"""Independent JOINT v2.1: indexed receive prefix -> NN -> adapter -> exact wire.

Controller/evaluator code is never imported. Verified policy is the archived
checkpoint. Native ATTITUDE_TARGET collective feedback corroborates acceptance;
its rates are shaped internally and are not treated as an echo of wire commands.
The adapter formula below is independent: no sim_joint/runtime code is imported.
"""
from pathlib import Path
import bisect
import ast
from collections import Counter
import csv
import hashlib
import json
import math
import struct
import sys
import xml.etree.ElementTree as ET
import numpy as np
from replay_observations import IndependentEncoder, independent_mlp
from packet_clock_ordered import ordered_clock
from independent_metrics import compute_metrics
from verify_joint_wire_exclusivity_v2 import decode_headers, expected_commands, exclusivity

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "audit"))
from dataflash_parameters import inspect as inspect_dataflash

RECEIVE_EVIDENCE_VERSION = 'joint-receive-index-v2.1'


def archive_start_contract(path):
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    def call(node, owner, name):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == owner and node.func.attr == name)
    evidence = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'measurement_start_evidence' for t in node.targets)]
    starts = [node for node in calls if call(node, 'rec', 'event') and node.args
              and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'measurement_start']
    resets = [node for node in calls if call(node, 'policy', 'reset')]
    connections = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'mavlink_connection']
    if len(evidence) != 1 or len(starts) != 1 or len(resets) != 1 or len(connections) != 1:
        return {'pass': False, 'reason': 'ambiguous archived receive-cutoff/start/reset/connection block'}
    capture, start, reset = evidence[0], starts[0], resets[0]
    expected = {'receive_evidence_version': 'RECEIVE_EVIDENCE_VERSION', 'raw_message_index': 'rec.n',
                'attitude_record_index': "rec.message_indices['ATTITUDE']", 'position_record_index': "rec.message_indices['LOCAL_POSITION_NED']",
                'attitude_time_boot_ms': "rec.latest['ATTITUDE'].time_boot_ms", 'position_time_boot_ms': "rec.latest['LOCAL_POSITION_NED'].time_boot_ms",
                'attitude_receive_monotonic_s': "rec.received['ATTITUDE']", 'position_receive_monotonic_s': "rec.received['LOCAL_POSITION_NED']",
                'boot_s': 'rec.boot_s', 'start_monotonic_s': 'time.monotonic()'}
    actual = {key.value: value for key, value in zip(capture.value.keys, capture.value.values)
              if isinstance(key, ast.Constant)} if isinstance(capture.value, ast.Dict) else {}
    expressions_match = set(actual) == set(expected) and all(
        ast.dump(actual[key]) == ast.dump(ast.parse(value, mode='eval').body) for key, value in expected.items())
    start_keywords = {keyword.arg: keyword.value for keyword in start.keywords}
    event_reuses_capture = ('measurement_start_evidence' in start_keywords
                           and ast.dump(start_keywords['measurement_start_evidence']) == ast.dump(ast.Name(id='measurement_start_evidence', ctx=ast.Load())))
    between = [node for node in calls if capture.lineno <= node.lineno <= reset.end_lineno]
    no_receive = all(call(node, 'time', 'monotonic') or call(node, 'rec', 'event') or call(node, 'rec', 'state')
                     or call(node, 'math', 'atan2') or call(node, 'policy', 'reset')
                     or (ast.dump(node.func) == ast.dump(ast.parse('rec.state()[0].tolist', mode='eval').body)
                         and not node.args and not node.keywords) for node in between)
    state = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'state']
    state_read_only = len(state) == 1 and not any(isinstance(node, ast.Attribute) and node.attr in ('drain', 'recv_match', 'recv_msg', 'wait') for node in ast.walk(state[0]))
    keywords = {keyword.arg: keyword.value for keyword in connections[0].keywords}
    sender = [keywords[key].value for key in ('source_system', 'source_component')
              if key in keywords and isinstance(keywords[key], ast.Constant)]
    return {'pass': bool(expressions_match and event_reuses_capture and no_receive and state_read_only
                         and capture.lineno < start.lineno < reset.lineno and len(sender) == 2),
            'capture_line': capture.lineno, 'measurement_start_line': start.lineno, 'policy_reset_line': reset.lineno,
            'exact_public_receive_sources': expressions_match, 'event_reuses_capture': event_reuses_capture,
            'no_receive_before_yaw_reset': no_receive, 'state_method_has_no_receive': state_read_only, 'sender': sender}


def measurement_start_from_indices(raw, events, summary):
    """Use the immutable explicit receive prefix; wall_time is diagnostic only."""
    failures = []
    starts = [row for row in events if row.get('event') == 'measurement_start']
    evidence = summary.get('measurement_start_evidence')
    if len(starts) != 1 or not isinstance(evidence, dict) or starts[0].get('measurement_start_evidence') != evidence:
        return {'pass': False, 'failures': ['missing/ambiguous/mismatched start event and summary evidence']}
    if evidence.get('receive_evidence_version') != RECEIVE_EVIDENCE_VERSION:
        failures.append('unsupported receive evidence version')
    valid_raw = bool(raw) and all(row.get('record_index') == index and
        isinstance(row.get('receive_monotonic_s'), (int, float)) and math.isfinite(row['receive_monotonic_s'])
        for index, row in enumerate(raw, 1))
    valid_raw = valid_raw and all(b['receive_monotonic_s'] > a['receive_monotonic_s'] for a, b in zip(raw, raw[1:]))
    cutoff = evidence.get('raw_message_index')
    if not valid_raw or type(cutoff) is not int or not 1 <= cutoff < len(raw):
        return {'pass': False, 'failures': failures+['invalid raw index/monotonic order or start prefix boundary']}
    now = evidence.get('start_monotonic_s')
    if not isinstance(now, (int, float)) or not math.isfinite(now) or not raw[cutoff-1]['receive_monotonic_s'] <= now < raw[cutoff]['receive_monotonic_s']:
        failures.append('start monotonic time does not bracket the explicit raw cutoff')
    selected = {}
    for kind, key, boot_key, receive_key in (
        ('ATTITUDE', 'attitude_record_index', 'attitude_time_boot_ms', 'attitude_receive_monotonic_s'),
        ('LOCAL_POSITION_NED', 'position_record_index', 'position_time_boot_ms', 'position_receive_monotonic_s')):
        indices = [index for index, row in enumerate(raw[:cutoff], 1) if row['message'].get('mavpackettype') == kind]
        index = evidence.get(key)
        if not indices or type(index) is not int or index != indices[-1]:
            failures.append(kind+' is not the latest record in the explicit prefix')
            continue
        selected[kind] = raw[index-1]['message']
        if evidence.get(boot_key) != selected[kind].get('time_boot_ms') or evidence.get(receive_key) != raw[index-1]['receive_monotonic_s']:
            failures.append(kind+' selected boot/receive identity differs from raw')
    prefix_boot = max((row['message']['time_boot_ms']/1000 for row in raw[:cutoff] if 'time_boot_ms' in row['message']), default=None)
    if prefix_boot != evidence.get('boot_s') or prefix_boot != starts[0].get('boot_s') or prefix_boot != summary.get('measurement_boot_start_s'):
        failures.append('start boot time is not the exact public raw prefix maximum')
    result = {'pass': False, 'failures': failures, 'raw_message_index': cutoff,
              'attitude_record_index': evidence.get('attitude_record_index'), 'position_record_index': evidence.get('position_record_index'),
              'prefix_boot_s': prefix_boot, 'start_monotonic_s': now,
              'wall_time_used_for_selection': False,
              'wall_backsteps_retained': sum(b.get('wall_time', 0) < a.get('wall_time', 0) for a, b in zip(raw, raw[1:]))}
    if len(selected) == 2:
        attitude, position = selected['ATTITUDE'], selected['LOCAL_POSITION_NED']
        enu = np.array([[0., 1, 0], [1, 0, 0], [0, 0, -1]])
        R = enu @ rotation(attitude['roll'], attitude['pitch'], attitude['yaw']) @ np.diag([1., -1, -1])
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        residual = abs(math.atan2(math.sin(yaw-summary['yaw_reference_enu_rad']), math.cos(yaw-summary['yaw_reference_enu_rad'])))
        if not math.isfinite(residual) or residual >= 1e-10:
            failures.append('yaw reference differs from indexed raw ATTITUDE')
        if starts[0].get('position_enu') != [position['y'], position['x'], -position['z']]:
            failures.append('start event position differs from indexed raw LOCAL_POSITION_NED')
        result.update(raw_yaw_enu_rad=yaw, reference_rad=summary['yaw_reference_enu_rad'], wrapped_residual_rad=residual)
    result['pass'] = not failures
    return result


def strict_wire_check(folder, summary, commands, wire, archived):
    connected = [row for row in json_lines(folder/'events.jsonl') if row.get('event') == 'connected']
    if len(connected) != 1 or len(archived.get('sender', [])) != 2:
        return {'check': 'exclusive_exact_CRC_valid_wire_commands', 'pass': False, 'reason': 'sender/target identity unavailable'}
    expected = expected_commands(commands, archived['sender'], (connected[0]['system'], connected[0]['component']))
    headers, tcp = decode_headers(folder/'network.pcap.gz', summary['tcp_port'])
    parity = (Counter((round(boot*1000), struct.pack('<ffff', p, q, r, thrust).hex(), mask) for boot, p, q, r, thrust, mask in wire)
              == Counter((row['boot_ms'], row['rates_thrust_hex'], row['type_mask']) for row in headers))
    low = int(summary['measurement_boot_start_s']*1000)
    high = max(math.ceil((summary['measurement_boot_start_s']+summary['requested_duration_s'])*1000), max(row['boot_ms'] for row in expected))
    result = exclusivity(expected, headers, low, high)
    result.update(check='exclusive_exact_CRC_valid_wire_commands', ordered_clock_header_agreement=parity, tcp_sequence_reassembly=tcp)
    result['pass'] = result['pass'] and parity and tcp['gaps'] == tcp['conflicting_overlap'] == 0
    return result


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rotation(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_joint_dataflash(path):
    record=inspect_dataflash(path)
    from pymavlink import mavutil
    stream=mavutil.mavlink_connection(str(path))
    while True:
        message=stream.recv_match(type='PARM')
        if message is None:break
        if message.Name=='ATC_ANGLE_BOOST':record['parameters']['ATC_ANGLE_BOOST']=float(message.Value)
    stream.close()
    return record


def independent_preboost(desired, measured_cos, age_s, max_age_s):
    """Literal inverse of native current-BODY lean compensation."""
    if not np.isfinite([desired, measured_cos, age_s, max_age_s]).all():
        raise ValueError('Nonfinite adapter replay input')
    if not 0<=desired<=1 or not -1.000000001<=measured_cos<=1.000000001 or not 0<max_age_s<=.1 or not 0<=age_s<=max_age_s:
        raise ValueError('Invalid or stale adapter replay input')
    denominator=max(.1,min(1.,measured_cos))
    fade=max(0.,min(1.,10.*measured_cos))
    if fade<=0:raise ValueError('Native boost inverse is undefined')
    factor=denominator/fade
    raw=desired*factor
    return min(1.,max(0.,raw)),denominator,fade,factor,raw


def audit(folder):
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    with (folder / "telemetry.csv").open(newline="") as stream:
        telemetry = list(csv.DictReader(stream))
    commands, raw = json_lines(folder / "commands.jsonl"), json_lines(folder / "mavlink.jsonl")
    events = json_lines(folder/'events.jsonl')
    checks = []
    interface=manifest.get('joint_interface',{})
    semantics=interface.get('thrust_semantics')
    if semantics not in ('legacy_preboost','body_collective'):raise ValueError('Missing/unsupported explicit joint thrust semantics')
    checks.append({'check':'explicit_joint_interface_identity','pass':interface.get('version')=='joint-hybrid-body-collective-v2.0' and interface.get('receive_evidence_version')==RECEIVE_EVIDENCE_VERSION and summary.get('joint_interface')==interface and interface.get('runner_sha256')==digest(folder/'source_snapshot/run_experiment_v2_1.py') and interface.get('adapter_sha256')==digest(folder/'source_snapshot/thrust_adapter.py') and manifest['script_sha256']==interface.get('runner_sha256')})
    checks.append({"check": "equal_command_telemetry_count", "pass": len(commands) == len(telemetry) == summary["samples"]})
    archive_hashes = {name: digest(folder / "source_snapshot" / name) == expected
                      for name, expected in manifest["source_snapshot_sha256"].items()}
    checks.append({"check": "immutable_source_archive", "pass": all(archive_hashes.values()), "files": archive_hashes})
    policy_path = folder / "source_snapshot/policy.npz"
    checks.append({"check": "immutable_policy", "pass": digest(policy_path) == manifest["policy_sha256"] == summary["policy_sha256"]})
    with np.load(policy_path, allow_pickle=False) as saved:
        policy = {key: saved[key] for key in saved.files}
    if str(policy["contract"]) != "measured-memory-yaw-v2.0" or str(policy["architecture"]) != "hybrid":
        raise ValueError("Unsupported temporal hybrid contract")
    cfg = json.loads(str(policy["encoder_config"]))
    positions, attitudes, feedback, origins = {}, {}, [], []
    by_index={};indices_by_kind={}
    raw_index_valid=True
    for record_index,entry in enumerate(raw,1):
        message = entry["message"]
        kind = message.get("mavpackettype")
        by_index[record_index]=entry
        indices_by_kind.setdefault(kind,[]).append(record_index)
        raw_index_valid=raw_index_valid and entry.get('record_index')==record_index and np.isfinite(entry.get('receive_monotonic_s',float('nan')))
        if kind == "LOCAL_POSITION_NED":
            positions[int(message["time_boot_ms"])] = message
        elif kind == "ATTITUDE":
            attitudes[int(message["time_boot_ms"])] = message
        elif kind == "ATTITUDE_TARGET":
            feedback.append(message)
        elif kind == "GPS_GLOBAL_ORIGIN":
            origins.append(message)
    enu_from_ned = np.array([[0., 1, 0], [1, 0, 0], [0, 0, -1]])
    flu_from_frd = np.diag([1., -1, -1])
    yaw_reference = summary["yaw_reference_enu_rad"]
    start_evidence = measurement_start_from_indices(raw, events, summary)
    archived_start = archive_start_contract(folder/'source_snapshot/run_experiment_v2_1.py')
    checks.append({'check': 'yaw_reference_from_explicit_receive_prefix', **start_evidence})
    checks.append({'check': 'archived_receive_cutoff_no_drain_before_yaw_reset', **archived_start})
    checks.append({'check': 'command_prefix_and_yaw_reference_follow_start', 'pass': all(
        row.get('raw_message_index', -1) >= summary.get('measurement_start_evidence', {}).get('raw_message_index', len(raw)+1)
        and row.get('yaw_reference_enu_rad') == yaw_reference for row in commands)})
    encoder = IndependentEncoder(cfg, yaw_reference)
    previous = np.zeros(4, np.float32)
    largest_action_error = largest_state_error = largest_dt_error = largest_mapping_error = 0.
    largest_adapter_error=0.;record_source_failures=[];diagnostic_failures=[]
    expected_wire, command_times = [], []
    hover = manifest["thrust_mapping"]["fixed_nominal_guid_hover"]
    missing_raw = []
    for index, (row, command) in enumerate(zip(telemetry, commands)):
        p_stamp, a_stamp = round(float(row["pos_boot_s"]) * 1000), round(float(row["att_boot_s"]) * 1000)
        if p_stamp not in positions or a_stamp not in attitudes:
            missing_raw.append(index)
            continue
        cutoff=int(command.get('raw_message_index',0))
        selected={}
        for kind,key in [('LOCAL_POSITION_NED','position_record_index'),('ATTITUDE','attitude_record_index')]:
            record=int(command.get(key,0));available=indices_by_kind.get(kind,[])
            before=bisect.bisect_right(available,cutoff)
            latest=available[before-1] if before else None
            if record!=latest or not 1<=record<=cutoff<=len(raw) or int(row.get(key,0))!=record or int(row.get('raw_message_index',0))!=cutoff:
                record_source_failures.append({'command':index,'kind':kind,'record':record,'latest_at_cutoff':latest})
            if record not in by_index:raise ValueError('Missing explicitly indexed raw state record')
            selected[kind]=by_index[record]
        p,a=selected['LOCAL_POSITION_NED']['message'],selected['ATTITUDE']['message']
        if round(p['time_boot_ms'])!=p_stamp or round(a['time_boot_ms'])!=a_stamp:record_source_failures.append({'command':index,'reason':'indexed state timestamp disagrees with telemetry'})
        position = np.array([p["y"], p["x"], -p["z"]])
        velocity = np.array([p["vy"], p["vx"], -p["vz"]])
        orientation = enu_from_ned @ rotation(a["roll"], a["pitch"], a["yaw"]) @ flu_from_frd
        rates = np.array([a["rollspeed"], -a["pitchspeed"], -a["yawspeed"]])
        recorded_state = np.array([float(row[key]) for key in ("x_e", "y_n", "z_up", "vx_e", "vy_n", "vz_up")])
        largest_state_error = max(largest_state_error, float(np.max(np.abs(np.r_[position, velocity] - recorded_state))))
        dt = .02 if index == 0 else command["t"] - commands[index-1]["t"]
        largest_dt_error = max(largest_dt_error, abs(dt - command["control_dt_s"]))
        observation = encoder.observe(position, velocity, orientation, rates, command["target_enu"], previous, dt)
        action = independent_mlp(policy, observation)
        largest_action_error = max(largest_action_error, float(np.max(np.abs(action - command["action"]))))
        desired = float(np.clip((float(action[0]) + 1) * hover, 0, 1))
        thrust=desired
        measured_cos=float(orientation[2,2])
        largest_adapter_error=max(largest_adapter_error,abs(float(command['measured_cos'])-measured_cos),abs(float(row['measured_cos'])-measured_cos))
        if semantics=='body_collective':
            received=float(selected['ATTITUDE']['receive_monotonic_s'])
            age=float(command['state_sample_monotonic_s'])-received
            thrust,denominator,fade,factor,unclipped=independent_preboost(desired,measured_cos,age,interface['adapter_attitude_max_age_s'])
            adapter=command.get('thrust_adapter') or {}
            expected={'desired_body_thrust':desired,'preboost_thrust':thrust,'measured_cos':measured_cos,'denominator':denominator,'inverted_factor':fade,'inverse_factor':factor,'unclipped_preboost_thrust':unclipped,'attitude_receive_age_s':age}
            differences=[abs(float(adapter.get(key,float('inf')))-value) for key,value in expected.items()]
            differences.extend([abs(float(command['desired_body_thrust'])-desired),abs(float(row['desired_body_thrust'])-desired),abs(float(row['boost_denominator'])-denominator),abs(float(row['inverted_factor'])-fade),abs(float(row['inverse_factor'])-factor),abs(float(command['attitude_receive_monotonic_s'])-received),abs(float(row['att_receive_age_s'])-age)])
            largest_adapter_error=max(largest_adapter_error,*differences)
            if adapter.get('clipped')!=(thrust!=unclipped):record_source_failures.append({'command':index,'reason':'adapter clipping flag'})
        elif command.get('thrust_adapter') is not None:
            record_source_failures.append({'command':index,'reason':'legacy mode unexpectedly compensated'})
        largest_adapter_error=max(largest_adapter_error,abs(float(command['preboost_thrust'])-thrust),abs(float(row['preboost_thrust'])-thrust))
        diagnostic=command.get('attitude_target_diagnostic')
        if diagnostic is not None:
            source=by_index.get(int(diagnostic['record_index']),{}).get('message',{})
            if source.get('mavpackettype')!='ATTITUDE_TARGET' or source.get('time_boot_ms')!=diagnostic['time_boot_ms'] or source.get('q')!=diagnostic['q'] or int(diagnostic['record_index'])>cutoff:diagnostic_failures.append(index)
        native_rates = np.array([2 * action[1], -2 * action[2], -action[3]], float)
        expected_wire.append(np.r_[native_rates, thrust])
        command_times.append(command["boot_s"])
        largest_mapping_error = max(largest_mapping_error, abs(thrust - command["guid_thrust"]),
                                    float(np.max(np.abs(action - [float(row[key]) for key in ("action_thrust", "action_p", "action_q", "action_r")]))))
        previous = action.copy()
    checks.extend([
        {'check':'raw_message_indices_and_latest_public_state_source','pass':raw_index_valid and not record_source_failures,'failures':record_source_failures[:20]},
        {'check':'independent_measured_body_boost_inverse','pass':largest_adapter_error<5e-6,'max_difference':largest_adapter_error,'semantics':semantics,'inputs':'NN action, fixed nominal calibration, raw measured ATTITUDE only; target quaternion is diagnostic'},
        {'check':'optional_target_quaternion_diagnostic_matches_raw','pass':not diagnostic_failures,'failures':diagnostic_failures[:20]},
        {"check": "raw_MAV_state_available_for_every_command", "pass": not missing_raw, "missing": missing_raw},
        {"check": "CSV_pose_velocity_matches_raw_MAV", "pass": largest_state_error < 1e-12, "max_difference": largest_state_error},
        {"check": "stateful_encoder_dt_matches_real_control_intervals", "pass": largest_dt_error < 1e-12, "max_difference_s": largest_dt_error},
        {"check": "raw_MAV_stateful_encoder_MLP_reproduces_all_actions", "pass": largest_action_error < 5e-6,
         "commands": len(commands), "observation_size": int(policy["w0"].shape[1]), "max_action_difference": largest_action_error},
        {"check": "collective_and_FRD_mapping_matches_commands", "pass": largest_mapping_error < 5e-6, "max_difference": largest_mapping_error},
    ])
    clock = ordered_clock(folder / "network.pcap.gz", command_times[0], command_times[-1],
                          summary["tcp_port"], summary["plugin_port"], include_commands=True)
    wire = clock.pop("wire_attitude_commands")
    checks.append(strict_wire_check(folder, summary, commands, wire, archived_start))
    wire_errors, wire_matched = [], set()
    for boot, p, q, r, thrust, mask in wire:
        if boot < command_times[0] - .002 or boot > command_times[-1] + .002:
            continue
        index = bisect.bisect_left(command_times, boot)
        candidates = [i for i in (index-1, index) if 0 <= i < len(command_times)]
        index = min(candidates, key=lambda i: abs(command_times[i] - boot))
        if abs(command_times[index] - boot) > .002:
            continue
        error = float(np.max(np.abs(np.array([p, q, r, thrust]) - expected_wire[index])))
        wire_errors.append(error)
        if error < 5e-6 and mask == 128:
            wire_matched.add(index)
    checks.append({"check": "CRC_valid_wire_commands_match_independent_NN_actions", "pass": len(wire_matched) == len(commands),
                   "matched_unique_commands": len(wire_matched), "planned_commands": len(commands),
                   "max_difference": max(wire_errors) if wire_errors else None, "CRC_failures": clock["crc_failures"]})
    feedback_errors, feedback_lags = [], []
    for message in feedback:
        stamp = message["time_boot_ms"] * .001
        if not command_times[0] + .1 <= stamp <= command_times[-1]:
            continue
        left, right = bisect.bisect_left(command_times, stamp-.1), bisect.bisect_right(command_times, stamp+1e-6)
        if left == right:
            continue
        error, nearest = min((abs(message["thrust"] - expected_wire[i][3]), i) for i in range(left, right))
        feedback_errors.append(float(error))
        feedback_lags.append(stamp-command_times[nearest])
    accepted = sum(error < 1e-5 for error in feedback_errors)
    checks.append({"check": "native_collective_feedback_confirms_acceptance", "pass": len(feedback_errors) > 100 and accepted / len(feedback_errors) >= .95,
                   "samples": len(feedback_errors), "matching_samples": accepted, "max_residual": max(feedback_errors),
                   "rates_note": "native ATTITUDE_TARGET rates are internally shaped, not direct command echoes"})
    physics_events = json_lines(folder / "force_physics.jsonl")
    last_force, last_off = np.zeros(3), None
    for event in physics_events:
        force = np.array(event["force_enu_N"])
        if np.linalg.norm(last_force) > 1e-9 and np.linalg.norm(force) < 1e-9:
            last_off = event["sim_time"]
        last_force = force
    off = (last_off - clock["median_gazebo_minus_boot_s"] - summary["measurement_boot_start_s"]
           if last_off is not None else None)
    value = lambda key: np.array([float(row[key]) for row in telemetry])
    route_points = None
    if summary['mission'] == 'route':
        for statement in ast.parse((folder/'source_snapshot/protocol.py').read_text()).body:
            if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROUTE_POINTS' for t in statement.targets):
                expression=statement.value
                route_points=ast.literal_eval(expression.args[0] if isinstance(expression,ast.Call) else expression)
    position = np.column_stack([value(key) for key in ("x_e", "y_n", "z_up")])
    target = np.column_stack([value(key) for key in ("target_e", "target_n", "target_up")])
    metrics = compute_metrics(time=value("t"), position=position, target=target, yaw=value("yaw_error_rad"), yaw_target=0.,
                              planned_seconds=summary["requested_duration_s"], failed=summary["failure"] is not None,
                              mission=summary["mission"], last_force_off_s=off, route_points=route_points)
    metric_map = {"rmse_m": "rmse_position_m", "penalized_rmse_m": "penalized_rmse_position_m",
                  "max_altitude_drop_m": "altitude_drop_m", "heading_rmse_deg": "heading_rmse_deg", "heading_peak_deg": "heading_peak_deg"}
    checks.append({"check": "independent_weighted_metrics", "pass": all(abs(metrics[key] - summary[name]) < 1e-9 for key, name in metric_map.items())})
    if summary['mission']=='hover':
        measured_recovery,recorded_recovery=metrics['recovery']['time_s'],summary['recovery_time_s']
        checks.append({"check": "recovery_from_actual_physics_OFF", "pass": measured_recovery==recorded_recovery or (measured_recovery is not None and recorded_recovery is not None and abs(measured_recovery-recorded_recovery)<1e-9)})
    else:
        independent_route=metrics['route']
        checks.append({'check':'independent_route_completion','pass':independent_route['completed']==summary['route_completed'] and
            (independent_route['completion_time_s']==summary['route_completion_time_s'] or
             (independent_route['completion_time_s'] is not None and summary['route_completion_time_s'] is not None and abs(independent_route['completion_time_s']-summary['route_completion_time_s'])<1e-9))})
    checks.append({"check": "landed_disarmed", "pass": summary["landed_disarmed"] is True})
    logs = [inspect_joint_dataflash(path) for path in sorted((folder / "sitl/logs").glob("*.BIN"))]
    checks.append({"check": "binary_SITL_log_AHRS10_GUID_options8", "pass": bool(logs) and all(record["parameters"].get("AHRS_EKF_TYPE") == 10 and record["parameters"].get("GUID_OPTIONS") == 8 for record in logs)})
    if semantics=='body_collective':
        readback=json.loads((folder/'parameter_readback.json').read_text())
        checks.append({'check':'native_angle_boost_readback_and_BIN_enabled','pass':readback.get('ATC_ANGLE_BOOST')==1 and all(record['parameters'].get('ATC_ANGLE_BOOST')==1 for record in logs)})
    truth_rows = json_lines(folder / "groundtruth.jsonl")
    unique = {row["sim_time"]: row for row in truth_rows if row.get("name") == "iris"}
    truth_time = np.array(sorted(unique))
    truth_position = np.array([unique[t]["position"] for t in truth_time])
    world = ET.parse(folder / "gazebo_assets/world.sdf").getroot().find("world/spherical_coordinates")
    gps = origins[0]
    if abs(float(world.findtext("latitude_deg")) - gps["latitude"]*1e-7) > 1e-7 or abs(float(world.findtext("longitude_deg")) - gps["longitude"]*1e-7) > 1e-7:
        raise ValueError("Nonidentity geographic coordinate conversion required")
    origin = np.array([0., 0., gps["altitude"]*.001 - float(world.findtext("elevation"))])
    sample_time = value("boot_s") + clock["median_gazebo_minus_boot_s"]
    interpolated = np.column_stack([np.interp(sample_time, truth_time, truth_position[:, i]) for i in range(3)])
    physical_metrics = compute_metrics(time=value("t"), position=interpolated-origin, target=target, yaw=value("yaw_error_rad"), yaw_target=0.,
                                       planned_seconds=summary["requested_duration_s"], failed=False, last_force_off_s=off,
                                       mission=summary['mission'],route_points=route_points)
    for check in checks:
        check["pass"] = bool(check["pass"])
    return {"status": "PASS" if all(check["pass"] for check in checks) else "FAIL", "checks": checks,
            "run": str(folder), "policy_sha256": digest(policy_path), "recorded_split": summary["split"],
            "derived_phase": "exploratory_validation" if 82001 <= summary["seed"] <= 82008 else summary["split"],
            "algorithm_claim": "Exported neural actor inference and physical transfer verified; BC/PPO training attribution requires the separate training provenance and is not inferred from the flight filename.",
            "independent_metrics": metrics, "truth_position_metrics": {key: physical_metrics[key] for key in ("rmse_m", "penalized_rmse_m", "max_error_m", "max_altitude_drop_m", "recovery", "route")},
            "geographic_origin_world_enu_m": origin.tolist(), "capture_order_clock": clock, "dataflash": logs,
            "measurement_start_index_evidence": start_evidence,
            "auditor_sha256": digest(Path(__file__)),
            "wire_helper_sha256": digest(HERE/'verify_joint_wire_exclusivity_v2.py'),
            "limitations": "A single validation flight establishes no general superiority; matching-controller comparisons require the same scenario/mission/seed. AHRS10 is idealized SITL state, external50Hz MAVLink. World route dwell is a separate geometric diagnostic, not the controller's actual target-switch schedule. Real measured encoder history and wire commands are verified, not physical flight outside simulation."}


def self_test():
    """Offline source differential and adversarial fixtures; no flight claim."""
    import copy
    import tempfile
    from types import SimpleNamespace
    from verify_joint_wire_exclusivity_v2 import self_test as wire_self_test
    root = HERE.parent
    old_path, new_path = root/'sim_joint/run_experiment_v2.py', root/'sim_joint/run_experiment_v2_1.py'
    old, new = ast.parse(old_path.read_text()), ast.parse(new_path.read_text())
    def target(node, expression):
        return isinstance(node, ast.Assign) and any(ast.unparse(value) == expression for value in node.targets)
    def assignment(tree, expression):
        return next(node for node in ast.walk(tree) if target(node, expression))
    old_guard = next(node for node in ast.walk(old) if isinstance(node, ast.If) and 'joint_freeze.get(' in ast.unparse(node.test))
    old_interface = assignment(old, "summary['joint_interface']")
    old_scope = next(value for key, value in zip(old_interface.value.keys, old_interface.value.values) if key.value == 'frozen_runtime_scope')
    class RemoveEvidenceOnly(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(target(node, value) for value in ('RECEIVE_EVIDENCE_VERSION', 'measurement_start_evidence', "summary['measurement_start_evidence']")):
                return None
            if target(node, 'required_joint_sources'):
                return copy.deepcopy(assignment(old, 'required_joint_sources'))
            if target(node, "summary['joint_interface']"):
                pairs = [(key, value) for key, value in zip(node.value.keys, node.value.values) if key.value != 'receive_evidence_version']
                node.value.keys = [key for key, value in pairs]
                node.value.values = [copy.deepcopy(old_scope) if key.value == 'frozen_runtime_scope' else value for key, value in pairs]
            return self.generic_visit(node)
        def visit_If(self, node):
            if 'joint_freeze.get(' in ast.unparse(node.test):
                return copy.deepcopy(old_guard)
            return self.generic_visit(node)
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'event' and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'measurement_start':
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != 'measurement_start_evidence']
            return self.generic_visit(node)
    normalized = RemoveEvidenceOnly().visit(copy.deepcopy(new))
    normalized.body[0] = copy.deepcopy(old.body[0])
    tests = {'full_runner_AST_equal_except_explicit_evidence_and_freeze_metadata': ast.dump(old) == ast.dump(normalized),
             'archived_start_uses_exact_public_indices_without_receive': archive_start_contract(new_path)['pass']}
    # Delayed attitude has a newer boot stamp below start, and arrives after a
    # wall-clock backstep. The explicit prefix must still select raw record 2.
    raw = [dict(record_index=1, wall_time=100., receive_monotonic_s=10., message=dict(mavpackettype='LOCAL_POSITION_NED', time_boot_ms=1000, x=1., y=2., z=-3.)),
           dict(record_index=2, wall_time=101., receive_monotonic_s=11., message=dict(mavpackettype='ATTITUDE', time_boot_ms=990, roll=.01, pitch=.02, yaw=.1)),
           dict(record_index=3, wall_time=99., receive_monotonic_s=12., message=dict(mavpackettype='ATTITUDE', time_boot_ms=995, roll=.3, pitch=.2, yaw=.9))]
    evidence = dict(receive_evidence_version=RECEIVE_EVIDENCE_VERSION, raw_message_index=2, attitude_record_index=2, position_record_index=1,
                    attitude_time_boot_ms=990, position_time_boot_ms=1000, attitude_receive_monotonic_s=11., position_receive_monotonic_s=10.,
                    boot_s=1., start_monotonic_s=11.5)
    summary = dict(measurement_start_evidence=evidence, measurement_boot_start_s=1., yaw_reference_enu_rad=math.pi/2-.1)
    events = [dict(event='measurement_start', wall_time=100.5, boot_s=1., position_enu=[2., 1., 3.], measurement_start_evidence=evidence)]
    fixture = measurement_start_from_indices(raw, events, summary)
    tests['late_attitude_and_event_crossing_wall_backstep_select_explicit_prefix'] = fixture['pass'] and fixture['attitude_record_index'] == 2 and fixture['wall_backsteps_retained'] == 1
    for name, changes in (('wrong_cutoff', {'raw_message_index': 3}), ('wrong_selected_ATTITUDE', {'attitude_record_index': 3}),
                          ('wrong_prefix_boot', {'boot_s': .995}), ('wrong_receive_time', {'attitude_receive_monotonic_s': 10.9}),
                          ('wrong_start_monotonic_bracket', {'start_monotonic_s': 12.1}), ('legacy_receive_version', {'receive_evidence_version': 'legacy'})):
        altered = {**evidence, **changes}
        tests[name+'_rejected'] = not measurement_start_from_indices(raw, [{**events[0], 'measurement_start_evidence': altered}], {**summary, 'measurement_start_evidence': altered})['pass']
    tests['missing_explicit_evidence_rejected_without_wall_fallback'] = not measurement_start_from_indices(raw, events, {key: value for key, value in summary.items() if key != 'measurement_start_evidence'})['pass']
    tests['event_summary_evidence_mismatch_rejected'] = not measurement_start_from_indices(raw, events, {**summary, 'measurement_start_evidence': {**evidence, 'raw_message_index': 1}})['pass']
    nonfinite = copy.deepcopy(raw)
    nonfinite[1]['message']['yaw'] = float('nan')
    tests['nonfinite_selected_yaw_rejected'] = not measurement_start_from_indices(nonfinite, events, summary)['pass']
    # Execute only the unchanged pure helpers and controller branch extracted
    # from AST. Neither runner module nor main() is imported or executed.
    def controller_branch(tree):
        return next(node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.unparse(node.test) == "a.controller == 'pid'")
    def exercise(tree, controller, semantics, sample):
        calls = []
        class Mav:
            def __getattr__(self, name):
                return lambda *args: calls.append((name, args))
        ns = dict(np=np, math=math, mavutil=SimpleNamespace(mavlink=SimpleNamespace(MAV_FRAME_LOCAL_NED=1)),
                  ENU_FROM_NED=np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]),
                  ControllerFailure=RuntimeError, ThrustAdapterError=ValueError)
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('rotation', 'send_position', 'pd_action')]
        exec(compile(ast.Module(body=copy.deepcopy(helpers), type_ignores=[]), '<pure-controller-helpers>', 'exec'), ns)
        policy = SimpleNamespace(predict_enu=lambda *args, **kwargs: {'action': np.array([.2, .1, -.2, .3])}) if controller == 'hybrid' else None
        ns.update(a=SimpleNamespace(controller=controller, thrust_semantics=semantics, hover_thrust=.3026512153, adapter_attitude_max_age_s=.1),
                  rec=SimpleNamespace(boot_s=12.345), conn=SimpleNamespace(target_system=1, target_component=0, mav=Mav()),
                  pos=np.array([sample*.01, -.2, 3.]), vel=np.array([.1, -.1, .01]), R=ns['rotation'](.01*sample, -.005*sample, .1),
                  omega=np.array([.01, -.02, .03]), target=np.array([1., 2., 3.]), previous=np.zeros(4, np.float32),
                  yaw_reference=1.2, policy=policy, control_dt=.02, att_age=.01,
                  body_collective_to_preboost=lambda desired, cosine, age, limit: SimpleNamespace(preboost_thrust=independent_preboost(float(desired), cosine, age, limit)[0]))
        exec(compile(ast.Module(body=[copy.deepcopy(controller_branch(tree))], type_ignores=[]), '<controller-branch>', 'exec'), ns)
        return repr(calls), ns['action'], ns['thrust'], ns['previous']
    differential_count = 0
    for controller in ('pid', 'hybrid', 'rate_pd'):
        for semantics in ('legacy_preboost', 'body_collective'):
            passed = ast.dump(controller_branch(old)) == ast.dump(controller_branch(new))
            for sample in range(24):
                before, after = exercise(old, controller, semantics, sample), exercise(new, controller, semantics, sample)
                passed = passed and before[0] == after[0] and np.array_equal(before[1], after[1], equal_nan=True) and before[2] == after[2] and np.array_equal(before[3], after[3], equal_nan=True)
                differential_count += 1
            tests[controller+'_'+semantics+'_wire_action_and_memory_unchanged'] = bool(passed)
    heldout = next(node for node in ast.walk(new) if isinstance(node, ast.If) and ast.unparse(node.test) == "a.split == 'heldout'" and any(target(child, 'required_joint_sources') for child in ast.walk(node)))
    required = ast.literal_eval(assignment(new, 'required_joint_sources').value)
    tests['heldout_required_sources_are_new_runner_and_unchanged_adapter'] = required == ['sim_joint/run_experiment_v2_1.py', 'sim_joint/thrust_adapter.py']
    frozen = dict(joint_interface_version='joint-hybrid-body-collective-v2.0', joint_receive_evidence_version=RECEIVE_EVIDENCE_VERSION,
                  joint_thrust_semantics='body_collective', joint_adapter_attitude_max_age_s=.1,
                  joint_runtime_sha256={name: digest(root/name) for name in required})
    class GuardError(ValueError):
        pass
    def reject(message):
        raise GuardError(message)
    with tempfile.TemporaryDirectory(prefix='joint_receive_guard_') as temp:
        freeze_path = Path(temp)/'freeze.json'
        def accepted(value):
            freeze_path.write_text(json.dumps(value))
            ns = dict(json=json, a=SimpleNamespace(split='heldout', freeze_file=freeze_path, thrust_semantics='body_collective', adapter_attitude_max_age_s=.1),
                      p=SimpleNamespace(error=reject), ROOT=root, digest=digest, INTERFACE_VERSION='joint-hybrid-body-collective-v2.0', RECEIVE_EVIDENCE_VERSION=RECEIVE_EVIDENCE_VERSION)
            try:
                exec(compile(ast.Module(body=[copy.deepcopy(heldout)], type_ignores=[]), '<heldout-guard>', 'exec'), ns)
                return True
            except GuardError:
                return False
        tests['heldout_exact_v2_1_freeze_accepted'] = accepted(frozen)
        for name, value in (('missing_age', {key: value for key, value in frozen.items() if key != 'joint_adapter_attitude_max_age_s'}),
                            ('wrong_age', {**frozen, 'joint_adapter_attitude_max_age_s': .05}),
                            ('missing_receive_version', {key: value for key, value in frozen.items() if key != 'joint_receive_evidence_version'}),
                            ('legacy_receive_version', {**frozen, 'joint_receive_evidence_version': 'v2.0'}),
                            ('wrong_runner_hash', {**frozen, 'joint_runtime_sha256': {**frozen['joint_runtime_sha256'], required[0]: '0'*64}}),
                            ('legacy_manifest', {**frozen, 'joint_runtime_sha256': {'sim_joint/run_experiment_v2.py': digest(old_path), required[1]: digest(root/required[1])}})):
            tests['heldout_'+name+'_rejected'] = not accepted(value)
    wire_fixture = wire_self_test()
    tests['strict_wire_helper_all_adversarial_PCAP_fixtures_PASS'] = wire_fixture['status'] == 'PASS'
    preserved = {name: digest(root/name) == digest(root/'artifacts/refinement/joint_v2/gazebo_thrust_ablation/source_snapshot'/archived)
                 for name, archived in (('sim_joint/run_experiment_v2.py', 'sim_joint/run_experiment_v2.py'),
                                        ('sim_joint/thrust_adapter.py', 'sim_joint/thrust_adapter.py'),
                                        ('audit/verify_joint_hybrid_v2.py', 'audit/verify_joint_hybrid_v2.py'))}
    tests['all_three_frozen_original_files_match_preexisting_archive'] = all(preserved.values())
    sources = ['sim_joint/run_experiment_v2_1.py', 'audit/verify_joint_hybrid_v2_1.py', 'audit/verify_joint_wire_exclusivity_v2.py',
               'sim_joint/thrust_adapter.py', 'sim_joint/run_experiment_v2.py', 'audit/verify_joint_hybrid_v2.py']
    return {'status': 'PASS' if all(tests.values()) else 'FAIL', 'schema': 'joint-receive-index-v2.1-offline-review', 'checks': tests,
            'source_sha256': {name: digest(root/name) for name in sources}, 'frozen_original_hash_preservation': preserved,
            'archive_start_contract': archive_start_contract(new_path),
            'controller_differential_cases': differential_count, 'explicit_start_fixture': fixture,
            'wire_helper_fixture_checks': wire_fixture['tests'], 'new_flight_verified': False,
            'limitations': 'Offline logging/index/freeze/control-invariance verification only. No new v2.1 Gazebo flight has been run; actual raw-to-actor-to-wire and physical transfer remain pending the next separately authorized pilot.'}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, nargs='?')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.run is None and not args.self_test:
        parser.error('run is required unless --self-test')
    if args.output.exists():
        parser.error('Output already exists; preserve immutable receipts and use a fresh path')
    report = self_test() if args.self_test else audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({'status': report['status'], 'output': str(args.output), 'checks': len(report['checks']),
                      'failed_checks': [name for name, passed in report['checks'].items() if not passed] if args.self_test
                      else [row['check'] for row in report['checks'] if not row['pass']]}))
    sys.exit(0 if report['status'] == 'PASS' else 1)
