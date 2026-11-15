"""Independent Path77 JOINT v3: indexed raw prefix -> causal77 -> NN -> wire.

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

OBSERVATION_CONTRACT = 'measured-memory-finite-leg-v3.0'
BASE_CONTRACT = 'measured-memory-yaw-v2.0'
PINNED_BASE_CONFIG = {'history': True, 'integral_limit': 6., 'acceleration_tau_s': .08,
    'position_scale': 3., 'velocity_scale': 3., 'rate_scale': 3.,
    'acceleration_scale': 5., 'history_times_s': [.1, .2]}
PINNED_ROUTE = np.array([[0.,0.,3.],[2.,0.,3.],[2.,2.,3.5],[0.,2.,3.],[0.,0.,3.]])


def validate_policy77(policy):
    """Fail closed on the exact supported exported actor schema, not its filename."""
    scalar_keys = {'layers','architecture','contract','base_contract','encoder_config','path_context_config'}
    tensor_keys = {f'{prefix}{i}' for prefix in ('w','b') for i in range(3)}
    if set(policy) != scalar_keys | tensor_keys or any(policy[k].shape != () for k in scalar_keys):
        raise ValueError('Unsupported Path77 archive fields/scalar schema')
    if (str(policy['contract']) != OBSERVATION_CONTRACT or str(policy['base_contract']) != BASE_CONTRACT
            or str(policy['architecture']) != 'hybrid' or policy['layers'].dtype.kind not in 'iu' or int(policy['layers']) != 3):
        raise ValueError('Unsupported Path77 actor contract/architecture/layers')
    cfg = json.loads(str(policy['encoder_config']))
    if cfg != PINNED_BASE_CONFIG or json.loads(str(policy['path_context_config'])) != {'finite_leg_scale_m': .25}:
        raise ValueError('Unsupported Path77 engineering scale or base74 configuration')
    shapes = {'w0': (128,77),'b0':(128,), 'w1':(128,128),'b1':(128,), 'w2':(4,128),'b2':(4,)}
    if any(policy[k].shape != shape or policy[k].dtype != np.float32 or not np.isfinite(policy[k]).all()
           for k, shape in shapes.items()):
        raise ValueError('Nonfinite or unsupported Path77 actor tensor shape/dtype')
    return cfg


def observation_identity(manifest, summary, folder):
    identity = summary.get('joint_observation')
    if not isinstance(identity, dict) or manifest.get('joint_observation') != identity:
        return False
    expected = {'contract': OBSERVATION_CONTRACT, 'size': 77, 'base_size': 74,
        'base_policy_sha256': digest(folder/'source_snapshot/policy.py'),
        'path_policy_sha256': digest(folder/'source_snapshot/path_policy.py')}
    return (set(identity) == set(expected) | {'actor_input_scope'}
        and all(identity.get(key) == value for key,value in expected.items())
        and isinstance(identity.get('actor_input_scope'), str))


class IndependentEncoderPath77:
    """Independent base74 plus a separate causal finite segment calculation.

    No controller or policy encoder import. Do not feed mission events/route array
    here: only target observations supplied to this method may update its memory.
    """
    def __init__(self, config, yaw_reference=None):
        self.base = IndependentEncoder(config, yaw_reference)
        self.current = self.start = None
        self.changes = 0

    def observe(self, position, velocity, orientation, rates, target, previous_action, dt):
        p, rot, goal = np.asarray(position,float), np.asarray(orientation,float), np.asarray(target,float)
        if p.shape != (3,) or rot.shape != (3,3) or goal.shape != (3,) or not np.isfinite(np.r_[p,rot.ravel(),goal]).all():
            raise ValueError('Invalid measured position/heading/current goal')
        if self.current is None:
            self.current = goal.copy()
        elif not np.array_equal(self.current, goal):
            self.start, self.current = self.current.copy(), goal.copy()
            self.changes += 1
        correction = np.zeros(3, np.float32)
        if self.start is not None:
            vector = self.current-self.start
            square_length = float(np.dot(vector,vector))
            if not math.isfinite(square_length):
                raise ValueError('Nonfinite finite-segment length')
            if square_length > 1e-15:
                fraction = min(1.,max(0.,float(np.dot(p-self.start,vector)/square_length)))
                displacement = self.start+fraction*vector-p
                yaw = math.atan2(rot[1,0],rot[0,0])
                cs,sn = math.cos(yaw),math.sin(yaw)
                # Written explicitly rather than importing the actor heading helper.
                correction = np.array([cs*displacement[0]+sn*displacement[1],
                    -sn*displacement[0]+cs*displacement[1],displacement[2]],np.float64)/.25
                if not np.isfinite(correction).all(): raise ValueError('Nonfinite finite-segment correction')
                correction = correction.astype(np.float32)
        base = self.base.observe(position,velocity,orientation,rates,target,previous_action,dt)
        return np.concatenate((base,correction)).astype(np.float32)


class IndependentMission:
    """Independent fixed protocol replay; called AFTER actor observation/action.

    This route is an auditor target-validity check; never supplied to encoder77.
    """
    def __init__(self, kind):
        if kind not in ('hover','route'): raise ValueError('Unsupported mission')
        self.kind, self.index = kind, 0
        self.inside = self.last_time = self.completion = None
        self.events = []

    @property
    def target(self): return PINNED_ROUTE[self.index].copy()

    def update(self, t, position):
        if not np.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            raise ValueError('Nonfinite or nonincreasing mission time')
        if self.last_time is not None and t-self.last_time > .03: self.inside = None
        self.last_time = t
        if self.kind == 'hover' or self.completion is not None: return
        if np.linalg.norm(np.asarray(position,float)-self.target) > .25:
            self.inside = None
            return
        if self.inside is None: self.inside = t
        if t-self.inside+1e-9 < .5: return
        self.events.append(dict(type='goal_reached',goal_index=self.index,time_s=t,
            dwell_started_s=self.inside,target=self.target.tolist()))
        self.inside = None
        if self.index == 4: self.completion = t
        else: self.index += 1



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


def replay_actor_inputs(folder, summary, manifest, policy, telemetry, commands, raw, checks):
    """Core raw-to-actor checks, also exercised on labelled offline fixtures."""
    cfg = validate_policy77(policy)
    events = json_lines(folder/'events.jsonl')
    interface = manifest["joint_interface"]
    semantics = interface["thrust_semantics"]
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
    archived_start = archive_start_contract(folder/'source_snapshot/run_experiment_v3.py')
    checks.append({'check': 'yaw_reference_from_explicit_receive_prefix', **start_evidence})
    checks.append({'check': 'archived_receive_cutoff_no_drain_before_yaw_reset', **archived_start})
    checks.append({'check': 'command_prefix_and_yaw_reference_follow_start', 'pass': all(
        row.get('raw_message_index', -1) >= summary.get('measurement_start_evidence', {}).get('raw_message_index', len(raw)+1)
        and row.get('yaw_reference_enu_rad') == yaw_reference for row in commands)})
    encoder = IndependentEncoderPath77(cfg, yaw_reference)
    mission = IndependentMission(summary["mission"])
    target_failures, observation_failures, time_failures = [], [], []
    recorded_action_failures = []
    largest_observation_error = largest_path_error = 0.
    previous_cutoff = summary["measurement_start_evidence"]["raw_message_index"]
    prefix_boots = np.maximum.accumulate([entry["message"].get("time_boot_ms", 0.)/1000 for entry in raw])
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
        if not previous_cutoff <= cutoff <= len(raw):
            raise ValueError('Command receive cutoff regresses or is out of range')
        sample_monotonic = float(command.get('state_sample_monotonic_s', float('nan')))
        boot = float(command['boot_s']); t = float(command['t'])
        good_time = (np.isfinite([sample_monotonic, boot, t]).all()
            and raw[cutoff-1]['receive_monotonic_s'] <= sample_monotonic
            and (cutoff == len(raw) or sample_monotonic < raw[cutoff]['receive_monotonic_s'])
            and boot == prefix_boots[cutoff-1]
            and abs(t-(boot-summary['measurement_boot_start_s'])) < 1e-10
            and abs(t-float(row['t'])) < 1e-12 and boot == float(row['boot_s'])
            and (index == 0 or t > float(commands[index-1]['t'])))
        if not good_time: time_failures.append(index)
        previous_cutoff = cutoff
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
        commanded_target = np.asarray(command["target_enu"], float)
        csv_target = np.array([float(row[k]) for k in ('target_e','target_n','target_up')])
        if (commanded_target.shape != (3,) or not np.array_equal(commanded_target, mission.target)
                or not np.array_equal(csv_target, mission.target)
                or command.get('goal_index') != mission.index or int(row['goal_index']) != mission.index):
            target_failures.append({'command': index, 'reason': 'target/index does not match independent PRE-transition mission'})
        observation = encoder.observe(position, velocity, orientation, rates, commanded_target, previous, dt)
        logged_observation = np.asarray(command.get('policy_observation', []), float)
        if (command.get('policy_observation_contract') != OBSERVATION_CONTRACT
                or command.get('policy_observation_size') != 77
                or logged_observation.shape != (77,) or not np.isfinite(logged_observation).all()):
            observation_failures.append(index)
        else:
            largest_observation_error = max(largest_observation_error, float(np.max(np.abs(observation-logged_observation))))
            largest_path_error = max(largest_path_error, float(np.max(np.abs(observation[74:]-logged_observation[74:]))))
        action = independent_mlp(policy, observation)
        recorded_action = np.asarray(command.get('action', []),float)
        if recorded_action.shape != (4,) or not np.isfinite(recorded_action).all():
            recorded_action_failures.append(index)
        mission.update(t, position)
        if not np.array_equal(np.asarray(command.get('target_next_enu'), float), mission.target):
            target_failures.append({'command': index, 'reason': 'next target differs from independent POST-transition mission'})
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
        # The actor observes its actually emitted previous action. Recompute
        # each action independently, but do not substitute roundoff from this
        # audit's MLP/GEMV into the next recorded input. Every emitted action is
        # independently matched to the NN and exact wire elsewhere below.
        previous = recorded_action.astype(np.float32)
    checks.extend([
        {'check': 'causal_current_targets_and_ordered_mission_replay', 'pass': not target_failures and mission.events == summary.get('mission_events'), 'failures': target_failures[:20], 'independent_events': mission.events},
        {'check': 'all77_actual_actor_inputs_match_independent_raw_reconstruction', 'pass': not observation_failures and largest_observation_error < 5e-6, 'bad_observation_rows': observation_failures[:20], 'max_difference': largest_observation_error, 'max_finite_leg_feature_difference': largest_path_error, 'rows': len(commands)},
        {'check': 'every_command_time_and_sample_follow_strict_received_prefix', 'pass': not time_failures, 'failures': time_failures[:20], 'wall_clock_used': False},
        {'check':'raw_message_indices_and_latest_public_state_source','pass':raw_index_valid and not record_source_failures,'failures':record_source_failures[:20]},
        {'check':'independent_measured_body_boost_inverse','pass':largest_adapter_error<5e-6,'max_difference':largest_adapter_error,'semantics':semantics,'inputs':'NN action, fixed nominal calibration, raw measured ATTITUDE only; target quaternion is diagnostic'},
        {'check':'optional_target_quaternion_diagnostic_matches_raw','pass':not diagnostic_failures,'failures':diagnostic_failures[:20]},
        {"check": "raw_MAV_state_available_for_every_command", "pass": not missing_raw, "missing": missing_raw},
        {"check": "CSV_pose_velocity_matches_raw_MAV", "pass": largest_state_error < 1e-12, "max_difference": largest_state_error},
        {"check": "stateful_encoder_dt_matches_real_control_intervals", "pass": largest_dt_error < 1e-12, "max_difference_s": largest_dt_error},
        {"check": "raw_MAV_stateful_encoder_MLP_reproduces_all_actions", "pass": not recorded_action_failures and largest_action_error < 5e-6,
         "commands": len(commands), "observation_size": int(policy["w0"].shape[1]), "max_action_difference": largest_action_error},
        {"check": "collective_and_FRD_mapping_matches_commands", "pass": largest_mapping_error < 5e-6, "max_difference": largest_mapping_error},
    ])
    return feedback, origins, expected_wire, command_times, archived_start, start_evidence


def audit(folder, *, offline_synthetic=False):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    fixture = manifest.get('offline_conversion')
    if bool(fixture) != bool(offline_synthetic):
        raise ValueError('Offline fixture requires explicit checker-only mode; actual flight auditor never accepts converted data')
    with (folder / "telemetry.csv").open(newline="") as stream:
        telemetry = list(csv.DictReader(stream))
    commands, raw = json_lines(folder / "commands.jsonl"), json_lines(folder / "mavlink.jsonl")
    events = json_lines(folder/'events.jsonl')
    checks = []
    interface=manifest.get('joint_interface',{})
    semantics=interface.get('thrust_semantics')
    if semantics not in ('legacy_preboost','body_collective'):raise ValueError('Missing/unsupported explicit joint thrust semantics')
    checks.append({'check':'explicit_joint_interface_identity','pass':interface.get('version')=='joint-hybrid-body-collective-v2.0' and interface.get('receive_evidence_version')==RECEIVE_EVIDENCE_VERSION and summary.get('joint_interface')==interface and interface.get('runner_sha256')==digest(folder/'source_snapshot/run_experiment_v3.py') and interface.get('adapter_sha256')==digest(folder/'source_snapshot/thrust_adapter.py') and manifest['script_sha256']==interface.get('runner_sha256')})
    checks.append({"check": "equal_nonempty_command_telemetry_count", "pass": 0 < len(commands) == len(telemetry) == summary["samples"]})
    archive_hashes = {name: digest(folder / "source_snapshot" / name) == expected
                      for name, expected in manifest["source_snapshot_sha256"].items()}
    checks.append({"check": "immutable_source_archive", "pass": all(archive_hashes.values()), "files": archive_hashes})
    policy_path = folder / "source_snapshot/policy.npz"
    checks.append({"check": "immutable_policy", "pass": digest(policy_path) == manifest["policy_sha256"] == summary["policy_sha256"]})
    with np.load(policy_path, allow_pickle=False) as saved:
        policy = {key: saved[key] for key in saved.files}
    cfg = validate_policy77(policy)
    checks.append({"check": "explicit_path77_observation_identity", "pass": observation_identity(manifest, summary, folder)})
    feedback, origins, expected_wire, command_times, archived_start, start_evidence = replay_actor_inputs(
        folder, summary, manifest, policy, telemetry, commands, raw, checks)
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
            "observation_contract": OBSERVATION_CONTRACT,
            "observation_dependency_scope": "Base74 from independent audit encoder; appended3 reconstructed independently from measured position/current heading and past observed target only. Mission replay checks commanded current target separately; future points never enter actor feature reconstruction.",
            "evidence_kind": "OFFLINE_SYNTHETIC_CONVERSION_NO_NEW_FLIGHT" if offline_synthetic else "ACTUAL_RECORDED_FLIGHT",
            "new_actual_flight": not offline_synthetic,
            "offline_conversion": fixture,
            "algorithm_claim": ("Offline actor/observation conversion of immutable old74 recorded data only; no new77 physical transfer established." if offline_synthetic else "Exported neural actor inference and physical transfer verified; BC/PPO training attribution requires separate training provenance and is not inferred from the flight filename."),
            "independent_metrics": metrics, "truth_position_metrics": {key: physical_metrics[key] for key in ("rmse_m", "penalized_rmse_m", "max_error_m", "max_altitude_drop_m", "recovery", "route")},
            "geographic_origin_world_enu_m": origin.tolist(), "capture_order_clock": clock, "dataflash": logs,
            "measurement_start_index_evidence": start_evidence,
            "auditor_sha256": digest(Path(__file__)),
            "wire_helper_sha256": digest(HERE/'verify_joint_wire_exclusivity_v2.py'),
            "limitations": "A single validation flight establishes no general superiority; matching-controller comparisons require the same scenario/mission/seed. AHRS10 is idealized SITL state, external50Hz MAVLink. World route dwell is a separate geometric diagnostic, not the controller's actual target-switch schedule. Real measured encoder history and wire commands are verified, not physical flight outside simulation."}



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Fresh output required; preserve receipts')
    try:
        report = audit(args.run)
    except Exception as exc:
        report = {'status':'FAIL','run':str(args.run),'auditor_sha256':digest(Path(__file__)),
            'error_type':type(exc).__name__,'error':str(exc),'checks':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'output':str(args.output),
        'failed_checks':[c['check'] for c in report['checks'] if not c['pass']], 'error':report.get('error')}))
    sys.exit(0 if report['status']=='PASS' else 1)
