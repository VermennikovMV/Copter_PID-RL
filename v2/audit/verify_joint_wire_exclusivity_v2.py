"""Read-only adjunct: exclusive CRC-valid wire commands and receive-order yaw.

This does not replace verify_joint_hybrid_v2.py's independent actor replay or a
physical/metrics audit. Expected rate/thrust bytes come from commands.jsonl;
combine this receipt with the actor receipt for the NN-to-wire claim. No runtime
controller module is imported, and no captured run is modified.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile

sys.dont_write_bytecode = True
from packet_clock_ordered import ordered_clock
from pcap_evidence import frames, ipv4, x25


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class HeaderStream:
    """First-observation TCP bytes; only identical sequence overlap is removed.

    Retain byte values to detect conflicting retransmissions. Gaps and sequence
    wrap/reset are explicit uncertainty, never treated as command deduplication.
    MAVLink messages with identical values at NEW TCP positions remain distinct.
    """
    def __init__(self):
        self.next_seq = None
        self.seen = {}
        self.data = b""
        self.gaps = self.conflicting_overlap = self.retransmitted_bytes = 0
        self.crc_bad = 0

    def feed(self, sequence, payload, frame, wall, flow):
        for offset, value in enumerate(payload):
            position = sequence + offset
            if position in self.seen:
                self.retransmitted_bytes += 1
                if self.seen[position] != value:
                    self.conflicting_overlap += 1
            else:
                self.seen[position] = value
        if self.next_seq is not None:
            if sequence < self.next_seq:
                payload = payload[self.next_seq-sequence:]
                sequence = self.next_seq
            elif sequence > self.next_seq:
                self.gaps += 1
                self.data = b""
        if not payload:
            return []
        self.next_seq = sequence + len(payload)
        self.data += payload
        decoded = []
        while len(self.data) >= 8:
            if self.data[0] not in (0xfe, 0xfd):
                self.data = self.data[1:]
                continue
            v2 = self.data[0] == 0xfd
            header = 10 if v2 else 6
            if len(self.data) < header:
                break
            size = self.data[1]
            total = header + size + 2 + (13 if v2 and self.data[2] & 1 else 0)
            if len(self.data) < total:
                break
            message, self.data = self.data[:total], self.data[total:]
            msgid = int.from_bytes(message[7:10], "little") if v2 else message[5]
            if msgid != 82:
                continue
            expected_crc = int.from_bytes(message[header+size:header+size+2], "little")
            if x25(message[1:header+size] + bytes([49])) != expected_crc:
                self.crc_bad += 1
                continue
            payload82 = message[header:header+size].ljust(39, b"\0")
            decoded.append({"boot_ms": struct.unpack("<I", payload82[:4])[0],
                            "rates_thrust_hex": payload82[20:36].hex(),
                            "rates_thrust": list(struct.unpack("<ffff", payload82[20:36])),
                            "target_system": payload82[36], "target_component": payload82[37],
                            "type_mask": payload82[38],
                            "source_system": message[5] if v2 else message[3],
                            "source_component": message[6] if v2 else message[4],
                            "capture_frame_completed": frame, "capture_wall_completed": wall,
                            "flow": flow})
        return decoded


def decode_headers(path, tcp_port):
    streams, wire = {}, []
    for frame, (wall, linktype, packet) in enumerate(frames(path), 1):
        parsed = ipv4(linktype, packet)
        if parsed is None:
            continue
        protocol, source, dest, body = parsed
        if protocol != 6 or len(body) < 20:
            continue
        sport, dport = struct.unpack("!HH", body[:4])
        if dport != tcp_port:
            continue
        header = (body[12] >> 4) * 4
        if header < 20 or len(body) <= header:
            continue
        key = (source, dest, sport, dport)
        flow = {"source_ip": ".".join(map(str, source)), "source_port": sport,
                "destination_ip": ".".join(map(str, dest)), "destination_port": dport}
        wire.extend(streams.setdefault(key, HeaderStream()).feed(
            struct.unpack("!I", body[4:8])[0], body[header:], frame, wall, flow))
    diagnostics = {key: sum(getattr(value, key) for value in streams.values())
                   for key in ("gaps", "conflicting_overlap", "retransmitted_bytes", "crc_bad")}
    diagnostics["flows"] = len(streams)
    return wire, diagnostics


def signature(row):
    return (row["boot_ms"], row["rates_thrust_hex"], row["type_mask"],
            row["source_system"], row["source_component"],
            row["target_system"], row["target_component"])


def expected_commands(commands, sender, target):
    expected = []
    for command in commands:
        action = command["action"]
        expected.append({"boot_ms": int(command["boot_s"] * 1000) & 0xffffffff,
                         "rates_thrust_hex": struct.pack("<ffff", 2*action[1], -2*action[2],
                                                         -action[3], command["guid_thrust"]).hex(),
                         "type_mask": 128, "source_system": sender[0], "source_component": sender[1],
                         "target_system": target[0], "target_component": target[1]})
    return expected


def exclusivity(expected, wire, low_ms, high_ms):
    selected = [row for row in wire if low_ms <= row["boot_ms"] <= high_ms]
    planned, observed = Counter(map(signature, expected)), Counter(map(signature, selected))
    missing, extras = planned-observed, observed-planned
    return {"pass": bool(expected) and not missing and not extras and len(selected) == len(expected),
            "expected_commands": len(expected), "in_window_wire_commands": len(selected),
            "outside_window_wire_commands": len(wire)-len(selected),
            "window_boot_ms_inclusive": [low_ms, high_ms],
            "missing_count": sum(missing.values()), "extra_or_conflicting_count": sum(extras.values()),
            "missing_examples": [{"signature": list(key), "count": count} for key, count in list(missing.items())[:10]],
            "extra_examples": [{"signature": list(key), "count": count} for key, count in list(extras.items())[:10]],
            "comparison": "exact integer send boot_ms, IEEE754 float32 FRD rates/thrust bytes, mask and MAVLink sender/target; multiset equality, no value-based deduplication"}


def call_is(node, owner, name):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == owner and node.func.attr == name)


def archive_contract(path):
    source = path.read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    starts = [node for node in calls if call_is(node, "rec", "event") and node.args
              and isinstance(node.args[0], ast.Constant) and node.args[0].value == "measurement_start"]
    resets = [node for node in calls if call_is(node, "policy", "reset") and len(node.args) == 1
              and isinstance(node.args[0], ast.Name) and node.args[0].id == "yaw_reference"]
    yaw = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "yaw_reference" for t in node.targets)]
    connections = [node for node in calls if isinstance(node.func, ast.Attribute)
                   and node.func.attr == "mavlink_connection"]
    senders = []
    for node in connections:
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if all(key in keywords and isinstance(keywords[key], ast.Constant)
               for key in ("source_system", "source_component")):
            senders.append((keywords["source_system"].value, keywords["source_component"].value))
    shape = len(starts) == len(resets) == len(yaw) == len(senders) == 1
    if not shape:
        return {"pass": False, "reason": "ambiguous archived start/yaw/reset/sender syntax"}, None
    start, assignment, reset = starts[0], yaw[0], resets[0]
    between = [node for node in calls if start.lineno < node.lineno <= reset.end_lineno]
    # Whitelist the observed synchronous block: state reads, atan2, policy.reset.
    # Unknown intervening calls could receive telemetry indirectly.
    permitted = all(call_is(node, "rec", "state") or call_is(node, "math", "atan2")
                    or call_is(node, "policy", "reset") for node in between)
    state = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "state"]
    state_read_only = len(state) == 1 and all(
        not (isinstance(node, ast.Attribute) and node.attr in ("drain", "recv_match", "recv_msg", "wait"))
        for node in ast.walk(state[0]))
    passed = start.lineno < assignment.lineno < reset.lineno and permitted and state_read_only
    return {"pass": passed, "measurement_start_line": start.lineno, "yaw_assignment_line": assignment.lineno,
            "policy_reset_line": reset.lineno, "no_receive_between_start_and_reset": permitted,
            "state_method_has_no_receive": state_read_only,
            "synchronous_block": source.splitlines()[start.lineno-1:reset.end_lineno]}, senders[0]


def yaw_from_receive_order(raw, events, reference):
    starts = [row for row in events if row.get("event") == "measurement_start"]
    if len(starts) != 1:
        return {"status": "INDETERMINATE", "reason": "measurement_start must be unique"}
    wall = starts[0].get("wall_time")
    if not isinstance(wall, (int, float)) or not math.isfinite(wall):
        return {"status": "INDETERMINATE", "reason": "missing finite measurement_start.wall_time"}
    valid = all(row.get("record_index") == index and
                all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                    for key in ("wall_time", "receive_monotonic_s"))
                for index, row in enumerate(raw, 1))
    if not valid or not raw:
        return {"status": "INDETERMINATE", "reason": "missing/invalid raw record index or receive clocks"}
    mono_backsteps = sum(right["receive_monotonic_s"] <= left["receive_monotonic_s"] for left, right in zip(raw, raw[1:]))
    wall_backsteps = sum(right["wall_time"] < left["wall_time"] for left, right in zip(raw, raw[1:]))
    equal_cutoff = any(row["wall_time"] == wall for row in raw)
    if mono_backsteps or equal_cutoff:
        return {"status": "INDETERMINATE", "reason": "receive clock order cannot uniquely establish wall-time cutoff",
                "monotonic_nonincreasing": mono_backsteps, "wall_backsteps": wall_backsteps, "raw_equals_event_wall": equal_cutoff}
    before = [row for row in raw if row["wall_time"] < wall]
    after = [row for row in raw if row["wall_time"] > wall]
    attitudes = [row for row in before if row["message"].get("mavpackettype") == "ATTITUDE"]
    if not attitudes or not after:
        return {"status": "INDETERMINATE", "reason": "missing ATTITUDE before start or raw sample after start"}
    prefix = before[-1]["record_index"] == len(before) and after[0]["record_index"] == len(before)+1
    bracket_residual = (after[0]["wall_time"]-before[-1]["wall_time"]
                        -(after[0]["receive_monotonic_s"]-before[-1]["receive_monotonic_s"]))
    if not prefix or abs(bracket_residual) > .001:
        return {"status": "INDETERMINATE", "reason": "event wall cutoff is not a unique receive-order prefix or its bracket contains a clock discontinuity",
                "wall_classification_is_receive_prefix": prefix, "wall_backsteps": wall_backsteps,
                "bracket_wall_minus_monotonic_delta_s": bracket_residual, "bracket_clock_tolerance_s": .001}
    selected = attitudes[-1]
    message = selected["message"]
    # NED/FRD -> ENU/FLU body-X direction, independent of roll and pitch.
    yaw = math.atan2(math.cos(message["pitch"])*math.cos(message["yaw"]),
                     math.cos(message["pitch"])*math.sin(message["yaw"]))
    residual = abs(math.atan2(math.sin(yaw-reference), math.cos(yaw-reference)))
    later = [row for row in after if row["message"].get("mavpackettype") == "ATTITUDE"
             and row["message"].get("time_boot_ms", float("inf")) <= starts[0]["boot_s"]*1000+1e-6]
    return {"status": "PASS" if residual < 1e-10 else "FAIL", "reference_rad": reference,
            "derived_enu_yaw_rad": yaw, "wrapped_residual_rad": residual,
            "measurement_start_wall_time": wall, "measurement_start_boot_s": starts[0]["boot_s"],
            "selected_record_index": selected["record_index"], "selected_receive_monotonic_s": selected["receive_monotonic_s"],
            "selected_wall_time": selected["wall_time"], "selected_boot_ms": message["time_boot_ms"],
            "raw_cutoff_record_index": before[-1]["record_index"], "next_raw_record_index": after[0]["record_index"],
            "wall_backsteps_elsewhere": wall_backsteps, "wall_classification_is_receive_prefix": prefix,
            "bracket_wall_minus_monotonic_delta_s": bracket_residual, "bracket_clock_tolerance_s": .001,
            "late_postevent_attitudes_at_or_before_start_boot": len(later),
            "selection": "last ATTITUDE in raw receive/record order strictly before unique measurement_start.wall_time; no maximum-boot selection"}


def audit(folder):
    summary = json.loads((folder/"summary.json").read_text())
    manifest = json.loads((folder/"manifest.json").read_text())
    commands = json_lines(folder/"commands.jsonl")
    raw, events = json_lines(folder/"mavlink.jsonl"), json_lines(folder/"events.jsonl")
    archive = folder/"source_snapshot/run_experiment_v2.py"
    archived, sender = archive_contract(archive)
    connected = [row for row in events if row.get("event") == "connected"]
    archive_hash_ok = digest(archive) == manifest.get("script_sha256") == manifest.get("source_snapshot_sha256", {}).get("run_experiment_v2.py")
    yaw = yaw_from_receive_order(raw, events, summary["yaw_reference_enu_rad"])
    if not archived["pass"] or not archive_hash_ok:
        yaw["status"] = "INDETERMINATE"
        yaw["archive_limitation"] = "hashed synchronous measurement-start/yaw/reset contract not established"
    if not sender or len(connected) != 1:
        return {"status": "INDETERMINATE", "reason": "archived sender or connected target identity unavailable",
                "archive_contract": archived, "yaw_reference": yaw}
    target = (connected[0]["system"], connected[0]["component"])
    expected = expected_commands(commands, sender, target)
    wire, tcp = decode_headers(folder/"network.pcap.gz", summary["tcp_port"])
    clock = ordered_clock(folder/"network.pcap.gz", commands[0]["boot_s"], commands[-1]["boot_s"],
                          summary["tcp_port"], summary["plugin_port"], include_commands=True)
    basic = clock.pop("wire_attitude_commands")
    basic_signatures = Counter((round(boot*1000), struct.pack("<ffff", p, q, r, thrust).hex(), mask)
                               for boot, p, q, r, thrust, mask in basic)
    header_signatures = Counter((row["boot_ms"], row["rates_thrust_hex"], row["type_mask"]) for row in wire)
    parity = basic_signatures == header_signatures
    low = int(summary["measurement_boot_start_s"]*1000)
    requested_end = math.ceil((summary["measurement_boot_start_s"]+summary["requested_duration_s"])*1000)
    # The runner tests duration before drain(); its last drain can advance past
    # the deadline and still produce one final command. Include that command,
    # rather than silently omitting it from the proof or calling it missing.
    high = max(requested_end, max(row["boot_ms"] for row in expected))
    exclusive = exclusivity(expected, wire, low, high)
    checks = [{"check": "immutable_archived_runner", "pass": archive_hash_ok},
              {"check": "ordered_clock_header_decode_agreement", "pass": parity,
               "ordered_clock_commands": len(basic), "header_decoded_commands": len(wire)},
              {"check": "wire_command_multiset_is_exactly_logged_actor_commands", **exclusive},
              {"check": "receive_order_yaw_reference", "pass": yaw["status"] == "PASS"},
              {"check": "all_commands_carry_summary_yaw_reference", "pass": all(
                  row.get("yaw_reference_enu_rad") == summary["yaw_reference_enu_rad"] for row in commands)},
              {"check": "archived_no_receive_before_yaw_reset", **archived}]
    ambiguous_tcp = tcp["gaps"] or tcp["conflicting_overlap"]
    if not archive_hash_ok or not parity or not exclusive["pass"] or not checks[-2]["pass"]:
        status = "FAIL"
    elif ambiguous_tcp or not archived["pass"] or yaw["status"] == "INDETERMINATE":
        status = "INDETERMINATE"
    else:
        status = "PASS" if yaw["status"] == "PASS" else "FAIL"
    inputs = ["summary.json", "manifest.json", "commands.jsonl", "mavlink.jsonl", "events.jsonl", "network.pcap.gz", "source_snapshot/run_experiment_v2.py", "source_snapshot/policy.npz"]
    return {"status": status, "run": str(folder), "checks": checks, "yaw_reference": yaw,
            "tcp_sequence_reassembly": tcp, "capture_order_clock": clock,
            "sender": list(sender), "target": list(target), "input_sha256": {name: digest(folder/name) for name in inputs},
            "auditor_sha256": digest(Path(__file__)),
            "dependency_sha256": {name: digest(Path(sys.modules[name].__file__))
                                  for name in ("packet_clock_ordered", "pcap_evidence", "capture_udp_copies")},
            "requested_measurement_end_boot_ms": requested_end,
            "last_logged_command_boot_ms": expected[-1]["boot_ms"],
            "last_command_past_requested_end_ms": max(0, expected[-1]["boot_ms"]-requested_end),
            "scope": "Captured incoming SET_ATTITUDE_TARGET on configured SITL TCP port within inclusive measurement boot-time window. Exact logged-command wire exclusivity and receive-order yaw initialization only.",
            "limitations": ["Conjoin with verify_joint_hybrid_v2 actor replay; this adjunct alone does not establish NN inference, training provenance, physical force delivery, metrics quality or flight success.",
                            "Commands on uncaptured interfaces/ports, or carrying boot timestamps outside the declared window, are outside this proof; outside-window decoded commands are counted explicitly.",
                            "A missing TCP interval or conflicting retransmitted byte is indeterminate; byte-identical overlap at the same TCP sequence is the only retransmission removal.",
                            "Yaw event has no monotonic timestamp. Strict monotonic raw receive order, a unique prefix split at event wall_time, a wall/monotonic-consistent bracket (1 ms tolerance) and the archived synchronous no-receive block are required. Wall backsteps elsewhere are retained and reported; an event-crossing backstep or bracket discontinuity is indeterminate."]}


def self_test():
    """Real CRC-valid synthetic PCAP, including distinct conflicting commands."""
    def mav(boot, rate=0.1, source=255, target=1, mask=128):
        # SET_ATTITUDE_TARGET base payload is exactly 39 bytes.
        payload = struct.pack("<IffffffffBBB", boot, 0., 0., 0., 0., rate, -.2, .3, .4, target, 0, mask)
        head = bytes([len(payload), 0, source, 190, 82])
        return b"\xfe"+head+payload+struct.pack("<H", x25(head+payload+b"\x31"))
    first, second = mav(11000), mav(11020)
    def packet(seq, payload):
        tcp = struct.pack("!HHIIHHHH", 45000, 5760, seq, 0, 0x5018, 65535, 0, 0)+payload
        ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20+len(tcp), 1, 0, 64, 6, 0, b"\x7f\0\0\1", b"\x7f\0\0\1")
        return b"\0"*12+b"\x08\0"+ip+tcp
    def decode(payloads):
        with tempfile.TemporaryDirectory(prefix="joint_wire_fixture_") as tmp:
            path = Path(tmp)/"fixture.pcap"
            content = struct.pack("<IHHIIII", 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1)
            for index, (seq, payload) in enumerate(payloads):
                value = packet(seq, payload)
                content += struct.pack("<IIII", 100, index*1000, len(value), len(value))+value
            path.write_bytes(content)
            return decode_headers(path, 5760)
    seq = 1000
    baseline, baseline_tcp = decode([(seq, first), (seq+len(first), second)])
    expected = [{key: row[key] for key in ("boot_ms", "rates_thrust_hex", "type_mask", "source_system", "source_component", "target_system", "target_component")} for row in baseline]
    retransmitted, retransmit_tcp = decode([(seq, first), (seq, first), (seq+len(first), second)])
    tests = {"baseline_crc_valid_pcap": len(baseline) == 2 and exclusivity(expected, baseline, 11000, 11020)["pass"],
             "same_tcp_sequence_retransmission_accepted": exclusivity(expected, retransmitted, 11000, 11020)["pass"] and retransmit_tcp["retransmitted_bytes"] == len(first)}
    witnesses = {}
    for name, extra in (("extra_conflicting_timestamp", mav(11010, .8)),
                        ("extra_conflicting_same_timestamp", mav(11000, .8)),
                        ("extra_identical_at_new_tcp_sequence", first),
                        ("extra_wrong_sender", mav(11000, source=254)),
                        ("extra_wrong_target", mav(11000, target=2))):
        actual, diagnostics = decode([(seq, first+second+extra)])
        result = exclusivity(expected, actual, 11000, 11020)
        tests[name+"_rejected"] = not result["pass"] and result["extra_or_conflicting_count"] == 1 and not diagnostics["gaps"]
        witnesses[name] = result
    for name, replacement in (("wrong_sender_equal_count", mav(11000, source=254)),
                              ("wrong_target_equal_count", mav(11000, target=2)),
                              ("wrong_mask_equal_count", mav(11000, mask=0)),
                              ("wrong_boot_equal_count", mav(11001)),
                              ("wrong_rate_equal_count", mav(11000, .1000001))):
        actual, diagnostics = decode([(seq, replacement+second)])
        result = exclusivity(expected, actual, 11000, 11020)
        tests[name+"_rejected"] = (not result["pass"] and result["in_window_wire_commands"] == 2
                                     and result["missing_count"] == result["extra_or_conflicting_count"] == 1)
        witnesses[name] = result
    conflicting, conflict_tcp = decode([(seq, first), (seq, mav(11000, .8)), (seq+len(first), second)])
    tests["conflicting_same_sequence_overlap_is_uncertain"] = conflict_tcp["conflicting_overlap"] > 0
    raw = [{"record_index": 1, "wall_time": 1., "receive_monotonic_s": 10., "message": {"mavpackettype": "ATTITUDE", "time_boot_ms": 1000, "yaw": .1, "pitch": 0.}},
           {"record_index": 2, "wall_time": 3., "receive_monotonic_s": 12., "message": {"mavpackettype": "ATTITUDE", "time_boot_ms": 1100, "yaw": .9, "pitch": 0.}}]
    events = [{"event": "measurement_start", "wall_time": 2., "boot_s": 1.2}]
    yaw = yaw_from_receive_order(raw, events, math.pi/2-.1)
    tests["late_attitude_before_start_boot_not_used_for_yaw"] = yaw["status"] == "PASS" and yaw["selected_record_index"] == 1 and yaw["late_postevent_attitudes_at_or_before_start_boot"] == 1
    tests["ambiguous_equal_yaw_cutoff_is_indeterminate"] = yaw_from_receive_order(raw, [{**events[0], "wall_time": 3.}], math.pi/2-.1)["status"] == "INDETERMINATE"
    remote_backstep = raw + [{**raw[-1], "record_index": 3, "wall_time": 4., "receive_monotonic_s": 13.},
                             {**raw[-1], "record_index": 4, "wall_time": 3.5, "receive_monotonic_s": 14.}]
    tests["remote_wall_backstep_with_unique_cutoff_retained"] = yaw_from_receive_order(remote_backstep, events, math.pi/2-.1)["status"] == "PASS"
    crossing = raw + [{**raw[-1], "record_index": 3, "wall_time": 1.5, "receive_monotonic_s": 13.}]
    tests["event_crossing_wall_backstep_is_indeterminate"] = yaw_from_receive_order(crossing, events, math.pi/2-.1)["status"] == "INDETERMINATE"
    discontinuity = [raw[0], {**raw[1], "receive_monotonic_s": 11.5}]
    tests["event_bracket_clock_discontinuity_is_indeterminate"] = yaw_from_receive_order(discontinuity, events, math.pi/2-.1)["status"] == "INDETERMINATE"
    return {"status": "PASS" if all(tests.values()) else "FAIL", "tests": tests,
            "negative_fixture_witnesses": witnesses, "baseline_tcp": baseline_tcp,
            "retransmission_tcp": retransmit_tcp, "conflicting_retransmission_tcp": conflict_tcp,
            "auditor_sha256": digest(Path(__file__))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.run is None:
        parser.error("run is required unless --self-test")
    if args.output.exists():
        parser.error("output already exists; preserve immutable receipts and choose a new path")
    report = self_test() if args.self_test else audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "auditor_sha256": digest(Path(__file__))}))
    sys.exit(0 if report["status"] == "PASS" else 1)
