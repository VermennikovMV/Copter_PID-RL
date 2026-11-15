"""Audit physical clock relation in capture order, retaining wall-clock anomalies.

Old wall-time nearest-neighbour matching can pair different simulation times after
a realtime correction. Capture order avoids sorting across those corrections, but
queueing still limits clock precision. A small residual is measured, never assumed.
"""
from pathlib import Path
import json
import struct
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "audit"))
from pcap_evidence import frames, ipv4, MavClockStream
from capture_udp_copies import CaptureCopyFilter


def ordered_clock(path, boot_start=None, boot_end=None, tcp_port=5760, plugin_port=9002, include_commands=False, include_fdm=False):
    streams, offsets = {}, []
    latest_sim, last_wall, last_sim = None, None, None
    backwards_wall, backwards_sim, maximum_wall_backstep = 0, 0, 0.
    example_fdm_keys = None
    wire_commands = []
    active_sockets, active_fdm, foreign_fdm = {}, {}, {}
    physical_samples = []
    copy_filter = CaptureCopyFilter()
    raw_last_sim, raw_backsteps, raw_backstep_witnesses = None, 0, []
    retained_active_fdm = {}
    for capture_frame, (wall, linktype, packet) in enumerate(frames(path), 1):
        if last_wall is not None and wall < last_wall:
            backwards_wall += 1
            maximum_wall_backstep = max(maximum_wall_backstep, last_wall - wall)
        last_wall = wall
        parsed = ipv4(linktype, packet)
        if parsed is None:
            continue
        protocol, source, dest, body = parsed
        if len(body) < 8:
            continue
        sport, dport = struct.unpack("!HH", body[:4])
        if protocol == 17 and (sport in (plugin_port, plugin_port + 1) or dport in (plugin_port, plugin_port + 1)):
            payload = body[8:]
            if dport == plugin_port and len(payload) in (40, 72) and struct.unpack('<H', payload[:2])[0] in (18458, 29569):
                socket = (source, sport, dest, dport)
                active_sockets[socket] = active_sockets.get(socket, 0) + 1
                continue
            try:
                value = json.loads(payload.strip(b"\x00\r\n "))
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(value, dict) and "timestamp" in value:
                # An orphan Gazebo can resend a cached FDM to a DEAD UDP socket.
                # Only a response addressed to an actual SERVO sender can clock
                # this flight. This is endpoint identity, not value-based trimming.
                response_socket = (dest, dport, source, sport)
                endpoint_name = '.'.join(map(str, dest)) + ':' + str(dport)
                destination = active_fdm if response_socket in active_sockets else foreign_fdm
                endpoint = destination.setdefault(endpoint_name, {'count': 0, 'min_sim_time': float('inf'), 'max_sim_time': float('-inf')})
                endpoint['count'] += 1
                endpoint['min_sim_time'] = min(endpoint['min_sim_time'], float(value['timestamp']))
                endpoint['max_sim_time'] = max(endpoint['max_sim_time'], float(value['timestamp']))
                if response_socket not in active_sockets:
                    continue
                raw_sim = float(value['timestamp'])
                is_copy = copy_filter.observe(packet, linktype, capture_frame, wall)
                if raw_last_sim is not None and raw_sim < raw_last_sim - 1e-9:
                    raw_backsteps += 1
                    raw_backstep_witnesses.append({'capture_frame': capture_frame,
                        'previous_sim_time': raw_last_sim, 'sim_time': raw_sim,
                        'later_capture_copy_removed': is_copy, 'copy_witness': copy_filter.last_match})
                raw_last_sim = raw_sim
                if is_copy:
                    continue
                retained_active_fdm[endpoint_name] = retained_active_fdm.get(endpoint_name, 0) + 1
                if example_fdm_keys is None:
                    example_fdm_keys = list(value)
                latest_sim = float(value["timestamp"])
                if last_sim is not None and latest_sim < last_sim - 1e-9:
                    backwards_sim += 1
                last_sim = latest_sim
                if include_fdm:
                    physical_samples.append((latest_sim, value.get('position'), value.get('velocity')))
        elif protocol == 6 and (sport == tcp_port or dport == tcp_port) and len(body) >= 20:
            header = (body[12] >> 4) * 4
            if len(body) <= header:
                continue
            sequence = struct.unpack("!I", body[4:8])[0]
            targets = dport == tcp_port
            stream = streams.setdefault((source, dest, sport, dport), MavClockStream(targets=targets))
            decoded = stream.feed(sequence, body[header:])
            if targets:
                wire_commands.extend(decoded)
                continue
            for msgid, boot in decoded:
                if msgid != 32 or latest_sim is None:
                    continue
                if boot_start is not None and boot < boot_start:
                    continue
                if boot_end is not None and boot > boot_end:
                    continue
                offsets.append(latest_sim - boot)
    if not offsets:
        raise ValueError("No paired clock evidence in capture")
    values = np.asarray(offsets)
    median = float(np.median(values))
    result = {
        "method": "latest FDM addressed to an observed SERVO source socket preceding CRC-valid LOCAL_POSITION_NED in first-observation capture order; strict reversed-MAC checksum-completion copies deduplicated; no realtime sorting or value-based outlier removal",
        "pairs": len(values), "median_gazebo_minus_boot_s": median,
        "residual_p95_s": float(np.quantile(np.abs(values - median), .95)),
        "offset_p05_s": float(np.quantile(values, .05)), "offset_p95_s": float(np.quantile(values, .95)),
        "offset_min_s": float(values.min()), "offset_max_s": float(values.max()),
        "outliers_over50ms": int(np.count_nonzero(np.abs(values - median) > .05)),
        "capture_realtime_backsteps": backwards_wall, "largest_realtime_backstep_s": maximum_wall_backstep,
        "fdm_simtime_backsteps": backwards_sim,
        "raw_fdm_simtime_backsteps": raw_backsteps,
        "raw_fdm_backstep_witnesses": raw_backstep_witnesses,
        "capture_copy_dedup": copy_filter.diagnostics(),
        "retained_active_endpoint_fdm_counts": retained_active_fdm,
        "tcp_gaps": sum(stream.gaps for stream in streams.values()),
        "crc_failures": sum(stream.crc_bad for stream in streams.values()),
        "fdm_schema_keys": example_fdm_keys,
        "limitation": "MAVLink sample/send latency and capture queueing remain; envelope is empirical clock association uncertainty, not proof of an exact offset",
        "tcp_port": tcp_port, "plugin_port": plugin_port,
        "crc_valid_wire_attitude_commands": len(wire_commands),
        "servo_sources": [{'source_ip': '.'.join(map(str, key[0])), 'source_port': key[1], 'plugin_ip': '.'.join(map(str,key[2])), 'plugin_port':key[3], 'packets':count} for key,count in active_sockets.items()],
        "active_endpoint_fdm": active_fdm,
        "foreign_endpoint_fdm": foreign_fdm,
    }
    if include_commands:
        result["wire_attitude_commands"] = wire_commands
    if include_fdm:
        result['physical_samples'] = physical_samples
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boot-start", type=float)
    parser.add_argument("--boot-end", type=float)
    parser.add_argument("--tcp-port", type=int, default=5760)
    parser.add_argument("--plugin-port", type=int, default=9002)
    args = parser.parse_args()
    result = ordered_clock(args.path, args.boot_start, args.boot_end, args.tcp_port, args.plugin_port)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
