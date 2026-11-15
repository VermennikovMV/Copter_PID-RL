"""Export public measured demo data after archive, integrity and media checks.

Standard library only. This never invokes a controller, simulator or encoder.
Integrity PASS is independent of controller quality and route completion.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import io
import json
import math
import ntpath
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('pid', 'hybrid', 'direct')
PII = re.compile(r'[A-Za-z]:[\\/]Users[\\/]|/mnt/[a-z]/Users/|/home/[^/\s]+/', re.I)
PRIVATE_PATH = re.compile(r'(?:[A-Za-z]:[\\/]|/mnt/[a-z]/Users/|/home/)[^\s\"\'<>\[\]{}]+')

class Rejected(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise Rejected(message)

def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result

def reject_constant(value):
    raise Rejected('Nonfinite JSON constant: ' + value)

def parse(data):
    return json.loads(data.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=reject_constant)

def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')

def relative(name):
    p = Path(name)
    require(not p.is_absolute() and not ntpath.isabs(name) and '..' not in p.parts
            and '\\' not in name and ':' not in name, 'Unsafe archive relative path')
    return p

def absolute(value):
    return value.startswith('/') or ntpath.isabs(value)

def unicode_paths(text):
    for _ in range(4):
        decoded = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), text).replace('\\/', '/')
        if decoded == text:
            break
        text = decoded
    return text

def privacy(data):
    text = unicode_paths(data.decode('utf-8-sig'))
    require(not PII.search(text), 'Public output still contains a private machine path')
    # Check decoded Unicode escapes too, without treating CSV as JSON.
    try:
        decoded = json.loads(text)
    except ValueError:
        return
    require(not PII.search(json.dumps(decoded, ensure_ascii=False)), 'Escaped private machine path')

class Inputs:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None, size=None):
        original = Path(path)
        require(not original.is_symlink(), 'Source symlink is forbidden: ' + original.name)
        path = original.resolve()
        require(path.is_file(), 'Missing regular source file: ' + path.name)
        if path not in self.files:
            h = hashlib.sha256()
            before = path.stat()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    'Source changed during hashing: ' + path.name)
            self.files[path] = dict(sha256=h.hexdigest(), bytes=after.st_size, mtime_ns=after.st_mtime_ns)
        row = self.files[path]
        require(expected is None or row['sha256'] == expected, 'Source SHA mismatch: ' + path.name)
        require(size is None or row['bytes'] == size, 'Source byte count mismatch: ' + path.name)
        return row['sha256']

    def read(self, path):
        self.bind(path)
        return parse(Path(path).read_bytes())

    def unchanged(self):
        for path, row in self.files.items():
            stat = path.stat()
            require((stat.st_size, stat.st_mtime_ns) == (row['bytes'], row['mtime_ns']),
                    'Source changed after validation: ' + path.name)

class Projection:
    """Replace path metadata, keeping original file digest and changed pointers."""
    def __init__(self, session, flight, audit, archive):
        self.aliases = []
        self.resolvers = []
        old_flight = str(audit['run']).rstrip('/\\')
        old_code = str(audit['replay_code_root']).rstrip('/\\')
        local_code = ROOT / 'v2' if (ROOT / 'v2/quadrl/policy.py').is_file() else ROOT
        for original, symbol, local in [
            (old_flight, '@session/flight', flight),
            (str(archive['archived']).rstrip('/\\'), '@session/flight', flight),
            (str(session), '@session', session),
            (str(Path(old_flight).parent), '@session', session),
            (old_code, '@code', local_code),
            (str(local_code), '@code', local_code),
            (str(Path(old_code).parent) if Path(old_code).name == 'v2' else old_code, '@project', ROOT),
            (str(ROOT), '@project', ROOT),
        ]:
            if original:
                self.aliases.append((original, symbol))
                self.resolvers.append((original, Path(local)))
        self.aliases.sort(key=lambda item: len(item[0]), reverse=True)
        self.resolvers.sort(key=lambda item: len(item[0]), reverse=True)
        self.changed = set()

    def string(self, value):
        expanded = unicode_paths(value)
        result = expanded if PII.search(expanded) else value
        for original, symbol in self.aliases:
            result = result.replace(original, symbol)
        if absolute(result):
            return '@external/' + hashlib.sha256(result.encode()).hexdigest()[:16]
        return PRIVATE_PATH.sub(lambda match: '@external/' + hashlib.sha256(match[0].encode()).hexdigest()[:16], result)

    def walk(self, value, pointer=''):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                public_key = self.string(key)
                require(public_key not in result, 'Projection key collision')
                if public_key != key:
                    self.changed.add(pointer or '/')
                result[public_key] = self.walk(item, pointer + '/' + public_key.replace('~', '~0').replace('/', '~1'))
            return result
        if isinstance(value, list):
            return [self.walk(item, pointer + '/' + str(i)) for i, item in enumerate(value)]
        if isinstance(value, str):
            result = self.string(value)
            if result != value:
                self.changed.add(pointer or '/')
            return result
        return value

    def document(self, value, original_sha):
        self.changed.clear()
        result = self.walk(deepcopy(value))
        require('_projection' not in result, 'Original document already contains projection metadata')
        result['_projection'] = dict(
            original_sha256=original_sha, changed_fields=sorted(self.changed),
            kind='PUBLIC_DISPLAY_PROJECTION_NOT_ORIGINAL_GUARD_BYTES',
            integrity_is_not_scientific_acceptance=True,
            symbolic_roots={'@session': 'original complete presentation session',
                            '@code': 'byte-bound inference source root',
                            '@project': 'source project root',
                            '@external': 'opaque original-path identifier; original bytes bound by digest'})
        return result

    def resolve(self, value):
        for original, local in self.resolvers:
            if value == original:
                return local
            if value.startswith(original + '/'):
                return local / relative(value[len(original) + 1:])
        return Path(value)

def vector(value, name):
    require(isinstance(value, list) and len(value) == 3
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value),
            'Invalid recorded vector: ' + name)
    return value

def csv_bytes(columns, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode('utf-8')

def prepare(session, name):
    source = Inputs()
    flight = session / 'flight'
    must = ['flight_archive.json', 'integrity_audit01.json', 'media/media.json',
            'gazebo_screen.mp4', 'gazebo_screen.mp4.json', 'flight/plan.json', 'flight/complete.json']
    missing = [p for p in must if not (session / p).is_file()]
    require(not missing, 'Incomplete session; missing required evidence: ' + ', '.join(missing))
    audit_path = session / 'integrity_audit01.json'
    audit = source.read(audit_path)
    require(audit.get('status') == 'PASS' and audit.get('integrity_passed') is True
            and audit.get('failed_checks') == [], 'Integrity audit is not PASS; failed attempts remain archived, not exported as valid evidence')
    archive = source.read(session / 'flight_archive.json')
    require(archive.get('status') == 'BYTE_EXACT_COPY', 'Byte-exact flight archive is required')
    archive_paths = set()
    for row in archive['files']:
        path = relative(row['path'])
        require(path.as_posix() not in archive_paths, 'Duplicate archived file')
        archive_paths.add(path.as_posix())
        require(not (flight / path).is_symlink(), 'Archived symlink is forbidden')
        source.bind(flight / path, row['sha256'], row['bytes'])
    actual_paths = {p.relative_to(flight).as_posix() for p in flight.rglob('*') if p.is_file()}
    require(actual_paths == archive_paths, 'Archived flight file set differs from BYTE_EXACT receipt')
    summary_path, plan_path = flight / 'complete.json', flight / 'plan.json'
    summary, plan = source.read(summary_path), source.read(plan_path)
    require(plan.get('mission') == name == audit.get('mission'), 'Requested name/plan/audit mission mismatch')
    require(summary.get('one_physical_world') is True and plan.get('physical_world_count') == 1,
            'One real physical world is required')
    require(set(summary['roles']) == set(audit['roles']) == set(ROLES), 'All three roles must be retained')
    projection = Projection(session, flight, audit, archive)
    guarded_inputs = []
    for original, expected in audit['input_sha256'].items():
        resolved = projection.resolve(original)
        source.bind(resolved, expected)
        guarded_inputs.append(dict(path=projection.string(original), sha256=expected, verified_now=True))
    source.bind(ROOT / 'demo_three/audit.py', audit['auditor_sha256'])
    media_path = session / 'media/media.json'
    media, capture = source.read(media_path), source.read(session / 'gazebo_screen.mp4.json')
    require(media.get('source') == 'actual Gazebo GUI X11 framebuffer recording'
            and capture.get('source') == 'real X11 window framebuffer'
            and capture.get('status') == 'CAPTURED', 'Completed actual X11 GUI capture is required')
    source.bind(session / 'gazebo_screen.mp4', media['source_video_sha256'], capture['bytes'])
    require(capture['sha256'] == media['source_video_sha256'], 'Raw video SHA differs between receipts')
    source.bind(session / 'gazebo_screen.mp4.json', media['capture_receipt_sha256'])
    source.bind(flight / 'groundtruth.jsonl', media['groundtruth_sha256'])
    source.bind(summary_path, media['summary_sha256'])
    require(media.get('mission') == name and media.get('scenario') == plan['scenario']
            and media.get('seed') == plan['seed'], 'Media and flight identity differ')
    require(media.get('playback_speed') == 1.0, 'Public media must preserve actual wall-time speed')
    for key in ('measurement_sim_start_s', 'measurement_sim_end_s'):
        require(media[key] == summary[key], 'Media measurement interval differs: ' + key)
    failures = {role: summary['roles'][role]['failure'] for role in ROLES}
    require(media['all_recorded_controller_failures'] == failures, 'Media failure ledger differs from actual summary')
    require({'gazebo.mp4', 'gazebo.gif'} <= set(media['files']), 'Both derived GUI MP4/GIF are required')
    for file, info in media['files'].items():
        path = relative(file)
        require(len(path.parts) == 1, 'Derived media must be a direct file of media/')
        source.bind(session / 'media' / path, info['sha256'], info['bytes'])

    outputs = {}
    outputs['audit.json'] = encode(projection.document(audit, source.bind(audit_path)))
    summary_bytes = summary_path.read_bytes()
    # Keep complete.json bytes exactly when public-safe, including valid failures.
    try:
        privacy(summary_bytes)
        outputs['summary.json'] = summary_bytes
    except Rejected:
        outputs['summary.json'] = encode(projection.document(summary, source.bind(summary_path)))
    outputs['plan.json'] = encode(projection.document(plan, source.bind(plan_path)))
    ledger = []
    for item in audit['actual_common_force_ledger']:
        force = vector(item['force_enu_N'], 'actual common force')
        ledger.append(dict(sequence=item['seq'], world_sim_time_s=item['sim_time'],
                           measurement_time_s=item['measurement_t'], mission_time_s=item.get('mission_time_s', ''),
                           physics_iteration=item['iteration'], force_x_N=force[0], force_y_N=force[1], force_z_N=force[2],
                           original_event_json=json.dumps(projection.walk(item), ensure_ascii=False, separators=(',', ':'), allow_nan=False)))
    outputs['actualforceledger.csv'] = csv_bytes(
        ['sequence', 'world_sim_time_s', 'measurement_time_s', 'mission_time_s', 'physics_iteration',
         'force_x_N', 'force_y_N', 'force_z_N', 'original_event_json'], ledger)
    trajectory, counts, mission_counts = [], {}, {}
    for role in ROLES:
        path = flight / role / 'commands.jsonl'
        require(path.relative_to(flight).as_posix() in archive_paths, 'Commands are not bound by archive')
        rows = []
        with path.open('rb') as stream:
            for index, line in enumerate(stream):
                require(bool(line.strip()), 'Blank command record')
                item = parse(line)
                require(item.get('role') == role, 'Recorded role mismatch')
                p, v, target = (vector(item[key], role + ':' + key)
                                for key in ('task_position', 'velocity', 'target_before_update'))
                stamp = item['measurement_t']
                require(isinstance(stamp, (int, float)) and math.isfinite(stamp), 'Nonfinite measurement clock')
                mission = item.get('mission_time_s', '')
                require(mission == '' or isinstance(mission, (int, float)) and math.isfinite(mission), 'Invalid explicit mission clock')
                rows.append(dict(role=role, row_index=index, world_sim_time_s=item['world_sim_time'],
                                 measurement_time_s=stamp, mission_time_s=mission,
                                 mission_epoch_boot_ms=item.get('mission_epoch_boot_ms', ''),
                                 mission_epoch_boot_s=item.get('mission_epoch_boot_s', ''),
                                 observation_boot_s=item.get('observation_boot_s', ''),
                                 shared_control_slot=item.get('shared_control_slot', ''),
                                 position_x_m=p[0], position_y_m=p[1], position_z_m=p[2],
                                 target_x_m=target[0], target_y_m=target[1], target_z_m=target[2],
                                 velocity_x_mps=v[0], velocity_y_mps=v[1], velocity_z_mps=v[2],
                                 goal_event_json='' if item.get('goal_event') is None else json.dumps(
                                     projection.walk(item['goal_event']), ensure_ascii=False, separators=(',', ':'), allow_nan=False)))
        counts[role] = len(rows)
        mission_counts[role] = sum(r['mission_time_s'] != '' for r in rows)
        require(len(rows) == summary['roles'][role]['samples'] == audit['roles'][role]['actual_commands'],
                'All actual command rows must be retained: ' + role)
        trajectory.extend(rows)
    columns = ['role', 'row_index', 'world_sim_time_s', 'measurement_time_s', 'mission_time_s',
               'mission_epoch_boot_ms', 'mission_epoch_boot_s', 'observation_boot_s', 'shared_control_slot',
               'position_x_m', 'position_y_m', 'position_z_m', 'target_x_m', 'target_y_m', 'target_z_m',
               'velocity_x_mps', 'velocity_y_mps', 'velocity_z_mps', 'goal_event_json']
    outputs['trajectories.csv'] = csv_bytes(columns, trajectory)
    source.unchanged()
    originals = [dict(path=projection.string(str(path)), sha256=row['sha256'], bytes=row['bytes'])
                 for path, row in sorted(source.files.items())]
    payload = dict(schema='public-shared-world-measured-export-v1.0', mission=name,
                   scope='One actual shared-world demonstration; integrity PASS is not scientific acceptance',
                   raw_files_copied=False, videos_copied=False, archive_status=archive['status'],
                   original_integrity_audit_sha256=source.bind(audit_path),
                   original_archive_receipt_sha256=source.bind(session / 'flight_archive.json'),
                   original_media_receipt_sha256=source.bind(media_path),
                   summary_bytes_unchanged=outputs['summary.json'] == summary_bytes,
                   actual_command_rows=counts, explicit_mission_clock_rows=mission_counts,
                   mission_clock_rule='Copy actual mission_time_s only; never reconstruct old mission time',
                   controller_failures=projection.walk(failures),
                   route_completed={r: summary['roles'][r]['route_completed'] for r in ROLES},
                   original_inputs=originals, independently_audited_input_sha256=guarded_inputs,
                   media=projection.walk(media),
                   output_files={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                                 for name, data in outputs.items()},
                   exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   new_environment_steps=0, new_NN_forwards=0, new_optimizer_steps=0,
                   scientific_table_updated=False, failures_or_incomplete_rows_filtered=False)
    outputs['sources.json'] = encode(payload)
    for data in outputs.values():
        privacy(data)
    return outputs, counts

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--name', choices=('route', 'hover'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a fresh output; existing exports are preserved')
    original_layout = output.name == args.name and output.parent.name == 'shared_world' and output.parent.parent.name == 'results'
    radius_layout = (args.name == 'route' and output.name in ('r025', 'r040')
                     and output.parent.name == 'waypoint_radius' and output.parent.parent.name == 'results')
    if not (original_layout or radius_layout):
        parser.error('Use results/shared_world/<name> or results/waypoint_radius/r025|r040')
    if radius_layout:
        plan = json.loads((args.session.resolve() / 'flight/plan.json').read_text(encoding='utf-8'))
        expected_radius = {'r025': .25, 'r040': .40}[output.name]
        if plan.get('mission_acceptance', {}).get('waypoint_radius_m') != expected_radius:
            parser.error('Output radius label does not match the recorded mission contract')
    try:
        blobs, counts = prepare(args.session.resolve(), args.name)
    except (Rejected, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, 'REJECTED_NO_OUTPUT_CREATED: ' + str(exc) + '\n')
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, data in blobs.items():
            with (output / name).open('xb') as stream:
                stream.write(data)
    except BaseException:
        (output / 'EXPORT_INCOMPLETE.txt').write_text('Export failed; preserve this attempt and choose a fresh output.\n')
        raise
    print(json.dumps(dict(status='PASS_PUBLIC_DATA_EXPORT_INTEGRITY_ONLY', files=len(blobs),
                          actual_command_rows=counts, new_NN_forwards=0, new_environment_steps=0,
                          new_optimizer_steps=0, scientific_acceptance_claim=False), indent=2))

if __name__ == '__main__':
    main()
