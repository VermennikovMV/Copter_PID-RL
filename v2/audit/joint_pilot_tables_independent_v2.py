"""Independent read-only check of the sealed development pilot and its tables.

Imports no production metric, table, simulator, actor, or controller code.
Only the named completed 83301 pilot and its report are opened; the sole output
is this script's JSON receipt. No reserved confirmation results are discovered.
"""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'artifacts/refinement/joint_v2/gazebo_geometry_pilot32'
REPORT = ROOT / 'reports/joint_geometry_pilot_v2'
CONTRACT = ROOT / 'docs/joint_applicability_contract_v2.json'
OUTPUT = ROOT / 'audit/joint_pilot_tables_independent_v2.json'
POINTS = np.array([[0, 0, 3], [2, 0, 3], [2, 2, 3.5], [0, 2, 3], [0, 0, 3.]])
METHODS = ('PID', 'BC111_body_collective', 'wind111_body_collective',
           'controlPPO111_body_collective', 'geometryPPO111_body_collective')
SCENARIOS = ('nominal', 'force_strong', 'gust')
checks = {}
field_comparisons = 0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def ensure(value, message):
    if not value:
        raise AssertionError(message)


def same(actual, expected, message):
    global field_comparisons
    field_comparisons += 1
    if expected is None:
        ensure(actual in (None, ''), message + ': expected null')
    elif isinstance(expected, (bool, np.bool_)):
        ensure(actual == bool(expected) if not isinstance(actual, str) else actual == str(bool(expected)), message)
    elif isinstance(expected, (int, float, np.integer, np.floating)):
        ensure(actual not in (None, '') and np.isfinite(float(actual))
               and abs(float(actual) - float(expected)) <= 1e-9, message)
    else:
        ensure(actual == expected, message)


def path_for(value):
    path = Path(value)
    ensure(path.resolve().is_relative_to(PILOT), 'Evidence outside explicit completed pilot')
    return path


def passed(receipt):
    ensure(receipt['status'] == 'PASS', 'Non-PASS independent receipt')
    items = receipt.get('checks', {})
    ensure(all(item['pass'] is True for item in items) if isinstance(items, list)
           else all(v is True for v in items.values()), 'Failed independent receipt check')


def overlaps(t, start, end):
    return np.maximum(0., np.minimum(t, end) - np.maximum(np.r_[0., t[:-1]], start))


def rmse(error, weight):
    return float(np.sqrt(np.sum(error * error * weight) / np.sum(weight)))


def distance(p, a, b, finite=True):
    v = b - a
    if np.dot(v, v) == 0:
        return np.linalg.norm(p - a, axis=-1)
    projection = np.sum((p - a) * v, axis=-1) / np.dot(v, v)
    if finite:
        projection = np.maximum(0., np.minimum(1., projection))
    return np.linalg.norm(p - a - projection[..., None] * v, axis=-1)


def continuous(t, start, end):
    interior = t[(t > start + 1e-8) & (t < end - 1e-8)]
    return bool(end > start and t[0] <= start + .060001 and t[-1] >= end - 1e-8
                and np.max(np.diff(np.r_[start, interior, end])) <= .060001)


def independent_metrics(raw, summary):
    t = np.array([float(r['t']) for r in raw])
    p = np.array([[float(r[k]) for k in ('x_e', 'y_n', 'z_up')] for r in raw])
    goal = np.array([[float(r[k]) for k in ('target_e', 'target_n', 'target_up')] for r in raw])
    yaw = np.array([float(r['yaw_error_rad']) for r in raw])
    ensure(all(np.isfinite(a).all() for a in (t, p, goal, yaw)) and np.all(np.diff(t) > 0), 'Nonfinite/nonmonotone raw')
    horizon = 30. if summary['mission'] == 'hover' else 40.
    same(summary['requested_duration_s'], horizon, 'Common horizon')
    duration = min(float(t[-1]), horizon)
    failure = summary.get('failure') or (summary.get('error') if summary['status'] != 'PASS' else None)
    survived = failure is None and duration >= horizon - .040000001
    error = np.linalg.norm(p - goal, axis=1)
    w = np.diff(np.r_[0., np.minimum(t, horizon)])
    covered = float(w.sum())
    integral = float(np.sum(w * error ** 2))
    tail = max(0., horizon - covered)
    heading = np.arctan2(np.sin(yaw), np.cos(yaw)) * 180. / np.pi
    completed = summary['mission'] == 'route' and summary.get('completion_time_s') is not None
    metric = dict(survived=survived, rmse_position_m=rmse(error, w),
                  penalized_rmse_position_m=float(np.sqrt((integral + tail * (error[-1] ** 2 if survived else 16.)) / horizon)),
                  altitude_drop_m=float(max(0., np.max(goal[:, 2] - p[:, 2]))),
                  heading_rmse_deg=rmse(heading, w), heading_peak_deg=float(np.max(np.abs(heading))),
                  heading_full_flight_eligible=survived, duration_s=duration, horizon_s=horizon,
                  heading_guard_pass=bool(rmse(heading, w) <= 5. and np.max(np.abs(heading)) <= 15.),
                  route_completed=completed if summary['mission'] == 'route' else None,
                  route_completion_time_s=float(summary['completion_time_s']) if completed else None)
    geometry = None
    if summary['mission'] == 'route':
        events = [e for e in summary['mission_events'] if e['type'] == 'goal_reached']
        ensure([e['goal_index'] for e in events] == list(range(5)), 'Five ordered confirmed arrivals')
        for e in events:
            same(e['time_s'] - e['dwell_started_s'], .5, 'Confirmed half-second dwell')
            ensure(np.allclose(e['target'], POINTS[e['goal_index']], rtol=0, atol=1e-9), 'Fixed route points')
        times = np.array([e['time_s'] for e in events])
        same(times[-1], summary['completion_time_s'], 'Fifth arrival completion time')
        active = np.array([min(4, int(np.count_nonzero(times < stamp - 1e-8))) for stamp in t])
        ensure(np.allclose(goal, POINTS[active], rtol=0, atol=1e-7), 'Recorded pre-transition active goal')
        seg = np.array([distance(p[i], POINTS[max(0, j - 1)], POINTS[j]) for i, j in enumerate(active)])
        line_xy = np.array([distance(p[i, :2], POINTS[max(0, j - 1), :2], POINTS[j, :2], False)
                            for i, j in enumerate(active)])
        wtransit = overlaps(t, times[0], times[-1])
        ensure(continuous(t, times[0], times[-1]), 'Complete transit sample coverage')
        same(wtransit.sum(), times[-1] - times[0], 'Complete transit duration')
        corner_peaks = []
        for event in events[1:4]:
            j, stamp = event['goal_index'], event['time_s']
            mask = (t >= stamp - 1e-8) & (t <= stamp + 1. + 1e-8)
            ensure(continuous(t, stamp, stamp + 1.) and np.any(mask), 'Complete closed one-second corner window')
            corner_peaks.append(float(np.max(distance(p[mask], POINTS[j], POINTS[j + 1]))))
        geometry = dict(segment_rmse_m=rmse(seg, wtransit), cross_track_xy_rmse_m=rmse(line_xy, wtransit),
                        mean_corner_post_peak_m=float(np.mean(corner_peaks)),
                        active_segment_peak_m=float(np.max(seg[wtransit > 0.])),
                        geometry_eligible=True, observed_corners=3, complete_corner_windows=3,
                        transit_duration_s=float(times[-1] - times[0]), observed_transit_coverage_s=float(wtransit.sum()),
                        max_sample_gap_s=float(np.max(np.diff(t))))
    return metric, geometry


def main():
    complete, plan, contract, report = read(PILOT / 'complete.json'), read(PILOT / 'plan.json'), read(CONTRACT), read(REPORT / 'operational_gate.json')
    ensure(complete['status'] == 'COMPLETE' and len(complete['results']) == len(plan['jobs']) == 32, 'Complete fixed 32')
    ensure({r['job']['name'] for r in complete['results']} == {j['name'] for j in plan['jobs']}, 'Exact planned matrix')
    same(complete['plan_sha256'], sha(PILOT / 'plan.json'), 'Plan hash')
    same(plan['fixed_quality_contract_sha256'], sha(CONTRACT), 'Fixed prospective contract')
    integrity = read(ROOT / 'audit/joint_geometry_pilot32_integrity_v2.json')
    passed(integrity)
    same(integrity['complete_sha256'], sha(PILOT / 'complete.json'), 'Independent integrity binds completed campaign')
    same(integrity['plan_sha256'], sha(PILOT / 'plan.json'), 'Independent integrity binds plan')
    for key, path in [('source_sha256', ROOT / 'scripts/build_joint_geometry_pilot_v2.py'),
                      ('case_audit_helper_sha256', ROOT / 'scripts/build_joint_applicability_v2.py'),
                      ('contract_sha256', CONTRACT), ('campaign_complete_sha256', PILOT / 'complete.json'),
                      ('campaign_plan_sha256', PILOT / 'plan.json')]:
        same(report[key], sha(path), 'Current report provenance ' + key)
    for source, copy in [('scripts/build_joint_geometry_pilot_v2.py', 'source_builder.py'),
                         ('scripts/build_joint_applicability_v2.py', 'source_case_audit.py'),
                         ('docs/joint_applicability_contract_v2.json', 'fixed_contract.json')]:
        same(sha(ROOT / source), sha(REPORT / copy), 'Archived report source ' + copy)
    cases = {}; wire = 0; audit_count = dict(physical=0, force=0, metric=0, actor=0)
    evidence = {c['name']: c for c in report['case_evidence']}
    for result in complete['results']:
        job = result['job']; run = path_for(result['run']); name = job['name']
        summary = read(run / 'summary.json')
        same(summary['seed'], 83301, 'Only explicit development seed')
        same(job['environment_seed'], 83301, 'Declared development seed')
        same(summary['scenario'], job['scenario'], 'Declared scenario')
        same(summary['mission'], job['mission'], 'Declared mission')
        same(sha(run / 'source_snapshot/pid_profile.json'), contract['frozen_pid_profile']['sha256'], 'Same competent frozen PID')
        for kind in ('physical', 'force', 'metric') + (() if job['stage'] == 'PID' else ('actor',)):
            receipt_path = path_for(result[kind + '_audit_receipt'])
            receipt = read(receipt_path); passed(receipt); audit_count[kind] += 1
            same(result[kind + '_audit_sha256'], sha(receipt_path), 'Bound ' + kind + ' receipt')
            if kind == 'physical':
                for filename in ('summary.json', 'telemetry.csv', 'groundtruth_independent.jsonl'):
                    same(receipt['sha256'][filename], sha(run / filename), 'Physical raw hash ' + filename)
            if kind == 'metric':
                independent = receipt['metrics']
                same(receipt['telemetry_sha256'], sha(run / 'telemetry.csv'), 'Independent metric raw CSV hash')
                same(receipt['summary_sha256'], sha(run / 'summary.json'), 'Independent metric summary hash')
            if kind == 'actor':
                same(receipt['policy_sha256'], summary['policy_sha256'], 'Actor policy identity')
                same(summary['policy_sha256'], sha(run / 'source_snapshot/policy.npz'), 'Actor archived NPZ')
                actor_check = next(c for c in receipt['checks'] if c['check'] == 'raw_MAV_stateful_encoder_MLP_reproduces_all_actions')
                same(actor_check['observation_size'], 74, 'Pure measured 74-input encoder')
                exact = next(c for c in receipt['checks'] if c['check'] == 'exclusive_exact_CRC_valid_wire_commands')
                wire += exact['expected_commands']
        metric, geometry = independent_metrics(rows(run / 'telemetry.csv'), summary)
        for own, other in [('rmse_position_m', 'rmse_m'), ('penalized_rmse_position_m', 'penalized_rmse_m'),
                           ('altitude_drop_m', 'max_altitude_drop_m'), ('heading_rmse_deg', 'heading_rmse_deg'),
                           ('heading_peak_deg', 'heading_peak_deg'), ('survived', 'survived')]:
            same(metric[own], independent[other], 'Independent five-metric audit ' + name + '/' + own)
        for key, value in metric.items():
            same(evidence[name]['metrics'][key], value, 'Report case metric ' + name + '/' + key)
        if geometry:
            for key, value in geometry.items():
                same(evidence[name]['geometry'][key], value, 'Report case geometry ' + name + '/' + key)
            same(independent['route']['completion_time_s'], metric['route_completion_time_s'], 'Independently verified dwell completion')
            same(independent['route']['completed'], True, 'Independent route complete')
        cases[(job['stage'], job['scenario'], job['mission'])] = dict(job=job, run=run, summary=summary, metric=metric, geometry=geometry)
    same(audit_count, dict(physical=32, force=32, metric=32, actor=26), 'All denominators')
    same(wire, 44967, 'All exact-wire NN commands')
    checks['sealed_matrix_raw_hashes_independent_receipts_and_shared_PID'] = True
    checks['independent_raw_goal_heading_and_finite3D_transit_corner_metrics'] = True

    route_rows, hover_rows, map_rows = [rows(REPORT / (stem + '.csv')) for stem in ('route_metrics', 'hover_metrics', 'applicability')]
    ensure(len(route_rows) == 15 and len(hover_rows) == 17 and len(map_rows) == 6, 'Report row denominators')
    expected_keys = {(m, s, mission) for m in METHODS for s in SCENARIOS for mission in ('hover', 'route')}
    expected_keys |= {('wind111_legacy_preboost', s, 'hover') for s in ('force_strong', 'gust')}
    same(set(cases), expected_keys, 'Exact controller/scenario/mission coverage')
    gate = contract['joint_operational_quality_gate']; outcomes = {}; rowlookup = {}
    for table in (route_rows, hover_rows):
        for row in table:
            method, scenario, mission = row['method'], row['scenario'], row['mission']
            key = method, scenario, mission
            case = cases[key]; metric = case['metric']; pid = cases[('PID', scenario, mission)]['metric']
            expected = dict(dataset='JOINT_GEOMETRY_PILOT32', method=method, mission=mission, scenario=scenario,
                            environment_seed=83301, training_seed=None if method == 'PID' else 111, episodes=1,
                            interface=case['summary']['joint_interface']['effective_semantics'],
                            goal_rmse_full_horizon_m=metric['penalized_rmse_position_m'],
                            heading_rmse_deg=metric['heading_rmse_deg'], heading_peak_deg=metric['heading_peak_deg'],
                            survived=metric['survived'], run=str(case['run'].relative_to(ROOT)), evidence_status='DEVELOPMENT_1ENV',
                            evidence_label='PHYSICAL_METRICS_PASS_ACTOR_NA' if method == 'PID' else 'PHYSICAL_ACTOR_METRICS_PASS')
            guards = dict(survival=metric['survived'], heading_full_flight=metric['heading_full_flight_eligible'])
            if mission == 'hover':
                limit = min(.05, max(1.2 * pid['penalized_rmse_position_m'], pid['penalized_rmse_position_m'] + .02)) if scenario == 'nominal' else .85 * pid['penalized_rmse_position_m']
                guards.update(heading_rms=metric['heading_rmse_deg'] <= 5., heading_peak=metric['heading_peak_deg'] <= 15.)
                if method != 'PID':
                    guards['position_RMSE'] = metric['penalized_rmse_position_m'] <= limit
                retention = retention_limit = None
                if scenario != 'nominal' and method != 'PID':
                    specialist = cases[('wind111_legacy_preboost', scenario, 'hover')]['metric']['penalized_rmse_position_m']
                    retention, retention_limit = metric['penalized_rmse_position_m'] / specialist, 1.10 * specialist
                    guards['specialist_retention'] = metric['penalized_rmse_position_m'] <= retention_limit
                expected.update(gain_vs_PID_pct=100 * (1 - metric['penalized_rmse_position_m'] / pid['penalized_rmse_position_m']),
                                rmse_limit_m=limit, specialist_retention_ratio=retention, specialist_retention_limit_m=retention_limit,
                                conjunctive_hover_guards_pass=all(guards.values()))
            else:
                geom = case['geometry']; pg = cases[('PID', scenario, 'route')]['geometry']
                elapsed = metric['route_completion_time_s'] if metric['route_completed'] else 40.
                ptime = pid['route_completion_time_s'] if pid['route_completed'] else 40.
                limits = dict(time_s=1.05 * ptime, goal_rmse_m=1.05 * pid['penalized_rmse_position_m'],
                              finite3d_rmse_m=max(1.10 * pg['segment_rmse_m'], pg['segment_rmse_m'] + .01),
                              corner_peak_m=max(1.10 * pg['mean_corner_post_peak_m'], pg['mean_corner_post_peak_m'] + .02), absolute_peak_m=.5)
                guards = dict(survival=metric['survived'], completion=metric['route_completed'], heading_full_flight=metric['heading_full_flight_eligible'],
                              heading_rmse=metric['heading_rmse_deg'] <= 5., heading_peak=metric['heading_peak_deg'] <= 15.,
                              restricted_time=elapsed <= limits['time_s'], full_horizon_goal_rmse=metric['penalized_rmse_position_m'] <= limits['goal_rmse_m'],
                              complete_geometry=geom['geometry_eligible'], finite3d_rmse=geom['segment_rmse_m'] <= limits['finite3d_rmse_m'],
                              corner_peak=geom['mean_corner_post_peak_m'] <= limits['corner_peak_m'], absolute_peak=geom['active_segment_peak_m'] <= .5)
                expected.update(environment_count=1, trained_model_count=0 if method == 'PID' else 1,
                                restricted_time_s=elapsed, finite3d_transit_rmse_m=geom['segment_rmse_m'], xy_cross_track_rmse_m=geom['cross_track_xy_rmse_m'],
                                mean_corner_peak_m=geom['mean_corner_post_peak_m'], max_segment_deviation_m=geom['active_segment_peak_m'],
                                speed_gain_vs_pid_pct=100 * (1 - elapsed / ptime),
                                goal_rmse_gain_vs_pid_pct=100 * (1 - metric['penalized_rmse_position_m'] / pid['penalized_rmse_position_m']),
                                finite3d_rmse_change_vs_pid_pct=100 * (geom['segment_rmse_m'] / pg['segment_rmse_m'] - 1),
                                corner_peak_change_vs_pid_pct=100 * (geom['mean_corner_post_peak_m'] / pg['mean_corner_post_peak_m'] - 1),
                                completed=metric['route_completed'], geometry_guard_pass=all(guards[k] for k in ('complete_geometry', 'finite3d_rmse', 'corner_peak', 'absolute_peak')),
                                conjunctive_route_guards_pass=all(guards.values()), **{'guard_' + k: v for k, v in guards.items()}, **{'limit_' + k: v for k, v in limits.items()})
            expected['failed_guards'] = ';'.join(k for k, value in guards.items() if not value)
            same(set(row), set(expected), 'Every table column covered')
            for field, value in expected.items():
                same(row[field], value, 'CSV ' + str(key) + '/' + field)
            outcomes[key] = all(guards.values()), expected['failed_guards']
            rowlookup[key] = expected
    checks['all_csv_absolute_values_ratios_denominators_limits_and_conjoined_guards'] = True

    chosen = 'geometryPPO111_body_collective'
    for row in map_rows:
        scenario, mission = row['scenario'], row['mission']; value = rowlookup[(chosen, scenario, mission)]; pid = rowlookup[('PID', scenario, mission)]
        passed_cell, failures = outcomes[(chosen, scenario, mission)]
        recommendation = ('PID for minimum nominal error; both meet tolerance' if (scenario, mission) == ('nominal', 'hover') else
                          'RL within measured development tolerance' if passed_cell else 'Accuracy-speed tradeoff; fixed geometry gate failed')
        expected = dict(scenario=scenario, mission=mission, policy_version='geometry111_body_collective_v2.1', status='DEVELOPMENT_1ENV',
                        cell_operational_quality_pass=passed_cell, recommendation=recommendation, model_flights=1, PID_flights=1, environment_seeds=1, trained_models=1,
                        model_goal_RMSE_m=value['goal_rmse_full_horizon_m'], PID_goal_RMSE_m=pid['goal_rmse_full_horizon_m'],
                        model_route_time_s=value.get('restricted_time_s'), PID_route_time_s=pid.get('restricted_time_s'),
                        model_finite3D_RMSE_m=value.get('finite3d_transit_rmse_m'), PID_finite3D_RMSE_m=pid.get('finite3d_transit_rmse_m'),
                        failed_guards=failures, model_run=value['run'], PID_run=pid['run'])
        same(set(row), set(expected), 'Every applicability column covered')
        for field, v in expected.items():
            same(row[field], v, 'Applicability ' + scenario + '/' + mission + '/' + field)
    for method in METHODS[1:]:
        expected_failures = [s + '/' + mission + ':' + outcomes[(method, s, mission)][1]
                             for mission in ('route', 'hover') for s in SCENARIOS if not outcomes[(method, s, mission)][0]]
        item = report['gate'][method]
        same(item['failed_cells'], expected_failures, 'All six-cell failures retained ' + method)
        same(item['development_primary_six_cells_pass'], not expected_failures, 'Six-cell conjunction ' + method)
        for key, value in dict(episodes=6, environment_count=1, training_models=1, final_applicability_confirmed=False,
                               original_wind_macro='NOT_COVERED_MILD_MISSING').items():
            same(item[key], value, 'Scope ' + method + '/' + key)
    for key, value in dict(status='DEVELOPMENT_ONLY', episodes=32, final_applicability_confirmed=False,
                           original_wind_macro='NOT_COVERED_MILD_MISSING', reserved_results_opened=False).items():
        same(report[key], value, 'Report scope ' + key)
    checks['map_recommendations_and_n1_development_scope_no_original_macro_claim'] = True
    checks['exact_report_source_contract_and_receipt_hashes'] = True
    receipt = dict(schema='joint-pilot-tables-independent-v2.0', status='PASS', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   checks=checks, field_comparisons=field_comparisons, report_rows=dict(route=15, hover=17, applicability=6),
                   independent_receipts=audit_count, exact_NN_wire_commands=wire,
                   opened_scope='Only explicitly named completed Gazebo geometry pilot32, fixed contract, report sources/outputs and integrity receipt',
                   reserved_result_discovery_or_read=False, raw_recomputation='Standalone NumPy integrals, finite active segments and all three closed corner windows; no production metric or builder imports',
                   report_sha256={f: sha(REPORT / f) for f in ('route_metrics.csv', 'hover_metrics.csv', 'applicability.csv', 'operational_gate.json')},
                   builder_sha256=sha(ROOT / 'scripts/build_joint_geometry_pilot_v2.py'),
                   case_helper_sha256=sha(ROOT / 'scripts/build_joint_applicability_v2.py'), contract_sha256=sha(CONTRACT), source_sha256=sha(Path(__file__)))
    OUTPUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: receipt[k] for k in ('status', 'field_comparisons', 'report_rows', 'independent_receipts', 'exact_NN_wire_commands')}))


if __name__ == '__main__':
    main()
