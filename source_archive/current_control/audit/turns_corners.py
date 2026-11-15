#!/usr/bin/env python3
"""Snapshot diagnostics around real Gazebo route corners; never run a controller."""
import argparse,collections,csv,datetime,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
POINTS=((0.,0.,3.),(2.,0.,3.),(2.,2.,3.5),(0.,2.,3.),(0.,0.,3.))
METRICS=('corner_min_approach_m','post_outgoing_max_deviation_m','post_outgoing_mean_deviation_m',
         'post_incoming_overshoot_m','post_new_target_rmse_m','pre_corner_rmse_m')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(v):return math.sqrt(sum(x*x for x in v))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def distance(a,b):return norm(sub(a,b))
def mean(values):
    values=[v for v in values if v is not None];return sum(values)/len(values) if values else None
def segment_distance(position,a,b):
    ab=sub(b,a);fraction=max(0.,min(1.,dot(sub(position,a),ab)/dot(ab,ab)))
    return distance(position,tuple(x+fraction*y for x,y in zip(a,ab)))
def angle(a,b):return math.degrees(math.acos(max(-1.,min(1.,dot(a,b)/(norm(a)*norm(b))))))
def weighted(rows,start,end,value,squared=False):
    terms=[];previous=0.
    for row in rows:
        weight=max(0.,min(row['t'],end)-max(previous,start));previous=row['t']
        if weight:terms.append((weight,value(row)))
    if not terms:return None
    result=sum(w*(v*v if squared else v) for w,v in terms)/sum(w for w,v in terms)
    return math.sqrt(result) if squared else result

def analyse(path,models,pid_hash):
    summary=json.loads(path.read_text());method=summary['controller'];seed=summary.get('train_seed')
    if method not in ('pid','hybrid','direct'):raise ValueError('Unknown method')
    if method!='pid' and summary.get('policy_sha256')!=models[(method,seed)]:raise ValueError('Unfrozen model identity')
    profile=path.parent/'source_snapshot/pid_profile.json'
    if sha(profile)!=pid_hash:raise ValueError('Different strong native PID initialization/profile')
    rows=[]
    trace=path.parent/'telemetry.csv'
    if trace.exists():
        with trace.open(newline='') as file:
            for raw in csv.DictReader(file):rows.append({'t':float(raw['t']),'position':tuple(float(raw[k]) for k in ('x_e','y_n','z_up'))})
    events=[]
    for raw in (json.loads(line) for line in (path.parent/'events.jsonl').read_text().splitlines()):
        if raw.get('event')=='goal_reached':events.append(raw['mission_event'])
        elif raw.get('event')=='mission_transition' and raw.get('type')=='goal_reached':events.append(raw)
    common={'run':str(path.parent.relative_to(ROOT)),'controller':method,'training_seed':seed,
            'scenario':summary['scenario'],'environment_seed':summary['seed'],'flight_status':summary['status']}
    corners=[]
    for event in events:
        index=event['goal_index']
        if index not in (1,2,3):continue
        t=event['time_s'];previous,corner,next_point=POINTS[index-1:index+2]
        if distance(event['target'],corner)>1e-9:raise ValueError('Route geometry mismatch')
        first,last=(rows[0]['t'],rows[-1]['t']) if rows else (t,t)
        full_pre=bool(rows and first<=t-.5+1e-9 and last>=t-1e-9)
        full_post=bool(rows and first<=t+1e-9 and last>=t+1-1e-9)
        full_approach=bool(rows and first<=t-.5+1e-9 and last>=t+.5-1e-9)
        observed_post=max(0.,min(last,t+1)-max(first,t)) if rows else 0.
        near=[r for r in rows if t-.5-1e-9<=r['t']<=t+.5+1e-9]
        after=[r for r in rows if t-1e-9<=r['t']<=t+1+1e-9]
        incoming=sub(corner,previous);outgoing=sub(next_point,corner);unit=tuple(x/norm(incoming) for x in incoming)
        corners.append({**common,'corner_index':index,'transition_s':t,
            'incoming_direction_change_3d_deg':angle(incoming,outgoing),'xy_direction_change_deg':90.,
            'pre_window_start_s':t-.5,'post_window_end_s':t+1.,
            'full_pre_window':full_pre,'full_post_window':full_post,'full_approach_window':full_approach,
            'post_window_censored':not full_post,'observed_post_window_s':observed_post,
            'corner_min_approach_m':min(distance(r['position'],corner) for r in near) if full_approach and near else None,
            'post_outgoing_max_deviation_m':max(segment_distance(r['position'],corner,next_point) for r in after) if full_post and after else None,
            'post_outgoing_mean_deviation_m':weighted(rows,t,t+1,lambda r:segment_distance(r['position'],corner,next_point)) if full_post else None,
            'post_incoming_overshoot_m':max(0.,max(dot(sub(r['position'],corner),unit) for r in after)) if full_post and after else None,
            'post_new_target_rmse_m':weighted(rows,t,t+1,lambda r:distance(r['position'],next_point),squared=True) if full_post else None,
            'pre_corner_rmse_m':weighted(rows,t-.5,t,lambda r:distance(r['position'],corner),squared=True) if full_pre else None,
            'observed_post_outgoing_max_deviation_m':max((segment_distance(r['position'],corner,next_point) for r in after),default=None),
            'observed_post_new_target_rmse_m':weighted(rows,max(first,t),min(last,t+1),lambda r:distance(r['position'],next_point),squared=True) if observed_post else None})
    if len({r['corner_index'] for r in corners})!=len(corners):raise ValueError(f'Duplicate corner events: {path}')
    flight={**common,'expected_corner_opportunities':3,'observed_corner_events':len(corners),'unreached_corner_events':3-len(corners),
            'full_post_corner_windows':sum(r['full_post_window'] for r in corners),
            'post_censored_corner_events':sum(r['post_window_censored'] for r in corners),
            'route_completed':summary.get('route_completed'),
            **{metric:mean(r[metric] for r in corners) for metric in METRICS}}
    provenance={**common,'files':{name:sha(path.parent/name) if (path.parent/name).exists() else None for name in ['summary.json','telemetry.csv','events.jsonl','validity.json']},
                'policy_sha256':summary.get('policy_sha256'),'pid_profile_sha256':sha(profile)}
    return corners,flight,provenance

def aggregates(flights,matched=False):
    groups=collections.defaultdict(list)
    for row in flights:groups[(row['controller'],row['scenario'])].append(row)
    result=[]
    for (method,scenario),group in sorted(groups.items()):
        cells=collections.defaultdict(list)
        for row in group:cells[row['environment_seed']].append(row)
        result.append({'controller':method,'scenario':scenario,'physical_flights':len(group),'environment_seeds':sorted(cells),
            'trained_seed_ids':sorted({r['training_seed'] for r in group if r['training_seed'] is not None}),
            'pass_flights':sum(r['flight_status']=='PASS' for r in group),'controller_fail_flights':sum(r['flight_status']=='FAIL' for r in group),
            'route_completed_flights':sum(r['route_completed'] is True for r in group),
            'expected_corner_opportunities':3*len(group),'interior_corner_observations':sum(r['observed_corner_events'] for r in group),
            'unreached_corner_events':sum(r['unreached_corner_events'] for r in group),
            'full_post_corner_windows':sum(r['full_post_corner_windows'] for r in group),
            'post_censored_corner_events':sum(r['post_censored_corner_events'] for r in group),
            'metric_flight_coverage':{metric:sum(r[metric] is not None for r in group) for metric in METRICS},
            'matched_condition_diagnostic':matched,
            **{metric:mean(mean(r[metric] for r in cell) for cell in cells.values()) for metric in METRICS}})
    return result

def csv_write(path,rows):
    if not rows:path.write_text('');return
    with path.open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    parser.add_argument('--require-main-complete',action='store_true');a=parser.parse_args()
    state_path=ROOT/'artifacts/provenance/main_direct_v2/state.json'
    state=json.loads(state_path.read_text())
    campaign_path=ROOT/'artifacts/gazebo_direct/main_direct_test_v2/campaign_summary.json'
    campaign=json.loads(campaign_path.read_text())
    if a.require_main_complete and (state.get('stage')!='complete' or campaign.get('status')!='COMPLETE' or campaign.get('completed')!=108):
        raise RuntimeError('Full-main snapshot requires completed108-case Direct campaign and completed coordinator audits')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=(a.output or ROOT/'audit/turns_corner_analysis'/stamp).resolve()
    if not out.is_relative_to(ROOT/'audit/turns_corner_analysis'):raise ValueError('Output must stay in audit/turns_corner_analysis')
    if out.exists():raise FileExistsError('Preserve the existing corner snapshot: '+str(out))
    freezes=[ROOT/'artifacts/provenance'/name/'frozen_contract.json' for name in ['main_hybrid_v1','main_direct_v2']]
    contracts=[json.loads(p.read_text()) for p in freezes]
    models={(m['architecture'],m['training_seed']):m['sha256'] for c in contracts for m in c['models']}
    pid_hash=contracts[0]['pid_profiles']['gazebo']['sha256']
    paths=sorted([*(ROOT/'artifacts/gazebo/main_hybrid_test_v1').glob('*/summary.json'),*(ROOT/'artifacts/gazebo_direct/main_direct_test_v2').glob('*/summary.json')])
    candidates=[];excluded=[];canonical_route_summaries=0
    for path in paths:
        try:s=json.loads(path.read_text())
        except ValueError:excluded.append({'path':str(path),'reason':'metadata write in progress'});continue
        if s.get('mission')!='route':continue
        canonical_route_summaries+=1
        validity_path=path.parent/'validity.json'
        validity=json.loads(validity_path.read_text()) if validity_path.exists() else {}
        invalid=any(source.get(key) is False for source in [s,validity] for key in ['valid','infrastructure_valid','valid_controller_comparison'])
        if invalid:excluded.append({'path':str(path),'reason':'explicit infrastructure/validity false','validity_sidecar':str(validity_path) if validity_path.exists() else None});continue
        if s.get('status') not in ('PASS','FAIL'):excluded.append({'path':str(path),'reason':'route status '+str(s.get('status'))});continue
        candidates.append(path)
    if a.require_main_complete and canonical_route_summaries!=126:raise ValueError('Expected126 canonical main route summaries after main completion')
    corners=[];flights=[];provenance=[]
    for path in candidates:
        c,f,p=analyse(path,models,pid_hash);corners.extend(c);flights.append(f);provenance.append(p)
    lookup={(r['controller'],r['training_seed'],r['scenario'],r['environment_seed']):r for r in flights}
    matched=[];matched_cells=[];full_cohort=[]
    for scenario,env in sorted({(r['scenario'],r['environment_seed']) for r in flights}):
        pid=lookup.get(('pid',None,scenario,env));seeds=[seed for seed in [111,222,333] if ('direct',seed,scenario,env) in lookup and ('hybrid',seed,scenario,env) in lookup]
        if not pid or not seeds:continue
        matched.append(pid)
        for seed in seeds:
            matched.extend([lookup['hybrid',seed,scenario,env],lookup['direct',seed,scenario,env]])
        matched_cells.append({'scenario':scenario,'environment_seed':env,'paired_training_seeds':seeds})
        if len(seeds)==3:full_cohort.append({'scenario':scenario,'environment_seed':env})
    main_complete=state.get('stage')=='complete' and campaign.get('status')=='COMPLETE' and campaign.get('completed')==108
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'MAIN_COMPLETED_DIAGNOSTIC' if main_complete else 'EXPLORATORY_PARTIAL_SNAPSHOT',
        'scope':'Canonical valid PASS and controller FAIL route attempts; archived infrastructure attempts are excluded by the one-level source glob. No new flights or policy selection.',
        'geometry':{'points_enu_m':POINTS,'interior_corners':[1,2,3],'fixed_pre_window_s':.5,'fixed_post_window_s':1.,'closest_corner_window_s':[-.5,.5]},
        'metric_definitions':{'corner_min_approach_m':'Minimum 3D distance to the interior waypoint in[-0.5,+0.5]s about its confirmed arrival event.',
            'post_outgoing_max_deviation_m':'Peak 3D distance to the outgoing finite geometric segment in[0,+1]s after the confirmed goal transition.',
            'post_outgoing_mean_deviation_m':'Time-weighted distance to the outgoing finite geometric segment in[0,+1]s.',
            'post_incoming_overshoot_m':'Positive excess beyond the waypoint projected onto the incoming segment direction in[0,+1]s.',
            'post_new_target_rmse_m':'Time-weighted distance to the NEW waypoint in[0,+1]s; large values are expected after a2m target step. This is not path deviation or whole-route RMSE.',
            'pre_corner_rmse_m':'Time-weighted distance to the corner waypoint in[-0.5,0]s.'},
        'aggregation':'Average observed complete-window interior-corner statistics within each flight; average trained models within each environment condition; equal-weight environment conditions within a scenario. Censored or unreached windows are retained in denominators and have null complete-window metrics. PID is used once per environment, not tripled as independent data.',
        'all_available_diagnostics':aggregates(flights),'matched_condition_diagnostics':aggregates(matched,True),
        'matched_cells':matched_cells,'full_three_model_cohort_cells':full_cohort,
        'counts':{'expected_full_main_route_conditions':126,'expected_full_main_corner_opportunities':378,
            'canonical_route_summaries':canonical_route_summaries,'valid_route_traces':len(flights),
            'physical_route_traces':len(flights),'pass_route_traces':sum(r['flight_status']=='PASS' for r in flights),
            'controller_fail_route_traces':sum(r['flight_status']=='FAIL' for r in flights),
            'route_completed_traces':sum(r['route_completed'] is True for r in flights),
            'interior_corner_observations':len(corners),'full_post_corner_windows':sum(r['full_post_window'] for r in corners),
            'post_censored_corner_events':sum(r['post_window_censored'] for r in corners),
            'unreached_corner_events':sum(r['unreached_corner_events'] for r in flights),
            'infrastructure_or_metadata_exclusions':len(excluded),
            'matched_condition_cells':len(matched_cells),'full_three_model_cohort_cells':len(full_cohort)},
        'main_state':state,'main_campaign_counts':{k:campaign.get(k) for k in ['status','planned','completed','total_attempts']},
        'limitations':[('Full main campaign and its coordinator audits were complete before this snapshot.' if main_complete else 'Direct campaign is still running; trained seed coverage may differ by condition and results are partial.'),
            'These missions use waypoint arrival dwell0.5s, not prescribed continuous-speed turns. Geometric cornering is not a yaw-command experiment.',
            'Window means are conditional on reaching and observing that corner window; PASS/FAIL counts, unreached corners and censored windows are separate and must remain visible.',
            'Disturbances use fixed absolute simulation time; different route speeds can change which force phase intersects an event-relative window.',
            'No confidence intervals, practical-improvement threshold, or claims about explicitly commanded yaw turns are made.',
            'Recorded states are ArduPilot AHRS10 MAVLink; independent physical identity is checked by the main study audits.'],
        'excluded_route_summaries':excluded,'sources':provenance,'frozen_contract_sha256':{str(p.relative_to(ROOT)):sha(p) for p in freezes},'script_sha256':sha(Path(__file__))}
    out.mkdir(parents=True,exist_ok=False)
    (out/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'generator_source.py').write_bytes(Path(__file__).read_bytes())
    table={'status':report['status'],'counts':report['counts'],'rows':report['matched_condition_diagnostics'],
        'interpretation':'Measured corner diagnostics only. These data do not establish NN superiority at corners; see unreached/censored denominators and phase/speed tradeoffs.',
        'metric':'Mean per-corner peak 3D deviation from outgoing segment in first1s after confirmed transition; new-target RMSE is a separate2m-step response metric.'}
    (out/'table.json').write_text(json.dumps(table,indent=2),encoding='utf-8')
    csv_write(out/'corners.csv',corners);csv_write(out/'flights.csv',flights);csv_write(out/'matched_summary.csv',report['matched_condition_diagnostics'])
    print(json.dumps({'output':str(out),'counts':report['counts'],'matched':report['matched_condition_diagnostics']},indent=2))
if __name__=='__main__':main()
