"""Prospective joint geometry judge + actual historical/analytic verification.

No simulation/controller imports or new flights. The five original metrics remain
in sim.metrics; this adds geometry without relabeling goal-distance as cross-track.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadrl.protocol import ROUTE_POINTS
from audit.turns_corners import segment_distance as old_segment_distance
from audit.turns_corners import analyse as old_corner_analysis
from sim.metrics import from_csv_rows

GAP_LIMIT=.060001
EPS=1e-8


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distances_to_leg(position,a,b):
    p=np.asarray(position,dtype=float);a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    leg=b-a;length2=float(leg@leg)
    if length2<=1e-15:
        distance=np.linalg.norm(p-a,axis=-1)
        return distance,distance
    fraction=np.sum((p-a)*leg,axis=-1)/length2
    line=np.linalg.norm(p-(a+fraction[...,None]*leg),axis=-1)
    segment=np.linalg.norm(p-(a+np.clip(fraction,0,1)[...,None]*leg),axis=-1)
    return segment,line


def weights(t,start,end):
    return np.maximum(0.,np.minimum(t,end)-np.maximum(np.r_[0.,t[:-1]],start))


def weighted_rmse(values,w):
    return float(np.sqrt(np.dot(w,np.asarray(values)**2)/w.sum())) if w.sum()>EPS else None


def gap_complete(t,start,end):
    if end<=start or t[0]>start+GAP_LIMIT or t[-1]<end-EPS:return False
    relevant=np.r_[start,t[(t>start+EPS)&(t<end-EPS)],end]
    return bool(np.max(np.diff(relevant),initial=0.)<=GAP_LIMIT)


def geometry_metrics(t,position,target,events,completion_time_s=None):
    """Generic Fast/Gazebo judge; ignores incompatible post-update goal_index.

    Current target belongs to the interval BEFORE a same-row confirmed event.
    Thus events strictly earlier than t determine the active finite segment.
    """
    t=np.asarray(t,dtype=float);p=np.asarray(position,dtype=float);goal=np.asarray(target,dtype=float)
    if len(t)<2 or p.shape!=(len(t),3) or goal.shape!=p.shape:
        raise ValueError("Nonempty aligned time/position/target arrays required")
    if not np.isfinite(t).all() or not np.isfinite(p).all() or not np.isfinite(goal).all() or t[0]<0 or np.any(np.diff(t)<=0):
        raise ValueError("Finite, strictly increasing measured samples required")
    events=[event for event in events if event.get("type")=="goal_reached"]
    indices=[int(e["goal_index"]) for e in events]
    if indices!=list(range(len(events))) or len(events)>5:
        raise ValueError("Ordered unique goal0..goal4 confirmations required")
    et=np.asarray([float(e["time_s"]) for e in events])
    if len(et) and (not np.isfinite(et).all() or np.any(np.diff(et)<=0)):
        raise ValueError("Strictly increasing mission-event times required")
    for e in events:
        if not np.allclose(e["target"],ROUTE_POINTS[int(e["goal_index"])],rtol=0,atol=1e-9):
            raise ValueError("Event geometry differs from fixed route")
    active=np.minimum(np.searchsorted(et,t-EPS,side="left"),4)
    if not np.allclose(goal,ROUTE_POINTS[active],rtol=0,atol=1e-7):
        raise ValueError("Recorded current target does not match pre-transition mission sequence")
    segment=np.zeros(len(t));cross=np.zeros(len(t));segment_xy=np.zeros(len(t));cross_xy=np.zeros(len(t))
    for index in range(5):
        mask=active==index
        if mask.any():
            segment[mask],cross[mask]=distances_to_leg(p[mask],ROUTE_POINTS[max(0,index-1)],ROUTE_POINTS[index])
            segment_xy[mask],cross_xy[mask]=distances_to_leg(p[mask,:2],ROUTE_POINTS[max(0,index-1),:2],ROUTE_POINTS[index,:2])
    completed=completion_time_s is not None and len(events)==5
    if completion_time_s is not None and (len(events)!=5 or abs(float(completion_time_s)-et[-1])>.040001):
        raise ValueError("Completion time must match the fifth confirmed arrival")
    start=float(et[0]) if len(et) else None
    end=float(completion_time_s) if completed else float(t[-1])
    w=weights(t,start,end) if start is not None else np.zeros(len(t))
    transit_complete=bool(completed and start is not None and gap_complete(t,start,end) and abs(w.sum()-(end-start))<=1e-7)
    corners=[]
    for event in events:
        index=int(event["goal_index"])
        if index not in (1,2,3):continue
        stamp=float(event["time_s"]);a=ROUTE_POINTS[index];b=ROUTE_POINTS[index+1]
        closed=(t>=stamp-EPS)&(t<=stamp+1+EPS)
        cw=weights(t,stamp,stamp+1)
        outgoing,_=distances_to_leg(p,a,b)
        incoming=a-ROUTE_POINTS[index-1];unit=incoming/np.linalg.norm(incoming)
        full=gap_complete(t,stamp,stamp+1) and bool(closed.any())
        corners.append(dict(corner_index=index,transition_s=stamp,window_end_s=stamp+1,
            full_post_window=bool(full),post_outgoing_max_deviation_m=float(np.max(outgoing[closed])) if full else None,
            post_outgoing_mean_deviation_m=float(cw@outgoing/cw.sum()) if full else None,
            post_incoming_overshoot_m=float(max(0.,np.max((p[closed]-a)@unit))) if full else None))
    full_corners=len(corners)==3 and all(c["full_post_window"] for c in corners)
    eligible=transit_complete and full_corners
    result=dict(route_completed=bool(completed),geometry_eligible=bool(eligible),
        transit_start_s=start,transit_end_s=end,transit_duration_s=end-start if start is not None else None,
        observed_transit_coverage_s=float(w.sum()),max_sample_gap_s=float(np.max(np.diff(t))),
        segment_rmse_m=weighted_rmse(segment,w) if eligible else None,
        cross_track_rmse_m=weighted_rmse(cross,w) if eligible else None,
        cross_track_3d_rmse_m=weighted_rmse(cross,w) if eligible else None,
        cross_track_xy_rmse_m=weighted_rmse(cross_xy,w) if eligible else None,
        finite_segment_xy_rmse_m=weighted_rmse(segment_xy,w) if eligible else None,
        active_segment_peak_m=float(np.max(segment[w>0])) if eligible else None,
        observed_transit_segment_rmse_m=weighted_rmse(segment,w),
        expected_corners=3,observed_corners=len(corners),unreached_corners=3-len(corners),
        complete_corner_windows=sum(c["full_post_window"] for c in corners),
        mean_corner_post_peak_m=float(np.mean([c["post_outgoing_max_deviation_m"] for c in corners])) if eligible else None,
        max_corner_post_peak_m=float(np.max([c["post_outgoing_max_deviation_m"] for c in corners])) if eligible else None,
        corners=corners,
        interpretation="segment_rmse_m and peaks use 3D finite segments; cross_track_rmse_m is the preserved 3D infinite-line alias; explicit XY fields exclude commanded altitude components. Target error and physical altitude drop remain separate original metrics")
    return result


def from_gazebo_run(path):
    path=Path(path);summary=json.loads((path/"summary.json").read_text())
    if summary["mission"]!="route":raise ValueError("Route source required")
    with (path/"telemetry.csv").open(newline="",encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
    events=summary.get("mission_events",[])
    if not events:
        for line in (path/"events.jsonl").read_text().splitlines():
            record=json.loads(line)
            if record.get("event")=="goal_reached":events.append(record["mission_event"])
            elif record.get("event")=="mission_transition" and record.get("type")=="goal_reached":events.append(record)
    geometry=geometry_metrics([float(r["t"]) for r in rows],[[float(r[k]) for k in ("x_e","y_n","z_up")] for r in rows],
        [[float(r[k]) for k in ("target_e","target_n","target_up")] for r in rows],events,summary.get("completion_time_s"))
    original=from_csv_rows(rows,summary)
    return dict(run=str(path.relative_to(ROOT)),controller=summary["controller"],scenario=summary["scenario"],
        environment_seed=summary["seed"],training_seed=summary.get("train_seed"),flight_status=summary["status"],
        route_restricted_time_s=float(summary["completion_time_s"]) if original["route_completed"] else 40.,
        original_metrics=original,geometry=geometry,
        source_sha256={name:sha(path/name) for name in ("summary.json","telemetry.csv","events.jsonl")})


def fixture():
    t=np.arange(.02,40.0001,.02)
    event_times=np.array([.5,2.5,4.5,6.5,8.5])
    events=[dict(type="goal_reached",goal_index=i,time_s=float(stamp),target=ROUTE_POINTS[i].tolist()) for i,stamp in enumerate(event_times)]
    active=np.minimum(np.searchsorted(event_times,t-EPS,side="left"),4)
    target=ROUTE_POINTS[active].copy();position=[]
    for stamp,index in zip(t,active):
        if index==0:position.append(ROUTE_POINTS[0].copy());continue
        left=event_times[index-1];finish=event_times[index]-.5
        fraction=np.clip((stamp-left)/(finish-left),0,1)
        position.append(ROUTE_POINTS[index-1]+fraction*(ROUTE_POINTS[index]-ROUTE_POINTS[index-1]))
    return t,np.asarray(position),target,events


def self_test(output=None):
    checks=[]
    finite,line=distances_to_leg(np.array([[1,.3,.4],[3,0,0]]),np.zeros(3),np.array([2,0,0]))
    assert np.allclose(finite,[.5,1.]) and np.allclose(line,[.5,0.])
    checks.append("Finite-segment overshoot is not silently called perpendicular cross-track")
    finite_xy,line_xy=distances_to_leg(np.array([[1,.3,.4],[3,0,0]])[:,:2],np.zeros(2),np.array([2,0]))
    assert np.allclose(finite_xy,[.3,1.]) and np.allclose(line_xy,[.3,0.])
    checks.append("XY cross-track excludes vertical error and retains finite/infinite distinctions")
    t,p,target,events=fixture();perfect=geometry_metrics(t,p,target,events,8.5)
    assert perfect["geometry_eligible"] and perfect["segment_rmse_m"]<1e-14 and perfect["complete_corner_windows"]==3
    assert np.sqrt(np.mean(np.sum((p-target)**2,axis=1)))>.1
    checks.append("Perfect polyline transit has zero geometry error despite nonzero goal-distance error")
    altered=p.copy();leg1=(t>.5+EPS)&(t<2.5-EPS);altered[leg1,1]+=.2
    full=geometry_metrics(t,altered,target,events,8.5);short=t<=10
    cropped=geometry_metrics(t[short],altered[short],target[short],events,8.5)
    assert full["segment_rmse_m"]>0 and full["segment_rmse_m"]==cropped["segment_rmse_m"]
    checks.append("Appending thirty seconds of perfect final hover cannot dilute transit geometry")
    partial=t<=3;incomplete=geometry_metrics(t[partial],p[partial],target[partial],events[:2],None)
    assert not incomplete["geometry_eligible"] and incomplete["segment_rmse_m"] is None and incomplete["unreached_corners"]==2
    checks.append("Unfinished routes retain missing corner denominators and null primary geometry")
    bad=target.copy();bad[10]=ROUTE_POINTS[2]
    try:geometry_metrics(t,p,bad,events,8.5)
    except ValueError:pass
    else:raise AssertionError("Mismatched target sequence accepted")
    checks.append("Future-goal or post-update-index mismatch is rejected")
    keep=(t<2.8)|(t>3.1);gapped=geometry_metrics(t[keep],p[keep],target[keep],events,8.5)
    assert not gapped["geometry_eligible"]
    checks.append("Missing measured geometry window cannot earn a quality pass")
    # Reuse actual historical controls only. No new/closed joint-test seed opened.
    records=[];legacy_errors=[]
    profile=ROOT/"artifacts/gazebo/native_pid_tuning_v1/frozen_native_pid.json"
    pid_hash=sha(profile)
    for scenario in ("nominal","force_strong","gust"):
        for seed in (202601,202602,202603):
            path=ROOT/f"artifacts/gazebo/main_hybrid_test_v1/pid_trainNone_route_{scenario}_env{seed}"
            record=from_gazebo_run(path);assert record["geometry"]["geometry_eligible"]
            _,old,_=old_corner_analysis(path/"summary.json",{},pid_hash)
            delta=abs(record["geometry"]["mean_corner_post_peak_m"]-old["post_outgoing_max_deviation_m"])
            assert delta<1e-10
            legacy_errors.append(delta);records.append(record)
    historical=[]
    for scenario in ("nominal","force_strong","gust"):
        group=[r for r in records if r["scenario"]==scenario]
        route=float(np.mean([r["route_restricted_time_s"] for r in group]))
        segment=float(np.mean([r["geometry"]["segment_rmse_m"] for r in group]))
        corner=float(np.mean([r["geometry"]["mean_corner_post_peak_m"] for r in group]))
        historical.append(dict(scenario=scenario,physical_existing_PID_routes=3,route_mean_s=route,
            illustrative_speed_limit_s=1.05*route,transit_segment_RMSE_m=segment,
            illustrative_geometry_limit_m=max(1.10*segment,segment+.01),corner_mean_peak_m=corner,
            illustrative_corner_limit_m=max(1.10*corner,corner+.02),
            interpretation="Previously viewed historical reference only; new joint gates use fresh paired PID, not these constants"))
    pilot=[]
    folder=ROOT/"artifacts/refinement/gazebo_pilot_v1/runs"
    for name in ("pid_trainNone_route_gust_env83202","bc_train111_route_gust_env83202","candidate_train111_route_gust_env83202"):
        record=from_gazebo_run(folder/name);assert record["geometry"]["geometry_eligible"]
        pilot.append(record)
    contract=ROOT/"docs/joint_applicability_contract_v2.json"
    result=dict(status="PASS",scope="Analytic fixtures and12 previously recorded route traces; no new flights/training; reserved930xx not opened",
        analytic_checks=checks,historical_PID=historical,old_corner_definition_max_delta_m=max(legacy_errors),
        existing_recorded_routes_checked=len(records)+len(pilot),previously_viewed_pilot=pilot,
        historical_source_records=records,source_sha256=sha(Path(__file__)),contract_sha256=sha(contract),
        frozen_pid_profile_sha256=pid_hash,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    out=Path(output) if output is not None else ROOT/"audit/verify_joint_metrics_v2_1.json"
    if out.exists():raise FileExistsError("Preserve prior joint metric receipt: "+str(out))
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status="PASS",receipt=str(out),analytic_checks=len(checks),historical_PID=historical,
                         pilot=[dict(controller=r["controller"],time_s=r["route_restricted_time_s"],segment_rmse_m=r["geometry"]["segment_rmse_m"],cross_track_xy_rmse_m=r["geometry"]["cross_track_xy_rmse_m"],corner_peak_m=r["geometry"]["mean_corner_post_peak_m"]) for r in pilot]),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--output",help="New immutable receipt path; fails if it already exists")
    parser.add_argument("--gazebo-runs",nargs="+",help="Existing completed raw route directories; no simulation is launched")
    args=parser.parse_args()
    if args.self_test and args.gazebo_runs:parser.error("Choose self-test or recorded runs")
    if args.self_test:self_test(args.output)
    elif args.gazebo_runs:
        records=[from_gazebo_run(Path(p) if Path(p).is_absolute() else ROOT/p) for p in args.gazebo_runs]
        result=dict(status="MEASURED_GEOMETRY_ONLY",scope="Recorded development traces; physical/world identity must be independently audited before applicability claims",
                    records=records,source_sha256=sha(Path(__file__)),contract_sha256=sha(ROOT/"docs/joint_applicability_contract_v2.json"),
                    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if args.output:
            out=Path(args.output)
            if not out.is_absolute():out=ROOT/out
            if out.exists():raise FileExistsError("Preserve prior geometry receipt: "+str(out))
            out.write_text(json.dumps(result,indent=2))
        print(json.dumps([dict(run=r["run"],time_s=r["route_restricted_time_s"],geometry_eligible=r["geometry"]["geometry_eligible"],
                               segment_rmse_m=r["geometry"]["segment_rmse_m"],cross_track_xy_rmse_m=r["geometry"]["cross_track_xy_rmse_m"],
                               corner_peak_m=r["geometry"]["mean_corner_post_peak_m"],active_segment_peak_m=r["geometry"]["active_segment_peak_m"])
                          for r in records],indent=2))
    else:parser.error("Choose --self-test or --gazebo-runs")
