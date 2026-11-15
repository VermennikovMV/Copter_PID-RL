"""Shared V2 controller metrics. NumPy only; no simulation or controller imports."""
import numpy as np

def compute_metrics(t,position,target,yaw_error,horizon_s,*,failure=None,mission='hover',completion_time=None,force_events=(),force_off_time=None,force_time_source='unverified'):
    t=np.asarray(t,dtype=float);pos=np.asarray(position,dtype=float);goal=np.asarray(target,dtype=float);yaw=np.asarray(yaw_error,dtype=float)
    if t.ndim!=1 or len(t)==0 or pos.shape!=(len(t),3) or goal.shape!=pos.shape or yaw.shape!=t.shape:raise ValueError('Invalid metric trace shape')
    if not all(np.all(np.isfinite(v)) for v in [t,pos,goal,yaw]) or np.any(np.diff(t)<=0) or t[0]<0:raise ValueError('Invalid metric samples/timestamps')
    horizon=float(horizon_s);duration=min(float(t[-1]),horizon)
    error=np.linalg.norm(pos-goal,axis=1)
    # Each sampled state represents its preceding control interval, including the
    # initial0..first-sample interval. Failure remainder receives fixed4m error.
    weights=np.diff(np.r_[0.,np.minimum(t,horizon)])
    covered=float(weights.sum());survived=failure is None and duration>=horizon-.04-1e-9
    missing=max(0.,horizon-covered)
    integral=float(np.dot(weights,error**2))
    filled=integral+missing*(float(error[-1])**2 if survived else 16.)
    heading=np.arctan2(np.sin(yaw),np.cos(yaw))*180/np.pi
    heading_rmse=float(np.sqrt(np.dot(weights,heading**2)/covered)) if covered>1e-12 else None
    force_present=bool(force_events);applicable=mission=='hover' and force_present
    recovery=None;confirmed=None;inside_since=None
    if applicable and force_off_time is not None:
        previous=None
        for stamp,err in zip(t,error):
            if stamp<force_off_time-1e-9:continue
            if previous is not None and stamp-previous>.030001:inside_since=None
            previous=stamp
            if err>.25:inside_since=None;continue
            if inside_since is None:inside_since=float(stamp)
            if stamp-inside_since>=1.-1e-9:
                recovery=max(0.,inside_since-float(force_off_time));confirmed=float(stamp);break
    route_complete=mission=='route' and completion_time is not None
    pre_height=None;pre_drop=None
    phases={}
    if force_events:
        first_on=min(float(event['on_s']) for event in force_events)
        last_off=max(float(event['off_s']) for event in force_events)
        before=(t<first_on)&(t>=first_on-.5);after=t>=first_on
        if np.any(before) and np.any(after):
            pre_height=float(np.median(pos[before,2]));pre_drop=float(max(0.,pre_height-np.min(pos[after,2])))
        starts=np.r_[0.,t[:-1]];ends=np.minimum(t,horizon)
        def phase(intervals):
            w=np.zeros(len(t))
            for left,right in intervals:w+=np.maximum(0.,np.minimum(ends,right)-np.maximum(starts,left))
            return float(np.sqrt(np.dot(w,error**2)/w.sum())) if w.sum()>0 else None
        phases={'rmse_before_force_m':phase([(0,first_on)]),'rmse_during_force_m':phase([(float(e['on_s']),float(e['off_s'])) for e in force_events]),'rmse_after_force_m':phase([(last_off,horizon)]),'phase_force_time_source':force_time_source}
    return {'survived':bool(survived),'rmse_position_m':float(np.sqrt(integral/covered)) if covered>1e-12 else None,
        'penalized_rmse_position_m':float(np.sqrt(filled/horizon)),
        'altitude_drop_m':float(max(0.,np.max(goal[:,2]-pos[:,2]))),
        'pre_force_altitude_m':pre_height,'altitude_drop_from_preforce_m':pre_drop,'preforce_height_window_s':.5,
        'recovery_time_s':recovery,'recovery_dwell_confirmed_at_s':confirmed,
        'recovery_applicable':applicable,'recovery_censored':bool(recovery is None) if applicable and force_off_time is not None else None,
        'recovery_time_verified':bool(force_off_time is not None),'force_off_time_s':force_off_time,'force_time_source':force_time_source,
        'route_completed':bool(route_complete) if mission=='route' else None,
        'route_completion_time_s':float(completion_time) if route_complete else None,
        'route_completion_censored':bool(not route_complete) if mission=='route' else None,
        'heading_rmse_deg':heading_rmse,
        'heading_peak_deg':float(np.max(np.abs(heading))),'heading_guard_pass':bool(heading_rmse is not None and heading_rmse<=5 and np.max(np.abs(heading))<=15),
        'heading_full_flight_eligible':bool(survived),
        'duration_s':duration,'horizon_s':horizon,'failure':failure,
        'metric_time_weighting':'preceding control interval; failed/incomplete remainder4m; surviving final<=40ms filled last error',**phases}

def from_csv_rows(rows,summary,*,force_off_time=None,force_time_source='unverified',force_events_override=None):
    keys=['x_e','y_n','z_up'];goals=['target_e','target_n','target_up']
    events=summary.get('scenario_spec',{}).get('force_events',()) if force_events_override is None else force_events_override
    return compute_metrics([float(r['t']) for r in rows],[[float(r[k]) for k in keys] for r in rows],[[float(r[k]) for k in goals] for r in rows],[float(r['yaw_error_rad']) for r in rows],summary['requested_duration_s'],failure=summary.get('failure') or (summary.get('error') if summary['status'] not in ('PASS',) else None),mission=summary['mission'],completion_time=summary.get('completion_time_s'),force_events=events,force_off_time=force_off_time,force_time_source=force_time_source)
