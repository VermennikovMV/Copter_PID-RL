"""DIAGNOSTIC: recorded native rate shaping; never starts flight or training.

Only writes the adjacent new JSON receipt. All inputs are completed archived
development runs. No frozen configuration or controller is changed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_suffix('.json')
COMPLETE = ROOT/'artifacts/refinement/joint_v2/gazebo_thrust_ablation_recovery/complete.json'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):
    try: return str(Path(p).relative_to(ROOT))
    except ValueError: return str(p)
def completed():
    result = read(COMPLETE)
    assert result['status']=='COMPLETE' and result['valid_attempts']==12
    return [(x, Path(x['run'])) for x in result['results'] if x['status']=='PASS']

def quantile(x):
    x=np.asarray(x)
    return dict(zip(('median','p90','p95','p99','max'),map(float,np.quantile(x,[.5,.9,.95,.99,1])))) if x.size else None

def zoh(t,x,out):
    return x[np.clip(np.searchsorted(t,out,side='right')-1,0,len(t)-1)]

def native_shape(command, dt, accel_max, tc, initial=None):
    """Exact rate-only reduction of AP4.7 attitude_command_model.

    The zero position error gives zero correction rate/acceleration. The
    remaining demand is clip((desired-rate)/tc,+/-accel_max), followed by
    shape_accel jerk clamp and Euler target velocity update. No fitted gain.
    """
    v=command[0].copy() if initial is None else np.asarray(initial).copy()
    accel=np.zeros(3); result=np.empty_like(command)
    for i,u in enumerate(command):
        target=np.minimum(np.maximum((u-v)/tc,-accel_max),accel_max)
        jerk_step=accel_max/tc*dt
        accel+=np.minimum(np.maximum(target-accel,-jerk_step),jerk_step)
        v+=accel*dt; result[i]=v
    return result

def load_run(folder, record=None):
    from pymavlink import mavutil
    summary=read(folder/'summary.json'); manifest=read(folder/'manifest.json'); commands=list(lines(folder/'commands.jsonl'))
    cb=np.array([x['boot_s'] for x in commands]); ct=np.array([x['t'] for x in commands])
    cmd=np.array([x['action'][1:4] for x in commands])*np.array([2.,-2.,-1.])
    assert np.all(np.diff(cb)>0) and np.all(np.isfinite(cmd))
    offset=float(np.median(cb-ct)); conn=mavutil.mavlink_connection(str(folder/'sitl/logs/00000001.BIN'))
    params={}; rate=[]; guia=[]
    while True:
        m=conn.recv_match(type=['PARM','RATE','GUIA'])
        if m is None: break
        k=m.get_type()
        if k=='PARM':params[m.Name]=float(m.Value)
        elif k=='RATE':rate.append([m.TimeUS*1e-6,*np.deg2rad([m.RDes,m.PDes,m.YDes,m.R,m.P,m.Y])])
        elif k=='GUIA':guia.append([m.TimeUS*1e-6,*np.deg2rad([m.RollRt,m.PitchRt,m.YawRt])])
    rate=np.asarray(rate); guia=np.asarray(guia)
    assert len(guia)==len(cmd), (folder,len(guia),len(cmd))
    assert np.max(abs(guia[:,1:]-cmd))<1e-5
    assert np.max(abs(cb-ct-offset))<1e-6
    feedback=[]; att=[]
    for row in lines(folder/'mavlink.jsonl'):
        m=row['message']; k=m['mavpackettype']
        if k=='ATTITUDE_TARGET': feedback.append([m['time_boot_ms']*.001,m['body_roll_rate'],m['body_pitch_rate'],m['body_yaw_rate']])
        elif k=='ATTITUDE':att.append([m['time_boot_ms']*.001,m['rollspeed'],m['pitchspeed'],m['yawspeed']])
    fb=np.asarray(feedback); att=np.asarray(att)
    with (folder/'telemetry.csv').open() as f: telemetry=list(csv.DictReader(f))
    end=summary.get('completion_time_s') or summary['duration_s']
    if record and record['job']['mission']=='route': end=record['metrics']['route_completion_time_s']
    active=(rate[:,0]>=offset+.5)&(rate[:,0]<=offset+end)
    dt=1./params['SCHED_LOOP_RATE']
    accel=np.deg2rad([params['ATC_ACC_R_MAX'],params['ATC_ACC_P_MAX'],params['ATC_ACC_Y_MAX']])
    tc=np.array([params['ACRO_RP_RATE_TC']]*2+[params['ACRO_Y_RATE_TC']]); tc=np.where(tc>0,tc,dt*10.)
    grid=np.arange(cb[0]-.1,cb[-1]+.1,dt)
    raw_grid=zoh(cb,cmd,grid); recv_grid=zoh(guia[:,0],guia[:,1:],grid)
    shaped=native_shape(raw_grid,dt,accel,tc)
    received_shaped=native_shape(recv_grid,dt,accel,tc)
    fm=(fb[:,0]>=cb[0]+.3)&(fb[:,0]<=offset+end)
    predicted=np.column_stack([np.interp(fb[fm,0],grid,received_shaped[:,a]) for a in range(3)])
    fmerr=np.sqrt(np.mean((predicted-fb[fm,1:])**2,axis=0))
    assert np.max(fmerr[:2])<.015, ('Native shaper does not reproduce archived target',folder,fmerr)
    # Raw msec timestamp is quantized; tolerate one scheduler sample but no fitted shift.
    am=(att[:,0]>=cb[0]+.3)&(att[:,0]<=offset+end)
    bin_att=np.column_stack([np.interp(att[am,0],rate[:,0],rate[:,4+a]) for a in range(3)])
    activecmd=(ct>=.5)&(ct<=end)
    jumps=np.diff(cmd,axis=0); cdt=np.diff(cb)
    axes=[]
    for a in range(3):
        j=abs(jumps[activecmd[1:],a]); derivative=abs(np.diff(fb[:,1+a])/np.diff(fb[:,0]))
        fda=(fb[1:,0]>=offset+.5)&(fb[1:,0]<=offset+end)
        axes.append({'axis':('roll','pitch','yaw')[a],
            'command_abs_rad_s':quantile(abs(cmd[activecmd,a])),
            'command_jump_abs_rad_s':quantile(j),
            'command_slope_abs_rad_s2':quantile(j/cdt[activecmd[1:]]),
            'command_fraction_steps_over_native_accel':float(np.mean(j/cdt[activecmd[1:]]>accel[a])),
            'feedback_slope_abs_rad_s2':quantile(derivative[fda]),
            'feedback_fraction_steps_at_accel_limit_5percent':float(np.mean(derivative[fda]>.95*accel[a])),
            'native_accel_limit_rad_s2':float(accel[a]),
            'native_input_tc_s':float(tc[a]),
            'native_jerk_limit_rad_s3':float(accel[a]/tc[a]),
            'source_native_shaper_unfitted_feedback_rmse_rad_s':float(fmerr[a]),
            'mav_gyro_vs_BIN_gyro_boot_interpolation_rmse_rad_s':float(np.sqrt(np.mean((bin_att[:,a]-att[am,1+a])**2))),
        })
    report={'run':rel(folder),'kind':record['job'] if record else 'recorded_system_identification',
        'active_end_s':end,'status':summary['status'],
        'archived_source_identity':{k:manifest.get(k) for k in ('source_commit','source_status','binary_sha256','script_sha256','policy_sha256')},
        'boot_matching':{'measurement_boot_offset_s':offset,'command_count':len(cmd),'BIN_GUIA_count':len(guia),
            'ordered_command_vs_BIN_GUIA_peak_error_rad_s':float(np.max(abs(guia[:,1:]-cmd))),
            'GUIA_receipt_minus_command_state_boot_s':quantile(guia[:,0]-cb),
            'BIN_RATE_sample_dt_s':quantile(np.diff(rate[:,0])),
            'native_scheduler_dt_s':dt,'clocks':'ArduPilot TimeUS/1e6 and MAVLink time_boot_ms/1e3 only; no wall-time fit'},
        'axes':axes,'geometry':record.get('geometry') if record else None,
        'hashes':{rel(folder/name):sha(folder/name) for name in ('commands.jsonl','mavlink.jsonl','telemetry.csv','summary.json','sitl/logs/00000001.BIN','manifest.json')},
    }
    snap=folder/'source_snapshot/native_angle_boost'
    if snap.exists():
        report['hashes'].update({rel(p):sha(p) for p in snap.glob('*.cpp')})
        report['native_attitude_cpp_snapshot_matches_current_source']=sha(snap/'AC_AttitudeControl.cpp')==sha(Path('[external-home]/vkr_sim/ardupilot/libraries/AC_AttitudeControl/AC_AttitudeControl.cpp'))
    return dict(folder=folder,record=record,report=report,cb=cb,ct=ct,cmd=cmd,rate=rate,guia=guia,fb=fb,att=att,
        offset=offset,end=end,grid=grid,raw=raw_grid,shaped=shaped,active=active,params=params,dt=dt,telemetry=telemetry)

def lag_metrics(t,x,y,maxlag=.3):
    """Descriptive lag minimizing raw RMSE; no gain fit, not causality proof."""
    # Logged scheduler samples alternate 2/3ms. Median can jump between them
    # merely because of endpoint parity, so use complete elapsed sample time.
    dt=float((t[-1]-t[0])/(len(t)-1)); fits=[]
    for steps in range(int(maxlag/dt)+1):
        xx=x[:-steps or None]; yy=y[steps:]
        fits.append((float(np.sqrt(np.mean((xx-yy)**2))),steps*dt))
    best=min(fits)
    return {'best_lag_s':best[1],'best_lag_rmse_rad_s':best[0],'zero_lag_rmse_rad_s':fits[0][0]}

def describe(data):
    r=data['rate']; m=data['active']; t=r[m,0]
    raw=zoh(data['cb'],data['cmd'],t)
    feedback=np.column_stack([np.interp(t,data['fb'][:,0],data['fb'][:,1+a]) for a in range(3)])
    for a,out in enumerate(data['report']['axes']):
        out['command_to_shaped_feedback']=lag_metrics(t,raw[:,a],feedback[:,a])
        out['shaped_feedback_to_rate_desired']=lag_metrics(t,feedback[:,a],r[m,1+a])
        out['rate_desired_to_gyro']=lag_metrics(t,r[m,1+a],r[m,4+a])
        out['command_to_gyro']=lag_metrics(t,raw[:,a],r[m,4+a])
        out['attitude_correction_rate_desired_minus_shaped_rmse_rad_s']=float(np.sqrt(np.mean((r[m,1+a]-feedback[:,a])**2)))
    # Restrict reversal association to substantial command sign changes. Report
    # observed crossing times; no counterfactual position attribution is made.
    events=[]
    for a in (0,1):
        u=data['cmd'][:,a]; cb=data['cb']
        cross=np.where((u[1:]*u[:-1]<0)&(data['ct'][1:]>.5)&(data['ct'][1:]<data['end']))[0]+1
        last=-1.
        for j in cross:
            sign=np.sign(u[j]); now=cb[j]; limit=min(now+.5,data['offset']+data['end'])
            before=u[(cb>=now-.25)&(cb<now)]
            after=u[(cb>now)&(cb<=now+.25)]
            if now-last<.2 or not len(before) or not len(after) or np.max(-sign*before)<.1 or np.max(sign*after)<.1:continue
            last=now
            near=(r[:,0]>=now)&(r[:,0]<=limit)
            times=r[near,0]; ff=np.interp(times,data['fb'][:,0],data['fb'][:,1+a]); desire=r[near,1+a]; gyro=r[near,4+a]
            def crossing(v):
                indices=np.where(v*sign>0)[0]
                return float(times[indices[0]]-now) if len(indices) else None
            events.append({'axis':('roll','pitch')[a],'t_s':float(now-data['offset']),
                'old_command_rad_s':float(u[j-1]),'new_command_rad_s':float(u[j]),
                'prior_250ms_peak_old_sign_rad_s':float(np.max(-sign*before)),
                'next_250ms_peak_new_sign_rad_s':float(np.max(sign*after)),
                'shaped_sign_cross_delay_s':crossing(ff),'rate_desired_sign_cross_delay_s':crossing(desire),'gyro_sign_cross_delay_s':crossing(gyro)})
    data['report']['substantial_reversals']=events
    corners=[]
    if data['record'] and data['record']['job']['mission']=='route':
        for corner in data['record']['geometry']['corners']:
            tc=corner['transition_s']; mask=(data['fb'][:-1,0]-data['offset']>=tc-.5)&(data['fb'][:-1,0]-data['offset']<=tc+.5)
            slopes=np.diff(data['fb'][:,1:],axis=0)/np.diff(data['fb'][:,0])[:,None]
            cap=np.deg2rad([data['params']['ATC_ACC_R_MAX'],data['params']['ATC_ACC_P_MAX']])
            fractions=np.mean(abs(slopes[mask,:2])>.95*cap,axis=0)
            corners.append({**corner,'native_RP_accel_cap_fraction_in_pm500ms':fractions.tolist(),
                'native_RP_shaped_slope_peak_in_pm500ms_rad_s2':np.max(abs(slopes[mask,:2]),axis=0).tolist(),
                'nearest_recorded_rate_reversal':min(events,key=lambda x:abs(x['t_s']-tc)) if events else None})
    data['report']['corner_rate_relation']=corners
    return data['report']

def first_order(data, axis, parameters, shaped=False):
    from scipy.signal import lfilter
    gain,tau,delay=parameters; dt=data['dt']; x=(data['shaped'] if shaped else data['raw'])[:,axis]
    delayed=np.interp(data['grid']-delay,data['grid'],x,left=x[0]); alpha=np.exp(-dt/tau)
    return lfilter([gain*(1-alpha)],[1,-alpha],delayed,zi=[gain*x[0]*alpha])[0]

def frozen_pi(data, axis, config):
    from scipy.signal import cont2discrete,lfilter
    kp=config['rate_kp'][axis]; ki=config['rate_ki'][axis]; gain=config['rate_gain'][axis]
    num,den,_=cont2discrete(([gain*kp,gain*ki],[1,kp,ki]),data['dt'])
    x=data['raw'][:,axis]; delayed=np.interp(data['grid']-config['rate_delay_s'][axis],data['grid'],x,left=x[0])
    return lfilter(num[0],den,delayed)

def fit_and_validate(data):
    """Isolated effective-rate prediction; no physical parameter changes.

    Fit both models on exactly the two nominal route traces plus original
    low-amplitude system-ID1001. Fixed native shaping introduces no fit
    parameters. Remainder gain/tau/delay are reidentified, not copied from the
    aggregate fit, so native dynamics are not counted twice.
    """
    from scipy.optimize import minimize
    training=[x for x in data if x['record'] and x['record']['job']['mission']=='route' and x['record']['job']['scenario']=='nominal']
    for name in ('system_id_multisine_1001','system_id_validation_1002'):
        print('Reading',name,flush=True)
        x=load_run(ROOT/'artifacts/gazebo'/name); describe(x);data.append(x)
        if name.endswith('1001'):training.append(x)
    assert len(training)==3
    config=read(ROOT/'quadrl/plant_config.json'); fit={}; comparisons=[]
    for x in data:
        grid=x['grid']; x['model_mask']=(grid>=x['offset']+.5)&(grid<=x['offset']+x['end'])
        x['true_grid']=np.column_stack([np.interp(grid,x['rate'][:,0],x['rate'][:,4+a]) for a in range(3)])
    for shaped,name in ((False,'refit_linear'),(True,'native_shaper_plus_refit_remainder')):
        fit[name]=[]
        for axis in range(3):
            def objective(p):
                losses=[]
                for x in training:
                    m=x['model_mask']; y=x['true_grid'][m,axis]; pred=first_order(x,axis,p,shaped)[m]
                    losses.append(np.mean((pred-y)**2)/max(.0025,np.mean(y*y)))
                return float(np.mean(losses))
            starts=([1.,.025,.01],[1.,.035,.02],[1.,.02,.0075]) if shaped else ([1.,.05,.03],[1.,.035,.02],[1.,.06,.04])
            candidates=[minimize(objective,start,method='L-BFGS-B',bounds=[(.85,1.15),(.001,.15),(0.,.10)],
                options={'maxiter':100,'ftol':1e-11}) for start in starts]
            result=min(candidates,key=lambda r:r.fun)
            fit[name].append({'axis':('roll','pitch','yaw')[axis],'gain':float(result.x[0]),'tau_s':float(result.x[1]),'delay_s':float(result.x[2]),
                'normalized_training_mse':float(result.fun),'optimizer_success':bool(result.success),'message':result.message,
                'multistart_objectives':[float(c.fun) for c in candidates]})
    for x in data:
        row={'run':rel(x['folder']),'split':'fit' if any(x is y for y in training) else 'held_out','models':{}}
        m=x['model_mask']
        for name in ('frozen_aggregate_PI','refit_linear','native_shaper_plus_refit_remainder'):
            vals=[]
            for a in range(3):
                if name=='frozen_aggregate_PI':pred=frozen_pi(x,a,config)
                else:
                    f=fit[name][a];pred=first_order(x,a,[f['gain'],f['tau_s'],f['delay_s']],name.startswith('native'))
                error=pred[m]-x['true_grid'][m,a]
                cell={'axis':('roll','pitch','yaw')[a],'mse_rad_s_squared':float(np.mean(error**2)),
                    'rmse_rad_s':float(np.sqrt(np.mean(error**2)))}
                if x['record'] and x['record']['job']['mission']=='route':
                    # Isolate sparse command discontinuities from the remaining
                    # smooth rate-reversal/braking portions of each route.
                    jumps=np.where(abs(np.diff(x['cmd'][:,a]))>.5)[0]+1
                    jumpwindow=np.zeros(len(x['grid']),dtype=bool)
                    for j in jumps:jumpwindow|=(x['grid']>=x['cb'][j])&(x['grid']<=x['cb'][j]+.30)
                    for flag,label in ((jumpwindow,'first300ms_after_jump_over_0p5'),(~jumpwindow,'remaining_smooth_command')):
                        mm=m&flag
                        cell[label]={'samples':int(mm.sum()),'mse_rad_s_squared':float(np.mean((pred[mm]-x['true_grid'][mm,a])**2)) if np.any(mm) else None}
                vals.append(cell)
            row['models'][name]=vals
        comparisons.append(row)
    return {'status':'DIAGNOSTIC_ONLY_NOT_DEPLOYED','training_runs':[rel(x['folder']) for x in training],
        'forward_model_inputs':{
            'prediction':'Only controller command time/value history and own simulated shaper/filter internal states. No recorded gyro, attitude, quaternion, ATTITUDE_TARGET, or BIN desired rate enters prediction.',
            'initialization':'Shaper target rate starts from first controller command, target acceleration zero; first-order remainder starts gain times first command. No measured state initialization.',
            'training_teacher_use':'Recorded BIN gyro enters optimization residuals only. Held-out gyro enters score only. Recorded ATTITUDE_TARGET enters separate descriptive shaper verification only.',
        },
        'discretization':{'native_dt_s':.0025,'native_tc_s':.025,
            'RP_accel_limit_rad_s2':float(np.deg2rad(1100.)),'RP_jerk_limit_rad_s3':float(np.deg2rad(1100.)/.025),
            'shaper_equations':['a_goal=clip((command-v_target)/tc,-A,A)','a_target+=clip(a_goal-a_target,-J*dt,J*dt)','v_target+=a_target*dt'],
            'remainder':'alpha=exp(-dt/tau); rate[k]=alpha*rate[k-1]+(1-alpha)*gain*shaped_rate(t[k]-delay); fractional delay uses linear interpolation of simulated target history',
            'old_aggregate_removed':'Native-shaper candidate never calls frozen_pi and does not reuse its tau/delay/ki. Its entire post-shaper remainder is independently refit. Frozen_PI is a comparison baseline only.',
            'yaw_recommendation':'Retain old yaw dynamics; fitted nonlinear yaw is reported for diagnostic comparison only.'},
        'heldout_definition':'All other completed wind route/hover traces plus system-ID validation1002; no policy performance claim',
        'objective':'Mean per-run rate MSE divided by max(0.0025,mean(true_rate^2)); same 3 training traces for both models. Each axis independently fits gain[.85,1.15], tau[.001,.15]s, delay[0,.10]s. Native shaper parameters are fixed from actual logs.',
        'limitations':'Rate-only recorded-input prediction omits attitude-correction coupling, torque saturation and aerodynamic gyro disturbances. Frozen_PI is its unsaturated scalar linear rate subsystem, not a rerun of full Fast plant.',
        'fitted_parameters':fit,'comparisons':comparisons,'frozen_config_sha256':sha(ROOT/'quadrl/plant_config.json')}

def summarize(runs,fit):
    comparisons=fit['comparisons']; windroutes=[r for r in comparisons if r['split']=='held_out' and '/runs/' in r['run'] and '_route_' in r['run']]
    means={}; smooth={}
    for name in ('frozen_aggregate_PI','refit_linear','native_shaper_plus_refit_remainder'):
        means[name]=[float(np.mean([r['models'][name][a]['mse_rad_s_squared'] for r in windroutes])) for a in range(3)]
        smooth[name]=[float(np.mean([r['models'][name][a]['remaining_smooth_command']['mse_rad_s_squared'] for r in windroutes])) for a in range(3)]
    route_reports=[x['report'] for x in runs if x['record'] and x['record']['job']['mission']=='route']
    reversals=[e for r in route_reports for e in r['substantial_reversals']]
    corners=[c for r in route_reports for c in r['corner_rate_relation']]
    small=next(x for x in comparisons if x['run'].endswith('system_id_validation_1002'))
    return {'held_out_wind_route_count':len(windroutes),'held_out_wind_route_mean_MSE_by_RPY':means,
        'held_out_wind_route_smooth_command_mean_MSE_by_RPY':smooth,
        'held_out_RP_mean_MSE_reduction_vs_frozen_PI_fraction':float(1-np.mean(means['native_shaper_plus_refit_remainder'][:2])/np.mean(means['frozen_aggregate_PI'][:2])),
        'held_out_RP_mean_MSE_reduction_vs_refit_linear_fraction':float(1-np.mean(means['native_shaper_plus_refit_remainder'][:2])/np.mean(means['refit_linear'][:2])),
        'held_out_small_signal_validation_1002_RPY_RMSE':{name:[r['rmse_rad_s'] for r in values] for name,values in small['models'].items()},
        'route_reversal_shaped_sign_delay_s':quantile([e['shaped_sign_cross_delay_s'] for e in reversals if e['shaped_sign_cross_delay_s'] is not None]),
        'route_reversal_gyro_sign_delay_s':quantile([e['gyro_sign_cross_delay_s'] for e in reversals if e['gyro_sign_cross_delay_s'] is not None]),
        'corner_RP_cap_fraction_in_pm500ms':quantile([max(c['native_RP_accel_cap_fraction_in_pm500ms']) for c in corners]),
        'validation_limits':'Four wind route traces use one fixed policy111 and one environment seed83301 across scenario/interface cases. They are held out from this physical-response fit, not a new independent policy acceptance set.',
        'conclusion':'Native shaping is directly observed and its fixed explicit model plus reidentified remainder strongly improves held-out roll/pitch rate prediction while preserving/improving measured small-signal validation. Hard acceleration saturation is sparse and concentrated in goal-activation jumps; smooth braking reversals retain median31ms command-to-shaped sign lag and median63ms command-to-gyro sign lag. Recorded-input rate prediction cannot establish a counterfactual corner-position improvement or any new policy result. Keep yaw aggregate dynamics pending a better validated yaw remainder.',
        'next_step_scope':'Promising for a separate new Fast plant namespace plus retraining/revalidation. No parameters installed; physical mass/inertia and all frozen files unchanged.'}

def main():
    runs=[]
    for record,folder in completed():
        if record['job']['method']=='PID':continue
        print('Reading',rel(folder),flush=True)
        data=load_run(folder,record);describe(data);runs.append(data)
    fit=fit_and_validate(runs)
    native=Path('[external-home]/vkr_sim/ardupilot')
    source_paths=[native/'libraries/AP_Math/control.cpp',native/'libraries/AC_AttitudeControl/AC_AttitudeControl.cpp',native/'libraries/AC_AttitudeControl/AC_AttitudeControl.h',native/'ArduCopter/GCS_MAVLink_Copter.cpp']
    result={'schema':'joint-nonlinear-rate-diagnosis-v2.0','scope':'DIAGNOSTIC, read-only archived development traces; no simulator, policy evaluation, training, or frozen plant changes',
        'inputs_complete_receipt':{'path':rel(COMPLETE),'sha256':sha(COMPLETE)},
        'source_evidence':{str(p):sha(p) for p in source_paths},
        'method':'Native fixed shaper is reproduced with measured 400Hz/ACC/TC parameters. ATTITUDE_TARGET carries feedforward _ang_vel_target_rads; BIN RATE desired includes attitude correction; BIN RATE R/P/Y is gyro. All lags use boot clocks, without fitted gains.',
        'runs':[x['report'] for x in runs],'effective_rate_model_comparison':fit,
        'findings':summarize(runs,fit),'analysis_script_sha256':sha(Path(__file__))}
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Wrote',rel(OUT),flush=True)

def inspect():
    from pymavlink import mavutil
    record, folder = next((r,p) for r,p in completed() if r['job']['method']=='body_collective' and r['job']['scenario']=='nominal')
    types = Counter(); examples = {}
    for row in lines(folder/'mavlink.jsonl'):
        m=row['message']; k=m['mavpackettype']; types[k]+=1
        if k in ('ATTITUDE_TARGET','ATTITUDE') and k not in examples and m.get('time_boot_ms',0)>35000: examples[k]=row
    print(json.dumps({'run':rel(folder),'mav_types':types,'examples':examples},indent=2))
    conn=mavutil.mavlink_connection(str(folder/'sitl/logs/00000001.BIN'))
    counters=Counter(); params={}; examples={}
    while True:
        m=conn.recv_match()
        if m is None:break
        k=m.get_type(); counters[k]+=1
        if k=='PARM' and any(x in m.Name for x in ('ACC_', 'RATE_TC','SCHED_LOOP','RATE_FF')):params[m.Name]=m.Value
        if k in ('RAT','ATT','RATE','GUIA') and k not in examples and getattr(m,'TimeUS',0)>35000000:examples[k]=m.to_dict()
    print(json.dumps({'BIN_types':counters,'params':params,'examples':examples},indent=2))
    ap=Path('[external-home]/vkr_sim/ardupilot')
    for path in (ap/'libraries/AP_Math/control.cpp',ap/'libraries/AP_Math/control.h'):
        if not path.exists(): continue
        text=path.read_text(); at=text.find('void shape_vel_accel')
        if at>=0:print(str(path),text[at:at+5500])
    for path,token in ((ap/'libraries/AP_Math/control.cpp','float sqrt_controller_accel'),(ap/'ArduCopter/GCS_MAVLink_Copter.cpp','void GCS_MAVLINK_Copter::send_attitude_target'),(ap/'libraries/AC_AttitudeControl/AC_AttitudeControl.h','get_attitude_target_ang_vel_rads')):
        text=path.read_text(); at=text.find(token)
        print(str(path),text[at:at+3300])
    print('Completed',[(x['job']['name'], x['job']['method']) for x,p in completed()])

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--inspect',action='store_true'); args=parser.parse_args()
    if args.inspect:inspect()
    else:main()
