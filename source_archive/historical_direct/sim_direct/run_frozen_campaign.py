"""Crossed, frozen direct-Gazebo study; three isolated actuator-proxy lanes."""
import argparse,concurrent.futures,json,random,subprocess,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from contract import verify_contract

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    freeze=json.loads(a.freeze_file.read_text());out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    models=[m for m in freeze['models'] if m['architecture']=='direct']
    if sorted(m['training_seed'] for m in models)!=sorted(freeze['training_seeds']):raise ValueError('Incomplete direct cohort')
    profile=ROOT/freeze['pid_profiles']['gazebo']['path'];cal=ROOT/freeze['deployment']['gazebo']['direct_calibration']['path']
    ap=Path('[external-home]/vkr_sim/ardupilot');gz=Path('[external-home]/ardupilot_gazebo')
    external={'ardupilot_binary':ap/'build/sitl/bin/arducopter','default_copter':ap/'Tools/autotest/default_params/copter.parm','default_gazebo':ap/'Tools/autotest/default_params/gazebo-iris.parm','gazebo_plugin':gz/'build/libArduPilotPlugin.so','iris_base_sdf':gz/'models/iris_with_standoffs/model.sdf','iris_control_sdf':gz/'models/iris_with_ardupilot/model.sdf'}
    jobs=[]
    for model in models:
        for mission in freeze['missions']:
            for scenario in freeze['scenarios']['gazebo']:
                for seed in freeze['environment_seeds']['gazebo']:
                    verify_contract(a.freeze_file,backend='gazebo',controller='direct',env_seed=seed,mission=mission,scenario=scenario,duration_s=freeze['horizon_s'][mission],policy=ROOT/model['path'],train_seed=model['training_seed'],pid_profile=profile,calibration=cal,external_paths=external)
                    jobs.append({'model':model,'mission':mission,'scenario':scenario,'env_seed':seed})
    random.Random(487014).shuffle(jobs);out.mkdir(parents=True)
    for index,job in enumerate(jobs):
        job['instance']=index%3;job['name']=f"direct_train{job['model']['training_seed']}_{job['mission']}_{job['scenario']}_env{job['env_seed']}"
    (out/'campaign_plan.json').write_text(json.dumps({'jobs':jobs,'job_count':len(jobs),'contract':str(a.freeze_file),'order_seed':487014},indent=2))
    stop=threading.Event();lock=threading.Lock();results=[]
    def lane(instance):
        for job in [j for j in jobs if j['instance']==instance]:
            if stop.is_set():return
            run=out/job['name'];model=job['model']
            args=[sys.executable,str(ROOT/'sim_direct/run.py'),'--controller','direct','--policy',str(ROOT/model['path']),'--train-seed',str(model['training_seed']),'--mission',job['mission'],'--scenario',job['scenario'],'--seed',str(job['env_seed']),'--instance',str(instance),'--split','heldout','--freeze-file',str(a.freeze_file.resolve()),'--pid-profile',str(profile),'--calibration',str(cal),'--output',str(run)]
            with (out/(job['name']+'.log')).open('w') as log:code=subprocess.call(args,stdout=log,stderr=subprocess.STDOUT)
            summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else {'status':'ERROR','error':'Missing summary'}
            with lock:
                results.append({'run':str(run),'status':summary['status'],'returncode':code,'error':summary.get('error')})
                (out/'campaign_progress.json').write_text(json.dumps({'completed':len(results),'planned':len(jobs),'results':results},indent=2))
            if summary['status']=='ERROR':stop.set();return
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for future in [executor.submit(lane,i) for i in range(3)]:future.result()
    complete=len(results)==len(jobs) and not stop.is_set()
    (out/'campaign_summary.json').write_text(json.dumps({'status':'COMPLETE' if complete else 'INCOMPLETE','planned':len(jobs),'completed':len(results),'results':results},indent=2))
    return 0 if complete else 1
if __name__=='__main__':raise SystemExit(main())
