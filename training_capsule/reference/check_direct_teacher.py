import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference.direct_teacher import DirectTeacher
from quadrl.rollout import run_episode
out=ROOT/'reference/artifacts/direct_teacher_dev_1201'
out.mkdir(parents=True,exist_ok=False)
results=[]
for scenario in ['nominal','gust','mass_40','motor_20']:
    for mission in ['hover','route']:
        result=run_episode(DirectTeacher(),architecture='direct',scenario=scenario,mission=mission,seed=1201,save_path=out/f'{scenario}_{mission}')
        results.append(result)
        print(json.dumps({'scenario':scenario,'mission':mission,'metrics':result['metrics']}),flush=True)
(out/'results.json').write_text(json.dumps(results,indent=2))
