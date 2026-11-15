"""Collect the prospective96-episode mixture on the detailed direct plant."""
import json,sys,hashlib,shutil,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference.collect_mixed_demonstrations import MixedDemonstrationEnv,episode_design
from reference.collect_demonstrations import collect_dataset
from reference.direct_teacher import DirectTeacher
from quadrl.env import QuadEnv
from quadrl.protocol import Mission

class DirectMixedEnv(MixedDemonstrationEnv):
    def __init__(self,specs):
        super().__init__(specs);self.architecture='direct'
    def reset(self,seed):
        self.close();spec=self.specs[seed]
        self._env=QuadEnv('direct',training=spec['training_domain'],mission=spec['mission'],seconds=self.seconds)
        self._env.training_progress=1.
        obs,info=self._env.reset(seed=seed)
        self._env.mission=Mission(spec['mission']);self._env.target=self._env.mission.target
        info['collection_episode_spec']=spec
        return obs,info

parser=argparse.ArgumentParser();parser.add_argument('--seed-start',type=int,default=4001)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
seeds=list(range(args.seed_start,args.seed_start+96));out=args.output.resolve()
env=DirectMixedEnv(episode_design(seeds,design_seed=3000))
try:
    before={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['reference/direct_teacher.py','reference/collect_direct_demonstrations.py','quadrl/pid_config.json']}
    manifest=collect_dataset(env,DirectTeacher(),seeds,out)
    extra={}
    for name in ['reference/direct_teacher.py','reference/collect_direct_demonstrations.py','quadrl/pid_config.json']:
        dst=out/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        extra[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
        if extra[name]!=before[name]:raise RuntimeError('Direct collection source changed')
    (out/'direct_sources.json').write_text(json.dumps({'architecture':'direct','source_hashes':extra},indent=2))
    print(json.dumps({k:manifest[k] for k in ['samples','failures','dataset_sha256']}),flush=True)
finally:env.close()
