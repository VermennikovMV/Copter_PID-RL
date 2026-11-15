"""Fresh manual V21 fine-tuning, fail-closed until ROOT binds its reviewed plan.

Preparation imports only the standard-library official preflight module. Only
--run starts actual PyBullet PPO; it never starts Gazebo or a web server.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import joint_research_preflight_v1 as gate

BINDING=ROOT/'memory21_recording_source_v1.json'
SOURCE='docs/joint_route_memory_train_plans_v21/zero_route_integral.bound.json'
ARM='zero_route_integral'
BUDGET=dict(ppo_steps=262144,critic_steps=131072,eval_every=131072,candidate_count=3)


def source_binding():
    binding=json.loads(BINDING.read_text(encoding='utf-8-sig'))
    if binding.get('schema')!='memory21-manual-training-source-v1' or binding.get('status')!='READY_BOUND_TO_ROOT_PLAN':
        raise RuntimeError('Manual training is not bound yet: ROOT reviewed BOUND v21 plan is required; no training or output has been created.')
    source=binding.get('source_plan',{})
    if source.get('path')!=SOURCE or not re.fullmatch('[0-9a-f]{64}',source.get('sha256') or ''):
        raise ValueError('Exact ROOT reviewed source path/SHA required')
    if not (ROOT/SOURCE).is_file() or gate.sha(ROOT/SOURCE)!=source['sha256']:
        raise ValueError('Reviewed source plan absent or SHA changed')
    return source,json.loads((ROOT/SOURCE).read_text(encoding='utf-8-sig'))


def prepare():
    source,original=source_binding()
    argv=original['argv']
    if original.get('status')!='READY_FOR_RUN' or original.get('budget')!=BUDGET:
        raise ValueError('Only reviewed bounded v21 pilot recipe is supported')
    if original.get('sources',{}).get('driver',{}).get('path')!='scripts/train_joint_route_memory_v21.py' or argv.count('--variant')!=1 or argv[argv.index('--variant')+1]!=ARM:
        raise ValueError('Only exact conditioned v21 hybrid continuation may be reproduced')
    for option in ('--output','--research-plan'):
        if argv.count(option)!=1:raise ValueError('One explicit '+option+' required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent=ROOT/'artifacts/refinement/joint_v2'/('recording_memory21_'+stamp+'_'+uuid.uuid4().hex[:8])
    plan_path=parent/'research_plan.json';output=parent/ARM
    plan=copy.deepcopy(original)
    plan['argv'][plan['argv'].index('--output')+1]=output.relative_to(ROOT).as_posix()
    plan['argv'][plan['argv'].index('--research-plan')+1]=plan_path.relative_to(ROOT).as_posix()
    plan['recording_reproduction']=dict(
        title='Fresh actual conditioned route-memory v21 fine-tuning for user recording',
        source_plan=source,prepared_utc=datetime.now(timezone.utc).isoformat(),
        changed_only='argv --output and --research-plan paths plus this recording annotation',
        initial_already_trained=True,source_actor_npz_sha256='68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5',
        viewed_development_seeds_reused=True,independent_scientific_replication=False,
        zero_update_conditioning_is_not_PPO=True,current_Gazebo_actor_automatically_replaced=False,
        interpretation='Real new PyBullet critic warmup131072 and PPO262144 from already trained weights; no from-scratch/replay/Gazebo-training/acceptance claim.')
    recovered=copy.deepcopy(plan);recovered.pop('recording_reproduction');recovered['argv']=copy.deepcopy(original['argv'])
    if recovered!=original:raise AssertionError('Recording reproduction changed scientific settings')
    # This validates the NEW output, all source hashes, every prerequisite and13 mappings.
    # The original source output may already exist and is intentionally preserved.
    gate.validate(plan)
    parent.mkdir(exist_ok=False)
    with plan_path.open('x',encoding='utf-8') as stream:json.dump(plan,stream,ensure_ascii=False,indent=2);stream.write('\n')
    gate.validate(plan)
    return dict(status='PREPARED_NOT_TRAINED',plan=plan_path.relative_to(ROOT).as_posix(),plan_sha256=gate.sha(plan_path),
                output=output.relative_to(ROOT).as_posix(),source_plan_sha256=source['sha256'],
                source_binding_sha256=gate.sha(BINDING),changed_only_declared_paths_and_annotation=True,
                monitor_command='.\\show_joint_training_native_v21.cmd -RunRoot '+output.relative_to(ROOT).as_posix(),
                new_environment_steps=0,new_optimizer_steps=0,Gazebo_started=False,current_actor_replaced=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--prepare-only',action='store_true');modes.add_argument('--run',action='store_true');modes.add_argument('--inspect-binding',action='store_true')
    args=parser.parse_args()
    if args.inspect_binding:
        record=json.loads(BINDING.read_text(encoding='utf-8-sig'))
        print(json.dumps(dict(status='BINDING_INSPECTION_ONLY',binding=record,new_environment_steps=0,new_optimizer_steps=0),ensure_ascii=False));return 0
    try:
        receipt=prepare()
    except (FileNotFoundError,RuntimeError) as exc:
        print(json.dumps(dict(status='WAIT_ROOT_BOUND_PLAN',reason=str(exc),new_environment_steps=0,new_optimizer_steps=0,output_created=False),ensure_ascii=False),flush=True)
        return 2
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    if args.run:
        return subprocess.run([sys.executable,'-B',str(ROOT/'scripts/joint_research_preflight_v1.py'),'--plan',receipt['plan'],'--run'],cwd=ROOT,check=False).returncode
    return 0


if __name__=='__main__':raise SystemExit(main())
