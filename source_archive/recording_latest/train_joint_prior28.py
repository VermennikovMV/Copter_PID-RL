"""Fresh manual reproduction of the bounded V28 prior0 PPO fine-tuning recipe.

Preparation uses only stdlib and the official stdlib preflight. Only --run
starts real PyBullet training; no Gazebo, viewer or policy promotion is started.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import joint_research_preflight_v1 as gate

BINDING = ROOT / 'docs/joint_prior28_manual_v1/source_binding.json'
PINS = {
    'source_plan': ('docs/joint_prior_ablation_plans_v28/prior0.bound.json', '034d59d74efd223360db63a03fee40d1f8eaf5ecd8cff2f0c11a2c8bdec34818'),
    'driver': ('scripts/train_joint_prior_ablation_v28.py', '0c948b91c39455aba867a354db270fc5253ddf61855d1b94b5df8222539d9134'),
    'preflight': ('scripts/joint_research_preflight_v1.py', '27be46baf162a3b096f25a16299c3b64fd38f42c955cdddee2f7550c924c7dd7'),
}
BUDGET = dict(ppo_steps=262144, critic_steps=131072, eval_every=131072, candidate_count=3)


def source_binding():
    binding = json.loads(BINDING.read_text(encoding='utf-8-sig'))
    if binding.get('schema') != 'prior28-manual-training-source-v1' or binding.get('status') != 'READY_BOUND_TO_ROOT_PLAN':
        raise ValueError('Exact reviewed V28 manual binding is required')
    for role, (path, digest) in PINS.items():
        if binding.get(role) != dict(path=path, sha256=digest) or gate.sha(ROOT / path) != digest:
            raise ValueError('Reviewed source binding changed: ' + role)
    original = json.loads((ROOT / PINS['source_plan'][0]).read_text(encoding='utf-8-sig'))
    if original.get('status') != 'READY_FOR_RUN' or original.get('budget') != BUDGET or original['sources']['driver'] != binding['driver']:
        raise ValueError('Only the exact bounded prior0 recipe is supported')
    argv = original['argv']
    for option in ('--variant', '--output', '--research-plan'):
        if argv.count(option) != 1:
            raise ValueError('One explicit ' + option + ' required')
    if argv[argv.index('--variant') + 1] != 'prior0':
        raise ValueError('Only the prior0 arm may be reproduced')
    return binding, original


def prepare():
    binding, original = source_binding()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent = ROOT / 'artifacts/refinement/joint_v2' / ('recording_prior28_' + stamp + '_' + uuid.uuid4().hex[:8])
    plan_path, output = parent / 'research_plan.json', parent / 'prior0'
    plan = copy.deepcopy(original)
    for option, path in (('--output', output), ('--research-plan', plan_path)):
        plan['argv'][plan['argv'].index(option) + 1] = path.relative_to(ROOT).as_posix()
    plan['recording_reproduction'] = dict(
        source_plan=binding['source_plan'], prepared_utc=datetime.now(timezone.utc).isoformat(),
        changed_only='argv --output and --research-plan paths plus this recording annotation',
        scope='Fresh prior0 arm only: 131072 critic + 262144 PPO steps from already trained source68; not from scratch or Gazebo training.',
        actor='Full 27396 mean parameters; source action std frozen; kappa=0; zero_route_integral conditioning.',
        viewed_development_seeds_reused=True, independent_scientific_replication=False,
        source_result='Native route gain, strong-hover regression, FULL research admission FAIL; no inherited Gazebo result or automatic promotion.',
        original_two_arm_experiment_text_is_historical_context=True,
        prior001_and_archived_v24_retrained=False)
    recovered = copy.deepcopy(plan); recovered.pop('recording_reproduction')
    expected = [original['argv'].index(option) + 1 for option in ('--output', '--research-plan')]
    changed = [i for i, (a, b) in enumerate(zip(original['argv'], plan['argv'])) if a != b]
    if len(plan['argv']) != len(original['argv']) or sorted(changed) != sorted(expected):
        raise AssertionError('Only the two declared argv values may change')
    for index in expected:
        recovered['argv'][index] = original['argv'][index]
    if recovered != original:
        raise AssertionError('Scientific recipe changed')
    gate.validate(plan)
    parent.mkdir(exist_ok=False)
    with plan_path.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2); stream.write('\n')
    gate.validate(plan)
    return dict(status='PREPARED_NOT_TRAINED', plan=plan_path.relative_to(ROOT).as_posix(), plan_sha256=gate.sha(plan_path),
                output=output.relative_to(ROOT).as_posix(), source_plan_sha256=binding['source_plan']['sha256'],
                source_binding_sha256=gate.sha(BINDING), validated_metric_count=len(plan['metric_alignment']),
                expected_progress_files={name: (output / name).relative_to(ROOT).as_posix() for name in
                                         ('phase.json', 'warmup/progress.csv', 'progress.csv', 'warmup.json', 'selection.json', 'complete.json', 'error.json')},
                terminal_output='Official preflight and driver; CSV logging, no viewer started.',
                tensorboard_event_files=[p.relative_to(ROOT).as_posix() for p in output.rglob('events.out.tfevents.*')],
                new_environment_steps=0, new_optimizer_steps=0, Gazebo_started=False, current_actor_replaced=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ('prepare-only', 'run', 'inspect-binding'):
        modes.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    if args.inspect_binding:
        binding, _ = source_binding()
        print(json.dumps(dict(status='BINDING_INSPECTION_ONLY', binding=binding, new_environment_steps=0, new_optimizer_steps=0), ensure_ascii=False)); return 0
    receipt = prepare()
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if args.run:
        return subprocess.run([sys.executable, '-B', str(ROOT / PINS['preflight'][0]), '--plan', receipt['plan'], '--run'], cwd=ROOT, check=False).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
