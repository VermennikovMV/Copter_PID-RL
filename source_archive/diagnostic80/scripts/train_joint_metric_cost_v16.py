"""Finite-episode objective pair; all physics requires a separate bound ROOT GO.

Reuse the frozen v15 controller, masks and independent full-horizon evaluator.
Both arms use gamma1 and true training-deadline terminals. Only route reward
differs. The shared24 initial evaluation is reused, not rerun or duplicated.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from scripts import train_joint_route_basis_v15 as v15
from scripts.bc_trust_region_probe_v1 import BCTrustPPO, actor_name, sha
from quadrl_metric_cost_v16.env import ARMS, BASIS_KIND, JointTaskEnv
from quadrl_metric_cost_v16.reward import synthetic_fixtures

SCHEMA = "joint-finite-metric-cost-ppo-v16.0"
INITIAL, INITIAL_SHA = v15.INITIAL, v15.INITIAL_SHA
BASELINE = ROOT / "artifacts/refinement/joint_v2/route_basis_v15_baseline111/velocity_residual/baseline.json"
BUDGET, PROBE_BUDGET = v15.BUDGET.copy(), v15.PROBE_BUDGET.copy()
write_json, read, utc, phase = v15.write_json, v15.read, v15.utc, v15.phase


def source_hashes():
    old = v15.source_hashes()
    own = [Path(__file__), ROOT / "scripts/run_joint_metric_cost_pair_v16.py",
           *sorted((ROOT / "quadrl_metric_cost_v16").glob("*.py"))]
    return {**old, **{p.relative_to(ROOT).as_posix(): sha(p) for p in own}}


def baseline():
    record = v15.validate_baseline(BASELINE, BASIS_KIND)
    assert len(record["episodes"]) == 24 and record["initial_sha256"] == INITIAL_SHA
    return record


def vector(arm, count, seed):
    env = VecMonitor(DummyVecEnv([lambda: JointTaskEnv(variant=arm, architecture="hybrid", training=True) for unused in range(count)]))
    env.set_attr("training_progress", 1.)
    env.seed(seed)
    return env


def assert_hyperparameters(model, probe=False):
    assert model.gamma == model.rollout_buffer.gamma == 1.
    assert model.gae_lambda == .97 and model.max_grad_norm == .5
    assert all(model.lr_schedule(p) == 3e-5 and model.clip_range(p) == .15 for p in (0., .5, 1.))
    assert (model.n_steps, model.batch_size, model.n_epochs) == ((128, 64, 2) if probe else (512, 1024, 8))


def new_model(env, seed, output, probe=False):
    options = dict(gamma=1.)
    if probe: options.update(n_steps=128, batch_size=64, n_epochs=2)
    model = BCTrustPPO.load(INITIAL, env=env, device="cpu", seed=seed, **options)
    assert_hyperparameters(model, probe)
    assert not model.policy.optimizer.state
    model.policy.optimizer.state.clear()
    model.anchor_steps = 0
    model.configure_bc_prior(INITIAL, 0., output / "on_policy_prior_kl.jsonl")
    model.bc_kl_enabled = False
    write_json(output / "optimizer_reset.json", dict(initializer_states=0, after_clear=0, reset=[0, 0],
        gamma=1., rollout_buffer_gamma=1., source13to0_reset_belongs_v14=True))
    return model


class ObjectiveTrace(v15.ProbeTrace):
    """Archive existing actions and objective accounting; no additional step."""
    def _on_step(self):
        result = super()._on_step()
        mapping = dict(parent_reward="parent_reward", metric_reward="metric_reward", cost_rate="running_cost",
            terminal_penalty="terminal_penalty", completed_before="completed_before", active_PRE_leg="active_PRE_leg",
            finite_distance_sq="finite_distance_sq", learning_terminated="learning_terminated", learning_truncated="learning_truncated")
        for destination, source in mapping.items():
            values = [np.asarray(info["metric_cost_ledger"][source]) for info in self.locals["infos"]]
            self.rows.setdefault(destination, []).append(np.stack(values))
        for name in ("failed", "finite_task_deadline", "TimeLimit.truncated"):
            self.rows.setdefault(name, []).append(np.asarray([info.get(name, False) for info in self.locals["infos"]]))
        return result


def objective_report(guard):
    report = guard.report()
    report.update(gamma=1., rollout_buffer_gamma=guard.model.rollout_buffer.gamma,
                  common_training_deadlines_true_finite_task_terminals=True,
                  common_change_vs_v15="gamma1 AND zero training-timeout bootstrap; physical/evaluation horizons unchanged",
                  warm_critic_Adam_equality_required=False,
                  warm_difference="Reward and critic/Adam can differ; actor/actions/physical inputs must match")
    return report


def require_physical_plan(args, sources):
    from scripts import joint_research_preflight_v1 as preflight
    plan = read(args.physical_plan)
    assert plan["schema"] == "joint-metric-cost-physical-preparation-plan-v16.0"
    assert plan["status"] == "READY_FOR_ROOT_AUTHORIZED_PHYSICAL_PREPARATION"
    assert plan["mode"] == "semantic_pair512" and plan["budget"] == PROBE_BUDGET
    assert plan["source_sha256"] == sources and plan["output"] == args.output.relative_to(ROOT).as_posix()
    binding = plan["root_physical_authorization"]
    path = preflight.bound_file(ROOT, binding)
    assert binding["equals"] is True and preflight.json_pointer(read(path), binding["pointer"]) is True
    return plan


def static_check(args, sources):
    args.output.mkdir(parents=True, exist_ok=False)
    model = new_model(None, 9511, args.output)
    initial = copy.deepcopy(model.policy.state_dict())
    guard = v15.LimitedBasisGuard(model)
    try:
        fixture = synthetic_fixtures()
        assert model.get_env() is None
        guard.set_stage("ppo")
        x = np.random.default_rng(174916).normal(size=(1000, 80)).astype(np.float32)
        with torch.no_grad():
            assert float(model.prior_kl(torch.as_tensor(x))) == 0.
        v15.export(model, args.output / "initial_policy.npz", BASIS_KIND)
        policy = v15.NumpyPolicyRoute80(args.output / "initial_policy.npz", expected_kind=BASIS_KIND)
        parity = float(np.max(np.abs(model.predict(x, deterministic=True)[0] - policy.predict(x))))
        assert parity < 1e-5 and not model.policy.optimizer.state
        assert all(torch.equal(value, initial[name]) for name, value in model.policy.state_dict().items())
        assert all(parameter.grad is None for parameter in model.policy.parameters())
        baseline_record = baseline()
        assert source_hashes() == sources
        report = dict(schema=SCHEMA, status="PASS_STATIC_NO_ENV_OR_UPDATES", utc=utc(), source_sha256=sources,
            initial_sha256=INITIAL_SHA, gamma=1., rollout_buffer_gamma=model.rollout_buffer.gamma,
            effective_actor_trainable_elements=384, old77_actor_logstd_prior_exact=True,
            critic_new3_hardzero=True, pure_reward_and_deadline_fixtures=fixture,
            initial_current_prior_KL=0., strict80_Torch_NumPy_parity1000=parity,
            evaluation_engine="EXACT frozen v15 evaluate/NativeEvalEnv, unmodified", evaluation_basis_kind=BASIS_KIND,
            reused_baseline=dict(path=BASELINE.relative_to(ROOT).as_posix(), sha256=sha(BASELINE), episodes=len(baseline_record["episodes"])),
            new_baseline_episodes=0, physical_steps=0, environment_constructions=0, backward_calls=0, optimizer_steps=0,
            actual_pilot_executed=False, policy_quality_accepted=False)
        write_json(args.output / "static_receipt.json", report)
        assert not args.static_receipt.exists()
        write_json(args.static_receipt, report)
        print(json.dumps(dict(status=report["status"], receipt=str(args.static_receipt), sha256=sha(args.static_receipt))), flush=True)
    finally:
        guard.close()


def verify_pair(args, sources):
    plan = require_physical_plan(args, sources)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "prospective_plan.json", plan)
    arms = {}
    for arm in ARMS:
        output = args.output / arm; output.mkdir()
        env, guard = vector(arm, 1, 9511), None
        try:
            model = new_model(env, 9511, output, probe=True)
            guard = v15.LimitedBasisGuard(model)
            warm_trace = ObjectiveTrace()
            model.set_logger(configure(str(output / "critic"), ["csv"]))
            model.learn(128, reset_num_timesteps=True, callback=warm_trace, progress_bar=False)
            warm = objective_report(guard)
            assert warm["actual_optimizer_steps"] == dict(critic=4, ppo=0) and warm["Adam_parameter_states"] == 6
            assert not warm["changed_actor_tensors"]
            model.save(output / "critic_warmed_model"); warm_trace.save(output / "warm_trace.npz", BASIS_KIND)
            v15.start_ppo(model, guard)
            ppo_trace = ObjectiveTrace()
            model.set_logger(configure(str(output / "ppo"), ["csv"]))
            model.learn(128, reset_num_timesteps=True, callback=ppo_trace, progress_bar=False)
            final = objective_report(guard)
            assert final["actual_optimizer_steps"] == dict(critic=4, ppo=4) and final["Adam_parameter_states"] == 7
            assert final["changed_actor_tensors"] == [v15.ACTOR_W0]
            assert model.bc_kl_gradient_steps == model.ppo_gradient_steps == 4
            assert_hyperparameters(model, True)
            model.save(output / "probe_model"); ppo_trace.save(output / "ppo_trace.npz", BASIS_KIND)
            v15.export(model, output / "probe_policy.npz", BASIS_KIND)
            arms[arm] = dict(warmup=warm, PPO=final, actual_training_physical_steps=256,
                output_sha256={name: sha(output / name) for name in ("critic_warmed_model.zip", "warm_trace.npz", "probe_model.zip", "ppo_trace.npz", "probe_policy.npz")})
            write_json(output / "receipt.json", arms[arm])
        finally:
            if guard is not None: guard.close()
            env.close()
    from scripts.run_joint_metric_cost_pair_v16 import warmup_physical_equivalence
    warm = warmup_physical_equivalence(args.output)
    for arm in ARMS:
        for phase_name in ("warm_trace.npz", "ppo_trace.npz"):
            with np.load(args.output / arm / phase_name, allow_pickle=False) as trace:
                wanted = trace["parent_reward"] if arm == "w8_control" else trace["metric_reward"]
                np.testing.assert_allclose(trace["reward"], wanted, rtol=1e-6, atol=2e-6)
                assert np.all(~trace["TimeLimit.truncated"])
    assert source_hashes() == sources
    report = dict(schema=SCHEMA, status="PASS", utc=utc(), source_sha256=sources,
        prospective_plan_sha256=sha(args.physical_plan), arms=arms, warmup_physical_actor_equivalence=warm,
        pure_reward_deadline_fixtures=synthetic_fixtures(), total_training_physical_steps=512,
        additional_nontraining_physical_steps=0, total_optimizer_steps=dict(critic=8, PPO=8),
        common_gamma=1., true_training_deadline_terminal=True, physical_timeout_failure_claimed=False,
        actual_pilot_executed=False, new_baseline_episodes=0, evaluation_engine_unchanged=True,
        warm_reward_and_critic_Adam_equality_required=False, Gazebo_executed=False)
    write_json(args.output / "semantic_receipt.json", report)
    assert not args.semantic_receipt.exists()
    write_json(args.semantic_receipt, report)


def require_bound_preflight(args):
    from scripts import joint_research_preflight_v1 as preflight
    assert args.research_plan is not None
    plan = read(args.research_plan); command = preflight.validate(plan)
    assert plan["stage"] == "bounded_pilot" and plan["budget"] == BUDGET
    assert Path(command[2]).resolve() == Path(__file__).resolve() and sys.argv[1:] == plan["argv"]
    parent = (Path("/proc") / str(os.getppid()) / "cmdline").read_bytes().decode().split("\0")
    assert any(Path(item).name == "joint_research_preflight_v1.py" for item in parent) and "--run" in parent
    assert (ROOT / parent[parent.index("--plan") + 1]).resolve() == args.research_plan


def run_arm(args, sources):
    initial_baseline = baseline()
    output = args.output; output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = dict(schema=SCHEMA, variant=args.variant, basis_kind=BASIS_KIND, observation_contract=v15.CONTRACT_VERSION,
        actor_critic_observations=80, initial_sha256=INITIAL_SHA, prior_sha256=INITIAL_SHA,
        budget=BUDGET, common_gamma=1., training_deadlines_true_terminal=True,
        changes_vs_v15="gamma1 AND finite-task deadlines/zero bootstrap; betweenarm entire route objective only",
        gamma_reward_causal_scope="Do not attribute change versus old gamma.995 to route reward alone",
        new_baseline_episodes=0, reused_baseline=dict(path=BASELINE.relative_to(ROOT).as_posix(), sha256=sha(BASELINE), episodes=24),
        evaluation_engine="EXACT frozen v15 evaluate; native horizons/metric/termination unchanged",
        source_sha256=sources, research_plan_sha256=sha(args.research_plan), started_utc=utc())
    write_json(output / "config.json", config)
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sources: archive.write(ROOT / name, name)
    write_json(output / "manifest.json", dict(source_sha256=sources, initial_sha256=INITIAL_SHA, source_snapshot_sha256=sha(output / "source_snapshot.zip")))
    env, guard = vector(args.variant, 16, 9511), None
    try:
        model = new_model(env, 9511, output)
        guard = v15.LimitedBasisGuard(model)
        warm_trace, warm_proof = ObjectiveTrace(), v15.frozen.CurriculumProof(output, "critic")
        model.set_logger(configure(str(output / "warmup"), ["csv"]))
        model.learn(131072, reset_num_timesteps=True, callback=[v15.frozen.Heartbeat(output, "critic"), warm_proof, warm_trace], progress_bar=False)
        warm = objective_report(guard)
        assert warm["actual_optimizer_steps"] == dict(critic=1024, ppo=0) and warm["Adam_parameter_states"] == 6
        assert not warm["changed_actor_tensors"]
        model.save(output / "critic_warmed_model"); warm_trace.save(output / "warm_trace.npz", BASIS_KIND)
        warm_trace.rows.clear(); write_json(output / "warmup.json", warm)
        v15.start_ppo(model, guard)
        model.set_logger(configure(str(output), ["csv"]))
        selection = v15.RouteBasisSelection(output, initial_baseline, BASIS_KIND, model._n_updates)
        ppo_proof = v15.frozen.CurriculumProof(output, "ppo")
        model.learn(262144, reset_num_timesteps=True, callback=[selection, ppo_proof], progress_bar=False)
        selection.evaluate("final")
        report = objective_report(guard)
        assert_hyperparameters(model)
        assert report["actual_optimizer_steps"] == dict(critic=1024, ppo=2048) and report["Adam_parameter_states"] == 7
        assert report["changed_actor_tensors"] == [v15.ACTOR_W0] and model.bc_kl_gradient_steps == model.ppo_gradient_steps == 2048
        assert len(selection.records) == 3 and model.num_timesteps == 262144
        assert ppo_proof.counts["completed_episodes"] and ppo_proof.counts["hard_tailwind_triggers"]
        assert source_hashes() == sources and sha(INITIAL) == INITIAL_SHA
        selected = read(output / "selection.json")
        files = ("critic_warmed_model.zip", "warm_trace.npz", "best_model.zip", "best_policy.npz", "checkpoint_131072.zip", "checkpoint_131072.npz", "checkpoint_262144.zip", "checkpoint_262144.npz", "final.zip", "final.npz")
        hashes = {name: sha(output / name) for name in files}
        write_json(output / "model_manifest.json", dict(initial_sha256=INITIAL_SHA, prior_sha256=INITIAL_SHA, output_sha256=hashes))
        write_json(output / "guard_summary.json", report)
        write_json(output / "complete.json", dict(status="complete", schema=SCHEMA, variant=args.variant,
            basis_kind=BASIS_KIND, critic_transitions=131072, ppo_transitions=262144, optimizer=report, warmup=warm,
            output_model_sha256=hashes, source_sha256=sources, source_unchanged=True, candidate_records=selection.records,
            selected_policy_sha256=sha(output / "best_policy.npz"), selected_metrics=selected["aggregate"],
            fixed_development_guards_pass=selected["fixed_development_guards_pass"], diagnostic_candidate=not selected["fixed_development_guards_pass"],
            all_candidates_failed_development_guards=not any(row["guards_pass"] for row in selection.records),
            actual_curriculum_counts=dict(critic=warm_proof.counts, ppo=ppo_proof.counts),
            reused_baseline_new_episodes=0, actual_Gazebo80_acceptance=False, automatic_extension_authorized=False,
            wall_seconds=time.monotonic()-started, completed_utc=utc()))
        phase(output, "complete", environment_steps=262144)
    except Exception as exc:
        write_json(output / "error.json", dict(status="error", error=repr(exc), traceback=traceback.format_exc(), utc=utc()))
        raise
    finally:
        if guard is not None: guard.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ARMS)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--retention-reference", type=Path, default=v15.frozen.RETENTION_REFERENCE)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9611); parser.add_argument("--warmup-seed", type=int, default=9511)
    parser.add_argument("--steps", type=int, default=262144); parser.add_argument("--warmup-steps", type=int, default=131072)
    parser.add_argument("--eval-every", type=int, default=131072); parser.add_argument("--research-plan", type=Path)
    parser.add_argument("--static-check", action="store_true"); parser.add_argument("--verify", action="store_true")
    parser.add_argument("--physical-plan", type=Path, default=ROOT / "docs/joint_metric_cost_v16/semantic_plan.DRAFT.json")
    parser.add_argument("--static-receipt", type=Path, default=ROOT / "audit/joint_metric_cost_static_v16_root01.json")
    parser.add_argument("--semantic-receipt", type=Path, default=ROOT / "audit/joint_metric_cost_semantics_v16_root01.json")
    args = parser.parse_args()
    for name in ("initial", "retention_reference", "baseline", "output", "research_plan", "physical_plan", "static_receipt", "semantic_receipt"):
        value = getattr(args, name)
        if value is not None: setattr(args, name, (value if value.is_absolute() else ROOT / value).resolve())
    if args.output.exists(): parser.error("Fresh output required")
    if args.static_check and args.verify: parser.error("One preparation mode required")
    if (args.steps, args.warmup_steps, args.eval_every, args.seed, args.warmup_seed) != (262144, 131072, 131072, 9611, 9511): parser.error("Only fixed pilot budgets/seeds supported")
    assert args.initial == INITIAL.resolve() and sha(INITIAL) == INITIAL_SHA
    assert args.baseline == BASELINE.resolve()
    assert args.retention_reference == v15.frozen.RETENTION_REFERENCE.resolve() and sha(args.retention_reference) == v15.frozen.RETENTION_SHA256
    torch.set_num_threads(1); sources = source_hashes()
    if args.static_check: static_check(args, sources); return
    if args.verify: verify_pair(args, sources); return
    if args.variant is None or args.output.name != args.variant: parser.error("Canonical arm/output required")
    for flag in ("--initial", "--retention-reference", "--baseline"):
        if flag not in sys.argv: parser.error("Explicit training binding required: " + flag)
    require_bound_preflight(args); run_arm(args, sources)


if __name__ == "__main__":
    main()
