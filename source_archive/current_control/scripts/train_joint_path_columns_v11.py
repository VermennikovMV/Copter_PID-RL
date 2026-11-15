"""Bounded W8 PPO with exactly384 trainable actor Path77 first-layer weights.

The full critic learns. The actor base74 columns, every bias/later layer/head,
and log_std stay bit-exact. Frozen v8 CONTROL environment/evaluation/selection
are called normally, without monkeypatching or executing source text.
"""
from __future__ import annotations
import argparse
import copy
import datetime
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
from scripts import train_joint_corner_v8 as frozen
from scripts.bc_trust_region_probe_v1 import BCTrustPPO, actor_name, sha
from quadrl_corner_v8.env import JointTaskEnv
from quadrl_path_v7.env import PathNativeEnv
from quadrl_path_v7.policy import EncoderConfigPath77, NumpyPolicyPath77, export_policy_path77

SCHEMA = "joint-path-columns-ppo-v11.0"
VARIANT = "path_columns"
W0 = "mlp_extractor.policy_net.0.weight"
BASELINE = ROOT / "artifacts/refinement/joint_v2/corner_v8_baseline111/baseline.json"
BASELINE_SHA = "414f1ceeaf3ef9c777143300cd57ed979432d3d79fb61a4a8dfd4bf3e5fdb756"
CONTROL = ROOT / "artifacts/refinement/joint_v2/corner_pilot111/w8_control"
V9_AUDIT = ROOT / "audit/joint_control_budget_v9_root01.json"
V9_SHA = "9b4ad1c3f1f003f2c99872a66c110fc7429f8d9508a4c0e783de743f7451b59e"
V8_AUDIT = ROOT / "audit/joint_corner_pilot_v8_root01.json"
V8_SHA = "ba68e8c3669269eedaf98cdf4918e664f9402f8bc0f454bc5a978570853bc61c"
FROZEN_MANIFEST = ROOT / "docs/joint_corner_training_v8/freeze.json"
BUDGET = dict(ppo_steps=262144, critic_steps=131072, eval_every=131072, candidate_count=3)
PROBE_BUDGET = dict(envs=2, n_steps=128, batch_size=64, n_epochs=1,
                    critic_transitions=256, ppo_transitions=256,
                    new_full_hover_episodes=3, new_full_hover_steps=4500,
                    reused_matching_baseline_hover_episodes=3)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_hashes():
    """Preserve all63 frozen source/evidence bindings and add this new driver."""
    old = load_json(FROZEN_MANIFEST)["common_runtime_source_sha256"]
    for name, digest in old.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen source/evidence changed: " + name)
    return {**old, str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}


def prerequisites(args):
    frozen.physics_evidence()
    sources = source_hashes()
    if args.initial != frozen.INITIAL.resolve() or sha(args.initial) != frozen.INITIAL_SHA256:
        raise ValueError("Exact learned v7 Path77 initial ZIP required")
    if args.retention_reference != frozen.RETENTION_REFERENCE.resolve() or sha(args.retention_reference) != frozen.RETENTION_SHA256:
        raise ValueError("Frozen original Gamma77 retention reference required")
    if args.baseline != BASELINE.resolve() or sha(args.baseline) != BASELINE_SHA:
        raise ValueError("Explicitly reused actual initial24 baseline414 required")
    baseline, unused = frozen.baseline_receipt(args.initial, ROOT / "unused_v11_baseline", args.baseline, variant="path")
    for episode in baseline["episodes"]:
        if episode["mission"] == "route":
            assert sha(Path(episode["trace_path"]).parent / "geometry.json") == episode["geometry_sha256"]
    if sha(V9_AUDIT) != V9_SHA or sha(V8_AUDIT) != V8_SHA:
        raise ValueError("Frozen prior failure/matched-control audit changed")
    v8_audit = load_json(V8_AUDIT)
    assert load_json(V9_AUDIT)["status"] == v8_audit["status"] == "PASS"
    for name, digest in v8_audit["input_sha256"].items():
        if (ROOT / name).parent == CONTROL:
            assert sha(ROOT / name) == digest, "Reused CONTROL evidence changed: " + name
    assert sha(CONTROL / "model_manifest.json") == "5ee0c8802da8c554f390ac8ff5093c8fa5681bec618d1d545cec71350f070002"
    for name, digest in load_json(CONTROL / "model_manifest.json")["output_sha256"].items():
        assert sha(CONTROL / name) == digest
    control = load_json(CONTROL / "config.json")
    assert control["initial_sha256"] == frozen.INITIAL_SHA256
    assert (control["ppo_steps"], control["warmup_steps"], control["eval_every"], control["ppo_seed"], control["warmup_seed"]) == (262144, 131072, 131072, 9611, 9511)
    assert control["variant"] == "w8_control" and control["corner_reward"]["enabled"] is False
    return baseline, sources


class ActorColumnsGuard:
    """Mask before joint gradient clipping and audit every actual Adam step.

    Adam retains its original full parameter groups. Only six critic tensors
    receive gradients in warmup; PPO adds one actor tensor. Its frozen74-column
    moments stay exactly zero after the common initial13-to0 state reset.
    """
    def __init__(self, model):
        self.model = model
        self.initial = copy.deepcopy(model.policy.state_dict())
        self.prior = copy.deepcopy(model.bc_prior.state_dict())
        self.parameters = dict(model.policy.named_parameters())
        assert self.parameters[W0].shape == (128, 77)
        assert sum(not actor_name(name) for name in self.parameters) == 6
        assert not model.policy.optimizer.state
        assert all(group.get("weight_decay", 0.) == 0. for group in model.policy.optimizer.param_groups)
        self.counts = dict(critic=0, ppo=0)
        self.max_grad = dict(raw_frozen74=0., allowed_path3=0., masked_frozen74=0.)
        self.stage = None
        self.mask_handle = None
        self.pre = model.policy.optimizer.register_step_pre_hook(self.before_step)
        self.post = model.policy.optimizer.register_step_post_hook(self.after_step)
        self.set_stage("critic")

    def set_stage(self, stage):
        assert stage in ("critic", "ppo")
        self.stage = stage
        for name, parameter in self.parameters.items():
            parameter.grad = None
            parameter.requires_grad_(not actor_name(name) or (stage == "ppo" and name == W0))
        if stage == "ppo" and self.mask_handle is None:
            self.mask_handle = self.parameters[W0].register_hook(self.mask_gradient)

    def mask_gradient(self, gradient):
        assert torch.isfinite(gradient).all()
        self.max_grad["raw_frozen74"] = max(self.max_grad["raw_frozen74"], float(torch.linalg.vector_norm(gradient[:, :74])))
        self.max_grad["allowed_path3"] = max(self.max_grad["allowed_path3"], float(torch.linalg.vector_norm(gradient[:, 74:])))
        masked = gradient.clone()
        masked[:, :74] = 0.
        return masked

    def assert_fixed(self):
        for name, value in self.model.policy.state_dict().items():
            if not actor_name(name):
                continue
            if name == W0 and self.stage == "ppo":
                assert torch.equal(value[:, :74], self.initial[name][:, :74]), "Frozen actor74 columns drifted"
            else:
                assert torch.equal(value, self.initial[name]), "Frozen actor tensor drifted: " + name
        assert all(torch.equal(value, self.prior[name]) for name, value in self.model.bc_prior.state_dict().items())
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in self.model.bc_prior.parameters())

    def before_step(self, optimizer, unused_args, unused_kwargs):
        self.assert_fixed()
        for name, parameter in self.parameters.items():
            if actor_name(name) and (self.stage == "critic" or name != W0):
                assert parameter.grad is None
        if self.stage == "ppo":
            gradient = self.parameters[W0].grad
            assert gradient is not None and torch.count_nonzero(gradient[:, :74]) == 0
            self.max_grad["masked_frozen74"] = max(self.max_grad["masked_frozen74"], float(torch.linalg.vector_norm(gradient[:, :74])))
            state = optimizer.state.get(self.parameters[W0])
            if state:
                assert torch.count_nonzero(state["exp_avg"][:, :74]) == 0
                assert torch.count_nonzero(state["exp_avg_sq"][:, :74]) == 0

    def after_step(self, optimizer, unused_args, unused_kwargs):
        self.counts[self.stage] += 1
        self.assert_fixed()
        expected = {self.parameters[name] for name in self.parameters if not actor_name(name)}
        if self.stage == "ppo":
            expected.add(self.parameters[W0])
            state = optimizer.state[self.parameters[W0]]
            assert torch.count_nonzero(state["exp_avg"][:, :74]) == 0
            assert torch.count_nonzero(state["exp_avg_sq"][:, :74]) == 0
        assert set(optimizer.state) == expected, "Expected exactly6 critic or7 PPO Adam parameter states"

    def report(self):
        self.assert_fixed()
        changed = [name for name, value in self.model.policy.state_dict().items()
                   if actor_name(name) and not torch.equal(value, self.initial[name])]
        return dict(actual_optimizer_steps=self.counts.copy(), Adam_parameter_states=len(self.model.policy.optimizer.state),
            trainable_actor_elements=384 if self.stage == "ppo" else 0,
            actor_requires_grad_tensor_elements=128 * 77 if self.stage == "ppo" else 0,
            critic_fully_trainable=True, actor_gradient_tensor=W0,
            allowed_actor_slice="[:,74:77]", changed_actor_tensors=changed,
            changed_allowed_actor_elements=int(torch.count_nonzero(self.model.policy.state_dict()[W0][:, 74:] - self.initial[W0][:, 74:])),
            frozen74_and_all_other_actor_tensors_bit_exact=True, frozen74_Adam_moments_zero=True,
            weight_decay=0., prior_bit_exact=True, prior_direction="current Normal || frozen SAME learned v7 initial Path77 actor Normal",
            KL_weight=.1, max_gradient_norm=self.max_grad.copy())

    def close(self):
        self.pre.remove(); self.post.remove()
        if self.mask_handle is not None:
            self.mask_handle.remove()


def vector(envs, seed):
    env = VecMonitor(DummyVecEnv([lambda: JointTaskEnv(variant="w8_control", architecture="hybrid", training=True) for unused in range(envs)]))
    env.set_attr("training_progress", 1.)
    env.seed(seed)
    return env


def new_model(initial, env, seed, output, **kwargs):
    model = BCTrustPPO.load(initial, env=env, device="cpu", seed=seed, **kwargs)
    assert model.gamma == .995 and model.gae_lambda == .97 and model.lr_schedule(1.) == 3e-5
    model.anchor_steps = 0
    assert len(model.policy.optimizer.state) == 13
    model.policy.optimizer.state.clear()
    assert len(model.policy.optimizer.state) == 0
    model.configure_bc_prior(initial, 0., output / "on_policy_prior_kl.jsonl")
    model.bc_kl_enabled = False
    return model


def start_ppo(model, guard, seed):
    guard.set_stage("ppo")
    model.bc_kl_weight, model.bc_kl_enabled = .1, True
    model.set_random_seed(seed)
    model._last_obs = None
    model.bc_kl_gradient_steps = model.bc_kl_minibatch_sample_uses = model.ppo_gradient_steps = 0


def export(model, destination):
    export_policy_path77(model, destination, "hybrid", EncoderConfigPath77())


def hover_functional_proof(model, exported):
    initial = BCTrustPPO.load(frozen.INITIAL, device="cpu")
    values = np.random.default_rng(17491).normal(size=(1000, 77)).astype(np.float32)
    values[:, 74:] = 0.
    current_action = model.predict(values, deterministic=True)[0]
    initial_action = initial.predict(values, deterministic=True)[0]
    assert np.array_equal(current_action, initial_action)
    current_numpy = NumpyPolicyPath77(exported).predict(values)
    initial_numpy = NumpyPolicyPath77(frozen.INITIAL_NPZ).predict(values)
    assert np.array_equal(current_numpy, initial_numpy)
    with torch.no_grad():
        hover_kl = float(model.prior_kl(torch.as_tensor(values)))
    assert hover_kl == 0.
    return dict(states=1000, zero3=True, initial_current_torch_actions_bit_exact=True,
                initial_current_numpy_actions_bit_exact=True, current_prior_hover_KL=hover_kl,
                scope="All finite zero3 inputs mathematically invariant; these1000 are numerical fixtures")


def hover_trajectory_proof(policy_path, output, baseline):
    """Three new30s flights versus three explicit reused measured initial flights."""
    policy = NumpyPolicyPath77(policy_path)
    cases = []
    for scenario in frozen.SCENARIOS:
        prior = next(record for record in baseline["episodes"] if (record["scenario"], record["mission"], record["eval_seed"]) == (scenario, "hover", 84001))
        destination = output / ("hover_" + scenario + "_84001")
        env = PathNativeEnv(variant="path", architecture="hybrid", training=False, scenario=scenario, mission="hover")
        try:
            current = frozen._run_episode(env, policy, "hybrid", True, scenario, "hover", 84001, destination)
        finally:
            env.close()
        assert current["steps"] == prior["steps"] == 1500
        assert current["metrics"] == prior["metrics"]
        assert current["mission_events"] == prior["mission_events"] == []
        assert current["hashes"]["trace.csv"] == prior["hashes"]["trace.csv"]
        with np.load(current["actor_io_path"], allow_pickle=False) as now, np.load(prior["actor_io_path"], allow_pickle=False) as old:
            assert now.files == old.files
            assert all(np.array_equal(now[name], old[name]) for name in now.files)
            assert np.count_nonzero(now["observation"][:, 74:]) == 0
        cases.append(dict(scenario=scenario, seed=84001, new_steps=1500, reused_baseline_steps=1500,
            current_trace_sha256=current["hashes"]["trace.csv"], baseline_trace_sha256=prior["hashes"]["trace.csv"],
            all77_arrays_actions_physical_trajectory_and_metrics_bit_exact=True))
    return dict(cases=cases, new_hover_episodes=3, new_hover_steps=4500,
                reused_baseline_hover_episodes=3, full_horizon_s=30.,
                isolated_hover_mission=True, not_final_route_hold=True)


def require_probe_plan(args, sources):
    plan = load_json(args.probe_plan)
    assert plan["schema"] == "joint-path-columns-semantic-probe-v11.0"
    assert plan["status"] == "READY_FOR_BOUNDED_PROBE" and plan["budget"] == PROBE_BUDGET
    assert plan["source_sha256"] == sources
    for field, path in (("initial", args.initial), ("baseline", args.baseline), ("retention", args.retention_reference), ("diagnosis", V9_AUDIT)):
        assert plan[field] == dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
    assert plan["output"] == str(args.output.relative_to(ROOT))
    assert plan["semantic_receipt"] == str(args.semantic_receipt.relative_to(ROOT))
    return plan


def exact_nested(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_nested(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(exact_nested(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def verify(args, baseline, sources):
    plan = require_probe_plan(args, sources)
    if args.semantic_receipt.exists():
        raise FileExistsError("Fresh semantic audit path required")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen.write_json(args.output / "prospective_probe_plan.json", plan)
    with zipfile.ZipFile(args.output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT / name, name)
    env = vector(2, 9511)
    guard = None
    try:
        model = new_model(args.initial, env, 9511, args.output, n_steps=128, batch_size=64, n_epochs=1)
        guard = ActorColumnsGuard(model)
        before = copy.deepcopy(model.policy.state_dict())
        model.set_logger(configure(str(args.output / "critic"), ["csv"]))
        model.learn(256, reset_num_timesteps=True, progress_bar=False)
        warm = guard.report()
        assert not frozen.compare_states(model.policy, before, True)
        assert frozen.compare_states(model.policy, before, False)
        assert warm["Adam_parameter_states"] == 6
        model.save(args.output / "probe_critic_warmed_model")
        start_ppo(model, guard, 9611)
        model.set_logger(configure(str(args.output / "ppo"), ["csv"]))
        model.learn(256, reset_num_timesteps=True, progress_bar=False)
        report = guard.report()
        assert report["actual_optimizer_steps"] == dict(critic=4, ppo=4)
        assert report["Adam_parameter_states"] == 7 and report["changed_actor_tensors"] == [W0]
        assert report["max_gradient_norm"]["raw_frozen74"] > 0. and report["max_gradient_norm"]["allowed_path3"] > 0.
        assert report["max_gradient_norm"]["masked_frozen74"] == 0.
        assert model.bc_kl_gradient_steps == model.ppo_gradient_steps == 4
        assert int(np.count_nonzero(model.rollout_buffer.observations[..., 74:])) > 0
        policy = args.output / "probe_policy.npz"
        export(model, policy)
        model.save(args.output / "probe_model")
        values = np.random.default_rng(17491).normal(size=(1000, 77)).astype(np.float32)
        parity = float(np.max(np.abs(model.predict(values, deterministic=True)[0] - NumpyPolicyPath77(policy).predict(values))))
        assert parity < 1e-5
        functional = hover_functional_proof(model, policy)
        env.close()
        hover = hover_trajectory_proof(policy, args.output / "matched_full_hover", baseline)
        # Same inherited class and same actions; no reward/physics/input change.
        a = JointTaskEnv(variant="w8_control", architecture="hybrid", training=True)
        b = JointTaskEnv(variant="w8_control", architecture="hybrid", training=True)
        try:
            a.training_progress = b.training_progress = 1.
            oa, unused = a.reset(seed=9511); ob, unused = b.reset(seed=9511)
            initial = NumpyPolicyPath77(frozen.INITIAL_NPZ)
            lockstep = 0
            for unused in range(256):
                assert np.array_equal(oa, ob)
                action = initial.predict(oa)
                oa, ra, ta, ua, ia = a.step(action)
                ob, rb, tb, ub, ib = b.step(action)
                assert ra == rb and (ta, ua) == (tb, ub) and exact_nested(ia, ib)
                assert np.array_equal(a.state()[0], b.state()[0])
                assert ia["corner_reward_adjustment"] == 0.
                lockstep += 1
                if ta or ua:
                    break
        finally:
            a.close(); b.close()
        assert source_hashes() == sources
        result = dict(schema="joint-path-columns-semantics-v11.0", status="PASS", utc=frozen.utc(),
            prospective_probe_plan_sha256=sha(args.probe_plan), source_sha256=sources,
            actual_training_transitions=dict(critic=256, ppo=256, total=512),
            warmup=warm, constrained_PPO=report, NumPy_Torch_max_error=parity,
            raw_probe_snapshots_sha256={name: sha(args.output / name) for name in
                ("probe_critic_warmed_model.zip", "probe_model.zip", "probe_policy.npz", "source_snapshot.zip")},
            hover_functional= functional, full_matched_hover=hover,
            same_control_action_lockstep_steps_per_environment=lockstep,
            same77_W8_physics_PRE_reward_POST_target_exact=True,
            reused_initial_baseline=dict(path=str(BASELINE.relative_to(ROOT)), sha256=BASELINE_SHA, reused_episodes=24, new_baseline_episodes=0),
            frozen_original_retention_reference=frozen.retention_reference()["source"],
            frozen_evaluator_full24=True, frozen_selector_original_Gamma_anchor=True,
            reward_changed=False, corner_reward_enabled=False,
            probe_policy_not_used_for_pilot=True, bounded262144_PPO_executed=False,
            Gazebo_executed=False, reserved_seeds_not_opened=[93001, 93002, 93003],
            limitation="Structural hover preservation is not proof that restricted PPO improves route geometry or converges")
        frozen.write_json(args.output / "semantic_receipt.json", result)
        frozen.write_json(args.semantic_receipt, result)
        print(json.dumps(dict(status="PASS", receipt=str(args.semantic_receipt), sha256=sha(args.semantic_receipt))), flush=True)
    except Exception as exc:
        frozen.write_json(args.output / "probe_error.json", dict(status="ERROR", error=repr(exc),
            traceback=traceback.format_exc(), prospective_probe_plan_sha256=sha(args.probe_plan),
            source_sha256=sources, utc=frozen.utc()))
        raise
    finally:
        if guard is not None:
            guard.close()
        env.close()


def require_bound_preflight(args):
    from scripts import joint_research_preflight_v1 as preflight
    if args.research_plan is None:
        raise ValueError("Training requires official preflight --run parent and bound13metric plan")
    plan = load_json(args.research_plan)
    command = preflight.validate(plan)
    assert plan["stage"] == "bounded_pilot" and plan["budget"] == BUDGET
    assert Path(command[2]).resolve() == Path(__file__).resolve() and sys.argv[1:] == plan["argv"]
    assert plan["sources"]["initial_policy"] == dict(path=str(args.initial.relative_to(ROOT)), sha256=frozen.INITIAL_SHA256)
    parent = (Path("/proc") / str(os.getppid()) / "cmdline").read_bytes().decode("utf-8").split("\0")
    supplied = parent[parent.index("--plan") + 1]
    if not any(Path(item).name == "joint_research_preflight_v1.py" for item in parent) or "--run" not in parent or (ROOT / supplied).resolve() != args.research_plan:
        raise ValueError("Direct training blocked: official preflight --run parent with same plan required")
    return plan


def run_arm(args, baseline, sources):
    """Own normal runloop; never calls frozen full-actor-unfreeze run_arm."""
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = dict(schema=SCHEMA, variant=VARIANT, factor="Train only384 actor first-layer last3 causal Path77 columns; full critic trainable",
        initial=str(args.initial.relative_to(ROOT)), initial_sha256=sha(args.initial), prior_sha256=sha(args.initial),
        prior_direction="current Normal || frozen SAME learned v7 initial Path77 actor Normal", KL_weight=.1,
        learning_rate=3e-5, gamma=.995, GAE_lambda=.97, actor_inputs=77, critic_inputs=77,
        reward="Frozen v8 CONTROL W8+arrival10/completion20; effective corner reward0", frozen_control_environment=True,
        budget=BUDGET, ppo_seed=9611, warmup_seed=9511, envs=16,
        optimizer="Same full Adam param groups; reset13-to0,6 critic states warmup,7 during constrained PPO; weight_decay0",
        reused_baseline=dict(path=str(BASELINE.relative_to(ROOT)), sha256=BASELINE_SHA, episodes=24, new_episodes=0),
        retention_reference=frozen.retention_reference()["source"],
        matched_full_network_control=dict(path=str(CONTROL.relative_to(ROOT)), explicitly_reused=True, audit_sha256=V8_SHA),
        diagnosis=dict(path=str(V9_AUDIT.relative_to(ROOT)), sha256=V9_SHA, quality_passed=False),
        research_plan=str(args.research_plan.relative_to(ROOT)), research_plan_sha256=sha(args.research_plan),
        source_sha256=sources, started_utc=frozen.utc(), no_Gazebo_acceptance=True)
    frozen.write_json(output / "config.json", config)
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT / name, name)
    env, guard = vector(16, 9511), None
    try:
        model = new_model(args.initial, env, 9511, output)
        assert model.n_steps * 16 == 8192
        guard = ActorColumnsGuard(model)
        model.set_logger(configure(str(output / "warmup"), ["csv"]))
        warm_proof = frozen.CurriculumProof(output, "critic")
        model.learn(131072, reset_num_timesteps=True, callback=[frozen.Heartbeat(output, "critic"), warm_proof], progress_bar=False)
        warm = guard.report()
        assert warm["actual_optimizer_steps"] == dict(critic=1024, ppo=0) and warm["Adam_parameter_states"] == 6
        warm_control = BCTrustPPO.load(CONTROL / "critic_warmed_model.zip", device="cpu")
        assert all(torch.equal(value, warm_control.policy.state_dict()[name]) for name, value in model.policy.state_dict().items())
        warm["explicit_reused_full_network_control_warmup_all13_tensors_bit_exact"] = True
        frozen.write_json(output / "warmup.json", warm)
        model.save(output / "critic_warmed_model")
        start_ppo(model, guard, 9611)
        model.set_logger(configure(str(output), ["csv"]))
        selection = frozen.JointSelection(output, 131072, baseline, model._n_updates, "w8_control")
        ppo_proof = frozen.CurriculumProof(output, "ppo")
        model.learn(262144, reset_num_timesteps=True, callback=[selection, ppo_proof], progress_bar=False)
        selection.evaluate("final")
        report = guard.report()
        assert report["actual_optimizer_steps"] == dict(critic=1024, ppo=2048)
        assert model.num_timesteps == 262144 and model.bc_kl_gradient_steps == model.ppo_gradient_steps == 2048
        assert report["Adam_parameter_states"] == 7 and report["changed_actor_tensors"] == [W0]
        assert len(selection.records) == 3
        parity_input = np.random.default_rng(17491).normal(size=(1000, 77)).astype(np.float32)
        parity = float(np.max(np.abs(model.predict(parity_input, deterministic=True)[0] - NumpyPolicyPath77(output / "final.npz").predict(parity_input))))
        assert parity < 1e-5
        hover = hover_functional_proof(model, output / "final.npz")
        logs = [json.loads(line) for line in (output / "on_policy_prior_kl.jsonl").read_text().splitlines()]
        assert sum(row["optimizer_steps"] for row in logs) == 2048
        assert sum(row["minibatch_sample_uses"] for row in logs) == model.bc_kl_minibatch_sample_uses
        assert source_hashes() == sources and sha(args.baseline) == BASELINE_SHA and sha(args.initial) == frozen.INITIAL_SHA256
        # Compare selected and matched checkpoints with explicitly reused control.
        from scripts.run_joint_corner_pair_v8 import compare_records
        selected = load_json(output / "selection.json")
        old_selected = load_json(CONTROL / "selection.json")
        labels = ("checkpoint_131072", "checkpoint_262144", "final")
        comparisons = {label: compare_records(load_json(CONTROL / (label + "_validation.json")), load_json(output / (label + "_validation.json"))) for label in labels}
        comparisons["selected"] = compare_records(load_json(CONTROL / (old_selected["checkpoint"] + "_validation.json")), load_json(output / (selected["checkpoint"] + "_validation.json")))
        comparison = dict(status="REUSED_FULL_NETWORK_CONTROL_COMPARISON_NOT_ACCEPTANCE", candidate_role=VARIANT,
            source_compare_API_legacy_candidate_label="corner", reward_in_candidate="W8 only, no effectiveK40", control_new_flights=0,
            control_explicitly_reused=str(CONTROL.relative_to(ROOT)), selected_labels=dict(control=old_selected["checkpoint"], path_columns=selected["checkpoint"]), comparisons=comparisons)
        frozen.write_json(output / "control_comparison.json", comparison)
        model_hashes = {name: sha(output / name) for name in (
            "critic_warmed_model.zip", "best_model.zip", "best_policy.npz",
            "checkpoint_131072.zip", "checkpoint_131072.npz",
            "checkpoint_262144.zip", "checkpoint_262144.npz", "final.zip", "final.npz")}
        frozen.write_json(output / "model_manifest.json", dict(initial_sha256=frozen.INITIAL_SHA256,
            prior_sha256=frozen.INITIAL_SHA256, output_sha256=model_hashes))
        complete = dict(status="complete", schema=SCHEMA, variant=VARIANT, critic_transitions=131072, ppo_transitions=262144,
            optimizer=report, warmup=warm, hover_functional=hover, NumPy_Torch_max_error=parity,
            source_sha256=sources, source_unchanged=True, selected_policy_sha256=sha(output / "best_policy.npz"),
            final_policy_sha256=sha(output / "final.npz"), candidate_records=selection.records,
            all_candidates_failed_development_guards=not any(row["guards_pass"] for row in selection.records),
            fixed_development_guards_pass=selected["fixed_development_guards_pass"],
            diagnostic_candidate=not selected["fixed_development_guards_pass"],
            actual_curriculum_counts=dict(critic=warm_proof.counts, ppo=ppo_proof.counts),
            comparison_sha256=sha(output / "control_comparison.json"), reused_baseline_episodes=24, new_baseline_episodes=0,
            output_model_sha256=model_hashes,
            Gazebo_acceptance=False, automatic_long_run_authorized=False, wall_seconds=time.monotonic() - started, completed_utc=frozen.utc())
        frozen.write_json(output / "complete.json", complete)
        frozen.phase(output, "complete", environment_steps=262144)
    except Exception as exc:
        frozen.write_json(output / "error.json", dict(status="error", error=repr(exc), traceback=traceback.format_exc(), utc=frozen.utc()))
        raise
    finally:
        if guard is not None:
            guard.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(VARIANT,), default=VARIANT)
    parser.add_argument("--initial", type=Path, default=frozen.INITIAL)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--retention-reference", type=Path, default=frozen.RETENTION_REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=262144)
    parser.add_argument("--warmup-steps", type=int, default=131072)
    parser.add_argument("--eval-every", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=9611)
    parser.add_argument("--warmup-seed", type=int, default=9511)
    parser.add_argument("--research-plan", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--probe-plan", type=Path, default=ROOT / "docs/joint_path_columns_v11/probe_plan.json")
    parser.add_argument("--semantic-receipt", type=Path, default=ROOT / "audit/joint_path_columns_semantics_v11_root01.json")
    args = parser.parse_args()
    for name in ("initial", "baseline", "retention_reference", "output", "research_plan", "probe_plan", "semantic_receipt"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, (value if value.is_absolute() else ROOT / value).resolve())
    if args.output.exists():
        parser.error("Fresh output required; preserve earlier work")
    if (args.steps, args.warmup_steps, args.eval_every, args.seed, args.warmup_seed) != (262144, 131072, 131072, 9611, 9511):
        parser.error("Only fixed262144 PPO/131072 critic/eval and seeds9611/9511 supported")
    torch.set_num_threads(1)
    baseline, sources = prerequisites(args)
    if args.verify:
        verify(args, baseline, sources)
        return
    for flag in ("--initial", "--baseline", "--retention-reference"):
        if flag not in sys.argv:
            parser.error("Training argv must explicitly bind " + flag)
    if args.output.name != VARIANT:
        parser.error("Training output basename must be path_columns")
    require_bound_preflight(args)
    run_arm(args, baseline, sources)


if __name__ == "__main__":
    main()
