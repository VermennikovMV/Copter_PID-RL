"""Evidence/plan regression checks; fixtures are not simulation results."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "research_preflight", PROJECT / "scripts/joint_research_preflight_v1.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        def put(path, data):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return {"path": path, "sha256": gate.sha(target)}
        contract = put(gate.CONTRACT, (PROJECT / gate.CONTRACT).read_bytes())
        native = json.loads((PROJECT / gate.CONTRACT).read_text(encoding="utf-8"))["frozen_pid_profile"]
        pid = put(native["path"], (PROJECT / native["path"]).read_bytes())
        script = put("scripts/train_joint_fixture.py", b"# Test fixture; never executed.\n")
        initial = put("fixtures/initial.zip", b"Unit test fixture, not a trained policy")
        receipt = put("fixtures/probe.json", b'{"checks":{"succeeded":true}}')
        source = put("fixtures/source.py", b"# Isolated fixture\n")
        evidence = dict(receipt, pointer="/checks/succeeded", equals=True,
                        interpretation="Unit test fixture validates evidence dereferencing only")
        self.plan = {
            "schema": "joint-research-plan-v1", "status": "READY_FOR_RUN",
            "stage": "long_training", "contract": contract,
            "hypothesis": "Fixture for validating the plan, not a research hypothesis",
            "changed_factors": "No scientific experiment or model is created by this fixture",
            "expected_effect": "Only plan structure and identity checks should succeed",
            "reconsider_if": "A malformed or mismatched plan must be rejected",
            "sources": {role: deepcopy(source) for role in gate.ROLES},
            "metric_alignment": {metric: {key: "Fixture text; no scientific meaning is claimed"
                for key in ("reward", "checkpoint", "evaluation", "known_gap")}
                for metric in gate.METRICS},
            "development_environment_seeds": [84001],
            "evidence": {role: deepcopy(evidence) for role in
                         ("baseline", "diagnosis", "reward_semantics", "short_pilot")},
            "budget": {"ppo_steps": 2097152, "critic_steps": 131072,
                       "eval_every": 1048576, "candidate_count": 3},
            "argv": ["--output", "artifacts/refinement/joint_v2/fixture",
                     "--initial", initial["path"], "--steps", "2097152",
                     "--warmup-steps", "131072", "--eval-every", "1048576"],
        }
        self.plan["sources"].update(driver=script, native_pid=pid, initial_policy=initial)

    def reject(self, fragment):
        with self.assertRaisesRegex(gate.PlanError, fragment):
            gate.validate(self.plan, self.root)

    def test_complete_fixture_checks_without_execution_or_output(self):
        command = gate.validate(self.plan, self.root)
        self.assertEqual(command[2], str(self.root / "scripts/train_joint_fixture.py"))
        self.assertFalse((self.root / "artifacts/refinement/joint_v2/fixture").exists())

    def test_missing_metric_rejected(self):
        del self.plan["metric_alignment"]["route_corner_peak"]
        self.reject("Every required metric")

    def test_placeholder_reward_rejected(self):
        self.plan["metric_alignment"]["route_finite_segment_rmse"]["reward"] = "TODO: fill this later"
        self.reject("Unfilled explanation")

    def test_changed_contract_rejected(self):
        (self.root / gate.CONTRACT).write_bytes(b"{}")
        self.reject("SHA mismatch")

    def test_changed_source_rejected(self):
        (self.root / "fixtures/source.py").write_bytes(b"changed")
        self.reject("SHA mismatch")

    def test_reserved_seed_rejected(self):
        self.plan["development_environment_seeds"].append(93001)
        self.reject("Reserved final seed")

    def test_budget_change_rejected(self):
        self.plan["budget"]["ppo_steps"] += 1048576
        self.reject("Budget/command mismatch")

    def test_evidence_false_rejected_even_with_updated_sha(self):
        receipt = self.root / "fixtures/probe.json"
        receipt.write_bytes(b'{"checks":{"succeeded":false}}')
        for binding in self.plan["evidence"].values():
            binding["sha256"] = gate.sha(receipt)
        self.reject("Evidence assertion failed")

    def test_missing_evidence_field_rejected(self):
        self.plan["evidence"]["diagnosis"]["pointer"] = "/checks/not_present"
        self.reject("Missing evidence field")

    def test_no_short_pilot_rejected_for_long_training(self):
        del self.plan["evidence"]["short_pilot"]
        self.reject("Missing prerequisite evidence")

    def test_existing_output_preserved(self):
        output = self.root / "artifacts/refinement/joint_v2/fixture"
        output.mkdir(parents=True)
        self.reject("Output already exists")
        self.assertTrue(output.exists())

    def test_altered_pid_binding_rejected(self):
        self.plan["sources"]["native_pid"] = deepcopy(self.plan["sources"]["reward"])
        self.reject("Competent PID")

    def test_command_initial_policy_mismatch_rejected(self):
        index = self.plan["argv"].index("--initial") + 1
        self.plan["argv"][index] = "fixtures/different.zip"
        self.reject("Command initial policy")

    def test_path_escape_rejected(self):
        self.plan["sources"]["reward"]["path"] = "../outside.py"
        self.reject("Path escapes project")

    def test_draft_template_rejected(self):
        draft = json.loads((PROJECT / "docs/joint_research_plan_v1.template.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(gate.PlanError, "still a draft"):
            gate.validate(draft, self.root)


if __name__ == "__main__":
    unittest.main()
