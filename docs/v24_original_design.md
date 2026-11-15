# Matched deployment startup coverage v24

This controlled restart changes the matched-route initial-state bundle only.
Both the reused v23 control A and new C start from source68 ZIP/prior0df1, with
the same zero_route_integral encoder, full27396 mean actor, frozen log_std,
original77-only critic, metric-cost reward, gamma1/true terminals, seeds,
131072 critic warmup at0.0003 and262144 PPO transitions at0.00003. A's completed
training and three29-case evaluations are reused, not independently replicated.
No v23-trained weights initialize C.

The existing per-environment route alternation remains: matched first, original
next, reset at stage seeds9511/9611. Matched actual yaw is already the exact v22
measured Gazebo yaw in both A and C. C additionally sets physical position,
velocity, roll, pitch and BODY rates from that same fixed actor-visible sample,
and sets yaw reference equal to actual yaw, removing initial yaw-reference
perturbation on matched routes. This is one bundled environment intervention;
it cannot identify the separate effect of any component. Original-route resets
and all hover remain unchanged, including the original yaw-error randomization.
Task/force sampling and RNG draws remain unchanged; on-policy episode ends and
realized trigger times can subsequently differ.

Mission('route') is recreated and updated at t=0 at the new position, preserving
the original0.5s arrival rule and correctly starting dwell inside goal0. Auxiliary
corner coverage history is initialized at that same new position. All raw/path/
conditioner history, native queues and sensor RNG are reset before one retained
effective observation. The terminal executed-action audit cache survives VecEnv
automatic reset. Physical p/v/R/body rates, old/new references and errors, source
and actual initial states, dwell state and bundle components are logged. Actor
inputs retain the same80 measured features; neither future route nor force enters
the policy. Hidden AP controller states are not matched.

The actually completed v23 original24 evaluator and five-profile adapter are
reused as unchanged function objects. Their87-episode v23 completion evidence
and earlier control auxiliaries carry their original scope; no new29-case
software portability run is needed. New C runs three29-case candidate evaluations,
87 new episodes/156000 full-horizon controls. Reused A87 plus F0 original24 and
F0 matched5 total116 episodes/208000 controls; combined coverage203/364000.
Original24 means remain separate from five matched profiles. Valid failures stay
visible; no retries or automatic budget extension occur.

Scientific admission and selection are unchanged: all original hard gates,
source77 retention,3% nominal gain against F0, own-tail finite/corner nonincrease,
hover1.10 limits and all five absolute profile gates must pass. Admission comes
first, then the original native objective and stable phase order. Failure remains
failure. Separately, ROOT prospectively records a native engineering screen:
all original24 absolute hard/retention limits and all five absolute profile gates,
without the additional research-progress3% and own-tail comparisons. ROOT may
later choose an engineering-pass candidate for a new DIAGNOSTIC Gazebo campaign
using the unchanged objective/order. This does not accept the model, relabel a
scientific failure, relax any PID/Gazebo threshold or authorize Gazebo automatically.
The producer implements the unchanged scientific selector; separate labels are
computed from frozen full validation records by ROOT.

Static preparation constructs0 environments and performs0 physical/backward/
optimizer steps. Source68 initializer1000-sample parity,7-to0 Adam reset and
prior KL0 are checked through an explicit unchanged-core subreceipt. Final02
adds clear heading-enabled/full-enabled flags to manifests and first-buffer
metadata; preliminary01 is preserved. The new driver is direct official-preflight
only; the unused old orchestration postprocessing branch is not cloned.

ROOT-only semantics use512 training controls: A full-disabled, reproducing old
v23 heading_probe256; C full-enabled, each128 warm+128 PPO and4+4 Adam updates.
A must match all13 policy tensors, entire Adam, every old IO/first-buffer array
and actual gradient row exactly. C physical/critic/GAE can differ. NPZ archives
are materialized once for readback. Extra reset-only fixtures declare2 environments,
36 explicit resets and at most4 constructor resets:0 controls,0 NN,0 SGD. They
verify original/hover parity, matched full state/reference0/dwell, sensor and task
RNG, stage counters and terminal cache. Reset counts and actual control time by
stratum are logged; exact50% global steps/time is not promised.

DRAFT is not permission. ROOT must bind source/plan/authorization, independently
review512 semantics and use `joint_research_preflight_v1.py --plan <BOUND> --run`
for new C. The semantic entrypoint is `train_joint_initial_coverage_v24.py
--verify --physical-plan <READY_PREP> --output
artifacts/refinement/joint_v2/route_initial_semantic_v24_root01`.
