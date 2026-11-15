# Model card: measured Path77 FULLCONTROL

Current policy: `v2/artifacts/refinement/joint_v2/corner_pilot111/w8_control/best_policy.npz`.
SHA-256: `e0aa635603d5ec299cd526347891ef1b090e2f8ebd3879d5aa34f207a41fa781`.
The archive is an exact copy of the measured actor, not a renamed older 74D model.
Its topology is 77 → 128 tanh → 128 tanh → 4, clipped normalized actions.
Contract: `measured-memory-finite-leg-v3.0`.

The actor receives measured state/history and a causal finite-leg correction in
current heading, normalized by 0.25 m. The previous leg start is the previous
actually observed target. No future target, phase oracle, true wind, mass or
motor effectiveness is fed to the actor. Hover extras start at zero; a final
observed leg remains during route final hold. Outputs are collective thrust and
FLU body rates; actual MAVLink use requires the measured-body-Rzz thrust adapter.
ArduPilot retains its native inner loops.

Current weights are the selected FINAL of v8 **CONTROL**, with W8 finite-segment
reward and **no effective K40 corner reward**. Both v8 arms received the same77
path inputs and initial7d41 policy/prior. New critic/PPO budgets were131072/262144
environment transitions and1024/2048 optimizer steps; three candidate snapshots.
Training RNG seeds9611/9511 are distinct from lineage label111. The initializer
was already trained: BC → wind/geometry PPO → manual six-column Γ1.15 calibration
→ learned Path77 stage. It is not learning from scratch. Earlier stages and their
causal scopes are in [causal evidence](../causal_evidence.md).

Actual Gazebo development: 8 primary conditions at83301 plus a strong tailwind
route at84001. Primary7/8 conditions pass; nominal finite geometry fails. Tailwind
finite geometry and corner peak fail. Original six-cell wind macro15.2357% passes
the15% criterion in this viewed domain. Operational acceptance and native nominal
finite guard remain FAIL. Reserved93001/93002/93003 were not opened.

The stronger-budget v9 and actor-column-only v11 variants did not meet the
remaining requirement and are not substituted for this policy's measured cells.
Their completed negative receipts are retained. A Soft-KL prior is a mechanism,
not a mathematical retention guarantee for full-network PPO.

CPU quickstart uses a separate native-rate-v3 PyBullet proxy with unchanged
numeric dynamics and a small observation-only adapter. It establishes portable
inference/physical stepping, not the Gazebo claims. Actual raw PCAP/BIN/world logs
are indexed but excluded; without them the included audit code cannot re-create
the physical/wire proof. No real-aircraft, fault, noise, latency, 90° yaw-step,
m/s wind, broad generalization, or PPO-versus-SAC/TD3 superiority claim is made.

Owner redistribution license remains unspecified.

Additional v12-v16 native experiments are separated in docs/experiments.md.
v16 showed a limited tail-case improvement in the native proxy but failed nominal
accuracy and the predeclared overall improvement rule. It does not replace the
published77D actor or inherit its Gazebo cells. Diagnostic Gazebo80 evaluation
is a separate policy/version, not a new confirmation of this model.