# v20: conditioning with unchanged neural weights

PREPARE ONLY. Completed v21p1, v22 and v23 outcomes are in [causal history](experiment_history_v20_v24.md). V21p1 and v23 completed with admission FAIL; v24 main and final package decisions remain pending.

Exact v20 NPZ f0e87341 uses all six source68 weight/bias arrays byte exact. No
new gradient or optimizer update produced this controller. On a valid leg from
its own actually observed target history, the effective actor integral columns
22:25 become zero; other inputs and the hidden raw encoder update are preserved.
The contract is measured-memory-route-integral-v5.0. No target preview or true
wind is supplied. The source68 checkpoint remains at its original relative path
inside v2, so the unchanged policy loader retains its exact SHA guard.

Actual Gazebo development has nine new NN flights and eleven shared controls,
twenty observations, one viewed environment per condition. Primary8/8 and88/90
fixed checks pass; six-wind macro17.49797% passes. Separate tail84001 fails:
finite error0.166330m and mean path-corner peak0.207619m, worse than source68
(0.150973m /0.192377m). It is unpromoted and not independent final confirmation.
The same environment seed does not by itself match native/Gazebo initial yaw,
position or disturbance direction. Do not infer transfer causality from its name.

The earlier v17/v18/v19 true PPO pilots and all their negative outcomes are
preserved under results/diagnostics_latest. PPO was the thesis's selected method;
the included algorithm-choice document is not a PPO/SAC/TD3 empirical benchmark.

Portable read-only inspection: python scripts/inspect_memory20.py.
Training is PyBullet; measured applicability validation is Gazebo/ArduPilot.
The v8 guarded training capsule is historical and does not reproduce v21.
Full Gazebo recording launchers require the original external WSL stack.
