# Политики, контракты и доказательства

| Версия | Идентичность | Измеренная область |
|---|---|---|
| F0/v20 hybrid | `f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461` | Предыдущая Gazebo v7:8/8 primary, tail finite/corner FAIL, n=1 |
| Source68 / v16 | `68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5` | Отдельная80D диагностика; F0 сохраняет её6 learned arrays побайтно |
| Path77 / e0aa | `e0aa635603d5ec299cd526347891ef1b090e2f8ebd3879d5aa34f207a41fa781` | Предыдущий baseline; собственная7/8 Gazebo-карта и CPU rollout |
| Direct / historical |3 training ×3 environment seeds,108 Gazebo runs |4 motor actions через измеренный PWM bridge; не F0 и не v29 |
| v28 |2 реальные PPO-тренировки +архивный v24 control; exact SHA в CSV | Native route/hover tradeoff; все FULL research admissions FAIL |
| v29 bundle | `c3fc1c9c799f21200c98fc5995dba9908b1b13a517eb78bd60759e91eb8818a4` |Native29 absolute PASS; новая Gazebo v8:8/8 primary, tail2 FAIL, scientific FAIL |

F0 contract — `measured-memory-route-integral-v5.0`:80 измеряемых входов,
4 выхода (тяга/body rates). Source68 NPZ включён по реальному относительному
пути `v2/artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/best_policy.npz`.
Loader проверяет настоящие байты; подмена `sha()` и ослабление guards не используются.
F0 создан conditioning integral22:25 на собственном valid-leg history без нового
SGD; нижний rate PID ArduCopter сохраняется.

V29 — сборка F0-hover и реально обученного v28-route expert. У route expert2048
actual PPO Adam updates; сборка0SGD. Старые successful cells не наследуются.
На каждый отсчёт вычисляется только выбранный expert.

[Path77 original card](docs/model_cards/path77.md) ·
[Diagnostic80 card](models/DIAGNOSTIC80_CARD.md) ·
[Direct implementation](reference_manual/architecture.md) ·
[v28 exact SHA](results/ppo_prior_v28/comparison.csv) ·
[v29 independent metrics](results/latest_diagnostics/joint_phase_expert_metrics_v29_root01.json)

Initial weights уже были обучены: это не learning from scratch. Масса, отказы,
дополнительные задержки/шум и реальный аппарат новой политикой не подтверждены.
Авторская лицензия пока не выбрана.

Переносимый `public-shared-phase-expert-inference-v1.0` — отдельный экспорт для
CPU, `models/phase29_public_inference.json` SHA ba7681e4d260606a4d382e4c94b7f8da918b52aaaedd29e179ce32a68a3cd36d.
Он использует exact F0/route NPZ, source68 и original8b99 proof. Original ZIP62e
не включён и локально не проверяется: его SHA здесь только происхождение.
Это не подмена strict исследовательского/Gazebo wire-контракта. Четыре численных
метода унаследованы без изменений; полный2000-step causal replay совпал точно.

[Включённые Direct/Hybrid111 и SHA остальных моделей](models/historical_model_registry.json).
Веса222/333 перечислены как НЕ включённые. `v2/sim_direct/controller.py` и
`v2/sim_direct/proxy.py` доступны как обычные модули. Direct API выдаёт
нормированные действия/доли, не PWM. [Код и команды](reference_manual/running.md).
