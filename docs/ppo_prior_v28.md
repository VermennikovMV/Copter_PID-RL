# v28: влияние source-prior на маршрут и висение

15 ноября 2026. **Фактические native/PyBullet результаты; это не Gazebo и не принятая модель.**
Сравниваются только FINAL, без изменения выбора checkpoints или допусков.

| Модель | finite RMSE, м | peak, м | время, с | hover RMSE, м | matched5 | hard24 | FULL research |
|---|---:|---:|---:|---:|---:|---|---|
| F0 · без нового PPO | 0.168152 | 0.553270 | 9.60 | 0.047875 | 2/5 | PASS | — |
| κ = 0.10 · FINAL v24 | 0.163175 | 0.519652 | 9.64 | 0.047926 | 2/5 | PASS | FAIL |
| κ = 0.01 · FINAL v28 | 0.159794 | 0.417969 | 8.96 | 0.050333 | 3/5 | PASS | FAIL |
| κ = 0 · FINAL v28 | 0.142773 | 0.354720 | 8.94 | 0.056150 | 5/5 | FAIL | FAIL |

Три маршрутных числа относятся к **full_initial + force_strong84001**, один просмотренный development case.
Пределы этого matched-profile: finite≤0.150 м, peak≤0.500 м, время≤9.30 с; пройти должны все13 guards.
Hover — среднее RMSE четырёх native env seeds84001–84004, предел **0.0506976846506 м**.
Matched5 — число профилей, прошедших все13 проверок, а не число отдельных пройденных метрик.
Hard24 — прежние абсолютные требования original24; FULL research дополнительно включает неизменные научные admission rules.
F0 — исходная conditioning baseline, не новый обученный кандидат; прочерк не означает PASS.

**Причинно сопоставимые обученные FINAL: κ=.10 → κ=0.**
При одинаковом бюджете finite RMSE выбранного маршрута уменьшился на **12.50%**, а hover RMSE вырос на **17.16%**.
Отдельное улучшение относительно F0 (.168152→.142773 м) не следует целиком приписывать κ:
F0 не проходил эти393216 дополнительных взаимодействий.

Каждая из трёх обученных моделей получила131072 critic warm +262144 PPO переходов и1024+2048 Adam updates.
κ=.10 — архивный v24 control; κ=.01 и0 — две новые v28 тренировки, без дополнительного времени обучения на модель.
Общие initializer/source68, reward, curriculum/reset/force distribution, LR, frozen σ и seed111.
Независимый аудит подтверждает одинаковые warm weights/Adam/physical traces и первый PPO buffer.
После расхождения параметров последующие on-policy траектории закономерно различаются: весь датасет не называется побитово одинаковым.
Это один learning seed и просмотренные development условия, не независимая репликация несколькими training seeds.

Source-prior добавляет `κ·mean_batch KL(current Normal || frozen source Normal)` к PPO loss.
Уменьшение κ показало компромисс адаптации маршрута и сохранения висения.
Доказательства доминирования KL-градиента или гарантии безопасности из этого опыта не следуют.
κ=0 FINAL прошёл все5 matched профилей, но нарушил old24 hover cap; κ=.01 FINAL сохранил hard24, но не все matched profiles.
**Все три FULL research admission остаются FAIL; новых принятых результатов Gazebo нет.**

PPO указан в ВКР и уже поддержан воспроизводимым стендом непрерывных тяги/body rates.
Его on-policy clipped updates описаны в [Schulman et al.,2017](https://arxiv.org/abs/1707.06347).
SAC с replay/entropy и TD3 с twin-Q/delayed updates остаются альтернативами:
[Haarnoja et al.,2018](https://proceedings.mlr.press/v80/haarnoja18b.html),
[Fujimoto et al.,2018](https://proceedings.mlr.press/v80/fujimoto18a.html).
**SAC/TD3 здесь не обучались; превосходство PPO над ними не установлено.**

Основания: independent metrics96480b6…, controlled ledgerc65f1b9…, whole NN/integrity8b9936e… .
Полные SHA и policy SHA — `reports/joint_ppo_prior_results_v28/sources.json` и `comparison.csv`.
Создание этого слайда не выполняло NN, env, physics или SGD.
