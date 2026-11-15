# От отдельных улучшений к проверяемой применимости

| Версия | Изменение | Фактический результат |
|---|---|---|
| v25 | Те же source68/reset/PPO, добавлен global-peak penalty | Native pool FAIL.3 новых Gazebo PID/F0/trained полёта: trained finite−2,69% и peak−15,01% против F0, corner+6,11%; обе NN finite/corner FAIL против fresh PID. Один case, не новая полная карта |
| v26 | Frozen base, обучаемые velocity/force residual heads | Все6 native admissions FAIL.4 свежих Gazebo диагностики: PID/F0/A/B; все3 NN finite/corner FAIL. Force feature не исправил перенос |
| v27 | Gain landscape и gradient diagnosis |10 native episodes,0SGD; coarse gains не исправили strong case. Offline gradients не доказали critic bug |
| v28 | Только source-prior κ=.10/.01/0 при общих initializer/reset/reward/seeds/budget | Реальное PPO улучшает маршрут ценой hover. Все6 новых FULL research/engineering admissions FAIL. κ0 FINAL matched5/5, но strong-hover cap FAIL; κ.01 FINAL hard24 PASS, не все matched |
| v29 | F0-hover +обученный v28-route expert; causal observed-target gate,0 новых SGD |29 новых native episodes,52000 controls,12 hover exact F0, все29 absolute engineering cases PASS. Scientific FAIL: nominal gain2,957% меньше3%, tail corner+4,497%.Gazebo: 9 новых NN + 11 общих полётов, основные 8/8 PASS, 88/90 проверок; tail finite/corner FAIL |

V29 не выбирает expert по истинному ветру, future route или environment ID.
На отсчёт вызывается одна выбранная NN. Route expert реально обучен PPO;
последующая сборка — отдельный фактор, не дополнительное обучение.
Native full-wind finite0,142827м/peak0,355415м/T8,94с; pulse0,145515м/0,37969м/
8,98с. Эти числа **не Gazebo**.

V28 FINAL κ=.10→0 при одинаковом training budget улучшил finite одного
full-initial wind route на12,50%, ухудшил strong-hover RMSE на17,16%.
Shared warm weights/Adam/physical prefix и первый PPO buffer проверены;
поздние on-policy данные после расхождения параметров различаются. Это
сравнение обученных FINAL, не приписывание всего выигрыша над F0 одному κ.

V24 control и каждая v28 модель получили131072 critic +262144 PPO interactions
и1024 +2048 Adam. Бюджет на модель не увеличивался. Один learning seed и viewed
development conditions не заменяют независимое подтверждение. SAC/TD3 не обучались.

[v28 разбор](ppo_prior_v28.md) ·
[v28 metrics](../results/latest_diagnostics/joint_prior_ablation_metrics_v28_root01.json) ·
[v28 whole NN](../results/latest_diagnostics/review_joint_prior_ablation_v28_pilot_01.json) ·
[v29 native proof](../results/latest_diagnostics/review_joint_phase_expert_v29_native_01.json) ·
[v29 gates](../results/latest_diagnostics/joint_phase_expert_metrics_v29_root01.json)

Последняя измеренная карта — [v29/Gazebo v8](../results/phase29_v8/presentation.png).
Все 9 новых полётов прошли аудит физики/силы/NN/метрик; 11 общих контрольных
наблюдений использованы явно. Средний выигрыш исходных ветровых условий
16,904031%, но tail84001: finite 0,138578 > 0,102738 м и средний пик на поворотах
0,189760 > 0,172821 м — FAIL. Относительные native FAIL также сохраняются.
Предыдущая F0/v7 карта не изменена. Точечные диагностики v25/v26 не считаются
полными матрицами. Резервные 114 финальных полётов не выполнены.
