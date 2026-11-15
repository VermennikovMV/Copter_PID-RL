# Диагностическая политика80: v16 metric_cost

`diagnostic80_metric_cost16.npz` — точные байты выбранного checkpoint131072,
SHA`68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5`.
Это **отдельная модель**, не замена опубликованной77D e0aa и не end-to-end.
Нижний rate PID ArduCopter сохраняется.

80 входов: frozen77 измеряемых признаков плюс три производных координаты
`H_current @ (normcap(1.5*(current_target-position),2m/s)-velocity)` при наличии
отрезка из действительно наблюдавшихся целей. Новых датчиков, будущих точек,
истинного ветра или PID-команд на входе нет. Это дополнительный базис уже
имеющейся информации. Масштаб residual1m/s; отдельный hover получает нули.

Обучались только384 новых first-layer weights; старые77 actor columns, bias,
head и log_std зафиксированы. Новые critic columns нулевые. Во всех плечах v16
γ=1 и training deadlines являются finite-task terminals без bootstrap;
в B изменён весь route reward bundle. Эти изменения относительно старого v15
нельзя свести к одному коэффициенту штрафа. Весь бюджет выполнен:131072 critic,
262144 PPO transitions,1024/2048 optimizer steps, три candidate snapshots.
Выбран ранний checkpoint с960 PPO updates по прежнему fixed quality judge.

Native proxy: tail84001 finite RMSE лучше selectedA на6,85%, corner peak —на14,59%.
Nominal0,11м и общая relative eligibility не пройдены. Это диагностический
результат. Контроль A не летал в новой Gazebo серии, поэтому physical causal
эффект одного reward bundle не установлен.

Чтение и чистый inference без Torch/Gazebo:

```bash
python scripts/inspect_diagnostic80.py
```

Для `.npy` входов с последней размерностью80 доступны `--observations` и
`--output runs/<fresh>.npy`. Такой forward pass не является симуляцией или
новой проверкой качества. Базовый `scripts/quickstart.py` продолжает использовать
77D e0aa; файлы и ячейки двух моделей не смешиваются.

Фактическая Gazebo серия завершена:9 новых RL +11 общих наблюдений, n=1.
Целостность PASS;7/8 основных условий,87/90 проверок; nominal и tail84001 FAIL.
Macro15,3037% не отменяет эти провалы. Tail finite лучше прежней77 на6,311%,
но хуже PID на62,795%. Никакого promotion не выполнено.
[Таблица и ограничения](../results/diagnostic80_v16/presentation.png).
