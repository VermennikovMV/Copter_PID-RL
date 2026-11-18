# Copter PID-RL

Нейросети в контуре управления квадрокоптера на ArduPilot: сравнение трёх подходов на
одинаковых испытаниях в Gazebo Harmonic + ArduPilot SITL.

| Подход | Что делает нейросеть | Обучение |
|---|---|---|
| **ArduPilot** (база) | — штатный каскад PID в режиме GUIDED | — |
| **Компенсатор возмущений** | оценивает ветер и изменение массы по поведению дрона; физическая модель переводит оценку в ограниченную поправку наклона и тяги внутри каскада ArduPilot | с учителем (PyBullet) |
| **End-to-end** | сама задаёт команды четырём моторам, без каскада | PPO (PyBullet) |

Прошивка с компенсатором: [VermennikovMV/ardupilot-ROSMeetUp](https://github.com/VermennikovMV/ardupilot-ROSMeetUp)
(ArduPilot 4.7.2 + библиотека `AC_ResidualPolicy`, параметры `RESID_*`).

## Результаты (Gazebo + SITL, штатный ArduPilot 4.7.2)

Одинаковые возмущения во всех сериях: порывы — внешняя сила на корпус по расписанию
(3 Н, три порыва по 3 с с разворотом направления), груз — 0,35 кг (20 % массы) на
отцепляемом звене, сброс на 12 с (висение) или 4 с (трасса). Трасса — квадрат 10×10 м:
две пролётные точки (радиус 0,5 м), остановка на 2 с посередине и финиш. 8 seed на
сценарий с ветром.

| Сценарий | ArduPilot | Компенсатор | End-to-end |
|---|---|---|---|
| Висение, порывы: RMSE / пик | 11,2 / 23,7 см | **3,2 / 10,3 см** (−72 % / −57 %) | 3,4 / 10,6 см (−70 % / −55 %) |
| Порывы + груз: RMSE | 10,5 см | **3,2 см** (−70 %) | 3,4 см (−68 %) |
| Сброс груза: скачок высоты | 6,5 см | 4,5 см (−31 %) | **3,5 см** (−46 %) |
| Трасса: время | 21,4–21,5 с | 21,4 с | **17,2–17,3 с** (−20 %) |
| Трасса с порывами: уход с линии RMS / пик | 14 / 47 см | **12 / 48 см** (−18 % по RMS) | 41 / 80 см |
| Пиковый наклон на трассе | 29° | 31° | 42° |
| Висение без возмущений | 0,5 см | 0,5 см | 1,3 см |
| Потеря управления | 0 | 0 | 0 |

- **Компенсатор** сохраняет траекторию и безопасность ArduPilot (поправка ограничена
  ±15° и ±15 % тяги, все контуры штатные) и в 3 раза лучше держит точку при ветре.
  Скорость трассы задаёт планировщик ArduPilot, компенсатор её не меняет.
- **End-to-end** быстрее на трассе и так же устойчив к ветру, но менее точен на линии,
  наклоняется сильнее и не имеет гарантий каскада.
- Оба переносятся из PyBullet (где обучены) в Gazebo + SITL почти без потерь.

Подробно: [docs/results.md](docs/results.md), метод: [docs/method.md](docs/method.md).

## Структура

```
configs/      конфигурации обучения и оценки, параметры стенда SITL
training/     PyBullet: модель Iris, копия каскада ArduPilot, обёртки, обучение, оценка
  environment.py   среда PyBullet (физика винтов, рандомизация, возмущения, трассы)
  pid.py           копия каскада ArduPilot (позиция → скорость → ускорение → угол → угловая скорость)
  guided_shaper.py формирователь цели GUIDED (AC_PosControl::input_pos_NED_m)
  wpnav.py         навигатор AUTO (AC_WPNav, S-кривые) — для сравнения режимов
  wrappers.py      награда, вход поправки, наблюдения, оценщик и компенсация
  stacks.py        готовые сборки сред: ardupilot / compensator / end_to_end
  estimator.py     сбор данных и обучение оценщика возмущений
  ppo.py           обучение end-to-end (PPO)
  evaluate.py      сравнение на одинаковых сценариях и seed
  tools.py         экспорт модели, изменение числа входов без потери обучения
controllers/  кодировщик наблюдений (общий для Python и прошивки)
experiments/  трасса, правила прохождения точек, расписания возмущений
gazebo/       испытания в Gazebo + SITL, плагин возмущений
simulation/   запуск Gazebo + SITL, модель Iris, параметры стенда
models/       обученные модели
```

## Быстрый старт

```bash
python -m pip install -r requirements.txt

# сравнение в PyBullet на одинаковых сценариях и seed: ArduPilot (копия), компенсатор, end-to-end
python -m training.evaluate --estimator models/estimator.npz --policy models/e2e_policy.npz \
    --seeds 8 --out runs/pybullet.json
```

Gazebo + SITL (нужны Gazebo Harmonic, ardupilot_gazebo и собранный
[ardupilot-ROSMeetUp](https://github.com/VermennikovMV/ardupilot-ROSMeetUp), см. [docs/setup.md](docs/setup.md)):

```bash
python gazebo/evaluate_ardupilot.py --controller stock   --out runs/gazebo/stock     # RESID_ENABLE=0
python gazebo/evaluate_ardupilot.py --controller network --out runs/gazebo/network   # компенсатор в прошивке
python gazebo/evaluate_e2e.py --policy models/e2e_policy.npz --out runs/gazebo/e2e
python gazebo/compare.py runs/gazebo/stock runs/gazebo/network runs/gazebo/e2e
```

Отдельный случай: `--cases hover/gust --seeds 90000`. Окно Gazebo для показа:
`python -m simulation.launch --vehicle drone1 --gui`.

## Воспроизведение обучения

Модели в `models/` получены ровно этими командами (время — на 32 потоках).

**Оценщик возмущений** (~40 мин). Два круга: сначала летает каскад ArduPilot, затем — компенсатор
первого круга, чтобы оценщик учился на том замкнутом контуре, в котором работает.

```bash
python -m training.estimator collect --out runs/estimator/round1.npz
python -m training.estimator train runs/estimator/round1.npz --out runs/estimator/round1_model.npz
python -m training.estimator collect --out runs/estimator/round2.npz --estimator runs/estimator/round1_model.npz \
    --perturbation 0.3 --steps 100000 --seed 8000
python -m training.estimator train runs/estimator/round1.npz runs/estimator/round2.npz --out models/estimator.npz
```

**End-to-end** (~5 ч, 50 млн шагов). Одна команда; этапы обучения заданы в
[configs/e2e_curriculum.json](configs/e2e_curriculum.json): сначала короткие отрезки и обычная
награда (научиться летать), затем длинные трассы и старты в движении, затем награда за скорость.

```bash
python -m training.ppo --out runs/e2e          # контрольные точки: runs/e2e/ckpt_*, итог: runs/e2e/policy.npz
python -m training.tools export-actor runs/e2e/ckpt_050176000 runs/e2e/policy.npz   # любая контрольная точка
```

Награда, входы и рандомизация описаны в [docs/method.md](docs/method.md).

## Свои модели

Все скрипты принимают путь к модели, поэтому новую модель можно проверить теми же командами.

| Модель | Формат `.npz` | Проверка |
|---|---|---|
| Оценщик | `W0, b0, W1, b1, …` (tanh между слоями), `y_mean`, `y_std`; вход 83, выход 4 | `training.evaluate --estimator`; в прошивке — 83 → 128 → 128 → 4 |
| End-to-end | `w0, b0, w1, b1, …` (tanh между слоями, выход ограничен ±1); вход 83, выход 4 мотора | `training.evaluate --policy`, `gazebo/evaluate_e2e.py --policy` |

- Другой размер сети: `training.estimator train … --hidden 256 256`, `training.ppo … --net 256 256`.
- Другие награда, этапы или рандомизация: `training.ppo --reward '{"route_time": 0.6}'`,
  `--curriculum my_stages.json`, `--config my_config.json` (формат — как в `configs/`).
- Продолжить обучение: `training.ppo --out runs/e2e_more --resume runs/e2e/ckpt_050176000 --steps 60000000`.
- Добавить входы без потери обучения: `python -m training.tools widen` (новые входы получают нулевые веса).
- Веса оценщика встраиваются в прошивку генерацией `estimator_weights.h` в
  [ardupilot-ROSMeetUp](https://github.com/VermennikovMV/ardupilot-ROSMeetUp) (`libraries/AC_ResidualPolicy`).

## Ограничения

- Модели обучены для квадрокоптера класса Iris (рама ~550 мм, винты 10″, ~1,75 кг) с
  рандомизацией вокруг него; для другого класса нужно повторить обучение.
- Компенсатор распознаёт помехи, заложенные в обучение (горизонтальные силы, изменение
  массы и тяги моторов). Другие помехи он не компенсирует; безопасность при этом
  обеспечивают ограничение поправки и штатный каскад.
- Проверка — в симуляторах (PyBullet → Gazebo + SITL); на реальном аппарате не проверялось.

## Лицензия

[MIT](LICENSE). Модель Iris в `simulation/models/iris` и внешние компоненты — см.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
