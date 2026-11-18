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

# сравнение в PyBullet: ArduPilot, компенсатор, end-to-end
python -m training.evaluate --estimator models/estimator.npz --policy models/e2e_policy.npz \
    --seeds 8 --out runs/pybullet.json

# обучение
python -m training.estimator collect --out runs/estimator_data.npz
python -m training.estimator train runs/estimator_data.npz runs/estimator.npz
python -m training.ppo --out runs/e2e --steps 30000000


# Gazebo + SITL (нужны Gazebo Harmonic и собранный ardupilot-ROSMeetUp)
python gazebo/evaluate_ardupilot.py --controller stock   --out runs/gazebo/stock
python gazebo/evaluate_ardupilot.py --controller network --out runs/gazebo/network
python gazebo/evaluate_e2e.py --policy models/e2e_policy.npz --out runs/gazebo/e2e
python gazebo/compare.py stock network
```

Установка стенда: [docs/setup.md](docs/setup.md).

## Ограничения

- Модели обучены для квадрокоптера класса Iris (рама ~550 мм, винты 10″, ~1,75 кг) с
  рандомизацией вокруг него; для другого класса нужно повторить обучение.
- Компенсатор распознаёт помехи, заложенные в обучение (горизонтальные силы, изменение
  массы и тяги моторов). Другие помехи он не компенсирует; безопасность при этом
  обеспечивают ограничение поправки и штатный каскад.
- Проверка — в симуляторах (PyBullet → Gazebo + SITL); на реальном аппарате не проверялось.

Лицензия: [LICENSE_STATUS.md](LICENSE_STATUS.md), сторонние компоненты:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
