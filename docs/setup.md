# Установка

## Python (обучение и PyBullet)

Python 3.12, Linux или WSL2.

```bash
python -m pip install -r requirements.txt
```

Этого достаточно для `training.*`.

## Gazebo + ArduPilot SITL

1. [Gazebo Harmonic](https://gazebosim.org/docs/harmonic/install_ubuntu/) и
   [ardupilot_gazebo](https://github.com/ArduPilot/ardupilot_gazebo) (плагин `libArduPilotPlugin.so`).
2. Прошивка с компенсатором:

   ```bash
   git clone --recurse-submodules https://github.com/VermennikovMV/ardupilot-ROSMeetUp ~/ardupilot-ROSMeetUp
   cd ~/ardupilot-ROSMeetUp
   ./waf configure --board sitl && ./waf copter
   ```

   Путь задаётся `--ardupilot` или переменной `ARDUPILOT_PATH` (по умолчанию `~/ardupilot-ROSMeetUp`).
3. Плагин возмущений (внешняя сила на корпус по расписанию и отцепка груза):

   ```bash
   cmake -S gazebo/disturbance_plugin -B ~/copter_disturbance_build && cmake --build ~/copter_disturbance_build
   ```

   Путь к библиотеке — переменная `COPTER_DISTURBANCE_PLUGIN`
   (по умолчанию `~/copter_disturbance_build/libCopterDisturbanceSystem.so`).

Стенд в `simulation/`: модель Iris, мир, параметры SITL (`ardupilot.parm`,
`AHRS_EKF_TYPE=10` — состояние из симулятора). Параметры, которые выставляют скрипты
испытаний: `WP_SPD=3`, `WP_RADIUS_M=0.4`, `WP_YAW_BEHAVIOR=0`, `RESID_MODES=16` (GUIDED),
`RESID_THR_HOVR=0.303` (тяга висения стенда).

### WSL2 с двумя дистрибутивами

Проверенная конфигурация: Gazebo в `Ubuntu` (22.04), ArduPilot SITL и Python в `Ubuntu-24.04`.
Скрипты `gazebo/*` запускаются в `Ubuntu-24.04` с `--gazebo-distro Ubuntu` (по умолчанию),
Gazebo поднимается через `wsl.exe`. Для окна Gazebo под WSLg нужен `--software-rendering`
у `python -m simulation.launch`.

## Параметры прошивки

| Параметр | По умолчанию | Назначение |
|---|---|---|
| `RESID_ENABLE` | 0 | включить компенсатор (0 — штатный ArduPilot) |
| `RESID_MODES` | 16 | битовая маска режимов, где поправка разрешена (16 = GUIDED) |
| `RESID_ANG_MAX` | 15 | предел поправки наклона и курса, градусы |
| `RESID_THR_MAX` | 0,15 | предел поправки тяги, доля |
| `RESID_THR_HOVR` | 0 | тяга висения для входа оценщика (0 — `MOT_THST_HOVER`) |
| `RESID_TST_R/P/T` | 0 | постоянная тестовая поправка (крен, тангаж в градусах; тяга) |

Журнал `RESD`: поправка (R, P, Y, T), оценка (AX, AY, AZ, M) и вход тяги (TR).
