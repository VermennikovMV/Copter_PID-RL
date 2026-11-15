# Запуск трёх дронов в Gazebo

Проверенная конфигурация — Windows с WSL2/WSLg и двумя Ubuntu. Из публикуемого
набора исходников и моделей выполнены номинальная проверка, полный маршрут
и удержание при порывах. [Записи и измерения](../results/shared_world/README.md).
Автоматическая установка обеих ОС с нуля не проверялась.

## Окружение

| Дистрибутив | Назначение | Проверенные версии |
|---|---|---|
| `Ubuntu` | Gazebo, GUI, запись окна | Ubuntu 22.04.5, Python 3.10.12, Gazebo Sim 8.14.0, Transport13 13.5.0 |
| `Ubuntu-24.04` | Три SITL и контроллеры | Ubuntu 24.04.4, Python 3.12.3, NumPy 2.5.4, pymavlink 2.4.50 |

Проверьте имена из PowerShell: `wsl --list --verbose`. Короткие команды ниже
ожидают именно эти имена; `--science-distro` меняет научный дистрибутив.
Клон должен быть доступен обоим по одному Linux-пути. Команды выполняются
в корне клона, если явно не указан переход в другой каталог. Имена пользователей
в код не вшиты; используется домашняя папка default-пользователя каждого WSL.

## Ubuntu 24.04: Python и SITL

```bash
sudo apt-get install python3.12-venv git
python3.12 -m venv "$HOME/vkr_sim/venv-cu128"
"$HOME/vkr_sim/venv-cu128/bin/python" -m pip install -r demo_three/requirements-controller.txt
```

Существующую подходящую среду используйте без пересоздания. Название `venv-cu128`
историческое: Torch и CUDA для этой сцены не нужны. Другой Python задаётся
параметром `--science-python` или переменной `COPTER_SCIENCE_PYTHON`.

Для нового каталога установите зафиксированные исходники ArduPilot:

```bash
mkdir -p "$HOME/vkr_sim"
git clone https://github.com/ArduPilot/ardupilot.git "$HOME/vkr_sim/ardupilot"
cd "$HOME/vkr_sim/ardupilot"
git checkout fb773567896014da007bca58358704d4a0d8516f
git submodule update --init --recursive
Tools/environment_install/install-prereqs-ubuntu.sh -y
./waf configure --board sitl
./waf copter
```

Runtime ожидает `build/sitl/bin/arducopter` и штатные параметры
`Tools/autotest/default_params/copter.parm`, `gazebo-iris.parm`.
Путь по умолчанию — `$HOME/vkr_sim/ardupilot` научного пользователя.

## Ubuntu 22.04: Gazebo и запись

Подключите apt-репозиторий по [официальной инструкции Gazebo Harmonic](https://gazebosim.org/docs/harmonic/install_ubuntu/), затем установите:

```bash
sudo apt-get install build-essential cmake git pkg-config gz-harmonic \
  libgz-sim8-dev libgz-plugin2-dev python3-gz-transport13 python3-gz-msgs10 \
  ffmpeg xterm xdotool x11-utils tcpdump python3-pil fontconfig fonts-dejavu-core
gz sim --version
/usr/bin/python3 -c "import gz.transport13; import gz.msgs10; import PIL"
```

Нужен работающий GUI WSLg. Pillow и DejaVu Sans рисуют подписи на объектах сцены;
видео захватывает настоящее X11-окно Gazebo.

```bash
git clone https://github.com/ArduPilot/ardupilot_gazebo.git "$HOME/ardupilot_gazebo"
git -C "$HOME/ardupilot_gazebo" checkout 082a0fe231f6e63bc8d1598f1cba461d9e2ea7f5
export GZ_VERSION=harmonic
cmake -S "$HOME/ardupilot_gazebo" -B "$HOME/ardupilot_gazebo/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$HOME/ardupilot_gazebo/build" -j2
```

Этот default-путь нужен для `models/iris_with_standoffs`, `iris_with_ardupilot`
и `build/libArduPilotPlugin.so`. Из корня клона соберите включённый плагин силы:

```bash
cmake -S demo_three/force_plugin -B demo_three/force_plugin/build
cmake --build demo_three/force_plugin/build -j2
```

Готовый `libThreeDroneForceSystem.so` не поставляется. Пины исходников сохранены
в [описании стека](../evidence/three_demo_external_pins.json); SHA бинарника при
другом компиляторе может отличаться.

## Общие журналы WSL

Default-пользователи обоих WSL должны иметь одинаковый численный UID; в
проверенной системе оба UID — 1000. Проверьте из PowerShell:

```powershell
wsl -d Ubuntu -- id -u
wsl -d Ubuntu-24.04 -- id -u
```

При разных UID сначала настройте совпадающую учётную запись. В GUI-дистрибутиве
создайте только конкретную новую папку или проверьте существующую:

```bash
if [ -e /mnt/wsl/copter_three ] || [ -L /mnt/wsl/copter_three ]; then
  test -d /mnt/wsl/copter_three && test ! -L /mnt/wsl/copter_three && \
    test "$(stat -c %u /mnt/wsl/copter_three)" = "$(id -u)" || exit 1
else
  sudo install -d -m 0750 -o "$(id -u)" -g "$(id -g)" -- /mnt/wsl/copter_three
fi
```

Не меняйте владельца чужой существующей папки. Этот общий Linux tmpfs нужен
для живых журналов 50 Гц. После завершения процессов точная копия переносится
в output. Не удаляйте живые журналы до подтверждения копирования.

## Запустить

Из корня клона в Ubuntu 22.04:

```bash
python3 -m demo_three.present route --wp-radius 0.40
python3 -m demo_three.present route --wp-radius 0.25
python3 -m demo_three.present hover
```

`--wp-radius` задаёт общий радиус достижения точки в метрах для PID, Hybrid и
End-to-end. По умолчанию — 0,25 м; допустимо больше 0 и не больше 1 м. В каждом
случае нужно непрерывно оставаться внутри 0,5 с. Это параметр нашего менеджера
миссии, не `WPNAV_RADIUS` ArduPilot: цели передаются в GUIDED. Допуск восстановления
при удержании остаётся 0,25 м. [Опыт с двумя радиусами](../results/waypoint_radius/README.md)
сохранён отдельно от исходного исследовательского протокола.

Маршрут использует постоянную сильную нагрузку в заданном временном окне,
удержание — порывы. Сначала три
дрона взлетают и стабилизируются, затем начинается общий измерительный участок.
Файлы сохраняются в новом `runs/gazebo_<mission>_<wind>_<time>/`.
После готовности дронов предусмотрена пауза 10 с реального времени для загрузки
графики; физика и удержание позиции продолжаются. Для своей записи добавьте
`--manual-start`: убедитесь в окне Gazebo, что все три дрона уже зависли,
и выполните в другом терминале напечатанную команду `touch .../START_MEASUREMENT`.
Так были начаты опубликованные записи. Затем стенд повторно проверяет общий
допуск старта; задержка окна не подменяет физические измерения.
`--live-root` меняет общий каталог журналов; default — `/mnt/wsl/copter_three`.
Свой Python выбирается через `--science-python`: передавайте абсолютный путь,
существующий именно у научного пользователя Ubuntu 24.04.

TCP 5760/5770/5780 и UDP 9002/9012/9022/9032 должны быть свободны. Перехват
localhost использует отдельный tcpdump-helper с правами root в GUI-Ubuntu.
Приложение останавливает только собственные процессы. Закрытый training-архив
не нужен: точные NPZ и явные публичные проекции PID/calibration включены.
Совместная демонстрация не добавляет реплик к исторической
[карте применимости](../reference_manual/metrics.md).

## Проверить полёт и сделать GIF

Подставьте имя созданного каталога вместо `<session>`. Аудит выполняется в Ubuntu 24.04 из того же корня клона:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$HOME/vkr_sim/venv-cu128/bin/python" \
  -B demo_three/audit.py --run runs/<session>/flight \
  --output runs/<session>/integrity_audit.json
```

После проверки создайте GIF в Ubuntu 22.04:

```bash
python3 -m demo_three.media runs/<session> --output runs/<session>/media
```

Укажите новое имя output: исходные записи не перезаписываются. GIF и MP4 получаются из кадров окна; исходное видео, телеметрия и контрольные суммы остаются в каталоге опыта. Период возмущения задан в `flight/plan.json`, фактически приложенная сила — в трёх `force_physics_*.jsonl`. Причины отказов и незавершённые маршруты сохраняются.

