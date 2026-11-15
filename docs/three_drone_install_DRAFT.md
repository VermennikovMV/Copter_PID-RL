# Установка общей сцены трёх дронов — DRAFT

Это инструкция для нового публичного `demo_three/`. Текущий исходный стенд
работает в двух WSL-дистрибутивах. Его окружение прочитано и версии сверены,
но полный запуск из нового чистого публичного клона ещё не подтверждён.
Публичные пути runtime уже реализованы; итоговая проверка чистого клона ещё
ожидается. Установка не требует исходных закрытых training-архивов.

## Проверенное окружение

| Часть | Фактически установленная версия |
|---|---|
| GUI/физика | WSL2, Ubuntu 22.04.5, системный Python 3.10.12 |
| Gazebo Harmonic | Gazebo Sim 8.14.0 |
| Python transport | `python3-gz-transport13` 13.5.0, `python3-gz-msgs10` 10.3.2 |
| Плагин Gazebo | `ArduPilot/ardupilot_gazebo`, commit `082a0fe231f6e63bc8d1598f1cba461d9e2ea7f5` |
| ArduPilot SITL | `ArduPilot/ardupilot`, commit `fb773567896014da007bca58358704d4a0d8516f` |
| Контроллеры | WSL2, Ubuntu 24.04.4, Python 3.12.3 |
| Python-пакеты контроллеров | NumPy 2.5.4, pymavlink 2.4.50 |

Оба внешних Git checkout на момент read-only проверки чистые. Коммиты также
сохранены в [исходных manifests](../evidence/external_stack.json). Это pin исходников,
а не обещание одинакового SHA бинарников при другом компиляторе. Новый групповой
плагин силы собирается из включённых C++/CMake; готовый `.so` не поставляется.

Репозиторий должен быть доступен обоим дистрибутивам по одному Linux-пути,
например на общем диске WSL. Имена Windows/WSL-пользователей в настройках
не фиксируются. Windows WSLg должен предоставлять настоящий X11/GUI-сеанс.

## Python контроллера — Ubuntu 24.04

Откройте корень клона в научном дистрибутиве:

```bash
python3.12 -m venv .venv-science
.venv-science/bin/python -m pip install -r demo_three/requirements-controller.txt
.venv-science/bin/python -c "import sys,numpy,pymavlink; print(sys.version); print(numpy.__version__,pymavlink.__version__)"
```

Требуется также системный пакет `python3.12-venv`. Лёгкий controller runtime
использует только NumPy/pymavlink; Torch и PyBullet для этой Gazebo-сцены не нужны.
Не устанавливайте NumPy 2.5.4 в системный Python 3.10 GUI-дистрибутива.

Из этого же корня можно отдельно проверить ранее включённые контроллеры:

```bash
.venv-science/bin/python -I -B scripts/replay_controllers.py --controller all
```

Это 3500 архивных NN-отсчётов, без новой физики. Они не проверяют установку Gazebo
и не являются результатом новой трёхдроновой демонстрации.

## Gazebo и запись GUI — Ubuntu 22.04

Нужен установленный Gazebo Harmonic с его официальным apt-репозиторием.
В рабочей системе присутствуют следующие пакеты; команда предназначена для
подготовки среды, а не является записью уже выполненной установки чистого клона:

```bash
sudo apt-get install build-essential cmake git pkg-config gz-harmonic \
  libgz-sim8-dev libgz-plugin2-dev python3-gz-transport13 python3-gz-msgs10 \
  ffmpeg xterm xdotool x11-utils tcpdump python3-pil fontconfig fonts-dejavu-core
/usr/bin/python3 -c "import gz.transport13; import gz.msgs10"
gz sim --version
```

Версии прочих фактически установленных утилит: CMake 3.22.1, FFmpeg 4.4.2,
tcpdump 4.99.1, xterm 372, xdotool 3.20160805. Экспорт не запускает apt/pip,
не загружает зависимости и не проверяет доступность этих версий в будущих зеркалах.

Из корня клона соберите новый плагин одинакового воздействия:

```bash
cmake -S demo_three/force_plugin -B demo_three/force_plugin/build
cmake --build demo_three/force_plugin/build -j2
```

Ожидаемый результат — `demo_three/force_plugin/build/libThreeDroneForceSystem.so`.
Плагин атомарно подаёт одну мировую ENU-силу трём уникальным физическим звеньям.
Старый одиночный плагин не подменяется. Сборка не запускает симулятор.

## Внешние исходники

В Ubuntu 22.04 выберите собственный каталог установки плагина:

```bash
export COPTER_GAZEBO_ROOT="$HOME/ardupilot_gazebo"
mkdir -p "$(dirname "$COPTER_GAZEBO_ROOT")"
git clone https://github.com/ArduPilot/ardupilot_gazebo.git "$COPTER_GAZEBO_ROOT"
git -C "$COPTER_GAZEBO_ROOT" checkout 082a0fe231f6e63bc8d1598f1cba461d9e2ea7f5
export GZ_VERSION=harmonic
cmake -S "$COPTER_GAZEBO_ROOT" -B "$COPTER_GAZEBO_ROOT/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$COPTER_GAZEBO_ROOT/build" -j2
```

В Ubuntu 24.04 установите и соберите SITL. Эти шаги используют внешние исходники
и могут требовать дополнительных системных пакетов официального prerequisite-скрипта:

```bash
export COPTER_ARDUPILOT_ROOT="$HOME/vkr_sim/ardupilot"
mkdir -p "$(dirname "$COPTER_ARDUPILOT_ROOT")"
git clone https://github.com/ArduPilot/ardupilot.git "$COPTER_ARDUPILOT_ROOT"
git -C "$COPTER_ARDUPILOT_ROOT" checkout fb773567896014da007bca58358704d4a0d8516f
git -C "$COPTER_ARDUPILOT_ROOT" submodule update --init --recursive
cd "$COPTER_ARDUPILOT_ROOT"
Tools/environment_install/install-prereqs-ubuntu.sh -y
./waf configure --board sitl
./waf copter
```

Runtime использует `build/sitl/bin/arducopter` и штатные
`Tools/autotest/default_params/copter.parm`, `gazebo-iris.parm`.
Собственный каталог установки передаётся конфигурацией runtime; имя пользователя
не должно быть вшито в исходник. Доступ GUI к моделям обеспечивается каталогом
`$COPTER_GAZEBO_ROOT/models` и плагином из `$COPTER_GAZEBO_ROOT/build`.

## Общие быстрые журналы WSL

Живые высокочастотные журналы пишутся в общий Linux tmpfs
`/mnt/wsl/copter_three`; после остановки собственных процессов точная копия
переносится в итоговый output. До подтверждённого копирования живые журналы
сохраняются. Оба пользователя WSL должны иметь одинаковый численный UID.
В текущем стенде оба UID равны 1000; фиксированное имя пользователя не требуется.

Проверка из PowerShell, с именами ваших дистрибутивов:

```powershell
wsl -d Ubuntu -- id -u
wsl -d Ubuntu-24.04 -- id -u
```

При разных UID сначала настройте совпадающую учётную запись. Не меняйте владельца
существующей общей папки наугад. Из GUI-дистрибутива создайте только этот новый
каталог или проверьте существующий; рекурсивный `chown` не нужен:

```bash
if [ -e /mnt/wsl/copter_three ]; then
  test -d /mnt/wsl/copter_three && test ! -L /mnt/wsl/copter_three && \
    test "$(stat -c %u /mnt/wsl/copter_three)" = "$(id -u)" || exit 1
else
  sudo install -d -m 0750 -o "$(id -u)" -g "$(id -g)" -- /mnt/wsl/copter_three
fi
```

Pillow (`python3-pil`), `fc-match` и DejaVu Sans нужны для гладких надписей на
реальных объектах сцены; это системный шрифт, а не сгенерированные кадры полёта.

## Запуск сцены — требуется итоговая проверка клона

Имена дистрибутивов могут отличаться от рабочего стенда; перед запуском проверьте
их через `wsl.exe --list --verbose`. По умолчанию научный интерпретатор ищется
в `Path.home()/vkr_sim/venv-cu128/bin/python` научного дистрибутива; CUDA не нужна.
`--science-python` или `COPTER_SCIENCE_PYTHON` выбирает другую установленную среду.
ArduPilot по умолчанию расположен в `Path.home()/vkr_sim/ardupilot` научного
дистрибутива, плагин — в `Path.home()/ardupilot_gazebo` GUI-дистрибутива.

Из корня клона в GUI-дистрибутиве Ubuntu 22.04 реализованы две точки входа:

```bash
python3 -m demo_three.present route
python3 -m demo_three.present hover
```

Маршрут использует постоянную сильную нагрузку, удержание — порывы. Для среды
из этой инструкции и явного общего каталога:

```bash
python3 -m demo_three.present route --science-python "$PWD/.venv-science/bin/python" --live-root /mnt/wsl/copter_three
python3 -m demo_three.present hover --science-python "$PWD/.venv-science/bin/python" --live-root /mnt/wsl/copter_three
```

Синтаксис соответствует существующему CLI. Полный полёт из нового чистого
публичного клона пока не объявляется проверенным. `demo_three/assets.py`
явно различает исследовательские оригиналы и публичные metadata-проекции;
существующий оригинал с неверным SHA не заменяется проекцией молча.

Перед каждым запуском нужен новый output. TCP 5760/5770/5780 и UDP
9002/9012/9022/9032 должны быть свободны. Перехват localhost использует tcpdump
с соответствующими правами; runtime не завершает чужие процессы.
GUI-запись читает окно конкретного собственного процесса через X11, а не
генерирует траектории по CSV. FFmpeg/xdotool/xwininfo/xprop нужны в GUI-дистрибутиве.

## Модели и данные в публичном клоне

Код демонстрации лежит в `demo_three/`; неизменённые контроллеры, кодировщики и
NPZ — в `v2/`. Общий `DATA_ROOT=v2` должен применяться явно и при импортах, и
при поиске моделей. В набор включены Direct111 и отдельный публичный экспорт
текущего гибрида с точными F0/route/source68 NPZ. Original ZIP обучения не включён
и локально не считается проверенным: этот экспорт имеет отдельный контракт.

PID использует уже опубликованный `frozen_native_pid.public.json`: четыре
параметра побайтно описывают те же численные значения, metadata обезличена.
Калибровка `calibration.public.json` сохраняет все численные поля оригинала:
меняется только `source_run`, добавляется `_meta` с digest оригинала и перечнем
изменённых полей. Оригинальный SHA не выдаётся за SHA публичного файла.

Исторические таблицы/карты не становятся результатом общего нового мира.
Это отдельная демонстрация с тремя перенесёнными траекториями; её измерения,
неудачи и ограничения сохраняются отдельно. Новых шагов обучения эта сцена не делает.
