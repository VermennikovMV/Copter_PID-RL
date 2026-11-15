# Установка и команды

## Переносимое выполнение контроллеров

Проверено: Python 3.12.3 и NumPy 2.5.4, Ubuntu 24.04/WSL2. Из корня пакета:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python scripts/check_local_package.py
python scripts/controllers.py --controller hybrid
python scripts/controllers.py --controller direct
python scripts/controllers.py --controller pid-fast
python scripts/replay_controllers.py --controller all
```

В PowerShell активация: `.\.venv\Scripts\Activate.ps1`. Метаданные доступны
также для `historical-hybrid` и `pid-native`; последний указывает внешний SITL.
Replay выполняет Direct111 на всех 1500 сохранённых отсчётах быстрого физического
стенда и текущий гибрид на полном причинном префиксе из 2000 отсчётов PyBullet.
Проверяются наблюдения, действия, собственная предыдущая команда и переключение
экспертов. Это 3500 вызовов сети без новой физики, обучения, Torch или Gazebo.
Отдельный `scripts/replay_memory20_example.py` проверяет 64 архивных encoded
наблюдения F0 из Gazebo; он не заменяет непрерывную проверку кодировщика.

## Три дрона в Gazebo из публичного клона

В `demo_three/` включён совместный стенд PID / Hybrid phase29 / Direct111.
После [установки Gazebo и SITL](../docs/gazebo_setup.md) выполните в GUI-Ubuntu:

```bash
python3 -m demo_three.present route
python3 -m demo_three.present hover
```

Открываются три модели в одном мире и запись окна. [Готовые видео, метрики и
проверка одинаковых условий](../results/shared_world/README.md). Полный частный
training-архив для этих двух демонстраций не требуется.

## Дополнительный CPU-симулятор

```bash
python -m pip install -r requirements.txt
python scripts/quickstart.py --scenario force_strong --mission route --seed 84001
```

Это новые физические шаги PyBullet исторического Path77/e0aa, а не v29 или
Gazebo-карта. Результаты сохраняются в `runs/`. Версии 77/80/Direct имеют разные
контракты и не подменяют друг друга. [Модели](../MODEL_CARD.md).

## Полный исследовательский workspace

Следующие команды выполняются в исходном полном проекте. Нужны точные guarded
данные, исходная регистрация, WSL, SITL и Gazebo. Копии в `source_archive/` нужны
для проверки реализации; лёгкий публичный клон не содержит всего внешнего стека.

```powershell
# Последняя измеренная версия v29; тот же фиксированный PID для сравнения
.\show_joint_phase29.cmd
.\show_joint_phase29.cmd -Controller pid
.\show_joint_phase29.cmd -Seed 84001

# Предыдущая F0/v20 версия
.\show_joint_memory20.cmd

# Исторические три архитектуры одной общей Gazebo-серии
.\show_gazebo.cmd pid gust route
.\show_gazebo.cmd hybrid gust route
.\show_gazebo.cmd direct gust route

# Новый запуск реального PPO с отдельными выходами
.\train_joint_prior28.cmd --prepare-only
.\train_joint_prior28.cmd
.\show_joint_training_native_v28.cmd
```

У viewer по умолчанию может быть открыт завершённый архив. Для живой записи
используйте `-RunRoot` свежего запуска, как в
[инструкции viewer](../source_archive/recording_latest/prior28_viewer_source.txt).
Показатели берутся из настоящего heartbeat/Adam, а не имитируются анимацией.
Включённый [снимок живого PPO](../results/training_viewer/prior28_viewer_live_cold_root03_ppo.png)
получен во время полного повторного обучения из восстановленного private03.

Обучение проходит в PyBullet; Gazebo — отдельная проверка переноса. V29 launcher
прошёл реальный 40-секундный запуск с пятью текущими целями и проверкой NN/силы/
физики. Этот демонстрационный полёт не добавлен к девяти полётам основной матрицы.
Видеозапись пользователь делает самостоятельно. Фиксируйте версию, SHA, сценарий
и backend; отделяйте измерительный участок от взлёта и посадки. Новые обученные
веса не наследуют готовую карту. [Восстановление данных](reproduction.md).
