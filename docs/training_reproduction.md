# Повторение короткого дообучения W8 CONTROL

Капсула `training_capsule/` содержит исходники, небольшие обязательные свидетельства
и SB3 initializer. Это ранее обученная политика, а не случайные начальные веса.
Она воспроизводит процедуру короткого v8 CONTROL, породившую текущую измеренную
политику e0aa. Новые v16-опыты — отдельная серия.

Личные пути в metadata заменены, затем зависимые guard SHA заново связаны с
**фактическими опубликованными байтами**. Исходные и export SHA разделены в
`training_capsule/CAPSULE_MANIFEST.json`. Функция `sha()` не подменяется; guards
не отключены. В Python совпадает AST после нормализации только путей и SHA;
числа JSON и все tensor/Adam ZIP members исходного initializer сохранены.

Original initializer: `7d41ffdf401e46eb999eff8daa64b84f644164f86c1775fa4efb2233aeb1ac26`.
Public initializer: `7df5dd0a382525d08bfb4b2d8e4514e7269438bbd44a63598eff90ac0a5f3088`.
Это разные SHA metadata, при одинаковых13 policy tensors и13 Adam states.

## Команды

Python3.12, Ubuntu24.04/WSL2. Начните из корня репозитория:

```bash
python -m pip install -r requirements-training.txt
python scripts/reproduce_training.py --baseline-only
```

Последняя команда создаёт новый каталог `runs/reproduction_…` и выполняет24
full-horizon CPU baseline-эпизода без gradient updates. Она не запускает сервер,
Gazebo или основное PPO. Для подготовки полного guarded запуска:

```bash
python scripts/reproduce_training.py --prepare
python scripts/reproduce_training.py --workspace runs/reproduction_<выданное_имя> --run
```

`--prepare` создаёт другой свежий workspace, выполняет baseline24 и оригинальную
ограниченную semantic verification: синтетические fixtures, три физические
lockstep-проверки и512 optimizer-probe transitions, всего8 Adam updates. Модели
этой пробы не становятся initializer. Чтобы продолжить свой baseline workspace,
передайте его `--workspace` вместе с `--prepare`.

`--run` требует подготовленный workspace и запускает только официальный
`joint_research_preflight_v1.py --run` с зафиксированным планом13 метрик. Бюджет:
131072 critic transitions,262144 PPO transitions, три candidate snapshots,
seeds9511/9611. Все неудачные checkpoints сохраняются. Автоматической подмены
демонстрационной модели или увеличения бюджета нет.

## Что проверено при упаковке

Из опубликованной капсулы выполнены **24 новых CPU baseline-эпизода**,42000
control steps/336000 physics substeps. Все24 CSV trace SHA, метрики, события,
геометрия и360 массивов actor-I/O точно совпали с исходным initializer baseline.
Проверены13 policy tensors и весь Adam; deliberate tamper привязанного source
отклонён настоящим guard до запуска физики. Это проверка переносимости ПО,
**не новые научные повторы и не подтверждение Gazebo**.

При упаковке дополнительно проверены bounded semantic probes:4556 физических
control steps =4044 lockstep +512 optimizer-probe transitions,36448 PyBullet
substeps и8 Adam updates(4critic+4PPO). C-call observer подтвердил точный заранее
зафиксированный бюджет. Проверена подготовка official preflight-плана13 метрик.
Основное PPO при упаковке не запускалось. Наличие капсулы
не обещает побитового совпадения будущего конечного checkpoint: это проверяется
после реального воспроизведения. Python/библиотеки/CPU и версии должны совпадать.

Ручной Gazebo launcher по-прежнему требует внешний ArduPilot/Gazebo stack и
исходный workspace. Капсула воспроизводит CPU обучение, а не архивные physical/
wire доказательства и не устанавливает симулятор Gazebo. Старые WORKSPACE_ONLY
launcher copies остаются справочным исходным текстом.
