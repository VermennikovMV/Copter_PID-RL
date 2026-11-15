"""Build an allowlisted, standalone public research export. No network writes.

Only named completed development evidence is opened. Existing experiments and
reserved confirmation results are neither discovered nor modified.
"""
from __future__ import annotations

import argparse
import copy as copy_module
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "github-ready-joint-v2.0"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def json_write(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def project(value):
    """Public projection: original receipt remains unchanged in local archive."""
    if isinstance(value, dict):
        return {project(k): project(v) for k, v in value.items()}
    if isinstance(value, list):
        return [project(v) for v in value]
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    normalized = normalized.replace(str(ROOT).replace("\\", "/"), "v2")
    normalized = re.sub(r"/home/[^/:;\s]+", "[external-home]", normalized)
    # Keep experiment-relative suffixes, never the owner's machine/account path.
    if normalized.startswith(("/mnt/", "/home/")) or re.match(r"^[A-Za-z]:/", normalized):
        if "/v2/" in normalized:
            return "v2/" + normalized.split("/v2/", 1)[1]
        return "[external-local-path]/" + normalized.rsplit("/", 1)[-1]
    return normalized


README = r'''# RL quadrotor control: wind retention and route geometry

Исследовательская реализация управления квадрокоптером: нейросеть заменяет
внешний контур положения, а штатный внутренний контур ArduPilot сохраняется.
Актор получает 74 признака из измеряемых состояния и истории, выдаёт коллективную
тягу и угловые скорости. PPO обучается в быстром PyBullet; перенос проверен
отдельными физическими полётами ArduPilot SITL + Gazebo.

**Текущий статус — development, не окончательное подтверждение.** Основная карта
применимости относится к одному seed среды `83301`. Репликации трёх обученных
акторов используют ту же среду. Полный исходный макронабор ветровых условий
не покрыт: `force_mild` отсутствует. Закрытые подтверждающие результаты не входят
в этот экспорт.

## Что получилось

Данные ниже — реальные Gazebo полёты; CPU quickstart создаёт отдельные результаты
PyBullet и не воспроизводит эту таблицу.

| Условие / метрика | W8 PPO, seed 111 | Native PID | Вывод в данной среде |
|---|---:|---:|---|
| Сильный ветер, hover RMSE, м | 0.032969 | 0.067516 | ошибка ниже на 51.2% |
| Порывы, hover RMSE, м | 0.060521 | 0.106141 | ошибка ниже на 43.0% |
| Сильный ветер, время маршрута, с | 8.06 | 9.46 | быстрее на 14.8% |
| Порывы, время маршрута, с | 8.04 | 9.54 | быстрее на 15.7% |
| Сильный ветер, finite 3D RMSE, м | 0.146357 | 0.145075 | фиксированные допуски пройдены |
| Номинал, finite 3D RMSE, м | 0.154257 | 0.104546 | допуск геометрии **не пройден** |

**Ограничение:** в отдельном exploratory stress seed `84001` при попутном сильном
ветре W8: 8.92 с, finite 3D RMSE 0.19866 м, пик 0.50125 м; PID: 9.12 с,
0.09274 м, пик 0.21521 м. Порог 0.50 м превышен. Аудиты физики/актора/метрик
пройдены, но качество не прошло; универсального превосходства при ветре нет.

См. [полные таблицы](results/pilot32/applicability.csv),
[маршруты](results/pilot32/route_metrics.csv),
[hover](results/pilot32/hover_metrics.csv),
[график](results/pilot32/applicability.png) и
[причины результата](docs/causal_evidence.md).

```mermaid
flowchart LR
  M[Measured state + temporal memory, 74] --> N[BC → wind PPO → geometry PPO W8]
  N --> T[Collective thrust + angular rates]
  T --> A[Thrust semantics adapter]
  A --> I[Native ArduPilot inner control]
  I --> G[Gazebo Iris physics]
  G --> M
  G --> V[Independent physics, force, actor and metric audits]
```

## Что обеспечило результат

1. BC и последующий wind PPO дают исходную политику и удержание при ветре.
2. В парном сравнении W0/W8 обе ветви стартуют с одного checkpoint, имеют
   одинаковые PPO seed, распределение, 2 097 152 шага и равное число кандидатов.
   Меняется штраф расстояния до конечного сегмента маршрута (`W8` вместо `W0`).
   На сильном ветре W0 → W8: время 8.34 → 8.06 с, finite RMSE
   0.165524 → 0.146357 м; hover 0.040653 → 0.032969 м.
3. KL-якорь 0.1 использован для ограничения ухода от фиксированного prior.
   Удержание wind навыка измерено, но отдельный причинный вклад KL здесь
   не изолирован. Доли 80% route / 20% hover и завершение учебного эпизода при
   успехе — общие настройки двух ветвей, их самостоятельный эффект не проверен.
4. Корректная семантика коллективной тяги и сохранённый native inner loop
   проверены отдельно. Это проверка интерфейса, не единственная причина выигрыша.

Вариант модели внутренней динамики, tailwind curriculum и arrival reward
**не дали улучшения качества**. Они включены в индекс отрицательных результатов.

## Быстрый запуск на CPU

Проверенная среда: Ubuntu 24.04 (в том числе WSL2), Python 3.12. Запуск из корня
клона. Для CPU inference не нужны Torch, ArduPilot, Gazebo или исходный архив.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_export.py
python scripts/cpu_smoke.py
python scripts/quickstart.py --scenario force_strong --mission route --seed 84001
```

В Windows окружение активируется командой `.venv\Scripts\Activate.ps1`;
для физического SITL использовался Windows + WSL2. Чистая установка Gazebo
на другой машине этим экспортом не проверена.

`quickstart.py` использует включённый NPZ, выполняет реальные шаги PyBullet,
записывает trace, измеренные входы и действия в **новый** `runs/`.
Результат явно помечен как `PYBULLET_CPU_DEVELOPMENT`, а не Gazebo validation.
Для сравнения доступны `--model control`, `--model wind`, `--model bc`.

## Повторить парное обучение

Ниже — команда воспроизведения исторической пары. Для нового исследования
сначала обязателен [план соответствия метрикам](v2/docs/joint_research_rules_v1.md)
и `v2/scripts/joint_research_preflight_v1.py`; неполный draft он отвергает.
Проверка фиксирует источники, свидетельства, бюджет и 13 метрик, но смысл reward
и фактическое качество проверяются отдельно. [Правила агента](v2/AGENTS.md)
сохраняют эти требования между сессиями.

Опциональные зависимости для исторического воспроизведения:

```bash
python -m pip install -r requirements-training.txt
python v2/scripts/run_joint_training_pair_v2.py \
  --output artifacts/refinement/joint_v2/reproduction_pair \
  --seed 9611 --warmup-seed 9511 --steps 2097152 \
  --warmup-steps 131072 --eval-every 1048576
```

По умолчанию включённый wind checkpoint выбирается внутри `v2/`. Driver
отказывается перезаписывать существующий каталог. Полное 2M обучение в CI
не запускается; импорт и CLI проверяются отдельно. Совпадение хэша после
повторного обучения на другой аппаратуре не гарантируется.

## Структура и проверяемость

- `v2/quadrl/`, `v2/reference/`: точные исходные байты ядра, encoder, модели и reward.
- `v2/scripts/`: точные драйверы PPO и их зависимости.
- `v2/audit/`, `audit/`: независимые аудиторы и общие внешние helpers,
  включённые в экспорт; относительная структура оригинала сохранена.
- `v2/artifacts/`: выбранные небольшие NPZ, публичная проекция начального
  checkpoint, public PID metadata и явно помеченный snapshot состояния.
- `results/`: таблицы и публичные проекции завершённых receipts.
- `evidence/`: исходные SHA-256, размерный индекс исключённого сырья,
  исходные версии внешнего стека и перечень допустимых утверждений.
- `MANIFEST.json`: SHA-256 и размеры каждого публикуемого файла.

[Карточка моделей](MODEL_CARD.md) связывает веса с обучением и ограничениями.

Основной pilot: 32/32 physical, force и metric; 26/26 actor;
44 967 проверенных NN wire commands; 14/14 integrity checks.
Независимая сверка таблиц: 2 660 полей, PASS. Репликация: **16 новых** полётов
плюс **14 общих** старых, покрытие 30; это не 30 новых полётов.

Raw PCAP/BIN/видео, виртуальные среды, сборки, диссертационные документы,
учётные данные и незавершённые runs не входят. Без raw archive нельзя заново
получить доказательство сетевой/физической идентичности: included auditors
сохранены для проверки исходного архива, а не для имитации отсутствующих данных.
[Границы воспроизводимости](docs/reproducibility.md).

Публичные JSON — **проекции**, не побитовые копии original receipts: локальные
абсолютные пути заменены относительными обозначениями. SHA оригинала и SHA
проекции записаны отдельно в [индексе](evidence/receipts.json). Оригиналы не менялись.
SHA доказывает тождество байтов, сам по себе не доказывает корректность эксперимента.
Начальный SB3 ZIP также имеет отдельную публичную проекцию: единственное
изменение — удаление локального debug-log path; tensor entries сохранены
побитово. [Квитанция](evidence/checkpoint_projection.json) содержит оба SHA.
Алгоритмы и числовые научные критерии сохраняются; публичные пути и SHA-привязки
метаданных отличаются от закрытого экспериментального архива. Исходный contract
сохранён как исторический reference, а `*.public.json` и public preflight явно
помечены как адаптации. Manifest различает source и export SHA каждого файла.

## Лицензии и цитирование

Лицензия авторского кода и весов владельцем пока не выбрана; этот экспорт
не выдаёт MIT или иную лицензию за автора. Реальные лицензии зависимостей и
источники зафиксированы в [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).
Полные внешние проекты и их бинарники здесь не распространяются.
DOI, публикация, полное авторское имя и релизный URL не вымышляются;
для ссылки используйте фактический URL репозитория и commit после публикации.

English: research-only development evidence for an upper-loop RL quadrotor
controller. Wind-hover gains and faster routes were observed in one Gazebo
development environment; nominal geometry and a tailwind stress case fail fixed
quality guards. CPU reproduction is separate from archived Gazebo evidence.
'''


CAUSAL = r'''# Причины, контроли и отрицательные результаты

Основной источник чисел: `results/pilot32/route_metrics.csv` и `hover_metrics.csv`.
Набор фиксирован заранее, все 32 попытки сохранены; status физического аудита
не заменяет конъюнктивный gate качества.

| Изменение | Контроль и наблюдение | Допустимый вывод |
|---|---|---|
| BC → wind PPO | неизменённая measured-only архитектура, отдельные сохранённые политики | предварительный wind навык присутствует до geometry PPO |
| W0 → W8 | одно начало, одинаковые seed, 2M steps/ветвь, 3 оценённых кандидата/ветвь | локальная причинная оценка добавки geometry reward в данной процедуре |
| Fixed prior KL=0.1 | prior не заменяется выбранными поздними checkpoint | настройка для ограничения ухода от prior; самостоятельный причинный вклад KL не изолирован |
| Thrust interface A/B | те же веса; 12 валидных полётов / 13 попыток, infra failure сохранён | интерфейс важен для корректной физической интерпретации; не объясняет сам весь speed gain |
| Training seeds 111/222/333 | одна среда 83301; 16 новых + 14 общих = 30 coverage | воспроизводимость между акторами, не обобщение по средам |
| Номинальная геометрия | finite RMSE 0.15426 / 0.13878 / 0.16684 м против PID 0.10455 м | все три показывают accuracy-speed tradeoff |
| Попутный ветер, env84001 | W8 peak 0.50125 м против порога 0.50, PID 0.21521 м | отрицательный stress case ограничивает рекомендацию |
| Native rate shaper V3/V4 | gyro replay улучшился на 94.73%, PPO quality не улучшился | улучшение subsystem replay не равно улучшению управления |
| Tailwind V5 | 1085 curriculum triggers, quality improvement не получен | срабатывание curriculum само по себе не доказывает обучение |
| Arrival V3 | 6 Gazebo probe, медленнее и хуже геометрия, чем W8 | модификация не принята как улучшение |
| Velocity conditioning gamma=1.15 | folding 6 first-layer columns; native PyBullet diagnostic | не новое RL обучение, не Gazebo accepted controller |

Таблица не утверждает изолированный эффект каждого элемента всей длинной
цепочки обучения. Строго контролируемая локальная абляция здесь — W0/W8.
Закрытая независимая проверка не анализируется и не подразумевается.
'''


REPRO = r'''# Воспроизводимость

## Проверено в экспортном дереве

`scripts/verify_export.py` проверяет allowlist, хэши, размеры, локальные ссылки,
отсутствие персональных абсолютных путей и секретных файлов.
`scripts/cpu_smoke.py` импортирует ядро из `v2/`, открывает все NPZ и выполняет
реальную короткую физику, проверяет finite geometry и encoder contract.
`scripts/quickstart.py` выполняет новый полный CPU rollout и сохраняет измерения.
CI делает эти проверки без доступа к исходной рабочей папке и без Gazebo.
Дополнительно 15 unittest проверяют preflight: draft, ложные/подменённые
свидетельства, пропавшие метрики, изменённые бюджеты и занятый output отвергаются.
Public contract перепривязывает только public PID metadata; все числовые
критерии, определения, матрица и параметры PID независимо сверяются с оригиналом.

## Доступно для отдельного повторения

Парный PPO driver и исходный wind checkpoint включены. Каждая ветвь получает
одинаковую инициализацию и warmup; отличие — route geometry reward W8/W0.
Полное обучение длительное и намеренно не запускается каждым push.

## Что требует внешнего архива и системной установки

Реальный стек использовал Windows + WSL2, Ubuntu 24.04 Python 3.12,
ArduPilot 4.7.2 и Gazebo Harmonic 8.14. Версии/commit/SHA извлечены из сохранённых
manifests в `evidence/external_stack.json` без локальных путей.
Системные Gazebo bindings относятся к Python дистрибутива, не к pip-окружению.
Полные ArduPilot/Gazebo repo и бинарники не включены.

Независимые аудиторы сохранены с исходной относительной структурой:
`v2/audit` находит общие `audit/pcap_evidence.py`, `dataflash_parameters.py`
и `verify_evidence.py`. До этого часть helpers лежала за пределами `v2`.
Метрики на реальных raw traces, packet identity, force causality и exact wire
replay требуют PCAP/BIN/JSONL/telemetry исходного архива. Их размеры/хэши
зафиксированы в `evidence/excluded_raw_index.json`; публичного download URL
пока нет. Повторный полный Gazebo replay из одного этого клона не заявляется.

`v2/scripts/build_github_ready_v2.py` сохраняет точный алгоритм упаковки:
он рассчитан на полный исходный архив и training environment, а не на
повторную упаковку уже сокращённого клона. Отсутствующие raw inputs он не скачивает.

Все receipts в `results/receipts/` содержат публичную проекцию: локальные
абсолютные пути заменены. SHA исходных оригинальных байтов записаны отдельно.
Выбранные исходники, NPZ, CSV и PNG копируются побитово; manifest показывает
их source SHA и export SHA. Политика final confirmation не открывалась.
SB3 initial ZIP публикуется как проекция: `data.bc_kl_debug_log_path` заменён
на null, все tensor entries остались побитово прежними. Точный SHA оригинала,
SHA публичного ZIP и хэши tensor entries — в `evidence/checkpoint_projection.json`.
'''


QUICKSTART = r'''"""Run the packaged actor in real CPU PyBullet; never launch Gazebo."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
from quadrl.policy import NumpyPolicy
from quadrl.rollout import run_episode

MODELS = {
    "geometry": "artifacts/refinement/joint_v2/training_pair111/geometry/best_policy.npz",
    "control": "artifacts/refinement/joint_v2/training_pair111/control/best_policy.npz",
    "wind": "artifacts/refinement/wind_specialist_v1/best_policy.npz",
    "bc": "artifacts/training/main_hybrid_seed111_v1/bc_only_policy.npz",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="geometry")
    parser.add_argument("--scenario", choices=("nominal", "force_strong", "gust"), default="force_strong")
    parser.add_argument("--mission", choices=("hover", "route"), default="route")
    parser.add_argument("--seed", type=int, default=84001)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or not 100 <= args.seed < 90000:
        parser.error("Use positive seconds and an open development seed in [100,90000)")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "runs" / (stamp + "_" + args.model)).resolve()
    if not output.is_relative_to(ROOT / "runs") or output.exists():
        parser.error("Select a new output beneath runs/")
    record = run_episode(NumpyPolicy(ROOT / "v2" / MODELS[args.model]),
                         scenario=args.scenario, mission=args.mission,
                         seed=args.seed, seconds=args.seconds, save_path=output)
    record["scope"] = "PYBULLET_CPU_DEVELOPMENT; not Gazebo validation"
    (output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"scope": record["scope"], "model": args.model,
                      "steps": record["steps"], "metrics": record["metrics"],
                      "output": str(output.relative_to(ROOT))}, indent=2))

if __name__ == "__main__":
    main()
'''


SMOKE = r'''"""Meaningful CPU checks against only the standalone export tree."""
from pathlib import Path
import importlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "v2"))
sys.path.insert(0, str(ROOT / "v2" / "audit"))
from quadrl.env import QuadEnv
from quadrl.policy import NumpyPolicy, ObservationEncoder, EncoderConfig
from audit.verify_joint_metrics_v2 import distances_to_leg, fixture, geometry_metrics
from quickstart import MODELS

def main():
    checks = []
    for module in ("quadrl.env", "quadrl.policy", "quadrl.rollout", "reference.direct_physics",
                   "audit.verify_joint_metrics_v2", "packet_clock_ordered",
                   "verify_joint_hybrid_v2_1", "verify_fdm_identity", "verify_force_plugin"):
        loaded = importlib.import_module(module)
        assert Path(loaded.__file__).resolve().is_relative_to(ROOT), module
    checks.append("All controller/physics/independent-auditor imports resolve inside export")
    finite, infinite = distances_to_leg(np.array([[1., .3, .4], [3., 0., 0.]]), np.zeros(3), np.array([2., 0., 0.]))
    assert np.allclose(finite, [.5, 1.]) and np.allclose(infinite, [.5, 0.])
    t, p, target, events = fixture()
    perfect = geometry_metrics(t, p, target, events, 8.5)
    assert perfect["geometry_eligible"] and perfect["segment_rmse_m"] < 1e-14
    assert perfect["complete_corner_windows"] == 3
    partial = t <= 3
    assert not geometry_metrics(t[partial], p[partial], target[partial], events[:2], None)["geometry_eligible"]
    checks.append("Finite segment/overshoot, perfect-route, missing-corner geometry guards")
    steps = 0
    for name, relative in MODELS.items():
        policy = NumpyPolicy(ROOT / "v2" / relative)
        env = QuadEnv(scenario="nominal", mission="hover", seconds=2.)
        try:
            obs, info = env.reset(seed=84001)
            assert obs.shape == (74,) and policy.encoder.config == EncoderConfig()
            for _ in range(env.max_steps):
                action = policy.predict(obs)
                assert action.shape == (4,) and np.all(np.isfinite(action)) and np.max(np.abs(action)) <= 1
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.all(np.isfinite(obs)) and np.isfinite(reward)
                assert not terminated, (name, info)
                steps += 1
                if truncated:
                    break
            assert env.steps == 100
        finally:
            env.close()
    checks.append("All 4 archived actors execute 100 real PyBullet steps each, finite and surviving")
    print(json.dumps({"status": "PASS", "scope": "CPU software checks, not Gazebo acceptance",
                      "physics_steps": steps, "checks": checks}, indent=2))

if __name__ == "__main__":
    main()
'''


VERIFY = r'''"""Verify the publication allowlist, exact hashes and public text hygiene."""
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in manifest["files"]}
    actual = {str(p.relative_to(ROOT)).replace("\\", "/") for p in ROOT.rglob("*") if p.is_file()
              and not any(part in (".git", ".venv", "__pycache__", ".pytest_cache", "runs") for part in p.relative_to(ROOT).parts)}
    assert actual == set(expected) | {"MANIFEST.json"}, (sorted(actual - set(expected)), sorted(set(expected) - actual))
    total = 0
    for name, entry in expected.items():
        path = ROOT / name
        data = path.read_bytes()
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], name
        assert len(data) < 50 * 1024 * 1024, name
        assert not re.search(r"(^|/)(\.env($|\.)|id_rsa|credentials|token)(/|$)", name, re.I), name
        assert not re.search(r"9300[123]", name), name
        if path.suffix in (".zip", ".npz"):
            with zipfile.ZipFile(path) as archive:
                for entry in archive.namelist():
                    assert not re.search(rb"[A-Za-z]:[\\/]Users[\\/]|/mnt/[a-z]/Users/|/home/(?:misha|misha3)/", archive.read(entry)), (name, entry)
        if path.suffix.lower() in (".md", ".json", ".py", ".yml", ".csv", ".txt"):
            content = data.decode("utf-8-sig")
            assert not re.search(r"[A-Za-z]:[\\/]Users[\\/]|/mnt/[a-z]/Users/|/home/(?:misha|misha3)/", content), name
            assert not re.search(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{25,})", content), name
            if path.suffix == ".md":
                for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
                    if not re.match(r"(?:https?://|#|mailto:)", link):
                        assert (path.parent / link.split("#")[0]).exists(), (name, link)
        total += len(data)
    print(json.dumps({"status": "PASS", "files": len(expected), "bytes": total,
                      "manifest_self_hash_excluded": True}, indent=2))

if __name__ == "__main__":
    main()
'''


def build(output, resume=False):
    if output.exists() and not (resume and output == ROOT / "github_ready"):
        raise FileExistsError("Choose a new output; never overwrite an existing export")
    output.mkdir(parents=True, exist_ok=resume)
    provenance = {}

    def copy(relative, target=None, base=ROOT):
        source = base / relative
        dest = output / (target or "v2/" + relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        provenance[str(dest.relative_to(output))] = dict(source_scope="v2" if base == ROOT else "parent-audit",
            source_relative=relative, source_sha256=sha(source), byte_identical=True)

    for directory in ("quadrl", "reference"):
        for path in sorted((ROOT / directory).iterdir()):
            if path.is_file() and path.suffix in (".py", ".json"):
                copy(str(path.relative_to(ROOT)))
    for name in ("metrics.py",):
        copy("sim/" + name)
    for name in ("ForceSystem.cc", "CMakeLists.txt"):
        copy("sim/force_plugin/" + name)
    for name in ("train.py", "refine_precision.py", "bc_trust_region_probe_v1.py", "train_wind_specialist_v1.py",
                 "train_joint_geometry_v2.py", "run_joint_training_pair_v2.py", "build_github_ready_v2.py"):
        copy("scripts/" + name)
    for name in ("verify_joint_metrics_v2.py", "turns_corners.py", "replay_observations.py", "packet_clock_ordered.py",
                 "independent_metrics.py", "capture_udp_copies.py", "verify_joint_wire_exclusivity_v2.py",
                 "verify_joint_hybrid_v2_1.py", "verify_fdm_identity.py", "verify_force_plugin.py",
                 "joint_pilot_tables_independent_v2.py"):
        copy("audit/" + name)
    for name in ("pcap_evidence.py", "dataflash_parameters.py", "verify_evidence.py"):
        copy(name, "audit/" + name, ROOT.parent / "audit")
    for name in ("protocol.md", "joint_applicability_contract_v2.json", "joint_geometry_pilot32_extension_v2.json"):
        copy("docs/" + name)
    for name in ("sim_joint/thrust_adapter.py", "sim_joint/__init__.py"):
        copy(name)
    copy("tests/test_joint_research_preflight_v1.py")
    canonical_contract_relative = "docs/joint_applicability_contract_v2.json"
    canonical_contract = json.loads((ROOT / canonical_contract_relative).read_text(encoding="utf-8-sig"))
    pid_relative = canonical_contract["frozen_pid_profile"]["path"]
    public_pid_relative = pid_relative.replace(".json", ".public.json")
    pid_original = json.loads((ROOT / pid_relative).read_text(encoding="utf-8-sig"))
    pid_public = project(pid_original)
    pid_public["_public_projection"] = dict(original_sha256=sha(ROOT / pid_relative),
        note="Only local metadata paths sanitized; original numerical PID parameters unchanged")
    public_pid_dest = output / "v2" / public_pid_relative
    json_write(public_pid_dest, pid_public)
    provenance[str(public_pid_dest.relative_to(output))] = dict(source_scope="v2", source_relative=pid_relative,
        source_sha256=sha(ROOT / pid_relative), byte_identical=False,
        projection_note="Public PID metadata projection; original numerical parameters unchanged")
    public_contract_relative = canonical_contract_relative.replace(".json", ".public.json")
    public_contract = copy_module.deepcopy(canonical_contract)
    public_contract["frozen_pid_profile"] = dict(canonical_contract["frozen_pid_profile"],
        path=public_pid_relative, sha256=sha(public_pid_dest))
    public_contract["_public_projection"] = dict(original_contract_sha256=sha(ROOT / canonical_contract_relative),
        original_pid_sha256=sha(ROOT / pid_relative),
        note="Only PID path/SHA rebound to public metadata projection; scientific definitions, thresholds and matrix unchanged")
    public_contract_dest = output / "v2" / public_contract_relative
    json_write(public_contract_dest, public_contract)
    public_contract_sha = sha(public_contract_dest)
    provenance[str(public_contract_dest.relative_to(output))] = dict(source_scope="v2",
        source_relative=canonical_contract_relative, source_sha256=sha(ROOT / canonical_contract_relative), byte_identical=False,
        projection_note="Only public PID path/SHA and projection provenance added; canonical contract retained separately")
    def public_source(relative, content, note):
        destination = output / "v2" / relative
        write(destination, content)
        provenance[str(destination.relative_to(output))] = dict(source_scope="v2", source_relative=relative,
            source_sha256=sha(ROOT / relative), byte_identical=False, projection_note=note)
    gate_relative = "scripts/joint_research_preflight_v1.py"
    gate_text = (ROOT / gate_relative).read_text(encoding="utf-8-sig")
    gate_text = gate_text.replace(canonical_contract_relative, public_contract_relative)
    gate_text = gate_text.replace(sha(ROOT / canonical_contract_relative), public_contract_sha)
    gate_text = "# PUBLIC EXPORT PROJECTION: contract/PID metadata rebound; original source SHA " + sha(ROOT / gate_relative) + "\n" + gate_text
    public_source(gate_relative, gate_text, "Public preflight projection: only CONTRACT path/SHA rebound plus explicit source provenance")
    template_relative = "docs/joint_research_plan_v1.template.json"
    template = json.loads((ROOT / template_relative).read_text(encoding="utf-8-sig"))
    template["contract"] = dict(path=public_contract_relative, sha256=public_contract_sha)
    template["sources"]["native_pid"] = dict(path=public_pid_relative, sha256=sha(public_pid_dest))
    template["_public_projection"] = dict(original_sha256=sha(ROOT / template_relative),
        note="Only contract/PID metadata path/SHA rebound; draft remains non-runnable")
    public_source(template_relative, json.dumps(template, ensure_ascii=False, indent=2) + "\n",
                  "Public draft template projection: contract/PID path/SHA rebound; no experiment claimed")
    agents_relative = "AGENTS.md"
    agents_text = (ROOT / agents_relative).read_text(encoding="utf-8-sig").replace(
        canonical_contract_relative, public_contract_relative)
    agents_text += "\nПубличная адаптация путей: scientific contract сохранён отдельно побитово; public contract меняет только PID metadata path/SHA.\nИсходный AGENTS SHA256: `" + sha(ROOT / agents_relative) + "`.\n"
    public_source(agents_relative, agents_text, "Public AGENTS projection: public contract path and provenance note; research rules unchanged")
    rules_source = ROOT / "docs/joint_research_rules_v1.md"
    rules_dest = output / "v2/docs/joint_research_rules_v1.md"
    rules_text = rules_source.read_text(encoding="utf-8-sig")
    rules_text, command_changes = re.subn(r"wsl -d [^\n]+--plan docs/[^\n]+\.json",
        "python v2/scripts/joint_research_preflight_v1.py --plan docs/NEW_PLAN.json", rules_text)
    if command_changes != 1:
        raise ValueError("Expected exactly one private local command in governance rules")
    rules_text = rules_text.replace("joint_applicability_contract_v2.json", "joint_applicability_contract_v2.public.json")
    rules_text = rules_text.replace(sha(ROOT / canonical_contract_relative), public_contract_sha)
    rules_text += "\nПубличная адаптация: пути/SHA PID metadata перепривязаны; числовые критерии и матрица исходного контракта не менялись.\nИсходный contract SHA256: `" + sha(ROOT / canonical_contract_relative) + "`; исходный rules SHA256: `" + sha(rules_source) + "`.\n"
    write(rules_dest, rules_text)
    provenance[str(rules_dest.relative_to(output))] = dict(source_scope="v2",
        source_relative="docs/joint_research_rules_v1.md", source_sha256=sha(rules_source), byte_identical=False,
        projection_note="Public rules projection: portable command, public contract binding and explicit source provenance; scientific rules unchanged")
    models = ["artifacts/refinement/joint_v2/training_pair111/geometry/best_policy.npz",
              "artifacts/refinement/joint_v2/training_pair111/control/best_policy.npz",
              "artifacts/refinement/wind_specialist_v1/best_policy.npz",
              "artifacts/refinement/wind_specialist_v1/best_model.zip",
              "artifacts/training/main_hybrid_seed111_v1/bc_only_policy.npz"]
    for name in models:
        if not name.endswith(".zip"):
            copy(name)
            continue
        source, dest = ROOT / name, output / "v2" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        unchanged = {}
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(dest, "w") as public:
            for entry in original.infolist():
                data = original.read(entry.filename)
                if entry.filename == "data":
                    record = json.loads(data)
                    record["bc_kl_debug_log_path"] = None
                    data = (json.dumps(record, indent=4) + "\n").encode("utf-8")
                else:
                    unchanged[entry.filename] = hashlib.sha256(data).hexdigest()
                public.writestr(entry, data)
        relative = str(dest.relative_to(output))
        provenance[relative] = dict(source_scope="v2", source_relative=name, source_sha256=sha(source),
            byte_identical=False, projection_note="Only data.bc_kl_debug_log_path set to null; tensor entries unchanged")
        json_write(output / "evidence/checkpoint_projection.json", dict(source_relative=name, source_sha256=sha(source),
            projection=relative, projection_sha256=sha(dest), byte_identical=False,
            changed_fields=["data.bc_kl_debug_log_path: local path -> null"], unchanged_entries_sha256=unchanged))
    import numpy as np
    registry = []
    for name in models:
        if not name.endswith(".npz"):
            continue
        with np.load(ROOT / name, allow_pickle=False) as archive:
            registry.append(dict(path="v2/" + name, sha256=sha(ROOT / name), bytes=(ROOT / name).stat().st_size,
                architecture=str(archive["architecture"]), observation_contract=str(archive["contract"]),
                layers=[list(archive[f"w{index}"].shape) for index in range(int(archive["layers"]))]))
    json_write(output / "evidence/models.json", dict(scope="Unmodified archived development actors", models=registry))
    for path in sorted((ROOT / "reports/joint_geometry_pilot_v2").iterdir()):
        if path.suffix in (".csv", ".png", ".pdf"):
            copy(str(path.relative_to(ROOT)), "results/pilot32/" + path.name)

    receipts = []

    def receipt(relative, target=None, projection=None):
        source = ROOT / relative
        original = json.loads(source.read_text(encoding="utf-8-sig"))
        target = target or "results/receipts/" + relative.replace("/", "__")
        dest = output / target
        json_write(dest, project(original) if projection is None else projection(original))
        receipts.append(dict(source_relative=relative, source_sha256=sha(source), source_bytes=source.stat().st_size,
                             projection=target, projection_sha256=sha(dest), byte_identical=False,
                             projection_note=("Local absolute paths replaced; public projection is not original receipt bytes"
                                              if projection is None else "Public export snapshot; internal process/service notes omitted; not original receipt bytes")))
        return original

    for name in ("joint_geometry_pilot32_integrity_v2.json", "joint_geometry_replication16_integrity_v2.json",
                 "joint_tailwind_stress4_integrity_v2.json", "joint_thrust_recovery_integrity_v2.json",
                 "joint_arrival_probe6_integrity_v3.json", "joint_pilot_tables_independent_v2_002.json",
                 "joint_training_pair111_complete_integrity_20261113_01.json",
                 "joint_training_pair111_final_comparison_20261113_01.json",
                 "joint_native_rate_physics_replay_v3.json",
                 "joint_nativeplant111_final_integrity_20261113_01.json",
                 "joint_tailwind111_final_integrity_20261113_01.json",
                 "joint_research_preflight_v1_002.json"):
        receipt("audit/" + name)
    for name in ("plan.json", "paired_complete.json"):
        receipt("artifacts/refinement/joint_v2/training_pair111/" + name)
    receipt("reports/joint_geometry_pilot_v2/operational_gate.json")
    receipt("artifacts/refinement/joint_v2/nativeplant111/complete.json")
    receipt("artifacts/refinement/joint_v2/tailwind111/complete.json")
    receipt("artifacts/refinement/joint_v2/tailwind111/curriculum_counts_ppo.json")
    state_relative = "artifacts/refinement/joint_v2/continuation_state.json"
    def state_projection(original):
        fields = ("current_actor", "current_actor_sha256", "completed", "development_seeds",
                  "reserved_final_seeds_not_opened", "current_table", "next")
        return dict(status="PUBLIC_EXPORT_SNAPSHOT_NOT_LIVE_STATE", original_status=original["status"],
                    original_updated_utc=original.get("updated_utc"), original_sha256=sha(ROOT / state_relative),
                    contract=public_contract_relative, original_contract_reference=original.get("contract"),
                    projection_note="Public export snapshot; internal service/process notes omitted; no final acceptance implied",
                    **{key:project(original[key]) for key in fields if key in original})
    receipt(state_relative, "v2/" + state_relative, state_projection)

    batches = ["artifacts/refinement/joint_v2/gazebo_geometry_pilot32",
               "artifacts/refinement/joint_v2/gazebo_geometry_replication16",
               "artifacts/refinement/joint_v2/gazebo_tailwind_stress4",
               "artifacts/refinement/joint_v2/gazebo_thrust_ablation_recovery",
               "artifacts/refinement/joint_v2/gazebo_arrival_probe6"]
    raw_index, stack = [], []
    for batch in batches:
        for filename in ("plan.json", "complete.json"):
            if (ROOT / batch / filename).is_file():
                receipt(batch + "/" + filename)
        runs = ROOT / batch / "runs"
        if not runs.is_dir():
            continue
        # Scoped to explicitly named completed development batches only.
        for run in sorted(runs.iterdir()):
            if not run.is_dir() or not (run / "summary.json").is_file():
                continue
            summary = json.loads((run / "summary.json").read_text())
            recorded = summary.get("hashes", {})
            if (run / "manifest.json").is_file():
                manifest = json.loads((run / "manifest.json").read_text())
                stack_record = dict(run=str(run.relative_to(ROOT)), manifest_sha256=sha(run / "manifest.json"),
                                    manifest=project(manifest))
                if (run / "gazebo_manifest.json").is_file():
                    stack_record["gazebo_manifest_sha256"] = sha(run / "gazebo_manifest.json")
                    stack_record["gazebo_manifest"] = project(json.loads((run / "gazebo_manifest.json").read_text()))
                stack.append(stack_record)
            for path in sorted(run.iterdir()):
                if path.is_file() and path.name in ("summary.json", "manifest.json", "network.pcap.gz", "network.pcap",
                        "telemetry.csv", "mavlink.jsonl", "commands.jsonl", "groundtruth.jsonl", "groundtruth_independent.jsonl"):
                    raw_index.append(dict(path="v2/" + str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                                          sha256=sha(path), included=False,
                                          original_summary_sha256=sha(run / "summary.json")))
            logs = run / "sitl/logs"
            if logs.is_dir():
                for path in sorted(logs.glob("*.BIN")):
                    raw_index.append(dict(path="v2/" + str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                                          sha256=sha(path), included=False))
    json_write(output / "evidence/excluded_raw_index.json", dict(schema=SCHEMA, scope="Only explicitly named completed development batches",
               hash_method="SHA-256 recomputed from original bytes; no raw data bundled", files=raw_index))
    json_write(output / "evidence/external_stack.json", dict(scope="Sanitized original run manifests; original manifest SHA retained",
               run_manifests=stack))
    json_write(output / "evidence/receipts.json", dict(schema=SCHEMA, receipts=receipts))

    write(output / "README.md", README)
    write(output / "MODEL_CARD.md", """# Model card\n\nIncluded W8 geometry actor SHA-256:\n`653f108879b43449454fffdebdd87b1a0f6968e1f4bb1f68a57fc793fba13034`.\nExact paths, shapes and hashes for BC, wind, W0 control and W8 actors are in\n[evidence/models.json](evidence/models.json). All NPZ files preserve original bytes.\n\nThe model uses 74 measured-state/history features and 4 normalized outputs.\nIt has no privileged wind-force, mass or motor-efficiency input. Upper-loop control\nis 50 Hz; the physical deployment preserves ArduPilot's native inner loops.\nOutput thrust semantics require the verified adapter; the raw action alone is\nnot an interchangeable MAVLink collective command.\n\nLineage label 111 identifies the original training family. The paired development\nexperiment's actual PPO RNG seed is 9611, critic warmup seed 9511 (source plan),\nwith 131072 critic-only steps and 2097152 PPO steps per W0/W8 arm. Initial wind\ncheckpoint is included to repeat that pair; the earlier BC dataset and all prior\ntraining archives are outside this small export.\n\nIntended use: simulation research and inspection of the measured development\nevidence. The one-environment Gazebo map and shared-control replication do not\nestablish broad environmental generalization. Nominal route geometry and a\nselected tailwind stress case fail fixed quality guards. No claim of real-aircraft\nvalidation, final confirmation, or a universal wind advantage is made.\n\nVelocity-conditioned diagnostic weights and new unfinished models are excluded.\nThe author has not selected a redistribution license for research code or weights.\n""")
    write(output / "docs/causal_evidence.md", CAUSAL)
    write(output / "docs/reproducibility.md", REPRO)
    write(output / "scripts/quickstart.py", QUICKSTART)
    write(output / "scripts/cpu_smoke.py", SMOKE)
    write(output / "scripts/verify_export.py", VERIFY)
    write(output / "requirements.txt", "# CPU inference and independent auditor imports; observed Python 3.12 environment.\nnumpy==2.5.4\ngymnasium==1.4.0\npybullet==3.2.7\npymavlink==2.4.50\n")
    write(output / "requirements-training.txt", "-r requirements.txt\ntorch==2.11.0\nstable-baselines3==2.9.0\nmatplotlib==3.11.2\n"
          + f"pandas=={metadata.version('pandas')}\nscipy=={metadata.version('scipy')}\n")
    public_artifacts = [public_pid_relative, state_relative]
    weight_allowlist = "".join("!v2/" + name + "\n" for name in models + public_artifacts)
    write(output / ".gitignore", ".venv/\nvenv/\n__pycache__/\n*.py[cod]\n.pytest_cache/\nruns/\n*.pcap\n*.pcap.gz\n*.BIN\n*.bin\n*.mp4\n*.avi\n.env\n.env.*\ncredentials*\n*.pem\n*.key\nbuild/\nv2/reference/artifacts/\n# Only the explicitly packaged small models may enter Git.\nv2/artifacts/**\n!v2/artifacts/**/\n" + weight_allowlist + "*.writing\n.DS_Store\nThumbs.db\n")
    write(output / ".gitattributes", "# Preserve exact archived source and checkpoint bytes across platforms.\n* -text\n*.npz binary\n*.zip binary\n*.png binary\n*.pdf binary\n")
    write(output / ".github/workflows/cpu.yml", "name: CPU research checks\non: [push, pull_request, workflow_dispatch]\npermissions:\n  contents: read\njobs:\n  cpu:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v7\n        with:\n          persist-credentials: false\n      - uses: actions/setup-python@v7\n        with:\n          python-version: '3.12'\n          cache: pip\n      - run: python -m pip install -r requirements.txt\n      - run: python scripts/verify_export.py\n      - run: python scripts/cpu_smoke.py\n      - run: python -I -B -m unittest discover -s v2/tests -p test_joint_research_preflight_v1.py -v\n      - run: python scripts/quickstart.py --seconds 2 --mission hover --scenario nominal\n")

    licenses = []
    for name in ("numpy", "torch", "stable-baselines3", "gymnasium", "pybullet", "pymavlink", "matplotlib", "pandas", "scipy", "pytest", "Pillow"):
        dist = metadata.distribution(name)
        info = dist.metadata
        licenses.append(dict(distribution=name, version=dist.version,
            license_expression=info.get("License-Expression"), license_metadata=info.get("License"),
            project_urls=info.get_all("Project-URL") or [], home_page=info.get("Home-page")))
        for file in dist.files or []:
            rel = str(file)
            if any(part.lower() in ("licenses", "license", "license.txt", "license.md", "copying", "copying.txt", "notice", "notice.txt") for part in Path(rel).parts) and ".dist-info/" in rel:
                src = dist.locate_file(file)
                if src.is_file() and src.stat().st_size < 500_000:
                    dest = output / "third_party" / name / Path(rel).name
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, dest)
    json_write(output / "third_party/distribution_metadata.json", licenses)
    write(output / "THIRD_PARTY_NOTICES.md", """# Third-party notices\n\nAuthor-owned project code and weights have no license grant assigned by this export.\nNo whole-project MIT license is asserted. External projects are installed separately.\n\n| Component | License recorded in installed distribution / upstream source | Use |\n|---|---|---|\n| NumPy | BSD-3-Clause plus bundled 0BSD/MIT/Zlib/CC0 components | inference, metrics |\n| PyTorch | BSD-3-Clause | optional training |\n| Stable-Baselines3 | MIT | optional PPO training |\n| Gymnasium | MIT | environment API |\n| PyBullet | Zlib | fast physical proxy |\n| pymavlink | LGPLv3 | offline auditor imports, MAVLink |\n| Matplotlib | Matplotlib license agreement (PSF-based) | optional plotting |\n| pandas | BSD-3-Clause | optional analysis |\n| SciPy | BSD-3-Clause with bundled third-party notices | optional analysis |\n| pytest | MIT | development dependency |\n| Pillow | MIT-CMU | optional images |\n| ArduPilot | GPL-3.0-or-later | external SITL; not bundled |\n| Gazebo gz-sim | Apache-2.0 | external simulator; not bundled |\n| ardupilot_gazebo | LGPL-3.0 (upstream LICENSE.md) | external bridge; not bundled |\n\nPython license values above come from installed package metadata, with exact versions\nand full available notices in [third_party](third_party). Original installed notices\nare retained byte-for-byte. These dependency licenses do not relicense this research code.\n\nExternal upstream license sources: [ArduPilot COPYING](https://github.com/ArduPilot/ardupilot/blob/master/COPYING.txt),\n[gz-sim LICENSE](https://github.com/gazebosim/gz-sim/blob/gz-sim8/LICENSE),\n[ardupilot_gazebo LICENSE.md](https://github.com/ArduPilot/ardupilot_gazebo/blob/main/LICENSE.md).\n""")

    files = []
    for path in sorted(output.rglob("*")):
        excluded_parts = {".git", ".venv", "__pycache__", ".pytest_cache", "runs"}
        if path.is_file() and path.name != "MANIFEST.json" and not (set(path.relative_to(output).parts) & excluded_parts):
            rel = str(path.relative_to(output))
            entry = dict(path=rel, bytes=path.stat().st_size, sha256=sha(path))
            entry.update(provenance.get(rel, {}))
            files.append(entry)
    json_write(output / "MANIFEST.json", dict(schema=SCHEMA, policy="Explicit allowlist; manifest excludes its own hash",
        source_changes="None; original experiments, sources and receipts preserved", files=files,
        file_count=len(files), total_bytes=sum(f["bytes"] for f in files)))
    print(json.dumps(dict(status="BUILT", output=str(output), files=len(files), bytes=sum(f["bytes"] for f in files),
                         excluded_raw_files=len(raw_index), receipt_projections=len(receipts)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "github_ready")
    parser.add_argument("--resume-export-build", action="store_true", help="Refresh only generated publication files, never experiment data")
    args = parser.parse_args()
    build(args.output.resolve(), resume=args.resume_export_build)
