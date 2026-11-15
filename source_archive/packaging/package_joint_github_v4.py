"""Local v4 publication staging/finalization, without network or training."""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "github_ready_v3"
OUT = ROOT / "github_ready_v4"
POLICY_SHA = "e0aa635603d5ec299cd526347891ef1b090e2f8ebd3879d5aa34f207a41fa781"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write(name, data):
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str): data = data.encode("utf-8")
    path.write_bytes(data)
def jswrite(name, data): write(name, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def project(value):
    if isinstance(value, dict): return {project(k): project(v) for k, v in value.items()}
    if isinstance(value, list): return [project(v) for v in value]
    if not isinstance(value, str): return value
    value = value.replace("\\", "/").replace(ROOT.as_posix(), "v2").replace(ROOT.parent.as_posix(), "external_source_root")
    value = re.sub(r"/home/[^/:;\s]+", "[external-home]", value)
    value = re.sub(r"(?i)[a-z]:/+Users/+[^/\s]+", "[local-user]", value)
    if value.startswith(("/mnt/", "/home/")):
        return "v2/" + value.split("/v2/", 1)[1] if "/v2/" in value else "[external-local-path]/" + value.rsplit("/", 1)[-1]
    return value

NEGATIVES = {
    "v12_dense_path_weight": "audit/joint_dense_path_pilot_v12_root01.json",
    "v13_learning_rate": "audit/joint_lr_pilot_v13_root01.json",
    "v14_actual_phase": "audit/joint_route_phase_v14_offline_20261113.json",
    "v15_restricted_basis": "audit/joint_route_basis_pilot_v15_root01.json",
}

def prepare():
    assert not OUT.exists(), "Preserve existing v4 staging"
    original_manifest = read(OLD / "MANIFEST.json")
    OUT.mkdir()
    for row in original_manifest["files"]:
        source = OLD / row["path"]; assert sha(source) == row["sha256"], row["path"]
        target = OUT / row["path"]; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    jswrite("evidence/prior_export_v3_manifest.json", original_manifest)
    copied = []
    for folder, destination, names in (
        ("reports/joint_path77_trajectories_v2", "results/trajectories", ("comparison.png", "comparison.pdf", "wind_conditions.png", "wind_conditions.pdf", "source_data.json", "README.txt")),
        ("reports/joint_thesis_scope_v1", "results/thesis_scope", ("applicability.png", "applicability.pdf", "applicability.csv")),
    ):
        for name in names:
            source = ROOT / folder / name; target = destination + "/" + name
            write(target, source.read_bytes())
            copied.append(dict(path=target, original_relative=folder + "/" + name, original_sha256=sha(source), byte_identical=True))
    projections = []
    for label, relative in NEGATIVES.items():
        target = "results/diagnostics/" + label + ".json"
        jswrite(target, project(read(ROOT / relative)))
        projections.append(dict(path=target, original_relative=relative, original_sha256=sha(ROOT / relative),
            export_sha256=sha(OUT / target), transformation="Public path projection only; all numeric values unchanged"))
    jswrite("evidence/v4_added_sources.json", dict(exact_copies=copied, receipt_projections=projections))
    text = (OUT / "README.md").read_text(encoding="utf-8")
    text = text.replace("Исходный критерий ВКР по всем шести ветровым ячейкам теперь покрыт:", "Согласованный практический критерий по шести ветровым ячейкам покрыт:")
    text = text.replace("## Быстрый запуск без Gazebo", "## Архитектура\n\n```mermaid\nflowchart LR\n  M[\"Измерения и текущая цель\"] --> NN[\"PPO: 77 признаков\"]\n  NN -->|\"Общая тяга и угловые скорости\"| P[\"Внутренний rate PID ArduCopter\"]\n  P --> R[\"Моторы и Gazebo\"]\n  R --> M\n```\n\nЭто **гибрид**. Взлёт и посадку выполняет штатный контроллер; сеть работает\nна измерительном участке. End-to-end с прямыми командами моторам — другая модель.\n\n[Траектории и ограничения](results/trajectories/comparison.png) ·\n[Ветровые условия](results/trajectories/wind_conditions.png) ·\n[Область проверки по ВКР](results/thesis_scope/applicability.png) ·\n[Завершённые отрицательные опыты](docs/experiments.md).\n\n7/8 означает выполнение инженерных допусков, **не семь побед над PID**.\nПорог15% и новые геометрические допуски отдельно зафиксированы в протоколе;\nв исходных слайдах ВКР перечислены пять метрик без этих числовых порогов.\n\n## Быстрый запуск без Gazebo")
    write("README.md", text)
    write("docs/experiments.md", """# Последующие опыты: результаты не скрыты

Во всех строках ниже — PyBullet/native proxy или чтение архивов, а не новые
подтверждения Gazebo. Текущая измеренная политика e0aa остаётся отдельной.

| Опыт | Что меняли | Итог |
|---|---|---|
| v12 | Вес штрафа пути W8→W80 | Все3 кандидата не прошли допуск пути без ветра |
| v13 | Общий learning rate3e-5→3e-4 | Все3 не прошли; nominal ошибка выбранного хуже на3,39% |
| v14 | Разобрали10 настоящих Gazebo маршрутов | Гипотеза нехватки velocity-признака не подтвердилась; новых полётов нет |
| v15 | Парное обучение384 новых весов на разных производных координатах | Все6 кандидатов не прошли nominal finite RMSE0,11м; попутный случай у B хуже выбранного A |

v15:192 оценочных эпизода включают144 candidate и48 initial baseline; это не
192 независимых обучения или Gazebo-полёта. Оба плеча получили131072 critic и
262144 PPO transitions, по1024/2048 реальных optimizer steps. Все valid failures
сохранены. У B tail finite хуже selectedA на1,63%, tail corner —на26,21%.
Сравнение с A того же checkpoint также сохранено; checkpoint выбирал прежний
quality judge, а не величина парного выигрыша. Увеличение бюджета не разрешалось.

Числовые первичные receipts и original/export SHA: [индекс](../evidence/v4_added_sources.json).
Прогресс незавершённых опытов не является улучшением модели. v16 будет добавлен
после сообщения основного исследовательского процесса; исходный snapshot не выдаёт
подготовку к обучению за завершённый результат.
""")
    jswrite("evidence/publication_scope_v4.json", dict(status="LOCAL_DRAFT_WAIT_LATEST_RESULT", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        current_measured_actor_sha256=POLICY_SHA, actual_Gazebo_acceptance=False, remote_created=False, pushed=False,
        current_result_unchanged=True, v16_result="PENDING_ROOT_RESULT", original_v3_manifest_sha256=sha(OLD / "MANIFEST.json")))
    (OUT / "runs").mkdir(exist_ok=True)
    subprocess.run(["git", "init", "--initial-branch=main", str(OUT)], check=True, capture_output=True)
    print(json.dumps(dict(status="STAGED_NOT_FROZEN", output=str(OUT), prior_files=len(original_manifest["files"]), extra_receipts=len(projections))))

def finalize():
    assert OUT.exists() and not (OUT / "MANIFEST.json").exists(), "First immutable v4 finalize only"
    scope = read(OUT / "evidence/publication_scope_v4.json")
    assert scope["v16_result"] != "PENDING_ROOT_RESULT", "Wait for root's actual v16 decision"
    prior = {row["path"]: row for row in read(OLD / "MANIFEST.json")["files"]}
    files = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or any(part in (".git", ".venv", "__pycache__", ".pytest_cache", "runs") for part in path.relative_to(OUT).parts): continue
        name = path.relative_to(OUT).as_posix()
        row = dict(path=name, bytes=path.stat().st_size, sha256=sha(path))
        if name in prior:
            row.update(prior_export_sha256=prior[name]["sha256"], byte_identical_to_v3=sha(path) == prior[name]["sha256"], prior_provenance=prior[name])
        files.append(row)
    jswrite("MANIFEST.json", dict(schema="github-ready-joint-v4.0", policy="Explicit allowlist; excludes its own hash, runtime runs and Git metadata", files=files))
    print(json.dumps(dict(status="FINALIZED_LOCAL_ONLY", files=len(files), bytes=sum(row["bytes"] for row in files), manifest_sha256=sha(OUT / "MANIFEST.json"))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(); assert args.prepare != args.finalize
    prepare() if args.prepare else finalize()
