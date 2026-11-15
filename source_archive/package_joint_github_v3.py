"""Build a new local publication snapshot. No network, training or Gazebo.

The old public export and every original experiment remain unchanged. Public
metadata projections have separate source/export hashes. CPU inference and a
small physical rollout are self-contained; full training/raw Gazebo replay are
explicitly not promised by this light bundle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "github_ready"
OUT = ROOT / "github_ready_v3"
POLICY = "artifacts/refinement/joint_v2/corner_pilot111/w8_control/best_policy.npz"
POLICY_SHA = "e0aa635603d5ec299cd526347891ef1b090e2f8ebd3879d5aa34f207a41fa781"
CURRENT_AUDIT = "audit/joint_path77_development11_v3_root01.json"
CURRENT_AUDIT_SHA = "8f51f4917382e068188230cfe86a142c26f80e95977690919098b91e181922cb"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(relative, text):
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def jswrite(relative, value):
    write(relative, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def inventory(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def project(value):
    if isinstance(value, dict):
        return {project(k): project(v) for k, v in value.items()}
    if isinstance(value, list):
        return [project(v) for v in value]
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    normalized = normalized.replace(ROOT.as_posix(), "v2")
    # Parent audit paths also occur inside descriptive labels, not only at
    # the beginning of a value. Preserve the relative filename, not the home.
    normalized = normalized.replace(ROOT.parent.as_posix(), "external_source_root")
    normalized = re.sub(r"/home/[^/:;\s]+", "[external-home]", normalized)
    normalized = re.sub(r"(?i)[a-z]:/+Users/+[^/\s]+", "[local-user]", normalized)
    if normalized.startswith(("/mnt/", "/home/")):
        if "/v2/" in normalized:
            return "v2/" + normalized.split("/v2/", 1)[1]
        return "[external-local-path]/" + normalized.rsplit("/", 1)[-1]
    return normalized


README = r'''# RL и PID: проверяемая карта применимости

PPO управляет внешним контуром квадрокоптера через коллективную тягу и угловые
скорости. В ArduPilot сохранены штатные внутренние контуры. Текущий актор получает
77 измеряемых признаков: прежние 74 признака состояния/истории и три координаты
отклонения от конечного текущего отрезка. Будущие точки, истинная сила ветра и
скрытые параметры физики на вход актора не подаются.

**Статус: исследовательский development, полная приёмка не пройдена.** В основном
наборе пройдены 7 из 8 условий. Номинальная точность пути и отдельный попутный
случай остаются неудачными. Это одна политика, seed обучения 111, просмотренная
среда 83301 и отдельный stress 84001; независимый финальный тест ещё не проведён.

![Карта применимости](results/current/presentation.png)

## Измеренный результат

Числа ниже получены в **ArduPilot SITL + Gazebo**, а не командами CPU quickstart.
В каждом условии — один полёт каждого сравниваемого метода.

| Условие | RL / PID | Результат в измеренной среде |
|---|---:|---|
| Постоянная сила, hover RMSE | 0,03124 / 0,06752 м | RL точнее |
| Порывы, hover RMSE | 0,05595 / 0,10614 м | RL точнее |
| Постоянная сила, время маршрута | 8,56 / 9,46 с | RL быстрее |
| Постоянная сила, finite 3D RMSE | 0,12474 / 0,14508 м | RL точнее |
| Порывы, время маршрута | 9,26 / 9,54 с | RL быстрее |
| Без силы, finite 3D RMSE | 0,13258 / 0,10455 м | допуск точности **не выполнен** |
| Попутная сила, finite 3D RMSE | 0,16114 / 0,09274 м | допуск точности **не выполнен** |

Исходный критерий ВКР по всем шести ветровым ячейкам теперь покрыт: отношение
средних penalized goal RMSE даёт **15,2357% улучшения при пороге 15%**. Это узкий
development-результат n=1, а не подтверждённая робастность. Этот макрокритерий не
заменяет требования к точности пути и поворотам: общий operational gate — FAIL.

В матрице **11 новых полётов**: 9 RL и 2 PID при малой силе. Ещё 7 PID и 2
wind-specialist полёта объявлены общими контролями. Итого 20 наблюдений,
не 20 новых или независимых повторов. Все 11 новых physical/force/metric audits
прошли; 9 полётов NN77 повторно проверены по **15 989 CRC-valid wire commands**.
Фиксированные проверки качества: **87/90 PASS, 3 FAIL**.

[Таблица CSV](results/current/applicability.csv), [PDF для слайда](results/current/presentation.pdf),
[все метрики](results/current/all_original_flights.csv), [все требования](results/current/fixed_quality_checks.csv),
[фазы сил](results/current/route_force_exposure.csv), [что дало результат](docs/causal_evidence.md).

## Быстрый запуск без Gazebo

Проверяемая среда — Python 3.12, Ubuntu 24.04/WSL2. Для текущего NPZ inference и
CPU rollout Torch, ArduPilot и Gazebo не нужны.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_export.py
python scripts/cpu_smoke.py
python scripts/quickstart.py --scenario force_strong --mission route --seed 84001
```

Quickstart выполняет реальные шаги PyBullet с native-rate-v3 proxy и точным
включённым актором77, пишет новые `runs/…/trace.csv`, `actor_io.npz` и `summary.json`.
Полный маршрут — 40 с, hover — 30 с. Этот запуск помечен **PYBULLET_CPU_DEVELOPMENT**;
он не воспроизводит и не подтверждает автоматически архивную таблицу Gazebo.
Проверка упаковки: 40 с, 2000 реальных control steps, пять подтверждённых целей.

## Воспроизводимость и исходники

- [MODEL_CARD](MODEL_CARD.md): точные веса, интерфейс, история и границы.
- [Воспроизводимость](docs/reproducibility.md): что запускается из этого пакета,
  что требует исходного архива и внешнего стека.
- `v2/quadrl*`: переносимое ядро CPU inference; исходные policy/physics сохраняют
  байты, metadata-only public fit имеет отдельно указанный исходный SHA.
- `source_archive/current_control/`: зафиксированные исходники обучающего опыта
  для просмотра; это не обещание самодостаточного guarded training без архива.
- `results/current/`: новая карта **только актора e0aa…**. `results/pilot32/` —
  историческая карта другой 74D политики; эти ячейки не смешиваются.
- `results/receipts/`, `evidence/`: public projections с раздельными original и
  export SHA, а также индекс исключённых входов. PCAP/BIN/видео не включены.
- `MANIFEST.json`: полный allowlist, размеры и SHA каждого файла; сам manifest
  исключает собственный hash. SHA проверяет байты, а не истинность утверждения.

Ручной Gazebo launcher проверен в исходном Windows/WSL workspace:40s полёт,
1997 NN wire commands и5 фактических маркеров; это отдельный integration QA,
не ещё один scientific replicate. Training launcher прошёл prepare-only/data
checks без запуска нового обучения. Их [архив исходников](source_archive/recording_workspace/README.md)
помечен **WORKSPACE_ONLY**: самодостаточный запуск этих скриптов из малого пакета
не обещается. Новый live HTTP monitor визуально не проверен: автоматический
запуск был отклонён с причиной `blocked by policy`; повторного старта не было.
Пользователь запускает монитор и записывает видео самостоятельно. Архивный
график нельзя выдавать за живое обучение. Сборка не запускает сервер или браузер.

## Лицензии

Лицензия авторского кода и весов владельцем пока не выбрана. Этот snapshot не
назначает MIT или другую лицензию за автора. Лицензии зависимостей включены в
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). Полные внешние проекты/бинарники не
распространяются. DOI, релизный URL и commit не вымышляются; локальный Git не имеет
remote или commit, публикация не выполнялась.

English: development-only evidence for an upper-loop PPO quadrotor controller.
Wind-hover gains and faster wind routes were measured in one viewed Gazebo
environment. Nominal path accuracy and a separate tailwind case still fail fixed
requirements. The original six-cell wind macro passes narrowly; final independent
confirmation, universal wind superiority, hardware validation and PPO optimality
are not claimed.
'''

MODEL_CARD = r'''# Model card: measured Path77 FULLCONTROL

Current policy: `v2/artifacts/refinement/joint_v2/corner_pilot111/w8_control/best_policy.npz`.
SHA-256: `e0aa635603d5ec299cd526347891ef1b090e2f8ebd3879d5aa34f207a41fa781`.
The archive is an exact copy of the measured actor, not a renamed older 74D model.
Its topology is 77 → 128 tanh → 128 tanh → 4, clipped normalized actions.
Contract: `measured-memory-finite-leg-v3.0`.

The actor receives measured state/history and a causal finite-leg correction in
current heading, normalized by 0.25 m. The previous leg start is the previous
actually observed target. No future target, phase oracle, true wind, mass or
motor effectiveness is fed to the actor. Hover extras start at zero; a final
observed leg remains during route final hold. Outputs are collective thrust and
FLU body rates; actual MAVLink use requires the measured-body-Rzz thrust adapter.
ArduPilot retains its native inner loops.

Current weights are the selected FINAL of v8 **CONTROL**, with W8 finite-segment
reward and **no effective K40 corner reward**. Both v8 arms received the same77
path inputs and initial7d41 policy/prior. New critic/PPO budgets were131072/262144
environment transitions and1024/2048 optimizer steps; three candidate snapshots.
Training RNG seeds9611/9511 are distinct from lineage label111. The initializer
was already trained: BC → wind/geometry PPO → manual six-column Γ1.15 calibration
→ learned Path77 stage. It is not learning from scratch. Earlier stages and their
causal scopes are in [causal evidence](docs/causal_evidence.md).

Actual Gazebo development: 8 primary conditions at83301 plus a strong tailwind
route at84001. Primary7/8 conditions pass; nominal finite geometry fails. Tailwind
finite geometry and corner peak fail. Original six-cell wind macro15.2357% passes
the15% criterion in this viewed domain. Operational acceptance and native nominal
finite guard remain FAIL. Reserved93001/93002/93003 were not opened.

The stronger-budget v9 and actor-column-only v11 variants did not meet the
remaining requirement and are not substituted for this policy's measured cells.
Their completed negative receipts are retained. A Soft-KL prior is a mechanism,
not a mathematical retention guarantee for full-network PPO.

CPU quickstart uses a separate native-rate-v3 PyBullet proxy with unchanged
numeric dynamics and a small observation-only adapter. It establishes portable
inference/physical stepping, not the Gazebo claims. Actual raw PCAP/BIN/world logs
are indexed but excluded; without them the included audit code cannot re-create
the physical/wire proof. No real-aircraft, fault, noise, latency, 90° yaw-step,
m/s wind, broad generalization, or PPO-versus-SAC/TD3 superiority claim is made.

Owner redistribution license remains unspecified.
'''

CAUSAL = r'''# Что действительно проверено как причина

Выигрыш оценивается метриками, а не PPO return. Условия, модель, backend,
бюджет и число кандидатов фиксируются отдельно. Последовательное улучшение
пакета не означает, что причинный вклад каждого изменения доказан в Gazebo.

| Опыт | Что сравнивалось | Что известно |
|---|---|---|
| W0/W8 | одинаковые initial, seeds,2M PPO и число кандидатов; только finite-path penalty | Matched Fast FINAL mean finite: W0 .156561 → W8 .150635м, −3.785%. Выбранный W0 сохранён раньше W8; selected comparison — отдельный результат. Не всё качество прошло. |
| Γ1.15 | ручное изменение шести XY velocity columns, без новых PPO шагов | В реальном tailwind стресс-полёте пик0.50125→0.30507м, около−39%; точность против PID всё ещё FAIL. Это **калибровка**, а не новое обучение RL. |
| blind/path77 | одинаковая77D топология и бюджет; только доступ к causal3 вместо нулей | Fast nominal finite лучше на1.82% в selected comparison, tail finite лучше около10%, но tail corners хуже23.7%. Все условия extension не пройдены. |
| v8 W8 CONTROL/K40 | одинаковый learned77 initial/prior и262144 PPO; только эффективный corner penalty | K40 проиграл выбранному CONTROL; отклонён. CONTROL улучшил finite примерно на4% от того же initial, но nominal.11м не достиг. |
| v9 больше времени | неизменный CONTROL,1M вместо262144 PPO; все13 tensors и Adam в prefix точно воспроизведены | Long FINAL не решил nominal: .123623м против short FINAL .123605м; strong/gust хуже1.62/2.65%. Selected-of9 не смешивается с selected-of3; extension отклонён. |
| v11 только384 weights | те же initial/reward/budget; менялись только actor path columns, critic полный | Hover function и36 полных hover trajectories сохранены точно. Но geometry хуже short CONTROL; все selected/matched extension checks FALSE. Ограничение параметров не дало route выигрыша. |

Текущая Gazebo карта использует **v8 FULLCONTROL FINAL e0aa…**. Новые или старые
веса не наследуют её успешные ячейки. Шесть ветровых ячеек впервые полностью
покрыты при добавлении mild PID/RL; macro15.2357% — результат измерения этой
политики и среды, а не доказательство вклада одного Path77 изменения.

PPO выбран как рабочий baseline непрерывного управления, а не как доказанно
лучший алгоритм. Его [исходная работа](https://arxiv.org/abs/1707.06347) и
[Swift](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/) — первичные ссылки
на алгоритм и связанный quadrotor пример. Равного сравнения PPO/SAC/TD3 в этом
проекте не было. Смена алгоритма сама по себе не исправляет несоответствие reward
конечной метрике. Неудачные reward/physics/curriculum варианты сохранены.

Номинальная ошибка здесь — не «RL бесполезен»: wind hover и часть wind-route
метрик лучше PID. Но это и не универсальная победа. Номинальный путь и попутный
случай ограничивают применимость; все обязательные gates соединяются через И.
'''

REPRO = r'''# Границы воспроизводимости

Из этой лёгкой копии проверяются allowlist/SHA, exact77 NPZ inference и реальные
CPU PyBullet шаги с native-rate-v3 model. `scripts/cpu_smoke.py` проверяет causal
feature semantics и полный40s маршрут; `scripts/quickstart.py` пишет новый run.
Все проектные imports должны разрешаться внутри экспортированного дерева.

Таблица Gazebo — отдельное архивное свидетельство. Public receipt projections
содержат exact-policy SHA, 11new/9shared counts, исходные input hashes и negative
gates. Они не содержат raw PCAP/BIN/world logs. Чтобы заново получить wire/physical
identity proof, нужен исходный архив и соответствующий внешний SITL/Gazebo stack.
Индекс исключённых входов и их размеры находится в `evidence/current_inputs.json`.

`source_archive/current_control/` сохраняет зафиксированные source/evidence
dependencies для просмотра. Файлы с private-local metadata публикуются как
явно помеченные projections; original SHA и export SHA разделены в MANIFEST.
Native-rate fit меняет только строки metadata: все числовые/динамические поля
равны исходным. Поэтому CPU physics не перенастраивается под пример.

Полный текущий guarded training не объявлен самодостаточным: initial7d41 SB3 ZIP,
полная baseline/evaluation evidence closure, raw archives и внешний stack могут
быть исключены из малого пакета. `evidence/training_recipe.json` содержит точные
initial/prior SHA, budgets/seeds и original relative recipe. Не обходите guards
отсутствующих доказательств. Текущая политика не начиналась с нулевых весов.

Для нового обучения действует metric-first план:13 метрик → reward/constraint →
selector → независимый judge, frozen sources/evidence/PID и fresh output. Return
или имя best_policy не означают приёмку. Резервные seeds93001..3 не используются
для разработки. Требуемая полная финальная матрица114 полётов ещё не выполнена.

Windows/WSL recording launchers включены только как WORKSPACE_ONLY text source
archive; public metadata/default paths проектируются для приватности, исходные
SHA сохранены. Совпадение registry SHA в исходном workspace не переносится на
эту public projection. На чистом клоне команды из этого архива не обещаются.
Отдельный GUI integration QA прошёл в исходном workspace; monitor проверен
по данным/синтаксису без автоматического HTTP/browser запуска. Видео записывает
пользователь. Сборка не запускает сервер, браузер, training, Gazebo, remote или
push. CPU QA не считается научным полётом.
'''

CPU_ENV = r'''"""Observation-only CPU evaluator, with no training/Torch dependency.

This packaging adaptation mirrors PathObservationMixin/PathNativeEnv exactly for
the full-horizon evaluator. Original source hashes remain in the manifest.
"""
import numpy as np
from gymnasium import spaces
from quadrl_joint.env_v3 import QuadEnvJointV3
from quadrl_path_v7.policy import FiniteLegContext


class Path77CpuEnv(QuadEnvJointV3):
    def __init__(self, **kwargs):
        self.path_context = FiniteLegContext()
        super().__init__(architecture="hybrid", **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (77,), np.float32)

    def reset(self, seed=None, options=None):
        self.path_context.reset()
        return super().reset(seed=seed, options=options)

    def observation(self, state=None):
        base = super().observation(state)
        measured = self.measured_state
        extra = self.path_context.observe(measured["position"], measured["rotation"], self.target)
        self.path_feature = extra.copy()
        return np.concatenate((base, extra)).astype(np.float32)
'''

QUICK = r'''"""Real CPU Path77 rollout; never launch Gazebo, training, HTTP or browser."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
from quadrl_path_v7.policy import NumpyPolicyPath77
from quadrl_path_v7.cpu_eval import Path77CpuEnv
from quadrl.rollout import _run_episode

POLICY = ROOT / "v2/artifacts/refinement/joint_v2/corner_pilot111/w8_control/best_policy.npz"
POLICY_SHA = "e0aa635603d5ec299cd526347891ef1b090e2f8ebd3879d5aa34f207a41fa781"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("nominal", "force_mild", "force_strong", "gust"), default="force_strong")
    parser.add_argument("--mission", choices=("hover", "route"), default="route")
    parser.add_argument("--seed", type=int, default=84001)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seconds = args.seconds if args.seconds is not None else (30. if args.mission == "hover" else 40.)
    if not 100 <= args.seed < 90000 or not 0 < seconds <= 40:
        parser.error("Use a viewed/development seed[100,90000), positive horizon<=40s")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "runs" / (stamp + "_path77")).resolve()
    if not output.is_relative_to(ROOT / "runs") or output.exists():
        parser.error("Choose a fresh output beneath runs/")
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest() == POLICY_SHA
    policy = NumpyPolicyPath77(POLICY)
    env = Path77CpuEnv(scenario=args.scenario, mission=args.mission, seconds=seconds)
    try:
        record = _run_episode(env, policy, "hybrid", True, args.scenario, args.mission, args.seed, output)
    finally:
        env.close()
    record.update(scope="PYBULLET_CPU_DEVELOPMENT_NOT_GAZEBO_VALIDATION", model="exact77_FULLCONTROL_e0aa",
        policy_sha256=POLICY_SHA, observation_size=77, no_Gazebo_or_training_started=True)
    record.pop("trace_path", None)
    record.pop("actor_io_path", None)
    (output / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(scope=record["scope"], steps=record["steps"], goals_reached=len(record["mission_events"]),
        completion_time_s=record["metrics"]["route_completion_time_s"], survived=record["metrics"]["survived"],
        output=str(output.relative_to(ROOT))), indent=2))


if __name__ == "__main__":
    main()
'''

SMOKE = r'''"""Check standalone77 inference, causal geometry and actual CPU physical steps."""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
from quadrl_path_v7.policy import FiniteLegContext, NumpyPolicyPath77
from quadrl_path_v7.cpu_eval import Path77CpuEnv


def main():
    for name in ("quadrl.policy", "quadrl_path_v7.policy", "quadrl_path_v7.cpu_eval",
                 "quadrl_joint.env_v3", "quadrl_joint.native_rate_model_v3", "quadrl.env", "reference.direct_physics"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(ROOT), name
    assert "torch" not in sys.modules, "CPU inference must not import training/Torch"
    context = FiniteLegContext()
    R = np.eye(3)
    assert np.array_equal(context.observe(np.array([0., .2, 3.]), R, [0.,0.,3.]), np.zeros(3))
    feature = context.observe(np.array([1., .2, 3.]), R, [2.,0.,3.])
    assert np.allclose(feature, [0., -.8, 0.])
    assert np.allclose(context.observe(np.array([3.,0.,3.]), R, [2.,0.,3.]), [-4.,0.,0.])
    policy_path = ROOT / "v2/artifacts/refinement/joint_v2/corner_pilot111/w8_control/best_policy.npz"
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == "e0aa635603d5ec299cd526347891ef1b090e2f8ebd3879d5aa34f207a41fa781"
    policy = NumpyPolicyPath77(policy_path)
    env = Path77CpuEnv(scenario="nominal", mission="route", seconds=40.)
    observed = 0
    try:
        obs, _ = env.reset(seed=84001)
        assert obs.shape == (77,)
        for _ in range(env.max_steps):
            action = policy.predict(obs)
            assert action.shape == (4,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (77,) and np.isfinite(obs).all() and np.isfinite(reward)
            assert not terminated, info
            observed += 1
            if truncated:
                break
        assert observed == 2000 and len(env.mission.events) == 5 and env.mission.completion_time is not None
        completion = env.mission.completion_time
    finally:
        env.close()
    print(json.dumps(dict(status="PASS", scope="SELF_CONTAINED_CPU_SOFTWARE_QA_NOT_GAZEBO",
        observations=77, actual_control_steps=observed, actual_physics_substeps=observed*8,
        goals_reached=5, route_completion_time_s=completion, training_or_Gazebo_started=False)))


if __name__ == "__main__":
    main()
'''


def workspace_recording_sources():
    """Read-only text archive of the now-frozen workspace launchers."""
    assert sha(ROOT / "development_models_v3.json") == "50c28967c84ac9259ca6614edb46b37167a6bb6c7c6298aac20bd1c02a525477"
    relatives = ("show_joint_path77.cmd", "show_joint_path77.ps1", "development_models_v3.json",
        "scripts/joint_path77_presentation_v3.py", "START_JOINT_PATH77.txt",
        "train_joint_recording_v4.cmd", "train_joint_recording_v4.ps1",
        "show_joint_training_v4.cmd", "show_joint_training_v4.ps1",
        "scripts/prepare_joint_training_recording_v4.py", "scripts/joint_training_viewer_v4.py",
        "START_TRAINING_RECORDING_V4.txt")
    rows = []
    for relative in relatives:
        source = ROOT / relative
        destination = "source_archive/recording_workspace/" + relative + ".txt"
        if source.suffix == ".json":
            jswrite(destination, project(read(source)))
        else:
            text = source.read_text(encoding="utf-8-sig")
            text = text.replace(ROOT.as_posix(), "v2").replace(ROOT.parent.as_posix(), "external_source_root")
            text = re.sub(r"/home/[^/\s'\"]+", "/opt/vkr", text)
            write(destination, text)
        rows.append(dict(path=destination, original_relative=relative, original_sha256=sha(source),
            export_sha256=sha(OUT / destination), byte_identical=source.read_bytes() == (OUT / destination).read_bytes(),
            scope="WORKSPACE_ONLY_TEXT_SOURCE_NOT_PORTABLE_LAUNCHER"))
    write("source_archive/recording_workspace/README.md", "# Recording launcher source reference\n\nThese are text copies for inspection, with original workspace filenames and SHA in the index. Public metadata/default paths are projected and `.txt` prevents accidental execution. They are **WORKSPACE_ONLY**, not runnable standalone launchers for this light export. The exact measured actor and portable CPU demo are in the main README. GUI integration succeeded in the original workspace; a new live HTTP monitor was not visually launched. The user records video manually.\n")
    jswrite("evidence/recording_workspace_sources.json", dict(status="WORKSPACE_ONLY_REFERENCE", files=rows,
        portable_or_full_training_Gazebo_launch_claim=False))
    return {row["path"]:dict(source_scope="workspace-recording-text-projection", source_relative=row["original_relative"],
        source_sha256=row["original_sha256"], byte_identical=row["byte_identical"]) for row in rows}


def build():
    assert not OUT.exists(), "New export directory required; preserve earlier snapshots"
    assert sha(ROOT / POLICY) == POLICY_SHA
    assert sha(ROOT / CURRENT_AUDIT) == CURRENT_AUDIT_SHA
    before_old = inventory(OLD)
    old_manifest = read(OLD / "MANIFEST.json")
    source_info = {}
    OUT.mkdir()
    for row in old_manifest["files"]:
        source = OLD / row["path"]
        assert sha(source) == row["sha256"]
        target = OUT / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_info[row["path"]] = dict(source_scope="prior-public-export-v2", source_relative=row["path"],
            source_sha256=row["sha256"], byte_identical=True)
    jswrite("evidence/prior_export_manifest.json", old_manifest)
    write("README.md", README)
    write("MODEL_CARD.md", MODEL_CARD)
    write("docs/causal_evidence.md", CAUSAL)
    write("docs/reproducibility.md", REPRO)
    write("v2/quadrl_path_v7/cpu_eval.py", CPU_ENV)
    write("scripts/quickstart.py", QUICK)
    write("scripts/cpu_smoke.py", SMOKE)
    old_ignore = (OLD / ".gitignore").read_text(encoding="utf-8")
    write(".gitignore", old_ignore + "\n!v2/artifacts/refinement/joint_v2/corner_pilot111/w8_control/best_policy.npz\n")
    numerical_fit = ("native_shaper", "RP_remainder", "yaw_retained", "collective_retained_tau_s", "parameter_bounds")
    for relative in ("quadrl_path_v7/__init__.py", "quadrl_path_v7/policy.py", "quadrl_joint/__init__.py",
                     "quadrl_joint/env_v3.py", "quadrl_joint/native_rate_model_v3.py", "quadrl_joint/native_rate_fit_v3.json", POLICY):
        source, destination = ROOT / relative, "v2/" + relative
        target = OUT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("native_rate_fit_v3.json"):
            original = read(source)
            public = project(original)
            for key in numerical_fit:
                assert original[key] == public[key] or key == "native_shaper"
            assert {k:v for k,v in original["native_shaper"].items() if k != "source_evidence"} == {k:v for k,v in public["native_shaper"].items() if k != "source_evidence"}
            jswrite(destination, public)
        else:
            shutil.copy2(source, target)
        source_info[destination] = dict(source_scope="original-workspace", source_relative=relative,
            source_sha256=sha(source), byte_identical=source.read_bytes() == target.read_bytes(),
            projection="Only local metadata paths projected; dynamics numbers preserved" if relative.endswith(".json") else None)
    public_models = read(OUT / "evidence/models.json")
    public_models["current_publication_policy77"] = dict(path="v2/"+POLICY, sha256=POLICY_SHA,
        observations=77, architecture="hybrid", role="DIAGNOSTIC_ONLY_FULLCONTROL", Gazebo_measured=True,
        original_training_config_sha256=sha(ROOT / "artifacts/refinement/joint_v2/corner_pilot111/w8_control/config.json"))
    jswrite("evidence/models.json", public_models)
    jswrite("v2/artifacts/refinement/joint_v2/continuation_state.json", dict(
        status="PUBLICATION_SNAPSHOT_NOT_LIVE_WORKSPACE", policy_sha256=POLICY_SHA,
        table="results/current/presentation.png", operational_gate="FAIL", development_primary_cells_passed=7,
        development_primary_cells_total=8, tailwind_stress_passed=False, original_six_cell_wind_macro_passed=True,
        original_six_cell_wind_macro_relative_improvement=.152356713427,
        final_independent_confirmation_complete=False, reserved_final_seeds_opened=False,
        current_training=None, further_experiments="Outside this immutable snapshot"))

    result_names = ("applicability.csv", "applicability.png", "applicability.pdf", "presentation.png", "presentation.pdf",
        "all_original_flights.csv", "fixed_quality_checks.csv", "route_force_exposure.csv", "metric_alignment.json",
        "operational_gate.json", "original_thesis_wind_macro.json", "README.txt")
    for name in result_names:
        relative = "reports/joint_path77_fullcontrol_v5/" + name
        destination = "results/current/" + name
        data = (ROOT / relative).read_bytes()
        if name.endswith(".json"):
            jswrite(destination, project(read(ROOT / relative)))
        else:
            target = OUT / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        source_info[destination] = dict(source_scope="original-workspace", source_relative=relative,
            source_sha256=sha(ROOT / relative), byte_identical=(OUT / destination).read_bytes() == data)
    receipts = read(OUT / "evidence/receipts.json")
    new_receipts = []
    proof_paths = (
        CURRENT_AUDIT, "artifacts/refinement/joint_v2/gazebo_path77_development11_v3/complete.json",
        "artifacts/refinement/joint_v2/gazebo_path77_development11_v3/plan.json",
        "audit/joint_path77_fullcontrol_map_v5_root01.json",
        "audit/joint_control_budget_v9_root01.json", "audit/joint_path_columns_pilot_v11_root01.json",
        "audit/joint_corner_pilot_v8_root01.json", "audit/joint_path_pilot_v7_root01.json",
        "audit/joint_velocity_conditioning7_integrity_v1.json",
        "artifacts/refinement/joint_v2/corner_pilot111/w8_control/config.json",
        "artifacts/refinement/joint_v2/corner_pilot111/w8_control/complete.json",
        "artifacts/refinement/joint_v2/corner_pilot111/w8_control/model_manifest.json",
        "artifacts/refinement/joint_v2/corner_pilot111/w8_control/selection.json",
        "artifacts/refinement/joint_v2/control_budget_v9_111/w8_control/wrapper_v9_prefix_proof.json",
        "docs/joint_path_columns_v11/probe_accounting_corrigendum_root01.json",
        "artifacts/user_runs/path77_gui_integration_v3_root01/complete.json",
        "audit/joint_training_recording_v4_root_readback_01.json",
    )
    for relative in proof_paths:
        destination = "results/receipts/" + relative.replace("/", "__")
        jswrite(destination, project(read(ROOT / relative)))
        entry = dict(original_relative=relative, original_sha256=sha(ROOT / relative),
            exported_path=destination, export_sha256=sha(OUT / destination), byte_identical=False,
            transformation="Recursive public path projection, including dictionary keys; scientific numeric values unchanged")
        new_receipts.append(entry)
        source_info[destination] = dict(source_scope="original-receipt-public-projection", source_relative=relative,
            source_sha256=entry["original_sha256"], byte_identical=False)
    jswrite("evidence/current_receipts.json", dict(schema="public-receipt-projections-current77-v3", files=new_receipts))
    # The previous index remains a historical index; link a separate current one.
    jswrite("evidence/receipts.json", dict(previous_publication_index=receipts,
        current_index="evidence/current_receipts.json", interpretation="Do not pool different policy versions"))

    frozen = read(ROOT / "docs/joint_corner_training_v8/freeze.json")
    source_archive = []
    original_sources = list(frozen["common_runtime_source_sha256"])
    original_sources.extend(("scripts/train_joint_control_budget_v9.py", "scripts/train_joint_path_columns_v11.py",
        "sim_joint/run_experiment_v3.py", "audit/verify_joint_hybrid_v3.py"))
    for relative in sorted(set(original_sources)):
        source = ROOT / relative
        if relative in frozen["common_runtime_source_sha256"]:
            assert sha(source) == frozen["common_runtime_source_sha256"][relative]
        destination = "source_archive/current_control/" + relative
        if source.suffix == ".json":
            jswrite(destination, project(read(source)))
        else:
            text = source.read_text(encoding="utf-8-sig")
            # Public source archive is a view, not a replacement frozen runtime.
            projected = re.sub(r"/home/[^/\s'\"]+", "/opt/vkr", text)
            write(destination, projected)
        identical = source.read_bytes() == (OUT / destination).read_bytes()
        source_info[destination] = dict(source_scope="frozen-source-archive", source_relative=relative,
            source_sha256=sha(source), byte_identical=identical,
            projection=None if identical else "Public metadata/default-path projection; not the original frozen runtime identity")
        source_archive.append(dict(path=destination, original_sha256=sha(source), export_sha256=sha(OUT / destination), byte_identical=identical))
    jswrite("evidence/current_source_archive.json", dict(files=source_archive,
        scope="Inspection/reference source closure. Missing raw/initial proof dependencies prevent a self-contained full training/Gazebo promise."))

    current = read(ROOT / CURRENT_AUDIT)
    index = []
    for original, digest in current["input_sha256"].items():
        candidate = Path(original)
        if not candidate.is_absolute():
            candidate = ROOT / original
        index.append(dict(source=project(original), sha256=digest, bytes=candidate.stat().st_size if candidate.exists() else None,
            included_raw=False, original_path_key_sha256=hashlib.sha256(original.encode()).hexdigest()))
    jswrite("evidence/current_inputs.json", dict(schema="excluded-current-experiment-input-index-v3", input_count=len(index),
        interpretation="Original input namespace projected for privacy; files are not automatically included and hashes alone do not replace raw evidence.", files=index))
    config = read(ROOT / "artifacts/refinement/joint_v2/corner_pilot111/w8_control/config.json")
    recipe_keys = ("schema", "variant", "initial_sha256", "prior_sha256", "fixed_kl_weight", "geometry_weight",
        "corner_reward", "observation_contract", "seed", "warmup_seed", "warmup_steps", "ppo_steps", "eval_every", "candidate_count")
    jswrite("evidence/training_recipe.json", dict(recipe={k:project(config[k]) for k in recipe_keys},
        original_driver="scripts/train_joint_corner_v8.py", original_driver_sha256=sha(ROOT / "scripts/train_joint_corner_v8.py"),
        original_plan="docs/joint_corner_plans_v8/w8_control.bound.json",
        original_plan_sha256=sha(ROOT / "docs/joint_corner_plans_v8/w8_control.bound.json"),
        included_initial7d41_checkpoint=False, full_guard_evidence_closure_included=False,
        self_contained_training_claim=False, self_contained_CPU_inference_claim=True,
        warning="Request original archive by the listed hashes; never bypass scientific guards to fill missing evidence. Do not use probe/new failed policy as initializer."))

    # Preserve historical public CI, replacing its smoke with the new77 smoke.
    workflow = (OLD / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    workflow = workflow.replace("--seconds 2 --mission hover --scenario nominal", "--seconds 2 --mission hover --scenario nominal")
    write(".github/workflows/cpu.yml", workflow)
    jswrite("evidence/publication_scope.json", dict(schema="local-publication-snapshot-v3", utc=datetime.now(timezone.utc).isoformat(),
        current_policy_sha256=POLICY_SHA, main_source_audit_sha256=CURRENT_AUDIT_SHA,
        original_development_gate="FAIL", original_thesis_wind_macro="PASS_IN_VIEWED_DEVELOPMENT_ONLY",
        active_new_experiments_included=False, GUI_training_recording_launchers_included=False,
        workspace_only_recording_text_references_included=True,
        no_remote_or_push_performed=True, no_project_license_chosen=True,
        prior_snapshot_before_sha256=sha(OLD / "MANIFEST.json"), old_snapshot_preserved=True))

    source_info.update(workspace_recording_sources())
    package_source = "source_archive/package_joint_github_v3.py"
    write(package_source, Path(__file__).read_text(encoding="utf-8"))
    source_info[package_source] = dict(source_scope="original-workspace", source_relative="scripts/package_joint_github_v3.py", source_sha256=sha(Path(__file__)), byte_identical=True)
    # Recompute every manifest row after all public projections are final.
    files = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or any(part in (".git", ".venv", "__pycache__", ".pytest_cache", "runs") for part in path.relative_to(OUT).parts):
            continue
        name = path.relative_to(OUT).as_posix()
        row = dict(path=name, bytes=path.stat().st_size, sha256=sha(path))
        if name in source_info:
            row.update(source_info[name])
            row["byte_identical"] = row["sha256"] == row["source_sha256"]
        files.append(row)
    jswrite("MANIFEST.json", dict(schema="github-ready-joint-v3.0", policy="Explicit allowlist; excludes manifest's own hash and local Git metadata", files=files))
    assert inventory(OLD) == before_old, "Old public snapshot changed"
    receipt = dict(status="PACKAGE_BUILT_PENDING_VERIFICATION", utc=datetime.now(timezone.utc).isoformat(),
        package="github_ready_v3", manifest_sha256=sha(OUT / "MANIFEST.json"), files=len(files), bytes=sum(x["bytes"] for x in files),
        policy_sha256=POLICY_SHA, current_audit_sha256=CURRENT_AUDIT_SHA, prior_export_all_files_unchanged=True,
        prior_export_file_count=len(before_old), current_source_files=len(source_archive), receipt_projections=len(new_receipts),
        numeric_fit_parameters_unchanged=True, actual_Gazebo_started=False, training_started=False, server_started=False,
        remote_or_push_performed=False, source_sha256=sha(Path(__file__)))
    target = ROOT / "audit/github_packaging_v3_build_01.json"
    assert not target.exists()
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


def refresh_public():
    """Correct only this new export's public metadata; preserve first attempt."""
    before_old = inventory(OLD)
    old_manifest = read(OUT / "MANIFEST.json")
    assert old_manifest["schema"] == "github-ready-joint-v3.0"
    for row in old_manifest["files"]:
        assert sha(OUT / row["path"]) == row["sha256"], row["path"]
    before_manifest_sha = sha(OUT / "MANIFEST.json")
    write("README.md", README)
    write("docs/causal_evidence.md", CAUSAL)
    write("docs/reproducibility.md", REPRO)
    index = read(OUT / "evidence/current_receipts.json")
    changed = []
    for row in index["files"]:
        source = ROOT / row["original_relative"]
        assert sha(source) == row["original_sha256"]
        old_sha = sha(OUT / row["exported_path"])
        jswrite(row["exported_path"], project(read(source)))
        row["export_sha256"] = sha(OUT / row["exported_path"])
        if old_sha != row["export_sha256"]:
            changed.append(dict(path=row["exported_path"], before_sha256=old_sha, after_sha256=row["export_sha256"]))
    jswrite("evidence/current_receipts.json", index)
    extra_proofs = ("artifacts/user_runs/path77_gui_integration_v3_root01/complete.json",
                    "audit/joint_training_recording_v4_root_readback_01.json")
    existing = {row["original_relative"] for row in index["files"]}
    for relative in extra_proofs:
        if relative in existing:
            continue
        source = ROOT / relative
        if relative.startswith("artifacts/user_runs"):
            assert sha(source) == "d58233f28c917d2516b2ce09fb947258e728f06f8f9b12946dc30a203efb8a4e"
        destination = "results/receipts/" + relative.replace("/", "__")
        jswrite(destination, project(read(source)))
        index["files"].append(dict(original_relative=relative, original_sha256=sha(source), exported_path=destination,
            export_sha256=sha(OUT / destination), byte_identical=False,
            transformation="Public path projection; workspace-only integration/prepare proof, not additional primary scientific flight"))
    jswrite("evidence/current_receipts.json", index)
    recording_sources = workspace_recording_sources()
    publication = read(OUT / "evidence/publication_scope.json")
    publication["workspace_only_recording_text_references_included"] = True
    publication["GUI_training_recording_launchers_included"] = False
    publication["workspace_original_GUI_QA_passed"] = True
    publication["public_portable_Gazebo_recording_claim"] = False
    jswrite("evidence/publication_scope.json", publication)
    package_source = "source_archive/package_joint_github_v3.py"
    write(package_source, Path(__file__).read_text(encoding="utf-8"))
    entries = {row["path"]:row for row in old_manifest["files"]}
    for relative, info in recording_sources.items():
        entries[relative] = info
    files = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or any(part in (".git", ".venv", "__pycache__", ".pytest_cache", "runs") for part in path.relative_to(OUT).parts) or path.relative_to(OUT).as_posix() == "MANIFEST.json":
            continue
        relative = path.relative_to(OUT).as_posix()
        row = dict(entries.get(relative, {}), path=relative, bytes=path.stat().st_size, sha256=sha(path))
        if relative == package_source:
            row.update(source_scope="original-workspace", source_relative="scripts/package_joint_github_v3.py", source_sha256=sha(Path(__file__)), byte_identical=True)
        if "source_sha256" in row:
            row["byte_identical"] = row["sha256"] == row["source_sha256"]
        files.append(row)
    jswrite("MANIFEST.json", dict(schema="github-ready-joint-v3.0", policy=old_manifest["policy"], files=files))
    assert inventory(OLD) == before_old
    receipt = dict(status="PUBLIC_PROJECTION_CORRECTED_PENDING_RECHECK", before_manifest_sha256=before_manifest_sha,
        after_manifest_sha256=sha(OUT / "MANIFEST.json"), changed_receipt_projections=changed,
        original_receipts_unchanged=True, old_public_export_unchanged=True, source_sha256=sha(Path(__file__)),
        build01_source_preserved="audit/github_packaging_v3_build01_source.py", no_new_simulation_or_training=True)
    number = 1
    path = ROOT / f"audit/github_packaging_v3_projection_correction_{number:02d}.json"
    while path.exists():
        number += 1
        path = ROOT / f"audit/github_packaging_v3_projection_correction_{number:02d}.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--refresh-public", action="store_true", help="Correct own new public metadata; preserve build01 receipt")
    args = parser.parse_args()
    if args.refresh_public:
        if args.build:
            parser.error("Choose either build or metadata refresh")
        refresh_public()
        return
    if not args.build:
        parser.print_help()
        return
    build()


if __name__ == "__main__":
    main()
