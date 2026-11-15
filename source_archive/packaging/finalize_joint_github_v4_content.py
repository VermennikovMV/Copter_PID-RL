"""Add the completed, separate diagnostic80 evidence to the new local export only."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'github_ready_v4'
spec = importlib.util.spec_from_file_location('v4pack', ROOT / 'scripts/package_joint_github_v4.py')
pack = importlib.util.module_from_spec(spec); spec.loader.exec_module(pack)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name, data):
    p = OUT / name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data, encoding='utf-8')
def js(name, data): write(name, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def nonstring_leaves(value):
    if isinstance(value, dict): return [nonstring_leaves(v) for v in value.values()]
    if isinstance(value, list): return [nonstring_leaves(v) for v in value]
    return None if isinstance(value, str) else (type(value).__name__, value)

assert not (OUT / 'MANIFEST.json').exists(), 'Finish content before immutable inventory'
assert not (OUT / 'evidence/diagnostic80_sources.json').exists(), 'Fresh content step only'
whole = ROOT / 'audit/joint_route80_development9_v5_root01.json'
assert sha(whole) == '5b917f1b2ce55025878f0e7e4b8c62a95852c3a1652d6def70ad9216ad2d5b74'
table_audit = ROOT / 'audit/joint_metric_cost16_gazebo_v6_table_root01.json'
assert sha(table_audit) == '589295ff9bc9a64aaa20372ddd03552641324adc9f9ae8b218038a2004c65698'
visual_audit = ROOT / 'audit/joint_metric_cost16_gazebo_v6_visual_root01.json'
assert sha(visual_audit) == 'dba2f8c8c7a5229e56012ac3548d16282949c1900e52bfb5cb2e84b2c1fbacea'
table = read(table_audit)
provenance = []
def copy(relative, target):
    p = ROOT / relative; q = OUT / target; q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q)
    provenance.append(dict(path=target, original_relative=relative, original_sha256=sha(p), export_sha256=sha(q), transformation='Exact bytes'))
def projection(relative, target):
    p = ROOT / relative; value = read(p); projected = pack.project(value)
    assert nonstring_leaves(value) == nonstring_leaves(projected), relative
    js(target, projected)
    provenance.append(dict(path=target, original_relative=relative, original_sha256=sha(p), export_sha256=sha(OUT / target), transformation='Public path strings only; recursive numeric/bool/null invariance checked'))

folder = 'reports/joint_metric_cost16_gazebo_v6/'
for name, digest in table['output_sha256'].items():
    assert sha(ROOT / folder / name) == digest, name
    destination = 'results/diagnostic80_v16/' + name
    if name.endswith('.json'): projection(folder + name, destination)
    else: copy(folder + name, destination)
for relative, target in (
    ('audit/joint_route80_development9_v5_root01.json', 'results/diagnostic80_v16/whole_integrity_audit.json'),
    ('audit/joint_metric_cost16_gazebo_v6_table_root01.json', 'results/diagnostic80_v16/table_audit.json'),
    ('audit/joint_metric_cost16_gazebo_v6_visual_root01.json', 'results/diagnostic80_v16/visual_readback_audit.json'),
    ('audit/github_packaging_v4_prepare_qa_01.json', 'evidence/capsule_prepare_qa.json'),
    ('audit/github_packaging_v4_preflight_tests_01.json', 'evidence/preflight_tests_qa.json'),
    ('audit/github_packaging_v4_diagnostic80_qa_01.json', 'evidence/diagnostic80_archive_replay_qa.json'),
): projection(relative, target)
for relative in (
    'models/diagnostic80_metric_cost16.npz',
):
    assert sha(OUT / relative) == '68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5'
provenance.append(dict(path='models/diagnostic80_metric_cost16.npz', original_relative='artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/checkpoint_131072.npz', original_sha256=sha(OUT / 'models/diagnostic80_metric_cost16.npz'), export_sha256=sha(OUT / 'models/diagnostic80_metric_cost16.npz'), transformation='Exact selected diagnostic weights; not promoted'))
for relative in ('quadrl_route_basis_v14/__init__.py', 'quadrl_route_basis_v14/policy.py'):
    assert sha(ROOT / relative) == sha(OUT / 'v2' / relative)
    provenance.append(dict(path='v2/' + relative, original_relative=relative, original_sha256=sha(ROOT / relative), export_sha256=sha(OUT / 'v2' / relative), transformation='Exact bytes; archive replay verified'))
for relative in (
    'scripts/train_joint_metric_cost_v16.py', 'scripts/run_joint_metric_cost_pair_v16.py',
    'scripts/build_joint_metric_cost16_gazebo_v6.py', 'audit/audit_joint_metric_cost_pilot_v16.py',
    'quadrl_metric_cost_v16/env.py', 'quadrl_metric_cost_v16/reward.py', 'quadrl_metric_cost_v16/__init__.py',
): copy(relative, 'source_archive/diagnostic80/' + relative)
write('source_archive/diagnostic80/README.md', '''# WORKSPACE_ONLY source evidence

Exact original v16 reward/environment/trainer, runner and audit/table-builder
sources are supplied for review of what produced the diagnostic result. Their
original workspace evidence/import closure and external Gazebo stack are not
included here. These files are not a self-contained v16 training command.

The small portable guarded capsule reproduces the earlier v8 preparation only;
the 80-input policy has a separate pure inference entry point:
`python scripts/inspect_diagnostic80.py`. It is not the default 77-input actor.
''')
inputs=[]
for name,digest in read(whole)['input_sha256'].items():
    p=Path(name)
    inputs.append(dict(original_path=pack.project(name), original_sha256=digest, bytes=p.stat().st_size if p.is_file() else None))
js('evidence/diagnostic80_original_inputs.json', dict(scope='Original workspace input index; raw PCAP/BIN/world logs excluded; this index does not recreate physical evidence', source_whole_audit_sha256=sha(whole), input_count=len(inputs), inputs=inputs))
js('evidence/diagnostic80_sources.json', dict(default77_unchanged=True, diagnostic80_promoted=False, separate_models_and_observations=True, files=provenance))

readme=(OUT/'README.md').read_text(encoding='utf-8')
anchor='## Архитектура'
assert anchor in readme
readme=readme.replace(anchor, '''## Последняя диагностическая версия80

Отдельная [измеренная таблица v16](results/diagnostic80_v16/presentation.png),
[CSV](results/diagnostic80_v16/applicability.csv) и
[сравнение с прежней77](results/diagnostic80_v16/comparison_to_previous77.csv)
содержат **9 новых RL-полётов +11 общих наблюдений**, а не20 независимых повторов.
При сильной силе finite RMSE ниже PID на14,47%, угловой peak —на25,84%, время —на9,51%.
Относительно77 улучшения finite RMSE основной среды составили0,27–0,86%;
в отдельном попутном случае —6,31%, peak ошибки —16,23%, время9,54→9,42с.
Но попутная finite ошибка всё ещё **хуже PID на62,80%**, номинальный маршрут и
попутные допуски не пройдены. Native-проверку не прошли все6 кандидатов.
Macro15,3037% при n=1 не меняет общий FAIL. **Модель80 не принята и не заменяет77**.

Контроль A в Gazebo не летал; это описательное сравнение точных политик, не
доказательство эффекта одной награды. [Весá и ограничения80](models/DIAGNOSTIC80_CARD.md),
[целостность фактических полётов](results/diagnostic80_v16/whole_integrity_audit.json),
[происхождение файлов](evidence/diagnostic80_sources.json).

''' + anchor)
write('README.md',readme)
experiments=(OUT/'docs/experiments.md').read_text(encoding='utf-8')
experiments=experiments.replace('Во всех строках ниже — PyBullet/native proxy или чтение архивов, а не новые\nподтверждения Gazebo.', 'v12–v15 ниже — PyBullet/native proxy или чтение архивов. Для v16 дополнительно\nвыполнены реальные диагностические Gazebo-полёты, описанные отдельно в конце.')
start=experiments.index('Серия Gazebo80 отдельно запущена')
experiments=experiments[:start]+'''Серия Gazebo80 завершена:9 новых RL-полётов,9 общих PID и2 общих retention,
20 наблюдений,15991 NN wire-команда,1409 входных SHA,10 маршрутов/30 окон поворотов.
Независимый аудит целостности PASS; научное качество и общий operational gate FAIL.
Основная среда7/8;87/90 проверок. Nominal finite0,131637м превышает0,115001м;
tail84001 finite0,150973м превышает0,102738м, corner0,192377м превышает0,172821м.
Ветровой macro15,3037% — узкий development n=1. Прежняя77-модель сохранена.

В сильной силе относительно PID finite RMSE ниже14,474%, mean corner peak ниже25,841%,
время ниже9,514%; при порывах finite RMSE ниже7,331%. В попутном случае finite
ошибка выше PID на62,795%. Относительно прежней77 попутный finite RMSE лучше6,311%,
active peak —16,232%; mean corner peak лучше лишь0,170%. Это разные метрики.
Парный контроль A не летал в Gazebo, поэтому перенесённый эффект только reward
bundle не доказан. Между версиями также изменены gamma/конечный горизонт и базис.

[Отдельная таблица80](../results/diagnostic80_v16/presentation.png),
[все исходные метрики20 наблюдений](../results/diagnostic80_v16/all_original_flights.csv),
[сравнение политик](../results/diagnostic80_v16/comparison_to_previous77.csv).
'''
write('docs/experiments.md', experiments)
with (OUT/'models/DIAGNOSTIC80_CARD.md').open('a',encoding='utf-8') as h:
    h.write('''\nФактическая Gazebo серия завершена:9 новых RL +11 общих наблюдений, n=1.
Целостность PASS;7/8 основных условий,87/90 проверок; nominal и tail84001 FAIL.
Macro15,3037% не отменяет эти провалы. Tail finite лучше прежней77 на6,311%,
но хуже PID на62,795%. Никакого promotion не выполнено.
[Таблица и ограничения](../results/diagnostic80_v16/presentation.png).
''')
with (OUT/'docs/reproducibility.md').open('a',encoding='utf-8') as h:
    h.write('''\n## Отдельная диагностическая80

`results/diagnostic80_v16/` — точная отдельная фактическая серия:9 новых RL +11
общих наблюдений, с обязательными FAIL. Ни одна её строка не дописывает77-карту.
SHA/размеры исходных1409 входов перечислены в
`evidence/diagnostic80_original_inputs.json`; raw архивы исключены. Pure inference
80 проверен по1000 архивных obs/actions с нулевой разницей и нулём новых шагов.
Скрипт `scripts/inspect_diagnostic80.py` не запускает симуляцию.
''')
scope=read(OUT/'evidence/publication_scope_v4.json')
scope.update(status='LOCAL_REVIEWABLE_COMPLETE_NOT_PUBLISHED',v16_result='NATIVE_AND_GAZEBO_DIAGNOSTIC_COMPLETE_NOT_PROMOTED',
    v16_Gazebo_integrity_audit_original_sha256=sha(whole),v16_table_audit_original_sha256=sha(table_audit),
    diagnostic80_policy_sha256=sha(OUT/'models/diagnostic80_metric_cost16.npz'),diagnostic80_new_flights=9,
    diagnostic80_shared_observations=11,diagnostic80_overall_quality='FAIL',diagnostic80_promoted=False,
    actual_Gazebo_acceptance=False,main_model_and_table_remain_77=True)
js('evidence/publication_scope_v4.json',scope)
copy('scripts/finalize_joint_github_v4_content.py','source_archive/packaging/finalize_joint_github_v4_content.py')
print(json.dumps(dict(status='CONTENT_FINAL_ACTUAL_DIAGNOSTIC_COMPLETE',provenance_files=len(provenance),original_inputs=len(inputs),default77_unchanged=True,remote=False)))
