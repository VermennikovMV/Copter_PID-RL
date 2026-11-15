# Что можно воспроизвести

| Действие | Проверенная область |
|---|---|
| CPU inference Direct111 и текущего гибрида | Самодостаточные NumPy-модули, точные веса, отдельный public inference-контракт |
| Полный архивный state→action replay | 1500 отсчётов Direct FAST и 2000 причинных отсчётов v29 PyBullet; все ошибки равны нулю |
| F0 replay | 64 настоящих encoded наблюдения Gazebo; не новая физика |
| Path77 CPU PyBullet | Историческая самодостаточная симуляция, не v29/Gazebo validation |
| Историческая training capsule v8 | Проверены baseline, ограниченный semantic probe и preflight; это не последнее обучение |
| Полное повторение PPO v28 prior0 | Фактически завершено из отдельного закрытого архива private03; веса, Adam и 87 эпизодов совпали с оригиналом |
| Новая Gazebo-проверка | Требует внешнего SITL/Gazebo и полного исследовательского контракта; не запускается переносимым CPU-экспортом |

Закрытый архив private03 уже создан: 271 точный исходный файл, 81 425 144 байта;
ZIP 56 620 048 байт. [Manifest](../evidence/research_inputs_v28_private03.manifest.json)
содержит реальные размеры и SHA. ZIP не включён в Git и не опубликован: оригинальная
metadata требует отдельного решения о раскрытии. URL для скачивания отсутствует.
[Состав и проверка](../docs/research_evidence_release.md).

После получения именно этого архива helper выполняется из корня пакета:

```bash
python scripts/research_evidence.py --manifest evidence/research_inputs_v28_private03.manifest.json --plan
python scripts/research_evidence.py --manifest evidence/research_inputs_v28_private03.manifest.json --check-archive research_inputs_v28_private03.zip
python scripts/research_evidence.py --manifest evidence/research_inputs_v28_private03.manifest.json --restore research_inputs_v28_private03.zip --destination runs/restored_prior28
python scripts/research_evidence.py --manifest evidence/research_inputs_v28_private03.manifest.json --check-root runs/restored_prior28
```

Helper использует stdlib, проверяет безопасные пути, точные размеры и SHA,
не заменяет существующий destination. При восстановлении архивный prefix удаляется:
файлы записываются непосредственно в `runs/restored_prior28`, и эта папка является
корнем восстановленного проекта. Из корня публичного пакета после восстановления:

```bash
python -m pip install -r requirements-training.txt
cd runs/restored_prior28
python -c "from pathlib import Path; Path('artifacts/refinement/joint_v2/prior_ablation_pilot_v28_111').mkdir(parents=True,exist_ok=True)"
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B scripts/joint_research_preflight_v1.py --plan docs/joint_prior_ablation_plans_v28/prior0.bound.json --run
```

Создаётся только пустой родительский каталог: папка результата `prior0` до запуска
должна отсутствовать. Для повторного запуска восстановите архив в новую папку
и используйте её вместо `runs/restored_prior28`; существующие результаты не заменяются.
Это тот же просмотренный рецепт и семена, а не независимая статистическая реплика.
Imports, source guards, preflight и реальное полное
обучение проверены отдельно. Guards и SHA-функции не ослаблены.

Полный повтор: 131072 шага обучения критика + 262144 PPO, 1024 + 2048 Adam и
87 новых проверочных эпизодов. Независимый аудит подтвердил 1171 проверку:
все 13 тензоров, полное состояние Adam, начальные данные и все 87 CSV/IO/метрики
совпали; итоговый NPZ побайтно совпал с оригиналом.
[Публичная проекция доказательства](../evidence/cold03_full_numeric_parity.public.json).
Это та же просмотренная среда и те же семена, а не независимая статистическая
реплика или новая Gazebo-серия. Более ранняя попытка private02 с недостающим
поздним JSON сохранена как инфраструктурная ошибка.

Original ZIP маршрутного эксперта не включён в лёгкий клон. Его digest в public
контракте означает происхождение, а не локальную проверку файла. Переносимый
контроллер проверяет точные NPZ, source68 и исходное доказательство обучения;
его отдельный контракт не заменяет исследовательский wire-контракт Gazebo.
Публичные обезличенные display-проекции имеют отдельные original/export SHA
и не используются вместо исходных guard-файлов.

На каждой версии сохраняются backend, веса/initializer SHA, семена, бюджет,
новые/повторно использованные полёты и валидные FAIL. Нет эмпирического
сравнения PPO/SAC/TD3 или подтверждения на реальном аппарате.
[Почему PPO](../docs/algorithm_choice.md).
