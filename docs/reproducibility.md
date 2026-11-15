# Границы воспроизводимости

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

Исторический source_archive/current_control сам по себе не был самодостаточным: initial7d41 SB3 ZIP,
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

## Добавление v4: переносимая training capsule

Теперь доступна отдельная [процедура воспроизведения](training_reproduction.md):
68 исходных файлов/необходимых небольших artifacts, initializer ZIP и свежая
генерация baseline/semantic proof перед official preflight. Все алгоритмы и
числа сохраняются; guard SHA привязаны к реальным projected bytes, original
SHA отдельно перечислены. Доказана точная baseline-переносимость24 эпизодов.
Baseline24, bounded semantic probes и готовый13-метричный preflight-план проверены.
Основное PPO при упаковке не запускалось.

Raw Gazebo PCAP/BIN/world archive по-прежнему исключён. Из этого добавления
нельзя получить повторное доказательство архивной физики или универсальности RL.
## Отдельная диагностическая80

`results/diagnostic80_v16/` — точная отдельная фактическая серия:9 новых RL +11
общих наблюдений, с обязательными FAIL. Ни одна её строка не дописывает77-карту.
SHA/размеры исходных1409 входов перечислены в
`evidence/diagnostic80_original_inputs.json`; raw архивы исключены. Pure inference
80 проверен по1000 архивных obs/actions с нулевой разницей и нулём новых шагов.
Скрипт `scripts/inspect_diagnostic80.py` не запускает симуляцию.
