# Что изменилось и что действительно проверено

Это промежуточный пакет PREP. Последняя измеренная Gazebo-модель — **v20/F0**
`f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461`.
Её [таблица v7](../results/memory20_v7/presentation.png) содержит8/8 основных
условий и88/90 проверок, но отдельные попутные finite/corner допуски84001
**FAIL**. Macro17,49797% получен в просмотренной development-среде n=1.
Новые PPO-кандидаты не наследуют её ячейки. Финальной независимой проверки нет.

| Опыт | Изменение и реальные затраты | Результат и граница вывода |
|---|---|---|
| v20/F0 | Все6 weight/bias массивов source68 сохранены побайтно;0 новых SGD. Только effective integral22:25 становится нулём при наличии собственного наблюдавшегося отрезка | Gazebo9 новых NN +11 общих наблюдений. Основной путь улучшился, отдельный попутный стал хуже source68. Это эффект обработки входа, не дополнительное обучение |
| Исходный v21 | В каждом плече131072 critic +131072 PPO controls,1024 critic +960 PPO Adam до ошибки evaluator | Ошибка metadata `sha`/`utc`. Два hover-эпизода по1500 controls сохранены. Нет24-case aggregate/quality вывода. Это инфраструктурный сбой |
| v21p1 | Коррекция metadata, проверка полных24 условий до main. Оба плеча от одного source68 ZIP, сброс всего Adam. Каждое131072 critic +262144 PPO controls,1024 +2048 Adam,3 snapshot | Полностью завершённый **отрицательный PPO-пилот**: все6 admissions FAIL. Zero FINAL `a9dc6666…` улучшил nominal на3,116% против собственного F0, но tail finite/corner ухудшились на11,064%/30,442%. В Gazebo эти обученные веса не летали |
| v22 |10 новых native эпизодов/20000 controls +4 reused/8000;0 SGD. Раздельно меняли heading, полное стартовое состояние и физическую фазу силы | Один seed не совпадает по начальному состоянию между native и Gazebo. Yaw-only перевернул wind finite-сравнение: zero был лучше raw на9,45%, стал хуже на19,99%. Full/pulse peaks0,553/0,574м превысили0,5м. Полная идентификация plant/скрытых AP-состояний не доказана |
| v23 | Один новый B от прежнего source68,131072 critic +262144 PPO/1024 +2048 Adam. Менялась тренировочная heading-страта; A использован повторно. Три29-case проверки B:87 новых эпизодов/156000 controls | Все3 combined admissions FAIL. Heading-only wind улучшился, полное deployment startup не исправлено. FINAL `c9f861bb…`: nominal gain2,671% меньше3%; tail corner хуже F0 на2,671%; full/pulse peaks0,520/0,541м и ограничения времени не пройдены. Обученный B в Gazebo не летал |
| v24 | Новая matched-route стартовая **совокупность** p/v/roll/pitch/body rates/reference-error0 при том же yaw-разбиении, source68, objective и бюджете. Контроль A — reused v23; C не начинается с trained c9f | Завершён новый C:393216 training controls/1024 critic +2048 PPO Adam;87 новых29-case эпизодов/156000 controls. Все3 original24 hard PASS, но все3 research/matched5/engineering FAIL; pool пуст, Gazebo не запускался |

Таким образом, здесь разделены три разных фактора: **conditioning без SGD**,
**новое обучение PPO**, **изменение распределения обучающих условий**. Рост
return, более долгий запуск или новая подпись checkpoint не доказывают улучшения
применимости. У v24 меняется несколько физических/reference компонентов вместе;
причинный эффект каждого компонента отдельно не устанавливается.

В v23 matched/original route назначались попеременно в каждой среде; это не ровно
50% всех физических шагов. Фактическая доля matched route-time была52,58% при
warmup и51,87% при PPO. Числа original24 и дополнительных5 условий не смешаны
для улучшения среднего. Все валидные неудачи сохранены; резервные seeds закрыты.

Для v24 полное research-admission остаётся прежним. Отдельное заранее заданное
engineering-screening может определять целесообразность **новой диагностической**
Gazebo-проверки. Оно не отменяет research FAIL и не изменяет PID/Gazebo-пороги.
Новая измеренная карта появится только после новых полётов точных весов.

Доказательства с original/export SHA:
[индекс](../evidence/v5_sources_prepare03.json),
[метрики v21p1](../results/history_v21_v24/joint_route_memory_train_metrics_v21p1_root01.json),
[restart parity](../results/history_v21_v24/review_joint_route_memory_train_v21p1_restart_parity_01.json),
[интерпретация v22](../results/history_v21_v24/route_transfer_initial_v22_interpretation_01.json),
[метрики v23](../results/history_v21_v24/joint_heading_coverage_metrics_v23_root01.json),
[semantics v24](../results/history_v21_v24/joint_initial_coverage_semantics_v24_root01.json).

Один живой экран показывает реальный процесс, но не заменяет количественную
серию. [Команды и viewers17–23](../source_archive/recording_latest/README.md)
требуют полного исходного workspace. Сеть дообучается в PyBullet; Gazebo используется
для отдельной проверки. В архивном viewer виден завершённый опыт, а не новое
обучение. Для live-видео нужен свежий запуск; его веса не наследуют старую карту.

[Почему PPO](algorithm_choice.md): метод выбран в исходной ВКР и соответствует
непрерывному интерфейсу. Это не эмпирическое сравнение PPO с SAC/TD3. Первичные
описания методов: [PPO](https://arxiv.org/abs/1707.06347),
[SAC](https://proceedings.mlr.press/v80/haarnoja18b.html),
[TD3](https://proceedings.mlr.press/v80/fujimoto18a.html).

V24 FINAL: nominal0,09972487м — новый PPO-выигрыш4,583% против собственного F0.
Full-wind finite лучше v23 на8,260%, но pulse/yaw-wind хуже на1,962%/3,408%.
Full/pulse peaks0,519652/0,526566м и времена9,64/9,66с не проходят допуски.
Это улучшение отдельных метрик, не общее исправление переноса.
[Фактические метрики](../results/history_v21_v24/joint_initial_coverage_metrics_v24_root01.json).
V25 — следующая подготовка изменения награды; её результатов в этом пакете нет.
Ни одна новая версия не заполняет ячейки F0/v7.

Самодостаточная малая v24 training capsule **не заявляется**:
[строгие зависимости](../evidence/v24_guarded_training_data_requirements.json)
требуют565 immutable файлов275486723 байта, включая242 trace/IO файла245106286
байт. Они не скопированы; SHA-проверки не обходились. См.
[воспроизведение v24](v24_reproduction_limits.md).
