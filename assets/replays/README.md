# Replays измеренных траекторий

GIF и poster PNG — точные копии финальных файлов renderer. Это анимация реальных
архивных CSV, **не screen recording Gazebo**, не новый полёт/оценка/обучение.
Legacy hover: train111/env202601, три старые архитектуры. F0 route: другая
модель и env83301. Их наблюдения не объединяются.151 frames/10fps/15.1s каждый.
Force strips показывают реальные ON/OFF, interpolation меняет только картинку.

[Source manifest](manifest.json) содержит original paths/hashes/metric scope;
[original source README](SOURCE_README.md) сохранён отдельно. Команда renderer в
нём относится к полному исходному workspace: необходимые raw CSV в лёгкий клон
не включены. Renderer review source — source_archive/media в корне пакета.
ROOT и автор renderer просмотрели posters/decoded frames; проверки metadata
приведены в evidence/media. GIF не заменяет количественную серию.
