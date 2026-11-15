param(
    [ValidateSet('rl', 'pid')][string]$Controller = 'rl',
    [ValidateSet('nominal', 'force_mild', 'force_strong', 'gust')][string]$Scenario = 'force_strong',
    [ValidateSet('hover', 'route')][string]$Mission = 'route',
    [int]$Seed = 83301,
    [ValidateRange(0, 2)][int]$Instance = 0,
    [switch]$NoGui,
    [switch]$VerifyOnly,
    [switch]$PlanOnly,
    [Alias('h', '?')][switch]$Help
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($Help) {
    Write-Host @'
Запуск Gazebo для самостоятельной видеозаписи:
  .\show_joint_metric16.cmd
  .\show_joint_metric16.cmd -Controller pid
  .\show_joint_metric16.cmd -Scenario gust -Mission hover
  .\show_joint_metric16.cmd -Scenario force_mild -Mission route
  .\show_joint_metric16.cmd -Scenario nominal -Mission route

По умолчанию: RL, постоянный сильный ветер, маршрут, среда83301.
Полный полёт: маршрут40с, удержание30с. Запись окна вы делаете сами.
Модель экспериментальная; в штиле и при попутном ветре точность маршрута
пока не проходит все допуски. Проверенная основная среда:83301.

-VerifyOnly: только локальные проверки файлов, без запуска WSL/Gazebo.
-PlanOnly: показать точные команды будущего запуска, без симуляции.
-NoGui: тот же полёт без окна. -Instance0|1|2: свободная полоса симулятора.
Таблица для презентации: reports/joint_metric16_fullcontrol_v5/presentation.png
'@
    exit 0
}
if ($Seed -lt 0 -or $Seed -in @(93001,93002,93003)) {
    throw 'Требуется неотрицательный разработочный seed;93001..93003 зарезервированы.'
}
function Resolve-Path77Input([string]$Relative) {
    if ([IO.Path]::IsPathRooted($Relative)) { throw 'Ожидается путь относительно проекта.' }
    $path = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Relative))
    $prefix = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
    if (-not $path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase) -or -not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Отсутствует проверенный файл проекта: $Relative"
    }
    return $path
}
function Convert-Path77Linux([string]$Path) {
    if ($Path -notmatch '^([A-Za-z]):\\(.*)$') { throw "Неподдерживаемый путь: $Path" }
    return "/mnt/$($Matches[1].ToLowerInvariant())/$($Matches[2].Replace('\','/'))"
}
$registryPath = Resolve-Path77Input 'development_models_metric16_v1.json'
$registry = Get-Content -LiteralPath $registryPath -Encoding UTF8 -Raw | ConvertFrom-Json
if ($registry.schema -ne 'joint-metric16-presentation-v1.0' -or $registry.status -ne 'DEVELOPMENT_MEASURED_EXPERIMENTAL' -or $registry.final_applicability_confirmed -ne $false -or $registry.basis_kind -ne 'velocity_residual' -or $registry.actor_observations -ne 80) {
    throw 'Ожидается отдельная проверенная экспериментальная модель80.'
}
$anchors = @{
    'sim_joint/run_experiment_v4.py'='75302a356022501143e71b1be7f24d6cd6eeb58e0b39aa98e4f05b085852f904'
    'audit/verify_joint_hybrid_v4.py'='d054b0f9552ff0ed71c969a259c5943d12d4a38255c59faf5d141240ddc5530c'
    'artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/best_policy.npz'='68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5'
    'artifacts/gazebo/native_pid_tuning_v1/frozen_native_pid.json'='086eeea17bf39436246f4638aa0d8bbb81cabfb2e873413488ce58b741fb29bd'
}
foreach ($name in $anchors.Keys) {
    if ($registry.files_sha256.$name -ne $anchors[$name]) { throw "Изменилась привязка модели: $name" }
}
foreach ($entry in $registry.files_sha256.PSObject.Properties) {
    if ($entry.Value -notmatch '^[a-f0-9]{64}$') { throw 'Некорректная контрольная сумма.' }
    $actual = (Get-FileHash -LiteralPath (Resolve-Path77Input $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value) { throw "Изменился проверенный файл: $($entry.Name)" }
}
Write-Host 'Проверка модели и сохранённых экспериментов: PASS.'
if ($VerifyOnly) { exit 0 }
$linuxRoot = Convert-Path77Linux $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$suffix = [Guid]::NewGuid().ToString('N').Substring(0,8)
$relative = "artifacts/user_runs/${stamp}_${suffix}_metric16_${Controller}_${Scenario}_${Mission}"
$output = Convert-Path77Linux (Join-Path $PSScriptRoot $relative)
$mode = if ($PlanOnly) { '--plan-only' } else { '--run' }
$cli = @($mode,'--controller',$Controller,'--scenario',$Scenario,'--mission',$Mission,
         '--seed',"$Seed",'--instance',"$Instance",'--output',$output)
if ($NoGui) { $cli += '--no-gui' }
Write-Host "Контроллер: $Controller; условие: $Scenario; задача: $Mission. Полёт: $($registry.horizon_s.$Mission)с."
Write-Host $registry.audience_note
if ($Seed -ne 83301) { Write-Host 'Это новая демонстрационная среда; её результат не наследует выводы таблицы83301.' }
if (-not $PlanOnly) { Write-Host 'Запишите открывшееся окно Gazebo. Проверка полёта сохранится автоматически.' }
& wsl.exe -d Ubuntu-24.04 -u misha --cd $linuxRoot -- [external-home]/vkr_sim/venv-cu128/bin/python -B scripts/joint_metric16_presentation_v1.py @cli
$code = $LASTEXITCODE
if (-not $PlanOnly) { Write-Host "Файлы полёта: $relative" }
exit $code
