[CmdletBinding()]
param(
    [string]$RunRoot = 'artifacts/refinement/joint_v2/critic_calibration_pilot_v18_111',
    [switch]$VerifyOnly,
    [switch]$PlanOnly,
    [ValidateRange(0,300)][int]$QaSeconds = 0,
    [string]$SnapshotPath = ''
)

# Read-only desktop monitor. No training, simulator, web server or process control.
$ErrorActionPreference = 'Stop'
$script:ProjectRoot = $PSScriptRoot
$script:ReadWarnings = New-Object 'System.Collections.Generic.List[string]'
$script:Movement = @{}
$script:Culture = [Globalization.CultureInfo]::InvariantCulture
Add-Type -AssemblyName System.Drawing
$script:Background = [Drawing.Color]::FromArgb(245,247,251)
if (-not [IO.Path]::IsPathRooted($RunRoot)) { $RunRoot = Join-Path $PSScriptRoot $RunRoot }
$RunRoot = [IO.Path]::GetFullPath($RunRoot)
if (-not [string]::IsNullOrWhiteSpace($SnapshotPath)) {
    if ($QaSeconds -lt 6) { throw 'SnapshotPath requires QaSeconds >= 6 for actual refreshes.' }
    if (-not [IO.Path]::IsPathRooted($SnapshotPath)) { $SnapshotPath=Join-Path $PSScriptRoot $SnapshotPath }
    $SnapshotPath=[IO.Path]::GetFullPath($SnapshotPath)
    $allowedAudit=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'audit'))+[IO.Path]::DirectorySeparatorChar
    if (-not $SnapshotPath.StartsWith($allowedAudit,[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($SnapshotPath) -ne '.png') { throw 'Snapshot must be a PNG within this project audit directory.' }
    if ([IO.File]::Exists($SnapshotPath)) { throw 'Preserve the existing QA image; choose a new name.' }
}

function Field($Object, [string]$Name) {
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -ne $property) { return $property.Value }
    return $null
}

function Shared-Text([string]$Path) {
    if (-not [IO.File]::Exists($Path)) { return $null }
    $stream = $null; $reader = $null
    try {
        $stream = New-Object IO.FileStream($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
        $reader = New-Object IO.StreamReader($stream,[Text.Encoding]::UTF8,$true)
        return $reader.ReadToEnd()
    } catch {
        [void]$script:ReadWarnings.Add(([IO.Path]::GetFileName($Path) + ': файл временно недоступен'))
        return $null
    } finally {
        if ($null -ne $reader) { $reader.Dispose() } elseif ($null -ne $stream) { $stream.Dispose() }
    }
}

function Shared-Json([string]$Path) {
    $raw = Shared-Text $Path
    if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
    try { return ConvertFrom-Json -InputObject $raw } catch {
        [void]$script:ReadWarnings.Add(([IO.Path]::GetFileName($Path) + ': запись JSON ещё не завершена'))
        return $null
    }
}

function Shared-Csv([string]$Path) {
    $raw = Shared-Text $Path
    if ([string]::IsNullOrWhiteSpace($raw)) { return @() }
    # Ignore a concurrently appended incomplete last line, never fill it.
    if (-not $raw.EndsWith("`n")) {
        $last = $raw.LastIndexOf("`n")
        if ($last -lt 0) { return @() }
        $raw = $raw.Substring(0,$last+1)
    }
    try { return @(ConvertFrom-Csv -InputObject $raw) } catch {
        [void]$script:ReadWarnings.Add(([IO.Path]::GetFileName($Path) + ': неполная CSV, повтор чтения'))
        return @()
    }
}

function Number($Value) {
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) { return $null }
    $number = 0.0
    if ([double]::TryParse([string]$Value,[Globalization.NumberStyles]::Float,$script:Culture,[ref]$number) -and -not [double]::IsNaN($number) -and -not [double]::IsInfinity($number)) { return $number }
    return $null
}

function Maximum($A,$B) {
    if ($null -eq $A) { return $B }; if ($null -eq $B) { return $A }
    return [Math]::Max([double]$A,[double]$B)
}

function Format-Count($Value) {
    if ($null -eq $Value) { return '—' }
    return ([long]$Value).ToString('N0',$script:Culture).Replace(',',' ')
}

function Format-Metric($Value) {
    if ($null -eq $Value) { return '—' }
    return ([double]$Value).ToString('F4',$script:Culture)
}

function Guard-Counters([string]$Path) {
    $raw = Shared-Text $Path
    if ($null -eq $raw) { return $null }
    $counts = @{critic=0L;ppo=0L}; $steps = @{critic=$null;ppo=$null}; $lastRecord = $null; $rows = 0
    $lines = $raw -split "`n"
    if (-not $raw.EndsWith("`n")) { $lines = $lines | Select-Object -SkipLast 1 }
    foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        try {
            $entry = ConvertFrom-Json -InputObject $line
            $stage = [string](Field $entry 'stage')
            if ($stage -notin @('critic','ppo')) { throw 'Unknown guard stage' }
            $actual = Number (Field $entry 'actual_stage_update')
            if ($null -eq $actual -or $actual -ne $counts[$stage]+1) { throw 'Guard sequence is not contiguous' }
            $counts[$stage] = [long]$actual
            $steps[$stage] = Maximum $steps[$stage] (Number (Field $entry 'environment_steps'))
            $lastRecord = $entry; $rows++
        } catch {
            [void]$script:ReadWarnings.Add('actual_gradient_norms.jsonl: некорректная последовательность; счётчик не показывается')
            return $null
        }
    }
    return [PSCustomObject]@{critic=$counts.critic;ppo=$counts.ppo;critic_env=$steps.critic;ppo_env=$steps.ppo;rows=$rows;last=$lastRecord}
}

function Arm-Paths {
    if ([IO.File]::Exists((Join-Path $RunRoot 'phase.json'))) { return @($RunRoot) }
    $names = @('critic_lr3e5','critic_lr3e4')
    if ([IO.Directory]::Exists((Join-Path $RunRoot 'limited_actor')) -or [IO.Directory]::Exists((Join-Path $RunRoot 'full_actor'))) { $names = @('limited_actor','full_actor') }
    if ([IO.Directory]::Exists((Join-Path $RunRoot 'w8_control')) -or [IO.Directory]::Exists((Join-Path $RunRoot 'metric_cost'))) { $names = @('w8_control','metric_cost') }
    return @($names | ForEach-Object { Join-Path $RunRoot $_ })
}

function Read-Arm([string]$Path) {
    $name = [IO.Path]::GetFileName($Path)
    $phase = Shared-Json (Join-Path $Path 'phase.json')
    $config = Shared-Json (Join-Path $Path 'config.json')
    $complete = Shared-Json (Join-Path $Path 'complete.json')
    $errorRecord = Shared-Json (Join-Path $Path 'error.json')
    $warm = Shared-Json (Join-Path $Path 'warmup.json')
    $selection = Shared-Json (Join-Path $Path 'selection.json')
    $ppoRows = @(Shared-Csv (Join-Path $Path 'progress.csv'))
    $warmRows = @(Shared-Csv (Join-Path $Path 'warmup/progress.csv'))
    $guards = Guard-Counters (Join-Path $Path 'actual_gradient_norms.jsonl')
    $stage = [string](Field $phase 'phase')
    $criticEnv = $null; $ppoEnv = $null; $criticUpdates = $null; $ppoUpdates = $null; $counterSource = 'нет записи реальных Adam steps'
    if ($warmRows.Count) { $criticEnv = Number (Field $warmRows[-1] 'time/total_timesteps') }
    if ($ppoRows.Count) { $ppoEnv = Number (Field $ppoRows[-1] 'time/total_timesteps') }
    if ($stage -eq 'critic') { $criticEnv = Maximum $criticEnv (Number (Field $phase 'environment_steps')) }
    if ($stage -in @('ppo','verification','complete')) { $ppoEnv = Maximum $ppoEnv (Number (Field $phase 'environment_steps')) }
    if ($null -ne $guards) {
        $criticEnv = Maximum $criticEnv $guards.critic_env; $ppoEnv = Maximum $ppoEnv $guards.ppo_env
        $criticUpdates = $guards.critic; $ppoUpdates = $guards.ppo; $counterSource = 'actual_gradient_norms.jsonl: непрерывная последовательность реальных optimizer updates'
    } elseif ($null -ne $complete) {
        $actual = Field (Field $complete 'optimizer') 'actual_optimizer_steps'
        $criticUpdates = Number (Field $actual 'critic'); $ppoUpdates = Number (Field $actual 'ppo')
        $counterSource = 'complete.json → optimizer.actual_optimizer_steps (завершённый архив)'
    } elseif ($null -ne $warm -and -not [IO.File]::Exists((Join-Path $Path 'actual_gradient_norms.jsonl'))) {
        $actual = Field $warm 'actual_optimizer_steps'
        $criticUpdates = Number (Field $actual 'critic'); $ppoUpdates = Number (Field $actual 'ppo')
        $counterSource = 'warmup.json → actual_optimizer_steps (только сохранённое состояние)'
    }
    if ($null -ne $complete) {
        $criticEnv = Maximum $criticEnv (Number (Field $complete 'critic_transitions'))
        $ppoEnv = Maximum $ppoEnv (Number (Field $complete 'ppo_transitions'))
    }
    $evaluations = @()
    foreach ($label in @('checkpoint_131072','checkpoint_262144','final')) {
        $evaluation = Shared-Json (Join-Path $Path ($label + '_validation.json'))
        if ($null -eq $evaluation) { continue }
        $cells = Field (Field $evaluation 'aggregate') 'cells'
        $tail = $null
        foreach ($episode in @(Field $evaluation 'episodes')) {
            if ((Field $episode 'scenario') -eq 'force_strong' -and (Field $episode 'mission') -eq 'route' -and (Field $episode 'environment_seed') -eq 84001) {
                $tail = Number (Field (Field $episode 'geometry') 'segment_rmse_m')
            }
        }
        $evaluations += [PSCustomObject]@{
            checkpoint=$label;timesteps=(Number (Field $evaluation 'timesteps'))
            nominal=(Number (Field (Field $cells 'nominal/route') 'segment_rmse_m'))
            strong=(Number (Field (Field $cells 'force_strong/route') 'segment_rmse_m'))
            gust=(Number (Field (Field $cells 'gust/route') 'segment_rmse_m'))
            tail84001=$tail;guards_pass=(Field $evaluation 'fixed_development_guards_pass')
            policy_sha256=(Field $evaluation 'policy_sha256');checkpoint_timing=(Field $evaluation 'checkpoint_timing')
        }
    }
    $points = @()
    foreach ($row in $ppoRows) {
        $x = Number (Field $row 'time/total_timesteps'); $y = Number (Field $row 'rollout/ep_rew_mean')
        if ($null -ne $x -and $null -ne $y) { $points += [PSCustomObject]@{x=$x;y=$y} }
    }
    $heartbeat = Field $phase 'heartbeat_utc'; $heartbeatAge = $null
    if ($null -ne $heartbeat) {
        try { $heartbeatAge = ([DateTimeOffset]::UtcNow-[DateTimeOffset]::Parse([string]$heartbeat,$script:Culture)).TotalSeconds } catch {}
    }
    $signature = @($stage,$criticEnv,$ppoEnv,$criticUpdates,$ppoUpdates,(Field $phase 'completed_cases'),$points.Count,$evaluations.Count) -join '|'
    if (-not $script:Movement.ContainsKey($Path)) { $script:Movement[$Path] = @{signature=$signature;changed=$null} }
    elseif ($script:Movement[$Path].signature -ne $signature) { $script:Movement[$Path] = @{signature=$signature;changed=[DateTimeOffset]::UtcNow} }
    $recentMovement = $null -ne $script:Movement[$Path].changed -and ([DateTimeOffset]::UtcNow-$script:Movement[$Path].changed).TotalSeconds -lt 15
    $fresh = $null -ne $heartbeatAge -and $heartbeatAge -ge -3 -and $heartbeatAge -lt 20
    $state = 'ОЖИДАНИЕ ФАЙЛОВ'
    if ($null -ne $complete -or $stage -eq 'complete') { $state = 'COMPLETE · сохранённый результат' }
    elseif ($null -ne $errorRecord -or $stage -eq 'error') { $state = 'ERROR · проверьте error.json' }
    elseif ($null -ne $phase) {
        if ($recentMovement -and $fresh) { $state = 'ОБНОВЛЯЕТСЯ · движение счётчиков подтверждено' }
        elseif ($fresh) { $state = 'СВЕЖИЙ HEARTBEAT · ждём движения счётчиков' }
        else { $state = 'АРХИВ / НЕТ СВЕЖИХ ИЗМЕНЕНИЙ' }
    }
    $hash = Field $selection 'policy_sha256'
    if (-not $hash -and $evaluations.Count) { $hash = $evaluations[-1].policy_sha256 }
    return [PSCustomObject]@{
        name=$name;path=$Path;stage=$stage;state=$state;heartbeat_utc=$heartbeat;heartbeat_age_s=$heartbeatAge
        observed_counter_movement=$recentMovement;critic_env=$criticEnv;ppo_env=$ppoEnv
        critic_updates=$criticUpdates;ppo_updates=$ppoUpdates;counter_source=$counterSource
        guard_rows=(Field $guards 'rows');last_gradient=(Field $guards 'last')
        budget=(Field $config 'budget');ppo_reward_points=$points;evaluations=$evaluations
        evaluated_policy_sha256=$hash;initial_sha256=(Field $config 'initial_sha256')
        completed_cases=(Field $phase 'completed_cases');total_cases=(Field $phase 'total_cases')
    }
}

function Snapshot {
    $script:ReadWarnings.Clear()
    return @((Arm-Paths) | ForEach-Object { Read-Arm $_ })
}

function Series-Label {
    $names = @((Arm-Paths) | ForEach-Object { [IO.Path]::GetFileName($_) })
    if ($names -contains 'critic_lr3e5' -or $names -contains 'critic_lr3e4') { return 'V18: critic warmup LR 3e-5 / 3e-4; PPO обоих LR 3e-5; полный actor 27 396' }
    return 'Архив / другая серия: '+($names -join ' / ')+'. Параметры эксперимента V18 к этому архиву не относятся.'
}

if ($PlanOnly) {
    [PSCustomObject]@{status='PLAN_ONLY_NO_GUI';run_root=$RunRoot;arms=(Arm-Paths);experiment_label=(Series-Label);poll_interval_ms=1500;qa_seconds=$QaSeconds;snapshot_path=$SnapshotPath
        read_only=$true;network_calls=0;starts_training=$false;starts_simulator=$false;stops_processes=$false
        sources=@('phase.json','config.json','actual_gradient_norms.jsonl','progress.csv','warmup/progress.csv','warmup.json','checkpoint_131072_validation.json','checkpoint_262144_validation.json','final_validation.json','selection.json','complete.json','error.json')
        semantics='PyBullet fine-tuning from already trained weights; Gazebo verification is separate. Reward is not acceptance. No file movement => no LIVE claim.'} | ConvertTo-Json -Depth 6
    exit 0
}

if ($VerifyOnly) {
    $snapshot = @(Snapshot)
    $details = @($snapshot | Select-Object name,path,stage,state,heartbeat_utc,heartbeat_age_s,observed_counter_movement,critic_env,ppo_env,critic_updates,ppo_updates,counter_source,guard_rows,budget,evaluations,evaluated_policy_sha256,initial_sha256,@{n='reward_points';e={$_.ppo_reward_points.Count}})
    [PSCustomObject]@{status='PASS_READ_ONLY_SNAPSHOT_NO_GUI';run_root=$RunRoot;experiment_label=(Series-Label);arms=$details;warnings=@($script:ReadWarnings.ToArray())
        process_started=$false;network_calls=0;new_env_steps=0;new_optimizer_steps=0;acceptance_is_native_not_Gazebo=$true} | ConvertTo-Json -Depth 12
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
[Windows.Forms.Application]::EnableVisualStyles()
$form = New-Object Windows.Forms.Form
$form.Text = 'RL · Реальное дообучение в PyBullet · '+(Series-Label)
$form.StartPosition = 'CenterScreen'; $form.ClientSize = New-Object Drawing.Size(1280,850)
$form.MinimumSize = New-Object Drawing.Size(1120,820); $form.BackColor = $script:Background
$form.Font = New-Object Drawing.Font('Segoe UI',10)

function Label-At($Parent,[string]$Text,[int]$X,[int]$Y,[int]$W,[int]$H,[int]$Size=10,[bool]$Bold=$false) {
    $label = New-Object Windows.Forms.Label
    $label.Text=$Text;$label.Location=New-Object Drawing.Point($X,$Y);$label.Size=New-Object Drawing.Size($W,$H)
    $style=[Drawing.FontStyle]::Regular;if($Bold){$style=[Drawing.FontStyle]::Bold}
    $label.Font=New-Object Drawing.Font('Segoe UI',$Size,$style);$label.ForeColor=[Drawing.Color]::FromArgb(35,47,68)
    [void]$Parent.Controls.Add($label);return $label
}

$null=Label-At $form 'Обучение RL: реальные шаги, реальные проверки' 18 12 1240 38 20 $true
$null=Label-At $form ('PyBullet · '+(Series-Label)) 20 54 1240 24 10
$null=Label-At $form 'Начальные веса уже обучены. Рост награды не означает прохождение допусков. Окно читает файлы и не запускает обучение.' 20 78 1240 23 10

$cards=@()
for($i=0;$i -lt 2;$i++) {
    $panel=New-Object Windows.Forms.Panel;$panel.Location=New-Object Drawing.Point((18+628*$i),110);$panel.Size=New-Object Drawing.Size(610,160);$panel.BackColor=[Drawing.Color]::White
    [void]$form.Controls.Add($panel)
    $cards+=@{panel=$panel;title=(Label-At $panel '' 12 8 586 25 13 $true);status=(Label-At $panel '' 12 36 586 23 9)
        counters=(Label-At $panel '' 12 62 586 43 10);heartbeat=(Label-At $panel '' 12 108 586 20 8);hash=(Label-At $panel '' 12 130 586 21 8)}
}

function New-Chart([string]$Title,[int]$X,[int]$Y,[int]$W,[int]$H,[string]$YTitle) {
    $chart=New-Object Windows.Forms.DataVisualization.Charting.Chart
    $chart.Location=New-Object Drawing.Point($X,$Y);$chart.Size=New-Object Drawing.Size($W,$H);$chart.BackColor=[Drawing.Color]::White
    $area=New-Object Windows.Forms.DataVisualization.Charting.ChartArea('main')
    $area.AxisX.Title='Фактически выполненные шаги PPO';$area.AxisY.Title=$YTitle;$area.AxisY.IsStartedFromZero=$false
    $area.AxisX.MajorGrid.LineColor=[Drawing.Color]::FromArgb(235,238,243);$area.AxisY.MajorGrid.LineColor=$area.AxisX.MajorGrid.LineColor
    $area.AxisX.LabelStyle.Format='0,,.00M';$area.AxisY.LabelStyle.Format='0.000'
    [void]$chart.ChartAreas.Add($area);[void]$chart.Titles.Add($Title)
    $legend=New-Object Windows.Forms.DataVisualization.Charting.Legend('legend');$legend.Docking='Bottom';$legend.Font=New-Object Drawing.Font('Segoe UI',8)
    [void]$chart.Legends.Add($legend);[void]$form.Controls.Add($chart);return $chart
}

$rewardChart=New-Chart 'Средняя награда эпизодов · progress.csv' 18 282 610 265 'Средняя награда'
$qualityChart=New-Chart 'Ошибка конечного 3D-отрезка · оценки checkpoints' 646 282 610 265 'Finite RMSE, м'
$colors=@([Drawing.Color]::FromArgb(0,130,135),[Drawing.Color]::FromArgb(117,74,190))
for($i=0;$i -lt 2;$i++) {
    $series=New-Object Windows.Forms.DataVisualization.Charting.Series("reward$i");$series.ChartType='Line';$series.BorderWidth=3;$series.Color=$colors[$i]
    [void]$rewardChart.Series.Add($series)
    foreach($condition in @('nominal','strong','gust')) {
        $series=New-Object Windows.Forms.DataVisualization.Charting.Series("quality${i}_$condition");$series.ChartType='Line';$series.BorderWidth=2;$series.Color=$colors[$i];$series.MarkerStyle='Circle';$series.MarkerSize=7
        if($condition -eq 'strong'){$series.BorderDashStyle='Dash'};if($condition -eq 'gust'){$series.BorderDashStyle='Dot'}
        [void]$qualityChart.Series.Add($series)
    }
}

$null=Label-At $form 'Оценка checkpoints в PyBullet · nominal / strong / gust — среднее 4 seeds; tail — отдельный seed 84001. PASS/FAIL — все фиксированные native guards.' 20 551 1234 27 9
$grid=New-Object Windows.Forms.DataGridView
$grid.Location=New-Object Drawing.Point(18,580);$grid.Size=New-Object Drawing.Size(1238,190);$grid.ReadOnly=$true;$grid.AllowUserToAddRows=$false;$grid.AllowUserToDeleteRows=$false;$grid.RowHeadersVisible=$false
$grid.BackgroundColor=[Drawing.Color]::White;$grid.BorderStyle='None';$grid.AutoSizeColumnsMode='Fill';$grid.SelectionMode='FullRowSelect'
$grid.AllowUserToResizeRows=$false;$grid.Font=New-Object Drawing.Font('Segoe UI',9)
foreach($column in @('Вариант','Checkpoint','Шаги PPO','Nominal, м','Strong, м','Gust, м','Tail84001, м','Native guards','SHA NPZ')){[void]$grid.Columns.Add($column,$column)}
$grid.Columns[0].FillWeight=115;$grid.Columns[1].FillWeight=145;$grid.Columns[8].FillWeight=145
[void]$form.Controls.Add($grid)
$foot=Label-At $form '' 20 777 1238 58 8
$tooltips=New-Object Windows.Forms.ToolTip

function Paint-Snapshot {
    try {
        $snapshot=@(Snapshot)
        for($i=0;$i -lt 2;$i++) {
            if($i -ge $snapshot.Count){$cards[$i].panel.Visible=$false;continue};$cards[$i].panel.Visible=$true;$arm=$snapshot[$i]
            $description=switch($arm.name){'critic_lr3e5'{'critic LR 3e-5 → PPO LR 3e-5'}'critic_lr3e4'{'critic LR 3e-4 → PPO LR 3e-5'}'limited_actor'{'архив: 384 параметра actor'}'full_actor'{'архив: 27 396 параметров actor'}default{'сохранённый эксперимент'}}
            $cards[$i].title.Text=$arm.name+' · '+$description
            $stage=switch($arm.stage){'critic'{'WARMUP'}'ppo'{'PPO'}'validation'{'EVAL'}'verification'{'VERIFY'}'complete'{'COMPLETE'}'error'{'ERROR'}default{'WAIT'}}
            $cards[$i].status.Text=$stage+' · '+$arm.state
            $cards[$i].status.ForeColor=if($arm.state -like 'ERROR*'){[Drawing.Color]::Firebrick}elseif($arm.observed_counter_movement -and $arm.state -like 'ОБНОВЛЯЕТСЯ*'){[Drawing.Color]::SeaGreen}else{[Drawing.Color]::DimGray}
            $cards[$i].counters.Text='Среда: critic '+(Format-Count $arm.critic_env)+' · PPO '+(Format-Count $arm.ppo_env)+"`r`n"+'Реальные Adam updates: critic '+(Format-Count $arm.critic_updates)+' · PPO '+(Format-Count $arm.ppo_updates)
            $age=if($null -eq $arm.heartbeat_age_s){'—'}else{([Math]::Max(0,$arm.heartbeat_age_s)).ToString('F1',$script:Culture)+' с'}
            $cards[$i].heartbeat.Text='Heartbeat UTC: '+[string]$arm.heartbeat_utc+' · возраст '+$age
            $short=if($arm.evaluated_policy_sha256){$arm.evaluated_policy_sha256.Substring(0,[Math]::Min(18,$arm.evaluated_policy_sha256.Length))+'…'}else{'ещё нет оценки'}
            $cards[$i].hash.Text='Последний оценённый / выбранный NPZ: '+$short
            $tooltips.SetToolTip($cards[$i].hash,[string]$arm.evaluated_policy_sha256);$tooltips.SetToolTip($cards[$i].counters,$arm.counter_source)
            $tooltips.SetToolTip($cards[$i].panel,$arm.path)
            $series=$rewardChart.Series["reward$i"];$series.LegendText=$arm.name;$series.Points.Clear()
            foreach($point in $arm.ppo_reward_points){[void]$series.Points.AddXY([double]$point.x,[double]$point.y)}
            foreach($condition in @('nominal','strong','gust')) {
                $series=$qualityChart.Series["quality${i}_$condition"];$series.LegendText=$arm.name+' / '+$condition;$series.Points.Clear()
                foreach($evaluation in $arm.evaluations){$value=Field $evaluation $condition;if($null -ne $value -and $null -ne $evaluation.timesteps){[void]$series.Points.AddXY([double]$evaluation.timesteps,[double]$value)}}
            }
        }
        $grid.Rows.Clear()
        foreach($arm in $snapshot){foreach($evaluation in $arm.evaluations){
            $guard=if($null -eq $evaluation.guards_pass){'НЕИЗВЕСТНО'}elseif($evaluation.guards_pass){'PASS'}else{'FAIL'}
            $short=if($evaluation.policy_sha256){$evaluation.policy_sha256.Substring(0,12)+'…'}else{'—'}
            $row=$grid.Rows.Add([string[]]@($arm.name,$evaluation.checkpoint,(Format-Count $evaluation.timesteps),(Format-Metric $evaluation.nominal),(Format-Metric $evaluation.strong),(Format-Metric $evaluation.gust),(Format-Metric $evaluation.tail84001),$guard,$short))
            $grid.Rows[$row].Cells[7].Style.ForeColor=if($guard -eq 'FAIL'){[Drawing.Color]::Firebrick}elseif($guard -eq 'PASS'){[Drawing.Color]::SeaGreen}else{[Drawing.Color]::DimGray}
            $grid.Rows[$row].Cells[8].ToolTipText=[string]$evaluation.policy_sha256
            $grid.Rows[$row].Cells[1].ToolTipText=[string]$evaluation.checkpoint_timing
        }}
        $warning=if($script:ReadWarnings.Count){' · чтение: '+($script:ReadWarnings.ToArray() -join '; ')}else{''}
        $foot.Text='Источник: '+$RunRoot+"`r`n"+'Чтение '+[DateTimeOffset]::Now.ToString('HH:mm:ss')+' · каждые 1,5 с · закрытие окна не останавливает обучение.'+$warning
    } catch {
        $foot.Text='Чтение будет повторено: '+$_.Exception.Message+"`r`n"+'Источник: '+$RunRoot
    }
}

$timer=New-Object Windows.Forms.Timer;$timer.Interval=1500
$timer.Add_Tick({Paint-Snapshot})
$script:QaFailure=$null
$qaTimer=$null
if($QaSeconds -gt 0){
    $qaTimer=New-Object Windows.Forms.Timer;$qaTimer.Interval=$QaSeconds*1000
    $qaTimer.Add_Tick({
        $qaTimer.Stop()
        try {
            Paint-Snapshot
            if(-not [string]::IsNullOrWhiteSpace($SnapshotPath)){
                $bitmap=New-Object Drawing.Bitmap($form.ClientSize.Width,$form.ClientSize.Height)
                try {$form.DrawToBitmap($bitmap,(New-Object Drawing.Rectangle(0,0,$form.ClientSize.Width,$form.ClientSize.Height)));$bitmap.Save($SnapshotPath,[Drawing.Imaging.ImageFormat]::Png)}finally{$bitmap.Dispose()}
            }
        }catch{$script:QaFailure=$_.Exception.Message}finally{$form.Close()}
    })
}
$form.Add_Shown({Paint-Snapshot;$timer.Start();if($null -ne $qaTimer){$qaTimer.Start()}})
$form.Add_FormClosed({$timer.Stop();$timer.Dispose();if($null -ne $qaTimer){$qaTimer.Stop();$qaTimer.Dispose()}})
[void]$form.ShowDialog()
$tooltips.Dispose();$form.Dispose()
if($script:QaFailure){throw ('Owned-window QA snapshot failed: '+$script:QaFailure)}
