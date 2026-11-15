[CmdletBinding()]
param(
    [string]$RunRoot = 'artifacts/refinement/joint_v2/prior_ablation_pilot_v28_111/prior0',
    [string]$ControlRoot = 'artifacts/refinement/joint_v2/route_initial_pilot_v24_111/zero_route_integral',
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
$script:ArchiveA = $null
$script:ProfileReferences = $null
$script:JsonCache = @{}
Add-Type -AssemblyName System.Drawing
$script:Background = [Drawing.Color]::FromArgb(245,247,251)
if (-not [IO.Path]::IsPathRooted($RunRoot)) { $RunRoot = Join-Path $PSScriptRoot $RunRoot }
$RunRoot = [IO.Path]::GetFullPath($RunRoot)
if (-not [IO.Path]::IsPathRooted($ControlRoot)) { $ControlRoot=Join-Path $PSScriptRoot $ControlRoot }
$ControlRoot=[IO.Path]::GetFullPath($ControlRoot)
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
    if(-not [IO.File]::Exists($Path)){return $null}
    $info=Get-Item -LiteralPath $Path
    $stamp=[string]$info.LastWriteTimeUtc.Ticks+':'+[string]$info.Length
    if($script:JsonCache.ContainsKey($Path) -and $script:JsonCache[$Path].stamp -eq $stamp){return $script:JsonCache[$Path].value}
    $raw = Shared-Text $Path
    if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
    try {$value=ConvertFrom-Json -InputObject $raw;$script:JsonCache[$Path]=@{stamp=$stamp;value=$value};return $value} catch {
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
    if ([IO.File]::Exists((Join-Path $RunRoot 'phase.json')) -or [IO.File]::Exists((Join-Path $RunRoot 'config.json'))) { return @($RunRoot) }
    if ([IO.Path]::GetFileName($RunRoot) -in @('prior0','prior001')) { return @($RunRoot) }
    return @((Join-Path $RunRoot 'prior0'))
}

function Config-Issue($Config,[bool]$ArchiveControl) {
    if ($null -eq $Config) { return '' }
    $expectedSchema=if($ArchiveControl){'joint-initial-coverage-ppo-v24.0'}else{'joint-full-policy-prior-ablation-v28.0'}
    if((Field $Config 'schema') -ne $expectedSchema -or (Field $Config 'conditioning_mode') -ne 'zero_route_integral' -or (Field $Config 'training_reward_arm') -ne 'metric_cost'){return 'Неверная версия/config: требуется full80 V24/V28 metric_cost'}
    if((Field $Config 'initial_sha256') -ne '0df1b6cd2811f5196395e8b707a3eee0387c8b0050375479a912511df779bf40' -or (Field $Config 'actor_trainable_mean_elements') -ne 27396 -or (Field $Config 'log_std_frozen') -ne $true){return 'Неверный source68/full27396/log_std binding'}
    if((Field $Config 'warm_learning_rate') -ne 0.0003 -or (Field $Config 'PPO_learning_rate') -ne 0.00003 -or (Field $Config 'gamma') -ne 1){return 'Изменены исходные LR/gamma'}
    if(-not $ArchiveControl){
        $variant=Field $Config 'variant';$weight=Number (Field $Config 'PPO_prior_weight')
        if($variant -notin @('prior0','prior001') -or $null -eq $weight -or ($variant -eq 'prior0' -and $weight -ne 0) -or ($variant -eq 'prior001' -and $weight -ne 0.01) -or (Field $Config 'anchor_steps') -ne 0){return 'Неверная ветвь KL; offline IL anchor должен быть 0'}
        $driver='scripts/train_joint_prior_ablation_v28.py';$digest='0c948b91c39455aba867a354db270fc5253ddf61855d1b94b5df8222539d9134'
        if((Field (Field $Config 'source_sha256') $driver) -ne $digest -or (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $driver) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest){return 'Изменён frozen V28 driver'}
    }
    return ''
}

function Read-Arm([string]$Path,[bool]$ArchiveControl=$false) {
    $name = [IO.Path]::GetFileName($Path)
    $phase = Shared-Json (Join-Path $Path 'phase.json')
    $config = Shared-Json (Join-Path $Path 'config.json')
    $complete = Shared-Json (Join-Path $Path 'complete.json')
    $errorPath = Join-Path $Path 'error.json'
    $errorRecord = Shared-Json $errorPath
    $pairErrorPath = Join-Path (Split-Path $Path -Parent) 'paired_complete.json'
    $pairRecord = Shared-Json $pairErrorPath
    $pairStatus = [string](Field $pairRecord 'status')
    $pairFailed = $pairStatus -in @('error','failed','infra_error') -or -not [string]::IsNullOrWhiteSpace([string](Field $pairRecord 'error'))
    $errorReason = [string](Field $errorRecord 'error');$errorSource = ''
    $hasError = [IO.File]::Exists($errorPath) -or $pairFailed
    if ([IO.File]::Exists($errorPath)) {
        $errorSource = $errorPath
        if ([string]::IsNullOrWhiteSpace($errorReason)) { $errorReason = 'error.json существует; ожидается полная запись причины' }
    } elseif ($pairFailed) {
        $errorSource = $pairErrorPath;$errorReason = [string](Field $pairRecord 'error')
        if ([string]::IsNullOrWhiteSpace($errorReason)) {
            $failedArms = @((Field $pairRecord 'arms') | Where-Object { $null -ne (Field $_ 'returncode') -and (Field $_ 'returncode') -ne 0 } | ForEach-Object { [string](Field $_ 'variant')+' (код '+[string](Field $_ 'returncode')+')' })
            $errorReason = 'Парный запуск: status='+$pairStatus
            if ($failedArms.Count) { $errorReason += '; '+($failedArms -join ', ') }
        }
    }
    $configIssue=Config-Issue $config $ArchiveControl
    if($configIssue -and -not $hasError){$hasError=$true;$errorReason=$configIssue;$errorSource=Join-Path $Path 'config.json'}
    $warm = Shared-Json (Join-Path $Path 'warmup.json')
    $selection = Shared-Json (Join-Path $Path 'selection.json')
    $ppoRows = @(Shared-Csv (Join-Path $Path 'progress.csv'))
    $warmRows = @(Shared-Csv (Join-Path $Path 'warmup/progress.csv'))
    $guards = Guard-Counters (Join-Path $Path 'actual_gradient_norms.jsonl')
    $stage = [string](Field $phase 'phase')
    $phaseSourceStage = $stage
    if ($stage -eq 'error' -and -not $hasError) { $hasError=$true;$errorReason='phase.json сообщает ERROR';$errorSource=Join-Path $Path 'phase.json' }
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
    $baseline = $null
    $baselineReferences = @{
        raw=@{path='artifacts/refinement/joint_v2/route_memory_baseline_v20_root01/raw/evaluation.json';sha='1c503e3c0f75be090fe36f439c9061bddeeedf12f1ee70fa6136500c52e5719d'}
        zero_route_integral=@{path='artifacts/refinement/joint_v2/route_memory_baseline_v20_root01/zero_route_integral/evaluation.json';sha='f0f6930a58d690f1c474c4ff954575c0af84e9e9c7f7d5cb66a266b0b82cbb9e'}
    }
    $baselineKey='zero_route_integral'
    if ($baselineReferences.ContainsKey($baselineKey)) {
        $reference = $baselineReferences[$baselineKey]; $baselinePath = Join-Path $PSScriptRoot $reference.path
        if ([IO.File]::Exists($baselinePath) -and (Get-FileHash -LiteralPath $baselinePath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $reference.sha) {
            $record = Shared-Json $baselinePath
            if ((Field $record 'conditioning_mode') -eq $baselineKey -and (Field $record 'training_transitions') -eq 0 -and (Field $record 'optimizer_steps') -eq 0) {
                $baseline = Evaluation-Row $record 'BASELINE0 · без PPO' 0 'ZERO_UPDATE_ARCHIVE_REFERENCE'
                $baseline | Add-Member -NotePropertyName source_path -NotePropertyValue $reference.path
                $baseline | Add-Member -NotePropertyName source_sha256 -NotePropertyValue $reference.sha
            } else { [void]$script:ReadWarnings.Add('Baseline: неверный режим или ненулевые training/optimizer steps') }
        } else { [void]$script:ReadWarnings.Add('Baseline: отсутствует или не совпадает фиксированный SHA') }
    }
    $evaluations = @()
    foreach ($label in @('checkpoint_131072','checkpoint_262144','final')) {
        $evaluation = Shared-Json (Join-Path $Path ($label + '_validation.json'))
        if ($null -eq $evaluation) { continue }
        $evaluations += Evaluation-Row $evaluation $label (Number (Field $evaluation 'timesteps')) 'ACTUAL_PPO_CHECKPOINT'
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
    if ($hasError) { $state = 'ERROR · '+$errorReason;$stage='error' }
    elseif ($null -ne $complete -or $stage -eq 'complete') { $state = 'COMPLETE ARCHIVE · обучение завершено, новые updates не идут' }
    elseif ($null -ne $phase) {
        if ($recentMovement -and $fresh) { $state = 'ОБНОВЛЯЕТСЯ · движение счётчиков подтверждено' }
        elseif ($fresh) { $state = 'СВЕЖИЙ HEARTBEAT · ждём движения счётчиков' }
        else { $state = 'АРХИВ / НЕТ СВЕЖИХ ИЗМЕНЕНИЙ' }
    }
    $hash = Field $selection 'policy_sha256'
    if (-not $hash -and $evaluations.Count) { $hash = $evaluations[-1].policy_sha256 }
    $priorWeight=Field $config 'PPO_prior_weight'
    if($ArchiveControl -and $null -ne $complete){$priorWeight=Field (Field $complete 'optimizer') 'KL_weight'}
    return [PSCustomObject]@{
        name=$name;path=$Path;prior_weight=$priorWeight;source_version=(Field $config 'schema');source_validation_error=$configIssue;stage=$stage;phase_source_stage=$phaseSourceStage;state=$state;error_reason=$errorReason;error_source=$errorSource;pair_failed=$pairFailed;heartbeat_utc=$heartbeat;heartbeat_age_s=$heartbeatAge
        observed_counter_movement=$recentMovement;critic_env=$criticEnv;ppo_env=$ppoEnv
        critic_updates=$criticUpdates;ppo_updates=$ppoUpdates;counter_source=($counterSource+'; последние опубликованные CSV/guard могут отставать от rollout во время checkpoint evaluation')
        guard_rows=(Field $guards 'rows');last_gradient=(Field $guards 'last')
        budget=(Field $config 'budget');ppo_reward_points=$points;evaluations=$evaluations;baseline_reference=$baseline
        evaluated_policy_sha256=$hash;initial_sha256=(Field $config 'initial_sha256')
        completed_cases=(Field $phase 'completed_cases');total_cases=(Field $phase 'total_cases')
    }
}

function Evaluation-Row($Evaluation,[string]$Label,$Timesteps,[string]$Kind) {
    $cells = Field (Field $Evaluation 'aggregate') 'cells'; $tail = $null
    foreach ($episode in @(Field $Evaluation 'episodes')) {
        if ((Field $episode 'scenario') -eq 'force_strong' -and (Field $episode 'mission') -eq 'route' -and (Field $episode 'environment_seed') -eq 84001) { $tail = Number (Field (Field $episode 'geometry') 'segment_rmse_m') }
    }
    $original=Field $Evaluation 'original24_admission'
    if($null -eq $original){$original=Field $Evaluation 'own_baseline_admission'}
    $matched=Field $Evaluation 'additional_matched_profiles_admission'
    $row=[PSCustomObject]@{
        checkpoint=$Label;timesteps=$Timesteps;source_kind=$Kind
        nominal=(Number (Field (Field $cells 'nominal/route') 'segment_rmse_m'))
        strong=(Number (Field (Field $cells 'force_strong/route') 'segment_rmse_m'))
        gust=(Number (Field (Field $cells 'gust/route') 'segment_rmse_m'))
        strong_hover=(Number (Field (Field $cells 'force_strong/hover') 'goal_rmse_m'))
        tail84001=$tail;guards_pass=(Field $Evaluation 'fixed_development_guards_pass')
        policy_sha256=(Field $Evaluation 'policy_sha256');checkpoint_timing=(Field $Evaluation 'checkpoint_timing')
        original24_admission=(Field $original 'passed');matched5_pass=$null;combined_admission=$null;engineering_eligibility=$null;engineering_eligibility_source='VIEWER_DERIVED_FROM_STORED_BOOLEAN_CHECKS_NOT_PRODUCER_FIELD';original_hard_checks=(Field $original 'checks')
        matched_worst_finite=$null;matched_worst_peak=$null;matched_episodes=@();role=''
    }
    if($null -ne $matched){Attach-Matched $row $matched}
    return $row
}

function Attach-Matched($Row,$Matched) {
    if((Field $Matched 'profile_count') -ne 5 -or @(Field $Matched 'episodes').Count -ne 5){return}
    if($Row.policy_sha256 -and (Field $Matched 'policy_sha256') -ne $Row.policy_sha256){[void]$script:ReadWarnings.Add('Matched policy SHA mismatch; profiles ignored');return}
    $episodes=@(Field $Matched 'episodes');$finite=@();$peak=@()
    foreach($episode in $episodes){$g=Field $episode 'geometry';$finite+=@(Number (Field $g 'segment_rmse_m'));$peak+=@(Number (Field $g 'active_segment_peak_m'))}
    $Row.matched5_pass=Field $Matched 'all_profiles_pass'
    $Row.matched_episodes=@($episodes|ForEach-Object {[PSCustomObject]@{scenario=(Field $_ 'scenario');intervention=(Field $_ 'intervention');eval_seed=(Field $_ 'eval_seed')
        geometry=[PSCustomObject]@{segment_rmse_m=(Number (Field (Field $_ 'geometry') 'segment_rmse_m'));active_segment_peak_m=(Number (Field (Field $_ 'geometry') 'active_segment_peak_m'))}
        profile_admission_pass=(Field (Field $_ 'profile_admission') 'passed')}})
    $Row.matched_worst_finite=($finite|Where-Object {$null -ne $_}|Measure-Object -Maximum).Maximum
    $Row.matched_worst_peak=($peak|Where-Object {$null -ne $_}|Measure-Object -Maximum).Maximum
    if($null -ne $Row.original24_admission -and $null -ne $Row.matched5_pass){$Row.combined_admission=[bool]$Row.original24_admission -and [bool]$Row.matched5_pass}
}

function Heading-Counts([string]$Path) {
    $records=@(Shared-Json $Path)
    if(-not [IO.File]::Exists($Path) -or $records.Count -eq 0 -or $null -eq $records[0]){return $null}
    $totals=[ordered]@{envs=$records.Count;matched_routes=0L;original_routes=0L;hover_resets=0L;matched_steps=0L;original_steps=0L;hover_steps=0L;matched_seconds=0.;original_seconds=0.;hover_seconds=0.;max_per_env_route_imbalance=0L;source=$Path}
    foreach($entry in $records){
        foreach($field in @('matched_routes','original_routes','hover_resets')){$value=Number (Field $entry $field);if($null -eq $value){return $null};$totals[$field]+=[long]$value}
        $steps=Field $entry 'control_steps_by_stratum';$seconds=Field $entry 'seconds_by_stratum'
        foreach($spec in @(@('matched_route','matched'),@('original_route','original'),@('hover','hover'))){$n=Number (Field $steps $spec[0]);$s=Number (Field $seconds $spec[0]);if($null -eq $n -or $null -eq $s){return $null};$totals[$spec[1]+'_steps']+=[long]$n;$totals[$spec[1]+'_seconds']+=[double]$s}
        $imbalance=[Math]::Abs([long](Field $entry 'matched_routes')-[long](Field $entry 'original_routes'))
        $totals.max_per_env_route_imbalance=[Math]::Max($totals.max_per_env_route_imbalance,$imbalance)
    }
    return [PSCustomObject]$totals
}

function Frozen-Profile-References {
    if($null -ne $script:ProfileReferences){return $script:ProfileReferences}
    $manifest=Shared-Json (Join-Path $PSScriptRoot 'docs/joint_prior_ablation_plans_v28/baseline_reuse_manifest.json')
    if($null -eq $manifest){[void]$script:ReadWarnings.Add('V28 reference manifest пока не опубликован; matched F0 references неизвестны');return @()}
    $rows=@()
    foreach($profile in @(Field $manifest 'sourceF0_v22_profiles')){
        $binding=Field (Field $profile 'source') 'intervention_summary.json';$relative=Field $binding 'path';$digest=Field $binding 'sha256'
        if(-not $relative){continue};$path=Join-Path $PSScriptRoot $relative
        if(-not [IO.File]::Exists($path) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest){[void]$script:ReadWarnings.Add('Frozen matched reference SHA mismatch');continue}
        $record=Shared-Json $path;$geometry=Field $record 'geometry'
        $rows+=[PSCustomObject]@{key=(Field $profile 'key');intervention=(Field $profile 'intervention');scenario=(Field $profile 'scenario');seed=(Field $profile 'environment_seed')
            finite=(Number (Field $geometry 'segment_rmse_m'));peak=(Number (Field $geometry 'active_segment_peak_m'));source=$relative;source_sha256=$digest;source_kind='REUSED_V22_ZERO_UPDATE_REFERENCE_NOT_PPO'}
    }
    if($rows.Count -eq 5){$script:ProfileReferences=$rows}
    return $rows
}

function Snapshot {
    $script:ReadWarnings.Clear()
    $cPaths=@(Arm-Paths);$c=Read-Arm $cPaths[0];$c|Add-Member -Force -NotePropertyName role -NotePropertyValue 'B_V28_FULL_POLICY_PRIOR_ABLATION'
    $weightText=if($null -ne $c.prior_weight){([double]$c.prior_weight).ToString('0.##',$script:Culture)}else{'?'}
    $c|Add-Member -Force -NotePropertyName display_name -NotePropertyValue ('B · V28 PPO κ='+$weightText+' · '+$c.name)
    $c|Add-Member -Force -NotePropertyName heading_critic -NotePropertyValue (Heading-Counts (Join-Path $c.path 'heading_counts_critic.json'))
    $c|Add-Member -Force -NotePropertyName heading_ppo -NotePropertyValue (Heading-Counts (Join-Path $c.path 'heading_counts_ppo.json'))
    foreach($row in $c.evaluations){
        $row.role='B NEW'
        $values=@((Field $row.original_hard_checks 'all_original_fixed_native_hard_pass'),(Field $row.original_hard_checks 'source77_nonregression'),$row.matched5_pass)
        if(@($values|Where-Object {$null -eq $_}).Count -eq 0){$row.engineering_eligibility=(@($values|Where-Object {-not [bool]$_}).Count -eq 0)}
    }
    if($null -eq $script:ArchiveA){$script:ArchiveA=Read-Arm $ControlRoot $true}
    $a=$script:ArchiveA;$a|Add-Member -Force -NotePropertyName role -NotePropertyValue 'A_REUSED_COMPLETED_V24'
    $a|Add-Member -Force -NotePropertyName display_name -NotePropertyValue 'A · V24 ARCHIVE κ=0.1'
    $a.observed_counter_movement=$false
    if(-not $a.error_reason){$a.stage='archive';$a.state='COMPLETE ARCHIVE · V24 κ=0.1 · новых Adam updates: 0'}
    foreach($row in $a.evaluations){
        $row.role='A REUSED'
        $values=@((Field $row.original_hard_checks 'all_original_fixed_native_hard_pass'),(Field $row.original_hard_checks 'source77_nonregression'),$row.matched5_pass)
        if(@($values|Where-Object {$null -eq $_}).Count -eq 0){$row.engineering_eligibility=(@($values|Where-Object {-not [bool]$_}).Count -eq 0)}
    }
    $c|Add-Member -Force -NotePropertyName matched_reference_profiles -NotePropertyValue @(Frozen-Profile-References)
    return @($c,$a)
}

function Series-Label {
    return 'V28 · full80 PPO prior κ=0/0.01; A — архив V24 κ=0.1 · reward одинаковый, on-policy данные меняются; original24 / matched5 / FULL отдельно'
}

if ($PlanOnly) {
    [PSCustomObject]@{status='PLAN_ONLY_NO_GUI';run_root=$RunRoot;B=(@(Arm-Paths)[0]);reused_A=$ControlRoot;new_A_aux_episodes=0;experiment_label=(Series-Label);poll_interval_ms=1500;qa_seconds=$QaSeconds;snapshot_path=$SnapshotPath
        maximum_display_cards=2;reward_is_quality_metric=$false;zero_update_reference_separate_from_PPO=$true;quality_x_minimum=0;quality_x_fallback_maximum=262144;quality_x_uses_declared_PPO_budget_when_available=$true
        read_only=$true;network_calls=0;starts_training=$false;starts_simulator=$false;stops_processes=$false;raw_returns_cross_arm_quality_comparable=$false;reward_objective_same_both=$true;on_policy_data_can_differ=$true;offline_IL_anchor_steps=0;PPO_update_counter_is_not_prior_KL_counter=$true
        sources=@('phase.json','config.json','actual_gradient_norms.jsonl','progress.csv','warmup/progress.csv','warmup.json','*_validation.json','selection.json','complete.json','error.json','../paired_complete.json','heading_counts_critic.json','heading_counts_ppo.json','docs/joint_prior_ablation_plans_v28/baseline_reuse_manifest.json')
        semantics='PyBullet fine-tuning from already trained weights; Gazebo verification is separate. Reward is not acceptance. No file movement => no LIVE claim.'} | ConvertTo-Json -Depth 6
    exit 0
}

if ($VerifyOnly) {
    $snapshot = @(Snapshot)
    $details = @($snapshot | Select-Object role,display_name,name,path,prior_weight,source_version,source_validation_error,stage,phase_source_stage,state,error_reason,error_source,pair_failed,heartbeat_utc,heartbeat_age_s,observed_counter_movement,critic_env,ppo_env,critic_updates,ppo_updates,counter_source,guard_rows,budget,evaluations,baseline_reference,heading_critic,heading_ppo,evaluated_policy_sha256,initial_sha256,@{n='reward_points';e={$_.ppo_reward_points.Count}})
    [PSCustomObject]@{status='PASS_READ_ONLY_SNAPSHOT_NO_GUI';run_root=$RunRoot;experiment_label=(Series-Label);arms=$details;warnings=@($script:ReadWarnings.ToArray())
        frozen_matched_profile_references=$snapshot[0].matched_reference_profiles;new_A_aux_episodes=0;engineering_is_separate_resource_classification=$true;full_admission_unchanged=$true
        process_started=$false;network_calls=0;new_env_steps=0;new_optimizer_steps=0;A_new_training_updates=0;acceptance_is_native_not_Gazebo=$true;raw_returns_cross_arm_quality_comparable=$false;reward_objective_same_both=$true;on_policy_data_can_differ=$true;offline_IL_anchor_steps=0;PPO_update_counter_is_not_prior_KL_counter=$true} | ConvertTo-Json -Depth 15
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
[Windows.Forms.Application]::EnableVisualStyles()
$form=New-Object Windows.Forms.Form
$form.Text='V28 · full-policy PPO prior / архив V24 κ=0.1'
$form.StartPosition='CenterScreen';$form.ClientSize=New-Object Drawing.Size(1400,980)
$form.MinimumSize=New-Object Drawing.Size(1416,1019);$form.BackColor=$script:Background
$form.Font=New-Object Drawing.Font('Segoe UI',9)

function Label-At($Parent,[string]$Text,[int]$X,[int]$Y,[int]$W,[int]$H,[int]$Size=9,[bool]$Bold=$false){
    $l=New-Object Windows.Forms.Label;$l.Text=$Text;$l.Location=New-Object Drawing.Point($X,$Y);$l.Size=New-Object Drawing.Size($W,$H)
    $style=[Drawing.FontStyle]::Regular;if($Bold){$style=[Drawing.FontStyle]::Bold}
    $l.Font=New-Object Drawing.Font('Segoe UI',$Size,$style);$l.ForeColor=[Drawing.Color]::FromArgb(35,47,68)
    [void]$Parent.Controls.Add($l);return $l
}
function Flag($Value){if($null -eq $Value){return '—'};if([bool]$Value){return 'PASS'};return 'FAIL'}
function Profile-Key($Record){$seed=Field $Record 'eval_seed';if($null -eq $seed){$seed=Field $Record 'environment_seed'};return [string](Field $Record 'intervention')+'/'+[string](Field $Record 'scenario')+'/'+[string]$seed}
function Heading-Text($Record,[string]$Stage){
    if($null -eq $Record){return $Stage+': файл снимка ещё не записан'}
    return $Stage+' · resets matched/original/hover: '+(Format-Count $Record.matched_routes)+' / '+(Format-Count $Record.original_routes)+' / '+(Format-Count $Record.hover_resets)+
        ' · фактические control-секунды: '+$Record.matched_seconds.ToString('F1',$script:Culture)+' / '+$Record.original_seconds.ToString('F1',$script:Culture)+' / '+$Record.hover_seconds.ToString('F1',$script:Culture)
}
$null=Label-At $form 'V28: сила prior в полной PPO-политике · B текущий / A архив κ=0.1' 18 8 1360 33 19 $true
$null=Label-At $form 'Full80 / 27396 mean-параметров · source68 уже обучен · log_std заморожен · PyBullet; Gazebo проверяется отдельно.' 20 47 1360 23 10
$null=Label-At $form 'FULL admission = Admit24 И Matched5. EngDiag — отдельная пригодность к диагностическому измерению; FULL FAIL сохраняется.' 20 72 1360 23 10 $true
$auxLabel=Label-At $form '' 20 97 1360 23 9
$cards=@()
for($i=0;$i -lt 2;$i++){
    $panel=New-Object Windows.Forms.Panel;$panel.Location=New-Object Drawing.Point((18+696*$i),126);$panel.Size=New-Object Drawing.Size(668,151);$panel.BackColor=[Drawing.Color]::White;[void]$form.Controls.Add($panel)
    $cards+=@{panel=$panel;title=(Label-At $panel '' 12 7 644 25 12 $true);status=(Label-At $panel '' 12 34 644 24 9)
        counters=(Label-At $panel '' 12 60 644 42 10);heartbeat=(Label-At $panel '' 12 105 644 18 8);hash=(Label-At $panel '' 12 125 644 19 8)}
}
$headingLabel=Label-At $form '' 20 283 1360 44 9
$null=Label-At $form 'Matched-страта: полный p/v/rpy/rates + ref0; это пакет, не только yaw. 50/50 — route-чередование внутри env, не доля времени.' 20 330 1360 21 8

function New-Chart([string]$Title,[int]$X,[string]$YTitle){
    $chart=New-Object Windows.Forms.DataVisualization.Charting.Chart;$chart.Location=New-Object Drawing.Point($X,355);$chart.Size=New-Object Drawing.Size(436,205);$chart.BackColor=[Drawing.Color]::White
    $area=New-Object Windows.Forms.DataVisualization.Charting.ChartArea('main');$area.AxisX.Minimum=0;$area.AxisX.Maximum=262144;$area.AxisX.Interval=65536;$area.AxisX.Title='Шаги PPO'
    $area.AxisX.LabelStyle.Format='0';$area.AxisX.LabelStyle.Font=New-Object Drawing.Font('Segoe UI',7);$area.AxisY.Title=$YTitle;$area.AxisY.IsStartedFromZero=$false;$area.AxisY.LabelStyle.Format='0.000'
    $area.AxisX.MajorGrid.LineColor=[Drawing.Color]::FromArgb(235,238,243);$area.AxisY.MajorGrid.LineColor=$area.AxisX.MajorGrid.LineColor
    [void]$chart.ChartAreas.Add($area);[void]$chart.Titles.Add($Title)
    $legend=New-Object Windows.Forms.DataVisualization.Charting.Legend('legend');$legend.Docking='Bottom';$legend.Font=New-Object Drawing.Font('Segoe UI',7);[void]$chart.Legends.Add($legend)
    [void]$form.Controls.Add($chart);return $chart
}
$rewardChart=New-Chart 'Одинаковый reward · разные траектории' 18 'Средняя награда'
$qualityChart=New-Chart 'Original24 · finite RMSE' 482 'м'
$matchedChart=New-Chart 'Matched5 · худшие finite / peak' 946 'м'
$colors=@([Drawing.Color]::FromArgb(0,130,135),[Drawing.Color]::FromArgb(125,135,147))
for($i=0;$i -lt 2;$i++){
    $tag=if($i -eq 0){'B new'}else{'A archive'}
    $s=New-Object Windows.Forms.DataVisualization.Charting.Series("reward$i");$s.ChartType='Line';$s.Color=$colors[$i];$s.BorderWidth=2;$s.LegendText=$tag;[void]$rewardChart.Series.Add($s)
    foreach($spec in @(@('nominal','N'),@('strong','S'),@('gust','G'))){
        $s=New-Object Windows.Forms.DataVisualization.Charting.Series("quality${i}_$($spec[0])");$s.ChartType='Line';$s.Color=$colors[$i];$s.BorderWidth=2;$s.MarkerStyle='Circle';$s.MarkerSize=6;$s.LegendText=$tag+' '+$spec[1]
        if($spec[0] -eq 'strong'){$s.BorderDashStyle='Dash'};if($spec[0] -eq 'gust'){$s.BorderDashStyle='Dot'};[void]$qualityChart.Series.Add($s)
    }
    foreach($field in @('matched_worst_finite','matched_worst_peak')){
        $s=New-Object Windows.Forms.DataVisualization.Charting.Series("matched${i}_$field");$s.ChartType='Line';$s.Color=$colors[$i];$s.BorderWidth=2;$s.MarkerStyle='Circle';$s.MarkerSize=6;$s.LegendText=$tag+' '+$(if($field -eq 'matched_worst_finite'){'finite'}else{'peak'})
        if($field -eq 'matched_worst_peak'){$s.BorderDashStyle='Dash'};[void]$matchedChart.Series.Add($s)
    }
}
foreach($condition in @('nominal','strong','gust')){$s=New-Object Windows.Forms.DataVisualization.Charting.Series('base_'+$condition);$s.ChartType='Point';$s.Color=[Drawing.Color]::SandyBrown;$s.MarkerStyle='Diamond';$s.MarkerSize=8;$s.IsVisibleInLegend=$false;[void]$qualityChart.Series.Add($s)}

$null=Label-At $form 'A — архив κ=0.1, новых A/Aux: 0. B — V28, калибровка и reward прежние; prior только в loss. Return не заменяет качество; F0 — ромбы x=0.' 20 566 1360 22 8
function New-Grid([int]$Y,[int]$Height,$Columns){
    $g=New-Object Windows.Forms.DataGridView;$g.Location=New-Object Drawing.Point(18,$Y);$g.Size=New-Object Drawing.Size(1364,$Height)
    $g.ReadOnly=$true;$g.AllowUserToAddRows=$false;$g.AllowUserToDeleteRows=$false;$g.RowHeadersVisible=$false;$g.BackgroundColor=[Drawing.Color]::White;$g.BorderStyle='None';$g.AutoSizeColumnsMode='Fill';$g.SelectionMode='FullRowSelect';$g.Font=New-Object Drawing.Font('Segoe UI',8)
    foreach($column in $Columns){[void]$g.Columns.Add($column,$column)};[void]$form.Controls.Add($g);return $g
}
$grid=New-Grid 590 169 @('Роль','Checkpoint','PPO','Nom24,м','Strong24,м','Gust24,м','S hover,м','Hard24','Admit24','Matched5','FULL','EngDiag','M5 finite','M5 peak','SHA NPZ')
$grid.Columns[11].HeaderCell.ToolTipText='EngDiag: вычисление монитора из сохранённых absoluteHard24/hover caps + source77 + всех matched5. Own-relative hover1.10/3%nominal/own-tail входят только в FULL. Это не producer admission и не Gazebo PASS.';$grid.Columns[1].FillWeight=145;$grid.Columns[0].FillWeight=75;$grid.Columns[14].FillWeight=110
$null=Label-At $form 'Пять matched reference-профилей: source68, 0 новых PPO. Это native-профили с измеренным началом Gazebo; не новые полёты Gazebo и не среднее original24.' 20 765 1360 22 8
$profilesGrid=New-Grid 790 137 @('Профиль','Сценарий','Ref0 finite','Ref0 peak','A latest finite','A latest peak','A pass','B latest finite','B latest peak','B pass')
$profilesGrid.Columns[0].FillWeight=160;$foot=Label-At $form '' 20 933 1360 42 8
$tooltips=New-Object Windows.Forms.ToolTip

function Paint-Snapshot {
    try{
        $snapshot=@(Snapshot);$b=$snapshot[0];$a=$snapshot[1];$planned=Number (Field $b.budget 'ppo_steps');if($null -eq $planned -or $planned -le 0){$planned=262144}
        foreach($chart in @($rewardChart,$qualityChart,$matchedChart)){$chart.ChartAreas['main'].AxisX.Maximum=$planned;$chart.ChartAreas['main'].AxisX.Interval=$planned/4}
        for($i=0;$i -lt 2;$i++){
            $arm=$snapshot[$i];$cards[$i].title.Text=$arm.display_name
            $stage=switch($arm.stage){'critic'{'WARMUP'}'ppo'{'PPO'}'validation'{'EVAL24'}'matched_profile_validation'{'EVAL5'}'archive'{'ARCHIVE'}'complete'{'COMPLETE ARCHIVE'}'error'{'ERROR'}default{'WAIT'}}
            $cards[$i].status.Text=if($arm.stage -in @('error','archive','complete')){$arm.state}else{$stage+' · '+$arm.state}
            $cards[$i].status.ForeColor=if($arm.error_reason){[Drawing.Color]::Firebrick}elseif($i -eq 0 -and $arm.state -like 'ОБНОВЛЯЕТСЯ*'){[Drawing.Color]::SeaGreen}else{[Drawing.Color]::DimGray}
            $cards[$i].counters.Text=if($i -eq 0 -and $arm.stage -ne 'complete'){'Записанные шаги среды critic/PPO: '+(Format-Count $arm.critic_env)+' / '+(Format-Count $arm.ppo_env)+"`r`n"+'Реальные Adam critic/PPO: '+(Format-Count $arm.critic_updates)+' / '+(Format-Count $arm.ppo_updates)}else{'Архивные шаги среды critic/PPO: '+(Format-Count $arm.critic_env)+' / '+(Format-Count $arm.ppo_env)+"`r`n"+'Архивные Adam critic/PPO: '+(Format-Count $arm.critic_updates)+' / '+(Format-Count $arm.ppo_updates)+' · новых сейчас: 0'}
            $cards[$i].heartbeat.Text=if($i -eq 0){'Heartbeat UTC: '+$arm.heartbeat_utc}else{'A — завершённый V24 κ=0.1; новых training/Adam updates: 0.'}
            $hash=$arm.evaluated_policy_sha256;$cards[$i].hash.Text='Последний оценённый / выбранный NPZ: '+$(if($hash){$hash.Substring(0,[Math]::Min(18,$hash.Length))+'…'}else{'ещё нет'})
            $tooltips.SetToolTip($cards[$i].hash,[string]$hash);$tooltips.SetToolTip($cards[$i].counters,$arm.counter_source);$tooltips.SetToolTip($cards[$i].status,$arm.error_reason+"`r`n"+$arm.error_source)
            $s=$rewardChart.Series["reward$i"];$s.Points.Clear();foreach($point in $arm.ppo_reward_points){[void]$s.Points.AddXY([double]$point.x,[double]$point.y)}
            foreach($condition in @('nominal','strong','gust')){$s=$qualityChart.Series["quality${i}_$condition"];$s.Points.Clear();foreach($row in $arm.evaluations){$value=Field $row $condition;if($null -ne $value -and $null -ne $row.timesteps){[void]$s.Points.AddXY([double]$row.timesteps,[double]$value)}}}
            foreach($field in @('matched_worst_finite','matched_worst_peak')){$s=$matchedChart.Series["matched${i}_$field"];$s.Points.Clear();foreach($row in $arm.evaluations){$value=Field $row $field;if($null -ne $value -and $null -ne $row.timesteps){[void]$s.Points.AddXY([double]$row.timesteps,[double]$value)}}}
        }
        foreach($condition in @('nominal','strong','gust')){$s=$qualityChart.Series['base_'+$condition];$s.Points.Clear();$value=Field $b.baseline_reference $condition;if($null -ne $value){[void]$s.Points.AddXY(0,[double]$value)}}
        $headingLabel.Text=(Heading-Text $b.heading_critic 'Critic snapshot')+"`r`n"+(Heading-Text $b.heading_ppo 'PPO snapshot')
        $auxLabel.Text='metric_cost одинаковый; κ='+[string]$b.prior_weight+' только в loss; реальных Adam PPO: '+(Format-Count $b.ppo_updates)+'. Offline IL=0; strong hover отдельно, предел Γ=0.0507м.'
        $grid.Rows.Clear()
        foreach($arm in @($a,$b)){foreach($row in $arm.evaluations){
            $hash=if($row.policy_sha256){$row.policy_sha256.Substring(0,10)+'…'}else{'—'}
            $index=$grid.Rows.Add([string[]]@($row.role,$row.checkpoint,(Format-Count $row.timesteps),(Format-Metric $row.nominal),(Format-Metric $row.strong),(Format-Metric $row.gust),(Format-Metric $row.strong_hover),(Flag $row.guards_pass),(Flag $row.original24_admission),(Flag $row.matched5_pass),(Flag $row.combined_admission),(Flag $row.engineering_eligibility),(Format-Metric $row.matched_worst_finite),(Format-Metric $row.matched_worst_peak),$hash))
            if($row.role -like 'A*'){$grid.Rows[$index].DefaultCellStyle.BackColor=[Drawing.Color]::FromArgb(237,240,246)}
            foreach($column in @(7,8,9,10,11)){$cell=$grid.Rows[$index].Cells[$column];$cell.Style.ForeColor=if($cell.Value -eq 'FAIL'){[Drawing.Color]::Firebrick}elseif($cell.Value -eq 'PASS'){[Drawing.Color]::SeaGreen}else{[Drawing.Color]::DimGray}}
            $grid.Rows[$index].Cells[14].ToolTipText=[string]$row.policy_sha256
            $checkText=@();foreach($entry in @($row.original_hard_checks.PSObject.Properties)){$checkText+=($entry.Name+': '+(Flag $entry.Value))}
            $grid.Rows[$index].Cells[8].ToolTipText=($checkText -join "`r`n")
            if($null -ne $row.strong_hover -and $row.strong_hover -gt 0.050697684650604764){$grid.Rows[$index].Cells[6].Style.ForeColor=[Drawing.Color]::Firebrick}
            $grid.Rows[$index].Cells[6].ToolTipText='Strong hover goal-RMSE: абсолютный Γ cap 0.0506977м; own-hover1.10 отдельно в Admit24.'
        }}
        $profilesGrid.Rows.Clear();$lastA=@($a.evaluations|Where-Object {$_.matched_episodes.Count -eq 5})[-1];$lastB=@($b.evaluations|Where-Object {$_.matched_episodes.Count -eq 5})[-1]
        foreach($reference in $b.matched_reference_profiles){
            $ra=@($lastA.matched_episodes|Where-Object {(Profile-Key $_) -eq $reference.key})[0];$rb=@($lastB.matched_episodes|Where-Object {(Profile-Key $_) -eq $reference.key})[0]
            $ga=Field $ra 'geometry';$gb=Field $rb 'geometry'
            [void]$profilesGrid.Rows.Add([string[]]@($reference.intervention,$reference.scenario,(Format-Metric $reference.finite),(Format-Metric $reference.peak),(Format-Metric (Number (Field $ga 'segment_rmse_m'))),(Format-Metric (Number (Field $ga 'active_segment_peak_m'))),(Flag (Field $ra 'profile_admission_pass')),(Format-Metric (Number (Field $gb 'segment_rmse_m'))),(Format-Metric (Number (Field $gb 'active_segment_peak_m'))),(Flag (Field $rb 'profile_admission_pass'))))
        }
        $warning=if($script:ReadWarnings.Count){' · '+($script:ReadWarnings.ToArray() -join '; ')}else{''};$errors=@($snapshot|Where-Object {$_.error_reason}|ForEach-Object {$_.display_name+': '+$_.error_reason});if($errors.Count){$warning+=' · ERROR '+($errors -join '; ')}
        $displayPath=$b.path;$projectPrefix=$PSScriptRoot.TrimEnd('\')+'\'
        if($displayPath.StartsWith($projectPrefix,[StringComparison]::OrdinalIgnoreCase)){$displayPath='.\'+$displayPath.Substring($projectPrefix.Length)}
        $foot.Text='B: '+$displayPath+"`r`n"+'Чтение '+[DateTimeOffset]::Now.ToString('HH:mm:ss')+' · A архив κ0.1 /0new SGD · COMPLETE ARCHIVE не является LIVE; окно ничего не запускает и не останавливает.'+$warning
    }catch{$foot.Text='Чтение будет повторено: '+$_.Exception.Message+"`r`n"+'Источник B: '+$RunRoot}
}
$timer=New-Object Windows.Forms.Timer;$timer.Interval=1500;$timer.Add_Tick({Paint-Snapshot})
$script:QaFailure=$null;$qaTimer=$null
if($QaSeconds -gt 0){$qaTimer=New-Object Windows.Forms.Timer;$qaTimer.Interval=$QaSeconds*1000;$qaTimer.Add_Tick({$qaTimer.Stop();try{Paint-Snapshot;if($SnapshotPath){$bitmap=New-Object Drawing.Bitmap($form.Width,$form.Height);try{$form.DrawToBitmap($bitmap,(New-Object Drawing.Rectangle(0,0,$form.Width,$form.Height)));$bitmap.Save($SnapshotPath,[Drawing.Imaging.ImageFormat]::Png)}finally{$bitmap.Dispose()}}}catch{$script:QaFailure=$_.Exception.Message}finally{$form.Close()}})}
$form.Add_Shown({Paint-Snapshot;$timer.Start();if($null -ne $qaTimer){$qaTimer.Start()}})
$form.Add_FormClosed({$timer.Stop();$timer.Dispose();if($null -ne $qaTimer){$qaTimer.Stop();$qaTimer.Dispose()}})
[void]$form.ShowDialog();$tooltips.Dispose();$form.Dispose();if($script:QaFailure){throw $script:QaFailure}

