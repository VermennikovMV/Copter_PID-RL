param(
    [ValidateSet('phase29','pid')][string]$Controller = 'phase29',
    [ValidateSet('nominal','force_mild','force_strong','gust')][string]$Scenario = 'force_strong',
    [ValidateSet('hover','route')][string]$Mission = 'route',
    [ValidateSet(83301,84001)][int]$Seed = 83301,
    [ValidateRange(0,2)][int]$Instance = 0,
    [switch]$NoGui,
    [switch]$VerifyOnly,
    [switch]$PlanOnly,
    [Alias('h','?')][switch]$Help
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($Help) {
    Write-Host @'
Gazebo GUI для самостоятельной видеозаписи:
  .\show_joint_phase29.cmd
  .\show_joint_phase29.cmd -Controller pid
  .\show_joint_phase29.cmd -Scenario nominal -Mission route
  .\show_joint_phase29.cmd -Scenario gust -Mission hover
  .\show_joint_phase29.cmd -Seed 84001 -Scenario force_strong -Mission route

По умолчанию: phase29, сильный ветер, маршрут, seed83301.
Маршрут40с; отдельное зависание30с. Видео записываете сами.
phase29: F0 без наблюдаемого отрезка, обученный prior0 FINAL при valid_leg.
Одна общая память; после маршрута удержание продолжает route-эксперт.
Композиция ручная, нового SGD в v29 нет. PID остаётся замороженным.

-PlanOnly показывает команду; -VerifyOnly проверяет локальные SHA.
Оба режима работают до ROOT binding и не запускают WSL/Gazebo/очередь.
Обычный запуск доступен после свежих9 полётов и независимого ROOT binding.
-NoGui: без окна; -Instance 0..2: свободная полоса.
Seeds83301/84001 уже просмотрены при разработке; демонстрация не новая реплика.
'@
    exit 0
}
if ($PlanOnly -and $VerifyOnly) { throw 'Выберите один режим: PlanOnly или VerifyOnly.' }
function Get-Phase29Path([string]$Relative) {
    if ([IO.Path]::IsPathRooted($Relative) -or $Relative.Contains(':')) { throw 'Ожидается относительный путь проекта.' }
    $resolved = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Relative))
    $prefix = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase) -or -not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "Отсутствует файл проекта: $Relative"
    }
    return $resolved
}
function Get-Phase29SHA([string]$Relative) {
    return (Get-FileHash -LiteralPath (Get-Phase29Path $Relative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Get-Phase29Json([string]$Relative) {
    return Get-Content -LiteralPath (Get-Phase29Path $Relative) -Encoding UTF8 -Raw | ConvertFrom-Json
}
function Convert-Phase29Linux([string]$Value) {
    if ($Value -notmatch '^([A-Za-z]):\\(.*)$') { throw "Неподдерживаемый путь: $Value" }
    return "/mnt/$($Matches[1].ToLowerInvariant())/$($Matches[2].Replace('\','/'))"
}
$planName = 'docs/joint_phase29_manual_v1/manual_plan.DRAFT.json'
$bindingName = 'docs/joint_phase29_manual_v1/root_binding.json'
$plan = Get-Phase29Json $planName
if ($plan.schema -ne 'joint-phase29-manual-plan-v1.0' -or $plan.observation_contract -ne 'measured-shared-phase-expert-v29.0' -or $plan.automatic_video_recording -ne $false) { throw 'Неверный план демонстрации phase29.' }
$critical = @{
    'sim_joint/run_experiment_v8_phase_expert.py'='825edf2d9674517dd6d11c650c3772e6da825841245f3aa0faf95911db23d6cd'
    'artifacts/refinement/joint_v2/phase_expert_preparation_v29_root01/bundle.json'='c3fc1c9c799f21200c98fc5995dba9908b1b13a517eb78bd60759e91eb8818a4'
    'artifacts/refinement/joint_v2/route_memory_prototype_v20/root02/zero_route_integral/policy.npz'='f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461'
    'artifacts/refinement/joint_v2/prior_ablation_pilot_v28_111/prior0/final.npz'='ce86bb56d7e0f5c1775266d4a45cd3417e979303d27cc662a58e8070bf34e345'
    'artifacts/gazebo/native_pid_tuning_v1/frozen_native_pid.json'='086eeea17bf39436246f4638aa0d8bbb81cabfb2e873413488ce58b741fb29bd'
}
foreach ($name in $critical.Keys) {
    if ($plan.files_sha256.$name -ne $critical[$name]) { throw "Неверная привязка: $name" }
}
foreach ($entry in $plan.files_sha256.PSObject.Properties) {
    if ($entry.Value -notmatch '^[a-f0-9]{64}$' -or (Get-Phase29SHA $entry.Name) -ne $entry.Value) { throw "Изменился источник: $($entry.Name)" }
}
$bound = $false
$binding = $null
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot $bindingName) -PathType Leaf) {
    $binding = Get-Phase29Json $bindingName
    if ($binding.schema -ne 'joint-phase29-manual-root-binding-v1.0' -or $binding.status -ne 'ROOT_BOUND_FRESH9_DEVELOPMENT_DEMONSTRATION' -or $binding.manual_plan_sha256 -ne (Get-Phase29SHA $planName) -or $binding.expert_bundle_sha256 -ne $critical['artifacts/refinement/joint_v2/phase_expert_preparation_v29_root01/bundle.json'] -or $binding.final_applicability_confirmed -ne $false) { throw 'Неверный ROOT binding phase29.' }
    foreach ($entry in $binding.files_sha256.PSObject.Properties) {
        if ((Get-Phase29SHA $entry.Name) -ne $entry.Value) { throw "Изменился ROOT-bound файл: $($entry.Name)" }
    }
    $complete = Get-Phase29Json $binding.complete.path
    $audit = Get-Phase29Json $binding.independent_audit.path
    if ($complete.status -ne 'COMPLETE' -or $complete.new_actual_flights -ne 9 -or @($complete.results).Count -ne 9 -or $audit.status -ne 'PASS' -or $audit.new_actual_flights -ne 9 -or $audit.policy_sha256 -ne $binding.expert_bundle_sha256) { throw 'Требуются завершённые свежие9 полётов и независимый аудит.' }
    $bound = $true
}
if (-not $PSBoundParameters.ContainsKey('Instance') -and $bound) {
    $matched = @($binding.evidence_cases | Where-Object { $_.environment_seed -eq $Seed -and $_.scenario -eq $Scenario -and $_.mission -eq $Mission })
    if ($matched.Count -eq 1) { $Instance = [int]$matched[0].instance }
}
$Controller = $Controller.ToLowerInvariant()
$runtimeController = if ($Controller -eq 'phase29') { 'hybrid' } else { 'pid' }
$duration = if ($Mission -eq 'route') { 40 } else { 30 }
$linuxRoot = Convert-Phase29Linux $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$suffix = [Guid]::NewGuid().ToString('N').Substring(0,8)
$relative = "artifacts/user_runs/${stamp}_${suffix}_phase29_${Controller}_${Scenario}_${Mission}"
$linuxOutput = Convert-Phase29Linux (Join-Path $PSScriptRoot $relative)
$runtimeArgs = @('sim_joint/run_experiment_v8_phase_expert.py','--controller',$runtimeController,'--scenario',$Scenario,'--mission',$Mission,'--seed',"$Seed",'--instance',"$Instance",'--split','demo','--duration',"$duration",'--pid-profile',"$linuxRoot/artifacts/gazebo/native_pid_tuning_v1/frozen_native_pid.json",'--hover-thrust','.302651215314865','--thrust-semantics','body_collective','--adapter-attitude-max-age-s','.1','--basis-kind','velocity_residual','--conditioning-mode','zero_route_integral','--output',"$linuxOutput/runs/flight")
if ($Controller -eq 'phase29') { $runtimeArgs += @('--expert-bundle',"$linuxRoot/artifacts/refinement/joint_v2/phase_expert_preparation_v29_root01/bundle.json",'--bundle-sha256','c3fc1c9c799f21200c98fc5995dba9908b1b13a517eb78bd60759e91eb8818a4','--train-seed','111') }
$activation = if ($bound) { 'ROOT_BOUND' } else { 'PENDING_FRESH9_AND_ROOT_BINDING' }
if ($VerifyOnly) {
    [ordered]@{status='LOCAL_FILES_VERIFIED';activation=$activation;new_actual_flights=0;WSL_started=$false;queue_acquired=$false;NN_forwards=0} | ConvertTo-Json
    exit 0
}
if ($PlanOnly) {
    [ordered]@{status='READONLY_PLAN';activation=$activation;controller=$Controller;scenario=$Scenario;mission=$Mission;seed=$Seed;instance=$Instance;horizon_s=$duration;show_Gazebo_GUI=(-not $NoGui);automatic_video_recording=$false;command=$runtimeArgs;GUI_helper='scripts/joint_gazebo_gui_v2.py';new_actual_flights=0;WSL_started=$false;queue_acquired=$false;scope='Viewed development demonstration; no new independent replication or policy promotion'} | ConvertTo-Json -Depth 5
    exit 0
}
if (-not $bound) { throw 'Запуск phase29 ожидает ROOT binding после свежих9 полётов. PlanOnly и VerifyOnly доступны сейчас.' }
Write-Host "Gazebo: $Controller, $Scenario, $Mission, seed$Seed, полоса$Instance, измерение${duration}с."
if ($Controller -eq 'phase29') { Write-Host 'phase29: F0 до наблюдаемого отрезка, затем обученный route FINAL; одна общая память, новый SGD=0.' }
else { Write-Host 'PID: исходный замороженный профиль сравнения.' }
Write-Host $binding.audience_note
Write-Host 'Это просмотренная разработочная среда, а не независимая реплика. Видео запишите сами.'
$python = @'
import datetime as dt,fcntl,hashlib,json,os,pathlib,signal,subprocess,sys
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);mode,scenario,mission=sys.argv[3:6]
seed,lane=int(sys.argv[6]),int(sys.argv[7]);no_gui=sys.argv[8]=='1'
for folder in (root,root/'scripts',root/'audit'):sys.path.insert(0,str(folder))
from scripts import joint_memory20_presentation_v1 as helpers
from joint_gazebo_screen_v2 import process_preflight
from scripts.build_joint_development_registry_v2 import verify_external
from run_joint_path77_smoke_v3 import scheduled_force_proof,physical_audit,audit_metrics,from_gazebo_run
from verify_joint_hybrid_v8_phase_expert import audit as actor_audit
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
plan=read(root/'docs/joint_phase29_manual_v1/manual_plan.DRAFT.json')
binding_path=root/'docs/joint_phase29_manual_v1/root_binding.json';binding=read(binding_path)
assert binding['status']=='ROOT_BOUND_FRESH9_DEVELOPMENT_DEMONSTRATION'
assert sha(root/'docs/joint_phase29_manual_v1/manual_plan.DRAFT.json')==binding['manual_plan_sha256']
for name,digest in {**plan['files_sha256'],**binding['files_sha256']}.items():assert sha(root/name)==digest,name
verify_external(dict(external_runtime=binding['external_runtime']))
base=(root/'artifacts/user_runs').resolve();assert out.resolve().is_relative_to(base) and out.resolve()!=base and not out.exists()
run=out/'runs/flight';duration=40 if mission=='route' else 30
command=[sys.executable,'-B',str(root/'sim_joint/run_experiment_v8_phase_expert.py'),'--controller','hybrid' if mode=='phase29' else 'pid','--scenario',scenario,'--mission',mission,'--seed',str(seed),'--instance',str(lane),'--split','demo','--duration',str(duration),'--pid-profile',str(root/'artifacts/gazebo/native_pid_tuning_v1/frozen_native_pid.json'),'--hover-thrust','.302651215314865','--thrust-semantics','body_collective','--adapter-attitude-max-age-s','.1','--basis-kind','velocity_residual','--conditioning-mode','zero_route_integral','--output',str(run)]
if mode=='phase29':command+=['--expert-bundle',str(root/'artifacts/refinement/joint_v2/phase_expert_preparation_v29_root01/bundle.json'),'--bundle-sha256',binding['expert_bundle_sha256'],'--train-seed','111']
def helper(kind,script,extra):return ['/mnt/c/Windows/System32/wsl.exe','-d','Ubuntu','-u','misha3','--','/usr/bin/python3','-B','-c',helpers.OWNED_CHILD,str(out/(kind+'_owned_linux_pid.json')),'/usr/bin/python3','-B',str(root/script),'--run',str(run),*extra]
observer_command=helper('observer','scripts/independent_pose_recorder.py',['--flush-gate',str(out/'observer_flush_gate')])
gui_command=None if no_gui else helper('gui','scripts/joint_gazebo_gui_v2.py',[])
with (root/'artifacts/refinement/joint_v2/.gazebo_native_queue.lock').open('a+') as lease:
    fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);queue=process_preflight()
    out.mkdir();(out/'runs').mkdir();(out/'audits').mkdir()
    write(out/'plan.json',dict(controller=command,observer=observer_command,gui=gui_command,root_binding_sha256=sha(binding_path),automatic_video_recording=False,measurement_horizon_s=duration,development_only=True))
    processes={};logs=[];error=None;code=None;reports={}
    try:
        for kind,cmd in (('observer',observer_command),('gui',gui_command),('controller',command)):
            if cmd is None:continue
            stream=(out/(kind+'.log')).open('x');logs.append(stream)
            processes[kind]=subprocess.Popen(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=kind=='controller')
        code=processes['controller'].wait(timeout=360)
    except BaseException as exc:error=repr(exc)
    finally:
        controller=processes.get('controller')
        if controller is not None and controller.poll() is None:
            os.killpg(controller.pid,signal.SIGINT)
            try:controller.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(controller.pid,signal.SIGKILL);controller.wait()
        (out/'observer_flush_gate').write_text('flush\n')
        if run.exists():(run/'STOP_GUI_CLIENT').write_text('stop\n')
        for kind in ('observer','gui'):
            process=processes.get(kind)
            if process is None:continue
            try:process.wait(timeout=105)
            except subprocess.TimeoutExpired:
                try:helpers.stop_owned(process,out/(kind+'_owned_linux_pid.json'),run,kind)
                except Exception as exc:error=error or repr(exc)
                error=error or (kind+' timeout; attempt retained')
            if process.returncode!=0:error=error or (kind+' helper failed')
        for stream in logs:stream.close()
    result={}
    try:
        if error:raise RuntimeError(error)
        summary=read(run/'summary.json');assert summary['status'] in ('PASS','FAIL') and code==(0 if summary['status']=='PASS' else 1)
        assert summary['joint_interface']['runner_sha256']==plan['files_sha256']['sim_joint/run_experiment_v8_phase_expert.py']
        from quadrl.protocol import scenario_spec
        assert summary['scenario_spec']==scenario_spec(scenario,seed)
        reports=dict(force=scheduled_force_proof(run,summary,summary['scenario_spec']),physical=physical_audit(run))
        reports['metric']=audit_metrics(run,summary,reports['physical'])
        if mode=='phase29' and summary['status']=='PASS':reports['actor']=actor_audit(run,expected_kind='velocity_residual',expected_mode='zero_route_integral',expected_bundle_sha256=binding['expert_bundle_sha256'])
        for kind,report in reports.items():write(out/'audits'/(kind+'.json'),report);assert report['status']=='PASS',kind
        if mission=='route':write(out/'audits/geometry.json',from_gazebo_run(run))
        if not no_gui:
            gui=read(run/'gui_client_state.json');marker=read(run/'joint_current_target_state.json')
            assert gui['status']==marker['status']=='PASS'
            if mission=='route':assert marker['target_updates']==5
        result=dict(controller_status=summary['status'],audits={name:value['status'] for name,value in reports.items()},queue_release=process_preflight())
    except BaseException as exc:error=repr(exc)
    receipt=dict(schema='joint-phase29-manual-demo-v1.0',status='ERROR' if error else 'COMPLETE',utc=dt.datetime.now(dt.timezone.utc).isoformat(),error=error,controller_returncode=code,result=result,root_binding_sha256=sha(binding_path),new_actual_flights=int((run/'summary.json').exists()),automatic_video_recording=False,independent_replication_claimed=False,policy_promotion=False)
    write(out/'complete.json',receipt);print(json.dumps(receipt),flush=True)
    if error:raise RuntimeError(error)
'@
$noGuiFlag = if ($NoGui) { '1' } else { '0' }
$python | & wsl.exe -d Ubuntu-24.04 -u misha --cd $linuxRoot -- [external-home]/vkr_sim/venv-cu128/bin/python -B - $linuxRoot $linuxOutput $Controller $Scenario $Mission "$Seed" "$Instance" $noGuiFlag
$code = $LASTEXITCODE
Write-Host "Файлы демонстрации: $relative"
exit $code
