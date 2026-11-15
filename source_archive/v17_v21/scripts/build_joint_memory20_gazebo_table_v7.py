"""Render only a COMPLETE independently audited exact memory20 Gazebo matrix.

Composes the frozen v6 scalar/gate helpers. No controller, environment, actor
inference or simulation is executed. Prior reports are immutable inputs only.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import build_joint_metric_cost16_gazebo_v6 as common
import matplotlib.pyplot as plt

OUT=ROOT/'reports/joint_memory20_gazebo_v7'
AUDIT=ROOT/'audit/joint_memory_development9_v20_root01.json'
CAMPAIGN=ROOT/'artifacts/refinement/joint_v2/gazebo_memory_development9_v20_root01'
PLAN_SHA='49228743db8afc64de5e9553e278d3efb129631de074d45d2a8ee4e6a002669a'
POLICY='f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461'
SOURCE_POLICY='68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5'
SOURCE_AUDIT=ROOT/'audit/joint_route80_development9_v5_root01.json'
SOURCE_AUDIT_SHA='5b917f1b2ce55025878f0e7e4b8c62a95852c3a1652d6def70ad9216ad2d5b74'
NATIVE=ROOT/'artifacts/refinement/joint_v2/route_memory_baseline_v20_root01/zero_route_integral/evaluation.json'
NATIVE_SHA='f0f6930a58d690f1c474c4ff954575c0af84e9e9c7f7d5cb66a266b0b82cbb9e'
OLD=ROOT/'reports/joint_metric_cost16_gazebo_v6'
CHECKS=0


def require(ok,why):
    global CHECKS
    CHECKS+=1
    if not ok:raise ValueError(why)


def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def read(path):return common.read(path)
def sha(path):return common.sha(Path(path))
def ck(case):return tuple(case['key'])


def status_text(row):
    if row['quality_status']=='PASS':return 'Пройдены'
    names={'finite_segment_RMSE_m':'точность пути','mean_corner_peak_m':'повороты','absolute_active_segment_peak_m':'максимум пути',
           'restricted_route_time_s':'время','penalized_goal_RMSE_m':'ошибка цели','wind_hover_gain_vs_PID':'удержание',
           'wind_hover_retention':'сохранение удержания','heading_RMS_deg':'курс','heading_peak_deg':'пик курса'}
    failed=row['failed_checks'].split('; ');labels=list(dict.fromkeys(names.get(name,'проверки') for name in failed))
    return 'Не пройдены:\n'+(' / '.join(labels) if len(labels)<3 else str(len(failed))+' проверок')


def table_figure(rows,headers,values,filename,title,subtitle,summary,notes,widths):
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig=plt.figure(figsize=(20,11.25),facecolor='white')
    fig.text(.025,.95,title,fontsize=23,weight='bold',color='#143746')
    fig.text(.025,.905,subtitle,fontsize=12.7,color='#3b5260')
    fig.text(.025,.853,summary,fontsize=12.5,color='#3b5260',linespacing=1.45)
    ax=fig.add_axes([.025,.27,.95,.515]);ax.axis('off')
    table=ax.table(cellText=values,colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(10.8)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#d4dee4');cell.set_linewidth(.7)
        if i==0:cell.set_facecolor('#153e50');cell.set_text_props(color='white',weight='bold',fontsize=10.7)
        else:
            passed=rows[i-1]['quality_status']=='PASS';cell.set_facecolor('#edf7f0' if passed else '#fff0ee')
            if j==0:cell.set_text_props(ha='left')
            if i==len(rows):cell.set_edgecolor('#9a7549');cell.set_linewidth(1.15)
            if j==len(headers)-1:cell.set_text_props(weight='bold',color='#286341' if passed else '#9d302b',fontsize=10.0)
    fig.text(.025,.225,notes,fontsize=9.6,color='#3b5260',va='top',linespacing=1.5)
    for ext in ('png','pdf'):fig.savefig(OUT/(filename+'.'+ext),dpi=128,facecolor='white')
    plt.close(fig)


def render(rows,macro,summary):
    num=common.num;values=[]
    for row in rows:
        data=[('Удержание' if row['mission']=='hover' else 'Маршрут')+'\n'+row['condition'],str(row['environment_seed'])]
        for field,digits in (('goal_RMSE_m',3),('route_time_s',2),('finite_segment_RMSE_m',3),('mean_corner_peak_m',3),('absolute_peak_m',3)):
            data.append(num(row['joint_'+field],digits)+' / '+num(row['PID_'+field],digits))
        values.append(data+[status_text(row)])
    head=['Миссия / воздействие','Среда','Ошибка цели\nRMSE, м','Маршрут\nс','Конечный путь\nRMSE, м','Повороты\nсредний пик, м','Максимум\nотклонения, м','Заданные\nдопуски']
    primary=summary['primary_passed_cells'];passed=summary['passed_checks'];tail='пройден' if summary['tail_passed'] else 'не пройден'
    caption=f'Основные условия: {primary}/8 по инженерным допускам; отдельный попутный тест: {tail}. Проверки: {passed}/90.\n'
    caption+=f'Шесть ветровых ячеек: выигрыш {num(100*macro["relative_improvement"],2)}% по исходной RMSE цели (порог 15%): {macro["status"]}. Это не число побед над PID.'
    notes=(
        'Весовые коэффициенты сети 68dafd98… сохранены. Обнуляются три компоненты интеграла при известном отрезке (включая финальное удержание); новых PPO updates: 0.\n'
        '9 новых RL-полётов + 9 общих PID + 2 общих контрольных удержания = 20 наблюдений; не 20 новых повторов. Один просмотренный случай на условие.\n'
        'Основная среда 83301; попутный стресс 84001* выделен отдельно. Независимого финального теста нет; масса, задержки, мотор и шум этой моделью не проверены.\n'
        'Путь: конечный активный 3D-отрезок PRE-target без финального удержания; повороты: три окна по 1 с. Силы в Н, не м/с; фазы воздействия на поворотах различаются.\n'
        'Новая карта относится только к conditioning ZERO (NPZ f0e87341…). Эффект обработки памяти не называется новым обучением PPO; результаты последующего V21 сюда не входят.')
    table_figure(rows,head,values,'presentation','RL + PID: применимость после коррекции памяти ошибки',
        'Gazebo · гибрид RL с внутренним PID против штатного PID · прежние веса, другой входной интеграл · ZERO f0e87341…',caption,notes,[.21,.055,.11,.095,.13,.155,.115,.13])
    thesis=[]
    for row in rows:
        data=[('Удержание' if row['mission']=='hover' else 'Маршрут')+'\n'+row['condition'],str(row['environment_seed'])]
        for field,digits in (('goal_RMSE_m',3),('target_relative_max_downward_error_m',3),('recovery_time_s',3),('route_time_s',2)):
            data.append(num(row['joint_'+field],digits)+' / '+num(row['PID_'+field],digits))
        data.append(('Да' if row['joint_survived'] else 'Нет')+' / '+('Да' if row['PID_survived'] else 'Нет'))
        data.append(num(row['joint_heading_RMS_deg'],2)+' / '+num(row['PID_heading_RMS_deg'],2))
        thesis.append(data+[status_text(row)])
    table_figure(rows,['Миссия / воздействие','Среда','Ошибка цели\nRMSE, м','Недобор высоты\nк текущей цели, м','Восстановление\nпосле OFF, с','Маршрут\nс','Полёт\nсохранён','Курс RMS\nградусы','Все\nдопуски'],thesis,'thesis_metrics',
        'Пять исходных метрик ВКР: измерения в Gazebo','Точная ZERO f0e87341… · пары RL / PID · PRE-target, полные окна удержания 30 с / маршрута 40 с',
        'Геометрия пути и поворотов показана на основной карте; исходная ошибка цели не является cross-track.\nВысота считается относительно текущей цели; восстановление применимо только к удержанию.',
        'На маршруте цель меняет высоту 3 → 3,5 → 3 м: недобор к новой цели не равен незаданной просадке аппарата.\n'
        'Удержание уже внутри полосы в момент OFF даёт малое время восстановления; это не доказательство мгновенной аэродинамической адаптации.\n'
        'Все исходные значения, censored-флаги, пиковый курс и force-аудиты сохранены в CSV/JSON. Итоговый допуск — конъюнкция, отдельная метрика не компенсирует провал другой.\n'
        '9 новых полётов + 11 общих контрольных наблюдений; n=1 development. Одна политика ZERO; не финальная независимая проверка и не новый выигрыш PPO.',
        [.21,.05,.105,.13,.12,.08,.09,.105,.11])


def render_conditioning_comparison(comparisons):
    routes=[r for r in comparisons if r['mission']=='route']
    require(len(routes)==5,'All five route conditions in comparison, no successful-only selection')
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig=plt.figure(figsize=(20,11.25),facecolor='white')
    fig.text(.025,.95,'Что изменилось: те же веса RL, другая память ошибки',fontsize=24,weight='bold',color='#143746')
    fig.text(.025,.905,'Новая версия ZERO и измеренная исходная сеть 68dafd98… · новых PPO / Adam updates: 0',fontsize=14,color='#3b5260')
    labels=[('Без силы' if r['scenario']=='nominal' else 'Малая сила' if r['scenario']=='force_mild' else 'Попутная*' if r['environment_seed']==84001 else 'Порывы' if r['scenario']=='gust' else 'Пост. сила')+'\n'+str(r['environment_seed']) for r in routes]
    for position,field,title in (([.06,.33,.41,.46],'finite_segment_RMSE_m','Отклонение от конечного 3D-пути, м'),([.55,.33,.41,.46],'mean_corner_peak_m','Средний пик на трёх поворотах, м')):
        ax=fig.add_axes(position)
        for offset,prefix,name,color in ((-.24,'PID_','PID','#7d8993'),(0.,'source68_','Исходная RL68','#c87a33'),(.24,'current_','RL68 + ZERO','#1d8585')):
            vals=[r[prefix+field] for r in routes];x=[i+offset for i in range(len(routes))]
            ax.bar(x,vals,width=.23,label=name,color=color)
        ax.set_xticks(list(range(len(routes))));ax.set_xticklabels(labels,fontsize=11)
        ax.set_ylim(bottom=0);ax.set_title(title,fontsize=15,pad=14);ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True);ax.legend(fontsize=10)
        ax.tick_params(axis='y',labelsize=11)
    notes=('Изменение — только обработка трёх входов интеграла ошибки (индексы 22, 23, 24) на своём наблюдаемом действующем маршруте. Веса сети сохранены.\n'
           'Показаны все пять измеренных маршрутных условий, включая отдельный попутный стресс 84001*. Новая модель не наследует исходные успешные ячейки.\n'
           'Исходная RL68 и PID — исторические просмотренные контроли; новые ZERO-полёты измерены заново. n=1, это не независимые повторения и не новый эффект PPO.\n'
           'Времена прохождения и фазы приложенной силы могут различаться. Полная приёмка требует одновременно точности, времени, курса, завершения и сохранения удержания.\n'
           'График объясняет измеренное изменение в ограниченной области; окончательной независимой проверки и подтверждения массы/моторов/задержек/шума нет.')
    fig.text(.025,.24,notes,fontsize=11,color='#3b5260',va='top',linespacing=1.6)
    for ext in ('png','pdf'):fig.savefig(OUT/('conditioning_effect.'+ext),dpi=128,facecolor='white')
    plt.close(fig)


def build(expected_audit_sha):
    require(not OUT.exists(),'Fresh output only; preserve existing maps')
    require((CAMPAIGN/'complete.json').exists(),'Actual matrix COMPLETE required')
    require(AUDIT.exists() and sha(AUDIT)==expected_audit_sha,'Bind the final independent audit SHA')
    require(sha(CAMPAIGN/'plan.json')==PLAN_SHA,'Prospective physical matrix unchanged')
    require(sha(SOURCE_AUDIT)==SOURCE_AUDIT_SHA and sha(NATIVE)==NATIVE_SHA,'Source68 and zero-update native references unchanged')
    audit=read(AUDIT);complete=read(CAMPAIGN/'complete.json');source=read(SOURCE_AUDIT);native=read(NATIVE)
    require(audit['status']=='PASS' and complete['status']=='COMPLETE','Independent PASS and complete actual producer required')
    require(audit['policy_sha256']==POLICY and audit['conditioning_mode']=='zero_route_integral','Exact conditioned controller')
    require(audit['new_actual_flights']==9 and audit['new_PID_flights']==0 and audit['total_unique_observation_coverage']==20,'9 new +11 shared,20 coverage')
    require(audit['policy_promotion'] is False and audit['final_applicability_confirmed'] is False,'No automatic promotion/final confirmation')
    require(native['training_transitions']==0 and native['optimizer_steps']==0 and native['fixed_development_guards_pass'] is True,'Native zero-update baseline passes; not new PPO')
    require(native['source_policy_sha256']==SOURCE_POLICY and source['policy_sha256']==SOURCE_POLICY,'Same trained source68 weights identity')
    inputs=dict(audit['input_sha256'])
    for path in (AUDIT,SOURCE_AUDIT,NATIVE,CAMPAIGN/'plan.json',CAMPAIGN/'complete.json',Path(__file__),ROOT/'scripts/build_joint_metric_cost16_gazebo_v6.py',*sorted(OLD.iterdir())):
        if path.is_file():inputs[rel(path)]=sha(path)
    for name,digest in inputs.items():require(sha(common.resolve(name))==digest,'Frozen input before:'+name)
    cases=copy.deepcopy(audit['cases']);joints,pids,gates=common.classify(cases,audit['gate_rows'])
    require(len(cases)==20 and len(joints)==len(pids)==9 and sum(c['new_actual_flight'] for c in cases)==9,'Exact observation denominators')
    primary=sum(c['quality_pass'] for c in joints if ck(c)[0]==83301);tail=all(c['quality_pass'] for c in joints if ck(c)[0]==84001)
    require(audit['mandatory_eight_condition_operational_gate']==('PASS' if primary==8 else 'FAIL'),'Independent primary classification')
    require(audit['additional_tailwind84001_operational_gate']==('PASS' if tail else 'FAIL'),'Independent extra stress classification')
    six=[dict(scenario=s,mission=m,joint_penalized_RMSE_m=next(c for c in joints if ck(c)==(83301,s,m))['metrics']['penalized_rmse_position_m'],PID_penalized_RMSE_m=pids[(83301,s,m)]['metrics']['penalized_rmse_position_m']) for s in ('force_mild','force_strong','gust') for m in ('hover','route')]
    jm=statistics.mean(r['joint_penalized_RMSE_m'] for r in six);pm=statistics.mean(r['PID_penalized_RMSE_m'] for r in six)
    macro=dict(status='PASS' if 1-jm/pm>=.15 else 'FAIL',coverage='ALL6_DEVELOPMENT_CELLS',relative_improvement=1-jm/pm,joint_mean_penalized_RMSE_m=jm,PID_mean_penalized_RMSE_m=pm,minimum_improvement=.15,cells=six,formula='1 - mean6_joint_penalized_RMSE / mean6_PID_penalized_RMSE')
    for field in ('status','relative_improvement','joint_mean_penalized_RMSE_m','PID_mean_penalized_RMSE_m'):common.close(macro[field],audit['original_thesis_wind_macro'][field],'Fresh six-cell macro:'+field)
    scientific='DEVELOPMENT_MEASURED_GATES_PASS' if primary==8 and tail and macro['status']=='PASS' else 'FAIL'
    require(scientific==audit['scientific_quality_status'],'Scientific gate preserved, even if failed')
    summary=dict(scientific_quality_status=scientific,primary_passed_cells=primary,primary_cells=8,tail_passed=tail,passed_checks=sum(r['pass_'] for r in gates),total_checks=90,checks=gates)
    old={ck(c):c for c in source['cases'] if c['role']=='joint80'};rows=[];comparisons=[];original=[];exposure=[]
    for c in joints:
        seed,scenario,mission=ck(c);p=pids[ck(c)];current=common.case_metrics(c);pm=common.case_metrics(p);before=common.case_metrics(old[ck(c)])
        row=dict(condition=common.label(c),mission=mission,environment_seed=seed,source_training_lineage=111,joint_new_flights=1,PID_new_flights=0,PID_shared_flights=1,
                 policy_sha256=POLICY,source_policy_sha256=SOURCE_POLICY,conditioning_mode='zero_route_integral',changed_effective_columns='22:25 only on own observed valid leg',new_optimizer_steps=0,backend='Gazebo',quality_status='PASS' if c['quality_pass'] else 'FAIL',failed_checks='; '.join(r['check'] for r in c['quality_checks'] if not r['pass_']))
        row.update({'joint_'+k:v for k,v in current.items()});row.update({'PID_'+k:v for k,v in pm.items()});row.update(measured_domain='n1 viewed development; primary83301 plus extra84001; not independent final confirmation')
        rows.append(row)
        comp=dict(environment_seed=seed,scenario=scenario,mission=mission,current_policy_sha256=POLICY,source68_policy_sha256=SOURCE_POLICY,same_network_weights=True,new_optimizer_steps=0)
        for field in ('goal_RMSE_m','route_time_s','finite_segment_RMSE_m','mean_corner_peak_m','absolute_peak_m'):
            new,oldvalue,pid=current[field],before[field],pm[field]
            comp.update({'current_'+field:new,'source68_'+field:oldvalue,'PID_'+field:pid,
                         'change_percent_vs_source68_'+field:100*(new/oldvalue-1) if new is not None and oldvalue else None,
                         'change_percent_vs_PID_'+field:100*(new/pid-1) if new is not None and pid else None})
        comp['scope']='Same source68 weights, input conditioning changed; descriptive historical viewed comparison, not fresh PPO gain or independent replication.';comparisons.append(comp)
    for c in cases:
        seed,scenario,mission=ck(c);original.append(dict(run=rel(common.resolve(c['run'])),role=c['role'],new_actual_flight=c['new_actual_flight'],environment_seed=seed,scenario=scenario,mission=mission,
            policy_sha256=POLICY if c['role']=='joint80' else None,**common.case_metrics(c),actual_final_force_OFF_s=c['force']['last_physical_off_s'],force_norm_N=common.magnitude(c),force_intervals=json.dumps(c['force']['physical_intervals']),
            quality_status=('PASS' if c['quality_pass'] else 'FAIL') if c['role']=='joint80' else 'REFERENCE',proof='Fresh independent whole physical/force/metrics/wire audit reused; no builder inference'))
        if mission=='route':
            g=c['geometry'];intervals=c['force']['physical_intervals']
            for corner in g['corners']:
                stamp=corner['transition_s'];vector=next((r['force_enu_N'] for r in intervals if r['on_s']<=stamp<r['off_s']),[0.,0.,0.])
                exposure.append(dict(run=rel(common.resolve(c['run'])),role=c['role'],environment_seed=seed,scenario=scenario,corner=corner['corner_index'],corner_time_s=stamp,
                    force_at_corner_enu_N=json.dumps(vector),force_norm_N=math.sqrt(sum(float(x)**2 for x in vector)),transit_exposed_seconds=sum(max(0.,min(g['transit_end_s'],r['off_s'])-max(g['transit_start_s'],r['on_s'])) for r in intervals),
                    complete_corner_window=corner['full_post_window'],scope='Absolute force time, not event-aligned matched corner exposure'))
        c['run']=rel(common.resolve(c['run']))
    require(len(rows)==9 and len(original)==20 and len(exposure)==30,'Exact presentation denominators')
    OUT.mkdir(parents=True)
    for filename,data in (('applicability.csv',rows),('all_original_flights.csv',original),('fixed_quality_checks.csv',gates),('comparison_to_source68_and_PID.csv',comparisons),('route_force_exposure.csv',exposure)):
        common.write_csv(OUT/filename,data)
    for filename,data in (('case_recomputation.json',cases),('original_thesis_wind_macro.json',macro),('operational_gate.json',summary),('native_zero_update_reference.json',dict(path=rel(NATIVE),sha256=NATIVE_SHA,training_transitions=0,optimizer_steps=0,native_guards_pass=True,source_policy_sha256=SOURCE_POLICY)),('input_freeze.json',dict(independent_whole_audit_sha256=expected_audit_sha,input_sha256=inputs))):
        common.write_json(OUT/filename,data)
    render(rows,macro,summary)
    render_conditioning_comparison(comparisons)
    (OUT/'README.txt').write_text(
        'Точная карта ZERO f0e87341…: гибрид RL с внутренним PID против штатного PID; та же обученная сеть68dafd98…, обнулены только эффективные integral22:25 на наблюдаемом valid-leg. Это три индекса22,23,24. Известный финальный leg сохраняется и в конечном удержании. Новых PPO/Adam updates0.\n'
        f'Primary83301: {primary}/8 условий; extra tail84001: {tail}; проверки {summary["passed_checks"]}/90. Полный measured status {scientific}, wind macro {macro["relative_improvement"]:.8%} / minimum15%.\n'
        '9 новых RL-полётов+9 общих PID+2 общих старых удержания=20наблюдений, не20новых. Все9 ячеек этой точной модели измерены заново, старые RL ячейки не перенесены.\n'
        'presentation.png/pdf: finite3D/corners/time/goal RMSE/quality. thesis_metrics.png/pdf: все5 исходных метрик ВКР и курс. conditioning_effect.png/pdf: все5маршрутных условий ZERO/source68/PID при прежних весах. Полные raw значения/censoring в CSV.\n'
        'Goal RMSE не cross-track; finite active PRE segment без final hold; три закрытых1s corner windows. Заданный высотный шаг не называется незаданной просадкой, already-inside OFF recovery не мгновенная адаптация.\n'
        'comparison_to_source68_and_PID.csv сравнивает те же веса при разной обработке памяти с историческим просмотренным контролем68; это не новое обучение PPO и не независимые повторения.\n'
        'Ньютон, не скорость ветра м/с. Абсолютное расписание силы означает разные фазы воздействия при разных временах поворота. Все physical/force/actor proofs берутся из полного независимого receipt.\n'
        '15%macro — согласованный практический критерий, новые geometry limits — фиксированные инженерные требования; они не приписываются численным порогам исходной ВКР.\n'
        'n1 development83301 + extra84001. Независимый финальный114flight тест не выполнен; масса/груз/мотор/задержка/шум/90deg yaw/hardware не подтверждены. Веса и таблицы V21 не подмешаны.\n',encoding='utf-8')
    for name,digest in inputs.items():require(sha(common.resolve(name))==digest,'Frozen input after:'+name)
    receipt=dict(schema='joint-memory20-gazebo-table-v7.0',status='PASS',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),checks=CHECKS+common.CHECKS,
                 source_builder_sha256=sha(Path(__file__)),independent_whole_audit_sha256=expected_audit_sha,policy_sha256=POLICY,source_policy_sha256=SOURCE_POLICY,
                 new_actual_flights=9,shared_PID_flights=9,shared_retention_flights=2,total_coverage=20,table_rows=9,route_geometries=10,corner_windows=30,fixed_checks=90,
                 primary_passed=primary,primary_total=8,tail_passed=tail,passed_checks=summary['passed_checks'],scientific_quality_status=scientific,
                 failed_quality_checks=[r for r in gates if not r['pass_']],original_thesis_wind_macro=macro,conditioning_mode='zero_route_integral',new_optimizer_steps=0,
                 policy_promotion=False,final_applicability_confirmed=False,input_files_unchanged=len(inputs),old_v6_files_preserved=len(list(OLD.iterdir())),
                 no_new_physics_training_or_actor_inference=True,visual_inspection='PENDING_ROOT_AND_SEPARATE_IMAGE_READBACK',output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    target=ROOT/'audit/joint_memory20_gazebo_v7_table_root01.json';common.write_json(target,receipt)
    print(json.dumps(dict(status='PASS',receipt=rel(target),sha256=sha(target),checks=receipt['checks'],primary=str(primary)+'/8',tail=tail,macro=macro['relative_improvement'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit-sha256');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:
        print(json.dumps(dict(status='PREPARED_BUILDER_NO_RENDER',actual_complete=(CAMPAIGN/'complete.json').exists(),independent_audit_exists=AUDIT.exists(),expected_plan_sha256=PLAN_SHA,
                             policy_sha256=POLICY,source_policy_sha256=SOURCE_POLICY,output=rel(OUT),outputs_created=False,new_physics_steps=0,optimizer_steps=0,actor_inferences=0)));return
    if not args.audit_sha256 or len(args.audit_sha256)!=64:parser.error('Exact final independent PASS audit SHA required; no render while matrix/audit incomplete')
    build(args.audit_sha256)


if __name__=='__main__':main()
