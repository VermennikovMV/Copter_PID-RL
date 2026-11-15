"""Project the completed independent Gazebo80 audit, without new simulation."""
from __future__ import annotations
import copy
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/joint_metric_cost16_gazebo_v6"
AUDIT = ROOT / "audit/joint_route80_development9_v5_root01.json"
AUDIT_SHA = "5b917f1b2ce55025878f0e7e4b8c62a95852c3a1652d6def70ad9216ad2d5b74"
POLICY_SHA = "68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5"
OLD = ROOT / "reports/joint_path77_fullcontrol_v5"
NATIVE_AUDIT = ROOT / "audit/joint_metric_cost_pilot_v16_root01.json"
NATIVE_SHA = "f87ae82684225284b9e9c4e1272b47139307f569b740f1f3ab0516156f144294"
SCENARIOS = ("nominal", "force_mild", "force_strong", "gust")
CHECKS = 0


def require(ok, why):
    global CHECKS
    CHECKS += 1
    if not ok: raise ValueError(why)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def resolve(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / path
def rel(path): return path.resolve().relative_to(ROOT).as_posix()
def key(case): return tuple(case["key"])
def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
def write_csv(path, rows):
    require(bool(rows), "Nonempty CSV required")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
def close(a, b, why):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        require(a == b, why)
    else: require(math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-11), why)


def classify(cases, recorded):
    joints = sorted((c for c in cases if c["role"] == "joint80"), key=lambda c:(key(c)[0], SCENARIOS.index(key(c)[1]), key(c)[2] == "route"))
    pids = {key(c):c for c in cases if c["role"] == "PID"}
    refs = {key(c):c for c in cases if c["role"] == "legacy_retention"}
    gates = []
    def add(name, c, value, limit, passed): gates.append(dict(check=name, case=list(key(c)), value=value, limit=limit, pass_=bool(passed)))
    def upper(name, c, value, limit): add(name,c,value,limit,value is not None and limit is not None and math.isfinite(value) and math.isfinite(limit) and value<=limit)
    for c in joints:
        m, p = c["metrics"], pids[key(c)]; pm = p["metrics"]
        add("paired_PID_reference_full_flight", c, pm["survived"], True, p["status"]=="PASS" and pm["survived"] and pm["heading_guard_pass"] and pm["duration_s"]>=p["horizon_s"]-.040001)
        add("all_valid_joint_flights_survived", c, m["survived"], True, c["status"]=="PASS" and m["survived"])
        add("full_flight_heading_guard", c, m["heading_guard_pass"], True, m["heading_guard_pass"] and m["duration_s"]>=c["horizon_s"]-.040001)
        upper("heading_RMS_deg",c,m["heading_rmse_deg"],5.)
        upper("heading_peak_deg",c,m["heading_peak_deg"],15.)
        seed, scenario, mission = key(c)
        if mission == "hover":
            if scenario == "nominal": upper("nominal_hover_precision",c,m["rmse_position_m"],min(.05,max(1.2*pm["rmse_position_m"],pm["rmse_position_m"]+.02)))
            elif scenario in ("force_strong","gust"):
                upper("wind_hover_gain_vs_PID",c,m["penalized_rmse_position_m"],.85*pm["penalized_rmse_position_m"])
                upper("wind_hover_retention",c,m["penalized_rmse_position_m"],1.1*refs[key(c)]["metrics"]["penalized_rmse_position_m"])
        else:
            g, pg = c["geometry"], p["geometry"]
            pg_ok = pm["route_completed"] is True and pg["geometry_eligible"] is True and pg["complete_corner_windows"]==3
            add("paired_PID_complete_route_geometry",c,pg_ok,True,pg_ok)
            add("route_completed",c,m["route_completed"],True,m["route_completed"] is True)
            upper("restricted_route_time_s",c,m["route_completion_time_s"] if m["route_completed"] else 40.,1.05*(pm["route_completion_time_s"] if pm["route_completed"] else 40.))
            upper("penalized_goal_RMSE_m",c,m["penalized_rmse_position_m"],1.05*pm["penalized_rmse_position_m"])
            add("complete_route_geometry",c,g["geometry_eligible"],True,g["geometry_eligible"] is True and g["complete_corner_windows"]==3)
            upper("finite_segment_RMSE_m",c,g["segment_rmse_m"],max(1.1*pg["segment_rmse_m"],pg["segment_rmse_m"]+.01))
            upper("mean_corner_peak_m",c,g["mean_corner_post_peak_m"],max(1.1*pg["mean_corner_post_peak_m"],pg["mean_corner_post_peak_m"]+.02))
            upper("absolute_active_segment_peak_m",c,g["active_segment_peak_m"],.5)
        c["quality_checks"] = [r for r in gates if tuple(r["case"])==key(c)]
        c["quality_pass"] = all(r["pass_"] for r in c["quality_checks"])
    require(len(gates)==len(recorded)==90,"Exactly90 fixed gates")
    sorting = lambda r:(tuple(r["case"]),r["check"])
    for actual, expected in zip(sorted(gates,key=sorting),sorted(recorded,key=sorting)):
        require(actual["check"]==expected["check"] and actual["case"]==expected["case"],"Same gate identity")
        for field in ("value","limit","pass_"): close(actual[field],expected[field],"Exact fixed gate/"+field)
    return joints,pids,gates


def magnitude(c):
    intervals=c["force"]["physical_intervals"]
    return math.sqrt(sum(float(x)**2 for x in intervals[0]["force_enu_N"])) if intervals else None
def num(value,digits=3): return "—" if value is None else f"{value:.{digits}f}".replace(".",",")
def label(c):
    seed, scenario, unused = key(c)
    prefix="Попутная сила*" if seed==84001 else dict(nominal="Без силы",force_mild="Малая сила",force_strong="Пост. сила",gust="Порывы")[scenario]
    value=magnitude(c)
    return prefix+(" ≈"+num(value,2)+" Н" if value is not None else "")


def case_metrics(c):
    m,g=c["metrics"],c["geometry"] or {}
    return dict(goal_RMSE_m=m["rmse_position_m"],penalized_goal_RMSE_m=m["penalized_rmse_position_m"],
        target_relative_max_downward_error_m=m["altitude_drop_m"],auxiliary_preforce_relative_height_drop_m=m.get("altitude_drop_from_preforce_m"),
        recovery_time_s=m.get("recovery_time_s"),recovery_censored=m.get("recovery_censored"),survived=m["survived"],route_completed=m["route_completed"],
        route_time_s=m["route_completion_time_s"] if m["route_completed"] else 40. if key(c)[2]=="route" else None,
        heading_RMS_deg=m["heading_rmse_deg"],heading_peak_deg=m["heading_peak_deg"],duration_s=m["duration_s"],
        finite_segment_RMSE_m=g.get("segment_rmse_m"),mean_corner_peak_m=g.get("mean_corner_post_peak_m"),absolute_peak_m=g.get("active_segment_peak_m"))


def render(rows,macro):
    values=[]
    for row in rows:
        failed=row["failed_checks"].split("; ") if row["failed_checks"] else []
        status="Пройдены" if not failed else "Не выполнены:\nточность пути" if len(failed)==1 else "Не выполнены:\nпуть / повороты"
        data=[("Удержание" if row["mission"]=="hover" else "Маршрут")+"\n"+row["condition"],str(row["environment_seed"])]
        for field,digits in (("goal_RMSE_m",3),("route_time_s",2),("finite_segment_RMSE_m",3),("mean_corner_peak_m",3),("absolute_peak_m",3)):
            data.append(num(row["joint_"+field],digits)+" / "+num(row["PID_"+field],digits))
        values.append(data+[status])
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
    fig=plt.figure(figsize=(20,11.25),facecolor="white")
    fig.text(.025,.95,"RL80 и PID: измеренная применимость в Gazebo",fontsize=24,weight="bold",color="#143746")
    fig.text(.025,.905,"Последняя диагностическая гибридная модель · в парах RL / PID · политика 68dafd98…",fontsize=13,color="#3b5260")
    fig.text(.025,.855,"Основные условия: допуски пройдены в 7 из 8. Попутный случай: точность не прошла.\n"
        +f"Общий ветровой критерий: {num(100*macro['relative_improvement'],2)}% (порог 15%); это не 7 побед над PID.",fontsize=13,color="#3b5260",linespacing=1.45)
    ax=fig.add_axes([.025,.272,.95,.515]);ax.axis("off")
    headers=["Миссия / воздействие","Среда","Ошибка цели\nRMSE, м","Маршрут\nс","Отклонение от пути\nRMSE, м","Повороты\nсредний пик, м","Максимум\nотклонения, м","Заданные\nдопуски"]
    table=ax.table(cellText=values,colLabels=headers,colWidths=[.21,.055,.11,.095,.13,.155,.115,.13],cellLoc="center",bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(11.2)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor("#d4dee4");cell.set_linewidth(.7)
        if i==0:cell.set_facecolor("#153e50");cell.set_text_props(color="white",weight="bold",fontsize=11)
        else:
            passed=rows[i-1]["quality_status"]=="PASS"
            cell.set_facecolor("#edf7f0" if passed else "#fff0ee")
            if j==0:cell.set_text_props(ha="left")
            if j==7:cell.set_text_props(weight="bold",color="#286341" if passed else "#9d302b",fontsize=10.4)
    notes=("Модель не принята: все 6 кандидатов native-пилота не прошли допуск пути. Gazebo: 87/90 проверок; провалы в штиле и попутном случае сохранены.\n"
        "9 новых полётов + 11 общих контрольных наблюдений = 20 наблюдений, не 20 новых повторов. Одна политика; по одному случаю на метод и условие.\n"
        "Просмотренная среда 83301 и отдельный попутный случай 84001*. Финальный независимый тест не проведён; масса, задержки, мотор и шум не проверены этой моделью.\n"
        "Путь: конечный активный 3D-отрезок, без финального зависания. Повороты: три окна по 1 с. Сила в Н, не м/с; её фаза на поворотах может различаться.\n"
        "Небольшое улучшение относительно прежней RL77 не доказывает отдельный вклад награды: контроль A в Gazebo не летал, изменились также общие условия обучения и ширина входа.")
    fig.text(.025,.224,notes,fontsize=10.0,color="#3b5260",va="top",linespacing=1.55)
    for ext in ("png","pdf"):fig.savefig(OUT/("presentation."+ext),dpi=128,facecolor="white")
    plt.close(fig)


def main():
    require(not OUT.exists(),"Fresh output only; preserve previous maps")
    require(sha(AUDIT)==AUDIT_SHA,"Independent completed80 audit identity")
    require(sha(NATIVE_AUDIT)==NATIVE_SHA,"Native pilot audit identity")
    audit=read(AUDIT);native=read(NATIVE_AUDIT)
    require(audit["status"]==native["status"]=="PASS","Integrity audits passed before render")
    require(audit["policy_sha256"]==POLICY_SHA and audit["new_actual_flights"]==9 and audit["total_unique_observation_coverage"]==20,"Exact model and coverage")
    require(audit["scientific_quality_status"]=="FAIL" and audit["policy_promotion"] is False,"Diagnostic-only quality failure preserved")
    inputs=dict(audit["input_sha256"])
    for path in [AUDIT,NATIVE_AUDIT,Path(__file__),*sorted(OLD.iterdir())]:
        if path.is_file():inputs[rel(path)]=sha(path)
    for name,digest in inputs.items():require(sha(resolve(name))==digest,"Immutable source before:"+name)
    cases=copy.deepcopy(audit["cases"])
    require(len(cases)==20 and sum(c["new_actual_flight"] for c in cases)==9,"20 cases/9 new")
    joints,pids,gates=classify(cases,audit["gate_rows"])
    require(len(joints)==9 and len(pids)==9,"9 currentRL+9 PID references")
    require(sum(c["quality_pass"] for c in joints if key(c)[0]==83301)==7,"Seven primary passing cells")
    require(sum(not row["pass_"] for row in gates)==3,"Exactly three failed gates")
    six=[dict(scenario=s,mission=m,joint_penalized_RMSE_m=next(c for c in joints if key(c)==(83301,s,m))["metrics"]["penalized_rmse_position_m"],PID_penalized_RMSE_m=pids[(83301,s,m)]["metrics"]["penalized_rmse_position_m"]) for s in ("force_mild","force_strong","gust") for m in ("hover","route")]
    jr=statistics.mean(c["joint_penalized_RMSE_m"] for c in six);pr=statistics.mean(c["PID_penalized_RMSE_m"] for c in six)
    macro=dict(status="PASS" if 1-jr/pr>=.15 else"FAIL",coverage="ALL6_DEVELOPMENT_CELLS",relative_improvement=1-jr/pr,
        joint_mean_penalized_RMSE_m=jr,PID_mean_penalized_RMSE_m=pr,minimum_improvement=.15,cells=six,formula="1 - mean6_joint_penalized_RMSE / mean6_PID_penalized_RMSE")
    for field in ("status","relative_improvement","joint_mean_penalized_RMSE_m","PID_mean_penalized_RMSE_m"):close(macro[field],audit["original_thesis_wind_macro"][field],"Fresh six-cell macro:"+field)
    with (OLD/"applicability.csv").open(encoding="utf-8-sig",newline="") as handle:oldrows=list(csv.DictReader(handle))
    oldby={(int(r["environment_seed"]),r["mission"],r["condition"].split(" ≈")[0]):r for r in oldrows}
    oldcases=read(OLD/"case_recomputation.json")
    oldjoint={(c["environment_seed"],c["scenario"],c["mission"]):c for c in oldcases if c["role"]=="joint77"}
    rows=[];comparisons=[]
    for c in joints:
        seed,scenario,mission=key(c);p=pids[key(c)];jm,pm=case_metrics(c),case_metrics(p)
        row=dict(condition=label(c),mission=mission,environment_seed=seed,training_lineage=111,joint_new_flights=1,PID_new_flights=0,PID_shared_flights=1,
            policy_sha256=POLICY_SHA,encoder=80,basis_kind="velocity_residual",backend="Gazebo",quality_status="PASS" if c["quality_pass"] else"FAIL",
            failed_checks="; ".join(x["check"] for x in c["quality_checks"] if not x["pass_"]))
        row.update({"joint_"+k:v for k,v in jm.items()});row.update({"PID_"+k:v for k,v in pm.items()})
        row.update(measured_domain="Viewed development n=1 per method/cell; not independent final confirmation",model_role="LATEST80_DIAGNOSTIC_NATIVE_ALL6_FAIL_NOT_PROMOTED")
        rows.append(row)
        old=oldjoint[key(c)];om=old["metrics"];og=old["geometry"] or {}
        comp=dict(environment_seed=seed,scenario=scenario,mission=mission,current80_policy_sha256=POLICY_SHA,previous77_policy_sha256="e0aa635603d5ec299cd526347891ef1b090e2f8ebd3879d5aa34f207a41fa781")
        pairs={"goal_RMSE_m":(jm["goal_RMSE_m"],om["rmse_position_m"]),"route_time_s":(jm["route_time_s"],om["route_completion_time_s"]),
            "finite_segment_RMSE_m":(jm["finite_segment_RMSE_m"],og.get("segment_rmse_m")),"mean_corner_peak_m":(jm["mean_corner_peak_m"],og.get("mean_corner_post_peak_m")),"absolute_peak_m":(jm["absolute_peak_m"],og.get("active_segment_peak_m"))}
        for field,(new,oldvalue) in pairs.items():comp.update({"current80_"+field:new,"previous77_"+field:oldvalue,"change_percent_"+field:100*(new/oldvalue-1) if new is not None and oldvalue else None})
        comp.update(comparison_scope="Different exact policies/encoder width/common training conventions; controlA not flown; descriptive development comparison, not causal reward-only or independent replicates")
        comparisons.append(comp)
    original=[];exposure=[]
    for c in cases:
        seed,scenario,mission=key(c)
        original.append(dict(run=rel(resolve(c["run"])),role=c["role"],new_actual_flight=c["new_actual_flight"],environment_seed=seed,scenario=scenario,mission=mission,
            policy_sha256=POLICY_SHA if c["role"]=="joint80" else None,**case_metrics(c),actual_final_force_OFF_s=c["force"]["last_physical_off_s"],
            force_norm_N=magnitude(c),force_intervals=json.dumps(c["force"]["physical_intervals"]),quality_status=("PASS" if c["quality_pass"] else"FAIL") if c["role"]=="joint80" else"REFERENCE",
            actor_physical_proof="Reused freshly completed whole-audit receipt; no builder wire rerun"))
        if mission=="route":
            g=c["geometry"];ints=c["force"]["physical_intervals"]
            for corner in g["corners"]:
                stamp=corner["transition_s"];vector=[0.,0.,0.]
                for interval in ints:
                    if interval["on_s"]<=stamp<interval["off_s"]:vector=interval["force_enu_N"]
                exposure.append(dict(run=rel(resolve(c["run"])),role=c["role"],environment_seed=seed,scenario=scenario,corner=corner["corner_index"],corner_time_s=stamp,
                    force_at_corner_enu_N=json.dumps(vector),force_norm_N=math.sqrt(sum(float(x)**2 for x in vector)),
                    transit_exposed_seconds=sum(max(0.,min(g["transit_end_s"],r["off_s"])-max(g["transit_start_s"],r["on_s"])) for r in ints),
                    complete_corner_window=corner["full_post_window"],scope="Absolute force clock; no matched event-aligned corner-force causal attribution"))
        c["run"]=rel(resolve(c["run"]))
    require(len(rows)==9 and len(original)==20 and len(exposure)==30,"All required table denominators")
    OUT.mkdir(parents=True)
    write_csv(OUT/"applicability.csv",rows);write_csv(OUT/"all_original_flights.csv",original)
    write_csv(OUT/"fixed_quality_checks.csv",gates);write_csv(OUT/"comparison_to_previous77.csv",comparisons);write_csv(OUT/"route_force_exposure.csv",exposure)
    write_json(OUT/"case_recomputation.json",cases);write_json(OUT/"original_thesis_wind_macro.json",macro)
    write_json(OUT/"operational_gate.json",dict(scientific_quality_status="FAIL",primary_passed_cells=7,primary_cells=8,tail_passed=False,passed_checks=87,total_checks=90,checks=gates))
    write_json(OUT/"input_freeze.json",dict(independent_whole_audit_sha256=AUDIT_SHA,input_sha256=inputs))
    render(rows,macro)
    (OUT/"README.txt").write_text(
        "Последняя измеренная Gazebo80 модель:68dafd98…; metric_cost, velocity_residual, гибрид с внутренним PID.\n"
        "Это диагностическая версия: native-пилот не прошли все6 кандидатов; модель не принята. Проверенная77-модель и её файлы сохранены отдельно.\n"
        "9 новых RL-полётов +9 общих PID +2 общих retention =20 наблюдений; новых PID0. Не20 новых или независимых повторов.\n"
        "Основная среда83301:7/8 заданных допусков; номинальный маршрут FAIL. Отдельный попутный84001: finite/corner FAIL. 87/90 проверок.\n"
        f"Шестиячеечная ветровая macro: {macro['relative_improvement']:.8%}, формула1-mean6(RL penalized goal RMSE)/mean6(PID); n=1, не финальная независимая проверка.\n"
        "Числа получены из полного независимого20-case аудита; builder повторно вычисляет90 допусков и macro, не запускает новые полёты/обучение/wire-аудит.\n"
        "Все пять исходных метрик, включая высоту/восстановление/устойчивость, сохранены в all_original_flights.csv; full horizon30/40s.\n"
        "Ошибка цели не cross-track. Геометрия: конечный активный3D-отрезок PRE-target, транзит без финального зависания; три закрытых1s окна поворотов.\n"
        "Недобор высоты относительно цели включает заданный шаг3→3.5м; малый recovery при OFF не доказывает мгновенную аэродинамическую адаптацию.\n"
        "Сила вН, не м/с; различающиеся фазы на поворотах не считаются matched wind-at-corner. Удержание исходного yaw, не90deg yaw-команда.\n"
        "comparison_to_previous77.csv — описательное сравнение двух точных политик, не независимые повторения и не доказательство вклада одной награды.\n"
        "В Gazebo не летал парный контрольA; изменены также общие gamma/конечный горизонт обучения и ширина77→80.\n"
        "Масса, смещённый/динамический груз, отказы мотора, дополнительные задержки/шум и реальный аппарат этой моделью не подтверждены.\n",
        encoding="utf-8")
    for name,digest in inputs.items():require(sha(resolve(name))==digest,"Immutable source after:"+name)
    receipt=dict(schema="joint-metric-cost16-gazebo-map-v6.0",status="PASS",utc=dt.datetime.now(dt.timezone.utc).isoformat(),checks=CHECKS,
        source_builder_sha256=sha(Path(__file__)),independent_whole_audit_sha256=AUDIT_SHA,native_audit_sha256=NATIVE_SHA,policy_sha256=POLICY_SHA,
        new_actual_flights=9,shared_PID_flights=9,shared_retention_flights=2,total_observation_coverage=20,table_rows=9,route_geometries=10,corner_windows=30,
        fixed_checks=90,passed_checks=87,failed_quality_checks=[r for r in gates if not r["pass_"]],primary_passed=7,primary_total=8,tail_passed=False,
        original_thesis_macro=macro,native_all6_failed=True,policy_promotion=False,final_applicability_confirmed=False,
        input_files_unchanged=len(inputs),old_v5_files_preserved=len([p for p in OLD.iterdir() if p.is_file()]),
        no_new_simulation_training_or_actor_inference=True,reused_whole_audit_instead_of_new_wire_replay=True,
        visual_inspection="PENDING_SEPARATE_IMAGE_REVIEW",output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    target=ROOT/"audit/joint_metric_cost16_gazebo_v6_table_root01.json"
    write_json(target,receipt)
    print(json.dumps(dict(status="PASS",receipt=rel(target),sha256=sha(target),checks=CHECKS,primary="7/8",macro=macro["relative_improvement"],old_v5_preserved=receipt["old_v5_files_preserved"])))


if __name__=="__main__":main()
