"""Render saved measured Gazebo telemetry only; no simulator, policy or NN."""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/github_replay_media_v1"
AUDIT=ROOT/"audit/github_replay_media_v1_root01.json"
COLORS=["#56c8ff","#ffa94d","#9be07a"]
BG="#101924";FG="#e8f0f5";GRID="#344354"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"text.color":FG,
    "axes.labelcolor":FG,"xtick.color":"#bacbd6","ytick.color":"#bacbd6",
    "axes.edgecolor":"#617586","axes.facecolor":BG,"figure.facecolor":BG,
    "savefig.facecolor":BG,"axes.titleweight":"bold"})


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def write(p,value):
    assert not p.exists(),str(p)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def jobs():
    old=ROOT/"artifacts/gazebo/main_hybrid_test_v1"
    return [dict(name="legacy_hover_3way",title="Зависание под внешней силой",
        subtitle="Архивные Gazebo: PID / Hybrid / Direct · train111 · env202601 · одинаковая сила ≈3 Н",
        horizon=30.,poster_time=6.2,route=False,
        caption="Компромисс: Direct — меньше RMSE положения, но просадка Z 3 см против 1 см у PID и Hybrid.",
        sources=[("PID",old/"pid_trainNone_hover_force_strong_env202601"),
            ("Hybrid (старый)",old/"hybrid_train111_hover_force_strong_env202601"),
            ("Direct (старый)",ROOT/"artifacts/gazebo_direct/main_direct_test_v2/direct_train111_hover_force_strong_env202601")],
        evidence=[ROOT/"audit/main_hybrid_gazebo_physical_identity.json",ROOT/"audit/main_direct_gazebo_physical_identity.json",
            ROOT/"reports/presentation_pack/complete.json"]),
        dict(name="f0_strong_route",title="Маршрут при внешней силе: F0 и PID",
        subtitle="Измеренные Gazebo-траектории · F0 zero-route-I (f0e873…) · env83301 · development, n=1",
        horizon=40.,poster_time=6.8,route=True,
        caption="Это один успешный primary-case; отдельный tailwind84001 у F0 провален. Общей победы при любом ветре нет.",
        sources=[("PID",ROOT/"artifacts/refinement/joint_v2/gazebo_geometry_pilot32/runs/pid_route_force_strong_env83301"),
            ("F0 (zero-route-I)",ROOT/"artifacts/refinement/joint_v2/gazebo_memory_development9_v20_root01/runs/memory20_route_force_strong_env83301")],
        evidence=[ROOT/"audit/joint_memory_development9_v20_root01.json",ROOT/"reports/joint_memory20_gazebo_v7/applicability.csv",
            ROOT/"artifacts/refinement/joint_v2/gazebo_memory_development9_v20_root01/plan.json"])]


def load_source(label,folder,dependencies):
    summary_path=folder/"summary.json";csv_path=folder/"telemetry.csv"
    summary=read(summary_path)
    assert summary["status"]=="PASS" and summary["survived"] and summary["failure"] is None
    with csv_path.open(encoding="utf-8-sig",newline="") as handle:rows=list(csv.DictReader(handle))
    keys=("t","x_e","y_n","z_up","target_e","target_n","target_up","error_m")
    data={key:np.asarray([float(row[key]) for row in rows]) for key in keys}
    assert all(np.isfinite(value).all() for value in data.values()) and np.all(np.diff(data["t"])>0.)
    physical=[event for event in summary["physical_force_events"] if np.linalg.norm(event["force_enu_N"])>.01]
    assert len(physical)==1
    on=float(physical[0]["measurement_t"])
    off=min(float(event["measurement_t"]) for event in summary["physical_force_events"]
        if float(event["measurement_t"])>on and np.linalg.norm(event["force_enu_N"])<.01)
    force=np.asarray(physical[0]["force_enu_N"],float)
    for p in (summary_path,csv_path):dependencies[rel(p)]=sha(p)
    return dict(label=label,folder=rel(folder),summary=summary,data=data,on=on,off=off,force=force,
        source_summary=rel(summary_path),source_csv=rel(csv_path),policy_sha256=summary.get("policy_sha256"),
        input_rows=len(rows))


def current(source,t,key):return float(np.interp(t,source["data"]["t"],source["data"][key]))
def goal(source,t):
    data=source["data"];i=max(0,np.searchsorted(data["t"],t,side="right")-1)
    return [data[key][i] for key in ("target_e","target_n","target_up")]


def render(job,dependencies):
    sources=[load_source(label,folder,dependencies) for label,folder in job["sources"]]
    reference=sources[0]["summary"]["scenario_spec"]
    assert all(source["summary"]["scenario_spec"]==reference for source in sources)
    assert all(source["summary"]["duration_s"]==job["horizon"] for source in sources)
    for p in job["evidence"]:dependencies[rel(p)]=sha(p)
    geom={}
    if job["route"]:
        with (ROOT/"reports/joint_memory20_gazebo_v7/applicability.csv").open(encoding="utf-8-sig",newline="") as handle:
            row=next(x for x in csv.DictReader(handle) if x["mission"]=="route" and x["environment_seed"]=="83301" and x["condition"].startswith("Пост. сила"))
        for source,prefix in zip(sources,("PID","joint")):
            geom[source["label"]]=dict(finite_segment_RMSE_m=float(row[prefix+"_finite_segment_RMSE_m"]),
                route_time_s=float(row[prefix+"_route_time_s"]))
    fig=plt.figure(figsize=(10.8,6.6),dpi=100)
    fig.text(.055,.955,job["title"],fontsize=20,weight="bold")
    fig.text(.055,.918,job["subtitle"],fontsize=9,color="#b7c8d5")
    fig.text(.055,.882,"РЕПЛЕЙ ИЗ CSV · не запись интерфейса Gazebo · без новых полётов",fontsize=9,color="#d7bc83")
    xy=fig.add_axes([.075,.315,.40,.455]);z=fig.add_axes([.565,.575,.385,.195]);error=fig.add_axes([.565,.30,.385,.20])
    strip=fig.add_axes([.075,.165,.875,.045])
    for ax in (xy,z,error):ax.grid(color=GRID,alpha=.55,linewidth=.65);ax.set_axisbelow(True)
    xy.set_title("Общий масштаб XY",loc="left",fontsize=11)
    xy.set_xlabel("Восток E, м");xy.set_ylabel("Север N, м");xy.set_aspect("equal",adjustable="box")
    allxy=np.concatenate([np.c_[s["data"]["x_e"],s["data"]["y_n"]] for s in sources])
    if job["route"]:
        points=np.array([[0,0],[2,0],[2,2],[0,2],[0,0]],float)
        xy.plot(points[:,0],points[:,1],"--",color="#bbc8d0",linewidth=1.3,alpha=.75)
        for i,(x,y) in enumerate(points[:-1]):xy.text(x+.055,y+.065,"0/4" if i==0 else str(i),color="#c4d2dc",fontsize=9)
        xy.set_xlim(-.45,2.45);xy.set_ylim(-.45,2.45)
    else:
        limit=max(.12,float(np.max(np.abs(allxy)))*1.25);xy.set_xlim(-limit,limit);xy.set_ylim(-limit,limit)
        xy.scatter([0],[0],marker="*",s=120,color=FG,zorder=7)
    z.set_title("Высота: сплошная — факт, пунктир — цель",loc="left",fontsize=10)
    z.set_ylabel("Z, м");error.set_ylabel("Ошибка до цели, м");error.set_xlabel("Измеренное время, с")
    for ax in (z,error):ax.set_xlim(0,job["horizon"])
    allz=np.concatenate([s["data"]["z_up"] for s in sources]+[s["data"]["target_up"] for s in sources])
    pad=max(.012,float(np.ptp(allz))*.12);z.set_ylim(float(allz.min())-pad,float(allz.max())+pad)
    error.set_ylim(0,max(.06,max(float(s["data"]["error_m"].max()) for s in sources)*1.1))
    trails=[];dots=[];goals=[];zcursors=[];ecursors=[]
    for i,source in enumerate(sources):
        color=COLORS[i];data=source["data"]
        z.plot(data["t"],data["z_up"],color=color,linewidth=1.,alpha=.8)
        z.step(data["t"],data["target_up"],where="post",color=color,linewidth=.85,linestyle="--",alpha=.55)
        error.plot(data["t"],data["error_m"],color=color,linewidth=1.,alpha=.8)
        trail,=xy.plot([],[],color=color,linewidth=2.4,label=source["label"]);trails.append(trail)
        dot,=xy.plot([],[],"o",color=color,markersize=7,markeredgecolor=FG,markeredgewidth=.7,zorder=8);dots.append(dot)
        mark,=xy.plot([],[],"*",color=color,markersize=11,alpha=.8);goals.append(mark)
        zz,=z.plot([],[],"o",color=color,markersize=5);zcursors.append(zz)
        ee,=error.plot([],[],"o",color=color,markersize=5);ecursors.append(ee)
        if job["route"]:
            metrics=f'{source["label"]}: путь {geom[source["label"]]["finite_segment_RMSE_m"]:.3f} м · {geom[source["label"]]["route_time_s"]:.2f} с'
        else:
            metrics=f'{source["label"]}: RMSE {source["summary"]["rmse_position_m"]:.3f} м · Z↓ {source["summary"]["altitude_drop_m"]*100:.0f} см'
        fig.text(.075+i*(.875/len(sources)),.827,metrics,fontsize=8.2,color=color)
    xy.legend(loc="upper right",fontsize=8,facecolor=BG,edgecolor=GRID,labelcolor=FG)
    cursor_z=z.axvline(0,color=FG,alpha=.6,linewidth=.8);cursor_e=error.axvline(0,color=FG,alpha=.6,linewidth=.8)
    strip.set_xlim(0,job["horizon"]);strip.set_ylim(-.5,len(sources)-.5)
    strip.set_yticks(range(len(sources)),[s["label"].split(" (")[0] for s in sources],fontsize=7.3);strip.set_xticks([])
    for i,source in enumerate(sources):strip.broken_barh([(source["on"],source["off"]-source["on"])],(i-.32,.64),facecolors=COLORS[i],alpha=.6)
    strip.set_title("Физически подтверждённая сила ON — отдельная временная полоса каждого полёта",loc="left",fontsize=8,pad=3)
    strip_cursor=strip.axvline(0,color=FG,linewidth=1.)
    clock=fig.text(.075,.105,"",fontsize=11,weight="bold")
    fig.text(.075,.068,job["caption"],fontsize=8.1,color="#ffd19b")
    fig.text(.075,.037,"Только измеренные логи; интерполяция нужна для анимации. Метрики — по исходному полному горизонту.",fontsize=7.7,color="#9fb3c2")
    wind=xy.annotate("",xy=(0,0),xytext=(0,0),arrowprops=dict(arrowstyle="->",color="#f7d26a",lw=2.2))
    windtext=xy.text(.03,.035,"",transform=xy.transAxes,fontsize=8,color="#f7d26a")
    def draw(t):
        for i,source in enumerate(sources):
            data=source["data"];mask=(data["t"]<=t)&(data["t"]>=t-(5. if job["route"] else 4.))
            trails[i].set_data(data["x_e"][mask],data["y_n"][mask])
            dots[i].set_data([current(source,t,"x_e")],[current(source,t,"y_n")]);g=goal(source,t)
            goals[i].set_data([g[0]],[g[1]])
            zcursors[i].set_data([t],[current(source,t,"z_up")]);ecursors[i].set_data([t],[current(source,t,"error_m")])
        for artist in (cursor_z,cursor_e,strip_cursor):artist.set_xdata([t,t])
        force=sources[0]["force"];active=any(s["on"]<=t<s["off"] for s in sources)
        center=np.array([.13,.12]) if job["route"] else np.array([0.,0.]);scale=.45 if job["route"] else .055
        end=center+force[:2]/np.linalg.norm(force[:2])*scale
        wind.xy=end;wind.set_position(center);wind.set_visible(active)
        windtext.set_text(f'Сила {np.linalg.norm(force):.2f} Н  '+("ON" if active else "OFF"))
        clock.set_text(f't = {t:4.1f} / {job["horizon"]:.0f} с     ·     воспроизведение ×{job["horizon"]/15.:.2f}     ·     общий момент времени')
        fig.canvas.draw();return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
    poster=draw(job["poster_time"]);poster_path=OUT/(job["name"]+".png");poster.save(poster_path)
    palette=poster.quantize(colors=128,method=Image.Quantize.MEDIANCUT)
    frames=[draw(float(t)).quantize(palette=palette,dither=Image.Dither.NONE) for t in np.linspace(0,job["horizon"],151)]
    gif=OUT/(job["name"]+".gif")
    frames[0].save(gif,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True,disposal=1)
    assert gif.stat().st_size<=8*1024*1024
    plt.close(fig)
    with Image.open(gif) as check:
        assert check.n_frames==151 and check.size==(1080,660)
        for index in (0,50,100,150):check.seek(index);check.convert("RGB").load()
    info=dict(name=job["name"],gif=dict(path=rel(gif),sha256=sha(gif),bytes=gif.stat().st_size),
        poster=dict(path=rel(poster_path),sha256=sha(poster_path)),source_horizon_s=job["horizon"],
        replay_frames=151,replay_fps=10,duration_ms=15100,playback_rate=job["horizon"]/15.,
        frame_time_range_s=[0,job["horizon"]],poster_measurement_time_s=job["poster_time"],
        scenario_spec=reference,caption=job["caption"],source_metrics=geom,
        cases=[{k:s[k] for k in ("label","folder","source_summary","source_csv","policy_sha256","input_rows","on","off")} for s in sources])
    print(json.dumps(dict(asset=info["gif"],frames=151)),flush=True)
    return info


def main():
    assert not AUDIT.exists() and not any(p.is_file() for p in OUT.glob("*"));OUT.mkdir(parents=True,exist_ok=True)
    inputs={rel(Path(__file__)):sha(Path(__file__))};assets=[render(job,inputs) for job in jobs()]
    manifest=dict(schema="github-measured-gazebo-replay-media-v1.0",status="COMPLETE_MEASURED_CSV_REPLAYS_NOT_GUI_RECORDINGS",
        created_utc=datetime.now(timezone.utc).isoformat(),renderer=rel(Path(__file__)),input_sha256=inputs,assets=assets,
        metric_definitions=dict(hover_RMSE="Saved time-weighted full30s 3D current-target goal-distance RMSE; not XY-only",
            altitude_drop="Saved maximum downward current-target altitude error; hovertarget3m",
            route_finite_RMSE="Saved independent3D distance to finite active PRE-target segment over transit only",
            route_time="All ordered0.5s dwell confirmations frommeasurement0; full40s replay continues after completion"),
        interpolation="Linear position/error interpolation between telemetry rows for pictures only; current targets step-held; each physicalforceinterval fromitsownreceipt",
        source_time_alignment="Each flight uses itsown saved measurement-time t; no boot/wall-clock alignment or spatial offsets",
        frames_are_not_scientific_samples=True,new_evaluation_cases=0,new_environment_steps=0,new_NN_forwards=0,new_SGD_steps=0,
        limitations=["One viewed development case per method/model, not final independent confirmation.",
            "First clip is archived legacy train111/env202601, not current F0 or newer PPO.",
            "Second is actual F0f0e/env83301, not v28FINAL or v29composite; F0tail84001failure remains recorded.",
            "GIF interpolation and compressed colors do not replace raw traces or scientific receipts."])
    for p,digest in inputs.items():assert sha(ROOT/p)==digest,p
    write(OUT/"manifest.json",manifest)
    text="# Measured Gazebo trajectory replays\n\nThese GIFs animate actual saved telemetry CSVs. They are **not Gazebo GUI screen recordings**, generated simulator flights, or new evaluations. The first clip is the archived legacy train111/env202601 comparison; the second is actual F0 zero-route-I / PID env83301. They are different model generations and are not pooled.\n\n"
    for asset in assets:
        text+=f'## {asset["name"]}\n\n![Measured trajectory replay]({asset["name"]}.gif)\n\n'+asset['caption']+'\n\n'
    text+="Each clip is151 frames at10fps, with an explicit measured-time cursor and playback rate. XY/Z axes and measurement time are shared across methods. Force strips use each flight's independently verified physical ON/OFF timestamps. Position interpolation affects pictures only; displayed metrics come from original full-horizon summaries/audits. Route targets change according to each recorded controller's actual mission progress. No force is converted to wind velocity.\n\nThe complete source paths, source policy hashes, dependency SHA256, metric definitions, frame counts and limitations are in [manifest.json](manifest.json). Reproduce from the repository root with `python -B scripts/render_project_replay_gifs_v1.py` into a fresh output namespace (existing files are preserved).\n"
    (OUT/"README.md").write_text(text,encoding="utf-8")
    receipt=dict(schema=manifest["schema"],status="PASS_RENDERED_AND_GIF_FRAMES_READBACK",utc=manifest["created_utc"],
        manifest=dict(path=rel(OUT/"manifest.json"),sha256=sha(OUT/"manifest.json")),input_sha256=inputs,
        outputs={rel(p):sha(p) for p in OUT.iterdir() if p.is_file()},actual_rendered_frames=302,
        GIF_frame_readback_indices=[0,50,100,150],new_scientific_samples=0,new_NN_forwards=0,new_environment_steps=0,new_optimizer_steps=0,
        visual_review_pending=True)
    write(AUDIT,receipt);print(json.dumps(dict(status=receipt["status"],receipt=rel(AUDIT),sha256=sha(AUDIT))),flush=True)


if __name__=="__main__":main()
