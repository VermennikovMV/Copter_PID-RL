"""Shared deterministic mission/disturbance protocol for PyBullet and Gazebo.

NumPy and stdlib only. Unknown profiles fail closed. Forces are in NEWTONS, not
wind speed. Controllers receive the current target only, never these parameters.
"""
from functools import lru_cache
import numpy as np

PROTOCOL_VERSION="thesis-v2-draft-20261111"
CONTROL_HZ=50
CONTROL_DT=1/CONTROL_HZ
POSITION_BAND_M=.25
ROUTE_DWELL_S=.50
RECOVERY_DWELL_S=1.00
ROUTE_POINTS=np.array([[0,0,3],[2,0,3],[2,2,3.5],[0,2,3],[0,0,3]],dtype=float)
SCENARIOS={
    "nominal":{},
    "force_mild":{"force_profile":"force_mild"},
    "force_strong":{"force_profile":"force_strong"},
    "gust":{"force_profile":"gust"},
    "mass_20":{"mass_scale":1.2},
    "mass_40":{"mass_scale":1.4},
    "motor_20":{"motor_scale":.8,"motor_index":0},
    "motor_40":{"motor_scale":.6,"motor_index":0},
    "delay_40":{"delay_steps":2},
    "delay_100":{"delay_steps":5},
    "sensor_noise":{"noise_position":.03,"noise_velocity":.06,"noise_orientation":.01,"noise_rates":.03},
}


def seeded_rng(seed,stream):return np.random.default_rng(np.random.SeedSequence([int(seed),int(stream)]))


@lru_cache(maxsize=4096)
def _events(profile,seed):
    if profile not in SCENARIOS:raise ValueError(f"unknown disturbance profile {profile}")
    if profile not in ("force_mild","force_strong","gust"):return ()
    rng=seeded_rng(seed,110)
    angle=rng.uniform(-np.pi,np.pi)
    magnitude=(1.5 if profile=="force_mild" else 3.)*rng.uniform(.9,1.1)
    direction=np.array([np.cos(angle),np.sin(angle),0.])
    # Exact50Hz boundaries; jitter stays within requested±.25s (grid±.24s).
    jitter=rng.integers(-12,13,size=6)*CONTROL_DT
    if profile=="gust":
        specs=[(5+jitter[0],8+jitter[1],direction),
               (11+jitter[2],14+jitter[3],-direction),
               (17+jitter[4],20+jitter[5],np.array([-direction[1],direction[0],0.]))]
    else:specs=[(5+jitter[0],20+jitter[5],direction)]
    return tuple((round(float(on),2),round(float(off),2),tuple((magnitude*vector).tolist())) for on,off,vector in specs)


def force_events(profile,seed):
    if isinstance(profile,dict):
        return [dict(e) for e in profile.get("force_events",[])]
    return [{"on_s":on,"off_s":off,"force_enu_N":list(vector)} for on,off,vector in _events(profile,int(seed))]


def force_N(t,profile="nominal",seed=0):
    if not np.isfinite(t):raise ValueError("non-finite simulation time")
    sampled=np.floor((float(t)+1e-9)*CONTROL_HZ)/CONTROL_HZ
    events=force_events(profile,seed)
    result=np.zeros(3)
    for event in events:
        if sampled>=event["on_s"]-1e-9 and sampled<event["off_s"]-1e-9:
            result+=np.asarray(event["force_enu_N"],dtype=float)
    return result


def scenario_spec(name,seed):
    if name not in SCENARIOS:raise ValueError(f"unknown scenario{name}")
    spec={"name":name,"seed":int(seed),"protocol":PROTOCOL_VERSION,**SCENARIOS[name]}
    spec["force_events"]=force_events(spec.get("force_profile","nominal"),seed)
    return spec


def last_force_off(profile,seed=0):
    events=force_events(profile,seed)
    return max((e["off_s"] for e in events),default=None)


class Mission:
    """Goal-triggered route. update() takes the shared judge's observed position.

    Transition is confirmed after a full0.5s span inside the band. The current
    target is the ONLY mission feature supplied to the policy. Completion keeps
    the final target until the common horizon, so a quick route cannot shorten
    its accuracy measurement. No reset occurs on force/mass/motor events.
    """
    def __init__(self,kind="hover"):
        if kind not in ("hover","route"):raise ValueError(kind)
        self.kind=kind
        self.horizon_s=30. if kind=="hover" else 40.
        self.reset()

    def reset(self):
        self.index=0
        self.inside_since=None
        self.last_time=None
        self.completion_time=None
        self.events=[]

    @property
    def target(self):return ROUTE_POINTS[self.index].copy()

    def update(self,t,position):
        t=float(t);position=np.asarray(position,dtype=float)
        if not np.isfinite(t) or not np.all(np.isfinite(position)):raise ValueError("invalid mission observation")
        if self.last_time is not None and t<=self.last_time:raise ValueError("mission time must increase")
        # A missing sample breaks evidence of continuous dwell.
        if self.last_time is not None and t-self.last_time>1.5*CONTROL_DT:self.inside_since=None
        self.last_time=t
        if self.kind=="hover" or self.completion_time is not None:return None
        if np.linalg.norm(position-self.target)>POSITION_BAND_M:
            self.inside_since=None
            return None
        if self.inside_since is None:self.inside_since=t
        if t-self.inside_since+1e-9<ROUTE_DWELL_S:return None
        event={"type":"goal_reached","goal_index":self.index,"time_s":t,
               "dwell_started_s":self.inside_since,"target":self.target.tolist()}
        self.events.append(event)
        self.inside_since=None
        if self.index==len(ROUTE_POINTS)-1:self.completion_time=t
        else:self.index+=1
        return event
