"""Common configurable observation preparation for flight controllers."""
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
import numpy as np

OBSERVATION_SCHEMA_ID = 'quadrotor-heading-observation-v2-clock'
COMMAND_SCHEMA_IDS = {
    'direct': 'motor-thrust-gazebo-order-v1',
    'residual': 'native-pid-plus-motor-pwm-v1',
}


def encoder_config_dict(config):
    """Serializable configuration metadata, independent of encoder class."""
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    try:
        return dict(vars(config))
    except TypeError as exc:
        raise ValueError('Encoder configuration needs dataclass, mapping or named attributes') from exc

@dataclass(frozen=True)
class EncoderConfig:
    include_task_context: bool=True
    history: bool=True
    integral_limit: float=6.
    acceleration_tau_s: float=.08
    position_scale: float=3.
    velocity_scale: float=3.
    rate_scale: float=3.
    acceleration_scale: float=5.
    history_times_s: tuple=(.1,.2)
    goal_clip_m: float=0.  # >0: the goal error vector seen by the policy is clipped to this length

    def __post_init__(self):
        positive = (self.integral_limit, self.acceleration_tau_s, self.position_scale,
                    self.velocity_scale, self.rate_scale, self.acceleration_scale)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Observation scales and time constant must be positive and finite")
        if not isinstance(self.history, bool):
            raise ValueError("history must be a boolean")
        if any(not np.isfinite(v) or v <= 0 for v in self.history_times_s):
            raise ValueError("History times must be positive and finite")
        if tuple(sorted(set(self.history_times_s))) != tuple(self.history_times_s):
            raise ValueError("History times must be unique and increasing")

    @property
    def size(self):return (30+22*len(self.history_times_s) if self.history else 30)+(2 if self.include_task_context else 0)


def heading_rotation(rotation):
    rot=np.asarray(rotation,dtype=float).reshape(3,3)
    yaw=float(np.arctan2(rot[1,0],rot[0,0]))
    c,s=np.cos(yaw),np.sin(yaw)
    return yaw,np.array([[c,s,0],[-s,c,0],[0,0,1.]])


class ObservationEncoder:
    """All memory is computed from controller-visible pose/velocity/target only.

    No forces, mass, motor efficiency, actual torque, or simulator parameters enter
    this interface. Keep one instance per episode and reset at mission start.
    """
    schema_id = OBSERVATION_SCHEMA_ID

    def __init__(self,config=None):
        self.config=config or EncoderConfig()
        self.reset()

    def reset(self,yaw_reference=None):
        self.yaw_reference=yaw_reference
        self.integral_world=np.zeros(3)
        self.acceleration_world=np.zeros(3)
        self.previous_velocity=None
        self.time=0.
        self.history=deque()
        self.horizon_s=30.

    def set_task(self,horizon_s,mission_kind='hover'):
        if not np.isfinite(horizon_s) or horizon_s<=0:raise ValueError('Positive finite task horizon required')
        self.horizon_s=float(horizon_s)

    def observe(self,position,velocity,rotation,body_rates,target,previous_action=None,dt=.02):
        cfg=self.config
        pos,vel,rot,rates,goal=map(np.asarray,(position,velocity,rotation,body_rates,target))
        raw=np.r_[pos,vel,rot.reshape(-1),rates,goal]
        if not np.all(np.isfinite(raw)) or not np.isfinite(dt) or dt<=0:
            raise ValueError("finite measured state and positive elapsed time required")
        if previous_action is None:previous_action=np.zeros(4)
        action=np.asarray(previous_action,dtype=float)
        if action.shape!=(4,) or not np.all(np.isfinite(action)):raise ValueError("invalid prior action")
        yaw,heading=heading_rotation(rot)
        if self.yaw_reference is None:self.yaw_reference=yaw
        error=goal-pos
        if cfg.goal_clip_m>0:
            length=float(np.linalg.norm(error))
            if length>cfg.goal_clip_m:error=error*(cfg.goal_clip_m/length)
        if self.previous_velocity is not None:
            self.integral_world+=error*dt
            # XY norm limit preserves invariance to global yaw; Z is separate.
            length=np.linalg.norm(self.integral_world[:2])
            if length>cfg.integral_limit:self.integral_world[:2]*=cfg.integral_limit/length
            self.integral_world[2]=np.clip(self.integral_world[2],-cfg.integral_limit,cfg.integral_limit)
            observed_acceleration=(vel-self.previous_velocity)/dt
            alpha=1-np.exp(-dt/cfg.acceleration_tau_s)
            self.acceleration_world+=alpha*(observed_acceleration-self.acceleration_world)
            self.time+=dt
        self.previous_velocity=vel.copy()
        base=np.r_[heading@error/cfg.position_scale,heading@vel/cfg.velocity_scale,
                   (heading@rot).reshape(-1),rates/cfg.rate_scale,np.clip(action,-1,1)]
        yaw_error=self.yaw_reference-yaw
        current=np.r_[base,heading@self.integral_world/cfg.integral_limit,
                      heading@self.acceleration_world/cfg.acceleration_scale,
                      np.sin(yaw_error),np.cos(yaw_error)]
        self.history.append((self.time,base.copy(),heading.copy()))
        oldest = self.time - max(cfg.history_times_s, default=0.) if cfg.history else self.time
        while len(self.history) > 1 and self.history[1][0] <= oldest:
            self.history.popleft()
        if cfg.history:
            history_list=list(self.history)
            old=[]
            for lag in cfg.history_times_s:
                selected,old_heading=history_list[0][1:]
                for stamp,frame,frame_heading in history_list:
                    if stamp<=self.time-lag+1e-9:selected,old_heading=frame,frame_heading
                    else:break
                # Historical world-vector features must use the CURRENT heading.
                # Historical body rates/actions retain their original body axes.
                transported=selected.copy()
                change=heading@old_heading.T
                transported[:3]=change@selected[:3]
                transported[3:6]=change@selected[3:6]
                transported[6:15]=(change@selected[6:15].reshape(3,3)).reshape(-1)
                old.append(transported)
            current=np.r_[current,*old]
        if cfg.include_task_context:
            current=np.r_[current,max(0.,1-self.time/self.horizon_s),self.horizon_s/40.]
        if current.shape!=(cfg.size,):raise AssertionError("encoder contract size mismatch")
        return current.astype(np.float32)

