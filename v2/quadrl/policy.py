"""Measured-only temporal observations for a pure upper-loop RL replacement."""
from collections import deque
from dataclasses import dataclass,asdict
import json
import numpy as np

G=9.80665
MASS=1.75
INERTIA=np.array([.01251888,.01742669,.02380042],dtype=float)
ROTOR_POSITIONS=np.array([[.13,-.22,.023],[-.13,.20,.023],[.13,.22,.023],[-.13,-.20,.023]])
RATE_LIMIT=np.array([2.,2.,1.])
MAX_FORCE=2*MASS*G/4
CONTRACT_VERSION="measured-memory-yaw-v2.0"


@dataclass(frozen=True)
class EncoderConfig:
    history: bool=True
    integral_limit: float=6.
    acceleration_tau_s: float=.08
    position_scale: float=3.
    velocity_scale: float=3.
    rate_scale: float=3.
    acceleration_scale: float=5.
    history_times_s: tuple=(.1,.2)

    @property
    def size(self):return 30+22*len(self.history_times_s) if self.history else 30


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
    def __init__(self,config=None):
        self.config=config or EncoderConfig()
        self.reset()

    def reset(self,yaw_reference=None):
        self.yaw_reference=yaw_reference
        self.integral_world=np.zeros(3)
        self.acceleration_world=np.zeros(3)
        self.previous_velocity=None
        self.time=0.
        self.history=deque(maxlen=128)

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
        if current.shape!=(cfg.size,):raise AssertionError("encoder contract size mismatch")
        return current.astype(np.float32)


def decode_hybrid(action):
    a=np.clip(np.asarray(action,dtype=float),-1,1)
    return float((a[0]+1)/2),a[1:]*RATE_LIMIT


class NumpyPolicy:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            self.weights=[z[f"w{i}"].copy() for i in range(int(z["layers"]))]
            self.biases=[z[f"b{i}"].copy() for i in range(int(z["layers"]))]
            self.architecture=str(z["architecture"])
            self.contract=str(z["contract"])
            config=json.loads(str(z["encoder_config"]))
            config["history_times_s"]=tuple(config["history_times_s"])
            self.encoder=ObservationEncoder(EncoderConfig(**config))
        if self.contract!=CONTRACT_VERSION:raise ValueError("policy observation contract mismatch")
        if self.architecture not in ("hybrid","direct"):raise ValueError("unknown control architecture")
        if self.weights[0].shape[1]!=self.encoder.config.size or self.weights[-1].shape[0]!=4:
            raise ValueError("invalid model input/output dimensions")
        self.reset()

    def reset(self,yaw_reference=None):
        self.previous_action=np.zeros(4,dtype=np.float32)
        self.encoder.reset(yaw_reference)

    def predict(self,observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.shape[-1]!=self.encoder.config.size:raise ValueError("incorrect observation size")
        if not np.all(np.isfinite(x)):raise ValueError("non-finite observation")
        for index,(weights,bias) in enumerate(zip(self.weights,self.biases)):
            x=x@weights.T+bias
            if index<len(self.weights)-1:x=np.tanh(x)
        return np.clip(x,-1,1)

    def predict_enu(self,position,velocity,rotation,body_rates,target,dt=.02):
        obs=self.encoder.observe(position,velocity,rotation,body_rates,target,self.previous_action,dt)
        action=self.predict(obs)
        self.previous_action=action.copy()
        if self.architecture=="direct":
            return {"architecture":"direct","action":action,"motor_fractions":(action+1)/2,"observation":obs}
        thrust,rates=decode_hybrid(action)
        return {"architecture":"hybrid","action":action,"thrust_fraction":thrust,"rates_flu":rates,
                "rates_frd":rates*np.array([1.,-1.,-1.]),"observation":obs}


def load_policy(path):return NumpyPolicy(path)


def export_policy(model,path,architecture,encoder_config):
    import torch
    layers=[m for m in model.policy.mlp_extractor.policy_net if isinstance(m,torch.nn.Linear)]
    layers.append(model.policy.action_net)
    values={"layers":np.array(len(layers)),"architecture":np.array(architecture),
            "contract":np.array(CONTRACT_VERSION),"encoder_config":np.array(json.dumps(asdict(encoder_config)))}
    for index,layer in enumerate(layers):
        values[f"w{index}"]=layer.weight.detach().cpu().numpy()
        values[f"b{index}"]=layer.bias.detach().cpu().numpy()
    np.savez(path,**values)
