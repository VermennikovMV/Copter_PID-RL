"""Causal AP4.7 rate-command shaping plus independently identified remainder.

No pose, quaternion, measured trace, mass or inertia enters this command model.
RP replaces the old aggregate PI/delay. Yaw retains the frozen delayed PI target.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np

MODEL_VERSION = 'native-rate-rp-refit-v3.0'
FIT_PATH = Path(__file__).with_name('native_rate_fit_v3.json')


class FractionalDelay:
    """Linear interpolation between past samples only, at a constant delay."""
    def __init__(self, seconds, dt):
        self.steps=float(seconds)/float(dt)
        self.whole=int(np.floor(self.steps)); self.fraction=self.steps-self.whole
        self.history=deque([0.]*(self.whole+2),maxlen=self.whole+2)

    def reset(self, value=0.):
        self.history=deque([float(value)]*self.history.maxlen,maxlen=self.history.maxlen)

    def step(self, value):
        self.history.append(float(value)); n=self.whole
        return (1-self.fraction)*self.history[-1-n]+self.fraction*self.history[-2-n]


class NativeRateModelV3:
    def __init__(self, dt=.0025, fit=None):
        if fit is None:
            fit_bytes=FIT_PATH.read_bytes();self.fit=json.loads(fit_bytes)
            self.fit_sha256=hashlib.sha256(fit_bytes).hexdigest()
        else:
            self.fit=fit;self.fit_sha256=hashlib.sha256(json.dumps(fit,sort_keys=True).encode()).hexdigest()
        if abs(float(dt)-.0025)>1e-12:
            raise ValueError('Native V3 rate shaping requires the identified 400 Hz step')
        self.dt=float(dt)
        p=self.fit['native_shaper']
        self.accel_limit=np.asarray(p['RP_accel_limit_rad_s2'],dtype=float)
        self.tc=np.asarray(p['RP_input_tc_s'],dtype=float)
        self.jerk_limit=self.accel_limit/self.tc
        r=self.fit['RP_remainder']
        self.gain=np.asarray(r['gain'],dtype=float)
        self.tau=np.asarray(r['tau_s'],dtype=float)
        self.feedback_coefficient=(1-np.exp(-self.dt/self.tau))/self.dt
        self.delays=[FractionalDelay(d,self.dt) for d in r['delay_s']]
        yaw=self.fit['yaw_retained']
        self.yaw_queue_length=round(yaw['delay_s']/self.dt)
        self.reset()

    def reset(self):
        self.target_rate=np.zeros(2); self.target_accel=np.zeros(2)
        for delay in self.delays: delay.reset()
        self.yaw_queue=deque([0.]*self.yaw_queue_length)
        self.last_command=np.zeros(3); self.last_delayed_target=np.zeros(3)

    def step(self, command_rates):
        command=np.asarray(command_rates,dtype=float)
        if command.shape!=(3,) or not np.all(np.isfinite(command)):
            raise ValueError('Three finite FLU angular-rate commands required')
        goal=np.clip((command[:2]-self.target_rate)/self.tc,-self.accel_limit,self.accel_limit)
        self.target_accel+=np.clip(goal-self.target_accel,-self.jerk_limit*self.dt,self.jerk_limit*self.dt)
        self.target_rate+=self.target_accel*self.dt
        target=np.empty(3)
        for axis,delay in enumerate(self.delays):target[axis]=self.gain[axis]*delay.step(self.target_rate[axis])
        self.yaw_queue.append(float(command[2])); target[2]=self.yaw_queue.popleft()*self.fit['yaw_retained']['gain']
        self.last_command=command.copy(); self.last_delayed_target=target.copy()
        return target
