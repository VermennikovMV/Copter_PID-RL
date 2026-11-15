"""Prospective native-rate V3 PyBullet branch, hybrid architecture only.

Inherits the frozen measured-only actor, mission judge, disturbances and reward.
Only the hybrid inner angular-rate dynamics and 400 Hz integration differ.
"""
import math
import numpy as np
import pybullet as pb
from quadrl.env import QuadEnv
from quadrl.policy import INERTIA, MAX_FORCE, decode_hybrid
from quadrl.protocol import force_N
from quadrl.reward import reward_value
from .native_rate_model_v3 import NativeRateModelV3, MODEL_VERSION


class QuadEnvJointV3(QuadEnv):
    def __init__(self, architecture='hybrid', **kwargs):
        if architecture!='hybrid':raise ValueError('Native-rate V3 is hybrid only')
        # Base __init__ dispatches reset before this subclass has its model.
        self.rate_model=None
        super().__init__(architecture=architecture,**kwargs)
        self.dt=.0025; self.decimation=8; self.control_dt=.02
        pb.setTimeStep(self.dt,physicsClientId=self.cid)
        self.rate_model=NativeRateModelV3(self.dt)
        self.plant_config['joint_rate_model_version']=MODEL_VERSION
        self.plant_config['joint_rate_fit_sha256']=self.rate_model.fit_sha256
        self.plant_config['joint_rate_diagnosis_sha256']=self.rate_model.fit['diagnosis_receipt_sha256']
        self.plant_config['joint_RP_replacement']={'kind':'native_jerk_shaper_plus_refit_first_order_torque_feedback',
            **self.rate_model.fit['RP_remainder']}
        self.reset(seed=0)

    def reset(self, seed=None, options=None):
        observation,info=super().reset(seed=seed,options=options)
        if self.rate_model is not None:self.rate_model.reset()
        info['joint_rate_model_version']=MODEL_VERSION
        return observation,info

    def step(self, action):
        # The mission/reward/observation tail below is copied without changing
        # semantics from frozen QuadEnv.step. Only inner hybrid dynamics differ.
        desired=np.clip(np.asarray(action,dtype=float),-1,1)
        if desired.shape!=(4,) or not np.all(np.isfinite(desired)):
            raise ValueError('Action must contain four finite normalized values')
        self.delay.append(desired)
        applied=self.delay.popleft()
        old_action=self.previous_action.copy()
        before_position=self.state()[0]
        error_before=float(np.linalg.norm(before_position-self.target))
        thrust,rates_cmd=decode_hybrid(applied)
        for _ in range(self.decimation):
            pos,vel,rot,rates=self.state()
            delayed_rates=self.rate_model.step(rates_cmd)
            # RP aggregate PI and aggregate delay are REPLACED. This feedback
            # creates the fitted exact discrete first-order response through
            # real rotor torque, finite mixer authority and rigid-body physics.
            torque=np.empty(3)
            torque[:2]=INERTIA[:2]*self.rate_model.feedback_coefficient*(delayed_rates[:2]-rates[:2])
            # Original yaw delayed PI, evaluated at the new physical dt. The
            # controller retains fixed nominal inertia, including fault cases.
            limit=self.plant_config['rate_integral_limit']
            self.rate_integral[2]=np.clip(self.rate_integral[2]+(delayed_rates[2]-rates[2])*self.dt,-limit,limit)
            torque[2]=INERTIA[2]*(self.plant_config['rate_kp'][2]*(delayed_rates[2]-rates[2])+self.plant_config['rate_ki'][2]*self.rate_integral[2])
            self.collective_force+=(1-math.exp(-self.dt/self.plant_config['collective_tau_s']))*(thrust*4*MAX_FORCE-self.collective_force)
            self.motor_forces=np.clip(self.mix_inv@np.r_[self.collective_force,torque],0,MAX_FORCE)
            actual=self.motor_forces.copy()
            actual[self.parameters.get('motor_index',0)]*=self.parameters.get('motor_scale',1.)
            body_torque=np.array([self.rotors[:,1]@actual,-self.rotors[:,0]@actual,self.yaw_coeff*(self.spin@actual)])
            world_force=rot[:,2]*actual.sum()
            wind=force_N(self.steps*self.control_dt,self.parameters)
            world_force+=wind
            world_force-=np.asarray(self.plant_config['linear_drag_N_per_mps'])*vel
            pb.applyExternalForce(self.body,-1,world_force,pos,pb.WORLD_FRAME,physicsClientId=self.cid)
            pb.applyExternalTorque(self.body,-1,rot@body_torque,pb.WORLD_FRAME,physicsClientId=self.cid)
            pb.stepSimulation(physicsClientId=self.cid)
        self.steps+=1
        self.previous_action=desired.copy()
        pos,vel,rot,rates=state=self.state()
        judged_target=self.target.copy()
        event=self.mission.update(self.steps*self.control_dt,pos)
        self.target=self.mission.target
        tracking_error=float(np.linalg.norm(pos-judged_target))
        error=float(np.linalg.norm(pos-self.target))
        yaw=float(np.arctan2(rot[1,0],rot[0,0]))
        yaw_error=float(np.arctan2(np.sin(self.yaw_reference-yaw),np.cos(self.yaw_reference-yaw)))
        terminated=bool(tracking_error>4 or pos[2]<.25 or pos[2]>10 or rot[2,2]<.2)
        reward=reward_value(tracking_error,float(vel@vel),float(rot[2,2]),yaw_error,
            float(rates@rates),float((desired-old_action)@(desired-old_action)),
            error_before-tracking_error,event is not None,
            event is not None and self.mission.completion_time is not None,
            terminated,self.control_dt,self.reward_variant,self.reward_version)
        truncated=self.steps>=self.max_steps
        info={'position':pos.tolist(),'velocity':vel.tolist(),'target':judged_target.tolist(),'next_target':self.target.tolist(),
              'error':tracking_error,'altitude_error':float(pos[2]-judged_target[2]),'tilt_cos':float(rot[2,2]),
              'yaw':yaw,'yaw_error':yaw_error,'mission_event':event,'route_completion_s':self.mission.completion_time,
              'force_enu_N':wind.tolist(),'goal_index':self.mission.index,
              'body_rates':rates.tolist(),'motor_forces':self.motor_forces.tolist(),
              'actual_motor_forces':actual.tolist(),
              'failed':terminated,'time':self.steps*self.control_dt,
              'joint_rate_model_version':MODEL_VERSION}
        return self.observation(state),float(reward),terminated,truncated,info


# Explicit alias permits existing wrappers to select this new namespace.
QuadEnvV3=QuadEnvJointV3
