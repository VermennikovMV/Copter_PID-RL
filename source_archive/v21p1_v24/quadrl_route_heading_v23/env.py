"""Balanced absolute route heading; preserve original relative yaw error."""
from __future__ import annotations
from collections import deque
import copy
import numpy as np
import pybullet as pb
from quadrl.policy import MASS,G
from quadrl_route_memory_train_v21.env import JointTaskEnv as ParentEnv
from quadrl_route_transfer_v22.initial import rotation


def wrapped(value):return float(np.arctan2(np.sin(value),np.cos(value)))


class JointTaskEnv(ParentEnv):
    def __init__(self,*,heading_enabled,matched_yaw,**kwargs):
        self.heading_enabled=bool(heading_enabled);self.matched_yaw=float(matched_yaw)
        self.heading_constructed=False;self.heading_constructor_resets=0
        self.heading_route_counter=0;self.heading_stage_seed=None;self.heading_resets=[]
        self.heading_steps=dict(matched_route=0,original_route=0,hover=0)
        self._capture_heading_sensor=False;self._heading={}
        super().__init__(conditioning_mode="zero_route_integral",**kwargs)
        self.heading_constructed=True

    def observation(self,state=None):
        if self._capture_heading_sensor:self._heading_sensor_state=copy.deepcopy(self.sensor_rng.bit_generator.state)
        result=super().observation(state)
        self._annotate()
        return result

    def _annotate(self):
        if self.current_actor_sample is not None:
            for key,value in self._heading.items():self.current_actor_sample["heading_"+key]=value

    def reset(self,seed=None,options=None):
        if not self.heading_constructed:
            self.heading_constructor_resets+=1
            return super().reset(seed=seed,options=options)
        if seed is not None:
            self.heading_route_counter=0;self.heading_stage_seed=int(seed);self.heading_resets=[]
            self.heading_steps=dict(matched_route=0,original_route=0,hover=0)
        self._heading={};self._capture_heading_sensor=True
        observation,info=super().reset(seed=seed,options=options)
        self._capture_heading_sensor=False
        initial=copy.deepcopy(self.initial_state)
        source_yaw=float(initial["rpy"][2]);source_ref=float(self.yaw_reference)
        route=self.mission.kind=="route"
        if route:self.heading_route_counter+=1
        matched=bool(self.heading_enabled and route and self.heading_route_counter%2==1)
        delta=self.matched_yaw-source_yaw if matched else 0.
        if matched:
            position,velocity,unused,body_rates=self.state()
            rpy=np.asarray(initial["rpy"],float).copy();rpy[2]=self.matched_yaw
            R=rotation(rpy)
            pb.resetBasePositionAndOrientation(self.body,position,pb.getQuaternionFromEuler(rpy),physicsClientId=self.cid)
            pb.resetBaseVelocity(self.body,velocity,R@body_rates,physicsClientId=self.cid)
            self.sensor_rng.bit_generator.state=copy.deepcopy(self._heading_sensor_state)
            self.yaw_reference=source_ref+delta
            self.encoder.reset(self.yaw_reference);self.path_context.reset();self.conditioner.reset(self.yaw_reference)
            # Preserve the previous action audit sample through DummyVecEnv's
            # automatic reset, exactly like ParentEnv. It is not actor memory.
            self.current_actor_sample=None
            self.previous_action=np.zeros(4);self.rate_model.reset();self.rate_integral=np.zeros(3)
            self.rate_queues=[deque([0.]*round(delay/self.dt)) for delay in self.plant_config.get("rate_delay_s",[0,0,0])]
            self.delay=deque([np.zeros(4) for unused in range(self.parameters.get("delay_steps",0))])
            self.motor_forces=np.ones(4)*MASS*G/4;self.collective_force=MASS*G;self.trajectory=[]
            self.initial_state=copy.deepcopy(initial);self.initial_state["rpy"]=rpy.tolist()
            actual=self.state()
            assert np.max(np.abs(actual[0]-position))<1e-12 and np.max(np.abs(actual[1]-velocity))<1e-12
            assert np.max(np.abs(actual[2]-R))<1e-12 and np.max(np.abs(actual[3]-body_rates))<1e-12
            observation=self.observation(actual)
            assert len(self.encoder.history)==1 and self.encoder.time==0. and self.conditioner.observation_calls==1
            assert np.count_nonzero(self.encoder.integral_world)==0 and not self.path_context.valid_leg
            assert np.count_nonzero(self.rate_model.target_rate)==np.count_nonzero(self.rate_model.target_accel)==0
        actual_yaw=float(self.initial_state["rpy"][2])
        before,after=wrapped(source_ref-source_yaw),wrapped(self.yaw_reference-actual_yaw)
        assert abs(wrapped(before-after))<1e-12
        self._heading=dict(matched=matched,route_ordinal=self.heading_route_counter if route else 0,
            source_actual_yaw=source_yaw,source_yaw_reference=source_ref,actual_yaw=actual_yaw,
            yaw_reference=float(self.yaw_reference),delta=float(delta),relative_error_before=before,relative_error_after=after)
        self._annotate()
        receipt=dict(**self._heading,mission_kind=self.mission.kind,stage_seed=self.heading_stage_seed,
            route_counter=self.heading_route_counter,original_initial=initial,actual_initial=copy.deepcopy(self.initial_state),
            enabled=self.heading_enabled,parent_observation_returned_exact=not matched,
            rebuilt_observation_count=int(matched),hidden_control_steps=0,extra_RNG_draws=0)
        self.heading_resets.append(receipt)
        info.update(initial_state=copy.deepcopy(self.initial_state),yaw_reference=float(self.yaw_reference),heading_reset=receipt)
        return observation,info

    def heading_snapshot(self):
        return dict(enabled=self.heading_enabled,stage_seed=self.heading_stage_seed,
            constructor_resets_discarded=self.heading_constructor_resets,route_counter=self.heading_route_counter,
            matched_routes=sum(r["matched"] for r in self.heading_resets),
            original_routes=sum(r["mission_kind"]=="route" and not r["matched"] for r in self.heading_resets),
            hover_resets=sum(r["mission_kind"]=="hover" for r in self.heading_resets),
            control_steps_by_stratum=self.heading_steps.copy(),
            seconds_by_stratum={name:count*self.control_dt for name,count in self.heading_steps.items()},
            resets=copy.deepcopy(self.heading_resets))

    def step(self,action):
        stratum="hover" if self.mission.kind=="hover" else "matched_route" if self._heading.get("matched",False) else "original_route"
        result=super().step(action)
        self.heading_steps[stratum]+=1
        return result
