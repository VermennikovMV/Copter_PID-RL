"""Same heading strata; full measured startup bundle on matched routes only."""
from __future__ import annotations
from collections import deque
import copy
import numpy as np
import pybullet as pb
from quadrl.policy import MASS,G
from quadrl.protocol import Mission
from quadrl_corner_v8.env import CornerPeakReward,CORNER_WEIGHT
from quadrl_route_memory_train_v21.env import JointTaskEnv as ParentEnv
from quadrl_route_transfer_v22.initial import rotation


def wrapped(value):return float(np.arctan2(np.sin(value),np.cos(value)))


class JointTaskEnv(ParentEnv):
    def __init__(self,*,heading_enabled,matched_yaw,full_initial_enabled,deployment_state,**kwargs):
        self.full_initial_enabled=bool(full_initial_enabled);self.deployment_state=copy.deepcopy(deployment_state)
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
        full=bool(matched and self.full_initial_enabled)
        source_state=self.state() if full else None
        if matched:
            position,velocity,unused,body_rates=self.state()
            rpy=np.asarray(initial["rpy"],float).copy();rpy[2]=self.matched_yaw
            if full:
                position=np.asarray(self.deployment_state["position"],float)
                velocity=np.asarray(self.deployment_state["velocity"],float)
                rpy=np.asarray(self.deployment_state["rpy"],float)
                body_rates=np.asarray(self.deployment_state["body_rates"],float)
                assert float(rpy[2])==self.matched_yaw
            R=rotation(rpy)
            pb.resetBasePositionAndOrientation(self.body,position,pb.getQuaternionFromEuler(rpy),physicsClientId=self.cid)
            pb.resetBaseVelocity(self.body,velocity,R@body_rates,physicsClientId=self.cid)
            self.sensor_rng.bit_generator.state=copy.deepcopy(self._heading_sensor_state)
            self.yaw_reference=float(rpy[2]) if full else source_ref+delta
            if full:
                self.mission=Mission("route");self.target=self.mission.target
                self.mission.update(0.,position)
                self.corner_tracker=CornerPeakReward("route",weight=CORNER_WEIGHT)
                self.corner_tracker.observe(0.,position,self.target)
                self.coverage_hold_steps=0;self.completion_event_count=0
                self.metric_cost_tracker.reset()
            self.encoder.reset(self.yaw_reference);self.path_context.reset();self.conditioner.reset(self.yaw_reference)
            # Preserve the previous action audit sample through DummyVecEnv's
            # automatic reset, exactly like ParentEnv. It is not actor memory.
            self.current_actor_sample=None
            self.previous_action=np.zeros(4);self.rate_model.reset();self.rate_integral=np.zeros(3)
            self.rate_queues=[deque([0.]*round(delay/self.dt)) for delay in self.plant_config.get("rate_delay_s",[0,0,0])]
            self.delay=deque([np.zeros(4) for unused in range(self.parameters.get("delay_steps",0))])
            self.motor_forces=np.ones(4)*MASS*G/4;self.collective_force=MASS*G;self.trajectory=[]
            self.initial_state=copy.deepcopy(initial);self.initial_state["rpy"]=rpy.tolist()
            if full:self.initial_state.update(position=position.tolist(),velocity=velocity.tolist(),body_rates=body_rates.tolist())
            actual=self.state()
            assert np.max(np.abs(actual[0]-position))<1e-12 and np.max(np.abs(actual[1]-velocity))<1e-12
            assert np.max(np.abs(actual[2]-R))<1e-12 and np.max(np.abs(actual[3]-body_rates))<1e-12
            observation=self.observation(actual)
            assert len(self.encoder.history)==1 and self.encoder.time==0. and self.conditioner.observation_calls==1
            assert np.count_nonzero(self.encoder.integral_world)==0 and not self.path_context.valid_leg
            assert np.count_nonzero(self.rate_model.target_rate)==np.count_nonzero(self.rate_model.target_accel)==0
        actual_yaw=float(self.initial_state["rpy"][2])
        before,after=wrapped(source_ref-source_yaw),wrapped(self.yaw_reference-actual_yaw)
        if full:assert abs(after)<1e-12
        else:assert abs(wrapped(before-after))<1e-12
        self._heading=dict(matched=matched,route_ordinal=self.heading_route_counter if route else 0,
            source_actual_yaw=source_yaw,source_yaw_reference=source_ref,actual_yaw=actual_yaw,
            yaw_reference=float(self.yaw_reference),delta=float(delta),relative_error_before=before,relative_error_after=after)
        self._annotate()
        receipt=dict(**self._heading,mission_kind=self.mission.kind,stage_seed=self.heading_stage_seed,
            route_counter=self.heading_route_counter,original_initial=initial,actual_initial=copy.deepcopy(self.initial_state),
            enabled=self.heading_enabled,parent_observation_returned_exact=not matched,
            rebuilt_observation_count=int(matched),hidden_control_steps=0,extra_RNG_draws=0)
        receipt.update(full_initial_bundle=full,full_initial_enabled=self.full_initial_enabled,
            source_actual_state_before_bundle=None if source_state is None else {name:value.tolist() for name,value in zip(("position","velocity","rotation","body_rates"),source_state)},
            controller_reference_before_bundle=source_ref,controller_reference_after_bundle=float(self.yaw_reference),
            reset_components=["position","velocity","roll","pitch","body_rates","yaw_reference_initial_error_zero"] if full else [],
            mission_inside_since=self.mission.inside_since,mission_last_time=self.mission.last_time,
            mission_target=self.target.tolist(),mission_reinitialized_at_new_position=full,
            hidden_AP_state_matched=False)
        self.heading_resets.append(receipt)
        info.update(initial_state=copy.deepcopy(self.initial_state),yaw_reference=float(self.yaw_reference),heading_reset=receipt)
        return observation,info

    def heading_snapshot(self):
        return dict(enabled=self.heading_enabled,full_initial_enabled=self.full_initial_enabled,stage_seed=self.heading_stage_seed,
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
