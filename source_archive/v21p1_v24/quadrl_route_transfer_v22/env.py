"""Reset-only interventions on the exact native evaluation environment."""
from collections import deque
import copy
import json
import numpy as np
import pybullet as pb
from quadrl.policy import MASS,G
from quadrl.protocol import Mission,seeded_rng
from quadrl_route_basis_v15.env import NativeEvalEnv
from .initial import rotation,target_initial


class TransferNativeEnv(NativeEvalEnv):
    def __init__(self,*,intervention,measured_initial,physical_pulse,budget,**kwargs):
        if kwargs.get('training') is not False or kwargs.get('mission')!='route':
            raise ValueError('Development full40s route evaluation only')
        self.transfer_intervention=intervention;self.transfer_measured=copy.deepcopy(measured_initial)
        self.transfer_pulse=copy.deepcopy(physical_pulse);self.transfer_budget=budget
        self.transfer_constructed=False;self.transfer_reset_receipts=[]
        super().__init__(**kwargs)
        self.transfer_constructed=True

    def reset(self,seed=None,options=None):
        if not self.transfer_constructed:
            return super().reset(seed=seed,options=options)
        if seed!=84001:raise ValueError('Only fixed development84001')
        # Parent reset performs an observation. Discard it, then restore sensor RNG
        # and ALL controller-visible memory before the first effective observation.
        discarded,info=super().reset(seed=seed,options=options)
        assert self.steps==0 and self.control_dt==.02 and self.max_steps==2000
        original=copy.deepcopy(self.initial_state)
        initial=target_initial(original,self.transfer_measured,self.transfer_intervention)
        R=rotation(initial['rpy'])
        pb.resetBasePositionAndOrientation(self.body,initial['position'],pb.getQuaternionFromEuler(initial['rpy']),physicsClientId=self.cid)
        pb.resetBaseVelocity(self.body,initial['velocity'],R@np.asarray(initial['body_rates']),physicsClientId=self.cid)
        if self.transfer_intervention=='full_initial_physical_pulse':
            assert self.scenario=='force_strong'
            self.parameters=copy.deepcopy(self.parameters)
            self.parameters['force_events']=[copy.deepcopy(self.transfer_pulse)]
        self.sensor_rng=seeded_rng(84001,220)
        self.steps=0;self.mission=Mission('route');self.target=self.mission.target
        self.yaw_reference=float(initial['rpy'][2]);self.encoder.reset(self.yaw_reference)
        self.path_context.reset();self.current_actor_sample=None;self.last_step_actor_sample=None
        self.previous_action=np.zeros(4)
        self.rate_model.reset();self.rate_integral=np.zeros(3)
        self.rate_queues=[deque([0.]*round(delay/self.dt)) for delay in self.plant_config.get('rate_delay_s',[0,0,0])]
        self.delay=deque([np.zeros(4) for _ in range(self.parameters.get('delay_steps',0))])
        self.motor_forces=np.ones(4)*MASS*G/4;self.collective_force=MASS*G;self.trajectory=[]
        self.initial_state=copy.deepcopy(initial)
        self.mission.update(0.,np.asarray(initial['position']))
        actual=self.state()
        expected=[initial['position'],initial['velocity'],R,initial['body_rates']]
        state_errors={name:float(np.max(abs(a-np.asarray(e)))) for name,a,e in zip(('position','velocity','rotation','body_rates'),actual,expected)}
        assert max(state_errors.values())<1e-11
        assert np.count_nonzero(self.rate_model.target_rate)==np.count_nonzero(self.rate_model.target_accel)==0
        assert all(np.count_nonzero(list(q))==0 for q in [self.rate_model.yaw_queue,*[d.history for d in self.rate_model.delays]])
        observation=self.observation(actual)
        reference_rng=seeded_rng(84001,220)
        for _ in range(4):reference_rng.normal(0.,0.,3)
        assert json.dumps(self.sensor_rng.bit_generator.state,sort_keys=True)==json.dumps(reference_rng.bit_generator.state,sort_keys=True)
        assert len(self.encoder.history)==1 and self.encoder.time==0. and self.encoder.previous_velocity is not None
        assert np.count_nonzero(self.encoder.integral_world)==0 and not self.path_context.valid_leg
        assert all(np.array_equal(self.measured_state[key],value) for key,value in zip(('position','velocity','rotation','body_rates'),actual))
        record=dict(native_original_initial=original,injected_initial=initial,intervention=self.transfer_intervention,
            first_state_max_errors=state_errors,parent_reset_observation_discarded=True,sensor_rng_restored_before_one_effective_observation=True,
            base74_history_rows=1,base74_time_s=0.,raw_env_80_observations_after_reset=1,
            steps=0,hidden_physics_steps=0,plant_rate_queues_zero_before_effective_observation=True,
            native_force_events=copy.deepcopy(self.parameters.get('force_events',[])),
            source_Gazebo_hidden_lower_controller_state_matched=False,
            force_application='Frozen original50Hz control-interval force lookup; actual source timestamps are not silently rounded in plan, realized force change logged by evaluator.')
        self.transfer_reset_receipts.append(record)
        info.update(initial_state=copy.deepcopy(self.initial_state),parameters=copy.deepcopy(self.parameters),yaw_reference=self.yaw_reference,transfer_reset=record)
        return observation,info

    def step(self,action):
        if self.transfer_budget['control_steps']>=20000:raise RuntimeError('Fixed20k control budget exhausted')
        self.transfer_budget['control_steps']+=1
        return super().step(action)
