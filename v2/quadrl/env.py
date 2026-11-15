"""Fast PyBullet rigid body benchmark. This is NOT the ArduPilot controller.

All dynamics use SI ENU/FLU. Four physical rotor forces, yaw reaction torque,
first-order motor lag, rigid body inertia, drag, external wind force. Hybrid keeps
a simple inner angular-rate PI. Direct sends only four rotor thrust demands.
"""
import math,json
from dataclasses import replace,asdict
from pathlib import Path
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as pb
from .policy import MASS, INERTIA, MAX_FORCE, G, ROTOR_POSITIONS, ObservationEncoder, EncoderConfig, decode_hybrid
from .protocol import SCENARIOS, Mission, scenario_spec, force_N, seeded_rng, CONTROL_DT
from .reward import reward_value,REWARD_VERSION
from reference.direct_physics import DirectMotorModel,DirectMotorConfig

PLANT_PRIOR=json.loads(Path(__file__).with_name("plant_config.json").read_text())


class QuadEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, architecture="hybrid", training=False, scenario="nominal",
                 mission="hover", seconds=None, render_mode=None, reward_variant="full",
                 history=True, plant_config=None,reward_version=REWARD_VERSION):
        super().__init__()
        if architecture not in ("hybrid", "direct"):
            raise ValueError(architecture)
        self.architecture, self.training = architecture, training
        self.scenario, self.mission_kind = scenario, mission
        self.encoder_config=EncoderConfig(history=history)
        self.encoder=ObservationEncoder(self.encoder_config)
        self.plant_config=dict(PLANT_PRIOR if plant_config is None else plant_config)
        self.reward_variant = reward_variant
        self.reward_version=reward_version
        self.render_mode = render_mode
        self.dt, self.decimation = .005, 4
        self.control_dt = self.dt*self.decimation
        self.seconds=seconds
        self.mission=Mission(mission)
        self.max_steps=round((seconds if seconds is not None else self.mission.horizon_s)/self.control_dt)
        self.action_space = spaces.Box(-1., 1., (4,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.encoder_config.size,), np.float32)
        self.cid = pb.connect(pb.GUI if render_mode == "human" else pb.DIRECT)
        pb.setGravity(0, 0, -G, physicsClientId=self.cid)
        pb.setTimeStep(self.dt, physicsClientId=self.cid)
        # Total rigid mass1.75kg/inertia/rotorgeometry from official Gazebo Iris.
        # Aerodynamics are an identified local proxy, not a blade-element replica.
        self.rotors = ROTOR_POSITIONS.copy()
        # Body reaction torque opposes the officialrotor0/1 positive spin.
        self.spin = np.array([-1., -1., 1., 1.])
        self.yaw_coeff = .016
        matrix = np.vstack([np.ones(4), self.rotors[:, 1], -self.rotors[:, 0], self.yaw_coeff*self.spin])
        self.mix_inv = np.linalg.inv(matrix)
        collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[.13, .10, .035], physicsClientId=self.cid)
        visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[.13,.10,.035], rgbaColor=[.15,.35,.85,1], physicsClientId=self.cid)
        self.body = pb.createMultiBody(MASS, collision, visual, basePosition=[0,0,3], physicsClientId=self.cid)
        plane = pb.createCollisionShape(pb.GEOM_PLANE, physicsClientId=self.cid)
        pb.createMultiBody(0, plane, physicsClientId=self.cid)
        self.training_progress = 0.
        self.reset(seed=0)

    def state(self):
        pos, quat = pb.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)
        vel, omega = pb.getBaseVelocity(self.body, physicsClientId=self.cid)
        rot = np.asarray(pb.getMatrixFromQuaternion(quat)).reshape(3,3)
        return np.asarray(pos), np.asarray(vel), rot, rot.T @ omega

    def observation(self, state=None):
        pos, vel, rot, rates = self.state() if state is None else state
        # Memory/finite differences use the same NOISY measurements as controllers.
        pos=pos+self.sensor_rng.normal(0,self.parameters.get("noise_position",0.),3)
        vel=vel+self.sensor_rng.normal(0,self.parameters.get("noise_velocity",0.),3)
        rates=rates+self.sensor_rng.normal(0,self.parameters.get("noise_rates",0.),3)
        orientation_noise=self.sensor_rng.normal(0,self.parameters.get("noise_orientation",0.),3)
        noise_rot=np.asarray(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler(orientation_noise))).reshape(3,3)
        rot=rot@noise_rot
        self.measured_state={"position":pos.copy(),"velocity":vel.copy(),"rotation":rot.copy(),"body_rates":rates.copy()}
        return self.encoder.observe(pos,vel,rot,rates,self.target,self.previous_action,self.control_dt)

    def _training_parameters(self,rng):
        level=min(1.,.15+self.training_progress)
        # Absolute-time curriculum will reach full domain after1M interactions.
        spec={"name":"training_randomized","force_events":[]}
        if rng.random()<.8:
            spec["mass_scale"]=rng.uniform(1-.2*level,1+.5*level)
        if rng.random()<.35:
            spec["motor_scale"]=rng.uniform(1-.3*level,1.)
            spec["motor_index"]=int(rng.integers(0,4))
        if level>.5 and rng.random()<.45:spec["delay_steps"]=int(rng.integers(0,4))
        if level>.5 and rng.random()<.3:
            strength=rng.uniform(.2,1.)*level
            spec.update(noise_position=.03*strength,noise_velocity=.06*strength,
                        noise_orientation=.01*strength,noise_rates=.03*strength)
        if rng.random()<.8:
            magnitude=rng.uniform(.3,3.5)*level
            angle=rng.uniform(-np.pi,np.pi)
            direction=np.array([np.cos(angle),np.sin(angle),0.])
            if rng.random()<.5:
                spec["force_events"]=[{"on_s":float(rng.uniform(0,4)),"off_s":float(self.max_steps*self.control_dt+1),"force_enu_N":(direction*magnitude).tolist()}]
            else:
                for on in (3.,10.,17.):
                    a=angle+rng.uniform(-np.pi,np.pi)
                    spec["force_events"].append({"on_s":on,"off_s":on+float(rng.uniform(2,5)),
                                                 "force_enu_N":[float(magnitude*np.cos(a)),float(magnitude*np.sin(a)),0.]})
        return spec

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        episode_seed=int(seed) if seed is not None else int(rng.integers(0,2**31))
        self.sensor_rng=seeded_rng(episode_seed,220)
        initial_rng=seeded_rng(episode_seed,330)
        self.parameters=scenario_spec(self.scenario,episode_seed)
        self.mission=Mission(self.mission_kind)
        if self.training:
            self.mission=Mission("route" if self.training_progress>.5 and rng.random()<.4 else "hover")
            self.max_steps=round((self.seconds if self.seconds is not None else self.mission.horizon_s)/self.control_dt)
            self.parameters=self._training_parameters(rng)
        self.max_steps=round((self.seconds if self.seconds is not None else self.mission.horizon_s)/self.control_dt)
        self.mass = MASS*self.parameters.get("mass_scale",1.)
        self.inertia = INERTIA*self.parameters.get("mass_scale",1.)
        if self.architecture=="direct":
            # Gazebo mass perturbations scale EVERY link's mass/inertia,
            # including rotor angular momentum. The nominal action calibration
            # remains fixed: these are plant properties, never actor inputs.
            cfg=DirectMotorConfig();scale=self.parameters.get("mass_scale",1.)
            self.inertia=np.asarray(cfg.body_inertia_diagonal_kgm2)*scale
            self.direct_model=DirectMotorModel(replace(cfg,rotor_inertia_kgm2=cfg.rotor_inertia_kgm2*scale))
            self.direct_model.reset(pwm=cfg.nominal_hover_pwm)
        pb.changeDynamics(self.body,-1,mass=self.mass,localInertiaDiagonal=self.inertia,
                          linearDamping=0,angularDamping=0,physicsClientId=self.cid)
        self.target=self.mission.target
        spread=.1 if self.training and self.training_progress<.25 else .2
        pos=self.target+initial_rng.uniform(-spread,spread,3)
        rpy=initial_rng.uniform(-.035,.035,3)
        rpy[2]=initial_rng.uniform(-np.pi,np.pi)
        vel=initial_rng.uniform(-.03,.03,3)
        self.yaw_reference=float(rpy[2])
        if self.training:rpy[2]+=rng.uniform(-.25,.25)*min(1.,self.training_progress+.2)
        self.initial_state = {"position":pos.tolist(),"rpy":rpy.tolist(),"velocity":vel.tolist()}
        pb.resetBasePositionAndOrientation(self.body,pos,pb.getQuaternionFromEuler(rpy),physicsClientId=self.cid)
        pb.resetBaseVelocity(self.body,vel,[0,0,0],physicsClientId=self.cid)
        self.steps=0
        self.encoder.reset(self.yaw_reference)
        self.previous_action=np.zeros(4)
        self.motor_forces=np.ones(4)*MASS*G/4
        self.collective_force=MASS*G
        self.rate_integral=np.zeros(3)
        self.rate_queues=[deque([0.]*round(delay/self.dt)) for delay in self.plant_config.get("rate_delay_s",[0,0,0])]
        self.delay=deque([np.zeros(4) for _ in range(self.parameters.get("delay_steps",0))])
        self.trajectory=[]
        self.mission.update(0.,pos)
        return self.observation(),{"initial_state":self.initial_state,"parameters":self.parameters.copy(),"yaw_reference":self.yaw_reference}

    def step(self, action):
        desired=np.clip(np.asarray(action,dtype=float),-1,1)
        if desired.shape!=(4,) or not np.all(np.isfinite(desired)):
            raise ValueError("Action must contain four finite normalized values")
        self.delay.append(desired)
        applied=self.delay.popleft()
        old_action=self.previous_action.copy()
        before_position=self.state()[0]
        error_before=float(np.linalg.norm(before_position-self.target))
        thrust, rates_cmd=decode_hybrid(applied)
        for _ in range(self.decimation):
            pos,vel,rot,rates=self.state()
            if self.architecture=="hybrid":
                delayed_rates=np.zeros(3)
                for axis,queue in enumerate(self.rate_queues):
                    queue.append(rates_cmd[axis]);delayed_rates[axis]=queue.popleft()
                delayed_rates*=np.asarray(self.plant_config["rate_gain"])
                limit=self.plant_config["rate_integral_limit"]
                self.rate_integral=np.clip(self.rate_integral+(delayed_rates-rates)*self.dt,-limit,limit)
                # Fixed nominal controller gains. The controller must NOT see
                # randomized true inertia/mass from the plant.
                torque=INERTIA*(np.asarray(self.plant_config["rate_kp"])*(delayed_rates-rates)+np.asarray(self.plant_config["rate_ki"])*self.rate_integral)
                self.collective_force+=(1-math.exp(-self.dt/self.plant_config["collective_tau_s"]))* (thrust*4*MAX_FORCE-self.collective_force)
                forces=self.mix_inv @ np.r_[self.collective_force,torque]
                forces=np.clip(forces,0,MAX_FORCE)
                # Identified rate dynamics already include aggregate motor effects.
                self.motor_forces=forces
            else:
                # No position, attitude or rate controller in this branch.
                efficiencies=np.ones(4)
                efficiencies[self.parameters.get("motor_index",0)]=self.parameters.get("motor_scale",1.)
                rotor_result=self.direct_model.step_action(applied,self.dt,body_rates=rates,
                    body_velocity=rot.T@vel,blade_efficiency=efficiencies)
                self.motor_forces=rotor_result["forces_N"].copy()
            actual=self.motor_forces.copy()
            if self.architecture=="hybrid":
                actual[self.parameters.get("motor_index",0)]*=self.parameters.get("motor_scale",1.)
                body_torque=np.array([self.rotors[:,1]@actual,-self.rotors[:,0]@actual,
                                      self.yaw_coeff*(self.spin@actual)])
            else:body_torque=rotor_result["body_torque_Nm"]
            world_force=rot[:,2]*actual.sum()
            wind=force_N(self.steps*self.control_dt,self.parameters)
            # Applied external FORCE is in newtons, not a claimed wind speed.
            world_force+=wind
            # Direct blade inflow already generates vertical drag and local
            # rotor damping. Adding the identified hybrid drag would double it.
            if self.architecture=="hybrid":world_force-=np.asarray(self.plant_config["linear_drag_N_per_mps"])*vel
            pb.applyExternalForce(self.body,-1,world_force,pos,pb.WORLD_FRAME,physicsClientId=self.cid)
            pb.applyExternalTorque(self.body,-1,rot@body_torque,pb.WORLD_FRAME,physicsClientId=self.cid)
            pb.stepSimulation(physicsClientId=self.cid)
        self.steps+=1
        self.previous_action=desired.copy()
        pos,vel,rot,rates=state=self.state()
        judged_target=self.target.copy()
        event=self.mission.update(self.steps*self.control_dt,pos)
        self.target=self.mission.target
        # Reward/trace error refer to the goal active DURING this action, before
        # a confirmed arrival switches the next observation to the next goal.
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
        info={"position":pos.tolist(),"velocity":vel.tolist(),"target":judged_target.tolist(),"next_target":self.target.tolist(),
              "error":tracking_error,"altitude_error":float(pos[2]-judged_target[2]),"tilt_cos":float(rot[2,2]),
              "yaw":yaw,"yaw_error":yaw_error,"mission_event":event,"route_completion_s":self.mission.completion_time,
              "force_enu_N":wind.tolist(),"goal_index":self.mission.index,
              "body_rates":rates.tolist(),"motor_forces":self.motor_forces.tolist(),
              "actual_motor_forces":actual.tolist(),
              "failed":terminated,"time":self.steps*self.control_dt}
        return self.observation(state),float(reward),terminated,truncated,info

    def render(self):
        pos=self.state()[0]
        view=pb.computeViewMatrixFromYawPitchRoll(pos,3.5,45,-25,0,2)
        proj=pb.computeProjectionMatrixFOV(55,16/9,.05,50)
        _,_,rgba,_,_=pb.getCameraImage(960,540,view,proj,renderer=pb.ER_TINY_RENDERER,physicsClientId=self.cid)
        # PyBullet builds without NumPy support return a flat RGBA tuple.
        return np.asarray(rgba,dtype=np.uint8).reshape(540,960,4)[:,:,:3]

    def close(self):
        if pb.isConnected(self.cid): pb.disconnect(self.cid)

