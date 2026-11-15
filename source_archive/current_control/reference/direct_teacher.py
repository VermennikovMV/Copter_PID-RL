"""Development-only measured-state teacher for the detailed direct motor plant.

This controller generates offline demonstrations. It is not part of a deployed
neural policy. Rotor state is estimated from past commands with nominal physics.
"""
import numpy as np
from reference.dob import DOBReference,DOBConfig
from reference.direct_physics import DirectMotorModel
from quadrl.baseline import read_pid_config
from quadrl.policy import INERTIA,MAX_FORCE,RATE_LIMIT


class DirectTeacher:
    architecture='direct'
    def __init__(self,dt=.02):
        self.dt=dt
        from dataclasses import asdict
        values=asdict(read_pid_config());values['enabled']=True
        self.upper=DOBReference(dt, DOBConfig(**values))
        self.motor=DirectMotorModel()
        self.reset()

    def reset(self):
        self.upper.reset();self.motor.reset()
        self.integral=np.zeros(2);self.shaped=np.zeros(3)

    def metadata(self):
        return {'architecture':'direct','method':'offline measured-state DOB plus nominal momentum-aware motor allocation',
                'runtime_teacher_required':False,'rate_p':20,'rate_i':10,'yaw_momentum_gain':.35}

    def predict(self,obs):
        x=np.asarray(obs,dtype=float)
        upper_obs=x.copy()
        # Only the offline DOB teacher needs collective, whereas the actor
        # observes its actual four motor commands without this conversion.
        upper_obs[18]=float(np.mean(x[18:22]))
        upper=np.clip(self.upper.predict(upper_obs),-1,1)
        rates=x[15:18]*3
        command=upper[1:]*RATE_LIMIT
        self.shaped+=np.clip(command-self.shaped,-np.array([19.2,19.2,4.71])*self.dt,np.array([19.2,19.2,4.71])*self.dt)
        error=self.shaped-rates
        self.integral=np.clip(self.integral+error[:2]*self.dt,-.5,.5)
        torque=INERTIA[:2]*(20*error[:2]+10*self.integral)
        cfg=self.motor.config
        speed=np.maximum(self.motor.speed_rad_s,50)
        spin=np.array(cfg.rotor_spin)
        yawrow=spin/(2*cfg.thrust_coefficient*speed)
        current_force=cfg.thrust_coefficient*speed**2
        yawtarget=yawrow@current_force-.35*INERTIA[2]/cfg.rotor_inertia_kgm2*error[2]
        pos=self.motor.positions
        matrix=np.vstack([np.ones(4),pos[:,1],-pos[:,0],yawrow])
        force=np.linalg.solve(matrix,np.r_[(upper[0]+1)*.5*4*MAX_FORCE,torque,yawtarget])
        action=np.clip(2*force/MAX_FORCE-1,-1,1)
        rotation=x[6:15].reshape(3,3)
        self.motor.step_action(action,self.dt,body_rates=rates,body_velocity=rotation.T@(x[3:6]*3))
        return action.astype(np.float32)
