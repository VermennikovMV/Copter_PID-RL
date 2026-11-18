"""Rotor dynamics of the Iris airframe (Gazebo Iris SDF parameters).

PWM -> motor speed -> blade lift/drag with inflow, rotor momentum and body torques.
NumPy only.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
import math
import hashlib
import json
import numpy as np

MASS=1.75
G=9.80665
# Iris SDF (ArduPilot/ardupilot_gazebo 082a0fe): parallel-axis sum about
# z_COM = 4*.025*.023/1.75 m; average each spinning rotor's Ixx/Iyy.
# The previous XY values froze every blade at its initial azimuth.
INERTIA=np.array([.012832785142857143,.01711278514285714,.023800416],dtype=float)
# Policy action scale: zero action remains nominal hover. This scale is not
# the physical rotor saturation limit and must not be changed with that limit.
MAX_FORCE=2*MASS*G/4
# Static Iris LiftDrag yaw/thrust = radius*cda/cla. This does not model the
# transient reaction of accelerating rotors.
YAW_MOMENT_PER_THRUST_M=.084*.10/4.25
# SDF torque limit 2.5 Nm balances .004*w + kQ*w**2; its positive root is
# w=620.8733492930925 rad/s. kT*w**2 gives this nominal static per-rotor cap.
# kT=2.166513048e-5 N/(rad/s)^2, kQ=4.2820493184e-8 Nm/(rad/s)^2.
MOTOR_FORCE_LIMIT_N=8.351555002074628
ROTOR_POSITIONS=np.array([[.13,-.22,.023],[-.13,.20,.023],[.13,.22,.023],[-.13,-.20,.023]])

@dataclass(frozen=True)
class DirectMotorConfig:
    rotor_inertia_kgm2: float = 0.000167604
    motor_velocity_p: float = 0.20
    motor_torque_limit_Nm: float = 2.5
    joint_damping_Nms: float = 0.004
    maximum_speed_command_rad_s: float = 838.0
    pwm_min: float = 1100.0
    pwm_max: float = 1900.0
    pwm_quantization: float = 1.0
    blade_radius_m: float = 0.084
    blade_area_m2: float = 0.002
    air_density_kg_m3: float = 1.2041
    alpha_zero_rad: float = 0.3
    lift_slope: float = 4.25
    drag_slope: float = 0.10
    nominal_mass_kg: float = 1.75
    gravity_mps2: float = 9.80665
    maximum_substep_s: float = 0.001
    # Physical rotor speed signs; body reaction signs are their negative.
    rotor_spin: tuple = (1.0, 1.0, -1.0, -1.0)
    rotor_positions_m: tuple = ((.13, -.22, .023), (-.13, .20, .023),
                               (.13, .22, .023), (-.13, -.20, .023))
    # XY rotor inertia averaged over spinning blade azimuth, from the same SDF.
    body_inertia_diagonal_kgm2: tuple = tuple(INERTIA)
    # Fixed nominal motor calibration; no condition-specific adaptation.
    nominal_hover_pwm: tuple | None = (1533.216, 1533.215, 1533.214, 1533.217)

    @property
    def thrust_coefficient(self):
        return self.air_density_kg_m3 * self.blade_area_m2 * self.blade_radius_m**2 * self.lift_slope * self.alpha_zero_rad

    @property
    def yaw_drag_coefficient(self):
        return self.air_density_kg_m3 * self.blade_area_m2 * self.blade_radius_m**3 * self.drag_slope * self.alpha_zero_rad


class DirectMotorModel:
    """Four SDF motors FR, RL, FL, RR, with rotor momentum and blade inflow.

    ``step_action`` accepts the same four normalized force requests as the direct controller.
    ``step_pwm`` accepts recorded physical PWM for independent validation.
    Returned forces already include vertical/local rotor inflow drag. Do not add
    the previous 1.55 N/(m/s) vertical drag again when integrating this model.
    """
    def __init__(self, config=None):
        self.config = config or DirectMotorConfig()
        self.spin = np.asarray(self.config.rotor_spin, dtype=float)
        self.positions = np.asarray(self.config.rotor_positions_m, dtype=float)
        self.backend = "numpy"
        self.reset()

    def runtime_provenance(self):
        return {"backend": self.backend, "config": asdict(self.config),
                "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "native": None}

    def nominal_pwm_from_forces(self, forces):
        cfg = self.config
        forces = np.maximum(np.asarray(forces, dtype=float), 0.0)
        speed = np.sqrt(forces / cfg.thrust_coefficient)
        # Static P error balances bearing and aerodynamic resistance.
        setpoint = speed + (cfg.joint_damping_Nms * speed + cfg.yaw_drag_coefficient * speed**2) / cfg.motor_velocity_p
        return cfg.pwm_min + (cfg.pwm_max - cfg.pwm_min) * setpoint / cfg.maximum_speed_command_rad_s

    def action_to_pwm(self, action):
        cfg = self.config
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("four finite direct actions required")
        hover = np.asarray(cfg.nominal_hover_pwm) if cfg.nominal_hover_pwm is not None else self.nominal_pwm_from_forces(np.full(4, cfg.nominal_mass_kg * cfg.gravity_mps2 / 4))
        pwm = np.clip(cfg.pwm_min + (hover - cfg.pwm_min) * np.sqrt(np.clip(action, -1, 1) + 1), cfg.pwm_min, cfg.pwm_max)
        if cfg.pwm_quantization > 0:
            pwm = np.rint(pwm / cfg.pwm_quantization) * cfg.pwm_quantization
        return pwm

    def steady_speed(self, pwm):
        cfg = self.config
        demand = np.clip((np.asarray(pwm, dtype=float) - cfg.pwm_min) / (cfg.pwm_max - cfg.pwm_min), 0, 1) * cfg.maximum_speed_command_rad_s
        b = cfg.motor_velocity_p + cfg.joint_damping_Nms
        # Positive root of kQ*w^2+(P+bearing)*w-P*command=0.
        speed = 2 * cfg.motor_velocity_p * demand / (b + np.sqrt(b*b + 4*cfg.yaw_drag_coefficient*cfg.motor_velocity_p*demand))
        # Motor torque saturation also bounds attainable steady speed.
        cap = 2 * cfg.motor_torque_limit_Nm / (cfg.joint_damping_Nms + np.sqrt(cfg.joint_damping_Nms**2 + 4*cfg.yaw_drag_coefficient*cfg.motor_torque_limit_Nm))
        return np.minimum(speed, cap)

    def reset(self, forces=None, pwm=None):
        if forces is not None and pwm is not None:
            raise ValueError("choose forces or PWM for initial rotor state")
        cfg = self.config
        if pwm is not None:
            self.speed_rad_s = self.steady_speed(pwm)
        else:
            forces = np.full(4, cfg.nominal_mass_kg * cfg.gravity_mps2 / 4) if forces is None else np.asarray(forces, dtype=float)
            self.speed_rad_s = np.sqrt(np.maximum(forces, 0) / cfg.thrust_coefficient)
        if self.speed_rad_s.shape != (4,):
            raise ValueError("four rotor states required")

    def aerodynamic_load(self, speed, vertical_velocity, blade_efficiency):
        cfg = self.config
        tangential = cfg.blade_radius_m * np.maximum(speed, 0)
        inflow = np.asarray(vertical_velocity, dtype=float)
        flow_speed = np.hypot(tangential, inflow)
        alpha = cfg.alpha_zero_rad - np.arctan2(inflow, np.maximum(tangential, 1e-12))
        # Full model's near-hover domain is far from blade stall; bounded extension
        # follows the SDF linear/stall slopes for large disturbances.
        cl = np.where(alpha > 1.4, np.maximum(0, cfg.lift_slope * 1.4 - .025 * (alpha - 1.4)),
                      np.where(alpha < -1.4, np.minimum(0, -cfg.lift_slope * 1.4 - .025 * (alpha + 1.4)), cfg.lift_slope * alpha))
        cd = cfg.drag_slope * np.minimum(np.abs(alpha), 1.4)
        scale = cfg.air_density_kg_m3 * cfg.blade_area_m2 * flow_speed * blade_efficiency
        thrust = scale * (cl * tangential - cd * inflow)
        opposing_torque = cfg.blade_radius_m * scale * (cd * tangential + cl * inflow)
        active = tangential > 1e-10
        return np.where(active, thrust, 0), np.where(active, opposing_torque, 0)

    def step_pwm(self, pwm, dt, body_rates=None, body_velocity=None,
                 blade_efficiency=None, motor_torque_scale=None):
        cfg = self.config
        pwm = np.asarray(pwm, dtype=float)
        if pwm.shape != (4,) or not np.isfinite(pwm).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError("finite physical PWM and positive dt required")
        rates = np.zeros(3) if body_rates is None else np.asarray(body_rates, dtype=float)
        velocity = np.zeros(3) if body_velocity is None else np.asarray(body_velocity, dtype=float)
        efficiency = np.ones(4) if blade_efficiency is None else np.asarray(blade_efficiency, dtype=float)
        torque_scale = np.ones(4) if motor_torque_scale is None else np.asarray(motor_torque_scale, dtype=float)
        if rates.shape != (3,) or velocity.shape != (3,) or efficiency.shape != (4,) or torque_scale.shape != (4,):
            raise ValueError("invalid plant state/fault shape")
        if not np.isfinite(np.r_[rates, velocity, efficiency, torque_scale]).all() or np.any(efficiency < 0) or np.any(torque_scale < 0):
            raise ValueError("invalid physical plant state/fault factor")
        demand = np.clip((pwm - cfg.pwm_min) / (cfg.pwm_max - cfg.pwm_min), 0, 1) * cfg.maximum_speed_command_rad_s
        local_vertical_velocity = velocity[2] + np.cross(rates, self.positions)[:, 2]
        steps = max(1, math.ceil(dt / cfg.maximum_substep_s - 1e-10))
        h = dt / steps
        first_speed = self.speed_rad_s.copy()
        average_force, average_aero_torque, average_momentum = np.zeros(4), np.zeros(4), 0.0
        for _ in range(steps):
            force, opposing_torque = self.aerodynamic_load(self.speed_rad_s, local_vertical_velocity, efficiency)
            motor_torque = np.clip(cfg.motor_velocity_p * (demand - self.speed_rad_s),
                                   -cfg.motor_torque_limit_Nm, cfg.motor_torque_limit_Nm) * torque_scale
            # Semi-implicit bearing damping follows the 1ms articulated solver;
            # P motor torque is held over the physics interval, as in the plugin.
            next_speed = (self.speed_rad_s + h / cfg.rotor_inertia_kgm2 * (motor_torque - opposing_torque)) / (1 + h * cfg.joint_damping_Nms / cfg.rotor_inertia_kgm2)
            next_speed = np.maximum(0, next_speed)
            average_force += force / steps
            average_aero_torque += opposing_torque / steps
            average_momentum += float(cfg.rotor_inertia_kgm2 * self.spin @ self.speed_rad_s) / steps
            self.speed_rad_s = next_speed
        acceleration_reaction = -cfg.rotor_inertia_kgm2 * float(self.spin @ (self.speed_rad_s - first_speed)) / dt
        steady_yaw_reaction = -float(self.spin @ average_aero_torque)
        gyro_torque = -np.cross(rates, [0.0, 0.0, average_momentum])
        body_torque = np.sum(np.cross(self.positions, np.column_stack([np.zeros(4), np.zeros(4), average_force])), axis=0)
        body_torque += gyro_torque
        body_torque[2] += acceleration_reaction + steady_yaw_reaction
        return {"forces_N": average_force, "body_torque_Nm": body_torque,
                "body_yaw_torque_Nm": float(body_torque[2]),
                "rotor_acceleration_reaction_Nm": acceleration_reaction,
                "aerodynamic_yaw_torque_Nm": steady_yaw_reaction,
                "rotor_speed_rad_s": self.speed_rad_s.copy(), "rotor_gyro_torque_Nm": gyro_torque}

    def step_action(self, action, dt, **state):
        return self.step_pwm(self.action_to_pwm(action), dt, **state)
