"""Rotor dynamics of the Iris airframe with the parameters of the Gazebo Iris model.

PWM -> motor speed command -> P-controlled motor with rotor inertia and bearing damping ->
blade lift and drag with inflow (Gazebo LiftDrag) -> rotor forces and body torques
(rotor arms, aerodynamic yaw torque, rotor acceleration reaction, gyroscopic torque).
"""
from dataclasses import dataclass
import math

import numpy as np

MASS = 1.75          # kg, Iris
G = 9.80665
# Body inertia of the Iris model (ardupilot_gazebo): parallel-axis sum about the centre of mass,
# rotor x/y inertia averaged over the blade azimuth.
INERTIA = np.array([.012832785142857143, .01711278514285714, .023800416], dtype=float)
# Scale of the cascade copy's per-motor force request (2x hover force); not the rotor limit.
MAX_FORCE = 2 * MASS * G / 4


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
    nominal_mass_kg: float = MASS
    gravity_mps2: float = G
    maximum_substep_s: float = 0.001
    rotor_spin: tuple = (1.0, 1.0, -1.0, -1.0)            # rotor speed signs; body reaction is opposite
    rotor_positions_m: tuple = ((.13, -.22, .023), (-.13, .20, .023), (.13, .22, .023), (-.13, -.20, .023))
    body_inertia_diagonal_kgm2: tuple = tuple(INERTIA)
    nominal_hover_pwm: tuple = (1533.216, 1533.215, 1533.214, 1533.217)   # action 0 of the end-to-end policy

    @property
    def thrust_coefficient(self):
        return self.air_density_kg_m3 * self.blade_area_m2 * self.blade_radius_m**2 * self.lift_slope * self.alpha_zero_rad

    @property
    def yaw_drag_coefficient(self):
        return self.air_density_kg_m3 * self.blade_area_m2 * self.blade_radius_m**3 * self.drag_slope * self.alpha_zero_rad


class DirectMotorModel:
    """Four motors in Gazebo order (front-right, rear-left, front-left, rear-right)."""

    def __init__(self, config=None):
        self.config = config or DirectMotorConfig()
        self.spin = np.asarray(self.config.rotor_spin, dtype=float)
        self.positions = np.asarray(self.config.rotor_positions_m, dtype=float)
        self.reset()

    def nominal_pwm_from_forces(self, forces):
        """Static PWM that holds the given rotor forces (the steady P error balances the resistance)."""
        cfg = self.config
        forces = np.maximum(np.asarray(forces, dtype=float), 0.0)
        speed = np.sqrt(forces / cfg.thrust_coefficient)
        setpoint = speed + (cfg.joint_damping_Nms * speed + cfg.yaw_drag_coefficient * speed**2) / cfg.motor_velocity_p
        return cfg.pwm_min + (cfg.pwm_max - cfg.pwm_min) * setpoint / cfg.maximum_speed_command_rad_s

    def action_to_pwm(self, action):
        """End-to-end action in [-1, 1] -> PWM: 0 = hover, the square root makes the force ~linear in the action."""
        cfg = self.config
        hover = np.asarray(cfg.nominal_hover_pwm)
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        pwm = np.clip(cfg.pwm_min + (hover - cfg.pwm_min) * np.sqrt(action + 1), cfg.pwm_min, cfg.pwm_max)
        return np.rint(pwm / cfg.pwm_quantization) * cfg.pwm_quantization

    def steady_speed(self, pwm):
        cfg = self.config
        demand = np.clip((np.asarray(pwm, dtype=float) - cfg.pwm_min) / (cfg.pwm_max - cfg.pwm_min), 0, 1) \
            * cfg.maximum_speed_command_rad_s
        b = cfg.motor_velocity_p + cfg.joint_damping_Nms
        # positive root of kQ*w^2 + (P + bearing)*w - P*command = 0
        speed = 2 * cfg.motor_velocity_p * demand / (b + np.sqrt(b * b + 4 * cfg.yaw_drag_coefficient * cfg.motor_velocity_p * demand))
        # the motor torque limit also bounds the steady speed
        cap = 2 * cfg.motor_torque_limit_Nm / (cfg.joint_damping_Nms + np.sqrt(
            cfg.joint_damping_Nms**2 + 4 * cfg.yaw_drag_coefficient * cfg.motor_torque_limit_Nm))
        return np.minimum(speed, cap)

    def reset(self, pwm=None):
        """Rotors at the steady speed of `pwm` (default: hover force of the nominal mass)."""
        cfg = self.config
        if pwm is not None:
            self.speed_rad_s = self.steady_speed(pwm)
        else:
            self.speed_rad_s = np.sqrt(np.full(4, cfg.nominal_mass_kg * cfg.gravity_mps2 / 4) / cfg.thrust_coefficient)

    def aerodynamic_load(self, speed, vertical_velocity, blade_efficiency):
        """Rotor thrust and opposing (drag) torque with axial inflow."""
        cfg = self.config
        tangential = cfg.blade_radius_m * np.maximum(speed, 0)
        inflow = np.asarray(vertical_velocity, dtype=float)
        flow_speed = np.hypot(tangential, inflow)
        alpha = cfg.alpha_zero_rad - np.arctan2(inflow, np.maximum(tangential, 1e-12))
        # linear lift up to the stall angle 1.4 rad, then the stall slope (Gazebo LiftDrag)
        cl = np.where(alpha > 1.4, np.maximum(0, cfg.lift_slope * 1.4 - .025 * (alpha - 1.4)),
                      np.where(alpha < -1.4, np.minimum(0, -cfg.lift_slope * 1.4 - .025 * (alpha + 1.4)),
                               cfg.lift_slope * alpha))
        cd = cfg.drag_slope * np.minimum(np.abs(alpha), 1.4)
        scale = cfg.air_density_kg_m3 * cfg.blade_area_m2 * flow_speed * blade_efficiency
        thrust = scale * (cl * tangential - cd * inflow)
        opposing_torque = cfg.blade_radius_m * scale * (cd * tangential + cl * inflow)
        active = tangential > 1e-10
        return np.where(active, thrust, 0), np.where(active, opposing_torque, 0)

    def step_pwm(self, pwm, dt, body_rates=None, body_velocity=None, blade_efficiency=None, motor_torque_scale=None):
        """Integrate the rotors over dt (1 ms substeps); returns averaged forces_N and body_torque_Nm."""
        cfg = self.config
        pwm = np.asarray(pwm, dtype=float)
        rates = np.zeros(3) if body_rates is None else np.asarray(body_rates, dtype=float)
        velocity = np.zeros(3) if body_velocity is None else np.asarray(body_velocity, dtype=float)
        efficiency = np.ones(4) if blade_efficiency is None else np.asarray(blade_efficiency, dtype=float)
        torque_scale = np.ones(4) if motor_torque_scale is None else np.asarray(motor_torque_scale, dtype=float)
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
            # semi-implicit bearing damping, motor torque held over the substep (as the Gazebo plugin)
            next_speed = (self.speed_rad_s + h / cfg.rotor_inertia_kgm2 * (motor_torque - opposing_torque)) \
                / (1 + h * cfg.joint_damping_Nms / cfg.rotor_inertia_kgm2)
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
        return {'forces_N': average_force, 'body_torque_Nm': body_torque, 'rotor_speed_rad_s': self.speed_rad_s.copy()}
