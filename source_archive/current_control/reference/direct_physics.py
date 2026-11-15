"""Physical Iris rotor model and replay validation against archived Gazebo PWM.

This is a PLANT component. Its body state and fault factors must never be policy
inputs. Motor velocity P control represents the SDF actuator, not a flight PID.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import ctypes
from functools import lru_cache
import os
import subprocess
import sys


NATIVE_SOURCE = r'''
#include <math.h>
void direct_motor_step(double *speed, const double *demand, const double *rates,
 const double *velocity, const double *efficiency, const double *torque_scale,
 const double *positions, const double *p, double *out) {
 // p: dt, substep, Jr, P, cap, bearing, radius, rho*area, a0, cla, cda.
 int n=(int)ceil(p[0]/p[1]-1e-10); if(n<1)n=1;
 double h=p[0]/n, first[4], vertical[4], average_q[4]={0}, momentum=0;
 const double spin[4]={1,1,-1,-1};
 for(int i=0;i<12;i++)out[i]=0;
 for(int i=0;i<4;i++) {
   first[i]=speed[i];
   vertical[i]=velocity[2]+rates[0]*positions[3*i+1]-rates[1]*positions[3*i];
 }
 for(int k=0;k<n;k++)for(int i=0;i<4;i++) {
   double vt=p[6]*fmax(0,speed[i]), vz=vertical[i], flow=hypot(vt,vz);
   double alpha=p[8]-atan2(vz,fmax(vt,1e-12)), cl=p[9]*alpha;
   if(alpha>1.4)cl=fmax(0,p[9]*1.4-.025*(alpha-1.4));
   if(alpha< -1.4)cl=fmin(0,-p[9]*1.4-.025*(alpha+1.4));
   double cd=p[10]*fmin(fabs(alpha),1.4), scale=p[7]*flow*efficiency[i];
   double force=vt>1e-10 ? scale*(cl*vt-cd*vz) : 0;
   double q=vt>1e-10 ? p[6]*scale*(cd*vt+cl*vz) : 0;
   double u=fmin(p[4],fmax(-p[4],p[3]*(demand[i]-speed[i])))*torque_scale[i];
   double next=fmax(0,(speed[i]+h/p[2]*(u-q))/(1+h*p[5]/p[2]));
   out[i]+=force/n; average_q[i]+=q/n; momentum+=p[2]*spin[i]*speed[i]/n;
   speed[i]=next;
 }
 double reaction=0,aero=0;
 for(int i=0;i<4;i++) {
   reaction-=p[2]*spin[i]*(speed[i]-first[i])/p[0];
   aero-=spin[i]*average_q[i];
   out[4]+=positions[3*i+1]*out[i]; out[5]-=positions[3*i]*out[i];
 }
 out[9]=-rates[1]*momentum; out[10]=rates[0]*momentum;
 out[4]+=out[9]; out[5]+=out[10]; out[6]=reaction+aero;
 out[7]=reaction; out[8]=aero;
}
'''


@lru_cache(maxsize=1)
def native_core():
    """Optional local C arithmetic acceleration, without changing Python packages."""
    if not sys.platform.startswith("linux"):
        return None
    directory = Path(__file__).resolve().parent / "artifacts" / "direct_fit"
    directory.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(NATIVE_SOURCE.encode()).hexdigest()[:16]
    source = directory / f"motor_core_{signature}.c"
    library = directory / f"motor_core_{signature}.so"
    metadata_path = directory / f"motor_core_{signature}.json"
    metadata = None
    if library.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            if metadata["source_sha256"] != hashlib.sha256(NATIVE_SOURCE.encode()).hexdigest() or metadata["binary_sha256"] != hashlib.sha256(library.read_bytes()).hexdigest():
                metadata = None
        except (OSError, ValueError, KeyError):
            metadata = None
    if metadata is None:
        source.write_text(NATIVE_SOURCE)
        temporary = directory / f"motor_core_{signature}.{os.getpid()}.so"
        try:
            command = ["cc", "-O3", "-shared", "-fPIC", "-std=c99", str(source), "-lm", "-o", str(temporary)]
            subprocess.run(command,
                           check=True, capture_output=True, timeout=30)
            temporary.replace(library)
            metadata = {"source_sha256": hashlib.sha256(NATIVE_SOURCE.encode()).hexdigest(),
                        "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                        "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
                        "compile_command": command, "source": str(source), "binary": str(library)}
            temporary_metadata = directory / f"motor_core_{signature}.{os.getpid()}.json"
            temporary_metadata.write_text(json.dumps(metadata, indent=2))
            temporary_metadata.replace(metadata_path)
        except (OSError, subprocess.SubprocessError):
            return None
    handle = ctypes.CDLL(str(library))
    function = handle.direct_motor_step
    function.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")] * 9
    function.restype = None
    function.library_path = str(library)
    function.provenance = metadata
    return function


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
    body_inertia_diagonal_kgm2: tuple = (.012832785142857143, .01711278514285714, .023800416)
    # Actual fixed nominal calibration1104; no condition-specific adaptation.
    nominal_hover_pwm: tuple | None = (1533.216, 1533.215, 1533.214, 1533.217)

    @property
    def thrust_coefficient(self):
        return self.air_density_kg_m3 * self.blade_area_m2 * self.blade_radius_m**2 * self.lift_slope * self.alpha_zero_rad

    @property
    def yaw_drag_coefficient(self):
        return self.air_density_kg_m3 * self.blade_area_m2 * self.blade_radius_m**3 * self.drag_slope * self.alpha_zero_rad


class DirectMotorModel:
    """Four SDF motors FR, RL, FL, RR, with rotor momentum and blade inflow.

    ``step_action`` accepts the same four normalized force requests as V2 direct.
    ``step_pwm`` accepts recorded physical PWM for independent validation.
    Returned forces already include vertical/local rotor inflow drag. Do not add
    the previous 1.55 N/(m/s) vertical drag again when integrating this model.
    """
    def __init__(self, config=None, use_compiled=True):
        self.config = config or DirectMotorConfig()
        self.spin = np.asarray(self.config.rotor_spin, dtype=float)
        self.positions = np.asarray(self.config.rotor_positions_m, dtype=float)
        self._native = native_core() if use_compiled else None
        self.backend = "source_equivalent_C" if self._native else "numpy"
        self.reset()

    def runtime_provenance(self):
        return {"backend": self.backend, "config": asdict(self.config),
                "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "native": self._native.provenance if self._native is not None else None}

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
        if self._native is not None and np.array_equal(self.spin, [1, 1, -1, -1]):
            output = np.empty(12, dtype=float)
            parameters = np.array([dt, cfg.maximum_substep_s, cfg.rotor_inertia_kgm2,
                                   cfg.motor_velocity_p, cfg.motor_torque_limit_Nm, cfg.joint_damping_Nms,
                                   cfg.blade_radius_m, cfg.air_density_kg_m3 * cfg.blade_area_m2,
                                   cfg.alpha_zero_rad, cfg.lift_slope, cfg.drag_slope])
            self._native(self.speed_rad_s, np.ascontiguousarray(demand), np.ascontiguousarray(rates),
                         np.ascontiguousarray(velocity), np.ascontiguousarray(efficiency),
                         np.ascontiguousarray(torque_scale), self.positions.reshape(-1), parameters, output)
            return {"forces_N": output[:4], "body_torque_Nm": output[4:7],
                    "body_yaw_torque_Nm": float(output[6]),
                    "rotor_acceleration_reaction_Nm": float(output[7]),
                    "aerodynamic_yaw_torque_Nm": float(output[8]),
                    "rotor_speed_rad_s": self.speed_rad_s.copy(), "rotor_gyro_torque_Nm": output[9:12]}
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


def load_pulses(directory):
    from scipy.spatial.transform import Rotation
    directory = Path(directory)
    def read(name):
        return [json.loads(line) for line in (directory / name).read_text().splitlines() if line.strip()]
    actuators, physics = read("actuators.jsonl"), read("physics.jsonl")
    assert len(physics) == len(actuators) + 1, "complete command/next-state exchange required"
    frame = np.asarray([row["frame"] for row in actuators])
    assert np.array_equal(frame, np.arange(len(actuators))), "noncontiguous actuator frame"
    time = np.asarray([row["timestamp"] for row in physics])
    assert np.all(np.diff(time) > 0), "nonmonotonic physics time"
    assert np.allclose(time[:-1], frame * .001 + time[0], atol=1e-10), "frame/time association failed"
    pwm = np.asarray([row["output_pwm"] for row in actuators], dtype=float)
    rates = np.asarray([row["imu"]["gyro"] for row in physics]) * [1, -1, -1]
    velocity_ned = np.asarray([row["velocity"] for row in physics])
    specific_force = np.asarray([row["imu"]["accel_body"] for row in physics]) * [1, -1, -1]
    quaternions = np.asarray([row["quaternion"] for row in physics])
    rot_ned_frd = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    velocity_flu = np.einsum("nji,nj->ni", rot_ned_frd, velocity_ned) * [1, -1, -1]
    override = np.asarray([row["override"] for row in actuators])
    starts = np.flatnonzero(override & ~np.r_[False, override[:-1]])
    windows = []
    for pulse_index, start in enumerate(starts):
        end = start
        while end + 1 < len(override) and override[end + 1]:
            end += 1
        begin, finish = np.searchsorted(time, [time[start] - .10, time[end] + .20])
        windows.append({"pulse_index": pulse_index, "time": time[begin:finish] - time[start],
                        "pwm": pwm[begin:finish], "rates": rates[begin:finish],
                        "body_velocity": velocity_flu[begin:finish],
                        "specific_force_flu": specific_force[begin:finish],
                        "on_index": int(start - begin), "off_index": int(end - begin),
                        "pulse_on_s": float(time[start]), "pulse_off_s": float(time[end]),
                        "shift_pwm": (np.asarray(actuators[start]["output_pwm"]) - actuators[start]["input_pwm"]).tolist()})
    return windows, {"records": len(physics), "unique_frames": len(set(frame)),
                     "min_dt_s": float(np.min(np.diff(time))), "max_dt_s": float(np.max(np.diff(time))),
                     "source_hashes": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ("actuators.jsonl", "physics.jsonl")}}


def replay_window(window, config=None, command_shift_steps=0):
    cfg = config or DirectMotorConfig()
    model = DirectMotorModel(cfg)
    pwm, measured, ts, velocity = (window[name] for name in ("pwm", "rates", "time", "body_velocity"))
    model.reset(pwm=pwm[0])
    predicted = np.empty_like(measured)
    predicted[0] = measured[0]
    inertia = np.asarray(cfg.body_inertia_diagonal_kgm2)
    forces, yaw_terms = [], []
    for index in range(1, len(ts)):
        action_index = np.clip(index - 1 + command_shift_steps, 0, len(ts) - 1)
        dt = ts[index] - ts[index - 1]
        result = model.step_pwm(pwm[action_index], dt, body_rates=measured[index - 1], body_velocity=velocity[index - 1])
        gyro = np.cross(measured[index - 1], inertia * measured[index - 1])
        predicted[index] = predicted[index - 1] + dt * (result["body_torque_Nm"] - gyro) / inertia
        forces.append(result["forces_N"])
        yaw_terms.append([result["rotor_acceleration_reaction_Nm"], result["aerodynamic_yaw_torque_Nm"]])
    return predicted, np.asarray(forces), np.asarray(yaw_terms)


def self_check():
    import time
    python_model, native_model = DirectMotorModel(use_compiled=False), DirectMotorModel(use_compiled=True)
    rng = np.random.default_rng(1105)
    max_difference = 0.0
    for _ in range(500):
        action = rng.uniform(-.8, .8, 4)
        dt = float(rng.choice([.001, .005, .02]))
        state = {"body_rates": rng.uniform(-.8, .8, 3), "body_velocity": rng.uniform(-1, 1, 3),
                 "blade_efficiency": rng.uniform(.6, 1, 4), "motor_torque_scale": rng.uniform(.8, 1, 4)}
        expected = python_model.step_action(action, dt, **state)
        actual = native_model.step_action(action, dt, **state)
        for key in expected:
            max_difference = max(max_difference, float(np.max(np.abs(np.asarray(actual[key]) - expected[key]))))
    timings = {}
    for name, model in (("numpy", python_model), ("native", native_model)):
        started = time.perf_counter()
        for _ in range(2000):
            model.step_action(np.zeros(4), .005)
        timings[name + "_us_per_5ms_step"] = (time.perf_counter() - started) / 2000 * 1e6
    cfg = DirectMotorConfig()
    equilibrium = DirectMotorModel()
    equilibrium.reset(pwm=cfg.nominal_hover_pwm)
    before = equilibrium.speed_rad_s.copy()
    result = equilibrium.step_pwm(cfg.nominal_hover_pwm, .02)
    reset_error = float(np.max(np.abs(equilibrium.speed_rad_s - before)))
    bounds = DirectMotorModel()
    for _ in range(1000):
        bounded = bounds.step_action(np.ones(4), .005)
    maximum_motor_torque = cfg.joint_damping_Nms * bounds.speed_rad_s + cfg.yaw_drag_coefficient * bounds.speed_rad_s**2
    checks = {"native_matches_numpy": max_difference < 1e-9, "static_equilibrium": reset_error < 1e-9,
              "torque_limited_steady_state": bool(np.max(maximum_motor_torque) <= cfg.motor_torque_limit_Nm + 1e-9),
              "rotor_body_reaction_signs": bool(DirectMotorModel().step_pwm([1553, 1553, 1513, 1513], .001)["body_yaw_torque_Nm"] < 0)}
    return {"pass": all(checks.values()), "checks": checks, "native_backend": native_model.backend,
            "runtime_provenance": native_model.runtime_provenance(),
            "maximum_backend_difference": max_difference, "equilibrium_speed_change": reset_error,
            "maximum_thrust_N_per_rotor": bounded["forces_N"].tolist(), **timings}


def fit_cli(direct_root, output):
    from dataclasses import replace
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = DirectMotorConfig()
    report = {"config": asdict(cfg), "self_check": self_check(), "thrust_coefficient_N_per_rad2s2": cfg.thrust_coefficient,
              "yaw_drag_coefficient_Nm_per_rad2s2": cfg.yaw_drag_coefficient,
              "steady_yaw_drag_per_thrust_m": cfg.yaw_drag_coefficient / cfg.thrust_coefficient,
              "linear_motor_tau_s": cfg.rotor_inertia_kgm2 / (cfg.motor_velocity_p + cfg.joint_damping_Nms), "runs": {}}
    report["source_model_not_test_fitted"] = True
    report["validation_scope"] = "Conditional actuator/component replay: recorded measured body rates and body velocity drive aerodynamic and gyroscopic loads every1ms, predicted angular velocity is integrated. Not a freely evolving coupled vehicle rollout or learned-policy flight."
    report["source_hashes"] = {"direct_physics.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "native_C_source": hashlib.sha256(NATIVE_SOURCE.encode()).hexdigest()}
    for model_name in ("iris_with_ardupilot", "iris_with_standoffs"):
        source_path = Path(direct_root) / "yaw_pulse_1102" / "gazebo_assets" / model_name / "model.sdf"
        report["source_hashes"][model_name + "/model.sdf"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    calibration_path = Path(direct_root) / "calibration_nominal_1104" / "calibration.json"
    if calibration_path.exists():
        report["source_hashes"]["calibration_nominal_1104/calibration.json"] = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    steady_x, steady_y = [], []
    for name in ("motor_pulse_1101", "yaw_pulse_1102"):
        windows, provenance = load_pulses(Path(direct_root) / name)
        shift_scores = []
        for shift in (-3, -2, -1, 0, 1, 2, 3):
            residuals = []
            for window in windows:
                predicted, _, _ = replay_window(window, command_shift_steps=shift)
                mask = (window["time"] >= -.02) & (window["time"] < .15)
                residuals.append(predicted[mask] - window["rates"][mask])
            error = np.concatenate(residuals)
            shift_scores.append({"command_shift_steps": shift, "rate_rmse_rad_s": np.sqrt(np.mean(error**2, axis=0)).tolist()})
        best = min(shift_scores, key=lambda row: row["rate_rmse_rad_s"][2])
        entries = []
        figure, axes = plt.subplots(len(windows), 3, figsize=(12, 8), sharex=True)
        for window in windows:
            predicted, force, yaw = replay_window(window, command_shift_steps=best["command_shift_steps"])
            mask = (window["time"] >= .005) & (window["time"] < .020)
            before = (window["time"] >= -.020) & (window["time"] < 0)
            plateau = (window["time"] >= .006) & (window["time"] < .019)
            slope = np.polyfit(window["time"][plateau], window["rates"][plateau, 2], 1)[0]
            model_slope = np.polyfit(window["time"][plateau], predicted[plateau, 2], 1)[0]
            measured_thrust = cfg.nominal_mass_kg * window["specific_force_flu"][1:, 2]
            thrust_rmse = np.sqrt(np.mean((force.sum(axis=1) - measured_thrust)**2))
            # Estimate steady aerodynamic scale only on unchanged PWM intervals;
            # remove modelled rotor momentum before comparing small steady torque.
            dt = np.diff(window["time"])
            rates = window["rates"]
            inertia = np.asarray(cfg.body_inertia_diagonal_kgm2)
            measured_torque = inertia[2] * np.diff(rates[:, 2]) / dt + np.cross(rates[:-1], rates[:-1] * inertia)[:, 2]
            constant_command = np.zeros(len(dt), dtype=bool)
            for index in range(8, len(dt)-2):
                # Sensor publication/packet association can differ by one 1ms
                # interval. Guard both sides of every physical PWM transition.
                constant_command[index] = np.all(window["pwm"][index-8:index+3] == window["pwm"][index])
            settled = constant_command & (np.abs(yaw[:, 0]) < .001) & (np.abs(yaw[:, 1]) > .0001)
            settled &= (window["time"][1:] > 0) & (window["time"][1:] < .1)
            x, y = yaw[settled, 1], (measured_torque - yaw[:, 0])[settled]
            steady_x.extend(x.tolist()); steady_y.extend(y.tolist())
            scale = float(x @ y / (x @ x)) if len(x) and x @ x > 1e-12 else None
            entries.append({"pulse_index": window["pulse_index"], "shift_pwm": window["shift_pwm"],
                            "actual_rate_change_5_20ms": (window["rates"][mask].mean(axis=0) - window["rates"][before].mean(axis=0)).tolist(),
                            "predicted_rate_change_5_20ms": (predicted[mask].mean(axis=0) - predicted[before].mean(axis=0)).tolist(),
                            "rate_rmse_rad_s": np.sqrt(np.mean((predicted - window["rates"])**2, axis=0)).tolist(),
                            "plateau_actual_yaw_acceleration_rad_s2": float(slope),
                            "plateau_predicted_yaw_acceleration_rad_s2": float(model_slope),
                            "specific_force_total_thrust_rmse_N": float(thrust_rmse),
                            "settled_samples_for_drag_check": int(settled.sum()),
                            "empirical_aerodynamic_torque_scale": scale})
            for axis in range(3):
                ax = axes[window["pulse_index"], axis]
                ax.plot(window["time"], rates[:, axis], label="Gazebo physical gyro", lw=1.3)
                ax.plot(window["time"], predicted[:, axis], "--", label="SDF source motor model", lw=1.1)
                ax.set_xlim(-.025, .16); ax.grid(alpha=.2)
                ax.set_ylabel(["roll", "pitch", "yaw"][axis] + " rad/s")
                ax.set_title(f"Pulse{window['pulse_index']}: {window['shift_pwm']}", fontsize=9)
                if window["pulse_index"] == len(windows)-1: ax.set_xlabel("Simulation seconds from actual pulse")
            np.savez_compressed(output / f"validated_{name}_pulse{window['pulse_index']}.npz",
                                time=window["time"], physical_pwm=window["pwm"], measured_rates=window["rates"],
                                predicted_rates=predicted, force=force, yaw_terms=yaw, measured_total_thrust_N=measured_thrust)
        report["runs"][name] = {"provenance": provenance, "alignment_scores": shift_scores, "best_alignment": best, "pulses": entries}
        axes[0, 0].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output / f"direct_replay_{name}.png", dpi=150)
        figure.savefig(output / f"direct_replay_{name}.pdf")
        plt.close(figure)
    x, y = np.asarray(steady_x), np.asarray(steady_y)
    scale = float(x @ y / (x @ x))
    report["steady_aerodynamic_check"] = {"settled_samples": len(x), "least_squares_scale_to_source": scale,
        "source_yaw_drag_per_thrust_m": cfg.yaw_drag_coefficient / cfg.thrust_coefficient,
        "empirical_equivalent_yaw_drag_per_thrust_m": scale * cfg.yaw_drag_coefficient / cfg.thrust_coefficient,
        "torque_residual_rmse_Nm": float(np.sqrt(np.mean((y-scale*x)**2))),
        "interpretation": "Small steady torque after subtracting rotor acceleration; short development windows and blade phase limit precision. Source coefficient is retained, not replaced by this noisy diagnostic fit."}
    all_pulses = [pulse for run in report["runs"].values() for pulse in run["pulses"]]
    gates = {"self_checks": report["self_check"]["pass"],
             "causal_zero_extra_delay": all(run["best_alignment"]["command_shift_steps"] == 0 for run in report["runs"].values()),
             "balanced_initial_yaw_jump": all(abs(pulse["actual_rate_change_5_20ms"][2]-pulse["predicted_rate_change_5_20ms"][2]) < .001 for pulse in report["runs"]["yaw_pulse_1102"]["pulses"]),
             "single_motor_yaw_replay": all(pulse["rate_rmse_rad_s"][2] < .001 for pulse in report["runs"]["motor_pulse_1101"]["pulses"]),
             "thrust_replay": all(pulse["specific_force_total_thrust_rmse_N"] < .005 for pulse in all_pulses)}
    report["validation_gates"] = gates
    report["pass"] = all(gates.values())
    (output / "direct_motor_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (output / "validated_source_model_replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "validation_gates": gates,
                      "steady_aerodynamic_check": report["steady_aerodynamic_check"],
                      "self_check": report["self_check"], "report": str(output / "validated_source_model_replay.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit_cli(args.direct_root, args.output)
