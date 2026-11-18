"""Controller-independent communication with ArduPilot and Gazebo.

Positions and velocities use ENU; body rates use FLU, radians per second.
This module exchanges messages. It does not select actions, define a mission,
load weights, or select a learning algorithm. An optional servo gate controls
physical intervals through the native lockstep transport. Timeouts are wall
time; ``State.time_s`` is the timestamp supplied by the simulated autopilot.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import threading
import time

import numpy as np
from pymavlink import mavutil


ENU_FROM_NED = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
FRD_FROM_FLU = np.diag([1., -1., -1.])
SERVO_PACKET = struct.Struct('<HHI16H')
SERVO_MAGIC = 18458


class StateUnavailable(RuntimeError):
    """A fresh, coherent pair of position and attitude is not available yet."""


@dataclass(frozen=True)
class State:
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    body_rates: np.ndarray
    time_s: float
    position_time_s: float
    attitude_time_s: float
    estimator_type: int | None = None


def _vector(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {length} finite numbers')
    return result


def _rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


class _MotorRelay:
    """Pass native FDM packets through, with an optional four-channel override.

Expired overrides are released and reported while native traffic continues.
This preserves the transport needed to request native landing. A fatal socket
error remains a failed simulation; it never counts as a successful landing.
"""
    def __init__(self, relay_port, physics_port, expiry_s=.5):
        self.address = ('127.0.0.1', int(relay_port))
        self.physics_address = ('127.0.0.1', int(physics_port))
        self.expiry_s = float(expiry_s)
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.override = None
        self.delta = None
        self.error = None
        self.autopilot_address = None
        self.stop = threading.Event()
        self.socket = None
        self.thread = None
        self.latest_fdm = None
        self.last_frame = None
        self.sensor_count = 0
        self.servo_count = 0
        self.duplicates = dict(sensor=0, servo=0)
        self.gated = False
        self.until = math.inf
        self.pending = None
        self.physics_dt = None
        self.history = deque(maxlen=2048)

    def start(self):
        if self.thread is not None:
            raise RuntimeError('Motor relay was already started')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(self.address)
            self.socket.settimeout(.1)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except BaseException:
            self.socket.close()
            raise

    def set_pwm(self, pwm):
        values = _vector(pwm, 4, 'motor PWM')
        if np.any(values < 1100) or np.any(values > 1900):
            raise ValueError('Motor PWM must lie in [1100, 1900] microseconds')
        with self.lock:
            if self.error:
                raise RuntimeError('Motor relay has failed: ' + self.error)
            self.override = (tuple(np.rint(values).astype(int)), time.monotonic())
            self.delta = None

    def set_delta(self, delta_pwm):
        values = _vector(delta_pwm, 4, 'motor PWM delta')
        if np.max(np.abs(values)) > 800:
            raise ValueError('PWM delta exceeds the physical PWM span')
        with self.cv:
            if self.error:
                raise RuntimeError('Motor relay has failed: ' + self.error)
            self.override = None
            self.delta = (tuple(values), time.monotonic())

    def release(self):
        with self.lock:
            self.override = None
            self.delta = None

    @property
    def held(self):
        with self.lock:
            return self.pending is not None

    def _snapshot(self):
        return dict(fdm=copy.deepcopy(self.latest_fdm),
                    frame=None if self.pending is None else self.pending[0][2],
                    native_pwm=None if self.pending is None else list(self.pending[0][3:7]),
                    sensor_count=self.sensor_count, servo_count=self.servo_count,
                    duplicates=dict(self.duplicates), history=list(self.history))

    def _wait_held(self, timeout):
        deadline = time.monotonic() + timeout
        with self.cv:
            while True:
                if self.error:
                    raise RuntimeError('Motor relay failed: ' + self.error)
                if self.stop.is_set():
                    raise RuntimeError('Motor relay is closed')
                if self.pending is not None:
                    return self._snapshot()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('No synchronous servo boundary')
                self.cv.wait(min(remaining, .1))

    def pause(self, timeout=10., physics_dt=.001):
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError('Positive boundary timeout and physics step required')
        with self.cv:
            self.physics_dt = float(physics_dt)
            self.gated = True
            self.until = -math.inf
        return self._wait_held(timeout)

    def advance(self, seconds=.02, timeout=10.):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Positive finite control interval and timeout required')
        with self.cv:
            if self.error or self.stop.is_set():
                raise RuntimeError('Motor relay is unavailable: ' + str(self.error))
            if not self.gated or self.pending is None or self.latest_fdm is None:
                raise RuntimeError('Advance requires a held servo boundary')
            if abs(seconds / self.physics_dt - round(seconds / self.physics_dt)) > 1e-7:
                raise ValueError('Control interval must contain whole physical steps')
            start = float(self.latest_fdm['timestamp'])
            before = self.sensor_count
            self.until = start + seconds
            packet, raw = self.pending
            self.pending = None
            self._send_servo(packet, raw)
        result = self._wait_held(timeout)
        elapsed = float(result['fdm']['timestamp']) - start
        if abs(elapsed - seconds) > 1e-8 or result['sensor_count'] - before != round(seconds / self.physics_dt):
            raise RuntimeError('Physical control interval differs from requested step')
        return result

    def resume(self):
        with self.cv:
            self.gated = False
            self.until = math.inf
            self.override = None
            self.delta = None
            if self.pending is not None:
                packet, raw = self.pending
                self.pending = None
                self._send_servo(packet, raw)
            self.cv.notify_all()

    def _send_servo(self, packet, raw):
        # The caller owns self.lock. A gated action is valid for its simulated
        # interval, including arbitrarily long PPO computation between steps.
        mode = 'native'
        native = list(packet[3:7])
        current = self.override or self.delta
        if current and not self.gated and time.monotonic() - current[1] > self.expiry_s:
            self.override = self.delta = None
            self.error = 'Motor command expired; native outputs restored'
        if self.override:
            packet = packet.copy()
            packet[3:7] = self.override[0]
            mode = 'override'
        elif self.delta:
            packet = packet.copy()
            # Preserve disarmed/stopped channels. In particular a zero delta
            # forwards the original bytes, including the native spool state.
            values = np.asarray(native, dtype=float)
            corrected = np.rint(np.clip(values + self.delta[0], 1100, 1900)).astype(int)
            packet[3:7] = np.where(values > 1100, corrected, values).astype(int).tolist()
            mode = 'delta'
        applied = list(packet[3:7])
        data = raw if native == applied else SERVO_PACKET.pack(*packet)
        self.socket.sendto(data, self.physics_address)
        self.servo_count += 1
        self.history.append(dict(frame=packet[2], native_pwm=native, applied_pwm=applied,
            after_sensor_t=None if self.latest_fdm is None else float(self.latest_fdm['timestamp']), mode=mode))

    def _run(self):
        try:
            while not self.stop.is_set():
                try:
                    data, source = self.socket.recvfrom(65535)
                except socket.timeout:
                    continue
                if source == self.physics_address:
                    if not data.lstrip().startswith(b'{'):
                        raise ValueError('Expected JSON physics data')
                    value = json.loads(data)
                    stamp = float(value['timestamp'])
                    if not math.isfinite(stamp):
                        raise ValueError('Nonfinite physics time')
                    with self.cv:
                        if self.latest_fdm is not None:
                            previous = float(self.latest_fdm['timestamp'])
                            if stamp < previous - 1e-9:
                                raise RuntimeError('Physics time moved backwards; restart the session')
                            if stamp <= previous + 1e-9:
                                self.duplicates['sensor'] += 1
                                continue
                            if self.gated and abs(stamp - previous - self.physics_dt) > 1e-8:
                                raise RuntimeError('Physics step changed or sensor packet was lost')
                        self.latest_fdm = value
                        self.sensor_count += 1
                        if self.autopilot_address is not None:
                            self.socket.sendto(data, self.autopilot_address)
                    continue
                if source[0] != '127.0.0.1' or len(data) != SERVO_PACKET.size:
                    raise ValueError('Unexpected motor packet sender or size')
                packet = list(SERVO_PACKET.unpack(data))
                if packet[0] != SERVO_MAGIC:
                    raise ValueError('Unsupported motor packet format')
                if self.autopilot_address is not None and source != self.autopilot_address:
                    raise ValueError('Motor packet source changed during the session')
                self.autopilot_address = source
                with self.cv:
                    if self.last_frame is not None:
                        if packet[2] == self.last_frame:
                            self.duplicates['servo'] += 1
                            if self.latest_fdm is None:
                                self.socket.sendto(data, self.physics_address)
                            continue
                        if packet[2] != ((self.last_frame + 1) & 0xffffffff):
                            raise RuntimeError('Nonconsecutive native servo frames')
                    self.last_frame = packet[2]
                    if self.gated and self.latest_fdm is not None and float(self.latest_fdm['timestamp']) >= self.until - 1e-9:
                        if self.pending is not None:
                            raise RuntimeError('More than one pending servo frame')
                        self.pending = (packet, data)
                        self.cv.notify_all()
                    else:
                        self._send_servo(packet, data)
        except BaseException as exc:
            if not self.stop.is_set():
                with self.cv:
                    self.error = repr(exc)
                    self.cv.notify_all()
        finally:
            self.socket.close()

    def close(self):
        self.release()
        self.stop.set()
        with self.cv:
            self.pending = None
            self.cv.notify_all()
        if self.thread is not None:
            self.thread.join(2.)
            if self.thread.is_alive():
                raise RuntimeError('Motor relay did not stop')
        elif self.socket is not None:
            self.socket.close()


class Vehicle:
    """One simulated vehicle, addressed by its MAVLink system ID.

``state()`` consumes a new coherent observation by default. Callers waiting for
the next sample catch ``StateUnavailable``; readiness and command helpers may
read the current snapshot with ``require_new=False``. Reading cannot advance
Gazebo time and is not a lockstep acknowledgement.
"""
    def __init__(self, vehicle_id, tcp_port, relay_port, physics_port, output=None,
                 *, system_id=None, maximum_pair_skew_s=.01):
        self.vehicle_id = str(vehicle_id)
        suffix = re.search(r'(\d+)$', self.vehicle_id)
        if system_id is None and suffix is None:
            raise ValueError('Supply system_id for a vehicle name without a numeric suffix')
        self.system_id = int(system_id if system_id is not None else suffix.group(1))
        if not 1 <= self.system_id <= 255:
            raise ValueError('MAVLink system_id must be in [1, 255]')
        for port in (tcp_port, relay_port, physics_port):
            if not 1 <= int(port) <= 65535:
                raise ValueError('Port must be in [1, 65535]')
        if len({int(tcp_port), int(relay_port), int(physics_port)}) != 3:
            raise ValueError('Vehicle ports must be distinct')
        if not math.isfinite(maximum_pair_skew_s) or not 0 <= maximum_pair_skew_s <= .02:
            raise ValueError('Position/attitude skew must be between 0 and 20ms')
        self.tcp_port = int(tcp_port)
        self.maximum_pair_skew_s = float(maximum_pair_skew_s)
        self.relay = _MotorRelay(relay_port, physics_port)
        self.connection = None
        self.latest = {}
        self.received = {}
        self.readback = {}
        self._parameters = {}
        self._message_count = 0
        self._last_heartbeat_sent = 0.
        self._latest_time_ms = 0
        self._last_pair = None
        self._history = {name: deque(maxlen=128) for name in ('LOCAL_POSITION_NED', 'ATTITUDE')}
        self._events = None
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            self._events = (output/'interface.jsonl').open('x', buffering=1, encoding='utf-8')

    def _event(self, kind, **fields):
        if self._events:
            self._events.write(json.dumps(dict(event=kind, vehicle=self.vehicle_id,
                wall_time=time.time(), **fields), allow_nan=False) + '\n')

    def start_relay(self):
        self.relay.start()

    def _require_connection(self):
        if self.connection is None:
            raise RuntimeError('Vehicle is not connected')
        return self.connection

    def _check_relay(self):
        if self.relay.error:
            raise RuntimeError('Motor relay failure: ' + self.relay.error)

    def connect(self, timeout=60):
        if self.connection is not None:
            raise RuntimeError('Vehicle was already connected')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Connection timeout must be positive wall-clock seconds')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_relay()
            try:
                self.connection = mavutil.mavlink_connection(
                    f'tcp:127.0.0.1:{self.tcp_port}', source_system=255, source_component=190)
                break
            except OSError:
                time.sleep(.1)
        if self.connection is None:
            raise TimeoutError(f'{self.vehicle_id}: no autopilot TCP connection')
        self._wait(lambda: 'HEARTBEAT' in self.latest, deadline, 'autopilot heartbeat')
        self.connection.target_system = self.system_id
        self.connection.target_component = 1
        for message_id in (30, 32):
            self._command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, message_id, 20000.)
        for name in ('MAV_SYSID', 'AHRS_EKF_TYPE', 'GUID_OPTIONS', 'ATC_ANGLE_BOOST'):
            self.readback[name] = self.get_parameter(name)
        if self.readback['MAV_SYSID'] != self.system_id:
            raise RuntimeError('Autopilot MAV_SYSID differs from the selected vehicle')
        self._wait(lambda: all(self._history.values()), deadline, 'position and attitude')
        self._event('connected', parameters=self.readback)
        return self

    def drain(self):
        conn = self._require_connection()
        self._check_relay()
        for _ in range(4096):
            message = conn.recv_match(blocking=False)
            if message is None:
                break
            if message.get_type() == 'BAD_DATA':
                continue
            if message.get_srcSystem() != self.system_id or message.get_srcComponent() != 1:
                continue
            name = message.get_type()
            now = time.monotonic()
            self._message_count += 1
            self.latest[name] = message
            self.received[name] = now
            if name in self._history:
                stamp = int(message.time_boot_ms)
                history = self._history[name]
                if history and stamp < history[-1][0]:
                    raise RuntimeError('Autopilot time moved backwards; reconnect the vehicle')
                if not history or stamp > history[-1][0]:
                    history.append((stamp, message, now))
                self._latest_time_ms = max(self._latest_time_ms, stamp)
            elif name == 'PARAM_VALUE':
                parameter = message.param_id
                if isinstance(parameter, bytes):
                    parameter = parameter.decode('ascii')
                parameter = str(parameter).rstrip('\x00')
                value = float(message.param_value)
                if not math.isfinite(value):
                    raise ValueError('Autopilot returned a nonfinite parameter')
                self._parameters[parameter] = (value, self._message_count)
        else:
            raise StateUnavailable('Telemetry queue exceeds the bounded drain limit')
        if time.monotonic() - self._last_heartbeat_sent >= 1.:
            conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self._last_heartbeat_sent = time.monotonic()

    def _wait(self, predicate, deadline, description):
        while time.monotonic() < deadline:
            self.drain()
            if predicate():
                return
            time.sleep(.005)
        status = self.latest.get('STATUSTEXT')
        detail = str(status.text) if status is not None else 'no status message'
        raise TimeoutError(f'{self.vehicle_id}: timeout waiting for {description}: {detail}')

    def state(self, *, require_new=True):
        self.drain()
        positions = self._history['LOCAL_POSITION_NED']
        attitudes = self._history['ATTITUDE']
        if not positions or not attitudes:
            raise StateUnavailable('Position and attitude have not both arrived')
        candidates = []
        for p in reversed(positions):
            if require_new and self._last_pair is not None and p[0] <= self._last_pair[0]:
                break
            for a in reversed(attitudes):
                if require_new and self._last_pair is not None and a[0] <= self._last_pair[1]:
                    break
                if p[0]-a[0] > self.maximum_pair_skew_s * 1000. + 1e-6:
                    break
                if abs(p[0]-a[0]) <= self.maximum_pair_skew_s * 1000. + 1e-6:
                    candidates.append((p, a))
                    break
            if candidates:
                break
        if not candidates:
            raise StateUnavailable('No new coherent position/attitude pair')
        p, a = candidates[0]
        now = time.monotonic()
        # Wall age does not grow the physical age while the native servo gate
        # holds the world for external computation. Simulation-age checks remain.
        if ((not self.relay.held and max(now-p[2], now-a[2]) > .25)
                or self._latest_time_ms-min(p[0], a[0]) > 60):
            raise StateUnavailable('Observation is stale')
        position, attitude = p[1], a[1]
        raw = [position.x, position.y, position.z, position.vx, position.vy, position.vz,
               attitude.roll, attitude.pitch, attitude.yaw,
               attitude.rollspeed, attitude.pitchspeed, attitude.yawspeed]
        if not np.isfinite(raw).all():
            raise ValueError('Autopilot returned a nonfinite state')
        pair = (p[0], a[0])
        if self._last_pair and max(pair) < max(self._last_pair):
            raise RuntimeError('Observation time moved backwards')
        if require_new:
            self._last_pair = pair
        result = State(
            position=ENU_FROM_NED @ np.array(raw[:3]),
            velocity=ENU_FROM_NED @ np.array(raw[3:6]),
            rotation=ENU_FROM_NED @ _rotation(*raw[6:9]) @ FRD_FROM_FLU,
            body_rates=np.array(raw[9:12]) * np.array([1., -1., -1.]),
            time_s=max(pair)*.001, position_time_s=p[0]*.001, attitude_time_s=a[0]*.001,
            estimator_type=int(self.readback['AHRS_EKF_TYPE']) if 'AHRS_EKF_TYPE' in self.readback else None)
        if require_new:
            self._event('state', time_s=result.time_s, position=result.position.tolist(),
                        velocity=result.velocity.tolist(), estimator_type=result.estimator_type)
        return result

    def get_parameter(self, name):
        return self._parameter(name, None)

    def set_parameter(self, name, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Parameter must be finite')
        if name == 'MAV_SYSID' and value != self.system_id:
            raise ValueError('Changing the connected vehicle identity requires a new session')
        result = self._parameter(name, value)
        self.readback[name] = result
        self._event('parameter', name=name, value=result)
        return result

    def _parameter(self, name, value):
        conn = self._require_connection()
        if not re.fullmatch(r'[A-Z0-9_]{1,16}', name):
            raise ValueError('Invalid ArduPilot parameter name')
        for _ in range(4):
            self.drain()
            sent_after = self._message_count
            if value is None:
                conn.mav.param_request_read_send(self.system_id, 1, name.encode('ascii'), -1)
            else:
                conn.mav.param_set_send(self.system_id, 1, name.encode('ascii'), value,
                                        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            deadline = time.monotonic()+1.5
            while time.monotonic() < deadline:
                self.drain()
                response = self._parameters.get(name)
                if response and response[1] > sent_after and (value is None or abs(response[0]-value) <= 1e-4):
                    return response[0]
                time.sleep(.005)
        raise TimeoutError(f'{self.vehicle_id}: parameter readback failed: {name}')

    def _command(self, command, *values):
        if len(values) > 7 or not np.isfinite(values).all():
            raise ValueError('Invalid command parameters')
        self._require_connection().mav.command_long_send(self.system_id, 1, command,
            0, *values, *([0.]*(7-len(values))))

    def set_position(self, position_enu, yaw_enu=None):
        target = ENU_FROM_NED @ _vector(position_enu, 3, 'position target')
        if yaw_enu is not None and not math.isfinite(yaw_enu):
            raise ValueError('Yaw must be finite radians')
        state = self.state(require_new=False)
        mask = 3576 if yaw_enu is None else 2552
        yaw = 0. if yaw_enu is None else math.pi/2-float(yaw_enu)
        self.release_motors()
        self.connection.mav.set_position_target_local_ned_send(
            round(state.time_s*1000) & 0xffffffff, self.system_id, 1,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, mask, *target, 0., 0., 0., 0., 0., 0., yaw, 0.)

    def set_rates(self, thrust_fraction, rates_flu):
        thrust = float(thrust_fraction)
        rates = _vector(rates_flu, 3, 'body rates')
        if not math.isfinite(thrust) or not 0 <= thrust <= 1:
            raise ValueError('Body collective must be in [0, 1]')
        if not int(self.readback.get('GUID_OPTIONS', 0)) & 8:
            raise RuntimeError('Body-collective commands require GUID_OPTIONS bit 3 (value 8)')
        state = self.state(require_new=False)
        cosine = float(state.rotation[2, 2])
        if cosine <= 0:
            raise ValueError('Body collective cannot be inverted for this orientation')
        if int(self.readback.get('ATC_ANGLE_BOOST', 1)):
            preboost = thrust * max(cosine, .1) / min(10*cosine, 1.)
        else:
            preboost = thrust
        if not 0 <= preboost <= 1:
            raise ValueError('Requested body collective exceeds the native interface range')
        self.release_motors()
        frd = rates * np.array([1., -1., -1.])
        self.connection.mav.set_attitude_target_send(
            round(state.time_s*1000) & 0xffffffff, self.system_id, 1,
            128, [0., 0., 0., 0.], *frd, preboost)

    def set_motor_pwm(self, pwm):
        self._require_connection()
        if self.relay.thread is None or not self.relay.thread.is_alive():
            raise RuntimeError('Motor relay is not running')
        self.relay.set_pwm(pwm)

    def set_motor_delta(self, delta_pwm):
        """Add four PWM offsets to each new native output, in motor order."""
        self._require_connection()
        if self.relay.thread is None or not self.relay.thread.is_alive():
            raise RuntimeError('Motor relay is not running')
        self.relay.set_delta(delta_pwm)

    def pause_control(self, timeout=10., physics_dt=.001):
        """Hold the next native servo frame after delivering its physical state."""
        return self.relay.pause(timeout=timeout, physics_dt=physics_dt)

    def step_control(self, seconds=.02, timeout=10.):
        """Advance exactly one requested interval from a held servo boundary."""
        return self.relay.advance(seconds=seconds, timeout=timeout)

    def resume_control(self):
        """Release the gate and return to unmodified native motor output."""
        self.relay.resume()

    def release_motors(self):
        self.relay.release()

    def arm_and_takeoff(self, altitude=3., timeout=120):
        altitude = float(altitude)
        if not math.isfinite(altitude) or altitude <= 0:
            raise ValueError('Takeoff altitude must be positive metres above home')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Takeoff timeout must be positive wall-clock seconds')
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            try:
                initial = self.state(require_new=False)
                break
            except StateUnavailable:
                time.sleep(.01)
        else:
            raise TimeoutError(f'{self.vehicle_id}: no coherent takeoff state')
        last_request = -math.inf
        while time.monotonic() < deadline:
            self.drain()
            heartbeat = self.latest.get('HEARTBEAT')
            if heartbeat and heartbeat.custom_mode == 4 and heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                break
            if time.monotonic()-last_request >= 1.:
                self.connection.mav.set_mode_send(self.system_id,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
                self._command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.)
                last_request = time.monotonic()
            time.sleep(.02)
        else:
            status = self.latest.get('STATUSTEXT')
            raise TimeoutError(f'{self.vehicle_id}: arming failed: {getattr(status, "text", "no status")}')
        self._command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0., 0., 0., 0., 0., 0., altitude)
        while time.monotonic() < deadline:
            try:
                current = self.state(require_new=False)
                if current.position[2] >= initial.position[2] + .9*altitude:
                    self._event('takeoff', altitude_m=altitude, time_s=current.time_s)
                    return current
            except StateUnavailable:
                pass
            time.sleep(.01)
        raise TimeoutError(f'{self.vehicle_id}: takeoff altitude not reached')

    def land(self):
        """Request native LAND; this call does not assert touchdown."""
        self.release_motors()
        if self.connection is not None:
            self.connection.mav.set_mode_send(self.system_id,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)
            self._event('land_requested')

    def close(self):
        self.release_motors()
        try:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
        finally:
            try:
                self.relay.close()
            finally:
                if self._events:
                    self._events.close()
                    self._events = None


class World:
    """Native Gazebo commands; a prefix can run ``gz`` in another WSL distro."""
    def __init__(self, world_name='copter', partition='copter', command_prefix=()):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', world_name):
            raise ValueError('Invalid world name')
        self.world_name = world_name
        self.partition = str(partition)
        self.command_prefix = tuple(command_prefix)

    def _run(self, arguments, timeout=10):
        env = os.environ.copy()
        env['GZ_PARTITION'] = self.partition
        command = [*self.command_prefix, 'gz', *arguments]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
        except OSError as exc:
            if exc.errno != 8 or not Path('/init').is_file() or not str(command[0]).endswith('wsl.exe'):
                raise
            result = subprocess.run(['/init', command[0], *command],
                                    capture_output=True, text=True, timeout=timeout, env=env)
        if result.returncode:
            raise RuntimeError('Gazebo command failed: ' + result.stderr.strip())
        return result.stdout

    def set_wind(self, velocity_enu):
        """Publish a uniform air-velocity command in m/s, not an external force.

The command's successful publication is not a measurement of applied forces.
The native WindEffects system and wind-enabled model links must be loaded.
"""
        x, y, z = _vector(velocity_enu, 3, 'wind velocity')
        text = f'linear_velocity {{ x: {x:.17g} y: {y:.17g} z: {z:.17g} }} enable_wind: true'
        return self._run(['topic', '-t', f'/world/{self.world_name}/wind',
                          '-m', 'gz.msgs.Wind', '-p', text])

    def set_marker(self, name, position_enu):
        if not re.fullmatch(r'[A-Za-z0-9_:-]+', name):
            raise ValueError('Invalid marker name')
        x, y, z = _vector(position_enu, 3, 'marker position')
        text = f'name: "{name}" position {{ x: {x:.17g} y: {y:.17g} z: {z:.17g} }}'
        response = self._run(['service', '-s', f'/world/{self.world_name}/set_pose',
            '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000', '--req', text])
        if not re.search(r'data:\s*true', response):
            raise RuntimeError('Gazebo did not confirm the marker update')
        return response
