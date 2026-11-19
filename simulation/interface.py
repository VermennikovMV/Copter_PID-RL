"""Communication with ArduPilot SITL (MAVLink) and Gazebo.

Frames: positions and velocities ENU, body rates FLU (rad/s); ArduPilot works in NED/FRD.

Vehicle      MAVLink client of one SITL instance: telemetry, parameters, GUIDED targets, takeoff.
MotorRelay   sits between SITL and Gazebo on the JSON FDM link (servo packets SITL -> Gazebo,
             sensor packets Gazebo -> SITL). It can replace the four motor PWM values (end-to-end
             control) and hold the simulation at a servo frame so that an external controller
             advances it in exact control steps (lockstep).
World        `gz` commands for the running world (wind, plugin topics).
"""
from collections import deque
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
SERVO_PACKET = struct.Struct('<HHI16H')   # magic, frame rate, frame count, 16 PWM values
SERVO_MAGIC = 18458
PWM_MIN, PWM_MAX = 1100, 1900
GUIDED, LAND = 4, 9


class StateUnavailable(RuntimeError):
    """A new, time-consistent pair of position and attitude messages has not arrived yet."""


@dataclass(frozen=True)
class State:
    position: np.ndarray      # ENU, m
    velocity: np.ndarray      # ENU, m/s
    rotation: np.ndarray      # body FLU -> world ENU
    body_rates: np.ndarray    # FLU, rad/s
    time_s: float             # autopilot time of the newer message


def rotation_from_euler(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def run_command(command, timeout=None, **kwargs):
    """subprocess.run that can also start wsl.exe from inside WSL (ENOEXEC fallback through /init)."""
    try:
        return subprocess.run(command, timeout=timeout, **kwargs)
    except OSError as exc:
        if exc.errno != 8 or not Path('/init').is_file() or not str(command[0]).endswith('wsl.exe'):
            raise
        return subprocess.run(['/init', command[0], *command], timeout=timeout, **kwargs)


class MotorRelay:
    """UDP relay on the SITL <-> Gazebo FDM link with motor override and lockstep stepping.

    Native mode forwards everything unchanged. `set_pwm` replaces motors 1-4 in every servo packet
    (an override expires after `expiry_s` unless the relay is gated). `pause` holds the next servo
    frame; `advance(seconds)` releases it and holds again after exactly `seconds` of physics."""

    def __init__(self, relay_port, physics_port, expiry_s=.5):
        self.address = ('127.0.0.1', int(relay_port))           # SITL sends servo packets here
        self.physics_address = ('127.0.0.1', int(physics_port))  # Gazebo ArduPilotPlugin
        self.expiry_s = float(expiry_s)
        self.cv = threading.Condition()
        self.stop = threading.Event()
        self.socket = self.thread = None
        self.autopilot_address = None
        self.override = None            # (pwm tuple, wall time set)
        self.error = None
        self.latest_fdm = None          # last sensor packet from Gazebo
        self.last_frame = None
        self.sensor_count = 0
        self.gated, self.until, self.physics_dt = False, math.inf, None
        self.pending = None             # held servo packet (packet fields, raw bytes)

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(self.address)
        self.socket.settimeout(.1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    @property
    def held(self):
        with self.cv:
            return self.pending is not None

    def set_pwm(self, pwm):
        values = np.asarray(pwm, float)
        if values.shape != (4,) or not np.all((values >= PWM_MIN) & (values <= PWM_MAX)):
            raise ValueError(f'four motor PWM values in [{PWM_MIN}, {PWM_MAX}] required')
        with self.cv:
            self._raise_error()
            self.override = (tuple(np.rint(values).astype(int)), time.monotonic())

    def release(self):
        with self.cv:
            self.override = None

    def pause(self, timeout=10., physics_dt=.001):
        """Hold the next servo frame; returns the held boundary (latest sensor packet)."""
        with self.cv:
            self.physics_dt, self.gated, self.until = float(physics_dt), True, -math.inf
        return self._wait_held(timeout)

    def advance(self, seconds=.02, timeout=10.):
        """Release the held servo frame (with the current override) and hold again after `seconds`."""
        with self.cv:
            self._raise_error()
            if not self.gated or self.pending is None or self.latest_fdm is None:
                raise RuntimeError('advance requires a held servo frame (call pause first)')
            steps = seconds / self.physics_dt
            if abs(steps - round(steps)) > 1e-7:
                raise ValueError('the control step must be a whole number of physics steps')
            start, before = float(self.latest_fdm['timestamp']), self.sensor_count
            self.until = start + seconds
            packet, raw = self.pending
            self.pending = None
            self._send_servo(packet, raw)
        result = self._wait_held(timeout)
        elapsed = float(result['fdm']['timestamp']) - start
        if abs(elapsed - seconds) > 1e-8 or result['sensor_count'] - before != round(steps):
            raise RuntimeError(f'physics advanced {elapsed:.6f} s instead of {seconds} s')
        return result

    def resume(self):
        """Back to free running with native motor outputs."""
        with self.cv:
            self.gated, self.until, self.override = False, math.inf, None
            if self.pending is not None:
                packet, raw = self.pending
                self.pending = None
                self._send_servo(packet, raw)
            self.cv.notify_all()

    def close(self):
        self.release()
        self.stop.set()
        with self.cv:
            self.pending = None
            self.cv.notify_all()
        if self.thread is not None:
            self.thread.join(2.)
        elif self.socket is not None:
            self.socket.close()

    # ---------------------------------------------------------------- internals
    def _raise_error(self):
        if self.error:
            raise RuntimeError('motor relay failed: ' + self.error)
        if self.stop.is_set():
            raise RuntimeError('motor relay is closed')

    def _wait_held(self, timeout):
        deadline = time.monotonic() + timeout
        with self.cv:
            while self.pending is None:
                self._raise_error()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('no servo frame from SITL')
                self.cv.wait(min(remaining, .1))
            return dict(fdm=dict(self.latest_fdm), native_pwm=list(self.pending[0][3:7]),
                        sensor_count=self.sensor_count)

    def _send_servo(self, packet, raw):
        """Forward a servo packet to Gazebo, with motors 1-4 replaced by the override (caller holds cv).
        While gated, an override stays valid for any wall time (the simulation is held)."""
        if self.override and not self.gated and time.monotonic() - self.override[1] > self.expiry_s:
            self.override = None
            self.error = 'motor override expired; native outputs restored'
        if self.override:
            packet = list(packet)
            packet[3:7] = self.override[0]
            raw = SERVO_PACKET.pack(*packet)
        self.socket.sendto(raw, self.physics_address)

    def _on_sensor(self, data):
        fdm = json.loads(data)
        stamp = float(fdm['timestamp'])
        with self.cv:
            if self.latest_fdm is not None:
                previous = float(self.latest_fdm['timestamp'])
                if stamp < previous - 1e-9:
                    raise RuntimeError('physics time moved backwards')
                if stamp <= previous + 1e-9:
                    return                                    # duplicate
                if self.gated and abs(stamp - previous - self.physics_dt) > 1e-8:
                    raise RuntimeError('physics step changed or a sensor packet was lost')
            self.latest_fdm = fdm
            self.sensor_count += 1
            if self.autopilot_address is not None:
                self.socket.sendto(data, self.autopilot_address)

    def _on_servo(self, data, source):
        if len(data) != SERVO_PACKET.size:
            raise ValueError('unexpected servo packet size')
        packet = list(SERVO_PACKET.unpack(data))
        if packet[0] != SERVO_MAGIC:
            raise ValueError('unsupported servo packet format')
        if self.autopilot_address not in (None, source):
            raise ValueError('servo packet source changed')
        self.autopilot_address = source
        with self.cv:
            if packet[2] == self.last_frame:                  # SITL repeats a frame until it gets sensors
                if self.latest_fdm is None:
                    self.socket.sendto(data, self.physics_address)
                return
            if self.last_frame is not None and packet[2] != (self.last_frame + 1) & 0xffffffff:
                raise RuntimeError('servo frames are not consecutive')
            self.last_frame = packet[2]
            if self.gated and self.latest_fdm is not None and float(self.latest_fdm['timestamp']) >= self.until - 1e-9:
                if self.pending is not None:
                    raise RuntimeError('more than one held servo frame')
                self.pending = (packet, data)                 # hold: the simulation waits for advance()
                self.cv.notify_all()
            else:
                self._send_servo(packet, data)

    def _run(self):
        try:
            while not self.stop.is_set():
                try:
                    data, source = self.socket.recvfrom(65535)
                except socket.timeout:
                    continue
                if source == self.physics_address:
                    self._on_sensor(data)
                elif source[0] == '127.0.0.1':
                    self._on_servo(data, source)
        except BaseException as exc:
            if not self.stop.is_set():
                with self.cv:
                    self.error = repr(exc)
                    self.cv.notify_all()
        finally:
            self.socket.close()


class Vehicle:
    """MAVLink client of one SITL instance (system id = numeric suffix of the vehicle name)."""

    def __init__(self, vehicle_id, tcp_port, relay_port, physics_port, system_id=None, max_pair_skew_s=.01):
        self.vehicle_id = str(vehicle_id)
        self.system_id = int(system_id if system_id is not None else re.search(r'(\d+)$', self.vehicle_id).group(1))
        self.tcp_port = int(tcp_port)
        self.max_pair_skew_ms = 1000. * max_pair_skew_s
        self.relay = MotorRelay(relay_port, physics_port)
        self.connection = None
        self.latest = {}                 # last message of every type
        self.readback = {}               # parameters read or set in this session
        self._parameters = {}            # name -> (value, message counter)
        self._message_count = 0
        self._latest_time_ms = 0
        self._last_heartbeat_sent = 0.
        self._last_pair = None
        self._history = {name: deque(maxlen=128) for name in ('LOCAL_POSITION_NED', 'ATTITUDE')}

    def start_relay(self):
        self.relay.start()

    def connect(self, timeout=60):
        deadline = time.monotonic() + timeout
        while self.connection is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{self.vehicle_id}: no autopilot TCP connection')
            try:
                self.connection = mavutil.mavlink_connection(f'tcp:127.0.0.1:{self.tcp_port}',
                                                             source_system=255, source_component=190)
            except OSError:
                time.sleep(.1)
        self._wait(lambda: 'HEARTBEAT' in self.latest, deadline, 'autopilot heartbeat')
        self.connection.target_system, self.connection.target_component = self.system_id, 1
        for message_id in (30, 32):      # ATTITUDE, LOCAL_POSITION_NED at 50 Hz
            self._command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, message_id, 20000.)
        if self.get_parameter('MAV_SYSID') != self.system_id:
            raise RuntimeError(f'{self.vehicle_id}: MAV_SYSID differs from the vehicle')
        self._wait(lambda: all(self._history.values()), deadline, 'position and attitude')
        return self

    def drain(self):
        """Read all queued MAVLink messages; send a GCS heartbeat once per second."""
        if self.relay.error:
            raise RuntimeError('motor relay failed: ' + self.relay.error)
        for _ in range(4096):
            message = self.connection.recv_match(blocking=False)
            if message is None:
                break
            name = message.get_type()
            if name == 'BAD_DATA' or message.get_srcSystem() != self.system_id or message.get_srcComponent() != 1:
                continue
            self._message_count += 1
            self.latest[name] = message
            if name in self._history:
                stamp, history = int(message.time_boot_ms), self._history[name]
                if history and stamp < history[-1][0]:
                    raise RuntimeError('autopilot time moved backwards')
                if not history or stamp > history[-1][0]:
                    history.append((stamp, message, time.monotonic()))
                self._latest_time_ms = max(self._latest_time_ms, stamp)
            elif name == 'PARAM_VALUE':
                param = message.param_id
                param = (param.decode('ascii') if isinstance(param, bytes) else str(param)).rstrip('\x00')
                self._parameters[param] = (float(message.param_value), self._message_count)
        else:
            raise StateUnavailable('telemetry queue is too long')
        if time.monotonic() - self._last_heartbeat_sent >= 1.:
            self.connection.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self._last_heartbeat_sent = time.monotonic()

    def _wait(self, predicate, deadline, description):
        while time.monotonic() < deadline:
            self.drain()
            if predicate():
                return
            time.sleep(.005)
        status = self.latest.get('STATUSTEXT')
        raise TimeoutError(f'{self.vehicle_id}: timeout waiting for {description}: '
                           f'{status.text if status is not None else "no status message"}')

    def state(self, *, require_new=True, max_age_s=.25):
        """Latest position/attitude pair not more than max_pair_skew apart. require_new: the pair must be
        newer than the one returned last time (raises StateUnavailable otherwise). max_age_s: wall-clock
        age limit (None when another vehicle holds the simulation)."""
        self.drain()
        positions, attitudes = self._history['LOCAL_POSITION_NED'], self._history['ATTITUDE']
        if not positions or not attitudes:
            raise StateUnavailable('position and attitude have not both arrived')
        last = self._last_pair if require_new else None
        pair = None
        for p in reversed(positions):
            if last is not None and p[0] <= last[0]:
                break
            for a in reversed(attitudes):
                if (last is not None and a[0] <= last[1]) or p[0] - a[0] > self.max_pair_skew_ms + 1e-6:
                    break
                if abs(p[0] - a[0]) <= self.max_pair_skew_ms + 1e-6:
                    pair = (p, a)
                    break
            if pair:
                break
        if pair is None:
            raise StateUnavailable('no new consistent position/attitude pair')
        p, a = pair
        now = time.monotonic()
        # while the relay holds the simulation, wall-clock age does not matter
        too_old = max_age_s is not None and not self.relay.held and max(now - p[2], now - a[2]) > max_age_s
        if too_old or self._latest_time_ms - min(p[0], a[0]) > 60:
            raise StateUnavailable('observation is stale')
        if require_new:
            self._last_pair = (p[0], a[0])
        pos, att = p[1], a[1]
        return State(position=ENU_FROM_NED @ np.array([pos.x, pos.y, pos.z]),
                     velocity=ENU_FROM_NED @ np.array([pos.vx, pos.vy, pos.vz]),
                     rotation=ENU_FROM_NED @ rotation_from_euler(att.roll, att.pitch, att.yaw) @ FRD_FROM_FLU,
                     body_rates=np.array([att.rollspeed, -att.pitchspeed, -att.yawspeed]),
                     time_s=max(p[0], a[0]) * .001)

    # ---------------------------------------------------------------- parameters and commands
    def get_parameter(self, name):
        return self._parameter(name, None)

    def set_parameter(self, name, value):
        return self._parameter(name, float(value))

    def _parameter(self, name, value):
        for _ in range(4):
            self.drain()
            sent_after = self._message_count
            if value is None:
                self.connection.mav.param_request_read_send(self.system_id, 1, name.encode('ascii'), -1)
            else:
                self.connection.mav.param_set_send(self.system_id, 1, name.encode('ascii'), value,
                                                   mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                self.drain()
                response = self._parameters.get(name)
                if response and response[1] > sent_after and (value is None or abs(response[0] - value) <= 1e-4):
                    self.readback[name] = response[0]
                    return response[0]
                time.sleep(.005)
        raise TimeoutError(f'{self.vehicle_id}: no reply for parameter {name}')

    def _command(self, command, *values):
        self.connection.mav.command_long_send(self.system_id, 1, command, 0, *values, *([0.] * (7 - len(values))))

    def set_mode(self, mode):
        self.connection.mav.set_mode_send(self.system_id, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode)

    def set_position(self, position_enu, yaw_enu=None):
        """GUIDED position target (SET_POSITION_TARGET_LOCAL_NED); yaw None keeps the heading."""
        target = ENU_FROM_NED @ np.asarray(position_enu, float)
        mask = 3576 if yaw_enu is None else 2552
        yaw = 0. if yaw_enu is None else math.pi / 2 - float(yaw_enu)
        stamp = round(self.state(require_new=False, max_age_s=None).time_s * 1000) & 0xffffffff
        self.release_motors()
        self.connection.mav.set_position_target_local_ned_send(stamp, self.system_id, 1, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                                                               mask, *target, 0., 0., 0., 0., 0., 0., yaw, 0.)

    def set_motor_pwm(self, pwm):
        """Replace the four motor outputs of SITL (end-to-end control)."""
        if not self.relay.running:
            raise RuntimeError('motor relay is not running')
        self.relay.set_pwm(pwm)

    def release_motors(self):
        self.relay.release()

    def pause_control(self, timeout=10., physics_dt=.001):
        return self.relay.pause(timeout=timeout, physics_dt=physics_dt)

    def step_control(self, seconds=.02, timeout=10.):
        return self.relay.advance(seconds=seconds, timeout=timeout)

    def resume_control(self):
        self.relay.resume()

    def arm_and_takeoff(self, altitude=3., timeout=120):
        """GUIDED, arm, take off to `altitude` above the start; returns the state at 90 % of it."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                initial = self.state(require_new=False)
                break
            except StateUnavailable:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'{self.vehicle_id}: no state before takeoff')
                time.sleep(.01)
        last_request = -math.inf
        while True:
            self.drain()
            heartbeat = self.latest.get('HEARTBEAT')
            if heartbeat and heartbeat.custom_mode == GUIDED and heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                break
            if time.monotonic() > deadline:
                status = self.latest.get('STATUSTEXT')
                raise TimeoutError(f'{self.vehicle_id}: arming failed: {getattr(status, "text", "no status")}')
            if time.monotonic() - last_request >= 1.:
                self.set_mode(GUIDED)
                self._command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.)
                last_request = time.monotonic()
            time.sleep(.02)
        self._command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0., 0., 0., 0., 0., 0., float(altitude))
        while time.monotonic() < deadline:
            try:
                current = self.state(require_new=False)
                if current.position[2] >= initial.position[2] + .9 * altitude:
                    return current
            except StateUnavailable:
                pass
            time.sleep(.01)
        raise TimeoutError(f'{self.vehicle_id}: takeoff altitude not reached')

    def land(self):
        self.release_motors()
        if self.connection is not None:
            self.set_mode(LAND)

    def close(self):
        self.release_motors()
        try:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
        finally:
            self.relay.close()


class World:
    """`gz` commands for the running world; `command_prefix` runs them in another WSL distribution."""

    def __init__(self, world_name='copter', partition='copter', command_prefix=()):
        self.world_name, self.partition, self.command_prefix = world_name, str(partition), tuple(command_prefix)

    def gz(self, arguments, timeout=10):
        env = dict(os.environ, GZ_PARTITION=self.partition)
        result = run_command([*self.command_prefix, 'gz', *arguments], timeout=timeout,
                             capture_output=True, text=True, env=env)
        if result.returncode:
            raise RuntimeError('gz command failed: ' + result.stderr.strip())
        return result.stdout

    def publish(self, topic, text):
        """Publish a string message (e.g. the disturbance plugin schedule)."""
        return self.gz(['topic', '-t', topic, '-m', 'gz.msgs.StringMsg', '-p', f'data: "{text}"'])

    def set_wind(self, velocity_enu):
        """Uniform wind velocity, m/s (Gazebo WindEffects; not used by the disturbance experiments)."""
        x, y, z = (float(v) for v in velocity_enu)
        return self.gz(['topic', '-t', f'/world/{self.world_name}/wind', '-m', 'gz.msgs.Wind',
                        '-p', f'linear_velocity {{ x: {x:.17g} y: {y:.17g} z: {z:.17g} }} enable_wind: true'])
