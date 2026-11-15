"""Transparent JSON FDM relay with an explicitly armed motor-command override.

V2 rotor order is the physical Gazebo order [FR, RL, FL, RR]. This differs from
V1's reflected geometry. JSON sensor datagrams are forwarded byte-for-byte.
"""
import json
from pathlib import Path
import socket
import struct
import threading
import time
import traceback

import numpy as np

PACKET = struct.Struct('<HHI16H')
MAGIC = 18458
MOTOR_ORDER = ('FR', 'RL', 'FL', 'RR')


def action_to_pwm(action, hover_pwm):
    action = np.asarray(action, dtype=float)
    hover_pwm = np.asarray(hover_pwm, dtype=float)
    if action.shape != (4,) or not np.isfinite(action).all():
        raise ValueError('Four finite direct motor actions required')
    if hover_pwm.shape != (4,) or not np.isfinite(hover_pwm).all() or np.any(hover_pwm <= 1100) or np.any(hover_pwm >= 1900):
        raise ValueError('Four nominal hover PWM values inside (1100,1900) required')
    pwm = 1100 + (hover_pwm - 1100) * np.sqrt(np.clip(action, -1, 1) + 1)
    return np.rint(np.clip(pwm, 1100, 1900)).astype(int).tolist()


class Proxy:
    def __init__(self, output, listen_port=19012, physics_port=19002, stale_after_s=.5):
        self.output = Path(output)
        self.active = False
        self.pwm = None
        self.updated = 0.
        self.stale_after_s = float(stale_after_s)
        self.physics_address = ('127.0.0.1', int(physics_port))
        self.stop = threading.Event()
        self.error = None
        self.count = self.overridden = self.sensors = 0
        self.latest_pwm = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', int(listen_port)))
        self.sock.settimeout(.1)
        self.listen_address = self.sock.getsockname()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def set_action(self, action, hover_pwm):
        pwm = action_to_pwm(action, hover_pwm)
        # A single immutable assignment avoids partially updated commands.
        self.pwm = tuple(pwm)
        self.updated = time.monotonic()
        self.active = True
        return pwm

    def run(self):
        autopilot_address = None
        with (self.output / 'actuators.jsonl').open('w', buffering=1) as controls, (self.output / 'physics.jsonl').open('w', buffering=1) as physics:
            try:
                while not self.stop.is_set():
                    try:
                        data, source = self.sock.recvfrom(65535)
                    except socket.timeout:
                        continue
                    if len(data) == PACKET.size and struct.unpack_from('<H', data)[0] == MAGIC:
                        autopilot_address = source
                        packet = list(PACKET.unpack(data))
                        original = packet[3:7]
                        self.latest_pwm = original.copy()
                        override = bool(self.active)
                        if override:
                            if self.pwm is None or time.monotonic() - self.updated > self.stale_after_s:
                                raise RuntimeError('Stale direct action: stopping actuator relay')
                            packet[3:7] = self.pwm
                            data = PACKET.pack(*packet)
                            self.overridden += 1
                        self.sock.sendto(data, self.physics_address)
                        self.count += 1
                        controls.write(json.dumps({'wall': time.time(), 'monotonic': time.monotonic(),
                            'frame': packet[2], 'frame_rate': packet[1], 'input_pwm': original,
                            'output_pwm': packet[3:7], 'override': override,
                            'action_age_s': time.monotonic()-self.updated if override else None}) + '\n')
                    elif data.lstrip().startswith(b'{'):
                        if source != self.physics_address:
                            raise ValueError(f'Unexpected sensor sender {source}')
                        if autopilot_address is not None:
                            self.sock.sendto(data, autopilot_address)
                        # Persist the original JSON, but forward the exact received bytes.
                        physics.write(data.decode('utf-8').strip() + '\n')
                        self.sensors += 1
                    else:
                        raise ValueError(f'Unexpected datagram length={len(data)} source={source}')
            except Exception:
                self.error = traceback.format_exc()
            finally:
                self.sock.close()

    def close(self):
        self.active = False
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(3)
        else:
            self.sock.close()

