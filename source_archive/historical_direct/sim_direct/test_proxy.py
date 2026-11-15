"""Local UDP contract checks; these are not claimed to be simulator flights."""
import json
import socket
import time

import numpy as np
import pytest

from sim_direct.proxy import Proxy, PACKET, MAGIC, action_to_pwm


def test_motor_order_and_scaling():
    hover = np.array([1500, 1510, 1520, 1530])
    assert action_to_pwm([0, 0, 0, 0], hover) == hover.tolist()
    assert action_to_pwm([-1, -1, -1, -1], hover) == [1100]*4
    value = action_to_pwm([1, .5, 0, -.5], hover)
    assert value == np.rint(1100+(hover-1100)*np.sqrt([2,1.5,1,.5])).astype(int).tolist()
    with pytest.raises(ValueError):
        action_to_pwm([0, np.nan, 0, 0], hover)


def test_real_udp_relay_and_exact_sensor_preservation(tmp_path):
    physics = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    physics.bind(('127.0.0.1', 0))
    physics.settimeout(2)
    autopilot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    autopilot.bind(('127.0.0.1', 0))
    autopilot.settimeout(2)
    relay = Proxy(tmp_path, listen_port=0, physics_port=physics.getsockname()[1])
    relay.thread.start()
    original = PACKET.pack(MAGIC, 400, 1234, *range(1200, 1216))
    try:
        autopilot.sendto(original, relay.listen_address)
        assert physics.recvfrom(4096)[0] == original
        sensor = b'\n {"timestamp":1.25,"imu":{"gyro":[0,0,0]}}\n'
        physics.sendto(sensor, relay.listen_address)
        assert autopilot.recvfrom(4096)[0] == sensor
        pwm = relay.set_action([0,.1,.2,.3], [1500]*4)
        autopilot.sendto(original, relay.listen_address)
        received = PACKET.unpack(physics.recvfrom(4096)[0])
        assert list(received[3:7]) == pwm
        assert received[:3] == PACKET.unpack(original)[:3]
        assert received[7:] == PACKET.unpack(original)[7:]
        relay.active = False
        autopilot.sendto(original, relay.listen_address)
        assert physics.recvfrom(4096)[0] == original
    finally:
        relay.close()
        physics.close()
        autopilot.close()
    assert relay.error is None
    assert (relay.count, relay.overridden, relay.sensors) == (3,1,1)
    records = [json.loads(s) for s in (tmp_path/'actuators.jsonl').read_text().splitlines()]
    assert [r['override'] for r in records] == [False,True,False]


def test_stale_override_stops_relay(tmp_path):
    physics = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    physics.bind(('127.0.0.1',0))
    physics.settimeout(.2)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    relay = Proxy(tmp_path, listen_port=0, physics_port=physics.getsockname()[1])
    relay.set_action([0]*4, [1500]*4)
    relay.updated -= 1.
    relay.thread.start()
    try:
        sender.sendto(PACKET.pack(MAGIC,400,0,*([1500]*16)), relay.listen_address)
        relay.thread.join(2)
        assert relay.error and 'Stale direct action' in relay.error
        with pytest.raises(socket.timeout):
            physics.recvfrom(4096)
    finally:
        relay.close()
        sender.close()
        physics.close()
