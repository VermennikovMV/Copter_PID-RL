"""Disturbances in Gazebo + SITL matching the PyBullet replica, applied while hovering in GUIDED.

Experiments (schedule time 0 = moment the schedule is armed in Gazebo, vehicle settled at 3 m):
  force3n        horizontal 3 N (ENU +x) from 5 s to 7 s
  gust           the replica's 'gust' profile for --seed (experiments/scenarios.py force_events)
  payload_force  extra weight as a vertical force -0.2*m*g from 0 s to 12 s (released at 12 s)
  payload_drop   real extra mass: a 0.35 kg link (inertia 0.2 * body) held by a DetachableJoint
                 from spawn, released at 12 s

The force/release is applied by the CopterDisturbanceSystem world plugin (gazebo/disturbance_plugin),
which counts time in Gazebo simulation time. Output: trace.csv (50 Hz, ArduPilot time aligned to
the schedule through the plugin log) and summary.json.

  python gazebo/disturbances.py --experiment force3n --out runs/gazebo/force3n_00
"""
import os
import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from simulation.launch import Session  # noqa: E402
from track_metrics import write_trace  # noqa: E402
from sitl import wait_state  # noqa: E402
from experiments.scenarios import force_events  # noqa: E402

# Built from gazebo/disturbance_plugin inside the Gazebo distribution (see docs/setup.md).
PLUGIN = os.environ.get('COPTER_DISTURBANCE_PLUGIN', os.path.expanduser('~/copter_disturbance_build/libCopterDisturbanceSystem.so'))
MASS, G = 1.75, 9.80665
BASE_INERTIA = (0.008, 0.015, 0.017)   # iris base_link (simulation/models/iris/model.sdf)
DETACH_TOPIC = '/copter/payload/detach'
ET.register_namespace('experimental', 'http://sdformat.org/schemas/experimental')


def schedule_for(experiment, seed):
    if experiment == 'force3n':
        return [('force', 5., 7., (3., 0., 0.))], None, 14.
    if experiment == 'gust':
        events = force_events('gust', seed)
        return [('force', e['on_s'], e['off_s'], tuple(e['force_enu_N'])) for e in events], None, 26.
    if experiment == 'payload_force':
        return [('force', 0., 12., (0., 0., -0.2 * MASS * G))], None, 18.
    if experiment == 'payload_drop':
        return [], 12., 18.
    raise ValueError(experiment)


def build_world(out, experiment, log_file):
    source = REPO / 'simulation' / 'world.sdf'
    tree = ET.parse(source)
    world = tree.getroot().find('world')
    plugin = ET.SubElement(world, 'plugin', filename=PLUGIN, name='copter_disturbance::DisturbanceSystem')
    for tag, value in (('model_name', 'drone1'), ('link_name', 'base_link'), ('topic', '/copter/disturbance'),
                       ('detach_topic', DETACH_TOPIC), ('log_file', log_file)):
        ET.SubElement(plugin, tag).text = value
    if experiment == 'payload_drop':
        drone = next(m for m in world.findall('model') if m.attrib['name'] == 'drone1')
        include = drone.find('include')
        joint = ET.SubElement(include, 'plugin', filename='gz-sim-detachable-joint-system',
                              name='gz::sim::systems::DetachableJoint')
        for tag, value in (('parent_link', 'base_link'), ('child_model', 'payload'), ('child_link', 'link'),
                           ('detach_topic', DETACH_TOPIC)):
            ET.SubElement(joint, tag).text = value
        payload = ET.SubElement(world, 'model', name='payload')
        ET.SubElement(payload, 'pose').text = drone.findtext('pose')
        link = ET.SubElement(payload, 'link', name='link')
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'mass').text = repr(0.2 * MASS)
        inertia = ET.SubElement(inertial, 'inertia')
        for name, value in zip(('ixx', 'iyy', 'izz'), BASE_INERTIA):
            ET.SubElement(inertia, name).text = repr(0.2 * value)
        for name in ('ixy', 'ixz', 'iyz'):
            ET.SubElement(inertia, name).text = '0'
    path = out / 'world_disturbance.sdf'
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True, choices=('force3n', 'gust', 'payload_force', 'payload_drop'))
    ap.add_argument('--seed', type=int, default=90000)
    ap.add_argument('--gazebo-distro', default='Ubuntu')
    ap.add_argument('--ardupilot', type=Path, default=Path.home() / 'ardupilot-ROSMeetUp')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    forces, detach, duration = schedule_for(a.experiment, a.seed)
    log_file = str((a.out / 'plugin.jsonl').resolve())
    world = build_world(a.out, a.experiment, log_file)
    prefix = ('/mnt/c/Windows/System32/wsl.exe', '-d', a.gazebo_distro, '--')
    rows = []
    with Session(a.out / 'simulation', gazebo_prefix=prefix, vehicle_ids=('drone1',),
                 ardupilot=a.ardupilot, world=world) as session:
        v = session.clients['drone1']
        initial = wait_state(v)
        origin = initial.position - np.array([0., 0., session.vehicles[0].position_enu[2]])
        goal = origin + np.array([0., 0., 3.])
        v.arm_and_takeoff(3.)
        deadline, stable = time.monotonic() + 60., None
        while time.monotonic() < deadline:
            s = wait_state(v)
            v.set_position(goal, math.pi / 2)
            if np.linalg.norm(s.position - goal) <= .03 and np.linalg.norm(s.velocity) <= .05:
                stable = s.time_s if stable is None else stable
                if s.time_s - stable >= 2.:
                    break
            else:
                stable = None
        else:
            raise TimeoutError('hover not settled')
        lines = [f'force {on} {off} {f[0]} {f[1]} {f[2]}' for _, on, off, f in forces]
        if detach is not None:
            lines.append(f'detach {detach}')
        text = '\\n'.join(lines) if lines else ''
        # a single `gz topic` publication can be lost before discovery completes: resend until the
        # plugin logs that the schedule is armed (resend only after 3 s, so it is never re-armed)
        for _ in range(5):
            published = wait_state(v).time_s
            session.world._run(['topic', '-t', '/copter/disturbance', '-m', 'gz.msgs.StringMsg',
                                '-p', f'data: "{text}"'])
            wait_end = time.monotonic() + 3.
            while time.monotonic() < wait_end and 'schedule_armed' not in Path(log_file).read_text():
                wait_state(v)
                v.set_position(goal, math.pi / 2)
            if 'schedule_armed' in Path(log_file).read_text():
                break
        else:
            raise RuntimeError('disturbance schedule was not armed')
        end = time.monotonic() + duration + 20.
        last = None
        while time.monotonic() < end:
            s = wait_state(v)
            v.set_position(goal, math.pi / 2)
            if last is not None and s.time_s <= last:
                continue
            last = s.time_s
            p = s.position - goal
            att = v.latest['ATTITUDE']
            rows.append(dict(t_ap=round(s.time_s, 4), x=p[0], y=p[1], z=p[2],
                             vx=s.velocity[0], vy=s.velocity[1], vz=s.velocity[2],
                             roll_deg=math.degrees(att.roll), pitch_deg=math.degrees(att.pitch)))
            if s.time_s - published > duration + 3.:
                break
        v.land()
    events = [json.loads(x) for x in Path(log_file).read_text().splitlines() if x.strip()]
    armed = next(e for e in events if e['event'] == 'schedule_armed')
    write_trace(a.out / 'trace_raw.csv', rows)
    summary = dict(experiment=a.experiment, seed=a.seed, forces=forces, detach_s=detach,
                   published_ap_time_s=published, armed_gz_sim_time_s=armed['sim_time'], plugin_events=events)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ('experiment', 'published_ap_time_s', 'armed_gz_sim_time_s')}))


if __name__ == '__main__':
    main()
