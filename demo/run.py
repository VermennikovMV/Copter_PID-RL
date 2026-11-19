"""Demonstrations: three vehicles in one Gazebo scene, flying at the same time in the same wind.

  drone1 (blue)    stock ArduPilot: GUIDED in hover, the track as an AUTO mission
  drone2 (orange)  the same + disturbance compensator (RESID_ENABLE=1)
  drone3 (purple)  end-to-end policy (motor commands; ArduPilot only takes off)

  python demo/run.py race       track with 8 fly-through points, a stop near the end and the finish
  python demo/run.py payload    hover, a box of 65 % of the vehicle mass is released at the same moment
  python demo/run.py wind       hover in strong gusty wind: from the front, calm, from the right, calm
  python demo/run.py final      the track in side wind gusts with a 30 % payload released in flight

Hover: the vehicles stand side by side. Track: all fly the same track one above the other, 2 m apart.

Options: the Gazebo window opens by default; --record video.mp4 (camera in the scene, video in
simulation time with captions, no window needed), --headless, --policy, --ardupilot,
--waypoint-limits sport|default (ArduPilot AUTO limits on the track).

Wind: the Gazebo Iris already has air drag ~1.52 N/(m/s) relative to the ground, so an extra force
1.52 * w(t) gives the drag relative to air moving with the wind w(t). Each blow: mean wind + turbulence
(random, ~2 s correlation, ~20 % of the mean, direction within a few degrees) + a "1 - cos" gust, with
smooth ramps; streaks in the scene move with the wind. All vehicles get the same wind, armed in the
same simulation step.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'gazebo'), str(REPO)]
from disturbances import plugin_path  # noqa: E402
from evaluate_ardupilot import STAND  # noqa: E402
from sitl import mission_items, upload_mission, wait_state  # noqa: E402
from track_metrics import load_track  # noqa: E402
from simulation.interface import StateUnavailable, run_command  # noqa: E402
from simulation.launch import Session, spawn, wsl_prefix  # noqa: E402
from controllers.base import EncoderConfig, ObservationEncoder  # noqa: E402
from experiments.mission import Mission  # noqa: E402
from training.evaluate import Policy  # noqa: E402
from training.physics import MASS, DirectMotorModel  # noqa: E402
from training.wrappers import track_features  # noqa: E402

DT = .02
YAW = math.pi / 2                                    # all vehicles face +y (north)
DRAG = 1.52                                          # N/(m/s), Iris air drag (as in the training model)
DRONES = ('drone1', 'drone2', 'drone3')
NAMES = {'drone1': 'ArduPilot', 'drone2': 'Компенсатор', 'drone3': 'End-to-end'}
COLORS = {'drone1': (.04, .27, .72), 'drone2': (.90, .32, .055), 'drone3': (.45, .12, .68)}
BASE_INERTIA = (0.008, 0.015, 0.017)
FPS = 25


# ------------------------------------------------------------------ wind
def wind(t_on, t_off, mean, heading, gusts=(), rng=None, ramp=2., step=.1):
    """Wind velocity pieces [(on, off, wx, wy)]: one blow of wind. heading: direction the air moves to,
    rad (it wanders by a few degrees); gusts: [(t_peak, extra m/s, duration s)]."""
    rng = rng or np.random.default_rng(0)
    u, d, out = 0., 0., []
    for t in np.arange(t_on, t_off - 1e-9, step):
        u += step / 2. * -u + .22 * mean * math.sqrt(2 * step / 2.) * rng.normal()     # turbulence, 2 s
        d += step / 4. * -d + math.radians(5) * math.sqrt(2 * step / 4.) * rng.normal()  # direction, 4 s
        envelope = min(1., (t - t_on) / ramp, (t_off - t) / ramp)
        envelope = .5 - .5 * math.cos(math.pi * max(0., envelope))
        speed = envelope * (mean + u)
        for t_peak, extra, duration in gusts:
            if abs(t - t_peak) < duration / 2:
                speed += extra * (.5 + .5 * math.cos(2 * math.pi * (t - t_peak) / duration))
        a = heading + d
        out.append((round(t, 2), round(t + step, 2), speed * math.cos(a), speed * math.sin(a)))
    return out


FROM_FRONT, FROM_RIGHT = -math.pi / 2, math.pi        # air moving to -y / to -x (vehicles face +y)

# Demo track: 8 fly-through points, a 2 s stop near the end, finish at the start (x, y, z in m)
TRACK = dict(points=[(0, 0, 3), (7, -2, 3), (14, 1, 3.5), (20, -1, 4), (19, 7, 4), (12, 11, 3.5), (5, 8, 3),
                     (-2, 11, 3.5), (-3, 4, 3), (0, 0, 3)],
             dwell=[.5, 0, 0, 0, 0, 0, 0, 2., 0, .5],
             radii=[.4, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, .4, 2.5, .4])   # fly-through 2.5 m; the stop 0.4 m: a real stop
HEAVY = .65                                          # payload, fraction of the vehicle mass
# ArduPilot AUTO on the track: "sport" waypoint limits instead of the smooth defaults (3 m/s, 2.5 m/s^2,
# jerk 1 m/s^3), within the limits the estimator was trained with
SPORT = dict(WP_SPD=5.0, WP_ACC=4.0, WP_JERK=5.0)
AUTO_STOP_TOLERANCE_S = .1


def scene(name):
    rng = np.random.default_rng(2026)
    if name == 'race':
        return dict(kind='route', wind=[], drop=None, seconds=80., title='Трасса: 8 пролётных точек, остановка, финиш')
    if name == 'payload':
        return dict(kind='hover', spacing=4., wind=[], drop=6., payload=HEAVY, seconds=18.,
                    title=f'Сброс груза {HEAVY * MASS:.2f} кг ({HEAVY:.0%} массы)'.replace('.', ',').replace('%', ' %'))
    if name == 'wind':
        w = (wind(3., 15., 3.5, FROM_FRONT, gusts=((10., 1.5, 3.),), rng=rng)         # from the front, calm,
             + wind(20., 32., 3.5, FROM_RIGHT, gusts=((27., 1.5, 3.),), rng=rng))     # from the right, calm
        return dict(kind='hover', spacing=4., drop=None, seconds=38., wind=w, title='Сильный ветер: спереди, затем справа')
    if name == 'final':
        w = []
        for t_on in (2., 8.):        # side gusts on the first legs, all vehicles still flying; calm at the stop
            w += wind(t_on, t_on + 5.5, 3.5, FROM_RIGHT, gusts=((t_on + 2.75, 1.5, 2.5),), rng=rng, ramp=1.5)
        return dict(kind='route', drop=5., payload=HEAVY, seconds=90., wind=w,
                    title='Общий тест: ветер, груз 65 %, остановка')
    raise ValueError(name)


def track(sc):
    """Points, dwell times and acceptance radii of the scene (hover: the start point of the test track).
    sc['pass_radius'] overrides the radius of the fly-through points."""
    if sc['kind'] == 'route':
        dwell, radii = np.array(TRACK['dwell']), np.array(TRACK['radii'], float)
        if sc.get('pass_radius'):
            radii[1:-1][dwell[1:-1] == 0] = sc['pass_radius']
        return np.array(TRACK['points'], float), dwell, radii
    return load_track()


ROUTE_LEVEL = {'drone1': 4., 'drone2': 2., 'drone3': 0.}   # m above the track


def lane(sc, i, name):
    """Where the vehicle stands and how its points are shifted. Hover: side by side. Route: all fly the
    same square, one above the other (2 m apart), so it is easy to see who is ahead."""
    if sc['kind'] == 'hover':
        return (i - 1) * sc['spacing'], np.zeros(3)
    x0 = (i - 1) * 2.
    return x0, np.array([-x0, 0., ROUTE_LEVEL[name]])


# ------------------------------------------------------------------ world
def build_world(out, sc, prefix, record):
    tree = ET.parse(REPO / 'simulation/world.sdf')
    world = tree.getroot().find('world')
    for m in list(world.findall('model')):
        if m.attrib['name'].startswith('target'):
            world.remove(m)
    plugin = plugin_path(prefix)
    points, dwell = track(sc)[:2]
    stops = [k for k, s in enumerate(dwell) if s > 1.] if sc['kind'] == 'route' else []
    route = sc['kind'] == 'route'
    area = ((-7, -5, 1.5), (24, 15, 9.)) if route else ((-8, -4, 1.5), (8, 6, 5.))
    for i, name in enumerate(DRONES):
        x0, shift = lane(sc, i, name)
        drone = next(m for m in world.findall('model') if m.attrib['name'] == name)
        drone.find('pose').text = f'{x0} 0 .195 0 0 {YAW}'
        p = ET.SubElement(world, 'plugin', filename=plugin, name='copter_disturbance::DisturbanceSystem')
        for tag, value in (('model_name', name), ('link_name', 'base_link'), ('topic', f'/demo/{name}/disturbance'),
                           ('detach_topic', f'/demo/{name}/detach'), ('log_file', str((out / f'{name}.jsonl').resolve()))):
            ET.SubElement(p, tag).text = value
        if i == 0 and sc['wind']:
            w = ET.SubElement(p, 'wind_streaks')
            for tag, value in (('prefix', 'wind_streak_'), ('count', '70' if route else '45'),
                               ('area_min', ' '.join(map(str, area[0]))), ('area_max', ' '.join(map(str, area[1]))),
                               ('drag_N_per_mps', str(DRAG))):
                ET.SubElement(w, tag).text = value
        if sc['drop'] is not None:
            joint = ET.SubElement(drone.find('include'), 'plugin', filename='gz-sim-detachable-joint-system',
                                  name='gz::sim::systems::DetachableJoint')
            for tag, value in (('parent_link', 'base_link'), ('child_model', f'payload_{name}'), ('child_link', 'box'),
                               ('detach_topic', f'/demo/{name}/detach')):
                ET.SubElement(joint, tag).text = value
            world.append(box_model(f'payload_{name}', (x0, 0., .095), sc['payload']))
        if route:                            # a coloured ball over the vehicle: visible from far away
            joint = ET.SubElement(drone.find('include'), 'plugin', filename='gz-sim-detachable-joint-system',
                                  name='gz::sim::systems::DetachableJoint')
            for tag, value in (('parent_link', 'base_link'), ('child_model', f'marker_{name}'), ('child_link', 'ball'),
                               ('detach_topic', f'/demo/{name}/keep_marker')):
                ET.SubElement(joint, tag).text = value
            world.append(marker_model(f'marker_{name}', (x0, 0., .42), COLORS[name]))
        marks = points if route else points[:1]
        world.append(points_model(f'points_{name}', [(x0 + shift[0] + q[0], shift[1] + q[1], shift[2] + q[2])
                                                     for q in marks], COLORS[name], stops,
                                  *(() if route else (.1, .7))))
    for k in range(70 if route else 45):
        world.append(streak_model(f'wind_streak_{k}', 1.4 if route else .8))
    if record:
        world.append(ET.fromstring('<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">'
                                   '<render_engine>ogre2</render_engine></plugin>'))
        pose = '9 -17 15 0 .47 1.5708' if route else '0 -7.5 3.7 0 .07 1.5708'
        world.append(camera_model(pose, 1.25 if route else 1.15, record))
    path = out / 'world_demo.sdf'
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return path


def material(color, alpha=1., emissive=0.):
    c = ' '.join(map(str, color))
    e = ' '.join(str(emissive * x) for x in color)
    return f'<material><ambient>{c} {alpha}</ambient><diffuse>{c} {alpha}</diffuse><emissive>{e} 1</emissive></material>'


def box_model(name, xyz, fraction):
    """Payload of `fraction` of the vehicle mass. The visible box grows with the mass; the collision box
    stays small and touches only the ground (collide_bitmask: vehicles use bits 0-7), so a released box
    never hits a vehicle flying below."""
    i = ''.join(f'<{k}>{fraction * v!r}</{k}>' for k, v in zip(('ixx', 'iyy', 'izz'), BASE_INERTIA))
    side = .12 * (fraction / .2) ** (1 / 3) * 1.4
    shape = f'<geometry><box><size>{side:.3f} {side:.3f} {side / 2:.3f}</size></box></geometry>'
    return ET.fromstring(
        f'<model name="{name}"><pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose><link name="box">'
        f'<inertial><mass>{fraction * MASS!r}</mass><inertia>{i}<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>'
        f'<collision name="c"><geometry><box><size>.12 .12 .06</size></box></geometry>'
        f'<surface><contact><collide_bitmask>0x0100</collide_bitmask></contact></surface></collision>'
        f'<visual name="v">{shape}{material((.95, .78, .1))}</visual></link></model>')


def marker_model(name, xyz, color):
    """A 5 g ball fixed above the vehicle (no collision): colour mark for the camera."""
    return ET.fromstring(
        f'<model name="{name}"><pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose><link name="ball">'
        f'<inertial><mass>.005</mass><inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz></inertia></inertial>'
        f'<visual name="v"><geometry><sphere><radius>.16</radius></sphere></geometry>{material(color, 1., .8)}'
        f'</visual></link></model>')


def points_model(name, positions, color, stops=(), size=.17, transparency=.25):
    """Track points (stops are bigger) joined by thin lines."""
    links = ''.join(f'<link name="p{k}"><pose>{x} {y} {z} 0 0 0</pose><visual name="v"><geometry><sphere>'
                    f'<radius>{.3 if k in stops else size}</radius></sphere></geometry>{material(color, 1., .5)}'
                    f'<transparency>{transparency}</transparency></visual></link>' for k, (x, y, z) in enumerate(positions))
    for k, (a, b) in enumerate(zip(positions[:-1], positions[1:])):
        d = np.subtract(b, a)
        mid, length = np.add(a, b) / 2, float(np.linalg.norm(d))
        pitch, yaw = math.atan2(math.hypot(d[0], d[1]), d[2]), math.atan2(d[1], d[0])
        links += (f'<link name="s{k}"><pose>{mid[0]} {mid[1]} {mid[2]} 0 {pitch} {yaw}</pose><visual name="v">'
                  f'<geometry><cylinder><radius>.025</radius><length>{length}</length></cylinder></geometry>'
                  f'{material(color, 1., .5)}<transparency>.45</transparency></visual></link>')
    return ET.fromstring(f'<model name="{name}"><static>true</static>{links}</model>')


def streak_model(name, length):
    return ET.fromstring(
        f'<model name="{name}"><pose>0 0 -20 0 0 0</pose><link name="l"><gravity>false</gravity>'
        f'<inertial><mass>.001</mass><inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz></inertia></inertial>'
        f'<visual name="v"><geometry><cylinder><radius>.018</radius><length>{length}</length></cylinder></geometry>'
        f'{material((.55, .8, 1.), .85, .6)}<transparency>.15</transparency></visual></link></model>')


def camera_model(pose, fov, record):
    return ET.fromstring(
        f'<model name="demo_camera"><static>true</static><pose>{pose}</pose><link name="l">'
        f'<sensor name="cam" type="camera"><always_on>1</always_on><update_rate>{FPS}</update_rate><camera>'
        f'<horizontal_fov>{fov}</horizontal_fov><image><width>1280</width><height>720</height></image>'
        f'<clip><near>.1</near><far>300</far></clip>'
        f'</camera></sensor></link></model>')


# ------------------------------------------------------------------ flight
def run(name, a):
    sc = scene(name)
    if a.pass_radius:
        sc['pass_radius'] = a.pass_radius
        sc['title'] = f'Трасса: пролётные точки с радиусом {a.pass_radius:g} м'.replace('.', ',')
    if a.payload is not None and sc['drop'] is not None:
        sc['payload'] = a.payload
        sc['title'] = sc['title'].replace('65 %', f'{a.payload:.0%}'.replace('%', ' %'))
    out = Path(a.out or REPO / 'runs/demo' / f'{name}_{time.strftime("%m%d_%H%M%S")}').resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    prefix = wsl_prefix(a.gazebo_distro)
    tag = f'/tmp/copter_demo_{name}_{int(time.time())}'
    record = dict(raw=tag + '.mp4', stamps=tag + '.txt', stop=tag + '.stop') if a.record else None
    world = build_world(out, sc, prefix, record)
    points, dwell, radii = track(sc)
    cfg = json.loads((REPO / 'configs/e2e_eval.json').read_text())
    policy, motors = Policy(a.policy), DirectMotorModel()
    lines = [f'force {on} {off} {DRAG * wx:.3f} {DRAG * wy:.3f} 0' for on, off, wx, wy in sc['wind']]
    if sc['drop'] is not None:
        lines.append(f'detach {sc["drop"]}')
    log = dict(scene=name, title=sc['title'], wind=sc['wind'], drop=sc['drop'], finish={}, crash={}, samples=[],
               hover_point=[float(x) for x in points[0]] if sc['kind'] == 'hover' else None)
    with Session(out / 'simulation', gazebo_prefix=prefix, ardupilot=a.ardupilot, world=world,
                 gui=not a.record and not a.headless, software_rendering=True, offscreen_rendering=bool(a.record)) as session:
        recorder = start_recorder(session, prefix, record) if record else None
        v = session.clients
        origin, start, climb = {}, {}, {}
        for i, n in enumerate(DRONES):
            for p, value in STAND.items():
                v[n].set_parameter(p, value)
            v[n].set_parameter('RESID_ENABLE', 1. if n == 'drone2' else 0.)
            if sc['kind'] == 'route':
                # the same acceptance radius for everybody: ArduPilot switches AUTO waypoints with it,
                # the end-to-end policy and the finish check use the track radii
                v[n].set_parameter('WP_RADIUS_M', float(max(radii[1:-1])))
                if a.waypoint_limits == 'sport':
                    for p, value in SPORT.items():
                        v[n].set_parameter(p, value)
            shift = lane(sc, i, n)[1]
            origin[n] = wait_state(v[n]).position - np.array([0., 0., .195]) + shift
            start[n] = origin[n] + points[0]
            climb[n] = start[n] - np.array([shift[0], shift[1], 0.])    # own height above the take-off spot
        for n in DRONES:
            v[n].arm_and_takeoff(3.)
        settle(v, climb)                     # first climb, then move: the paths to the start never cross
        settle(v, start)
        auto = sc['kind'] == 'route'          # ArduPilot flies a known track as an AUTO mission
        if auto:
            for n in ('drone1', 'drone2'):
                upload_mission(v[n], mission_items(v[n], origin[n], points, dwell))
            settle(v, start)

        # ArduPilot times an AUTO stop itself (exactly the hold time); 0.1 s tolerance for the 50 Hz check
        auto_dwell = np.where(dwell > 1., dwell - AUTO_STOP_TOLERANCE_S, dwell)
        missions = {n: Mission(sc['kind'], .4, points=points, dwell_s=auto_dwell if auto and n != 'drone3' else dwell,
                               radii_m=radii) for n in DRONES}
        encoder = ObservationEncoder(EncoderConfig(**cfg['observation']))
        encoder.reset(YAW)
        e2e = v['drone3']
        e2e.pause_control()                  # from now on the whole world advances in drone3's 20 ms steps
        text = '\\n'.join(lines) if lines else 'force 0 0 0 0 0'
        for _ in range(2):                   # resent while held: all three are armed in the same step
            for n in DRONES:
                session.world.publish(f'/demo/{n}/disturbance', text)
            if auto:
                for n in ('drone1', 'drone2'):
                    v[n].set_mode(3)          # AUTO, taken in the same step as the disturbances
        states = {n: v[n].state(require_new=False, max_age_s=None) for n in DRONES}   # the world is held
        t0 = states['drone3'].time_s
        for n in DRONES:
            missions[n].update(0., states[n].position - origin[n])
        print(f'{name}: go', flush=True)
        prev = np.zeros(4)
        while True:
            t = states['drone3'].time_s - t0
            if not auto:
                for n in ('drone1', 'drone2'):
                    v[n].set_position(origin[n] + missions[n].target, YAW)
            s = states['drone3']
            p = s.position - origin['drone3']
            obs = encoder.observe(p, s.velocity, s.rotation, s.body_rates, missions['drone3'].target, prev, DT)
            obs = np.r_[obs, track_features(missions['drone3'], p, s.rotation)]
            prev = policy(obs.astype(np.float32).astype(np.float64))
            e2e.set_motor_pwm(motors.action_to_pwm(prev))
            states = {'drone3': step(e2e)}
            states.update({n: latest(v[n]) for n in ('drone1', 'drone2')})
            log['samples'].append([round(t, 3)] + [round(float(x), 3) for n in DRONES
                                                   for x in states[n].position - origin[n]])
            for n in DRONES:
                missions[n].update(states[n].time_s - t0, states[n].position - origin[n])
                if missions[n].completion_time is not None and NAMES[n] not in log['finish']:
                    log['finish'][NAMES[n]] = round(missions[n].completion_time, 2)
                    print(f'  {NAMES[n]}: finish {missions[n].completion_time:.2f} s', flush=True)
                height = states[n].position[2] - origin[n][2] + lane(sc, DRONES.index(n), n)[1][2]
                if t > 2. and height < .4 and NAMES[n] not in log['finish'] and NAMES[n] not in log['crash']:
                    log['crash'][NAMES[n]] = round(t, 2)       # on the ground away from the finish
                    print(f'  {NAMES[n]}: lost control at {t:.2f} s', flush=True)
            ended = {**log['finish'], **log['crash']}
            if t >= sc['seconds'] or (sc['kind'] == 'route' and len(ended) == 3 and t > max(ended.values()) + 2.):
                break
        e2e.set_position(origin['drone3'] + missions['drone3'].target, YAW)
        e2e.resume_control()
        end = time.monotonic() + (0 if a.record else a.hold)
        while time.monotonic() < end:
            for n in DRONES:
                v[n].set_position(origin[n] + missions[n].target, YAW)
            time.sleep(.1)
        armed = next(json.loads(x)['sim_time'] for x in (out / 'drone1.jsonl').read_text().splitlines()
                     if 'schedule_armed' in x)
        log['armed_sim_time'] = armed
        if recorder is not None:
            run_command([*prefix, 'touch', record['stop']], check=True)
            recorder.wait(timeout=120)
    (out / 'demo.json').write_text(json.dumps(log, ensure_ascii=False))
    print('result:', log['finish'] or 'hover', flush=True)
    if a.record:
        make_video(log, record, Path(a.record).resolve(), prefix, out)


def step(vehicle, tries=5):
    """Advance the held world by one 20 ms step and return the new state. Under heavy load SITL
    sometimes sends no telemetry within a step; then the same motor command is held for one more step."""
    for _ in range(tries):
        vehicle.step_control(DT)
        try:
            return wait_state(vehicle, timeout=3.)
        except StateUnavailable:
            pass
    raise StateUnavailable(f'no telemetry after {tries} steps')


def latest(vehicle, timeout=10.):
    """Most recent state (ArduPilot vehicles are only observed; a 20 ms step may bring no new pair)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return vehicle.state(require_new=False, max_age_s=None)   # drone3 holds the world
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(.002)


def settle(v, start, timeout=90.):
    """Hold every vehicle at its start point until all are steady for 2 s."""
    deadline, stable = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        ok = True
        for n, c in v.items():
            s = wait_state(c)
            c.set_position(start[n], YAW)
            ok &= bool(np.linalg.norm(s.position - start[n]) <= .05 and np.linalg.norm(s.velocity) <= .08)
        now = time.monotonic()
        stable = (stable or now) if ok else None
        if stable and now - stable >= 2.:
            return
    raise TimeoutError('vehicles did not settle at the start')


# ------------------------------------------------------------------ video
def ass_time(t):
    t = max(0., t)
    return f'{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}'


def captions(log, lead):
    """ASS subtitles: title, vehicle labels, timer, wind speed, payload release, finish times."""
    def bgr(c):
        r, g, b = (int(255 * min(1., x * 1.3)) for x in c)
        return f'&H{b:02X}{g:02X}{r:02X}&'
    end = log['samples'][-1][0] + 1.
    ev = [f'Dialogue: 0,{ass_time(0)},{ass_time(end + lead)},Top,,0,0,0,,{log["title"]}']
    for i, n in enumerate(DRONES):
        x = (250, 640, 1030)[i]
        ev.append(f'Dialogue: 0,{ass_time(0)},{ass_time(end + lead)},Label,,0,0,0,,'
                  f'{{\\pos({x},690)\\c{bgr(COLORS[n])}}}{NAMES[n]}')
    for k in np.arange(0., end, .1):
        ev.append(f'Dialogue: 0,{ass_time(k + lead)},{ass_time(k + lead + .1)},Timer,,0,0,0,,t = {k:4.1f} с')
    pieces = {round(on, 2): (wx, wy) for on, off, wx, wy in log['wind']}
    for k in np.arange(0., end, .5):
        w = pieces.get(round(round(k / .1) * .1, 2))
        if w and math.hypot(*w) > .3:
            arrow = '→↗↑↖←↙↓↘'[int(round(math.atan2(w[1], w[0]) / (math.pi / 4))) % 8]
            ev.append(f'Dialogue: 0,{ass_time(k + lead)},{ass_time(k + lead + .5)},Wind,,0,0,0,,'
                      'Ветер ' + f'{math.hypot(*w):.1f}'.replace('.', ',') + f' м/с {arrow}')
    if log.get('hover_point') is not None:       # hover: distance to the point under every vehicle
        s = np.array(log['samples'])
        for i, n in enumerate(DRONES):
            x = (250, 640, 1030)[i]
            e = 100. * np.linalg.norm(s[:, 1 + 3 * i:4 + 3 * i] - log['hover_point'], axis=1)
            for k in np.arange(0., end - 4., .1):
                j = min(int(np.searchsorted(s[:, 0], k)), len(e) - 1)
                ev.append(f'Dialogue: 0,{ass_time(k + lead)},{ass_time(k + lead + .1)},Label,,0,0,0,,'
                          f'{{\\pos({x},645)\\fs26\\3c&H000000&}}отклонение {e[j]:.0f} см')
            after = e[s[:, 0] > 2.]
            if log['wind']:      # wind: mean deviation; payload release: the largest one
                summary = f'в среднем {math.sqrt(float(np.mean(after ** 2))):.1f} см'
            else:
                summary = f'максимум {float(after.max()):.1f} см'
            ev.append(f'Dialogue: 1,{ass_time(end - 4. + lead)},{ass_time(end + lead)},Label,,0,0,0,,'
                      f'{{\\pos({x},645)\\c{bgr(COLORS[n])}}}' + summary.replace('.', ','))
    if log['drop'] is not None:
        ev.append(f'Dialogue: 0,{ass_time(log["drop"] + lead)},{ass_time(log["drop"] + lead + 2.5)},Event,,0,0,0,,'
                  'Сброс груза')
    for i, n in enumerate(DRONES):
        f = log['finish'].get(NAMES[n])
        if f is not None:
            x = (250, 640, 1030)[i]
            ev.append(f'Dialogue: 1,{ass_time(f + lead)},{ass_time(end + lead)},Label,,0,0,0,,'
                      f'{{\\pos({x},640)\\c{bgr(COLORS[n])}}}финиш ' + f'{f:.1f} с'.replace('.', ','))
        f = log.get('crash', {}).get(NAMES[n])
        if f is not None:
            x = (250, 640, 1030)[i]
            ev.append(f'Dialogue: 1,{ass_time(f + lead)},{ass_time(end + lead)},Label,,0,0,0,,'
                      f'{{\\pos({x},640)\\c{bgr(COLORS[n])}}}потеря управления')
    head = ('[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n\n[V4+ Styles]\n'
            'Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, '
            'Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n'
            'Style: Top,DejaVu Sans,30,&H00FFFFFF,&H00000000,&H80000000,1,1,2,0,8,20,20,18\n'
            'Style: Label,DejaVu Sans,30,&H00FFFFFF,&H00FFFFFF,&H80000000,1,1,2,0,2,0,0,0\n'
            'Style: Timer,DejaVu Sans Mono,26,&H00FFFFFF,&H00000000,&H80000000,0,1,2,0,9,20,20,18\n'
            'Style: Wind,DejaVu Sans,30,&H00FFD38C,&H00000000,&H80000000,1,1,2,0,7,20,20,18\n'
            'Style: Event,DejaVu Sans,34,&H0040D0FF,&H00000000,&H80000000,1,1,2,0,8,20,20,70\n\n'
            '[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n')
    return head + '\n'.join(ev) + '\n'


def start_recorder(session, prefix, record):
    """Record the scene camera where Gazebo runs (demo/camera_recorder.py)."""
    topic = f'/world/{session.world_name}/model/demo_camera/link/l/sensor/cam/image'
    script = str(Path(__file__).resolve().parent / 'camera_recorder.py')
    return spawn([*prefix, 'env', f'GZ_PARTITION={session.partition}', 'python3', script, topic,
                  record['raw'], record['stamps'], record['stop']], stdin=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_video(log, record, target, prefix, out, lead=1.):
    """Cut the camera video from `lead` s before the start to the end and burn in the captions."""
    ass = out / 'captions.ass'
    ass.write_text(captions(log, lead), encoding='utf-8')
    work = record['raw'][:-4]
    stamps = run_command([*prefix, 'cat', record['stamps']], capture_output=True, text=True, check=True).stdout.split()
    first = sum(1 for s in stamps if float(s) < log['armed_sim_time'] - lead)
    last = first + int(round((log['samples'][-1][0] + 1. + lead) * FPS))
    (out / 'frame_stamps.txt').write_text('\n'.join(stamps))
    script = (f'set -e; cp "{ass}" {work}.ass; '
              # a silent audio track: messengers send video without sound as an animation (GIF)
              f'ffmpeg -y -loglevel error -i {record["raw"]} -f lavfi -i anullsrc=r=44100:cl=stereo '
              f'-vf "trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS,subtitles={work}.ass" '
              f'-map 0:v -map 1:a -c:v libx264 -pix_fmt yuv420p -crf 20 -c:a aac -b:a 32k -shortest '
              f'-movflags +faststart {work}_final.mp4; '
              f'cp {work}_final.mp4 "{target}"; rm -f {work}*')
    target.parent.mkdir(parents=True, exist_ok=True)
    run_command([*prefix, 'bash', '-c', script], check=True)
    print('video:', target, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scene', choices=('race', 'payload', 'wind', 'final'))
    ap.add_argument('--record', help='write a video (mp4) instead of opening the Gazebo window')
    ap.add_argument('--headless', action='store_true', help='no window, no video (check the scene only)')
    ap.add_argument('--policy', default=str(REPO / 'models/e2e_policy.npz'))
    ap.add_argument('--ardupilot', type=Path, default=None)
    ap.add_argument('--gazebo-distro', default='Ubuntu')
    ap.add_argument('--hold', type=float, default=5., help='seconds the window stays after the run')
    ap.add_argument('--waypoint-limits', choices=('sport', 'default'), default='sport',
                    help='ArduPilot AUTO limits on the track: sport (5 m/s, 4 m/s^2, jerk 5) or the smooth defaults')
    ap.add_argument('--payload', type=float, help='payload, fraction of the vehicle mass (payload and final scenes)')
    ap.add_argument('--pass-radius', type=float, help='acceptance radius of the fly-through points, m (default 2.5)')
    ap.add_argument('--out')
    a = ap.parse_args()
    run(a.scene, a)


if __name__ == '__main__':
    main()
