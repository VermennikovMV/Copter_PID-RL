"""Evaluation scenarios: disturbance force schedules and vehicle changes.

Forces are external forces on the airframe in NEWTONS (ENU), not wind speeds. The schedule of a
scenario is a deterministic function of (scenario, seed), so every controller meets the same
disturbances. Controllers never see these parameters; they only see the vehicle state and target.
"""
from functools import lru_cache

import numpy as np

CONTROL_HZ = 50
CONTROL_DT = 1 / CONTROL_HZ
ROUTE_DWELL_S = .50
ROUTE_POINTS = np.array([[0, 0, 3], [2, 0, 3], [2, 2, 3.5], [0, 2, 3], [0, 0, 3]], dtype=float)
SCENARIOS = {
    'nominal': {},
    'force_mild': {'force_profile': 'force_mild'},        # constant 1.5 N from 5 to 20 s
    'force_strong': {'force_profile': 'force_strong'},    # constant 3 N from 5 to 20 s
    'gust': {'force_profile': 'gust'},                    # three 3 N gusts of 3 s: forward, reverse, sideways
    'mass_20': {'mass_scale': 1.2},
    'mass_40': {'mass_scale': 1.4},
    'motor_20': {'motor_scale': .8, 'motor_index': 0},
    'motor_40': {'motor_scale': .6, 'motor_index': 0},
    'delay_40': {'delay_steps': 2},
    'delay_100': {'delay_steps': 5},
    'sensor_noise': {'noise_position': .03, 'noise_velocity': .06, 'noise_orientation': .01, 'noise_rates': .03},
}


def seeded_rng(seed, stream):
    """Independent random stream `stream` of episode `seed`."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


@lru_cache(maxsize=4096)
def _events(profile, seed):
    if profile not in SCENARIOS:
        raise ValueError(f'unknown disturbance profile {profile}')
    if profile not in ('force_mild', 'force_strong', 'gust'):
        return ()
    rng = seeded_rng(seed, 110)
    angle = rng.uniform(-np.pi, np.pi)
    magnitude = (1.5 if profile == 'force_mild' else 3.) * rng.uniform(.9, 1.1)
    direction = np.array([np.cos(angle), np.sin(angle), 0.])
    jitter = rng.integers(-12, 13, size=6) * CONTROL_DT      # +-0.24 s on the 50 Hz grid
    if profile == 'gust':
        specs = [(5 + jitter[0], 8 + jitter[1], direction),
                 (11 + jitter[2], 14 + jitter[3], -direction),
                 (17 + jitter[4], 20 + jitter[5], np.array([-direction[1], direction[0], 0.]))]
    else:
        specs = [(5 + jitter[0], 20 + jitter[5], direction)]
    return tuple((round(float(on), 2), round(float(off), 2), tuple((magnitude * vector).tolist()))
                 for on, off, vector in specs)


def force_events(profile, seed=0):
    """[{on_s, off_s, force_enu_N}] of a profile name, or the 'force_events' of a parameter dict."""
    if isinstance(profile, dict):
        return [dict(e) for e in profile.get('force_events', [])]
    return [{'on_s': on, 'off_s': off, 'force_enu_N': list(vector)} for on, off, vector in _events(profile, int(seed))]


def force_N(t, profile='nominal', seed=0):
    """External force at time t (sampled on the 50 Hz control grid)."""
    sampled = np.floor((float(t) + 1e-9) * CONTROL_HZ) / CONTROL_HZ
    result = np.zeros(3)
    for event in force_events(profile, seed):
        if event['on_s'] - 1e-9 <= sampled < event['off_s'] - 1e-9:
            result += np.asarray(event['force_enu_N'], dtype=float)
    return result


def scenario_spec(name, seed):
    if name not in SCENARIOS:
        raise ValueError(f'unknown scenario {name}')
    spec = {'name': name, 'seed': int(seed), **SCENARIOS[name]}
    spec['force_events'] = force_events(spec.get('force_profile', 'nominal'), seed)
    return spec
