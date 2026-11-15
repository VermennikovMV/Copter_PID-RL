"""Explicit body-collective to ArduCopter GUIDED preboost conversion.

Source: AC_AttitudeControl_Multi::get_throttle_boosted and
AC_AttitudeControl::thrust_vector_rotation_angles. Despite the local variable
name ``cos_tilt_target``, ``_thrust_angle_rad`` is the CURRENT BODY lean angle.
Only the measured rotation already supplied to the actor is used here. This
adapter cannot establish equivalence of motor dynamics or guarantee tracking.
"""
from dataclasses import asdict, dataclass
import math

INTERFACE_VERSION = 'joint-hybrid-body-collective-v2.0'
SEMANTICS = ('legacy_preboost', 'body_collective')


class ThrustAdapterError(ValueError):
    """No command may be sent for invalid or stale compensation inputs."""


@dataclass(frozen=True)
class ThrustMapping:
    desired_body_thrust: float
    preboost_thrust: float
    measured_cos: float
    denominator: float
    inverted_factor: float
    inverse_factor: float
    unclipped_preboost_thrust: float
    clipped: bool
    attitude_receive_age_s: float

    def to_dict(self):
        return asdict(self)


def body_collective_to_preboost(desired_body_thrust, measured_cos,
                               attitude_receive_age_s, max_age_s=.1):
    """Invert native angle boost using public measured body R[2,2].

    ``preboost * clip(10*Rzz,0,1) / clip(Rzz,.1,1)`` is the native
    postboost throttle. Zero inverted factor has no inverse, so fails closed.
    Saturation is explicit: in the fading regime it can prevent equality of
    the desired and native throttle. Flight envelope Rzz>=.2 avoids that fade.
    """
    values = (desired_body_thrust, measured_cos, attitude_receive_age_s, max_age_s)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ThrustAdapterError('Nonfinite or missing public thrust/attitude input')
    if not 0 <= desired_body_thrust <= 1:
        raise ThrustAdapterError('Desired body collective outside [0,1]')
    if not 0 < max_age_s <= .1:
        raise ThrustAdapterError('Attitude age limit must be in (0,0.1] seconds')
    if not 0 <= attitude_receive_age_s <= max_age_s:
        raise ThrustAdapterError('Measured attitude is stale or its receive age is invalid')
    if not -1.000000001 <= measured_cos <= 1.000000001:
        raise ThrustAdapterError('Measured rotation cosine outside [-1,1]')
    measured_cos = min(1., max(-1., measured_cos))
    denominator = min(1., max(.1, measured_cos))
    inverted_factor = min(1., max(0., 10. * measured_cos))
    if inverted_factor <= 0:
        raise ThrustAdapterError('Native inverted factor is zero; body collective has no inverse')
    inverse_factor = denominator / inverted_factor
    unclipped = desired_body_thrust * inverse_factor
    preboost = min(1., max(0., unclipped))
    return ThrustMapping(float(desired_body_thrust), preboost, measured_cos,
                         denominator, inverted_factor, inverse_factor,
                         unclipped, preboost != unclipped, float(attitude_receive_age_s))
