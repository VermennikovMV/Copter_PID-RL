"""Prospective development reward, separate from the five evaluation metrics.

Version precision-arrival-v3 was introduced after the three initial V2 3M-step
pilots exhibited route stalling. It uses only public task state and actions.
"""
import math

REWARD_VERSION="precision-arrival-v3"


def reward_value(error,velocity_sq,tilt_cos,yaw_error,rate_sq,action_delta_sq,
                 distance_progress,arrival,completion,failed,dt=.02,
                 variant="full",version=REWARD_VERSION):
    if version=="initial-v2":
        cost=2.5*error**2+.08*velocity_sq+.15*(1-tilt_cos)+5*(1-math.cos(yaw_error))+.003*rate_sq
        if variant!="no_smooth":cost+=.008*action_delta_sq
        value=dt*(2-cost)+.8*distance_progress+2*arrival+5*completion
    elif version==REWARD_VERSION:
        # A point just outside the arrival band is substantially less valuable
        # than precise tracking; confirmed progress earns an explicit bonus.
        rate=1.5*math.exp(-(error/.2)**2)-.35*error**2-.04*velocity_sq
        rate-=10*(1-math.cos(yaw_error))+.05*(1-tilt_cos)+.003*rate_sq
        if variant!="no_smooth":rate-=.008*action_delta_sq
        value=dt*rate+.8*distance_progress+10*arrival+20*completion
    else:raise ValueError(f"Unknown reward version: {version}")
    return float(value-20*failed)
