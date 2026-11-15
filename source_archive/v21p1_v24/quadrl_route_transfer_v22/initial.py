"""Pure initial-state coordinate transforms; importing creates no environment."""
import numpy as np


def rotation(rpy):
    r,p,y=map(float,rpy); cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def euler(matrix):
    R=np.asarray(matrix,float)
    assert R.shape==(3,3) and np.max(abs(R.T@R-np.eye(3)))<1e-10 and abs(np.linalg.det(R)-1)<1e-10
    return np.array([np.arctan2(R[2,1],R[2,2]),np.arcsin(np.clip(-R[2,0],-1,1)),np.arctan2(R[1,0],R[0,0])])


def target_initial(native, measured, intervention):
    """Only declared measured state is injected; no plant hidden state is claimed."""
    if intervention=='yaw_only':
        rpy=np.asarray(native['rpy'],float).copy();rpy[2]=measured['rpy'][2]
        return dict(position=list(native['position']),velocity=list(native['velocity']),rpy=rpy.tolist(),body_rates=[0.,0.,0.])
    if intervention in ('full_initial','full_initial_physical_pulse'):
        return {name:list(measured[name]) for name in ('position','velocity','rpy','body_rates')}
    raise ValueError('Fixed yaw_only/full_initial/full_initial_physical_pulse intervention required')
