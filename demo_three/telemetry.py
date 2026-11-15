"""Public MAV helpers derived from v8 runtime825edf2d, with raw source-ID logging."""
import json,math,time
from pathlib import Path
import numpy as np
from pymavlink import mavutil
ENU_FROM_NED=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
FRD_FROM_FLU=np.diag([1.,-1.,-1.])

def rotation(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp,cp*sr,cp*cr]])

class Telemetry:
    def __init__(self, conn, output, stop_file=None):
        self.conn = conn
        self.raw = open(output/'mavlink.jsonl', 'w', buffering=1)
        self.events = open(output/'events.jsonl', 'w', buffering=1)
        self.latest = {}
        self.boot_s = 0.
        self.n = 0
        self.last_recv = time.monotonic()
        self.last_hb = 0.
        self.received = {}
        self.message_indices = {}
        self.stop_file=Path(stop_file) if stop_file else None

    def check_stop(self):
        if self.stop_file is not None and self.stop_file.exists():
            raise InterruptedError('Owned STOP_DEMO requested')

    def event(self, event, **fields):
        item = {'event':event, 'wall_time':time.time(), 'boot_s':self.boot_s, **fields}
        self.events.write(json.dumps(item,default=str)+'\n')
        # Full evidence stays in events.jsonl; runtime prints concise live status.

    def drain(self):
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None: break
            if msg.get_type() == 'BAD_DATA': continue
            now = time.monotonic()
            self.last_recv = now
            self.n += 1
            self.latest[msg.get_type()] = msg
            self.received[msg.get_type()] = now
            self.message_indices[msg.get_type()] = self.n
            if hasattr(msg,'time_boot_ms'): self.boot_s=max(self.boot_s,msg.time_boot_ms/1000.)
            self.raw.write(json.dumps({'wall_time':time.time(),'message':msg.to_dict(),'record_index':self.n,'receive_monotonic_s':now,
                'source_system':msg.get_srcSystem(),'source_component':msg.get_srcComponent()},default=str)+'\n')
        if time.monotonic()-self.last_hb > 1:
            self.conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,0)
            self.last_hb=time.monotonic()

    def wait(self, predicate, timeout, description):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            self.check_stop()
            self.drain()
            if predicate(): return
            time.sleep(.01)
        raise TimeoutError(description)

    def param(self, name, value=None):
        for _ in range(4):
            self.check_stop()
            self.latest.pop('PARAM_VALUE',None)
            if value is None:
                self.conn.mav.param_request_read_send(self.conn.target_system,self.conn.target_component,name.encode(),-1)
            else:
                self.conn.mav.param_set_send(self.conn.target_system,self.conn.target_component,name.encode(),float(value),mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            end=time.monotonic()+1.5
            while time.monotonic()<end:
                self.check_stop()
                self.drain()
                m=self.latest.get('PARAM_VALUE')
                if m and m.param_id.strip('\x00')==name and (value is None or abs(m.param_value-value)<1e-4):
                    return float(m.param_value)
                time.sleep(.005)
        raise RuntimeError('Missing parameter readback '+name)

    def state(self):
        p=self.latest.get('LOCAL_POSITION_NED'); att=self.latest.get('ATTITUDE')
        if p is None or att is None: raise RuntimeError('Missing position or attitude')
        pos=ENU_FROM_NED@np.array([p.x,p.y,p.z])
        vel=ENU_FROM_NED@np.array([p.vx,p.vy,p.vz])
        R=ENU_FROM_NED@rotation(att.roll,att.pitch,att.yaw)@FRD_FROM_FLU
        omega=np.array([att.rollspeed,-att.pitchspeed,-att.yawspeed])
        return pos,vel,R,omega

    def close(self):
        self.raw.close(); self.events.close()

def command(conn, cmd, *values):
    values=list(values)+[0.]*(7-len(values))
    conn.mav.command_long_send(conn.target_system,conn.target_component,cmd,0,*values)

def send_position(conn, target_enu, boot_s, yaw_enu=None):
    x,y,z=ENU_FROM_NED@target_enu
    mask=3576 if yaw_enu is None else 2552
    yaw=0 if yaw_enu is None else math.pi/2-yaw_enu
    conn.mav.set_position_target_local_ned_send(int(boot_s*1000)&0xffffffff,conn.target_system,conn.target_component,mavutil.mavlink.MAV_FRAME_LOCAL_NED,mask,x,y,z,0,0,0,0,0,0,yaw,0)
