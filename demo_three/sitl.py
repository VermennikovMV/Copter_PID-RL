"""Owned SITL/capture helpers for a single physical three-aircraft world."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

VEHICLES=(
    dict(role='pid',model='iris_pid',instance=0,tcp_port=5760,physics_port=9002,sitl_port=9002,offset_enu=[-4.,0.,0.]),
    dict(role='hybrid',model='iris_hybrid',instance=1,tcp_port=5770,physics_port=9012,sitl_port=9012,offset_enu=[0.,0.,0.]),
    dict(role='direct',model='iris_direct',instance=2,tcp_port=5780,physics_port=9022,sitl_port=9032,offset_enu=[4.,0.,0.]))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');temporary.replace(path)
def check_ports():
    for vehicle in VEHICLES:
        sock=socket.socket();sock.settimeout(.15)
        try:
            if sock.connect_ex(('127.0.0.1',vehicle['tcp_port']))==0:raise RuntimeError('Foreign SITL TCP already active: '+str(vehicle['tcp_port']))
        finally:sock.close()
    for port in (9002,9012,9022,9032):
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:sock.bind(('127.0.0.1',port))
        except OSError as exc:raise RuntimeError('Foreign physics/proxy UDP active: '+str(port)) from exc
        finally:sock.close()
def stop_owned(process,timeout=10):
    if process is None or process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGINT)
    except ProcessLookupError:return
    try:process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
def launch_sitl(vehicle,output,ardupilot,pid_parameters):
    output=Path(output);folder=output/vehicle['role'];folder.mkdir(exist_ok=True)
    work=folder/'sitl';work.mkdir()
    params=folder/'experiment.parm'
    params.write_text('FRAME_CLASS 1\nFRAME_TYPE 1\nSIM_SPEEDUP 1\nFS_THR_ENABLE 0\nFS_GCS_ENABLE 0\nFENCE_ENABLE 0\nLOG_DISARMED 1\nLOG_BITMASK 65535\nGUID_OPTIONS 8\nMOT_PWM_MIN 1100\nMOT_PWM_MAX 1900\nMOT_THST_HOVER 0.5\nMOT_HOVER_LEARN 0\nMAV_SYSID '+str(vehicle['instance']+1)+'\n'+''.join(str(k)+' '+str(float(v))+'\n' for k,v in pid_parameters.items()))
    root=Path(ardupilot)
    defaults=','.join(str(p) for p in (root/'Tools/autotest/default_params/copter.parm',root/'Tools/autotest/default_params/gazebo-iris.parm',params))
    command=[str(root/'build/sitl/bin/arducopter'),'--model','JSON','--speedup','1','--defaults',defaults,
        '--home','-35.363262,149.165237,584,0','--sim-address=127.0.0.1','-I'+str(vehicle['instance']),
        '--sim-port-out',str(vehicle['sitl_port']),'--sim-port-in',str(vehicle['sitl_port']+1)]
    log=(folder/'sitl.log').open('x',buffering=1)
    process=subprocess.Popen(command,cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(folder/'sitl_launch.json',dict(command=command,pid=process.pid,vehicle=vehicle,binary_sha256=sha(root/'build/sitl/bin/arducopter'),params_sha256=sha(params)))
    return process,log

def capture(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    interface='loopback0' if Path('/sys/class/net/loopback0').exists() else 'lo'
    expression='host 127.0.0.1 and (udp port 9002 or udp port 9012 or udp port 9022 or udp port 9032 or tcp port 5760 or tcp port 5770 or tcp port 5780)'
    path=out/'network.pcap';log=(out/'network_capture.log').open('x')
    command=['/usr/bin/tcpdump','-i',interface,'-n','-p','-B','32768','-s','0','-w',str(path),expression]
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'network_capture_manifest.json',dict(command=command,pid=process.pid,capture_helper_pid=os.getpid(),uid=os.getuid(),scope='Read-only localhost three SITL + direct proxy; one common physical world'))
    stopped=False
    def stop(*args):
        nonlocal stopped;stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        while not stopped and not (out/'STOP_CAPTURE').exists():
            if process.poll() is not None:raise RuntimeError('Owned tcpdump exited')
            time.sleep(.1)
    finally:
        stop_owned(process,5);log.close()
        if path.exists():
            target=path.with_name(path.name+'.gz')
            with path.open('rb') as source,gzip.open(target,'wb',compresslevel=3) as destination:shutil.copyfileobj(source,destination)
            path.unlink()
            write_json(out/'network_capture_finished.json',dict(sha256=sha(target),bytes=target.stat().st_size,returncode=process.returncode))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--capture',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if not args.capture:parser.error('This standalone helper only records packets')
    capture(args.output)
