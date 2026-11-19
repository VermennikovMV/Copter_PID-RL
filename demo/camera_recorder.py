"""Write the frames of a Gazebo camera topic to a video, one frame per image message.

Runs where Gazebo runs (system python3 with python3-gz-transport13):
  python3 camera_recorder.py TOPIC out.mp4 stamps.txt STOP_FILE
The camera publishes 25 images per simulated second, so the video plays at the simulated speed
however slowly the simulation ran. stamps.txt: simulation time of every frame. Stops when STOP_FILE
appears, or when no frame has arrived for 120 s (the simulation has ended).
"""
import os
import subprocess
import sys
import threading
import time

from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node

topic, out, stamps, stop_file = sys.argv[1:5]
lock = threading.Lock()
state = dict(ffmpeg=None, last=time.monotonic())
log = open(stamps, 'w')


def frame(msg):
    with lock:
        if state['ffmpeg'] is None:
            state['ffmpeg'] = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                 '-s', f'{msg.width}x{msg.height}', '-r', '25', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast',
                 '-pix_fmt', 'yuv420p', '-crf', '18', out], stdin=subprocess.PIPE)
        state['ffmpeg'].stdin.write(msg.data)
        log.write(f'{msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9:.3f}\n')
        state['last'] = time.monotonic()


node = Node()
node.subscribe(Image, topic, frame)
while not os.path.exists(stop_file) and time.monotonic() - state['last'] < 120.:
    time.sleep(.2)
with lock:
    log.close()
    if state['ffmpeg'] is not None:
        state['ffmpeg'].stdin.close()
        state['ffmpeg'].wait()
