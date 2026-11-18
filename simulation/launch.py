"""Start the Gazebo world and one ArduPilot SITL instance per vehicle in it.

  python -m simulation.launch --gui                    # all vehicles of simulation/world.sdf
  python -m simulation.launch --vehicle drone1 --gui   # one vehicle

The launcher does not fly: the console accepts `state`, `wind E N U`, `param droneN NAME [VALUE]`,
`quit`. Experiments (gazebo/*.py) use `Session` directly. SITL runs with the JSON physics backend
of ardupilot_gazebo and AHRS_EKF_TYPE=10 (simulator state, simulation/ardupilot.parm).

On WSL with Gazebo in another distribution, `--gazebo-distro NAME` runs `gz` through wsl.exe.
"""
import argparse
from dataclasses import dataclass
import datetime as dt
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

from simulation.interface import Vehicle, World

HERE = Path(__file__).resolve().parent
ET.register_namespace('experimental', 'http://sdformat.org/schemas/experimental')
HOME = '-35.363262,149.165237,584,0'

# Runs Gazebo in its own process group inside the other WSL distribution and writes its PID, so
# that closing the session stops Gazebo itself and not only the local wsl.exe.
REMOTE_BOOTSTRAP = '''import os, signal, subprocess, sys
from pathlib import Path
p = subprocess.Popen(sys.argv[2:], start_new_session=True)
def stop(*_):
    if p.poll() is None:
        try: os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError: pass
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    Path(sys.argv[1]).write_text(str(p.pid))
    sys.exit(p.wait())
finally:
    stop()
    try: p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait()
'''


@dataclass(frozen=True)
class VehicleSpec:
    vehicle_id: str
    tcp_port: int        # MAVLink (SITL serial0)
    relay_port: int      # SITL FDM output -> motor relay
    physics_port: int    # ArduPilotPlugin fdm_port_in
    position_enu: tuple
    system_id: int
    instance: int


def spawn(command, **kwargs):
    """subprocess.Popen that can also start wsl.exe from inside WSL (ENOEXEC fallback through /init)."""
    try:
        return subprocess.Popen(command, **kwargs)
    except OSError as exc:
        if exc.errno != 8 or not Path('/init').is_file() or not str(command[0]).endswith('wsl.exe'):
            raise
        return subprocess.Popen(['/init', command[0], *command], **kwargs)


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


class Session:
    """Gazebo + SITL instances + their clients (`clients[vehicle_id]`, `world`); a context manager."""

    def __init__(self, output, *, gui=False, ardupilot=None, gazebo_prefix=(), speedup=1., world=None,
                 gazebo_plugin_dir=None, partition=None, software_rendering=False, vehicle_ids=None):
        self.output = Path(output).expanduser().resolve()
        self.gui, self.software_rendering = bool(gui), bool(software_rendering)
        self.ardupilot = Path(ardupilot or os.environ.get('ARDUPILOT_PATH', Path.home() / 'ardupilot-ROSMeetUp')).expanduser()
        self.gazebo_prefix = tuple(gazebo_prefix)
        self.speedup = float(speedup)
        self.gazebo_plugin_dir = gazebo_plugin_dir
        self.partition = partition or f'copter_{os.getpid()}_{int(time.time())}'
        self.tree = ET.parse(Path(world or HERE / 'world.sdf').expanduser())
        scene = self.tree.getroot().find('world')
        self.world_name = scene.attrib['name']
        self.vehicles = []
        for model in scene.findall('model'):
            plugin = model.find("plugin[@name='ArduPilotPlugin']")
            if plugin is None:
                continue
            if vehicle_ids is not None and model.attrib['name'] not in vehicle_ids:
                scene.remove(model)          # not selected: removed from the world
                continue
            index = len(self.vehicles)
            pose = tuple(float(v) for v in model.findtext('pose', '0 0 0 0 0 0').split())
            self.vehicles.append(VehicleSpec(model.attrib['name'], 5760 + 10 * index, 9032 + 10 * index,
                                             int(plugin.findtext('fdm_port_in')), pose[:3], index + 1, index))
        if not self.vehicles:
            raise ValueError('no ArduPilot vehicle selected in the world')
        self.clients, self.world = {}, None
        self._processes, self._logs, self._gazebo = [], [], None

    def _prefix_run(self, args, timeout=15):
        result = subprocess.run([*self.gazebo_prefix, *args], capture_output=True, text=True, timeout=timeout) \
            if not self.gazebo_prefix else _run_prefixed([*self.gazebo_prefix, *args], timeout)
        if result.returncode:
            raise RuntimeError(f'command failed: {args}: {result.stderr.strip()}')
        return result.stdout.strip()

    def _start(self, name, command, cwd):
        log = (Path(cwd) / f'{name}.log').open('x', encoding='utf-8')
        self._logs.append(log)
        process = spawn(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self._processes.append(process)
        return process

    def __enter__(self):
        binary = self.ardupilot / 'build/sitl/bin/arducopter'
        defaults = self.ardupilot / 'Tools/autotest/default_params/copter.parm'
        for path in (binary, defaults):
            if not path.is_file():
                raise FileNotFoundError(f'{path} (build ardupilot-ROSMeetUp: ./waf configure --board sitl && ./waf copter)')
        for spec in self.vehicles:
            for port, kind in ((spec.tcp_port, socket.SOCK_STREAM), (spec.relay_port, socket.SOCK_DGRAM),
                               (spec.physics_port, socket.SOCK_DGRAM)):
                with socket.socket(socket.AF_INET, kind) as sock:
                    try:
                        sock.bind(('127.0.0.1', port))
                    except OSError as exc:
                        raise RuntimeError(f'port {port} is busy: another simulation is running') from exc
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            if self.gazebo_plugin_dir is None:
                home = self._prefix_run(['printenv', 'HOME']) if self.gazebo_prefix else str(Path.home())
                self.gazebo_plugin_dir = str(Path(home) / 'ardupilot_gazebo/build')
            self._prefix_run(['test', '-f', str(Path(self.gazebo_plugin_dir) / 'libArduPilotPlugin.so')])
            resources = ':'.join(filter(None, [str(HERE / 'models'), os.environ.get('GZ_SIM_RESOURCE_PATH', '')]))
            gazebo_env = ['env', 'GZ_PARTITION=' + self.partition, 'GZ_SIM_RESOURCE_PATH=' + resources,
                          'GZ_SIM_SYSTEM_PLUGIN_PATH=' + self.gazebo_plugin_dir]
            if self.gui and self.software_rendering:
                gazebo_env += ['LIBGL_ALWAYS_SOFTWARE=1', 'GALLIUM_DRIVER=llvmpipe', 'QT_QPA_PLATFORM=xcb',
                               'QT_OPENGL=software']
            self.world = World(self.world_name, self.partition, (*self.gazebo_prefix, *gazebo_env))
            for spec in self.vehicles:
                client = Vehicle(spec.vehicle_id, spec.tcp_port, spec.relay_port, spec.physics_port, spec.system_id)
                client.start_relay()
                self.clients[spec.vehicle_id] = client

            if self.speedup != 1.:
                physics = self.tree.getroot().find('world/physics')
                factor = physics.find('real_time_factor')
                if factor is None:
                    factor = ET.SubElement(physics, 'real_time_factor')
                factor.text = str(self.speedup)
            world_path = self.output / 'world.sdf'
            self.tree.write(world_path, encoding='utf-8', xml_declaration=True)
            gz = ['gz', 'sim', '-r', '-v', '2', *([] if self.gui else ['-s']), str(world_path)]
            if self.gazebo_prefix:
                command = [*self.gazebo_prefix, *gazebo_env, 'python3', '-c', REMOTE_BOOTSTRAP,
                           str(self.output / 'gazebo.pid'), *gz]
            else:
                command = [*gazebo_env, *gz]
            self._gazebo = self._start('gazebo', command, self.output)

            for spec in self.vehicles:
                work = self.output / spec.vehicle_id
                work.mkdir()
                self._start('sitl', [str(binary), '--model', 'JSON', '--speedup', str(self.speedup),
                                     '--defaults', f'{defaults},{HERE / "ardupilot.parm"}', '--home', HOME,
                                     '--sim-address=127.0.0.1', f'-I{spec.instance}', '--sysid', str(spec.system_id),
                                     '--sim-port-out', str(spec.relay_port), '--sim-port-in', str(spec.relay_port + 1)],
                            work)
            for client in self.clients.values():
                client.connect(timeout=90)
                if self.speedup != 1.:
                    client.set_parameter('SIM_SPEEDUP', self.speedup)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        for client in self.clients.values():
            try:
                client.close()
            except Exception as exc:
                print(f'{client.vehicle_id}: {exc!r}', file=sys.stderr)
        self.clients = {}
        pid_file = self.output / 'gazebo.pid'
        if self.gazebo_prefix and pid_file.is_file():   # stop Gazebo in the other WSL distribution
            try:
                self._prefix_run(['python3', '-c', 'import os,signal,sys\ntry: os.killpg(int(sys.argv[1]),signal.SIGTERM)\n'
                                  'except ProcessLookupError: pass', pid_file.read_text().strip()])
                pid_file.unlink()
            except Exception as exc:
                print(f'remote Gazebo: {exc!r}', file=sys.stderr)
        for process in reversed(self._processes):
            stop_process(process)
        self._processes = []
        for log in self._logs:
            log.close()
        self._logs = []

    def __exit__(self, *_):
        self.close()


def _run_prefixed(command, timeout):
    process = spawn(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        stop_process(process)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def console_command(session, line):
    """One console command; returns False on `quit`."""
    words = line.split()
    if words == ['quit']:
        return False
    if words == ['state']:
        for name, client in session.clients.items():
            s = client.state(require_new=False)
            print(name, f't={s.time_s:.2f}', 'position', s.position.round(3).tolist(), 'velocity',
                  s.velocity.round(3).tolist(), flush=True)
    elif len(words) == 4 and words[0] == 'wind':
        session.world.set_wind([float(v) for v in words[1:]])
    elif len(words) in (3, 4) and words[0] == 'param':
        client = session.clients[words[1]]
        value = client.get_parameter(words[2]) if len(words) == 3 else client.set_parameter(words[2], float(words[3]))
        print(words[1], words[2], value, flush=True)
    elif words:
        print('commands: state | wind E N U | param droneN NAME [VALUE] | quit', flush=True)
    return True


def wsl_prefix(distro, user=None):
    """Command prefix that runs a program in another WSL distribution."""
    if not distro:
        return []
    return ['/mnt/c/Windows/System32/wsl.exe', '-d', distro, *(['-u', user] if user else []), '--']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path)
    p.add_argument('--gui', action='store_true')
    p.add_argument('--ardupilot', type=Path, help='ardupilot-ROSMeetUp checkout (default: $ARDUPILOT_PATH)')
    p.add_argument('--speedup', type=float, default=1.)
    p.add_argument('--vehicle', nargs='+', metavar='ID', help='only these vehicles (default: all in the world)')
    p.add_argument('--gazebo-plugin-dir')
    p.add_argument('--gazebo-distro', help='WSL distribution with Gazebo')
    p.add_argument('--gazebo-user')
    p.add_argument('--software-rendering', action='store_true', help='CPU rendering for the Gazebo window (WSLg)')
    a = p.parse_args(argv)
    output = a.output or HERE.parent / 'runs' / ('scene_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    try:
        with Session(output, gui=a.gui, ardupilot=a.ardupilot, gazebo_prefix=wsl_prefix(a.gazebo_distro, a.gazebo_user),
                     speedup=a.speedup, gazebo_plugin_dir=a.gazebo_plugin_dir,
                     software_rendering=a.software_rendering, vehicle_ids=a.vehicle) as session:
            print('ready:', ', '.join(f'{v.vehicle_id} (MAVLink tcp:{v.tcp_port})' for v in session.vehicles), flush=True)
            print('commands: state | wind E N U | param droneN NAME [VALUE] | quit', flush=True)
            commands = queue.Queue()
            threading.Thread(target=lambda: [commands.put(line) for line in sys.stdin], daemon=True).start()
            while True:
                for client in session.clients.values():
                    client.drain()
                if session._gazebo.poll() is not None:
                    raise RuntimeError(f'Gazebo exited, see {session.output / "gazebo.log"}')
                try:
                    if not console_command(session, commands.get_nowait()):
                        break
                except queue.Empty:
                    pass
                except (ValueError, KeyError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                    print('command failed:', exc, flush=True)
                time.sleep(.05)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
