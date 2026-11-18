"""Start a controller-independent Gazebo world and its ArduPilot instances.

The launcher does not arm, fly, load policies, or judge an experiment. The
default parameter file explicitly selects ArduPilot's idealized simulator-state
backend (AHRS_EKF_TYPE=10). External control is asynchronous; a speedup setting
does not promise that an external controller executes every simulation step.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Sequence
import uuid
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ET.register_namespace('experimental', 'http://sdformat.org/schemas/experimental')


@dataclass(frozen=True)
class VehicleSpec:
    vehicle_id: str
    tcp_port: int
    relay_port: int
    physics_port: int
    position_enu: tuple[float, float, float]
    system_id: int
    instance: int


def _spawn(command, **kwargs):
    """The fallback is only for an explicitly requested Windows WSL prefix."""
    try:
        return subprocess.Popen(command, **kwargs)
    except OSError as exc:
        if exc.errno != 8 or not Path('/init').is_file() or not str(command[0]).endswith('wsl.exe'):
            raise
        return subprocess.Popen(['/init', command[0], *command], **kwargs)


class Session:
    """Own one Gazebo process group, native SITL instances, and their clients."""

    def __init__(self, output: Path, *, gui=False, ardupilot: Path | None = None,
                 gazebo_prefix: Sequence[str] = (), speedup=1., world: Path | None = None,
                 gazebo_plugin_dir: str | None = None, partition: str | None = None,
                 software_rendering=False, vehicle_ids: Sequence[str] | None = None):
        self.output = Path(output).expanduser().resolve()
        self.gui = bool(gui)
        self.software_rendering = bool(software_rendering)
        self.ardupilot = Path(ardupilot or os.environ.get('ARDUPILOT_PATH',
                              Path.home() / 'ardupilot-ROSMeetUp')).expanduser().resolve()
        self.gazebo_prefix = tuple(gazebo_prefix)
        self.speedup = float(speedup)
        if not math.isfinite(self.speedup) or self.speedup <= 0:
            raise ValueError('Positive finite speedup required')
        self.world_path = Path(world or HERE / 'world.sdf').expanduser().resolve()
        self.tree = ET.parse(self.world_path)
        scene = self.tree.getroot().find('world')
        if scene is None:
            raise ValueError('World SDF has no world')
        self.world_name = scene.attrib['name']
        self.partition = partition or ('copter_' + uuid.uuid4().hex[:10])
        self.gazebo_plugin_dir = gazebo_plugin_dir
        all_vehicles = []
        for model in scene.findall('model'):
            plugin = model.find("plugin[@name='ArduPilotPlugin']")
            if plugin is None:
                continue
            index = len(all_vehicles)
            pose = tuple(float(v) for v in model.findtext('pose', '0 0 0 0 0 0').split())
            all_vehicles.append(VehicleSpec(model.attrib['name'], 5760 + 10 * index,
                9032 + 10 * index, int(plugin.findtext('fdm_port_in')), pose[:3], index + 1, index))
        if not all_vehicles:
            raise ValueError('World contains no ArduPilot-equipped vehicles')
        self.vehicle_ids = None
        if vehicle_ids is not None:
            if isinstance(vehicle_ids, (str, bytes)):
                raise ValueError('vehicle_ids must be a sequence of vehicle names')
            self.vehicle_ids = tuple(vehicle_ids)
            if not self.vehicle_ids or not all(isinstance(name, str) for name in self.vehicle_ids):
                raise ValueError('Select at least one vehicle name')
            if len(set(self.vehicle_ids)) != len(self.vehicle_ids):
                raise ValueError('Vehicle selection contains duplicates')
            available = {spec.vehicle_id for spec in all_vehicles}
            unknown = set(self.vehicle_ids) - available
            if unknown:
                raise ValueError('Unknown vehicle selection: ' + ', '.join(sorted(unknown)))
            selected = set(self.vehicle_ids)
            for model in scene.findall('model'):
                if (model.find("plugin[@name='ArduPilotPlugin']") is not None
                        and model.attrib['name'] not in selected):
                    scene.remove(model)
            self.vehicles = [spec for spec in all_vehicles if spec.vehicle_id in selected]
        else:
            self.vehicles = all_vehicles
        self.clients = {}
        self.world = None
        self._processes = []
        self._logs = []
        self._gazebo = None
        self._closed = False
        self._entered = False
        self._cleanup_errors = []

    def _write(self, name, value):
        path = self.output / name
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def _prefix_run(self, args, *, timeout=15):
        process = _spawn([*self.gazebo_prefix, *args], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            self._stop_process(process)
            raise
        if process.returncode:
            raise RuntimeError(f'Gazebo command failed ({process.returncode}): {stderr.strip()}')
        return stdout.strip()

    def _log_process(self, name, command):
        log = (self.output / (name + '.log')).open('x', encoding='utf-8')
        self._logs.append(log)
        process = _spawn(command, cwd=self.output, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
        self._processes.append(process)
        return process

    def _check_ports(self):
        for spec in self.vehicles:
            for port, kind in ((spec.tcp_port, socket.SOCK_STREAM),
                               (spec.relay_port, socket.SOCK_DGRAM),
                               (spec.physics_port, socket.SOCK_DGRAM)):
                with socket.socket(socket.AF_INET, kind) as sock:
                    try:
                        sock.bind(('127.0.0.1', port))
                    except OSError as exc:
                        raise RuntimeError(f'Port already in use: {port}') from exc

    def __enter__(self):
        if self._entered:
            raise RuntimeError('A Session can only be started once')
        self._entered = True
        binary = self.ardupilot / 'build/sitl/bin/arducopter'
        defaults = self.ardupilot / 'Tools/autotest/default_params/copter.parm'
        for path in (binary, defaults, HERE / 'ardupilot.parm'):
            if not path.is_file():
                raise FileNotFoundError(path)
        self._check_ports()
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            from simulation.interface import Vehicle, World
            if self.gazebo_plugin_dir is None:
                remote_home = self._prefix_run(['printenv', 'HOME']) if self.gazebo_prefix else str(Path.home())
                self.gazebo_plugin_dir = str(Path(remote_home) / 'ardupilot_gazebo/build')
            plugin = str(Path(self.gazebo_plugin_dir) / 'libArduPilotPlugin.so')
            self._prefix_run(['test', '-f', plugin])
            resource = str(HERE / 'models')
            existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
            if existing:
                resource += ':' + existing
            gazebo_env = ['env', 'GZ_PARTITION=' + self.partition,
                'GZ_SIM_RESOURCE_PATH=' + resource,
                'GZ_SIM_SYSTEM_PLUGIN_PATH=' + self.gazebo_plugin_dir]
            if self.gui and self.software_rendering:
                gazebo_env += ['LIBGL_ALWAYS_SOFTWARE=1', 'GALLIUM_DRIVER=llvmpipe',
                               'QT_QPA_PLATFORM=xcb', 'QT_OPENGL=software']
            world_command_prefix = (*self.gazebo_prefix, *gazebo_env)
            self.world = World(world_name=self.world_name, partition=self.partition,
                               command_prefix=world_command_prefix)
            for spec in self.vehicles:
                folder = self.output / spec.vehicle_id
                folder.mkdir()
                client = Vehicle(spec.vehicle_id, spec.tcp_port, spec.relay_port,
                                 spec.physics_port, output=folder, system_id=spec.system_id)
                self.clients[spec.vehicle_id] = client
                client.start_relay()
            world_path = self.world_path
            if self.speedup != 1.:
                physics = self.tree.getroot().find('world/physics')
                factor = physics.find('real_time_factor')
                if factor is None:
                    factor = ET.SubElement(physics, 'real_time_factor')
                factor.text = str(self.speedup)
            if self.speedup != 1. or self.vehicle_ids is not None:
                world_path = self.output / 'world.sdf'
                self.tree.write(world_path, encoding='utf-8', xml_declaration=True)
            gz_args = ['gz', 'sim', '-r', '-v', '2']
            if not self.gui:
                gz_args.append('-s')
            gz_args.append(str(world_path))
            if self.gazebo_prefix:
                # Keep the remote process group ID so cleanup reaches Gazebo,
                # rather than only stopping the local wsl.exe transport.
                bootstrap = '''import os,signal,subprocess,sys
from pathlib import Path
p = subprocess.Popen(sys.argv[2:], start_new_session=True)
def stop(*_):
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    Path(sys.argv[1]).write_text(str(p.pid))
    sys.exit(p.wait())
finally:
    stop()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
'''
                command = [*self.gazebo_prefix, *gazebo_env, 'python3', '-c', bootstrap,
                           str(self.output / 'gazebo.pid'), *gz_args]
            else:
                command = [*gazebo_env, *gz_args]
            self._gazebo = self._log_process('gazebo', command)
            for spec in self.vehicles:
                work = self.output / spec.vehicle_id / 'sitl'
                work.mkdir()
                log = (work / 'console.log').open('x', encoding='utf-8')
                self._logs.append(log)
                command = [str(binary), '--model', 'JSON', '--speedup', str(self.speedup),
                    '--defaults', str(defaults) + ',' + str(HERE / 'ardupilot.parm'),
                    '--home', '-35.363262,149.165237,584,0', '--sim-address=127.0.0.1',
                    '-I' + str(spec.instance), '--sysid', str(spec.system_id),
                    '--sim-port-out', str(spec.relay_port), '--sim-port-in', str(spec.relay_port + 1)]
                process = _spawn(command, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
                self._processes.append(process)
            for client in self.clients.values():
                client.connect(timeout=90)
                if self.speedup != 1.:
                    client.set_parameter('SIM_SPEEDUP', self.speedup)
            self._write('launch.json', dict(status='READY', world=self.world_name,
                partition=self.partition, gui=self.gui, speedup=self.speedup,
                vehicles=[asdict(v) for v in self.vehicles],
                state_source='AHRS_EKF_TYPE=10: idealized simulator state',
                external_control='asynchronous; no strict external-controller stepping guarantee',
                source_sha256=dict(world=hashlib.sha256(world_path.read_bytes()).hexdigest(),
                                   autopilot_parameters=hashlib.sha256((HERE/'ardupilot.parm').read_bytes()).hexdigest(),
                                   autopilot_binary=hashlib.sha256(binary.read_bytes()).hexdigest(),
                                   physical_model=hashlib.sha256((HERE/'models/iris/model.sdf').read_bytes()).hexdigest()),
                process_ids=[p.pid for p in self._processes]))
            return self
        except BaseException as exc:
            try:
                self._write('error.json', dict(error=repr(exc)))
            finally:
                self.close()
            raise

    @staticmethod
    def _stop_process(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for name, client in self.clients.items():
            try:
                client.close()
            except Exception as exc:
                self._cleanup_errors.append(name + ': ' + repr(exc))
        pid_file = self.output / 'gazebo.pid'
        if self.gazebo_prefix and pid_file.is_file():
            try:
                remote_pid = int(pid_file.read_text().strip())
                if remote_pid <= 1:
                    raise ValueError('Invalid owned Gazebo process ID')
                # This ID was produced by our fresh-session remote bootstrap.
                stop_remote = ('import os,signal,sys\n'
                               'try: os.killpg(int(sys.argv[1]),signal.SIGTERM)\n'
                               'except ProcessLookupError: pass\n')
                self._prefix_run(['python3', '-c', stop_remote, str(remote_pid)])
            except Exception as exc:
                self._cleanup_errors.append('remote Gazebo: ' + repr(exc))
        for process in reversed(self._processes):
            try:
                self._stop_process(process)
            except Exception as exc:
                self._cleanup_errors.append(str(process.pid) + ': ' + repr(exc))
        for log in self._logs:
            try:
                log.close()
            except Exception as exc:
                self._cleanup_errors.append('log: ' + repr(exc))
        if self.output.is_dir():
            self._write('cleanup.json', dict(errors=self._cleanup_errors,
                process_returncodes={str(p.pid): p.poll() for p in self._processes}))

    def __exit__(self, exc_type, exc, tb):
        self.close()


def console_command(session, line):
    """Handle one human command; return False when the user requests exit."""
    words = line.split()
    if words == ['quit']:
        return False
    if words == ['state']:
        for name, client in session.clients.items():
            state = client.state(require_new=False)
            print(name, 't=', state.time_s, 'position=', state.position.tolist(),
                  'velocity=', state.velocity.tolist(), 'estimator=', state.estimator_type, flush=True)
    elif len(words) == 4 and words[0] == 'wind':
        session.world.set_wind([float(v) for v in words[1:]])
        print('Wind command published, ENU m/s:', ' '.join(words[1:]), flush=True)
    elif len(words) in (3, 4) and words[0] == 'param':
        client = session.clients[words[1]]
        value = (client.get_parameter(words[2]) if len(words) == 3 else
                 client.set_parameter(words[2], float(words[3])))
        print(words[1], words[2], value, flush=True)
    elif words:
        print('Commands: state | wind E N U | param droneN NAME [VALUE] | quit', flush=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--ardupilot', type=Path)
    parser.add_argument('--speedup', type=float, default=1.)
    parser.add_argument('--vehicle', nargs='+', metavar='ID',
                        help='Run only these vehicle IDs; default: all vehicles in the world')
    parser.add_argument('--gazebo-plugin-dir')
    parser.add_argument('--gazebo-distro', help='Optional WSL distribution owning Gazebo')
    parser.add_argument('--gazebo-user')
    parser.add_argument('--software-rendering', action='store_true',
                        help='Optional CPU rendering for graphics drivers that fail to create the GUI')
    args = parser.parse_args(argv)
    prefix = []
    if args.gazebo_distro:
        prefix = ['/mnt/c/Windows/System32/wsl.exe', '-d', args.gazebo_distro]
        if args.gazebo_user:
            prefix += ['-u', args.gazebo_user]
        prefix += ['--']
    output = args.output or HERE.parent / 'runs' / ('scene_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    try:
        with Session(output, gui=args.gui, ardupilot=args.ardupilot, gazebo_prefix=prefix,
                     speedup=args.speedup, gazebo_plugin_dir=args.gazebo_plugin_dir,
                     software_rendering=args.software_rendering, vehicle_ids=args.vehicle) as session:
            print('READY: ' + ', '.join(f'{v.vehicle_id}=TCP:{v.tcp_port}' for v in session.vehicles), flush=True)
            print('Commands: state | wind E N U | param droneN NAME [VALUE] | quit', flush=True)
            print('No flight started. Ctrl+C stops this session.', flush=True)
            commands = queue.Queue()

            def read_console():
                for line in sys.stdin:
                    commands.put(line)

            threading.Thread(target=read_console, daemon=True).start()
            while True:
                for client in session.clients.values():
                    client.drain()
                if session._gazebo.poll() is not None:
                    raise RuntimeError('Gazebo exited; inspect gazebo.log')
                try:
                    line = commands.get_nowait()
                except queue.Empty:
                    line = None
                if line is not None:
                    try:
                        if not console_command(session, line):
                            break
                    except (ValueError, KeyError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                        print('Command failed:', exc, flush=True)
                time.sleep(.05)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
