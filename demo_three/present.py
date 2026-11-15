"""Launch the live three-drone demo and record its actual Gazebo GUI window.

Run this entry point in the Ubuntu WSL distribution that owns the desktop.
The numerical controller uses the existing scientific Python in Ubuntu-24.04.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo_three.capture import record, find_window


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mission", choices=("route", "hover"))
    p.add_argument("--wind", choices=("strong", "gust", "none"))
    p.add_argument("--output", type=Path)
    p.add_argument("--duration", type=float)
    p.add_argument("--manual-start", action="store_true", help="Hold before measurement until START_MEASUREMENT appears")
    p.add_argument("--gui-warmup", type=float, default=10., help="Wall seconds held after flight readiness for GUI initialization")
    p.add_argument("--science-distro", default="Ubuntu-24.04")
    p.add_argument("--science-python", default=os.environ.get("COPTER_SCIENCE_PYTHON"))
    p.add_argument("--public-root", type=Path)
    p.add_argument("--live-root", type=Path, default=Path("/mnt/wsl/copter_three"),
                   help="Shared Linux filesystem for live telemetry; archived into --output afterwards")
    a = p.parse_args()
    if not 0 <= a.gui_warmup <= 120:
        p.error("--gui-warmup must be between 0 and 120 seconds")
    wind = a.wind or ("strong" if a.mission == "route" else "gust")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (a.output or ROOT / "runs" / f"gazebo_{a.mission}_{wind}_{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    live_root = a.live_root.resolve()
    live_root.mkdir(parents=True, exist_ok=True)
    live_session = live_root / (stamp + "_" + uuid.uuid4().hex[:8])
    live_session.mkdir(exist_ok=False)
    flight = live_session / "flight"
    gate = out / "START_MEASUREMENT"
    stop_recording = out / "STOP_RECORDING"
    runtime_args = ["-u", "-B", str(ROOT / "demo_three/runtime.py"), "--mission", a.mission,
                    "--scenario", {"strong": "force_strong", "gust": "gust", "none": "nominal"}[wind],
                    "--seed", "83301", "--gui", "--output", str(flight), "--start-file", str(gate)]
    if a.duration:
        runtime_args += ["--duration", str(a.duration)]
    if a.public_root:
        runtime_args += ["--public-root", str(a.public_root.resolve())]
    elif (ROOT / "models/phase29_public_inference.json").exists():
        runtime_args += ["--public-root", str(ROOT)]
    # This bootstrap resolves the scientific interpreter inside its own WSL
    # home; no username or machine-specific absolute home is embedded in code.
    bootstrap = ("import os,sys;from pathlib import Path;"
                 "p=Path(sys.argv[1]) if sys.argv[1]!='-' else Path.home()/'vkr_sim/venv-cu128/bin/python';"
                 "assert p.is_file(), 'Set --science-python to the installed scientific Python';"
                 "os.execv(str(p),[str(p),*sys.argv[2:]])")
    command = ["/mnt/c/Windows/System32/wsl.exe", "-d", a.science_distro, "--cd", str(ROOT), "--",
               "env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "/usr/bin/python3", "-c", bootstrap,
               a.science_python or "-", *runtime_args]
    manifest = {"mission": a.mission, "wind": wind, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source": "actual Gazebo GUI framebuffer", "runtime_command": command,
                "output": str(out), "live_flight": str(flight), "manual_start": a.manual_start}
    (out / "presentation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Gazebo | {a.mission} | wind: {wind} | PID / Hybrid / End-to-end", flush=True)
    try:
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    except OSError as exc:
        if exc.errno != 8 or not Path("/init").is_file():
            raise
        # WSL's binfmt interpreter with the preserved argv[0] convention.
        # Some WSL sessions expose Windows binaries but lack exec registration.
        command = ["/init", command[0], *command]
        manifest["runtime_command"] = command
        (out / "presentation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    recorder = None
    capture_log = None
    started = time.monotonic()
    complete_observed = None
    gui_ready_since = None
    flight_ready_since = None
    ready_snapshot_saved = False
    released = False
    try:
        while process.poll() is None:
            if time.monotonic() - started > 900:
                raise TimeoutError("Live presentation exceeded 15 minutes; preserved current attempt")
            state = read(flight / "SCENE_STATE.json")
            if state and state.get("gui_pid") and recorder is None:
                try:
                    window = find_window(state["gui_pid"], timeout=2)
                except RuntimeError:
                    if time.monotonic() - started > 90:
                        raise
                    continue
                # Qt initially maps a different-sized black loading window.
                # x11grab must start after the final client size is stable.
                if (window["width"], window["height"]) != (1280, 800) or '"Gazebo Sim"' not in window["properties"]:
                    gui_ready_since = None
                    time.sleep(.2)
                    continue
                gui_ready_since = gui_ready_since or time.monotonic()
                if time.monotonic() - gui_ready_since < 3.:
                    time.sleep(.2)
                    continue
                # This captures a real application window and also verifies its
                # owner PID; it never captures other desktop applications.
                with (out / "snapshot.log").open("x") as snapshot_log, contextlib.redirect_stdout(snapshot_log):
                    record(state["gui_pid"], out / "gazebo_first.png", 20, snapshot=True)
                capture_log = (out / "screen_capture.log").open("x")
                recorder = subprocess.Popen([sys.executable, str(ROOT / "demo_three/capture.py"), "record",
                    "--pid", str(state["gui_pid"]), "--output", str(out / "gazebo_screen.mp4"),
                    "--duration", "900", "--stop-file", str(stop_recording)],
                    stdout=capture_log, stderr=subprocess.STDOUT, start_new_session=True)
                print("Recording the real Gazebo window.", flush=True)
            capture_receipt = read(out / "gazebo_screen.mp4.json")
            if not released and (flight / "READY.json").exists() and capture_receipt:
                if capture_receipt.get("status") not in ("RECORDING", "CAPTURED"):
                    raise RuntimeError("Window capture is not running")
                if flight_ready_since is None:
                    flight_ready_since = time.monotonic()
                    print("All three ready. Holding for GUI initialization; physics and PID remain running.", flush=True)
                if time.monotonic() - flight_ready_since < a.gui_warmup:
                    time.sleep(.1)
                    continue
                if not ready_snapshot_saved:
                    with (out / "ready_snapshot.log").open("x") as snapshot_log, contextlib.redirect_stdout(snapshot_log):
                        record(state["gui_pid"], out / "gazebo_ready.png", 20, snapshot=True)
                    ready_snapshot_saved = True
                    manifest["gui_readiness"] = {"warmup_wall_seconds": a.gui_warmup,
                        "snapshot": "gazebo_ready.png", "automatic_visual_interpretation": False,
                        "manual_start": a.manual_start}
                    (out / "presentation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    if a.manual_start:
                        print("Inspect the actual Gazebo window, then in another terminal: touch " + str(gate.relative_to(ROOT)), flush=True)
                if not a.manual_start:
                    gate.write_text("All three ready; actual GUI recorder is running.\n")
                    released = True
                elif gate.exists():
                    released = True
            if (flight / "complete.json").exists():
                complete_observed = complete_observed or time.monotonic()
                if time.monotonic() - complete_observed > 1.:
                    stop_recording.touch(exist_ok=True)
            if recorder and recorder.poll() is not None and complete_observed is None:
                raise RuntimeError("Real-window recording ended before flight completion")
            time.sleep(.1)
        code = process.wait()
        manifest.update(runtime_returncode=code, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (out / "presentation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if code:
            raise RuntimeError("Simulation stopped; see flight/error.json; this attempt is preserved")
        print("Flight complete. Saved: " + str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out.name), flush=True)
    finally:
        stop_recording.touch(exist_ok=True)
        if process.poll() is None:
            if flight.exists():
                (flight / "STOP_DEMO").touch()
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=15)
        if recorder:
            recorder.wait(timeout=20)
        if capture_log:
            capture_log.close()
        # All high-frequency records stay on the shared Linux filesystem until
        # the owned writers stop. Preserve every attempt, including exceptions.
        if flight.exists():
            (flight / "STOP_SHARED_WORLD").touch(exist_ok=True)
            (flight / "STOP_CAPTURE").touch(exist_ok=True)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if (flight / "scene_finished.json").exists() and (flight / "network_capture_finished.json").exists():
                    break
                time.sleep(.2)
            archived = out / "flight"
            shutil.copytree(flight, archived)
            entries = []
            for source in sorted(flight.rglob("*")):
                if not source.is_file():
                    continue
                relative = source.relative_to(flight)
                source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
                if hashlib.sha256((archived / relative).read_bytes()).hexdigest() != source_sha:
                    raise RuntimeError("Live-flight archive differs: " + str(relative))
                entries.append({"path": relative.as_posix(), "sha256": source_sha, "bytes": source.stat().st_size})
            (out / "flight_archive.json").write_text(json.dumps({"status": "BYTE_EXACT_COPY",
                "source": str(flight), "archived": str(archived), "files": entries}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
