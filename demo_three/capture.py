"""Record real, explicitly selected X11 application windows in WSLg.

No rendered trajectories or generated frames are used. Terminal input is sent
to the owned xterm window; GUI capture reads that window's actual framebuffer.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def descendants(pid):
    parents = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            parents[int(path.name)] = int((path / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    owned = {int(pid)}
    while True:
        expanded = owned | {child for child, parent in parents.items() if parent in owned}
        if expanded == owned:
            return owned
        owned = expanded


def find_window(pid, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True, check=True).stdout
        owned = descendants(pid)
        for line in tree.splitlines():
            geometry = re.search(r"(\d{3,})x(\d{3,})", line)
            identity = re.search(r"0x[0-9a-fA-F]+", line)
            if not geometry or not identity:
                continue
            props = subprocess.run(["xprop", "-id", identity[0], "_NET_WM_PID", "WM_NAME"],
                                   capture_output=True, text=True).stdout
            owner = re.search(r"_NET_WM_PID.*=\s*(\d+)", props)
            if owner and int(owner[1]) in owned:
                return {"window_id": identity[0], "owner_pid": int(owner[1]),
                        "requested_pid": pid, "width": int(geometry[1]), "height": int(geometry[2]),
                        "properties": props.strip(), "observed_utc": utc()}
        time.sleep(.2)
    raise RuntimeError(f"No visible X11 window owned by PID {pid} or its descendants")


def terminal(folder, cwd):
    folder.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PS1="copter$ ", HISTFILE="/dev/null", PROMPT_COMMAND="", TERM="xterm-256color")
    command = ["xterm", "-title", "Copter PID / RL - commands", "-fa", "DejaVu Sans Mono",
               "-fs", "12", "-geometry", "116x12", "-bg", "#171c25", "-fg", "#e9edf3",
               "-xrm", "XTerm*allowSendEvents: true",
               "-e", "bash", "--noprofile", "--norc", "-i"]
    with (folder / "xterm.log").open("x") as log:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    window = find_window(proc.pid)
    value = {"pid": proc.pid, "command": command, "cwd": str(cwd), "window": window, "started_utc": utc()}
    write_json(folder / "terminal.json", value)
    print(json.dumps(value), flush=True)


def send(manifest, text, enter):
    item = json.loads(manifest.read_text())
    window = find_window(item["pid"], timeout=3)
    if window["window_id"] != item["window"]["window_id"]:
        raise RuntimeError("Owned terminal window changed")
    # Direct delivery avoids typing into the user's unrelated Windows window.
    # Only this xterm is launched with allowSendEvents enabled.
    commands = [["xdotool", "type", "--window", window["window_id"], "--delay", "30", "--", text]]
    if enter:
        commands.append(["xdotool", "key", "--window", window["window_id"], "Return"])
    started = utc()
    for command in commands:
        subprocess.run(command, check=True)
    event = {"started_utc": started, "finished_utc": utc(), "window": window,
             "typed_text": text, "return_pressed": enter,
             "scope": "Actual X11 terminal input; command completion must be checked separately"}
    with (manifest.parent / "input_events.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def record(pid, output, duration, stop_file=None, snapshot=False):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    window = find_window(pid)
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-f", "x11grab", "-draw_mouse", "1",
               "-window_id", str(int(window["window_id"], 16)), "-framerate", "20",
               "-i", os.environ.get("DISPLAY", ":0")]
    if snapshot:
        command += ["-frames:v", "1", "-threads", "1", str(output)]
    else:
        command += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-threads", "2",
                    "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    receipt = {"status": "STARTING", "source": "real X11 window framebuffer", "window": window,
               "command": command, "started_utc": utc(), "started_wall_s": time.time(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    receipt_path = output.with_suffix(output.suffix + ".json")
    write_json(receipt_path, receipt)
    with output.with_suffix(output.suffix + ".log").open("x") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(status="RECORDING", recorder_pid=proc.pid)
        write_json(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        try:
            start = time.monotonic()
            requested_stop = False
            while proc.poll() is None:
                if time.monotonic() - start >= duration or (stop_file and stop_file.exists()):
                    requested_stop = True
                    proc.send_signal(signal.SIGINT)
                    break
                time.sleep(.1)
            proc.wait(timeout=15)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
    receipt.update(finished_utc=utc(), finished_wall_s=time.time(), returncode=proc.returncode)
    if not output.exists() or output.stat().st_size < 500:
        receipt.update(status="ERROR", error="No usable screen capture")
        write_json(receipt_path, receipt)
        raise RuntimeError(receipt["error"])
    receipt.update(sha256=hashlib.sha256(output.read_bytes()).hexdigest(), bytes=output.stat().st_size)
    if not snapshot:
        receipt["probe"] = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)], text=True))
    receipt["requested_stop"] = requested_stop
    receipt["status"] = "CAPTURED" if proc.returncode in (0, 255) and (snapshot or requested_stop) else "ERROR"
    if receipt["status"] == "ERROR":
        receipt["error"] = "Capture ended before the requested stop; partial frames are retained"
    write_json(receipt_path, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "probe"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("terminal")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cwd", type=Path, default=Path.cwd())
    p = sub.add_parser("send")
    p.add_argument("--terminal", type=Path, required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--enter", action="store_true")
    for name in ("snapshot", "record"):
        p = sub.add_parser(name)
        p.add_argument("--pid", type=int, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--duration", type=float, default=180)
        p.add_argument("--stop-file", type=Path)
    a = parser.parse_args()
    if a.operation == "terminal":
        terminal(a.output.resolve(), a.cwd.resolve())
    elif a.operation == "send":
        send(a.terminal.resolve(), a.text, a.enter)
    else:
        record(a.pid, a.output.resolve(), a.duration, a.stop_file, a.operation == "snapshot")


if __name__ == "__main__":
    main()
