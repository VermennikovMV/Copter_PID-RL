"""Extract GIF/MP4 from a real screen recording, preserving its wall-time speed."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output for derived media")
    source = session / "gazebo_screen.mp4"
    receipt = read(source.with_suffix(".mp4.json"))
    flight = session / "flight"
    summary = read(flight / "complete.json")
    plan = read(flight / "plan.json")
    if receipt["status"] != "CAPTURED" or sha(source) != receipt["sha256"]:
        raise ValueError("Completed actual framebuffer recording with matching SHA is required")
    if not summary.get("one_physical_world"):
        raise ValueError("One actual physical world is required")
    start = summary["measurement_sim_start_s"]
    end = summary["measurement_sim_end_s"]
    poses = [json.loads(line) for line in (flight / "groundtruth.jsonl").read_text().splitlines() if line]
    selected = [row for row in poses if start <= row["sim_time"] <= end and "wall_time" in row]
    if not selected or {row["name"] for row in selected} != {"iris_pid", "iris_hybrid", "iris_direct"}:
        raise ValueError("Recorded physical poses with receive times are required for all three aircraft")
    # Receive time is not an exact display latency measurement. Include margins
    # and retain the full source video; never align by a fabricated trajectory.
    first_wall = min(row["wall_time"] for row in selected)
    last_wall = max(row["wall_time"] for row in selected)
    offset = max(0., first_wall - receipt["started_wall_s"] - .75)
    length = last_wall - first_wall + 1.5
    video_duration = float(receipt["probe"]["format"]["duration"])
    length = min(length, video_duration - offset)
    if length <= 0:
        raise ValueError("Measurement interval is outside the recorded video")
    gif_length = length
    gif_scope = "Full recorded measurement interval"
    if plan["mission"] == "route" and all(row.get("route_completed") for row in summary["roles"].values()):
        # Show the moving part in the README, while the MP4 retains the entire
        # measurement, including post-route wind and recovery. Use actual goal
        # event/world timestamps; never animate logged positions.
        completion_times = []
        for role in summary["roles"]:
            commands_log = [json.loads(line) for line in (flight / role / "commands.jsonl").read_text().splitlines() if line]
            completed = [row["world_sim_time"] for row in commands_log
                         if row.get("goal_event") and row["goal_event"]["goal_index"] == 4]
            if len(completed) != 1:
                raise ValueError("Completed route needs exactly one actual final-goal event per aircraft")
            completion_times.append(completed[0])
        gif_end = min(end, max(completion_times) + 2.)
        visual_rows = [row for row in selected if row["sim_time"] <= gif_end]
        gif_length = min(length, max(row["wall_time"] for row in visual_rows) - receipt["started_wall_s"] - offset + .75)
        gif_scope = "Route traversal through the last aircraft's completion plus 2 simulation seconds; full measurement in MP4"
    output.mkdir(parents=True)
    mp4 = output / "gazebo.mp4"
    gif = output / "gazebo.gif"
    commands = [
        ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-ss", str(offset), "-i", str(source),
         "-t", str(length), "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
         "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)],
        ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-i", str(mp4), "-t", str(gif_length), "-filter_complex_threads", "1",
         "-vf", "fps=12,scale=1000:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
         "-loop", "0", "-threads", "2", str(gif)],
        ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-ss", "1", "-i", str(mp4),
         "-frames:v", "1", "-threads", "1", str(output / "preview.png")],
    ]
    for index, command in enumerate(commands):
        with (output / f"encode_{index}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = {"source": "actual Gazebo GUI X11 framebuffer recording", "mission": plan["mission"],
              "scenario": plan["scenario"], "seed": plan["seed"], "source_video_sha256": sha(source),
              "capture_receipt_sha256": sha(source.with_suffix(".mp4.json")),
              "groundtruth_sha256": sha(flight / "groundtruth.jsonl"), "summary_sha256": sha(flight / "complete.json"),
              "offset_wall_seconds": offset, "duration_wall_seconds": length, "playback_speed": 1.0,
              "gif_duration_wall_seconds": gif_length, "gif_scope": gif_scope,
              "measurement_sim_start_s": start, "measurement_sim_end_s": end,
              "alignment": "Received pose wall times with 0.75 s margins; exact GUI display latency not measured",
              "all_recorded_controller_failures": {key: value.get("failure") for key, value in summary["roles"].items()},
              "changes": "Temporal trim, video encoding, GIF downscale/palette only. No trajectory reconstruction or generated scene.",
              "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in (mp4, gif, output / "preview.png")},
              "commands": commands}
    (output / "media.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"mission": plan["mission"], "duration_seconds": length, "files": result["files"]}, indent=2))


if __name__ == "__main__":
    main()
