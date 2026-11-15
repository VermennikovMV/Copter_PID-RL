#!/usr/bin/env python3
"""Build / serve one Gazebo world containing three independently driven Iris.

Building uses only the standard library and starts no simulator. --serve is the
explicit physics boundary; it runs in the Gazebo environment with Transport13.
No function here controls a drone pose or supplies truth to a neural policy.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROUTE = ((0., 0., 3.), (2., 0., 3.), (2., 2., 3.5), (0., 2., 3.), (0., 0., 3.))
VEHICLES = (
    {"name": "iris_pid", "key": "pid", "label": "PID", "color": [0.04, 0.27, 0.72, 1.], "origin_enu": [-4., 0., 0.], "plugin_port": 9002, "sitl_instance": 0, "mavlink_tcp_port": 5760},
    {"name": "iris_hybrid", "key": "hybrid", "label": "HYBRID", "color": [0.90, 0.32, 0.055, 1.], "origin_enu": [0., 0., 0.], "plugin_port": 9012, "sitl_instance": 1, "mavlink_tcp_port": 5770},
    {"name": "iris_direct", "key": "direct", "label": "END-TO-END", "color": [0.45, 0.12, 0.68, 1.], "origin_enu": [4., 0., 0.], "plugin_port": 9022, "sitl_instance": 2, "mavlink_tcp_port": 5780},
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def text(parent, tag, value, **attributes):
    element = ET.SubElement(parent, tag, attributes)
    element.text = str(value)
    return element


def numbers(values):
    return " ".join(format(float(v), ".17g") for v in values)


def material(visual, color):
    old = visual.find("material")
    if old is not None:
        visual.remove(old)
    item = ET.SubElement(visual, "material")
    text(item, "ambient", numbers([v * .45 for v in color[:3]] + [color[3]]))
    text(item, "diffuse", numbers(color))
    text(item, "specular", "0.08 0.08 0.08 1")
    text(item, "emissive", "0 0 0 1")
    return item


def visual(link, name, shape, dimensions, pose, color):
    item = ET.SubElement(link, "visual", name=name)
    text(item, "pose", numbers(pose))
    geometry = ET.SubElement(item, "geometry")
    primitive = ET.SubElement(geometry, shape)
    if shape == "box":
        text(primitive, "size", numbers(dimensions))
    elif shape == "sphere":
        text(primitive, "radius", dimensions[0])
    elif shape == "cylinder":
        text(primitive, "radius", dimensions[0])
        text(primitive, "length", dimensions[1])
    material(item, color)
    text(item, "cast_shadows", "false")
    return item


def static_link(world, name, pose=(0., 0., 0., 0., 0., 0.)):
    model = ET.SubElement(world, "model", name=name)
    text(model, "static", "true")
    text(model, "pose", numbers(pose))
    return ET.SubElement(model, "link", name="visual_only")


def add_label(world, assets, name, label, center_x, color):
    # Smooth, ordinary system-font lettering on a real two-triangle mesh. This
    # is a static scene material, not a screen overlay or a generated scene.
    from PIL import Image, ImageDraw, ImageFont
    font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "DejaVu Sans"], text=True).strip()
    width = {"pid": 1.20, "hybrid": 2.25, "direct": 3.15}[name]
    height = .55
    image_width = round(280 * width / height)
    font = ImageFont.truetype(font_path, 200)
    image = Image.new("RGB", (image_width, 280), (242, 244, 246))
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), label, font=font)
    if box[2] - box[0] > image_width - 60:
        font = ImageFont.truetype(font_path, round(200 * (image_width - 60) / (box[2] - box[0])))
        box = draw.textbbox((0, 0), label, font=font)
    x, y = (image_width - (box[2] - box[0])) / 2 - box[0], (280 - (box[3] - box[1])) / 2 - box[1]
    draw.text((x, y), label, font=font, fill=tuple(round(v * 210) for v in color[:3]))
    image_path = assets / ("label_" + name + ".png")
    image.save(image_path)
    mesh_path = assets / ("label_" + name + ".dae")
    # COLLADA stores a UV-mapped quad with its normal along +Z. All files are
    # generated beside the copied model assets and referenced explicitly.
    mesh_path.write_text(f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit meter="1" name="meter"/><up_axis>Z_UP</up_axis></asset>
<library_images><image id="image"><init_from>{image_path.name}</init_from></image></library_images>
<library_effects><effect id="effect"><profile_COMMON>
<newparam sid="surface"><surface type="2D"><init_from>image</init_from></surface></newparam>
<newparam sid="sampler"><sampler2D><source>surface</source></sampler2D></newparam>
<technique sid="common"><lambert><emission><color>0 0 0 1</color></emission><ambient><color>.5 .5 .5 1</color></ambient>
<diffuse><texture texture="sampler" texcoord="UVSET0"/></diffuse></lambert></technique>
</profile_COMMON></effect></library_effects>
<library_materials><material id="material"><instance_effect url="#effect"/></material></library_materials>
<library_geometries><geometry id="quad"><mesh>
<source id="positions"><float_array id="positions-array" count="12">{-width/2} {-height/2} 0 {width/2} {-height/2} 0 {width/2} {height/2} 0 {-width/2} {height/2} 0</float_array><technique_common><accessor source="#positions-array" count="4" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="normals"><float_array id="normals-array" count="3">0 0 1</float_array><technique_common><accessor source="#normals-array" count="1" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="uv"><float_array id="uv-array" count="8">0 0 1 0 1 1 0 1</float_array><technique_common><accessor source="#uv-array" count="4" stride="2"><param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>
<vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
<triangles count="2" material="Label"><input semantic="VERTEX" source="#vertices" offset="0"/><input semantic="NORMAL" source="#normals" offset="1"/><input semantic="TEXCOORD" source="#uv" offset="2" set="0"/><p>0 0 0 1 0 1 2 0 2 0 0 0 2 0 2 3 0 3</p></triangles>
</mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="Scene"><node><instance_geometry url="#quad"><bind_material><technique_common><instance_material symbol="Label" target="#material"><bind_vertex_input semantic="UVSET0" input_semantic="TEXCOORD" input_set="0"/></instance_material></technique_common></bind_material></instance_geometry></node></visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#Scene"/></scene></COLLADA>''', encoding="utf-8")
    link = static_link(world, "label_" + name)
    item = ET.SubElement(link, "visual", name="smooth_font_board")
    text(item, "pose", numbers((center_x, -1.0, .025, 0., 0., 0.)))
    geometry = ET.SubElement(item, "geometry")
    # Gazebo's resolver does not decode percent-escaped non-ASCII file URIs.
    text(ET.SubElement(geometry, "mesh"), "uri", str(mesh_path.resolve()))
    text(item, "cast_shadows", "false")


def physics_signature(root):
    """Exclude rendering only, retaining every original physical parameter."""
    root = copy.deepcopy(root)
    for link in root.findall(".//link"):
        for item in list(link.findall("visual")):
            link.remove(item)
    return ET.tostring(root, encoding="unicode")


def build_scene(output_dir, world_name="demo_three", partition="demo_three",
                models_root=None, force_plugin=None):
    for value in (world_name, partition):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("world and partition names must contain only letters, digits, '_' or '-'")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "gazebo_manifest.json").exists():
        raise FileExistsError("Use a fresh output; an existing world manifest is preserved")
    upstream_models = Path(models_root or Path.home() / "ardupilot_gazebo/models").resolve()
    upstream = upstream_models.parent
    force_library = Path(force_plugin or HERE / "force_plugin/build/libThreeDroneForceSystem.so").resolve()
    base_source = upstream_models / "iris_with_standoffs/model.sdf"
    control_source = upstream_models / "iris_with_ardupilot/model.sdf"
    ardupilot_library = upstream / "build/libArduPilotPlugin.so"
    for source in (base_source, control_source, ardupilot_library, force_library, HERE / "gui.config"):
        if not source.is_file():
            raise FileNotFoundError(source)
    assets = out / "gazebo_assets"
    assets.mkdir(exist_ok=False)
    shutil.copytree(upstream_models / "iris_with_standoffs", assets / "iris_with_standoffs")
    shutil.copy2(force_library, assets / force_library.name)
    force_source = HERE / "force_plugin/GroupForceSystem.cc"
    shutil.copy2(force_source, assets / "GroupForceSystem.cc")
    shutil.copy2(HERE / "gui.config", out / "gui.config")
    original_base = ET.parse(base_source).getroot()
    original_control = ET.parse(control_source).getroot()
    total_mass = sum(float(n.text) for n in original_base.findall(".//link/inertial/mass"))
    sdf = ET.Element("sdf", version="1.9")
    world = ET.SubElement(sdf, "world", name=world_name)
    physics = ET.SubElement(world, "physics", name="same_1ms_physics", type="ignore")
    text(physics, "max_step_size", .001)
    text(physics, "real_time_factor", 1.)
    for filename, classname in (("physics", "Physics"), ("user-commands", "UserCommands"),
                                ("scene-broadcaster", "SceneBroadcaster"), ("imu", "Imu"),
                                ("apply-link-wrench", "ApplyLinkWrench")):
        ET.SubElement(world, "plugin", filename="gz-sim-" + filename + "-system", name="gz::sim::systems::" + classname)
    group_force_topic = f"/world/{world_name}/group_force"
    force = ET.SubElement(world, "plugin", filename="ThreeDroneForceSystem", name="demo_three::GroupForceSystem")
    text(force, "topic", group_force_topic)
    text(force, "log_dir", str(out))
    scene = ET.SubElement(world, "scene")
    text(scene, "ambient", ".42 .42 .42 1")
    text(scene, "background", ".82 .86 .90 1")
    text(scene, "shadows", "true")
    text(scene, "grid", "false")
    light = ET.SubElement(world, "light", name="sun", type="directional")
    text(light, "pose", "0 0 12 0 0 0")
    text(light, "diffuse", ".75 .74 .72 1")
    text(light, "specular", ".12 .12 .12 1")
    text(light, "direction", "-.35 .2 -1")
    text(light, "cast_shadows", "true")
    fill = ET.SubElement(world, "light", name="soft_fill", type="directional")
    text(fill, "diffuse", ".18 .19 .21 1")
    text(fill, "direction", ".4 -.2 -1")
    text(fill, "cast_shadows", "false")
    spherical = ET.SubElement(world, "spherical_coordinates")
    for key, value in (("surface_model", "EARTH_WGS84"), ("latitude_deg", -35.363262),
                       ("longitude_deg", 149.165237), ("elevation", 584), ("heading_deg", 0)):
        text(spherical, key, value)
    ground = static_link(world, "ground")
    collision = ET.SubElement(ground, "collision", name="ground_collision")
    geometry = ET.SubElement(collision, "geometry")
    plane = ET.SubElement(geometry, "plane")
    text(plane, "normal", "0 0 1")
    text(plane, "size", "200 200")
    item = ET.SubElement(ground, "visual", name="ground_visual")
    geometry = ET.SubElement(item, "geometry")
    plane = ET.SubElement(geometry, "plane")
    text(plane, "normal", "0 0 1")
    text(plane, "size", "200 200")
    material(item, (.76, .78, .80, 1.))
    vehicles = {}
    for spec in VEHICLES:
        vehicle = copy.deepcopy(spec)
        key, name = vehicle["key"], vehicle["name"]
        ox, oy, oz = vehicle["origin_enu"]
        body_folder = assets / ("iris_body_" + key)
        body_folder.mkdir()
        shutil.copy2(assets / "iris_with_standoffs/model.config", body_folder / "model.config")
        colored = copy.deepcopy(original_base)
        base_link = colored.find("model/link[@name='base_link']")
        # Keep the original Iris frame and all four visible propeller meshes.
        # A small matte canopy identifies the role without hiding the airframe.
        material(base_link.find("visual[@name='base_visual']"), (.10, .12, .15, 1.))
        visual(base_link, "demo_colored_canopy", "box", (.23, .15, .055),
               (0., 0., .055, 0., 0., 0.), vehicle["color"])
        visual(base_link, "demo_front_white", "box", (.035, .075, .008),
               (.095, 0., .086, 0., 0., 0.), (.92, .94, .96, 1.))
        for rotor in colored.findall("model/link"):
            if rotor.get("name", "").startswith("rotor_"):
                for rotor_visual in rotor.findall("visual"):
                    material(rotor_visual, (.075, .085, .10, 1.))
        if physics_signature(colored) != physics_signature(original_base):
            raise AssertionError("Rendering edits changed base physics")
        ET.indent(colored)
        ET.ElementTree(colored).write(body_folder / "model.sdf", encoding="utf-8", xml_declaration=True)
        control_folder = assets / name
        control_folder.mkdir()
        root = copy.deepcopy(original_control)
        model = root.find("model")
        model.set("name", name)
        include = model.find("include")
        include.find("uri").text = "model://iris_body_" + key
        text(include, "name", "iris_with_standoffs")
        model.find("plugin[@name='ArduPilotPlugin']/fdm_port_in").text = str(vehicle["plugin_port"])
        pose_plugin = ET.SubElement(model, "plugin", filename="gz-sim-pose-publisher-system", name="gz::sim::systems::PosePublisher")
        for tag, value in (("publish_link_pose", "true"), ("publish_model_pose", "true"),
                           ("publish_nested_model_pose", "true"), ("use_pose_vector_msg", "true"),
                           ("static_publisher", "false"), ("update_frequency", "50")):
            text(pose_plugin, tag, value)
        # Reverse the declared substitutions and demand exact parent XML.
        reversed_control = copy.deepcopy(root)
        rev_model = reversed_control.find("model")
        rev_model.set("name", original_control.find("model").get("name"))
        rev_model.find("include/uri").text = "model://iris_with_standoffs"
        rev_model.find("include").remove(rev_model.find("include/name"))
        rev_model.find("plugin[@name='ArduPilotPlugin']/fdm_port_in").text = original_control.find("model/plugin[@name='ArduPilotPlugin']/fdm_port_in").text
        rev_model.remove(rev_model.find("plugin[@name='gz::sim::systems::PosePublisher']"))
        if ET.tostring(reversed_control) != ET.tostring(original_control):
            raise AssertionError("Undeclared motor / ArduPilot plugin change")
        ET.indent(root)
        ET.ElementTree(root).write(control_folder / "model.sdf", encoding="utf-8", xml_declaration=True)
        config = ET.Element("model")
        text(config, "name", name)
        text(config, "version", "1.0")
        text(config, "sdf", "model.sdf", version="1.9")
        ET.ElementTree(config).write(control_folder / "model.config", encoding="utf-8", xml_declaration=True)
        include = ET.SubElement(world, "include")
        text(include, "uri", "model://" + name)
        text(include, "name", name)
        text(include, "pose", numbers((ox, oy, .195, 0., 0., math.pi / 2)))
        force_log = out / ("force_physics_" + name + ".jsonl")
        route = [[x + ox, y + oy, z + oz] for x, y, z in ROUTE]
        path_link = static_link(world, "route_" + name)
        for i, (a, b) in enumerate(zip(route, route[1:])):
            dx, dy, dz = (b[j] - a[j] for j in range(3))
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            # Thin 3D reference lines are visuals only; no route collision.
            visual(path_link, "segment_" + str(i), "box", (length, .022, .022),
                   (*[(a[j] + b[j]) / 2 for j in range(3)], 0., -math.atan2(dz, math.hypot(dx, dy)), math.atan2(dy, dx)),
                   (*vehicle["color"][:3], .42))
            visual(path_link, "ground_segment_" + str(i), "box", (math.hypot(dx, dy), .045, .006),
                   ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, .012, 0., 0., math.atan2(dy, dx)), vehicle["color"])
        for i, point in enumerate(route[:-1]):
            visual(path_link, "waypoint_" + str(i), "sphere", (.055,), (*point, 0., 0., 0.), (*vehicle["color"][:3], .68))
        target_name = "target_" + name
        target_link = static_link(world, target_name, (*route[0], 0., 0., 0.))
        visual(target_link, "active_target", "sphere", (.095,), (0., 0., 0., 0., 0., 0.), vehicle["color"])
        add_label(world, assets, key, vehicle["label"], ox + 1., vehicle["color"])
        vehicle.update(route_world_enu=route, target_model=target_name,
                       spawn_world_enu=[ox, oy, .195], spawn_yaw_rad=math.pi / 2,
                       force_topic=group_force_topic, force_log=str(force_log),
                       control_model_sdf=str(control_folder / "model.sdf"), body_model_sdf=str(body_folder / "model.sdf"))
        vehicles[name] = vehicle
    ET.indent(sdf)
    world_path = assets / "world.sdf"
    ET.ElementTree(sdf).write(world_path, encoding="utf-8", xml_declaration=True)
    manifest = {
        "schema": "three-drone-gazebo-scene-v1.0", "status": "BUILT_NO_SIMULATOR_STARTED", "created_utc": utc(),
        "world": world_name, "world_name": world_name, "partition": partition,
        "world_sdf": str(world_path), "gui_config": str(out / "gui.config"), "output": str(out),
        "world_sha256": sha(world_path), "gui_config_sha256": sha(out / "gui.config"),
        "scene_source_sha256": sha(__file__), "vehicles": vehicles,
        "route_local_enu": ROUTE, "translation_only": True,
        "spawn_heading_equal": True, "total_mass_kg_each": total_mass,
        "physics_unchanged_except_rendering_and_independent_ports": True,
        "visual_markers_have_no_collision": True,
        "sources": {str(p): sha(p) for p in (base_source, control_source, ardupilot_library, force_library, force_source)},
        "environment": {"GZ_PARTITION": partition,
                        "GZ_SIM_SYSTEM_PLUGIN_PATH": str(assets) + ":" + str(upstream / "build"),
                        "GZ_SIM_RESOURCE_PATH": str(assets) + ":" + str(upstream_models)},
        "ipc": {"state": "SCENE_STATE.json", "command": "SCENE_COMMAND.json", "stop": "STOP_SHARED_WORLD"},
        "group_force_topic": group_force_topic,
        "group_force_entity_order": list(vehicles),
        "force_scope": "one group EntityWrench, three observed base_link IDs, one PreUpdate all valid or none; actual per-model application ticks remain authoritative",
        "gui_pid": None, "gazebo_pid": None,
    }
    atomic_json(out / "gazebo_manifest.json", manifest)
    return manifest


def serve(manifest, seed=83301, gui=False):
    """Own one real Gazebo server and optional GUI until STOP_SHARED_WORLD."""
    from gz.transport13 import Node
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.entity_wrench_pb2 import EntityWrench
    from gz.msgs10.entity_pb2 import Entity
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.empty_pb2 import Empty
    from gz.msgs10.scene_pb2 import Scene
    from gz.msgs10.world_stats_pb2 import WorldStatistics

    out = Path(manifest["output"])
    env = os.environ.copy()
    env.update(manifest["environment"])
    os.environ["GZ_PARTITION"] = env["GZ_PARTITION"]
    world = manifest["world"]
    stopped = threading.Event()
    lock = threading.Lock()
    models = {}
    model_ids = {}
    stats = {"sim_time": 0., "pose_messages": 0, "force_seq": -1,
             "paused": False, "iterations": 0, "stats_messages": 0}
    truth = (out / "groundtruth.jsonl").open("w", buffering=1)
    events = (out / "scene_events.jsonl").open("w", buffering=1)
    log = (out / "gazebo.log").open("w", buffering=1)
    gui_log = (out / "gazebo_gui.log").open("w", buffering=1)
    command = ["gz", "sim", "-s", "-r", "-v", "3", "--seed", str(seed), manifest["world_sdf"]]
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    gui_process = None
    manifest.update(status="SERVER_STARTED", gazebo_pid=server.pid, worker_pid=os.getpid(), server_command=command)
    atomic_json(out / "gazebo_manifest.json", manifest)
    node = Node()

    def event(value):
        events.write(json.dumps({"wall_time": time.time(), "sim_time": stats["sim_time"], **value}) + "\n")

    def on_pose(message):
        if stopped.is_set():
            return
        stamp = message.header.stamp.sec + message.header.stamp.nsec * 1e-9
        with lock:
            stats["sim_time"] = max(stats["sim_time"], stamp)
            stats["pose_messages"] += 1
            for pose in message.pose:
                name = pose.name
                if name not in manifest["vehicles"]:
                    name = next((key for key, identifier in model_ids.items() if pose.id == identifier), None)
                if name not in manifest["vehicles"]:
                    continue
                old = models.get(name, {})
                value = {"position": [pose.position.x, pose.position.y, pose.position.z],
                         "quaternion_xyzw": [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                         "id": int(pose.id), "base_link_id": old.get("base_link_id"), "sim_time": stamp}
                models[name] = value
                truth.write(json.dumps({"name": name, "wall_time": time.time(),
                                        "receive_monotonic_s": time.monotonic(), "sim_time": stamp, **value}) + "\n")

    node.subscribe(Pose_V, f"/world/{world}/pose/info", on_pose)

    def on_stats(message):
        with lock:
            stats["sim_time"] = max(stats["sim_time"], message.sim_time.sec + message.sim_time.nsec * 1e-9)
            stats["paused"] = bool(message.paused)
            stats["iterations"] = int(message.iterations)
            stats["stats_messages"] += 1

    node.subscribe(WorldStatistics, f"/world/{world}/stats", on_stats)
    force_publisher = node.advertise(manifest["group_force_topic"], EntityWrench)
    if gui:
        gui_env = env.copy()
        gui_env.update(LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe", QT_QPA_PLATFORM="xcb", QT_OPENGL="software")
        gui_env.pop("WAYLAND_DISPLAY", None)
        gui_command = ["gz", "sim", "-g", "-v", "4", "--render-engine-gui", "ogre", "--gui-config", manifest["gui_config"]]
        gui_process = subprocess.Popen(gui_command, stdout=gui_log, stderr=subprocess.STDOUT, env=gui_env, start_new_session=True)
        manifest.update(gui_pid=gui_process.pid, gui_command=gui_command)
        atomic_json(out / "gazebo_manifest.json", manifest)

    def stop(*_):
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_discovery = -1e9
    last_state = -1e9
    current_force = [0., 0., 0.]
    marker_positions = {}
    ipc_failures = 0
    last_command = None

    def discover_links():
        # Resolve nested model hierarchy once. Bare base_link names are ambiguous
        # in a three-drone world and are deliberately never used for association.
        ok, scene = node.request(f"/world/{world}/scene/info", Empty(), Empty, Scene, 500)
        if not ok:
            return
        for parent in scene.model:
            name = parent.name.split("::")[-1]
            if name not in manifest["vehicles"]:
                continue
            found = []

            def visit(model):
                found.extend(int(link.id) for link in model.link if link.name.split("::")[-1] == "base_link")
                for child in model.model:
                    visit(child)

            visit(parent)
            if len(found) != 1:
                raise RuntimeError(f"Expected one physical base_link for {name}, got {found}")
            with lock:
                model_ids[name] = int(parent.id)
                value = models.setdefault(name, {"id": int(parent.id), "position": None, "quaternion_xyzw": None, "sim_time": None})
                value["base_link_id"] = found[0]
            event({"event": "unique_model_link_binding", "name": name, "model_id": int(parent.id), "base_link_id": found[0]})
        if len(model_ids) == 3:
            ids = [models[name]["base_link_id"] for name in manifest["vehicles"]]
            if len(set(ids)) != 3:
                raise RuntimeError("Physical base_link IDs are not unique")

    try:
        while not stopped.is_set() and not (out / "STOP_SHARED_WORLD").exists():
            if server.poll() is not None:
                raise RuntimeError(f"Owned Gazebo server exited {server.returncode}; see gazebo.log")
            now = time.monotonic()
            if len(model_ids) < 3 and now - last_discovery > 1.:
                discover_links()
                last_discovery = now
            request_path = out / "SCENE_COMMAND.json"
            if request_path.exists():
                try:
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                    ipc_failures = 0
                except (OSError, json.JSONDecodeError) as exc:
                    ipc_failures += 1
                    if ipc_failures > 20:
                        raise RuntimeError("Shared scene command unreadable for >100ms") from exc
                    time.sleep(.005)
                    continue
                sequence = int(request["seq"])
                if sequence > stats["force_seq"] and len(model_ids) == 3:
                    desired = [float(v) for v in request.get("force_enu_N", [0., 0., 0.])]
                    if len(desired) != 3 or not all(math.isfinite(v) for v in desired):
                        raise ValueError("force_enu_N must contain three finite world-ENU Newton values")
                    # A single message is applied atomically to all three links
                    # in one PreUpdate by the new demo-only group plugin.
                    message = EntityWrench()
                    message.entity.id = models[manifest["group_force_entity_order"][0]]["base_link_id"]
                    message.entity.type = Entity.LINK
                    message.entity.name = "base_link"
                    field = message.header.data.add()
                    field.key = "seq"
                    field.value.append(str(sequence))
                    identities = message.header.data.add()
                    identities.key = "entity_ids"
                    identities.value.extend(str(models[name]["base_link_id"]) for name in manifest["group_force_entity_order"])
                    message.wrench.force.x, message.wrench.force.y, message.wrench.force.z = desired
                    force_publisher.publish(message)
                    current_force = desired
                    stats["force_seq"] = sequence
                    last_command = request
                    event({"event": "shared_force_command_published", "seq": sequence, "models": list(manifest["vehicles"]),
                           "force_enu_N": desired, "measurement_sim_start_s": request.get("measurement_sim_start_s"),
                           "entity_ids": [models[name]["base_link_id"] for name in manifest["group_force_entity_order"]],
                           "group_message": True, "scope": "one atomic group request; actual per-model physics log determines application tick"})
                for name, target in request.get("targets_world_enu", {}).items():
                    if name not in manifest["vehicles"]:
                        raise ValueError("Unknown target vehicle: " + name)
                    target = tuple(float(v) for v in target)
                    if len(target) != 3 or not all(math.isfinite(v) for v in target):
                        raise ValueError("A visual target needs three finite ENU coordinates")
                    if marker_positions.get(name) == target:
                        continue
                    pose = Pose()
                    pose.name = manifest["vehicles"][name]["target_model"]
                    pose.position.x, pose.position.y, pose.position.z = target
                    pose.orientation.w = 1.
                    ok, response = node.request(f"/world/{world}/set_pose", pose, Pose, Boolean, 100)
                    event({"event": "visual_target_updated", "name": name, "target_world_enu": target, "success": bool(ok and response.data)})
                    if ok and response.data:
                        marker_positions[name] = target
            # The shared 50 Hz controller clock must not be quantized by a
            # second 50 Hz wall-clock IPC timer. Fast shared Linux storage keeps
            # this observer update inexpensive; physics itself is unchanged.
            if now - last_state >= .005:
                with lock:
                    snapshot = copy.deepcopy(models)
                    state_stats = dict(stats)
                atomic_json(out / "SCENE_STATE.json", {
                    "schema": "three-drone-scene-state-v1.0", "updated_utc": utc(), "wall_time": time.time(),
                    "world": world, "partition": manifest["partition"], **state_stats, "models": snapshot,
                    "gazebo_pid": server.pid, "gui_pid": gui_process.pid if gui_process else None,
                    "force_enu_N": current_force, "measurement_sim_start_s": (last_command or {}).get("measurement_sim_start_s"),
                    "all_model_links_observed": len(model_ids) == 3,
                    "gui_alive": gui_process.poll() is None if gui_process else False,
                })
                last_state = now
            time.sleep(.005)
    finally:
        stopped.set()
        for child in (gui_process, server):
            if child is None:
                continue
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        atomic_json(out / "scene_finished.json", {
            "finished_utc": utc(), **stats, "models": models, "server_returncode": server.returncode,
            "gui_returncode": gui_process.returncode if gui_process else None,
            "owned_processes_only": True, "force_enu_N": current_force,
        })
        truth.close()
        events.close()
        log.close()
        gui_log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Fresh output directory shared by runtime and the Gazebo worker")
    parser.add_argument("--world-name", default="demo_three")
    parser.add_argument("--partition", default="demo_three")
    parser.add_argument("--upstream", default=str(Path.home() / "ardupilot_gazebo"))
    parser.add_argument("--force-plugin", default=None)
    parser.add_argument("--seed", type=int, default=83301)
    parser.add_argument("--serve", action="store_true", help="Explicitly start ONE real Gazebo server")
    parser.add_argument("--gui", action="store_true", help="Launch the owned 1280x800 Ogre GUI (requires --serve)")
    args = parser.parse_args(argv)
    if args.gui and not args.serve:
        parser.error("--gui requires --serve")
    manifest = build_scene(args.output, args.world_name, args.partition, Path(args.upstream) / "models", args.force_plugin)
    print(json.dumps({"manifest": str(Path(args.output).resolve() / "gazebo_manifest.json"),
                      "world_sdf": manifest["world_sdf"], "vehicles": list(manifest["vehicles"]),
                      "starting_physics": args.serve}, indent=2), flush=True)
    if args.serve:
        serve(manifest, args.seed, args.gui)


if __name__ == "__main__":
    main()
