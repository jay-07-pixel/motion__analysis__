"""Interactive 3D view of one IK fit.

The page is a single HTML file. Drag turns the skeleton, the scroll wheel
zooms, and each marker row is the gap between the recorded point and the
model marker. Bone meshes are the same posed files as the still picture.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from src.opensim_upper.bone_figure import stand_points
from src.opensim_upper.model_xml import UPPER_MODEL_MARKERS

MARKER_LABELS = {
    "RShoulder": "Right shoulder",
    "LShoulder": "Left shoulder",
    "RElbow": "Right elbow",
    "LElbow": "Left elbow",
    "r_lelbow_study": "Right elbow lateral",
    "r_melbow_study": "Right elbow medial",
    "r_lwrist_study": "Right wrist lateral",
    "r_mwrist_study": "Right wrist medial",
    "L_lelbow_study": "Left elbow lateral",
    "L_melbow_study": "Left elbow medial",
    "L_lwrist_study": "Left wrist lateral",
    "L_mwrist_study": "Left wrist medial",
}

_JOINT_WORDS = {
    "torso": "Torso",
    "head": "Head",
    "humerus": "upper arm",
    "ulna": "ulna",
    "radius": "radius",
    "hand": "hand",
    "pelvis": "Pelvis",
    "sacrum": "Sacrum",
    "femur": "thigh",
    "tibia": "shin",
    "talus": "ankle",
    "calcn": "heel",
    "toes": "toes",
}


def joint_label(name: str) -> str:
    """Short name for a body origin, such as ``Right thigh``."""
    side = ""
    body = name
    if name.endswith("_r"):
        side = "Right "
        body = name[:-2]
    elif name.endswith("_l"):
        side = "Left "
        body = name[:-2]
    word = _JOINT_WORDS.get(body, body)
    if side and word[:1].isupper():
        word = word[:1].lower() + word[1:]
    text = (side + word).strip()
    if not text:
        return name
    return text[:1].upper() + text[1:]


def marker_label(name: str) -> str:
    return MARKER_LABELS.get(name, name)


def _finite_point(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    point = np.asarray(value, dtype=float).reshape(3)
    if not np.isfinite(point).all():
        return None
    return (float(point[0]), float(point[1]), float(point[2]))


def _show(point: tuple[float, float, float] | None, transforms: dict | None) -> list[float] | None:
    if point is None:
        return None
    array = np.asarray(point, dtype=float).reshape(1, 3)
    if transforms and "torso" in transforms:
        shown = stand_points(array, transforms).reshape(3)
    else:
        shown = array.copy()
        shown[:, 1] *= -1.0
        shown = shown.reshape(3)
    return [round(float(value), 5) for value in shown]


def fit_markers(
    recorded_points: dict,
    model_points: dict,
    used_in_ik: dict | None = None,
) -> list[dict]:
    """One row per marker name. ``error_mm`` is the 3D gap, or None.

    ``used_in_ik`` is the same frame from ``markers_ik.trc``. A blank there
    means the point was left out of the solve (spacing/step gate).
    """
    order = {name: index for index, name in enumerate(UPPER_MODEL_MARKERS)}
    names = sorted(
        set(recorded_points) | set(model_points),
        key=lambda name: (order.get(name, 999), name),
    )
    rows = []
    for name in names:
        recorded = _finite_point(recorded_points.get(name))
        model = _finite_point(model_points.get(name))
        used = True
        if used_in_ik is not None:
            used = _finite_point(used_in_ik.get(name)) is not None
        error = None
        if recorded is not None and model is not None:
            error = float(np.linalg.norm(np.subtract(recorded, model)) * 1000.0)
        rows.append(
            {
                "name": name,
                "label": marker_label(name),
                "recorded_stored": recorded,
                "model_stored": model,
                "error_mm": None if error is None else round(error, 2),
                "used_in_ik": used,
            }
        )
    return rows


def build_fit_scene(
    recorded: dict,
    model_points: dict,
    transforms: dict | None,
    model_path: Path | None = None,
    bodies: dict | None = None,
    take_rms_mm: float | None = None,
    used_in_ik: dict | None = None,
) -> dict:
    """Posed bones and named points for the still picture and the 3D page."""
    triangles = None
    if transforms and "torso" in transforms:
        from src.opensim_upper.bone_figure import merged_body_meshes, posed_triangles
        from src.opensim_upper.camera_model import SOURCE_MODEL
        from src.opensim_upper.environment import POSE2SIM_GEOMETRY

        if SOURCE_MODEL.is_file():
            raw = posed_triangles(
                transforms,
                merged_body_meshes(SOURCE_MODEL, model_path),
                POSE2SIM_GEOMETRY,
            )
            if raw is not None and len(raw):
                triangles = stand_points(raw, transforms)

    joints = []
    sources: list[tuple[str, tuple]] = []
    if transforms:
        sources = [(name, origin) for name, (_rotation, origin) in transforms.items()]
    elif bodies:
        sources = list(bodies.items())
    for name, origin in sources:
        shown = _show(_finite_point(origin), transforms)
        if shown is None:
            continue
        joints.append({"name": name, "label": joint_label(name), "position": shown})

    markers = []
    for row in fit_markers(recorded.get("points") or {}, model_points, used_in_ik=used_in_ik):
        markers.append(
            {
                "name": row["name"],
                "label": row["label"],
                "recorded": _show(row["recorded_stored"], transforms),
                "model": _show(row["model_stored"], transforms),
                "error_mm": row["error_mm"],
                "used_in_ik": bool(row["used_in_ik"]),
            }
        )
    # Picture RMS uses only points that entered the IK solve on this frame.
    errors = [
        row["error_mm"]
        for row in markers
        if row["error_mm"] is not None and row.get("used_in_ik", True)
    ]
    rms = None
    if errors:
        rms = float(np.sqrt(np.mean(np.square(errors))))
    left_out = sum(1 for row in markers if row.get("used_in_ik") is False and row["recorded"] is not None)

    bones = []
    if triangles is None and joints:
        from src.opensim_upper.visualize import _BONE_LINKS

        by_name = {item["name"]: item["position"] for item in joints}
        for start, stop in _BONE_LINKS:
            if start in by_name and stop in by_name:
                bones.append({"a": by_name[start], "b": by_name[stop]})

    cloud_parts = []
    if triangles is not None:
        cloud_parts.append(np.asarray(triangles, dtype=float).reshape(-1, 3))
    point_cloud = [item["position"] for item in joints]
    point_cloud.extend(row["recorded"] for row in markers if row["recorded"] is not None)
    point_cloud.extend(row["model"] for row in markers if row["model"] is not None)
    if point_cloud:
        cloud_parts.append(np.asarray(point_cloud, dtype=float))
    if cloud_parts:
        cloud = np.concatenate(cloud_parts, axis=0)
        low = cloud.min(axis=0)
        high = cloud.max(axis=0)
        center = [round(float(value), 5) for value in (low + high) / 2.0]
        span = float(max(np.max(high - low), 0.3))
    else:
        center = [0.0, 0.0, 1.6]
        span = 1.0
    return {
        "time_sec": float(recorded.get("time_sec") or 0.0),
        "triangles": triangles,
        "joints": joints,
        "markers": markers,
        "bones": bones,
        "center": center,
        "span": span,
        "rms_mm": None if rms is None else round(rms, 2),
        "take_rms_mm": None if take_rms_mm is None else round(float(take_rms_mm), 2),
        "n_left_out": int(left_out),
    }


def write_fit_html(scene: dict, path: Path) -> Path:
    """Write a self-contained page that can be rotated in a browser."""
    positions = scene.get("triangles")
    if positions is None or len(positions) == 0:
        encoded = ""
    else:
        raw = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1), dtype=np.float32)
        encoded = base64.b64encode(raw.tobytes()).decode("ascii")
    payload = {
        "time_sec": round(float(scene["time_sec"]), 3),
        "positions_b64": encoded,
        "joints": scene["joints"],
        "markers": scene["markers"],
        "bones": scene.get("bones") or [],
        "center": scene["center"],
        "span": round(float(scene["span"]), 5),
        "rms_mm": scene["rms_mm"],
        "take_rms_mm": scene.get("take_rms_mm"),
        "n_left_out": scene.get("n_left_out") or 0,
    }
    page = _PAGE.replace("__FIT_JSON__", json.dumps(payload, separators=(",", ":")))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenSim fit</title>
<style>
  html, body { margin: 0; height: 100%; background: #f7f4f0; color: #1c1c1c;
    font-family: "Segoe UI", Arial, sans-serif; }
  body { display: flex; min-height: 100%; }
  #view { flex: 1; position: relative; min-width: 0; }
  canvas { width: 100%; height: 100%; display: block; touch-action: none; }
  #tip { position: absolute; pointer-events: none; background: #fff; border: 1px solid #ddd;
    border-radius: 8px; padding: 8px 10px; font-size: 13px; display: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08); max-width: 240px; }
  #tip b { display: block; font-size: 14px; }
  #tip span { color: #555; }
  aside { width: 320px; box-sizing: border-box; padding: 16px 16px 20px;
    background: #fff; border-left: 1px solid #ece7e1; overflow: auto; }
  h1 { font-size: 16px; font-weight: 650; margin: 0 0 6px; }
  p { margin: 0 0 8px; color: #444; font-size: 13px; line-height: 1.4; }
  .rms { font-size: 18px; margin: 10px 0 12px; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px 12px; font-size: 13px; margin-bottom: 12px; }
  button.reset { border: 0; background: #1f4e79; color: #fff; border-radius: 6px;
    padding: 6px 10px; font: inherit; cursor: pointer; }
  #markers { display: flex; flex-direction: column; gap: 4px; }
  #markers button { display: flex; justify-content: space-between; gap: 8px; text-align: left;
    border: 0; background: #f6f3ef; border-radius: 6px; padding: 7px 8px; font: inherit; cursor: pointer; }
  #markers button.on { background: #f3e0d4; }
  #markers b { font-weight: 650; white-space: nowrap; }
  .swatch { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
</style>
</head>
<body>
<div id="view">
  <canvas id="canvas"></canvas>
  <div id="tip"></div>
</div>
<aside>
  <h1 id="title">OpenSim fit</h1>
  <p>Not a clinical measurement. Shown standing. Legs are the model pose. Arms and points follow this fit.</p>
  <p>Drag to turn. Scroll to zoom. Right-drag to slide. The right side of the starting view is the person's right.</p>
  <div class="rms" id="rms"></div>
  <div class="controls">
    <label><input type="checkbox" id="show-bones" checked> bones</label>
    <label><input type="checkbox" id="show-joints" checked> <i class="swatch" style="background:#0b1f3a"></i>joints</label>
    <label><input type="checkbox" id="show-model" checked> <i class="swatch" style="background:#2a6f97"></i>model</label>
    <label><input type="checkbox" id="show-recorded" checked> <i class="swatch" style="background:#c45c26"></i>recorded</label>
    <label><input type="checkbox" id="show-gaps" checked> gaps</label>
    <label><input type="checkbox" id="show-names"> names</label>
    <label><input type="checkbox" id="show-through" checked> keep points visible</label>
  </div>
  <button class="reset" id="reset" type="button">Reset view</button>
  <div id="markers" style="margin-top:12px"></div>
</aside>
<script>
const FIT = __FIT_JSON__;
const canvas = document.getElementById("canvas");
const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
if (!gl) {
  document.body.textContent = "This browser has no WebGL, so the 3D fit cannot be shown.";
  throw new Error("webgl");
}
const tip = document.getElementById("tip");
document.getElementById("title").textContent = "Full OpenSim skeleton at " + FIT.time_sec.toFixed(2) + " s";
const rms = document.getElementById("rms");
if (FIT.take_rms_mm != null) {
  rms.innerHTML = "Whole take IK gap (RMS): <b>" + FIT.take_rms_mm.toFixed(1) + " mm</b>"
    + (FIT.rms_mm == null ? "" : "<br><span style='font-size:14px;color:#555'>This picture (points used in IK): " + FIT.rms_mm.toFixed(1) + " mm</span>")
    + (FIT.n_left_out ? "<br><span style='font-size:13px;color:#8a4b2e'>" + FIT.n_left_out + " recorded point(s) left out of IK (spacing/step gate)</span>" : "");
} else {
  rms.textContent = FIT.rms_mm == null ? "No paired markers." : "Average gap (RMS): " + FIT.rms_mm.toFixed(1) + " mm";
}

const list = document.getElementById("markers");
const rows = [];
const nameTags = FIT.markers.map((marker) => {
  const tag = document.createElement("div");
  tag.textContent = marker.label;
  tag.style.position = "absolute";
  tag.style.transform = "translate(-50%, -100%)";
  tag.style.fontSize = "11px";
  tag.style.color = "#222";
  tag.style.background = "rgba(255,255,255,0.86)";
  tag.style.borderRadius = "4px";
  tag.style.padding = "1px 4px";
  tag.style.pointerEvents = "none";
  tag.style.whiteSpace = "nowrap";
  tag.style.display = "none";
  document.getElementById("view").appendChild(tag);
  return tag;
});
FIT.markers.forEach((marker, index) => {
  const button = document.createElement("button");
  button.type = "button";
  const name = document.createElement("span");
  name.textContent = marker.label;
  const gap = document.createElement("b");
  if (marker.used_in_ik === false) {
    gap.textContent = marker.error_mm == null ? "left out" : marker.error_mm.toFixed(1) + " mm · left out";
    button.style.opacity = "0.72";
  } else {
    gap.textContent = marker.error_mm == null ? "—" : marker.error_mm.toFixed(1) + " mm";
  }
  button.appendChild(name);
  button.appendChild(gap);
  button.addEventListener("click", () => selectMarker(index));
  list.appendChild(button);
  rows.push(button);
});

function b64f32(text) {
  if (!text) return new Float32Array(0);
  const bin = atob(text);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm(v) {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}
function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}
function basis(eye, target) {
  let f = norm(sub(target, eye));
  const up = Math.abs(f[1]) > 0.95 ? [0, 0, -1] : [0, 1, 0];
  let r = norm(cross(up, f));
  const u = cross(f, r);
  return { f, r, u };
}
function viewMatrix(eye, target) {
  const { f, r, u } = basis(eye, target);
  const m = new Float32Array(16);
  m[0] = r[0]; m[4] = r[1]; m[8] = r[2]; m[12] = -dot(r, eye);
  m[1] = u[0]; m[5] = u[1]; m[9] = u[2]; m[13] = -dot(u, eye);
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2]; m[14] = dot(f, eye);
  m[15] = 1;
  return m;
}
function mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

const vsSource = `
attribute vec3 aPos;
attribute vec3 aNrm;
uniform mat4 uMvp;
uniform float uSize;
varying vec3 vN;
void main() {
  vN = aNrm;
  gl_Position = uMvp * vec4(aPos, 1.0);
  gl_PointSize = uSize;
}`;
const fsSource = `
precision mediump float;
varying vec3 vN;
uniform vec3 uColor;
uniform float uMode;
void main() {
  if (uMode > 1.5) {
    gl_FragColor = vec4(uColor, 1.0);
    return;
  }
  if (uMode > 0.5) {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = dot(c, c);
    if (r > 1.0) discard;
    if (r > 0.58) gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    else gl_FragColor = vec4(uColor, 1.0);
    return;
  }
  vec3 n = normalize(vN);
  vec3 key = normalize(vec3(0.25, 0.72, -0.65));
  vec3 fill = normalize(vec3(-0.45, 0.2, -0.4));
  float light = 0.38 + 0.46 * abs(dot(n, key)) + 0.18 * abs(dot(n, fill));
  gl_FragColor = vec4(uColor * light, 1.0);
}`;
function shader(type, source) {
  const item = gl.createShader(type);
  gl.shaderSource(item, source);
  gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(item) || "shader");
  }
  return item;
}
const program = gl.createProgram();
gl.attachShader(program, shader(gl.VERTEX_SHADER, vsSource));
gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fsSource));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
  throw new Error(gl.getProgramInfoLog(program) || "link");
}
gl.useProgram(program);
const aPos = gl.getAttribLocation(program, "aPos");
const aNrm = gl.getAttribLocation(program, "aNrm");
const uMvp = gl.getUniformLocation(program, "uMvp");
const uColor = gl.getUniformLocation(program, "uColor");
const uMode = gl.getUniformLocation(program, "uMode");
const uSize = gl.getUniformLocation(program, "uSize");

const positions = b64f32(FIT.positions_b64);
const mesh = gl.createBuffer();
let vertexCount = 0;
if (positions.length) {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 9) {
    const ax = positions[i + 3] - positions[i];
    const ay = positions[i + 4] - positions[i + 1];
    const az = positions[i + 5] - positions[i + 2];
    const bx = positions[i + 6] - positions[i];
    const by = positions[i + 7] - positions[i + 1];
    const bz = positions[i + 8] - positions[i + 2];
    let nx = ay * bz - az * by;
    let ny = az * bx - ax * bz;
    let nz = ax * by - ay * bx;
    const length = Math.hypot(nx, ny, nz) || 1;
    nx /= length; ny /= length; nz /= length;
    for (let k = 0; k < 9; k += 3) {
      normals[i + k] = nx;
      normals[i + k + 1] = ny;
      normals[i + k + 2] = nz;
    }
  }
  const interleaved = new Float32Array((positions.length / 3) * 6);
  vertexCount = positions.length / 3;
  for (let v = 0; v < vertexCount; v++) {
    interleaved[v * 6] = positions[v * 3];
    interleaved[v * 6 + 1] = positions[v * 3 + 1];
    interleaved[v * 6 + 2] = positions[v * 3 + 2];
    interleaved[v * 6 + 3] = normals[v * 3];
    interleaved[v * 6 + 4] = normals[v * 3 + 1];
    interleaved[v * 6 + 5] = normals[v * 3 + 2];
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, mesh);
  gl.bufferData(gl.ARRAY_BUFFER, interleaved, gl.STATIC_DRAW);
}
const dynamic = gl.createBuffer();

const fov = 32 * Math.PI / 180;
let target = FIT.center.slice();
let azimuth = -0.18;
let elevation = 0.08;
function fitDistance() { return (FIT.span * 0.62) / Math.tan(fov / 2); }
let distance = fitDistance();
const home = { target: target.slice(), azimuth, elevation, distance };
let selected = -1;

function eyePosition() {
  const ce = Math.cos(elevation);
  return [
    target[0] + distance * Math.sin(azimuth) * ce,
    target[1] + distance * Math.sin(elevation),
    target[2] - distance * Math.cos(azimuth) * ce,
  ];
}
function checked(id) { return document.getElementById(id).checked; }
function selectMarker(index) {
  selected = index;
  rows.forEach((row, rowIndex) => row.classList.toggle("on", rowIndex === index));
  const marker = FIT.markers[index];
  const a = marker.recorded || marker.model;
  const b = marker.model || marker.recorded;
  if (!a) return;
  target = b ? [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2] : a.slice();
  distance = Math.min(distance, Math.max(0.35, (marker.error_mm || 20) / 1000 * 18));
}
document.getElementById("reset").addEventListener("click", () => {
  selected = -1;
  rows.forEach((row) => row.classList.remove("on"));
  target = home.target.slice();
  azimuth = home.azimuth;
  elevation = home.elevation;
  distance = home.distance;
});

let dragging = 0;
let lastX = 0;
let lastY = 0;
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  dragging = event.button === 2 || event.shiftKey ? 2 : 1;
  lastX = event.clientX;
  lastY = event.clientY;
});
canvas.addEventListener("pointerup", () => { dragging = 0; });
canvas.addEventListener("pointerleave", () => { dragging = 0; });
canvas.addEventListener("pointermove", (event) => {
  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  if (dragging === 1) {
    azimuth -= dx * 0.005;
    elevation = Math.max(-1.2, Math.min(1.2, elevation + dy * 0.005));
  } else if (dragging === 2) {
    const eye = eyePosition();
    const { r, u } = basis(eye, target);
    const scale = distance * 0.0016;
    target[0] -= r[0] * dx * scale - u[0] * dy * scale;
    target[1] -= r[1] * dx * scale - u[1] * dy * scale;
    target[2] -= r[2] * dx * scale - u[2] * dy * scale;
  } else {
    hover(event.clientX, event.clientY);
  }
});
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  distance = Math.max(0.12, Math.min(12, distance * Math.exp(event.deltaY * 0.001)));
}, { passive: false });

function project(point, mvp, width, height) {
  const x = mvp[0] * point[0] + mvp[4] * point[1] + mvp[8] * point[2] + mvp[12];
  const y = mvp[1] * point[0] + mvp[5] * point[1] + mvp[9] * point[2] + mvp[13];
  const z = mvp[2] * point[0] + mvp[6] * point[1] + mvp[10] * point[2] + mvp[14];
  const w = mvp[3] * point[0] + mvp[7] * point[1] + mvp[11] * point[2] + mvp[15];
  if (w <= 0) return null;
  return { x: (x / w + 1) * 0.5 * width, y: (1 - y / w) * 0.5 * height, z: z / w };
}
let lastMvp = null;
function hover(clientX, clientY) {
  if (!lastMvp) return;
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  let best = null;
  FIT.markers.forEach((marker) => {
    [marker.recorded, marker.model].forEach((point) => {
      if (!point) return;
      const screen = project(point, lastMvp, rect.width, rect.height);
      if (!screen) return;
      const d = Math.hypot(screen.x - x, screen.y - y);
      if (d < 16 && (!best || d < best.d)) best = { d, marker, screen };
    });
  });
  if (!best) {
    tip.style.display = "none";
    return;
  }
  tip.style.display = "block";
  tip.style.left = (best.screen.x + 14) + "px";
  tip.style.top = (best.screen.y + 14) + "px";
  tip.replaceChildren();
  const title = document.createElement("b");
  title.textContent = best.marker.label;
  const detail = document.createElement("span");
  const gap = best.marker.error_mm == null ? "unpaired" : best.marker.error_mm.toFixed(1) + " mm between recorded and model";
  detail.textContent = best.marker.name + " — " + gap;
  tip.appendChild(title);
  tip.appendChild(detail);
}

function flatten(items) {
  const out = [];
  items.forEach((point) => { if (point) out.push(point[0], point[1], point[2]); });
  return new Float32Array(out);
}
function drawArrays(mode, data, color, drawMode, size) {
  if (!data.length) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, dynamic);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 12, 0);
  gl.disableVertexAttribArray(aNrm);
  gl.vertexAttrib3f(aNrm, 0, 1, 0);
  gl.uniform3fv(uColor, color);
  gl.uniform1f(uMode, mode);
  gl.uniform1f(uSize, size);
  gl.drawArrays(drawMode, 0, data.length / 3);
}
function pointsOf(key) {
  const out = [];
  FIT.markers.forEach((marker, index) => {
    if (index === selected) return;
    const point = marker[key];
    if (point) out.push(point[0], point[1], point[2]);
  });
  return new Float32Array(out);
}

function frame() {
  const width = Math.max(1, Math.floor(canvas.clientWidth * Math.min(window.devicePixelRatio || 1, 2)));
  const height = Math.max(1, Math.floor(canvas.clientHeight * Math.min(window.devicePixelRatio || 1, 2)));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  gl.viewport(0, 0, width, height);
  gl.clearColor(0.969, 0.957, 0.941, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);
  const eye = eyePosition();
  const near = Math.max(0.02, distance * 0.02);
  const far = distance + FIT.span * 6;
  const mvp = mul(perspective(fov, width / height, near, far), viewMatrix(eye, target));
  lastMvp = mvp;
  gl.useProgram(program);
  gl.uniformMatrix4fv(uMvp, false, mvp);
  if (checked("show-bones") && vertexCount) {
    gl.depthMask(true);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh);
    gl.enableVertexAttribArray(aPos);
    gl.enableVertexAttribArray(aNrm);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 24, 0);
    gl.vertexAttribPointer(aNrm, 3, gl.FLOAT, false, 24, 12);
    gl.uniform1f(uMode, 0);
    gl.uniform3f(uColor, 0.84, 0.81, 0.77);
    gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
  } else if (checked("show-bones") && FIT.bones.length) {
    const stick = [];
    FIT.bones.forEach((bone) => stick.push(bone.a[0], bone.a[1], bone.a[2], bone.b[0], bone.b[1], bone.b[2]));
    drawArrays(2, new Float32Array(stick), [0.04, 0.12, 0.23], gl.LINES, 1);
  }
  const through = checked("show-through");
  if (through) gl.disable(gl.DEPTH_TEST);
  if (checked("show-gaps")) {
    const gaps = [];
    FIT.markers.forEach((marker) => {
      if (!marker.recorded || !marker.model) return;
      gaps.push(marker.recorded[0], marker.recorded[1], marker.recorded[2]);
      gaps.push(marker.model[0], marker.model[1], marker.model[2]);
    });
    drawArrays(2, new Float32Array(gaps), [0.72, 0.2, 0.14], gl.LINES, 1);
  }
  if (checked("show-joints")) {
    drawArrays(1, flatten(FIT.joints.map((joint) => joint.position)), [0.043, 0.122, 0.227], gl.POINTS, 9);
  }
  if (checked("show-model")) drawArrays(1, pointsOf("model"), [0.165, 0.435, 0.592], gl.POINTS, 13);
  if (checked("show-recorded")) drawArrays(1, pointsOf("recorded"), [0.769, 0.361, 0.149], gl.POINTS, 15);
  if (selected >= 0) {
    const marker = FIT.markers[selected];
    if (checked("show-model") && marker.model) drawArrays(1, new Float32Array(marker.model), [0.165, 0.435, 0.592], gl.POINTS, 20);
    if (checked("show-recorded") && marker.recorded) drawArrays(1, new Float32Array(marker.recorded), [0.769, 0.361, 0.149], gl.POINTS, 22);
  }
  if (through) gl.enable(gl.DEPTH_TEST);
  const rect = canvas.getBoundingClientRect();
  FIT.markers.forEach((marker, index) => {
    const tag = nameTags[index];
    const point = marker.recorded || marker.model;
    const screen = point && checked("show-names") ? project(point, mvp, rect.width, rect.height) : null;
    if (!screen || screen.z < -1 || screen.z > 1) {
      tag.style.display = "none";
      return;
    }
    tag.style.display = "block";
    tag.style.left = screen.x + "px";
    tag.style.top = (screen.y - 16) + "px";
  });
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
</body>
</html>
"""
