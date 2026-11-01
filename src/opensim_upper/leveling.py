"""Measure camera up from the RealSense accelerometer.

A static accelerometer reads the normal force, which points opposite gravity,
so the mean reading is up. Image Y is not used. The accelerometer frame is
rotated into the colour camera with the device extrinsics. Translation is
not added: acceleration is a direction, not a point.

The live colour/depth pipeline is left unchanged. A reading opens its own
short pipeline and must run only while that camera is free.
"""

from __future__ import annotations

import numpy as np

GRAVITY_M_S2 = 9.80665
ACCEL_MAGNITUDE_MIN = 8.5
ACCEL_MAGNITUDE_MAX = 11.0
ACCEL_STILL_STD_MAX = 0.35
MIN_ACCEL_SAMPLES = 20


class LevelingError(Exception):
    """The camera up direction was not measured."""


def rotate_motion_to_color(vector_m_s2, rotation_column_major) -> np.ndarray:
    """Apply an rs2_extrinsics rotation. The 9 values are column-major."""
    rotation = np.asarray(rotation_column_major, dtype=float).reshape((3, 3), order="F")
    return rotation @ np.asarray(vector_m_s2, dtype=float)


def assess_static_accel(samples_m_s2) -> dict:
    """Accept a still accelerometer reading and return its up direction.

    ``samples_m_s2`` are already in the colour camera frame.
    """
    samples = np.asarray(samples_m_s2, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise LevelingError("Accelerometer samples must be Nx3.")
    if len(samples) < MIN_ACCEL_SAMPLES:
        raise LevelingError(
            f"Need at least {MIN_ACCEL_SAMPLES} still accelerometer samples. "
            f"Got {len(samples)}."
        )
    if not np.isfinite(samples).all():
        raise LevelingError("Accelerometer samples contain a non-finite value.")
    spread = samples.std(axis=0)
    if float(np.max(spread)) > ACCEL_STILL_STD_MAX:
        raise LevelingError(
            "The camera was moving. Leave it still and measure up again. "
            f"Largest axis std was {float(np.max(spread)):.3f} m/s^2."
        )
    mean = samples.mean(axis=0)
    magnitude = float(np.linalg.norm(mean))
    if magnitude < ACCEL_MAGNITUDE_MIN or magnitude > ACCEL_MAGNITUDE_MAX:
        raise LevelingError(
            "The accelerometer magnitude is not gravity "
            f"({magnitude:.3f} m/s^2). Up was not stored. Image Y was not used."
        )
    up = mean / magnitude
    return {
        "up_camera": up,
        "magnitude_m_s2": magnitude,
        "n_samples": int(len(samples)),
        "std_m_s2": [float(value) for value in spread],
        "source": "realsense_accel_rotated_to_color",
        "image_y_used_as_gravity": False,
    }


def _accel_to_color_rotation():
    """Column-major rotation from the accelerometer into the colour camera.

    Read from the device profile. No colour stream is started, so the live
    colour/depth pipeline is not reconfigured.
    """
    import pyrealsense2 as rs

    try:
        devices = rs.context().query_devices()
    except Exception as error:
        raise LevelingError(f"Could not query RealSense devices. Detail: {error}") from error
    if len(devices) == 0:
        raise LevelingError(
            "Camera not connected. Plug in the D455f on USB 3 and close RealSense Viewer."
        )
    color_profile = None
    accel_profile = None
    for sensor in devices[0].query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.accel and accel_profile is None:
                accel_profile = profile
            if profile.stream_type() == rs.stream.color and color_profile is None:
                color_profile = profile
    if accel_profile is None:
        raise LevelingError(
            "This RealSense has no accelerometer stream. Camera up was not measured. "
            "Image Y was not used as gravity."
        )
    if color_profile is None:
        raise LevelingError(
            "Could not find a colour stream profile, so the accelerometer cannot be "
            "rotated into the colour camera. Up was not measured."
        )
    return list(accel_profile.get_extrinsics_to(color_profile).rotation)


def read_static_up_color(n_samples: int = 40, timeout_sec: float = 8.0) -> dict:
    """Read a still accelerometer and return up in the colour camera frame.

    Opens an accelerometer-only pipeline. Fails if the camera is already in use.
    """
    import time

    import pyrealsense2 as rs

    if n_samples < MIN_ACCEL_SAMPLES:
        raise LevelingError(f"n_samples must be at least {MIN_ACCEL_SAMPLES}.")
    rotation = _accel_to_color_rotation()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel)
    try:
        pipeline.start(config)
    except Exception as error:
        raise LevelingError(
            "Could not read the RealSense accelerometer. Close RealSense Viewer "
            "and stop the live session, then measure up again. "
            f"Detail: {error}"
        ) from error
    try:
        raw_samples = []
        warmup = 10
        seen = 0
        deadline = time.monotonic() + float(timeout_sec)
        while len(raw_samples) < n_samples and time.monotonic() < deadline:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=500)
            except Exception:
                continue
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame is None:
                continue
            motion = accel_frame.as_motion_frame().get_motion_data()
            seen += 1
            if seen <= warmup:
                continue
            sample = (float(motion.x), float(motion.y), float(motion.z))
            raw_samples.append(rotate_motion_to_color(sample, rotation))
        if len(raw_samples) < MIN_ACCEL_SAMPLES:
            raise LevelingError(
                "This RealSense did not provide enough still accelerometer frames. "
                "Camera up was not measured. Image Y was not used as gravity."
            )
        assessed = assess_static_accel(raw_samples[:n_samples])
        assessed["extrinsics_rotation_column_major"] = [float(value) for value in rotation]
        assessed["extrinsics_translation_ignored"] = True
        return assessed
    finally:
        pipeline.stop()
