"""Match one LSTM result to the camera frame that produced it.

The network output is unchanged. This only decides whether that result may
be drawn on the frame now on screen.
"""

from __future__ import annotations


def accept_newer_result(stored_time_sec: float | None, arrived_time_sec: float | None) -> bool:
    """Keep a worker result only when it is not older than the one already stored."""
    if arrived_time_sec is None or arrived_time_sec != arrived_time_sec:
        return False
    if stored_time_sec is None:
        return True
    return float(arrived_time_sec) >= float(stored_time_sec)


def classify_prediction(
    source_time_sec: float | None,
    current_time_sec: float | None,
    *,
    delay_ms: float,
    max_age_ms: float,
    source_frame: int | None = None,
    current_frame: int | None = None,
) -> dict:
    """Label a prediction current, delayed, stale, or unavailable.

    A delayed result may still be drawn. A stale result is not drawn.
    A result whose source frame is ahead of the displayed frame is not drawn.
    """
    if source_time_sec is None or current_time_sec is None:
        return _decision("unavailable", None, False, source_frame, current_frame)
    if source_time_sec != source_time_sec or current_time_sec != current_time_sec:
        return _decision("unavailable", None, False, source_frame, current_frame)
    age_ms = (float(current_time_sec) - float(source_time_sec)) * 1000.0
    if source_frame is not None and current_frame is not None and int(source_frame) > int(current_frame):
        return _decision("unavailable", age_ms, False, source_frame, current_frame)
    if age_ms < -1.0:
        return _decision("unavailable", age_ms, False, source_frame, current_frame)
    if age_ms > float(max_age_ms):
        return _decision("stale", age_ms, False, source_frame, current_frame)
    if age_ms > float(delay_ms):
        return _decision("delayed", age_ms, True, source_frame, current_frame)
    return _decision("current", age_ms, True, source_frame, current_frame)


def _decision(status: str, age_ms: float | None, draw: bool, source_frame, current_frame) -> dict:
    return {
        "status": status,
        "age_ms": None if age_ms is None else float(age_ms),
        "draw": bool(draw),
        "source_frame": None if source_frame is None else int(source_frame),
        "current_frame": None if current_frame is None else int(current_frame),
    }
