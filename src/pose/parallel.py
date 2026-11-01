"""Run Pose and Hands on one frame at the same time.

Each model keeps its own graph and its own previous-frame state. The image
is the current frame for both. A hand result is not carried onto a later frame.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


def run_together(pose_call: Callable[[], T], hands_call: Callable[[], U]) -> tuple[T, U]:
    """Start hands, run pose on this thread, then wait for hands.

    The two calls overlap. Their return values are unchanged by the overlap.
    """
    box: dict[str, U] = {}
    error: list[BaseException] = []

    def _hands() -> None:
        try:
            box["value"] = hands_call()
        except BaseException as exc:
            error.append(exc)

    worker = threading.Thread(target=_hands, name="mediapipe-hands", daemon=True)
    worker.start()
    try:
        pose_value = pose_call()
    finally:
        worker.join()
    if error:
        raise error[0]
    return pose_value, box["value"]
