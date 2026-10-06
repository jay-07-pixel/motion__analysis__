"""Day and night colours for the analysis window.

PAL is one shared object. Switching night mode updates it in place so
every screen reads the current colours. Widget backgrounds are still
applied by the window, because Tk copies a colour when the widget is built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Palette:
    """Tk colours for one appearance."""

    bg: str
    panel: str
    card: str
    line: str
    text: str
    muted: str
    accent: str
    start: str
    stop: str
    idle: str
    video: str
    arc: str
    needle: str
    min_c: str
    max_c: str
    mode_c: str
    on_fg: str = "#ffffff"


NIGHT = Palette(
    bg="#101418",
    panel="#171e28",
    card="#1e2734",
    line="#2a3545",
    text="#eef3f8",
    muted="#8b98a8",
    accent="#4c9aff",
    start="#2f9e6a",
    stop="#c44c4c",
    idle="#2a3340",
    video="#07090c",
    arc="#3a4658",
    needle="#4c9aff",
    min_c="#3dd6c8",
    max_c="#e06c5c",
    mode_c="#e8a338",
)

DAY = Palette(
    bg="#f3f5f8",
    panel="#ffffff",
    card="#ffffff",
    line="#d5dde6",
    text="#1a2330",
    muted="#5c6b7a",
    accent="#1d6fdb",
    start="#1f8a56",
    stop="#c44747",
    idle="#e4eaf1",
    video="#e7edf3",
    arc="#c5d0dc",
    needle="#1d6fdb",
    min_c="#0e8f86",
    max_c="#c44747",
    mode_c="#b57412",
)

PAL = Palette(**NIGHT.__dict__)


def use_night(on: bool) -> None:
    """Copy the night or day palette into PAL."""
    source = NIGHT if on else DAY
    PAL.__dict__.update(source.__dict__)
