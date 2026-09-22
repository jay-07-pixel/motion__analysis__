"""Build the short progress PPT (16:9). Slide 1 first; 2–3 can be added later."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

# Professional palette (not rainbow: navy + teal + amber + coral).
NAVY = RGBColor(0x0B, 0x1F, 0x3A)
NAVY_MID = RGBColor(0x14, 0x32, 0x58)
TEAL = RGBColor(0x1F, 0xB5, 0xA6)
AMBER = RGBColor(0xE8, 0xA3, 0x38)
CORAL = RGBColor(0xE0, 0x6C, 0x5C)
CREAM = RGBColor(0xF6, 0xF3, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x23, 0x32)
MUTED = RGBColor(0x5A, 0x66, 0x78)
CARD = RGBColor(0xFF, 0xFF, 0xFF)

OUT = Path(__file__).resolve().parent / "Progress_Update_2D.pptx"
OUT_ALT = Path(__file__).resolve().parent / "Progress_Update_2D_full.pptx"


def _set_run(run, text: str, size: int, color: RGBColor, bold: bool = False) -> None:
    """Apply Calibri + size + colour to one text run."""
    run.text = text
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold


def _fill(shape, color: RGBColor) -> None:
    """Solid fill, no line."""
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _round_card(slide, left, top, width, height, fill: RGBColor):
    """Rounded rectangle used as a content card."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    _fill(shape, fill)
    shape.line.color.rgb = RGBColor(0xE4, 0xE0, 0xD8)
    shape.line.width = Pt(1)
    # Slightly tighter corners than the default.
    try:
        shape.adjustments[0] = 0.08
    except Exception:
        pass
    return shape


def _textbox(slide, left, top, width, height):
    """Empty text box; caller adds paragraphs."""
    return slide.shapes.add_textbox(left, top, width, height)


def _add_canvas(slide) -> None:
    """Cream page + navy header band + teal rule (shared chrome)."""
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
    )
    _fill(bg, CREAM)
    header = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.72)
    )
    _fill(header, NAVY)
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.72), Inches(13.333), Inches(0.08)
    )
    _fill(accent, TEAL)


def _add_header_text(slide, kicker: str, title: str, subtitle: str) -> None:
    """White title block inside the navy header."""
    box = _textbox(slide, Inches(0.5), Inches(0.18), Inches(12.3), Inches(0.32))
    box.text_frame.clear()
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(r, kicker, 12, TEAL, True)

    box = _textbox(slide, Inches(0.5), Inches(0.48), Inches(12.3), Inches(0.7))
    box.text_frame.word_wrap = True
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(r, title, 32, WHITE, True)

    box = _textbox(slide, Inches(0.5), Inches(1.16), Inches(12.3), Inches(0.4))
    box.text_frame.word_wrap = True
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(r, subtitle, 15, RGBColor(0xC5, 0xD4, 0xE8), False)


def _add_footer(slide, text: str) -> None:
    """Navy footer strip."""
    foot = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.12), Inches(13.333), Inches(0.38)
    )
    _fill(foot, NAVY)
    box = _textbox(slide, Inches(0.5), Inches(7.16), Inches(12.3), Inches(0.3))
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(r, text, 11, WHITE, False)


def add_slide_1(prs: Presentation) -> None:
    """Problem statement + 2D-only scope of the 2D+3D assignment."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_canvas(slide)
    _add_header_text(
        slide,
        "PROGRESS UPDATE   ·   SINGLE INTEL REALSENSE D455f",
        "2D and 3D Motion Analysis",
        "Problem statement   ·   Assigned: 2D + 3D    |    Delivered this phase: 2D only",
    )

    # Three equal cards.
    gap = Inches(0.22)
    left0 = Inches(0.45)
    top = Inches(2.08)
    width = Inches(4.05)
    height = Inches(4.55)
    cards = [
        (
            TEAL,
            "01   THE ASSIGNMENT",
            "Measure human motion with one RGB-D camera.",
            [
                "Hardware: Intel RealSense D455f",
                "Live USB  and  uploaded video",
                "Camera coordinates (image pixels)",
                "No hardcoding — config.yaml",
            ],
        ),
        (
            AMBER,
            "02   THIS PHASE",
            "Full problem is 2D + 3D. We completed 2D.",
            [
                "2D = colour frame → joint pixels (u, v)",
                "Origin: top-left of the RGB image",
                "3D (depth → metres) is not built yet",
                "Student prototype — not a medical device",
            ],
        ),
        (
            CORAL,
            "03   WHY 2D FIRST",
            "3D is the same pixel plus RealSense depth.",
            [
                "If the 2D dot is wrong, 3D will be wrong",
                "So: RGB → keypoints → angles → one app",
                "Then add aligned depth for metres / m/s",
                "Do not treat MediaPipe “world” as RealSense 3D",
            ],
        ),
    ]

    for i, (bar_color, heading, lead, bullets) in enumerate(cards):
        x = left0 + i * (width + gap)
        _round_card(slide, x, top, width, height, CARD)

        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, x, top, width, Inches(0.1)
        )
        _fill(bar, bar_color)

        hbox = _textbox(slide, x + Inches(0.22), top + Inches(0.28), width - Inches(0.4), Inches(0.42))
        hp = hbox.text_frame.paragraphs[0]
        r = hp.add_run()
        _set_run(r, heading, 13, bar_color, True)

        lbox = _textbox(slide, x + Inches(0.22), top + Inches(0.72), width - Inches(0.4), Inches(0.85))
        lbox.text_frame.word_wrap = True
        lp = lbox.text_frame.paragraphs[0]
        r = lp.add_run()
        _set_run(r, lead, 15, INK, True)

        bbox = _textbox(slide, x + Inches(0.22), top + Inches(1.62), width - Inches(0.4), Inches(2.7))
        tf = bbox.text_frame
        tf.word_wrap = True
        for j, line in enumerate(bullets):
            para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            para.level = 0
            para.space_after = Pt(10)
            r = para.add_run()
            _set_run(r, "▸  " + line, 14, MUTED, False)

    _add_footer(
        slide,
        "Jay   ·   IIT Guwahati   ·   2D phase complete   ·   Next: depth → camera metres",
    )


def add_slide_2(prs: Presentation) -> None:
    """Solution built this phase + horizontal 2D workflow."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_canvas(slide)
    _add_header_text(
        slide,
        "SOLUTION   ·   2D APPLICATION",
        "What we built",
        "One window: live D455f or upload  ·  skeleton, elbow degrees, wrist path  ·  optional save",
    )

    lead = _round_card(slide, Inches(0.45), Inches(2.00), Inches(12.43), Inches(0.78), CARD)
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(2.00), Inches(0.12), Inches(0.78)
    )
    _fill(bar, TEAL)
    box = _textbox(slide, Inches(0.78), Inches(2.10), Inches(11.9), Inches(0.58))
    box.text_frame.word_wrap = True
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(
        r,
        "A user-facing 2D app. Thresholds stay in config.yaml. Camera pixels only — not metres, not a medical device.",
        16,
        INK,
        False,
    )

    steps = [
        (TEAL, "01  RGB SOURCE", "Live D455f\nor .mp4 / .avi / .bag", "Colour frames only"),
        (NAVY_MID, "02  POSE  2D", "MediaPipe Pose\n33 joints (u, v)", "Origin: top-left"),
        (AMBER, "03  ANALYSIS", "Elbow angle (image °)\nWrist trail + px/s", "Skip low confidence"),
        (CORAL, "04  OUTPUT", "GUI live numbers\nCSV + overlay.mp4", "Start / Stop"),
    ]
    box_w = Inches(2.62)
    arrow_w = Inches(0.36)
    gap = Inches(0.10)
    x = Inches(0.45)
    y = Inches(2.98)
    h = Inches(2.55)
    for i, (color, title, body, note) in enumerate(steps):
        _round_card(slide, x, y, box_w, h, CARD)
        topbar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, box_w, Inches(0.10))
        _fill(topbar, color)

        tbox = _textbox(slide, x + Inches(0.14), y + Inches(0.22), box_w - Inches(0.28), Inches(0.42))
        r = tbox.text_frame.paragraphs[0].add_run()
        _set_run(r, title, 13, color, True)

        bbox = _textbox(slide, x + Inches(0.14), y + Inches(0.70), box_w - Inches(0.28), Inches(1.15))
        bbox.text_frame.word_wrap = True
        for j, line in enumerate(body.split("\n")):
            para = bbox.text_frame.paragraphs[0] if j == 0 else bbox.text_frame.add_paragraph()
            para.space_after = Pt(2)
            rr = para.add_run()
            _set_run(rr, line, 15, INK, True)

        nbox = _textbox(slide, x + Inches(0.14), y + Inches(1.95), box_w - Inches(0.28), Inches(0.42))
        nbox.text_frame.word_wrap = True
        r = nbox.text_frame.paragraphs[0].add_run()
        _set_run(r, note, 12, MUTED, False)

        x = x + box_w
        if i < len(steps) - 1:
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW,
                x + gap,
                y + Inches(1.05),
                arrow_w,
                Inches(0.32),
            )
            _fill(arrow, TEAL)
            x = x + arrow_w + gap + gap

    chips = [
        (TEAL, "Live  or  Upload"),
        (AMBER, "Elbow ° in the image   ·   ~180° = straight"),
        (CORAL, "Wrist speed in px/s   ·   not m/s"),
    ]
    chip_w = Inches(4.05)
    cx = Inches(0.45)
    for color, text in chips:
        _round_card(slide, cx, Inches(5.70), chip_w, Inches(1.20), CARD)
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, cx, Inches(5.70), Inches(0.10), Inches(1.20)
        )
        _fill(bar, color)
        box = _textbox(slide, cx + Inches(0.28), Inches(6.00), chip_w - Inches(0.42), Inches(0.62))
        box.text_frame.word_wrap = True
        r = box.text_frame.paragraphs[0].add_run()
        _set_run(r, text, 14, INK, True)
        cx += chip_w + Inches(0.22)

    _add_footer(slide, "Workflow is 2D camera pixels only. Depth is not used yet.")


def add_slide_3(prs: Presentation) -> None:
    """Tech stack used for the 2D phase."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_canvas(slide)
    _add_header_text(
        slide,
        "IMPLEMENTATION",
        "Tech stack",
        "Python  ·  nothing hardcoded in code  ·  run from the project folder:  python app.py",
    )

    items = [
        (TEAL, "CAMERA", "Intel RealSense D455f", "pyrealsense2  ·  live RGB\n.bag playback for later 3D"),
        (NAVY_MID, "VIDEO", "OpenCV", "Uploaded .mp4 / .avi\nSkeleton overlay on RGB"),
        (AMBER, "POSE", "MediaPipe Pose", "BlazePose  ·  33 landmarks\n2D pixels (u, v) only"),
        (CORAL, "CONFIG", "YAML  ·  config.yaml", "Resolution, FPS, thresholds\nWhich joints and trail"),
        (TEAL, "INTERFACE", "Tkinter + Pillow", "Live / Upload  ·  Start / Stop\nLive elbow ° and wrist px/s"),
        (AMBER, "DATA OUT", "CSV + MP4", "keypoints_2d.csv\nangles_2d.csv  ·  overlay.mp4"),
    ]
    gap = Inches(0.22)
    left0 = Inches(0.45)
    width = Inches(4.05)
    height = Inches(2.05)
    for i, (color, tag, title, body) in enumerate(items):
        col = i % 3
        row = i // 3
        x = left0 + col * (width + gap)
        y = Inches(2.02) + row * (height + Inches(0.18))
        _round_card(slide, x, y, width, height, CARD)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, width, Inches(0.09))
        _fill(bar, color)

        tbox = _textbox(slide, x + Inches(0.20), y + Inches(0.20), width - Inches(0.40), Inches(0.28))
        r = tbox.text_frame.paragraphs[0].add_run()
        _set_run(r, tag, 11, color, True)

        nbox = _textbox(slide, x + Inches(0.20), y + Inches(0.48), width - Inches(0.40), Inches(0.38))
        r = nbox.text_frame.paragraphs[0].add_run()
        _set_run(r, title, 16, INK, True)

        bbox = _textbox(slide, x + Inches(0.20), y + Inches(0.92), width - Inches(0.40), Inches(0.95))
        bbox.text_frame.word_wrap = True
        for j, line in enumerate(body.split("\n")):
            para = bbox.text_frame.paragraphs[0] if j == 0 else bbox.text_frame.add_paragraph()
            para.space_after = Pt(2)
            rr = para.add_run()
            _set_run(rr, line, 13, MUTED, False)

    note = _round_card(slide, Inches(0.45), Inches(6.48), Inches(12.43), Inches(0.52), NAVY)
    box = _textbox(slide, Inches(0.65), Inches(6.56), Inches(12.05), Inches(0.38))
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(
        r,
        "Not used as 3D: MediaPipe world landmarks (RGB-only guess).  Real 3D = RealSense depth at these 2D pixels.  Close Viewer before Live.",
        13,
        WHITE,
        False,
    )

    _add_footer(slide, "Jay   ·   IIT Guwahati   ·   python app.py   ·   USB 3  ·  close RealSense Viewer")


def main() -> None:
    """Write the 3-slide 16:9 progress deck."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    add_slide_1(prs)
    add_slide_2(prs)
    add_slide_3(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    target = OUT
    try:
        prs.save(str(target))
    except PermissionError:
        target = OUT_ALT
        prs.save(str(target))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
