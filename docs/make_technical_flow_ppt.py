"""Three-slide technical flow. Replaces the long box deck.

Run from the project folder:
    python docs/make_technical_flow_ppt.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(0x0B, 0x1F, 0x3A)
TEAL = RGBColor(0x0E, 0x8F, 0x86)
TEAL_SOFT = RGBColor(0xD7, 0xF3, 0xEF)
CREAM = RGBColor(0xF4, 0xF1, 0xEA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x23, 0x32)
MUTED = RGBColor(0x5C, 0x67, 0x78)
AMBER = RGBColor(0xC4, 0x7D, 0x2A)

OUT = Path(__file__).resolve().parent / "Flow_3pages.pptx"
W = Inches(13.333)
H = Inches(7.5)


def _fill(shape, color: RGBColor) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _text(shape, lines: list[tuple[str, int, RGBColor, bool]], align=PP_ALIGN.LEFT) -> None:
    frame = shape.text_frame
    frame.word_wrap = True
    frame.auto_size = None
    frame.margin_left = Inches(0.08)
    frame.margin_right = Inches(0.08)
    for index, (text, size, color, bold) in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_after = Pt(2)
        run = paragraph.add_run()
        run.text = text
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold


def _blank(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    _fill(bg, CREAM)
    return slide


def _slide_path(prs: Presentation) -> None:
    slide = _blank(prs)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.35))
    _fill(bar, NAVY)
    title = slide.shapes.add_textbox(Inches(0.55), Inches(0.22), Inches(12), Inches(0.95))
    _text(
        title,
        [
            ("ONE PATH", 13, TEAL, True),
            ("Camera, then dots, then the skeleton", 32, WHITE, True),
        ],
    )

    steps = [
        ("1", "Camera", "Colour and\naligned depth"),
        ("2", "Dots", "Body and\nboth hands"),
        ("3", "Metres", "Pixel + depth\nbecomes X Y Z"),
        ("4", "Diamonds", "Eight arm\nestimates"),
        ("5", "Save", "Start, then\nStop"),
        ("6", "Skeleton", "Sized copy,\nthen the fit"),
    ]
    left = Inches(0.38)
    top = Inches(2.05)
    width = Inches(2.05)
    gap = Inches(0.12)
    for index, (number, name, detail) in enumerate(steps):
        x = left + index * (width + gap)
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, top, width, Inches(2.55))
        _fill(card, WHITE if index < 5 else NAVY)
        accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, top, width, Inches(0.08))
        _fill(accent, TEAL if index < 5 else AMBER)
        blob = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.16), top + Inches(0.28), Inches(0.46), Inches(0.46))
        _fill(blob, TEAL if index < 5 else AMBER)
        num = slide.shapes.add_textbox(x + Inches(0.16), top + Inches(0.34), Inches(0.46), Inches(0.38))
        _text(num, [(number, 16, WHITE, True)], PP_ALIGN.CENTER)
        name_box = slide.shapes.add_textbox(x + Inches(0.12), top + Inches(0.9), width - Inches(0.2), Inches(0.45))
        _text(name_box, [(name, 20, INK if index < 5 else WHITE, True)])
        detail_box = slide.shapes.add_textbox(x + Inches(0.12), top + Inches(1.4), width - Inches(0.2), Inches(0.9))
        _text(detail_box, [(detail, 15, MUTED if index < 5 else RGBColor(0xD5, 0xDE, 0xE8), False)])
        if index < 5:
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW,
                x + width - Inches(0.02),
                top + Inches(1.1),
                Inches(0.16),
                Inches(0.18),
            )
            _fill(arrow, TEAL)

    note = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.38), Inches(5.05), Inches(12.55), Inches(1.55))
    _fill(note, TEAL_SOFT)
    note_text = slide.shapes.add_textbox(Inches(0.65), Inches(5.25), Inches(12.05), Inches(1.2))
    _text(
        note_text,
        [
            ("The model does not run while you watch.", 22, NAVY, True),
            ("It starts at step 6, after Stop. The separate ground calibration is not on this path.", 18, INK, False),
        ],
    )


def _slide_live(prs: Presentation) -> None:
    slide = _blank(prs)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.2))
    _fill(bar, NAVY)
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.18), Inches(12), Inches(0.9))
    _text(title, [("WHILE THE CAMERA IS ON", 13, TEAL, True), ("What one frame does", 30, WHITE, True)])

    panels = [
        (
            "Camera",
            [
                "One RealSense. Colour and depth together.",
                "Depth is aligned to the colour picture.",
                "Origin at the colour centre.",
                "+X right, +Y down, +Z forward.",
            ],
        ),
        (
            "Dots",
            [
                "Pose finds the body.",
                "Hands find both hands together.",
                "Each hand stays on its own wrist.",
                "A bad depth stays empty.",
            ],
        ),
        (
            "Diamonds and dials",
            [
                "Height and weight feed the network only.",
                "Eight diamonds: elbow and wrist, both arms.",
                "A dial is three dots. Straight is near 180°.",
                "That dial is not a skeleton angle.",
            ],
        ),
    ]
    for index, (heading, lines) in enumerate(panels):
        x = Inches(0.4) + index * Inches(4.25)
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.6), Inches(4.05), Inches(4.55))
        _fill(card, WHITE)
        cap = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.6), Inches(4.05), Inches(0.7))
        _fill(cap, TEAL if index < 2 else NAVY)
        cap_text = slide.shapes.add_textbox(x + Inches(0.2), Inches(1.72), Inches(3.65), Inches(0.48))
        _text(cap_text, [(heading, 20, WHITE, True)])
        body = slide.shapes.add_textbox(x + Inches(0.25), Inches(2.55), Inches(3.55), Inches(3.3))
        frame = body.text_frame
        frame.word_wrap = True
        for line_index, line in enumerate(lines):
            paragraph = frame.paragraphs[0] if line_index == 0 else frame.add_paragraph()
            paragraph.space_after = Pt(12)
            run = paragraph.add_run()
            run.text = line
            run.font.name = "Calibri"
            run.font.size = Pt(18)
            run.font.color.rgb = INK
    foot = slide.shapes.add_textbox(Inches(0.5), Inches(6.4), Inches(12.3), Inches(0.7))
    _text(
        foot,
        [("Start writes these points. Stop closes the file. The camera stays open.", 18, MUTED, False)],
    )


def _slide_model(prs: Presentation) -> None:
    slide = _blank(prs)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.2))
    _fill(bar, NAVY)
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.18), Inches(12), Inches(0.9))
    _text(title, [("AFTER STOP", 13, TEAL, True), ("Where the model works", 30, WHITE, True)])

    steps = [
        ("12 points", "Shoulders, elbows, and eight diamonds. Wrist dots are not sent."),
        ("One copy", "Chest stays 40.3 cm. Forearms can change for this recording."),
        ("Leave out", "An elbow pair far from 9.1 cm, or a 15 cm jump, stays out of that frame."),
        ("Joints move", "Bone markers stay put. The angles change to reach the points."),
    ]
    for index, (heading, detail) in enumerate(steps):
        x = Inches(0.35) + index * Inches(3.25)
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.65), Inches(3.08), Inches(3.15))
        _fill(card, WHITE)
        num = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.18), Inches(1.85), Inches(0.42), Inches(0.42))
        _fill(num, TEAL)
        num_text = slide.shapes.add_textbox(x + Inches(0.18), Inches(1.9), Inches(0.42), Inches(0.34))
        _text(num_text, [(str(index + 1), 14, WHITE, True)], PP_ALIGN.CENTER)
        head = slide.shapes.add_textbox(x + Inches(0.7), Inches(1.88), Inches(2.2), Inches(0.4))
        _text(head, [(heading, 18, NAVY, True)])
        body = slide.shapes.add_textbox(x + Inches(0.18), Inches(2.5), Inches(2.72), Inches(2.05))
        _text(body, [(detail, 16, INK, False)])

    note = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.35), Inches(5.1), Inches(12.65), Inches(1.7))
    _fill(note, NAVY)
    note_text = slide.shapes.add_textbox(Inches(0.6), Inches(5.3), Inches(12.2), Inches(1.35))
    _text(
        note_text,
        [
            ("Calibration is not this step.", 22, TEAL, True),
            ("A separate window can build another ground. This model already uses the colour camera, so those metres are copied as they are.", 18, WHITE, False),
        ],
    )


def build() -> Path:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    _slide_path(prs)
    _slide_live(prs)
    _slide_model(prs)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
