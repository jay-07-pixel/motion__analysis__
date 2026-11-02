"""Plain-language capture flow. One step per slide.

Run from the project folder:
    python docs/make_capture_flow_ppt.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(0x0B, 0x1F, 0x3A)
TEAL = RGBColor(0x1F, 0xB5, 0xA6)
CREAM = RGBColor(0xF6, 0xF3, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x23, 0x32)
MUTED = RGBColor(0x5A, 0x66, 0x78)

OUT = Path(__file__).resolve().parent / "Capture_Flow.pptx"
W = Inches(13.333)
H = Inches(7.5)

STEPS = [
    (
        "1",
        "Type height and weight, then press Connect",
        [
            "Height is in centimetres. Weight is in kilograms.",
            "Connect reads those two numbers and opens the camera.",
            "Colour and depth both turn on.",
            "The extra arm markers turn on.",
            "Nothing is saved yet.",
        ],
    ),
    (
        "2",
        "You see yourself on the live screen",
        [
            "The picture is on the left. The six dials are on the right.",
            "Each new picture finds the body and both hands.",
            "Each hand stays with its own wrist.",
            "Depth at a dot becomes a 3D point: right, down, and away from the camera.",
            "A bad depth is left empty. The last depth is not reused.",
        ],
    ),
    (
        "3",
        "The diamonds and the dials update on that same picture",
        [
            "The recent frames add the elbow diamonds and the wrist diamonds.",
            "An old diamond is not drawn as if it belongs to this picture.",
            "Each dial shows the angle made by three points you can see.",
            "If depth drops for a moment, the needle can hold the last number.",
            "That held number is only on the dial. It is not written into the file.",
        ],
    ),
    (
        "4",
        "Press Start Capture when the dots are on you",
        [
            "Both shoulders must be visible, and depth must be streaming.",
            "The camera stays the one already open. A second camera is not started.",
            "The same live steps keep running.",
            "Now the points, the diamonds, and the dial numbers are written to the recording.",
            "The picture on screen is the picture being saved.",
        ],
    ),
    (
        "5",
        "Press Stop Capture",
        [
            "The recording files are closed.",
            "The camera stays on, so you can still see yourself.",
            "The saved points stay where they were measured.",
            "This folder is the one the next step reads.",
        ],
    ),
    (
        "6",
        "Press Run 3D Analysis",
        [
            "The saved points are read from the recording.",
            "Twelve upper-body points are copied into the marker file. They are not moved.",
            "Those points are the shoulders, the elbows, and the eight diamonds.",
            "A sized copy of the skeleton is made for this recording.",
            "The original skeleton file is left unchanged.",
        ],
    ),
    (
        "7",
        "The skeleton is fitted to those saved points",
        [
            "An elbow pair that is too far from the skeleton spacing is left out of the fit.",
            "A point that jumped a long way in one moment is left out of the fit.",
            "The recording itself is not edited.",
            "The joints move until the skeleton points are as close as they can get.",
            "The live screen is not changed by this step.",
        ],
    ),
    (
        "8",
        "Press View Results",
        [
            "The joint angles from the fitted skeleton are shown.",
            "A picture shows the skeleton on the recorded points.",
            "Graphs show the angles across the recording.",
            "The same files are in the results folder.",
        ],
    ),
]


def _fill(shape, color: RGBColor) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _run(paragraph, text: str, size: int, color: RGBColor, bold: bool = False) -> None:
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


def _footer(slide, page: int, total: int) -> None:
    box = slide.shapes.add_textbox(Inches(0.5), Inches(7.12), Inches(10), Inches(0.28))
    p = box.text_frame.paragraphs[0]
    _run(p, "Capture flow", 14, MUTED)
    num = slide.shapes.add_textbox(Inches(11.2), Inches(7.12), Inches(1.6), Inches(0.28))
    np = num.text_frame.paragraphs[0]
    np.alignment = PP_ALIGN.RIGHT
    _run(np, f"{page}  /  {total}", 14, MUTED)


def _title_slide(prs: Presentation) -> None:
    slide = _blank(prs)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), H)
    _fill(band, TEAL)
    box = slide.shapes.add_textbox(Inches(0.9), Inches(2.15), Inches(11.5), Inches(2.4))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    _run(p, "What happens, start to end", 20, TEAL, True)
    p2 = tf.add_paragraph()
    _run(p2, "The capture flow", 54, NAVY, True)
    p3 = tf.add_paragraph()
    _run(p3, "Connect, watch, record, fit the skeleton, then open the result.", 24, INK)


def _map_slide(prs: Presentation) -> None:
    slide = _blank(prs)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.15))
    _fill(band, NAVY)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.15), W, Inches(0.06))
    _fill(rule, TEAL)
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.28), Inches(12), Inches(0.7))
    _run(title.text_frame.paragraphs[0], "The eight steps", 32, WHITE, True)

    labels = [item[1] for item in STEPS]
    card_w = Inches(2.9)
    card_h = Inches(2.15)
    gap_x = Inches(0.22)
    gap_y = Inches(0.22)
    left0 = Inches(0.45)
    top0 = Inches(1.6)
    for index, label in enumerate(labels):
        col = index % 4
        row = index // 4
        left = left0 + col * (card_w + gap_x)
        top = top0 + row * (card_h + gap_y)
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, card_w, card_h)
        _fill(card, WHITE)
        num = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.16), Inches(2.5), Inches(0.5))
        _run(num.text_frame.paragraphs[0], str(index + 1), 26, TEAL, True)
        body = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.72), Inches(2.55), Inches(1.25))
        frame = body.text_frame
        frame.word_wrap = True
        _run(frame.paragraphs[0], label, 16, INK, True)
    _footer(slide, 2, 10)


def _step_slide(prs: Presentation, number: str, title: str, lines: list[str], page: int) -> None:
    slide = _blank(prs)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.35))
    _fill(band, NAVY)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.35), W, Inches(0.06))
    _fill(rule, TEAL)
    badge = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.48), Inches(0.28), Inches(0.78), Inches(0.78))
    _fill(badge, TEAL)
    badge_text = slide.shapes.add_textbox(Inches(0.48), Inches(0.42), Inches(0.78), Inches(0.52))
    bp = badge_text.text_frame.paragraphs[0]
    bp.alignment = PP_ALIGN.CENTER
    _run(bp, number, 26, NAVY, True)
    heading = slide.shapes.add_textbox(Inches(1.5), Inches(0.38), Inches(11.2), Inches(0.7))
    heading.text_frame.word_wrap = True
    _run(heading.text_frame.paragraphs[0], title, 28, WHITE, True)

    top = Inches(1.62)
    for line in lines:
        row = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), top, Inches(12.3), Inches(0.9))
        _fill(row, WHITE)
        box = slide.shapes.add_textbox(Inches(0.78), top + Inches(0.18), Inches(11.75), Inches(0.58))
        frame = box.text_frame
        frame.word_wrap = True
        _run(frame.paragraphs[0], line, 22, INK)
        top += Inches(1.02)
    _footer(slide, page, 10)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    _title_slide(prs)
    _map_slide(prs)
    for page, (number, title, lines) in enumerate(STEPS, start=3):
        _step_slide(prs, number, title, lines, page)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
