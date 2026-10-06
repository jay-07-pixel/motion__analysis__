"""New pipeline deck. Does not touch Progress_Update_2D.pptx.

Run from the project folder:
    python docs/make_pipeline_ppt.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(0x0B, 0x1F, 0x3A)
TEAL = RGBColor(0x1F, 0xB5, 0xA6)
TEAL_DEEP = RGBColor(0x0E, 0x7C, 0x72)
AMBER = RGBColor(0xE8, 0xA3, 0x38)
CORAL = RGBColor(0xE0, 0x6C, 0x5C)
CREAM = RGBColor(0xF6, 0xF3, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x23, 0x32)
MUTED = RGBColor(0x5A, 0x66, 0x78)
LINE = RGBColor(0xE4, 0xE0, 0xD8)
CYAN = RGBColor(0x3D, 0xD6, 0xC8)

OUT = Path(__file__).resolve().parent / "Motion_Analysis_Pipeline.pptx"
W = Inches(13.333)
H = Inches(7.5)


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
    layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    _fill(bg, CREAM)
    return slide


def _header(slide, kicker: str, title: str) -> None:
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.22))
    _fill(band, NAVY)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.22), W, Inches(0.06))
    _fill(rule, TEAL)
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.16), Inches(12.4), Inches(0.95))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    _run(p, kicker.upper(), 12, TEAL, True)
    p2 = tf.add_paragraph()
    _run(p2, title, 28, WHITE, True)


def _footer(slide, page: int, total: int) -> None:
    box = slide.shapes.add_textbox(Inches(0.45), Inches(7.15), Inches(12.4), Inches(0.28))
    tf = box.text_frame
    p = tf.paragraphs[0]
    _run(p, "Jay  ·  Intel RealSense D455f  ·  camera coordinates only", 11, MUTED)
    num = slide.shapes.add_textbox(Inches(11.4), Inches(7.12), Inches(1.5), Inches(0.28))
    np = num.text_frame.paragraphs[0]
    np.alignment = PP_ALIGN.RIGHT
    _run(np, f"{page}  /  {total}", 11, MUTED)


def _card(slide, left, top, width, height, fill: RGBColor = WHITE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    _fill(shape, fill)
    shape.line.color.rgb = LINE
    shape.line.width = Pt(1)
    try:
        shape.adjustments[0] = 0.08
    except Exception:
        pass
    return shape


def _text(slide, left, top, width, height, lines: list[tuple[str, int, RGBColor, bool]], align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    for i, (text, size, color, bold) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(2)
        _run(p, text, size, color, bold)
    return box


def _bullets(slide, left, top, width, height, items: list[str], size: int = 16) -> None:
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        p.space_after = Pt(8)
        _run(p, item, size, INK, False)


def _flow_box(slide, left, top, width, height, step: str, title: str, detail: str, fill: RGBColor, ink: RGBColor):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    _fill(shape, fill)
    shape.line.fill.background()
    try:
        shape.adjustments[0] = 0.12
    except Exception:
        pass
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(p, step, 10, ink, True)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    _run(p2, title, 13, ink, True)
    p3 = tf.add_paragraph()
    p3.alignment = PP_ALIGN.CENTER
    _run(p3, detail, 11, ink, False)
    return shape


def _arrow(slide, left, top, width, height, color: RGBColor = TEAL) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    _fill(shape, color)


def _down_arrow(slide, left, top, width, height) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, left, top, width, height)
    _fill(shape, TEAL)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    prs.core_properties.title = "2D and 3D Motion Analysis — pipeline"
    prs.core_properties.author = "Jay"
    total = 11

    _slide_title(prs)
    _slide_task(prs, 2, total)
    _slide_coordinates(prs, 3, total)
    _slide_flowchart(prs, 4, total)
    _slide_capture(prs, 5, total)
    _slide_pose(prs, 6, total)
    _slide_regions(prs, 7, total)
    _slide_angles(prs, 8, total)
    _slide_depth(prs, 9, total)
    _slide_dials(prs, 10, total)
    _slide_limits(prs, 11, total)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


def _slide_title(prs: Presentation) -> None:
    slide = _blank(prs)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    _fill(band, NAVY)
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.16), H)
    _fill(accent, TEAL)
    _text(
        slide,
        Inches(0.7),
        Inches(1.7),
        Inches(11.5),
        Inches(3.4),
        [
            ("MOTION ANALYSIS", 14, TEAL, True),
            ("2D and 3D from one RealSense D455f", 36, WHITE, True),
            ("", 10, WHITE, False),
            ("Pipeline, camera coordinates, and how each angle is calculated", 20, RGBColor(0xC5, 0xD4, 0xE8), False),
            ("", 12, WHITE, False),
            ("Jay", 18, WHITE, True),
            ("Live camera and uploaded video  ·  camera coordinates only", 16, TEAL, False),
        ],
    )


def _slide_task(slide_prs, page, total) -> None:
    slide = _blank(slide_prs)
    _header(slide, "01   Scope", "What this build does")
    _footer(slide, page, total)
    cards = [
        ("2D first", "Colour image only. Every joint is a pixel (u, v) on the RGB frame. Angles are the angle you see in the photo."),
        ("Then 3D", "Same pixel, plus RealSense depth aligned to colour. Deproject to camera metres: X, Y, Z. Not a guessed skeleton."),
        ("Both in one window", "Live USB or an uploaded file. Choose a body region. Dials update while the camera runs. After Stop: min, max, and mode."),
    ]
    for i, (title, body) in enumerate(cards):
        left = Inches(0.4 + i * 4.25)
        _card(slide, left, Inches(1.7), Inches(4.05), Inches(3.35))
        _text(
            slide,
            left + Inches(0.25),
            Inches(1.95),
            Inches(3.55),
            Inches(2.9),
            [
                (f"0{i + 1}", 14, TEAL, True),
                (title, 22, NAVY, True),
                ("", 8, INK, False),
                (body, 15, INK, False),
            ],
        )
    _card(slide, Inches(0.4), Inches(5.25), Inches(12.5), Inches(1.65), RGBColor(0xE7, 0xF6, 0xF4))
    _text(
        slide,
        Inches(0.65),
        Inches(5.45),
        Inches(12.0),
        Inches(1.3),
        [
            ("Rule we follow", 14, TEAL_DEEP, True),
            ("MediaPipe “world landmarks” are a model guess. We do not use them as 3D. Real 3D is depth at the colour pixel, then the camera’s own intrinsics.", 16, INK, False),
        ],
    )


def _slide_coordinates(prs, page, total) -> None:
    slide = _blank(prs)
    _header(slide, "02   Coordinates", "Two camera frames — do not mix them")
    _footer(slide, page, total)
    _card(slide, Inches(0.4), Inches(1.6), Inches(6.15), Inches(5.25))
    _text(
        slide,
        Inches(0.65),
        Inches(1.8),
        Inches(5.7),
        Inches(4.9),
        [
            ("2D   ·   camera pixels", 20, NAVY, True),
            ("Used when the 2D button is on", 14, TEAL_DEEP, True),
            ("", 6, INK, False),
            ("Origin    top-left corner of the colour image", 16, INK, False),
            ("u    column, left → right, in pixels", 16, INK, False),
            ("v    row, top → bottom, in pixels", 16, INK, False),
            ("", 6, INK, False),
            ("An elbow angle here is the angle in the photograph. If the arm points toward the lens, this number is wrong.", 16, INK, False),
            ("", 6, INK, False),
            ("An mp4 has only this frame. It has no depth.", 16, MUTED, False),
        ],
    )
    _card(slide, Inches(6.8), Inches(1.6), Inches(6.1), Inches(5.25), NAVY)
    _text(
        slide,
        Inches(7.05),
        Inches(1.8),
        Inches(5.65),
        Inches(4.9),
        [
            ("3D   ·   camera metres", 20, WHITE, True),
            ("Used when the 3D button is on", 14, TEAL, True),
            ("", 6, WHITE, False),
            ("Origin    optical centre of the colour lens", 16, WHITE, False),
            ("X    right of the lens, centimetres", 16, WHITE, False),
            ("Y    down from the lens, centimetres", 16, WHITE, False),
            ("Z    straight out from the lens (distance)", 16, WHITE, False),
            ("", 6, WHITE, False),
            ("Shown above the video, not drawn on the picture. Needs live USB or a RealSense .bag.", 16, RGBColor(0xC5, 0xD4, 0xE8), False),
        ],
    )


def _slide_flowchart(prs, page, total) -> None:
    slide = _blank(prs)
    _header(slide, "03   Flowchart", "One frame, from camera to dial")
    _footer(slide, page, total)

    y1 = Inches(1.55)
    bw, bh = Inches(2.55), Inches(1.15)
    gap = Inches(0.42)
    x0 = Inches(0.4)
    boxes = [
        ("1", "Capture", "Live D455f or file"),
        ("2", "Pose", "33 joints, u and v"),
        ("3", "Hands", "21 points per hand"),
        ("4", "Region", "What to draw and dial"),
    ]
    for i, (step, title, detail) in enumerate(boxes):
        left = x0 + i * (bw + gap)
        _flow_box(slide, left, y1, bw, bh, step, title, detail, TEAL, NAVY)
        if i < 3:
            _arrow(slide, left + bw + Inches(0.04), y1 + Inches(0.42), Inches(0.34), Inches(0.28))

    _text(
        slide,
        Inches(0.4),
        Inches(2.78),
        Inches(12.5),
        Inches(0.38),
        [("After the region is chosen, the 2D or 3D button picks the path.", 13, MUTED, False)],
        PP_ALIGN.CENTER,
    )

    _flow_box(
        slide, Inches(0.4), Inches(3.25), Inches(6.05), Inches(1.2),
        "5A   2D BUTTON", "Image-plane angle", "Pixels only. Cards show u and v.",
        AMBER, NAVY,
    )
    _flow_box(
        slide, Inches(6.85), Inches(3.25), Inches(6.05), Inches(1.2),
        "5B   3D BUTTON", "Depth, then camera X Y Z", "Align, median, deproject, smooth.",
        CORAL, WHITE,
    )

    _down_arrow(slide, Inches(6.5), Inches(4.52), Inches(0.32), Inches(0.32))

    y3 = Inches(4.95)
    tail = [
        ("6", "Live dials", "Shoulder, elbow, wrist"),
        ("7", "After Stop", "Min, max, mode"),
        ("8", "Save", "CSV and overlay video"),
    ]
    tw = Inches(3.7)
    for i, (step, title, detail) in enumerate(tail):
        left = Inches(0.4) + i * (tw + Inches(0.35))
        _flow_box(slide, left, y3, tw, Inches(1.05), step, title, detail, NAVY, WHITE)
        if i < 2:
            _arrow(slide, left + tw + Inches(0.02), y3 + Inches(0.38), Inches(0.3), Inches(0.26), AMBER)


def _slide_capture(prs, page, total) -> None:
    slide = _blank(prs)
    _header(slide, "04   Step 1", "Capture a colour frame")
    _footer(slide, page, total)
    rows = [
        ("Live camera", "Intel RealSense D455f on USB 3. Colour stream 1280 × 720 at 30 fps, already in OpenCV BGR. RealSense Viewer must be closed — only one program can hold the camera."),
        ("Uploaded movie", "mp4, avi, mov, mkv. Colour only. Good for 2D keypoints and image angles. There is no depth, so the 3D button refuses these files."),
        ("Uploaded .bag", "A RealSense recording. It stores colour and depth with the camera’s own timestamps. 2D uses the colour stream. 3D can align the stored depth later."),
    ]
    for i, (title, body) in enumerate(rows):
        top = Inches(1.6 + i * 1.75)
        _card(slide, Inches(0.4), top, Inches(12.5), Inches(1.6))
        _text(
            slide,
            Inches(0.7),
            top + Inches(0.2),
            Inches(12.0),
            Inches(1.25),
            [(title, 20, NAVY, True), (body, 15, INK, False)],
        )


def _slide_pose(prs, page, total) -> None:
    slide = _blank(prs)
    _header(slide, "05   Steps 2–3", "Find the body, then the fingers")
    _footer(slide, page, total)
    _card(slide, Inches(0.4), Inches(1.6), Inches(6.15), Inches(5.25))
    _text(
        slide,
        Inches(0.65),
        Inches(1.8),
        Inches(5.7),
        Inches(4.85),
        [
            ("MediaPipe Pose", 20, NAVY, True),
            ("", 4, INK, False),
            ("33 landmarks on the colour frame.", 16, INK, False),
            ("u = x × image width", 16, INK, True),
            ("v = y × image height", 16, INK, True),
            ("Origin of that pair is the top-left pixel.", 16, INK, False),
            ("", 6, INK, False),
            ("A joint is skipped if its visibility is below 0.5.", 16, INK, False),
            ("", 6, INK, False),
            ("Pose only marks the wrist, thumb tip, index tip, and pinky tip. That is not a finger skeleton.", 16, MUTED, False),
        ],
    )
    _card(slide, Inches(6.8), Inches(1.6), Inches(6.1), Inches(5.25))
    _text(
        slide,
        Inches(7.05),
        Inches(1.8),
        Inches(5.65),
        Inches(4.85),
        [
            ("MediaPipe Hands", 20, NAVY, True),
            ("", 4, INK, False),
            ("21 joints on each hand: knuckles and every fingertip.", 16, INK, False),
            ("", 6, INK, False),
            ("Hands labels left and right as if the picture were a selfie. The D455f is not mirrored, so that label would cross the arms.", 16, INK, False),
            ("", 6, INK, False),
            ("Each detected hand is attached to the nearest Pose wrist, so elbow and wrist stay on the same arm.", 16, INK, False),
            ("", 6, INK, False),
            ("Face mode does not run Hands.", 16, MUTED, False),
        ],
    )


def _slide_regions(prs, page, total) -> None:
    slide = _blank(prs)
    _header(slide, "06   Step 4", "Region picker does not re-run the model")
    _footer(slide, page, total)
    _text(
        slide,
        Inches(0.5),
        Inches(1.5),
        Inches(12.3),
        Inches(0.7),
        [("Pose still sees the whole person. The button only chooses which joints are drawn and which dials stay on.", 16, INK, False)],
    )
    regions = [
        ("Full body", "All joints. Six dials."),
        ("Upper body", "Drops the legs below the hip. Both hips stay, because the shoulder angle needs them."),
        ("Both arms", "Shoulders through the fingers. Same six dials."),
        ("Both hands", "Wrist dial only. Full finger skeleton."),
        ("Face", "Pose face points only. No angle dials."),
    ]
    for i, (name, detail) in enumerate(regions):
        col = i % 3
        row = i // 3
        left = Inches(0.4 + col * 4.25)
        top = Inches(2.35 + row * 2.2)
        width = Inches(4.05) if i < 3 else Inches(6.15)
        if i == 3:
            left = Inches(0.4)
        if i == 4:
            left = Inches(6.8)
            width = Inches(6.1)
        _card(slide, left, top, width, Inches(2.0))
        _text(
            slide,
            left + Inches(0.22),
            top + Inches(0.25),
            width - Inches(0.4),
            Inches(1.55),
            [(name, 20, NAVY, True), (detail, 15, INK, False)],
        )


def _slide_angles(prs, page, total) -> None:
    slide = _blank(prs)
    _header(slide, "07   Step 5", "How each angle is calculated")
    _footer(slide, page, total)
    _card(slide, Inches(0.4), Inches(1.55), Inches(12.5), Inches(1.55), NAVY)
    _text(
        slide,
        Inches(0.65),
        Inches(1.7),
        Inches(12.1),
        Inches(1.3),
        [
            ("Interior angle at the middle joint B, from vectors BA and BC", 16, TEAL, True),
            ("θ  =  arccos(  (BA · BC)  /  (|BA| |BC|)  )", 22, WHITE, True),
            ("About 180° is straight. Smaller means more bent. Skip the dial if any of the three joints is below confidence 0.5.", 14, RGBColor(0xC5, 0xD4, 0xE8), False),
        ],
    )
    triples = [
        ("Shoulder", "hip  →  shoulder  →  elbow", "The hip is the support point. That is why Upper body still keeps both hips."),
        ("Elbow", "shoulder  →  elbow  →  wrist", "Standard three-point elbow angle. Middle joint is the elbow."),
        ("Wrist", "elbow  →  wrist  →  index tip", "A 2D/3D proxy. The index fingertip stands in for the hand direction."),
    ]
    for i, (name, points, note) in enumerate(triples):
        left = Inches(0.4 + i * 4.25)
        _card(slide, left, Inches(3.3), Inches(4.05), Inches(2.15))
        _text(
            slide,
            left + Inches(0.2),
            Inches(3.45),
            Inches(3.65),
            Inches(1.9),
            [(name, 18, NAVY, True), (points, 13, TEAL_DEEP, True), (note, 13, INK, False)],
        )
    _text(
        slide,
        Inches(0.45),
        Inches(5.6),
        Inches(12.4),
        Inches(1.35),
        [
            ("Same three points in both modes. 2D uses pixel (u, v), so the angle is only in the photo. 3D uses camera (X, Y, Z) in metres, so it is the angle between the bones in space — and only if depth exists at all three joints.", 15, INK, False),
            ("The dials themselves are not smoothed. Each frame is a fresh angle.", 15, INK, True),
        ],
    )


def _slide_depth(prs, page, total) -> None:
    slide = _blank(prs)
    _header(slide, "08   Step 5B", "From a pixel to camera X, Y, Z")
    _footer(slide, page, total)
    steps = [
        ("1", "Depth stream", "Colour and depth run together only in 3D mode. Depth is aligned onto the colour image, so the wrist pixel and the depth pixel are the same place."),
        ("2", "Median 5×5", "Depth at the joint is the median of a 5×5 patch. One bad depth pixel does not throw the point."),
        ("3", "Deproject", "RealSense turns (u, v, Z) into metres with the colour camera intrinsics. That is X right, Y down, Z forward, from the lens centre."),
        ("4", "Smooth the point", "A move under 0.8 cm stays put. A real move is 35% new and 65% previous. If depth blinks off, the last good point is kept for 5 frames."),
    ]
    for i, (num, title, body) in enumerate(steps):
        col = i % 2
        row = i // 2
        left = Inches(0.4 + col * 6.45)
        top = Inches(1.55 + row * 2.6)
        _card(slide, left, top, Inches(6.25), Inches(2.4))
        _text(
            slide,
            left + Inches(0.25),
            top + Inches(0.2),
            Inches(5.8),
            Inches(2.05),
            [(num + "   " + title, 18, NAVY, True), (body, 14, INK, False)],
        )


def _slide_dials(prs, page, total) -> None:
    slide = _blank(prs)
    _header(slide, "09   Steps 6–7", "What the dials mean after Stop")
    _footer(slide, page, total)
    _text(
        slide,
        Inches(0.5),
        Inches(1.5),
        Inches(12.3),
        Inches(0.7),
        [("While the camera runs, the needle is the live angle. The coloured marks appear only after Stop. The needle then stays on the last angle.", 16, INK, False)],
    )
    marks = [
        (CYAN, NAVY, "Min", "Cyan line", "Smallest angle in that take."),
        (CORAL, WHITE, "Max", "Coral line", "Largest angle in that take."),
        (AMBER, NAVY, "Mode", "Amber triangle", "Most common 10° bin. The angle you held most often."),
    ]
    for i, (fill, ink, title, kind, body) in enumerate(marks):
        left = Inches(0.4 + i * 4.25)
        shape = _card(slide, left, Inches(2.4), Inches(4.05), Inches(2.7))
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        _text(
            slide,
            left + Inches(0.25),
            Inches(2.6),
            Inches(3.6),
            Inches(2.3),
            [(title, 26, ink, True), (kind, 16, ink, True), (body, 15, ink, False)],
        )
    _card(slide, Inches(0.4), Inches(5.3), Inches(12.5), Inches(1.55))
    _text(
        slide,
        Inches(0.65),
        Inches(5.5),
        Inches(12.0),
        Inches(1.2),
        [
            ("Same numbers are printed under each dial, for example  min 23°    max 106°    mode 40°.", 16, INK, False),
            ("A colour key with these three marks sits under the “LIVE ANGLES” title so it stays on screen.", 16, INK, False),
        ],
    )


def _slide_limits(prs, page, total) -> None:
    slide = _blank(prs)
    _header(slide, "10   Read this with the demo", "What the recording can and cannot show")
    _footer(slide, page, total)
    items = [
        ("Saved each run", "A timestamped folder under data/output: keypoint CSV (u, v, and x, y, z in metres when 3D ran), angle CSV, and the overlay video. Those recordings stay on this machine."),
        ("“No depth” on a wrist", "The colour joint was found, but the aligned depth at that pixel was missing, too close, or too far. The other wrist can still have X, Y, Z."),
        ("Shoulder dial blank", "The hip was not visible or was below confidence 0.5. Elbow and wrist do not need the hip, so they can keep moving."),
        ("Not a clinic measure", "Shoulder here is hip–shoulder–elbow. Wrist uses the index fingertip. 2D angles fail when the limb points at the camera. Smoothing is only on the 3D position, not on the dials."),
    ]
    for i, (title, body) in enumerate(items):
        top = Inches(1.5 + i * 1.35)
        _card(slide, Inches(0.4), top, Inches(12.5), Inches(1.25))
        _text(
            slide,
            Inches(0.65),
            top + Inches(0.12),
            Inches(12.0),
            Inches(1.05),
            [(title, 16, NAVY, True), (body, 13, INK, False)],
        )


if __name__ == "__main__":
    path = build()
    print(path)
