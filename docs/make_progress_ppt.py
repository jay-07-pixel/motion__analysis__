"""Build the short progress PPT (16:9). Slide 1 first; 2–3 can be added later."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
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
ASSETS = Path(__file__).resolve().parent / "assets"
CAMERA_ART = ASSETS / "d455f_card.png"
SOFT = RGBColor(0xC5, 0xD4, 0xE8)
LINE = RGBColor(0xE4, 0xE0, 0xD8)


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


def _add_canvas(slide, header_h: float = 1.72) -> None:
    """Cream page + navy header band + teal rule (shared chrome)."""
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
    )
    _fill(bg, CREAM)
    header = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(header_h)
    )
    _fill(header, NAVY)
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0),
        Inches(header_h),
        Inches(13.333),
        Inches(0.07),
    )
    _fill(accent, TEAL)


def generate_d455f_art(path: Path) -> Path:
    """Product-style D455f graphic only — labels are added in PowerPoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1400, 520
    img = Image.new("RGB", (w, h), (11, 31, 58))
    overlay = Image.new("RGB", (w, h), (11, 31, 58))
    od = ImageDraw.Draw(overlay)
    for i, r in enumerate(range(620, 40, -8)):
        shade = 18 + min(36, i)
        od.ellipse(
            (w // 2 - r, h // 2 - int(r * 0.38), w // 2 + r, h // 2 + int(r * 0.38)),
            fill=(11 + shade, 40 + shade, 70 + shade // 2),
        )
    img = Image.blend(img, overlay.filter(ImageFilter.GaussianBlur(26)), 0.58)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w, 8), fill=(31, 181, 166))

    # Long slim D455-style body, larger so it reads on the slide.
    bx1, by1, bx2, by2 = 50, 110, 1350, 410
    body = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body)
    bd.rounded_rectangle((bx1, by1, bx2, by2), radius=36, fill=(18, 20, 24, 255))
    bd.rounded_rectangle(
        (bx1 + 10, by1 + 8, bx2 - 10, by1 + 44),
        radius=14,
        fill=(52, 56, 64, 220),
    )
    img = Image.alpha_composite(img.convert("RGBA"), body).convert("RGB")
    d = ImageDraw.Draw(img)

    def port(cx: int, cy: int, r: int, ring, glass) -> None:
        d.ellipse((cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8), fill=(8, 9, 12))
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ring)
        d.ellipse((cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8), fill=glass)
        d.ellipse((cx - r // 3, cy - r // 2, cx, cy - r // 8), fill=(230, 236, 242))

    cy = 260
    port(250, cy, 46, (36, 40, 48), (22, 70, 78))
    port(455, cy, 30, (70, 58, 36), (120, 92, 40))
    port(660, cy, 46, (36, 40, 48), (22, 70, 78))
    port(980, cy, 58, (31, 181, 166), (18, 90, 110))
    d.rounded_rectangle((1230, 228, 1290, 292), radius=7, fill=(8, 9, 12))
    d.rounded_rectangle((1240, 242, 1280, 278), radius=5, fill=(70, 76, 86))
    img.save(path, "PNG")
    return path


def _card_title(slide, x, y, w, accent: RGBColor, tag: str) -> None:
    """Small coloured section label inside a card."""
    box = _textbox(slide, x, y, w, Inches(0.28))
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(r, tag, 12, accent, True)


def _set_line(shape, color: RGBColor, width_pt: float = 1.25) -> None:
    """Visible stroke (used for the 3D ‘next’ outline)."""
    shape.line.color.rgb = color
    shape.line.width = Pt(width_pt)


def _dash_line(shape, color: RGBColor, width_pt: float = 1.5) -> None:
    """Dashed outline — used only for the not-yet-built 3D path."""
    shape.line.color.rgb = color
    shape.line.width = Pt(width_pt)
    shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH


def _center_text(slide, x, y, w, h, text: str, size: int, color: RGBColor, bold: bool = False):
    """One centred line in a box."""
    box = _textbox(slide, x, y, w, h)
    box.text_frame.word_wrap = True
    box.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(r, text, size, color, bold)
    return box


def _right_arrow(slide, x, y, color: RGBColor = TEAL):
    """Small chevron between pipeline nodes."""
    arrow = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, x, y, Inches(0.28), Inches(0.20)
    )
    _fill(arrow, color)
    return arrow


def _add_header_text(
    slide,
    kicker: str,
    title: str,
    subtitle: str,
    title_size: int = 32,
    compact: bool = False,
) -> None:
    """White title block inside the navy header."""
    k_top = 0.12 if compact else 0.18
    t_top = 0.38 if compact else 0.48
    t_h = 0.62 if compact else 0.70
    s_top = 1.02 if compact else 1.16
    box = _textbox(slide, Inches(0.5), Inches(k_top), Inches(12.3), Inches(0.28))
    box.text_frame.clear()
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(r, kicker, 12, TEAL, True)

    box = _textbox(slide, Inches(0.5), Inches(t_top), Inches(12.3), Inches(t_h))
    box.text_frame.word_wrap = True
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(r, title, title_size, WHITE, True)

    box = _textbox(slide, Inches(0.5), Inches(s_top), Inches(12.3), Inches(0.36))
    box.text_frame.word_wrap = True
    r = box.text_frame.paragraphs[0].add_run()
    _set_run(r, subtitle, 14 if compact else 15, RGBColor(0xC5, 0xD4, 0xE8), False)


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
    """Problem, objective, 2D progress, D455f visual, 2D→3D status."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_canvas(slide, header_h=1.52)
    _add_header_text(
        slide,
        "SLIDE 1   ·   PROBLEM STATEMENT  &  PROJECT OBJECTIVE",
        "2D & 3D Motion Analysis Using a Single RGB-D Camera",
        "Intel RealSense D455f   ·   compact, real-time, single-camera motion analysis",
        title_size=26,
        compact=True,
    )

    generate_d455f_art(CAMERA_ART)

    left = Inches(0.38)
    col_w = Inches(8.42)
    right = Inches(8.98)
    rail_w = Inches(3.96)
    y0 = Inches(1.76)

    # --- Problem ---
    ph = Inches(1.42)
    _round_card(slide, left, y0, col_w, ph, CARD)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, y0, Inches(0.10), ph)
    _fill(bar, CORAL)
    _card_title(slide, left + Inches(0.28), y0 + Inches(0.10), col_w - Inches(0.44), CORAL, "PROBLEM STATEMENT")
    pbox = _textbox(slide, left + Inches(0.28), y0 + Inches(0.40), col_w - Inches(0.48), Inches(0.92))
    pbox.text_frame.word_wrap = True
    r = pbox.text_frame.paragraphs[0].add_run()
    _set_run(
        r,
        "Traditional motion-analysis systems often rely on expensive multi-camera "
        "motion-capture setups or specialized sensors. There is a need for a compact, "
        "real-time and accessible system that can capture human movement and extract "
        "meaningful joint-level motion data using a single camera.",
        13,
        INK,
        False,
    )

    # --- Objective ---
    oy = y0 + ph + Inches(0.12)
    oh = Inches(1.78)
    _round_card(slide, left, oy, col_w, oh, CARD)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, oy, Inches(0.10), oh)
    _fill(bar, TEAL)
    _card_title(slide, left + Inches(0.28), oy + Inches(0.10), col_w - Inches(0.44), TEAL, "PROJECT OBJECTIVE")
    goals = [
        "Develop a 2D + 3D human motion analysis system using a single Intel RealSense D455f",
        "Detect human body joints and quantify movement parameters",
        "Support both live camera input and recorded video",
        "Provide a foundation for real-world 3D motion measurement",
    ]
    gbox = _textbox(slide, left + Inches(0.28), oy + Inches(0.40), col_w - Inches(0.48), Inches(1.30))
    tf = gbox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(goals):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(5)
        r = para.add_run()
        _set_run(r, "▸   " + line, 13, INK, False)

    # --- Current progress ---
    py = oy + oh + Inches(0.12)
    ph2 = Inches(1.88)
    _round_card(slide, left, py, col_w, ph2, CARD)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, py, Inches(0.10), ph2)
    _fill(bar, AMBER)
    _card_title(
        slide,
        left + Inches(0.28),
        py + Inches(0.10),
        col_w - Inches(0.44),
        AMBER,
        "CURRENT PROGRESS  —  2D PHASE COMPLETED",
    )
    done = [
        "33 body landmarks detected using MediaPipe Pose",
        "Joint coordinates represented in RGB image pixels (u, v)",
        "Elbow angles calculated in the image plane",
        "Wrist trajectory and pixel velocity calculated",
        "CSV / video output and post-session analysis report implemented",
    ]
    dbox = _textbox(slide, left + Inches(0.28), py + Inches(0.40), col_w - Inches(0.48), Inches(1.38))
    tf = dbox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(done):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(3)
        r = para.add_run()
        _set_run(r, "▸   " + line, 13, INK, False)

    # --- Right rail: camera + 2D / 3D ---
    cam_h = Inches(2.72)
    _round_card(slide, right, y0, rail_w, cam_h, NAVY)
    slide.shapes.add_picture(
        str(CAMERA_ART),
        right + Inches(0.10),
        y0 + Inches(0.08),
        rail_w - Inches(0.20),
        Inches(1.95),
    )
    cap = _textbox(slide, right + Inches(0.14), y0 + Inches(2.06), rail_w - Inches(0.28), Inches(0.26))
    cap.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = cap.text_frame.paragraphs[0].add_run()
    _set_run(r, "INTEL REALSENSE  D455f", 12, TEAL, True)
    sub = _textbox(slide, right + Inches(0.14), y0 + Inches(2.32), rail_w - Inches(0.28), Inches(0.28))
    sub.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = sub.text_frame.paragraphs[0].add_run()
    _set_run(r, "Single RGB-D camera  ·  USB 3", 11, SOFT, False)

    # Progress indicator under the camera.
    iy = y0 + cam_h + Inches(0.14)
    pill_h = Inches(0.92)
    # 2D — completed (filled teal).
    done_pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, right, iy, rail_w, pill_h)
    _fill(done_pill, TEAL)
    try:
        done_pill.adjustments[0] = 0.12
    except Exception:
        pass
    t1 = _textbox(slide, right + Inches(0.16), iy + Inches(0.10), rail_w - Inches(0.32), Inches(0.28))
    r = t1.text_frame.paragraphs[0].add_run()
    _set_run(r, "2D   ·   COMPLETED", 13, WHITE, True)
    t1.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    t2 = _textbox(slide, right + Inches(0.16), iy + Inches(0.40), rail_w - Inches(0.32), Inches(0.42))
    t2.text_frame.word_wrap = True
    r = t2.text_frame.paragraphs[0].add_run()
    _set_run(r, "RGB joints, elbow °, wrist path", 11, RGBColor(0xE8, 0xFF, 0xFB), False)
    t2.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # Arrow 2D → 3D.
    arrow = slide.shapes.add_shape(
        MSO_SHAPE.DOWN_ARROW,
        right + Inches(1.72),
        iy + pill_h + Inches(0.06),
        Inches(0.52),
        Inches(0.28),
    )
    _fill(arrow, AMBER)

    # 3D — next phase (outline amber on cream).
    ny = iy + pill_h + Inches(0.40)
    next_pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, right, ny, rail_w, pill_h)
    next_pill.fill.solid()
    next_pill.fill.fore_color.rgb = WHITE
    _set_line(next_pill, AMBER, 2.0)
    try:
        next_pill.adjustments[0] = 0.12
    except Exception:
        pass
    t3 = _textbox(slide, right + Inches(0.16), ny + Inches(0.10), rail_w - Inches(0.32), Inches(0.28))
    r = t3.text_frame.paragraphs[0].add_run()
    _set_run(r, "3D   ·   NEXT PHASE", 13, AMBER, True)
    t3.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    t4 = _textbox(slide, right + Inches(0.16), ny + Inches(0.40), rail_w - Inches(0.32), Inches(0.42))
    t4.text_frame.word_wrap = True
    r = t4.text_frame.paragraphs[0].add_run()
    _set_run(r, "Depth → camera metres / m/s", 11, MUTED, False)
    t4.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # Status strip sitting on the footer band.
    _add_footer(
        slide,
        "Current status    ·    2D analysis completed     →     3D depth-based analysis is the next phase",
    )


def add_slide_2(prs: Presentation) -> None:
    """End-to-end 2D pipeline plus a dashed, not-yet-built 3D path."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_canvas(slide, header_h=1.48)
    _add_header_text(
        slide,
        "SLIDE 2   ·   SOLUTION  &  SYSTEM WORKFLOW",
        "Proposed Solution — End-to-End Motion Analysis Pipeline",
        "2D path is built     ·     dashed 3D path is the next phase, not implemented",
        title_size=24,
        compact=True,
    )

    # One-line description.
    lead = _round_card(slide, Inches(0.38), Inches(1.70), Inches(12.56), Inches(0.58), CARD)
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.38), Inches(1.70), Inches(0.10), Inches(0.58)
    )
    _fill(bar, TEAL)
    lbox = _textbox(slide, Inches(0.64), Inches(1.78), Inches(12.16), Inches(0.42))
    lbox.text_frame.word_wrap = True
    r = lbox.text_frame.paragraphs[0].add_run()
    _set_run(
        r,
        "A single RGB-D camera captures the subject, MediaPipe extracts 33 body "
        "landmarks, and the system converts these landmarks into measurable motion parameters.",
        13,
        INK,
        False,
    )

    # ---- Built 2D workflow ----
    tagbox = _textbox(slide, Inches(0.38), Inches(2.36), Inches(6.0), Inches(0.24))
    r = tagbox.text_frame.paragraphs[0].add_run()
    _set_run(r, "BUILT   ·   2D MOTION ANALYSIS", 11, TEAL, True)

    nodes = [
        (TEAL, "01  INPUT", "Live D455f\nUploaded video", "two sources, one pipeline"),
        (NAVY_MID, "02  RGB FRAME", "Colour frame\ncapture", "image plane only"),
        (TEAL, "03  MEDIAPIPE", "33 landmarks\n2D pixels (u, v)", "origin: top-left"),
        (AMBER, "04  ANALYSIS", "Elbow angle (image °)\nWrist trail + px/s", "angles  +  trajectory"),
        (CORAL, "05  OUTPUT", "Visualization & report\nCSV + overlay video", "post-session graphs"),
    ]
    n = len(nodes)
    left0 = 0.38
    right_edge = 12.94
    arrow_w = 0.30
    gap = 0.08
    usable = right_edge - left0 - (n - 1) * (arrow_w + 2 * gap)
    nw = usable / n
    ny = 2.64
    nh = 2.28
    ax_y = ny + nh / 2 - 0.10

    x = left0
    for i, (color, tag, title, note) in enumerate(nodes):
        xi = Inches(x)
        _round_card(slide, xi, Inches(ny), Inches(nw), Inches(nh), CARD)
        top = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, xi, Inches(ny), Inches(nw), Inches(0.08))
        _fill(top, color)

        tbox = _textbox(slide, xi + Inches(0.10), Inches(ny + 0.16), Inches(nw - 0.20), Inches(0.26))
        r = tbox.text_frame.paragraphs[0].add_run()
        _set_run(r, tag, 11, color, True)

        # Two stacked chips so forks (live/upload, angles/wrist) stay visible.
        chip_h = 0.46
        for k, line in enumerate(title.split("\n")):
            cy = ny + 0.48 + k * (chip_h + 0.08)
            chip = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                xi + Inches(0.10),
                Inches(cy),
                Inches(nw - 0.20),
                Inches(chip_h),
            )
            chip.fill.solid()
            chip.fill.fore_color.rgb = CREAM
            chip.line.color.rgb = LINE
            chip.line.width = Pt(0.75)
            try:
                chip.adjustments[0] = 0.18
            except Exception:
                pass
            cbox = _textbox(
                slide,
                xi + Inches(0.16),
                Inches(cy + 0.08),
                Inches(nw - 0.32),
                Inches(0.32),
            )
            cbox.text_frame.word_wrap = True
            cbox.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
            rr = cbox.text_frame.paragraphs[0].add_run()
            _set_run(rr, line, 12, INK, True)

        nbox = _textbox(slide, xi + Inches(0.10), Inches(ny + 1.90), Inches(nw - 0.20), Inches(0.30))
        nbox.text_frame.word_wrap = True
        r = nbox.text_frame.paragraphs[0].add_run()
        _set_run(r, note, 10, MUTED, False)

        x = x + nw
        if i < n - 1:
            _right_arrow(slide, Inches(x + gap), Inches(ax_y), TEAL)
            x = x + arrow_w + 2 * gap

    # ---- Dashed 3D extension (not built) ----
    band_y = 5.08
    band_h = 1.90
    band = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.38),
        Inches(band_y),
        Inches(12.56),
        Inches(band_h),
    )
    band.fill.solid()
    band.fill.fore_color.rgb = WHITE
    _dash_line(band, AMBER, 1.75)
    try:
        band.adjustments[0] = 0.04
    except Exception:
        pass

    _center_text(
        slide,
        Inches(0.50),
        Inches(band_y + 0.08),
        Inches(12.32),
        Inches(0.26),
        "3D EXTENSION   ·   NEXT PHASE   ·   NOT IMPLEMENTED",
        11,
        AMBER,
        True,
    )

    ext = [
        "2D pixel (u, v)\n+ RealSense depth",
        "Depth\nalignment",
        "3D\ndeprojection",
        "Camera\ncoordinates (m)",
        "3D motion\n/ speed",
    ]
    en = len(ext)
    e_left = 0.58
    e_right = 12.74
    e_arrow = 0.26
    e_gap = 0.06
    e_usable = e_right - e_left - (en - 1) * (e_arrow + 2 * e_gap)
    ew = e_usable / en
    ey = band_y + 0.42
    eh = 1.32
    ex = e_left
    for i, text in enumerate(ext):
        node = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(ex),
            Inches(ey),
            Inches(ew),
            Inches(eh),
        )
        node.fill.solid()
        node.fill.fore_color.rgb = CREAM
        _dash_line(node, AMBER, 1.25)
        try:
            node.adjustments[0] = 0.10
        except Exception:
            pass
        tbox = _textbox(slide, Inches(ex + 0.08), Inches(ey + 0.22), Inches(ew - 0.16), Inches(0.92))
        tbox.text_frame.word_wrap = True
        for j, line in enumerate(text.split("\n")):
            para = tbox.text_frame.paragraphs[0] if j == 0 else tbox.text_frame.add_paragraph()
            para.alignment = PP_ALIGN.CENTER
            para.space_after = Pt(1)
            rr = para.add_run()
            _set_run(rr, line, 12, MUTED, True)
        ex = ex + ew
        if i < en - 1:
            _right_arrow(slide, Inches(ex + e_gap), Inches(ey + eh / 2 - 0.10), AMBER)
            ex = ex + e_arrow + 2 * e_gap

    _add_footer(
        slide,
        "Current implementation: RGB-based 2D motion analysis     ·     "
        "Next phase: RealSense depth integration for true 3D coordinates",
    )


def add_slide_3(prs: Presentation) -> None:
    """Technology stack groups + modular architecture (2D now, depth later)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_canvas(slide, header_h=1.48)
    _add_header_text(
        slide,
        "SLIDE 3   ·   TECHNOLOGY STACK  &  IMPLEMENTATION",
        "Technology Stack & System Architecture",
        "Modular 2D pipeline   ·   hardware, vision, app, data   ·   ready for depth later",
        title_size=26,
        compact=True,
    )

    groups = [
        (
            TEAL,
            "HARDWARE",
            [
                ("Intel RealSense D455f", "RGB-D camera  ·  USB 3.0  ·  future depth integration"),
            ],
        ),
        (
            AMBER,
            "COMPUTER VISION  &  AI",
            [
                ("MediaPipe Pose", "33 body landmarks  ·  real-time pose estimation"),
                ("OpenCV", "Frame processing  ·  video I/O  ·  overlay generation"),
            ],
        ),
        (
            CORAL,
            "APPLICATION DEVELOPMENT",
            [
                ("Python", "Core processing pipeline"),
                ("Tkinter", "Live UI  ·  camera / file  ·  session control"),
                ("Matplotlib", "Post-session graphs  ·  angle / time  ·  statistics"),
            ],
        ),
        (
            NAVY_MID,
            "CONFIGURATION  &  DATA",
            [
                ("YAML", "Central config — camera, pose, analysis, output"),
                ("CSV / JSON / MP4", "Motion data  ·  run metadata  ·  annotated video"),
            ],
        ),
    ]

    left = 0.38
    col_w = 8.28
    y = 1.68
    gap = 0.10
    # Four stacked category cards; heights follow how many tools sit in each.
    heights = (0.82, 1.12, 1.42, 1.12)
    for (color, heading, tools), hh in zip(groups, heights):
        xi = Inches(left)
        yi = Inches(y)
        _round_card(slide, xi, yi, Inches(col_w), Inches(hh), CARD)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, xi, yi, Inches(0.10), Inches(hh))
        _fill(bar, color)
        _card_title(slide, xi + Inches(0.26), yi + Inches(0.08), Inches(col_w - 0.40), color, heading)
        tbox = _textbox(
            slide,
            xi + Inches(0.26),
            yi + Inches(0.36),
            Inches(col_w - 0.44),
            Inches(hh - 0.44),
        )
        tf = tbox.text_frame
        tf.word_wrap = True
        for i, (name, detail) in enumerate(tools):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.space_after = Pt(3)
            r = para.add_run()
            _set_run(r, name, 13, INK, True)
            r = para.add_run()
            _set_run(r, "    " + detail, 12, MUTED, False)
        y += hh + gap

    # ---- Architecture rail ----
    rx = 8.84
    rw = 4.10
    ry = 1.68
    rh = 4.78
    rail = _round_card(slide, Inches(rx), Inches(ry), Inches(rw), Inches(rh), NAVY)
    _center_text(
        slide,
        Inches(rx + 0.12),
        Inches(ry + 0.10),
        Inches(rw - 0.24),
        Inches(0.26),
        "ARCHITECTURE",
        11,
        TEAL,
        True,
    )

    layers = [
        "RealSense  /  Video",
        "Capture layer",
        "Pose estimation\n(MediaPipe)",
        None,  # split analysis drawn separately
        "Visualization\n& reporting",
        "CSV  /  video  /  report",
    ]
    node_w = rw - 0.40
    nx = rx + 0.20
    ny = ry + 0.44
    simple_h = 0.48
    split_h = 0.78
    arrow_h = 0.14
    for item in layers:
        if item is None:
            # Motion analysis fork: 2D angles | wrist motion
            parent = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(nx),
                Inches(ny),
                Inches(node_w),
                Inches(split_h),
            )
            parent.fill.solid()
            parent.fill.fore_color.rgb = NAVY_MID
            parent.line.fill.background()
            try:
                parent.adjustments[0] = 0.10
            except Exception:
                pass
            _center_text(
                slide,
                Inches(nx),
                Inches(ny + 0.02),
                Inches(node_w),
                Inches(0.22),
                "Motion analysis",
                10,
                SOFT,
                True,
            )
            chip_w = (node_w - 0.18) / 2
            for k, label in enumerate(("2D angles", "Wrist motion")):
                cx = nx + 0.06 + k * (chip_w + 0.06)
                chip = slide.shapes.add_shape(
                    MSO_SHAPE.ROUNDED_RECTANGLE,
                    Inches(cx),
                    Inches(ny + 0.28),
                    Inches(chip_w),
                    Inches(0.42),
                )
                _fill(chip, TEAL if k == 0 else AMBER)
                try:
                    chip.adjustments[0] = 0.18
                except Exception:
                    pass
                _center_text(
                    slide,
                    Inches(cx),
                    Inches(ny + 0.34),
                    Inches(chip_w),
                    Inches(0.30),
                    label,
                    10,
                    WHITE,
                    True,
                )
            ny += split_h
        else:
            node = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(nx),
                Inches(ny),
                Inches(node_w),
                Inches(simple_h),
            )
            node.fill.solid()
            node.fill.fore_color.rgb = WHITE
            node.line.fill.background()
            try:
                node.adjustments[0] = 0.12
            except Exception:
                pass
            tbox = _textbox(slide, Inches(nx + 0.08), Inches(ny + 0.04), Inches(node_w - 0.16), Inches(simple_h - 0.06))
            tbox.text_frame.word_wrap = True
            for j, line in enumerate(item.split("\n")):
                para = tbox.text_frame.paragraphs[0] if j == 0 else tbox.text_frame.add_paragraph()
                para.alignment = PP_ALIGN.CENTER
                para.space_after = Pt(0)
                rr = para.add_run()
                _set_run(rr, line, 11, INK, True)
            ny += simple_h

        if item != layers[-1]:
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.DOWN_ARROW,
                Inches(rx + rw / 2 - 0.11),
                Inches(ny + 0.01),
                Inches(0.22),
                Inches(arrow_h),
            )
            _fill(arrow, TEAL)
            ny += arrow_h + 0.04

    # Bottom banner — modular, ready for 3D depth (not claimed as built).
    banner = _round_card(slide, Inches(0.38), Inches(6.56), Inches(12.56), Inches(0.46), NAVY)
    _center_text(
        slide,
        Inches(0.50),
        Inches(6.64),
        Inches(12.32),
        Inches(0.32),
        "Built as a modular pipeline   →   ready for RealSense depth integration in the 3D phase.",
        13,
        WHITE,
        True,
    )

    _add_footer(
        slide,
        "Jay   ·   IIT Guwahati   ·   python app.py   ·   config.yaml   ·   2D implemented, 3D not claimed",
    )


def add_slide_4(prs: Presentation) -> None:
    """Literature: BlazePose (2D) and Dill et al. (why monocular 3D is weak)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_canvas(slide, header_h=1.48)
    _add_header_text(
        slide,
        "SLIDE 4   ·   LITERATURE REVIEW",
        "Literature Review — 2D Pose Estimation to 3D Motion Analysis",
        "Two papers:  the 2D method we use now    ·    why a single RGB view is not enough for 3D",
        title_size=22,
        compact=True,
    )

    papers = [
        (
            TEAL,
            "PAPER  1",
            "BlazePose: On-device Real-time Body Pose Tracking",
            "Bazarevsky et al.,  Google Research  (2020)",
            "WHAT IT PROVIDES",
            [
                "Real-time single-person pose from RGB image / video",
                "33 keypoints: shoulders, elbows, wrists, hips, knees, ankles",
                "Detector + tracker pipeline for on-device tracking",
                "Aimed at fitness, gesture recognition and AR",
            ],
            "LIMITATION FOR OUR PROJECT",
            "RGB pose is 2D image information. Depth cannot be reliably obtained from a single RGB view — body parts moving toward / away from the camera stay ambiguous.",
            "RELEVANCE",
            "Foundation of our current 2D phase through MediaPipe Pose.",
        ),
        (
            AMBER,
            "PAPER  2",
            "Accuracy Evaluation of 3D Pose Reconstruction through Stereo Fusion for Physical Exercises with MediaPipe Pose",
            "Dill et al.,  Sensors  (2024)",
            "WHAT IT STUDIES",
            [
                "3D pose reconstruction for physical-exercise analysis with MediaPipe Pose",
                "Two camera views + MediaPipe 2D landmarks + triangulation",
                "Compares reconstructed 3D pose with motion-capture ground truth",
                "Stereo fusion beats monocular 3D estimation",
            ],
            "KEY FINDING  /  LIMITATION",
            "3D from one RGB view is ill-posed: many 3D poses share the same 2D projection. MediaPipe “world” landmarks guess depth from an internal body model and are noisy.",
            "RELEVANCE",
            "Supports moving from 2D image measures to sensor-based 3D (RealSense depth).",
        ),
    ]

    left0 = 0.38
    gap = 0.20
    cw = (12.56 - gap) / 2
    top = 1.68
    ch = 4.78
    for i, (
        color,
        tag,
        title,
        cite,
        sec1,
        bullets,
        sec2,
        limit,
        sec3,
        relev,
    ) in enumerate(papers):
        x = Inches(left0 + i * (cw + gap))
        y = Inches(top)
        _round_card(slide, x, y, Inches(cw), Inches(ch), CARD)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(cw), Inches(0.08))
        _fill(bar, color)

        _card_title(slide, x + Inches(0.22), y + Inches(0.16), Inches(cw - 0.40), color, tag)

        tbox = _textbox(slide, x + Inches(0.22), y + Inches(0.40), Inches(cw - 0.44), Inches(0.70))
        tbox.text_frame.word_wrap = True
        r = tbox.text_frame.paragraphs[0].add_run()
        _set_run(r, title, 13, INK, True)

        cbox = _textbox(slide, x + Inches(0.22), y + Inches(1.10), Inches(cw - 0.44), Inches(0.24))
        r = cbox.text_frame.paragraphs[0].add_run()
        _set_run(r, cite, 11, MUTED, False)

        _card_title(slide, x + Inches(0.22), y + Inches(1.36), Inches(cw - 0.40), color, sec1)
        bbox = _textbox(slide, x + Inches(0.22), y + Inches(1.62), Inches(cw - 0.44), Inches(1.22))
        tf = bbox.text_frame
        tf.word_wrap = True
        for j, line in enumerate(bullets):
            para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            para.space_after = Pt(2)
            rr = para.add_run()
            _set_run(rr, "▸   " + line, 12, INK, False)

        _card_title(slide, x + Inches(0.22), y + Inches(2.90), Inches(cw - 0.40), CORAL, sec2)
        lbox = _textbox(slide, x + Inches(0.22), y + Inches(3.16), Inches(cw - 0.44), Inches(0.88))
        lbox.text_frame.word_wrap = True
        r = lbox.text_frame.paragraphs[0].add_run()
        _set_run(r, limit, 12, INK, False)

        _card_title(slide, x + Inches(0.22), y + Inches(4.08), Inches(cw - 0.40), TEAL, sec3)
        rbox = _textbox(slide, x + Inches(0.22), y + Inches(4.32), Inches(cw - 0.44), Inches(0.36))
        rbox.text_frame.word_wrap = True
        r = rbox.text_frame.paragraphs[0].add_run()
        _set_run(r, relev, 12, INK, True)

    banner = _round_card(slide, Inches(0.38), Inches(6.56), Inches(12.56), Inches(0.46), NAVY)
    _center_text(
        slide,
        Inches(0.50),
        Inches(6.64),
        Inches(12.32),
        Inches(0.32),
        "Takeaway    ·    BlazePose / MediaPipe is our 2D base     →     "
        "Dill et al.: true 3D needs more than one RGB view  —  we will use RealSense depth",
        12,
        WHITE,
        True,
    )

    _add_footer(
        slide,
        "Literature is exploratory for this progress talk   ·   we have not claimed a full survey",
    )


def main() -> None:
    """Write the 16:9 progress deck (problem, workflow, stack, literature)."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    add_slide_1(prs)
    add_slide_2(prs)
    add_slide_3(prs)
    add_slide_4(prs)
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
