"""Bousigues et al. 2025 Figure-1 marker names vs this pipeline.

Paper names are correspondences, not palpated ISB skin markers.
MediaPipe joints are never renamed as SAT, epicondyles, styloids, ASIS/PSIS,
femoral epicondyles, FCC, FM1, or FM5. Coordinates are never copied onto a
second anatomical identity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperLandmark:
    paper_name: str
    side: str
    region: str
    method: str  # estimated | unavailable
    source_name: str | None
    notes: str


# OpenCap / Pose2Sim v0.3 study-name correspondence (not palpation):
#   lelbow=lateral humeral epicondyle family, melbow=medial
#   lwrist=lateral/radial-styloid family, mwrist=medial/ulnar-styloid family
#   knee_study=lateral femoral epicondyle family, mknee=medial
#   calc_study=calcaneus family, 5meta=5th metatarsal, toe_study=OpenCap toe
PAPER_LANDMARKS: tuple[PaperLandmark, ...] = (
    PaperLandmark(
        "L_SAT",
        "left",
        "shoulder",
        "unavailable",
        None,
        "Acromion SAT is not MediaPipe shoulder and not Pose2Sim v0.3. "
        "Recommend a palpated acromion click or a physical marker.",
    ),
    PaperLandmark(
        "R_SAT",
        "right",
        "shoulder",
        "unavailable",
        None,
        "Acromion SAT is not MediaPipe shoulder and not Pose2Sim v0.3. "
        "Recommend a palpated acromion click or a physical marker.",
    ),
    PaperLandmark(
        "L_GHJC",
        "left",
        "shoulder",
        "unavailable",
        None,
        "Glenohumeral joint centre is not predicted. L_shoulder_study is not GHJC. "
        "Recommend functional calibration or extra sensors.",
    ),
    PaperLandmark(
        "R_GHJC",
        "right",
        "shoulder",
        "unavailable",
        None,
        "Glenohumeral joint centre is not predicted. r_shoulder_study is not GHJC. "
        "Recommend functional calibration or extra sensors.",
    ),
    PaperLandmark(
        "L_H-LE",
        "left",
        "elbow",
        "estimated",
        "L_lelbow_study",
        "Pose2Sim v0.3 upper LSTM lateral-elbow study marker. Not MediaPipe elbow.",
    ),
    PaperLandmark(
        "L_H-ME",
        "left",
        "elbow",
        "estimated",
        "L_melbow_study",
        "Pose2Sim v0.3 upper LSTM medial-elbow study marker. Not MediaPipe elbow.",
    ),
    PaperLandmark(
        "R_H-LE",
        "right",
        "elbow",
        "estimated",
        "r_lelbow_study",
        "Pose2Sim v0.3 upper LSTM lateral-elbow study marker. Not MediaPipe elbow.",
    ),
    PaperLandmark(
        "R_H-ME",
        "right",
        "elbow",
        "estimated",
        "r_melbow_study",
        "Pose2Sim v0.3 upper LSTM medial-elbow study marker. Not MediaPipe elbow.",
    ),
    PaperLandmark(
        "L_RS",
        "left",
        "wrist",
        "estimated",
        "L_lwrist_study",
        "Pose2Sim v0.3 upper LSTM lateral-wrist study marker (radial-styloid family). "
        "Not MediaPipe wrist or a hand landmark.",
    ),
    PaperLandmark(
        "L_US",
        "left",
        "wrist",
        "estimated",
        "L_mwrist_study",
        "Pose2Sim v0.3 upper LSTM medial-wrist study marker (ulnar-styloid family). "
        "Not MediaPipe wrist or a hand landmark.",
    ),
    PaperLandmark(
        "R_RS",
        "right",
        "wrist",
        "estimated",
        "r_lwrist_study",
        "Pose2Sim v0.3 upper LSTM lateral-wrist study marker (radial-styloid family). "
        "Not MediaPipe wrist or a hand landmark.",
    ),
    PaperLandmark(
        "R_US",
        "right",
        "wrist",
        "estimated",
        "r_mwrist_study",
        "Pose2Sim v0.3 upper LSTM medial-wrist study marker (ulnar-styloid family). "
        "Not MediaPipe wrist or a hand landmark.",
    ),
    PaperLandmark(
        "L_ASIS",
        "left",
        "pelvis",
        "estimated",
        "L.ASIS_study",
        "Pose2Sim v0.3 lower LSTM. Not left_hip and not LHJC_study.",
    ),
    PaperLandmark(
        "R_ASIS",
        "right",
        "pelvis",
        "estimated",
        "r.ASIS_study",
        "Pose2Sim v0.3 lower LSTM. Not right_hip and not RHJC_study.",
    ),
    PaperLandmark(
        "L_PSIS",
        "left",
        "pelvis",
        "estimated",
        "L.PSIS_study",
        "Pose2Sim v0.3 lower LSTM. Not left_hip.",
    ),
    PaperLandmark(
        "R_PSIS",
        "right",
        "pelvis",
        "estimated",
        "r.PSIS_study",
        "Pose2Sim v0.3 lower LSTM. Not right_hip.",
    ),
    PaperLandmark(
        "L_L-FE",
        "left",
        "knee",
        "estimated",
        "L_knee_study",
        "Pose2Sim v0.3 lower LSTM lateral-knee study marker. Not MediaPipe left_knee.",
    ),
    PaperLandmark(
        "L_M-FE",
        "left",
        "knee",
        "estimated",
        "L_mknee_study",
        "Pose2Sim v0.3 lower LSTM medial-knee study marker. Not MediaPipe left_knee.",
    ),
    PaperLandmark(
        "R_L-FE",
        "right",
        "knee",
        "estimated",
        "r_knee_study",
        "Pose2Sim v0.3 lower LSTM lateral-knee study marker. Not MediaPipe right_knee.",
    ),
    PaperLandmark(
        "R_M-FE",
        "right",
        "knee",
        "estimated",
        "r_mknee_study",
        "Pose2Sim v0.3 lower LSTM medial-knee study marker. Not MediaPipe right_knee.",
    ),
    PaperLandmark(
        "L_FCC",
        "left",
        "foot",
        "estimated",
        "L_calc_study",
        "Pose2Sim v0.3 lower LSTM calcaneus study marker. Not MediaPipe left_heel.",
    ),
    PaperLandmark(
        "R_FCC",
        "right",
        "foot",
        "estimated",
        "r_calc_study",
        "Pose2Sim v0.3 lower LSTM calcaneus study marker. Not MediaPipe right_heel.",
    ),
    PaperLandmark(
        "L_FM1",
        "left",
        "foot",
        "estimated",
        "L_toe_study",
        "OpenCap toe_study correspondence, not certified as first-metatarsal head. "
        "Not left_foot_index and not calibrated LBigToe.",
    ),
    PaperLandmark(
        "R_FM1",
        "right",
        "foot",
        "estimated",
        "r_toe_study",
        "OpenCap toe_study correspondence, not certified as first-metatarsal head. "
        "Not right_foot_index and not calibrated RBigToe.",
    ),
    PaperLandmark(
        "L_FM5",
        "left",
        "foot",
        "estimated",
        "L_5meta_study",
        "Pose2Sim v0.3 5th-metatarsal study marker. Not left_foot_index and not LSmallToe.",
    ),
    PaperLandmark(
        "R_FM5",
        "right",
        "foot",
        "estimated",
        "r_5meta_study",
        "Pose2Sim v0.3 5th-metatarsal study marker. Not right_foot_index and not RSmallToe.",
    ),
)

BY_SOURCE: dict[str, PaperLandmark] = {
    spec.source_name: spec for spec in PAPER_LANDMARKS if spec.source_name
}
BY_PAPER: dict[str, PaperLandmark] = {spec.paper_name: spec for spec in PAPER_LANDMARKS}

# Kept visible as themselves; never used as paper-source names.
NOT_PAPER_ALIASES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
    "LBigToe",
    "LSmallToe",
    "RBigToe",
    "RSmallToe",
    "LHJC_study",
    "RHJC_study",
    "L_shoulder_study",
    "r_shoulder_study",
)


def paper_label_for_source(source_name: str) -> str | None:
    spec = BY_SOURCE.get(source_name)
    return spec.paper_name if spec else None


def resolve_paper_landmarks(
    source_xyz: dict[str, tuple[float, float, float]],
) -> list[dict]:
    """One row per paper landmark. Missing/unavailable XYZ stay absent."""
    rows: list[dict] = []
    for spec in PAPER_LANDMARKS:
        xyz = source_xyz.get(spec.source_name) if spec.source_name else None
        present = xyz is not None
        if spec.method == "unavailable":
            reason = spec.notes
        elif spec.source_name is None:
            reason = spec.notes
        elif not present:
            reason = f"{spec.source_name} not present this frame"
        else:
            reason = spec.notes
        rows.append(
            {
                "paper_name": spec.paper_name,
                "side": spec.side,
                "region": spec.region,
                "method": spec.method,
                "source_landmark": spec.source_name or "",
                "x_m": xyz[0] if present else None,
                "y_m": xyz[1] if present else None,
                "z_m": xyz[2] if present else None,
                "present": present,
                "reason": reason,
            }
        )
    return rows


def annotate_overlay_records(records: list[dict]) -> list[dict]:
    """Attach paper_name on existing LSTM rows. Does not add a second XYZ."""
    out: list[dict] = []
    for rec in records:
        item = dict(rec)
        spec = BY_SOURCE.get(str(item.get("name") or ""))
        if spec is not None:
            item["paper_name"] = spec.paper_name
            item["anatomical_status"] = spec.method
        else:
            item.setdefault("paper_name", "")
            item.setdefault("anatomical_status", str(item.get("class") or ""))
        out.append(item)
    return out


def hud_lines(source_names: set[str]) -> list[str]:
    est_present = sum(
        1
        for spec in PAPER_LANDMARKS
        if spec.method == "estimated" and spec.source_name in source_names
    )
    est_total = sum(1 for spec in PAPER_LANDMARKS if spec.method == "estimated")
    unav = sum(1 for spec in PAPER_LANDMARKS if spec.method == "unavailable")
    return [
        f"PAPER set  estimated {est_present}/{est_total} live  "
        f"unavailable {unav} (SAT, GHJC not drawn)",
        "LSTM study names remain; paper tags are correspondences, not palpation",
    ]


def assert_no_forbidden_aliases() -> None:
    sources = {spec.source_name for spec in PAPER_LANDMARKS if spec.source_name}
    overlap = sources.intersection(NOT_PAPER_ALIASES)
    if overlap:
        raise AssertionError(f"Paper set must not alias MediaPipe/toe/HJC names: {sorted(overlap)}")
    for spec in PAPER_LANDMARKS:
        if spec.method == "unavailable":
            assert spec.source_name is None
        if spec.method == "estimated":
            assert spec.source_name is not None
            assert spec.source_name not in NOT_PAPER_ALIASES
