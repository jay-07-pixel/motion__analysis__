"""Validate participant height and mass for live Pose2Sim LSTM sessions.

UI fields are centimetres and kilograms. Inference uses metres and kilograms.
Live sessions must pass explicit values; they are never filled with a default person.
"""

from __future__ import annotations


class ParticipantError(ValueError):
    """User-facing anthropometric input error."""


HEIGHT_CM_MIN = 100.0
HEIGHT_CM_MAX = 250.0
MASS_KG_MIN = 25.0
MASS_KG_MAX = 250.0
HEIGHT_M_MIN = HEIGHT_CM_MIN / 100.0
HEIGHT_M_MAX = HEIGHT_CM_MAX / 100.0


def _parse_positive_number(raw: object, label: str) -> float:
    if raw is None:
        raise ParticipantError(f"{label} is required.")
    text = str(raw).strip().replace(",", ".")
    if text == "":
        raise ParticipantError(f"{label} is required.")
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ParticipantError(f"{label} must be a number.") from exc
    if not (value == value) or value in (float("inf"), float("-inf")):
        raise ParticipantError(f"{label} must be a finite number.")
    if value <= 0.0:
        raise ParticipantError(f"{label} must be greater than zero.")
    return value


def parse_ui_height_cm(raw: object) -> float:
    """Parse a height typed in centimetres. Returns metres."""
    cm = _parse_positive_number(raw, "Height (cm)")
    if cm < HEIGHT_CM_MIN or cm > HEIGHT_CM_MAX:
        raise ParticipantError(
            f"Height (cm) must be between {HEIGHT_CM_MIN:.0f} and {HEIGHT_CM_MAX:.0f}."
        )
    return cm / 100.0


def parse_ui_mass_kg(raw: object) -> float:
    """Parse a mass typed in kilograms. Returns kilograms."""
    kg = _parse_positive_number(raw, "Weight (kg)")
    if kg < MASS_KG_MIN or kg > MASS_KG_MAX:
        raise ParticipantError(
            f"Weight (kg) must be between {MASS_KG_MIN:.0f} and {MASS_KG_MAX:.0f}."
        )
    return kg


def require_participant_metres_kg(height_m: object, mass_kg: object) -> tuple[float, float]:
    """Accept already-converted metres and kilograms. No silent default person."""
    if height_m is None or mass_kg is None:
        raise ParticipantError(
            "Participant height (m) and mass (kg) are required for Live Pose2Sim."
        )
    h = _parse_positive_number(height_m, "Height (m)")
    m = _parse_positive_number(mass_kg, "Weight (kg)")
    if h < HEIGHT_M_MIN or h > HEIGHT_M_MAX:
        raise ParticipantError(
            f"Height (m) must be between {HEIGHT_M_MIN:.2f} and {HEIGHT_M_MAX:.2f}."
        )
    if m < MASS_KG_MIN or m > MASS_KG_MAX:
        raise ParticipantError(
            f"Weight (kg) must be between {MASS_KG_MIN:.0f} and {MASS_KG_MAX:.0f}."
        )
    return h, m
