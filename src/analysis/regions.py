"""Pick which body region to draw and analyse (from config.yaml).

MediaPipe Pose still runs on the whole person. Hands runs too (except Face).
We only keep the joints, angles, gauges, and highlights listed for the region.
"""

from __future__ import annotations


def region_id(config: dict) -> str:
    """Current region key, default full_body."""
    return str(config.get("analysis", {}).get("region", "full_body"))


def region_spec(config: dict, region: str | None = None) -> dict:
    """YAML block for this region, or empty dict if missing."""
    analysis = config.get("analysis") or {}
    key = region if region is not None else str(analysis.get("region", "full_body"))
    return dict((analysis.get("regions") or {}).get(key) or {})


def region_label(config: dict, region: str | None = None) -> str:
    """Button text, e.g. 'Both hands'."""
    spec = region_spec(config, region)
    return str(spec.get("label", region or region_id(config)))


def apply_region(config: dict, region: str) -> dict:
    """Copy config and narrow analysis lists to this region. In-place on the copy."""
    analysis = config.setdefault("analysis", {})
    analysis["region"] = region
    spec = region_spec(config, region)
    if not spec:
        return config

    if "angles" in spec:
        allowed = set(spec.get("angles") or [])
        analysis["angles"] = [row for row in analysis.get("angles", []) if row.get("name") in allowed]

    gauges = spec.get("gauges")
    if gauges is not None:
        analysis["gauge_joints"] = {
            "left": list(gauges.get("left") or []),
            "right": list(gauges.get("right") or []),
        }

    if "highlight" in spec:
        names = set(spec.get("highlight") or [])
        analysis["highlight_joints"] = [
            row for row in analysis.get("highlight_joints", []) if row.get("name") in names
        ]

    if "trail_joint" in spec:
        analysis["trail_joint"] = spec.get("trail_joint")

    joints = spec.get("joints")
    if spec.get("include_hand_landmarks") and joints:
        extra = [str(name) for name in (analysis.get("hand_landmarks") or [])]
        seen = set(joints)
        joints = list(joints) + [name for name in extra if name not in seen]
    analysis["draw_joints"] = joints
    analysis["extra_bones"] = list(spec.get("bones") or [])
    return config
