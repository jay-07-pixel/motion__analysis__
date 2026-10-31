"""Click four distinct toe pixels on a RealSense .bag and deproject to metres.

Experiment-only. Does not change the live capture pipeline.

Usage (project Python, RealSense SDK required):

  python experiments/anatomical_landmark_audit/scripts/label_toes_from_bag.py ^
    --bag PATH\\to\\feet.bag --out DIR

Keys:
  space  next frame
  b      previous buffered frame (limited)
  1-4    select landmark then click: 1 LBigToe  2 LSmallToe  3 RBigToe  4 RSmallToe
  u      undo last click for the active landmark
  s      save CSV
  q      quit

Invalid depth is stored empty, never guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from src.capture.bag_file import BagFileRGB
from src.capture.depth_util import median_depth_m

LANDMARKS = ("LBigToe", "LSmallToe", "RBigToe", "RSmallToe")
KEYS = {ord("1"): "LBigToe", ord("2"): "LSmallToe", ord("3"): "RBigToe", ord("4"): "RSmallToe"}
COLOURS = {
    "LBigToe": (0, 220, 255),
    "LSmallToe": (0, 165, 255),
    "RBigToe": (255, 80, 80),
    "RSmallToe": (255, 0, 180),
}


def deproject(intrinsics, u, v, z_m):
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(z_m))
    return float(point[0]), float(point[1]), float(point[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-depth-m", type=float, default=0.3)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument("--sample-window", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    source = BagFileRGB(args.bag, loop=False, realtime=False, enable_depth=True)
    source.start()
    active = "LBigToe"
    labels: dict[int, dict[str, dict]] = {}
    frames: list[np.ndarray] = []
    index = -1

    def grab():
        image = source.get_frame()
        if image is None:
            return False
        frames.append(image.copy())
        return True

    if not grab():
        source.stop()
        raise SystemExit("No frames in bag.")
    index = 0

    window = "label_toes (1 LBig 2 LSmall 3 RBig 4 RSmall | click | s save | q quit)"
    cv2.namedWindow(window)

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or index < 0:
            return
        z_m = source.depth_distance_m(float(x), float(y), args.sample_window)
        rec = labels.setdefault(index, {})
        if z_m < args.min_depth_m or z_m > args.max_depth_m or not np.isfinite(z_m) or z_m <= 0:
            rec[active] = {
                "u_px": x,
                "v_px": y,
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "invalid_depth": True,
            }
            print(f"frame {index} {active} ({x},{y}) INVALID DEPTH z={z_m}")
            return
        intrinsics = source.color_intrinsics()
        xc, yc, zc = deproject(intrinsics, x, y, z_m)
        rec[active] = {
            "u_px": x,
            "v_px": y,
            "x_m": xc,
            "y_m": yc,
            "z_m": zc,
            "invalid_depth": False,
        }
        print(f"frame {index} {active} ({x},{y}) cam XYZ {xc:.4f} {yc:.4f} {zc:.4f}")

    cv2.setMouseCallback(window, on_mouse)

    while True:
        image = frames[index].copy()
        hud = f"frame {index}/{len(frames)-1}  active={active}  space=next  s=save"
        cv2.putText(image, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for name, pt in labels.get(index, {}).items():
            colour = COLOURS[name]
            cv2.circle(image, (int(pt["u_px"]), int(pt["v_px"])), 6, colour, -1)
            tag = name + (" ?" if pt["invalid_depth"] else "")
            cv2.putText(image, tag, (int(pt["u_px"]) + 8, int(pt["v_px"]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)
        cv2.imshow(window, image)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key in KEYS:
            active = KEYS[key]
        elif key == ord(" "):
            if index == len(frames) - 1:
                if not grab():
                    print("end of bag")
                else:
                    index += 1
            else:
                index += 1
        elif key == ord("b") and index > 0:
            index -= 1
        elif key == ord("u"):
            labels.get(index, {}).pop(active, None)
        elif key == ord("s"):
            csv_path = args.out / "labeled_toes_camera.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "frame",
                        "marker",
                        "u_px",
                        "v_px",
                        "x_m",
                        "y_m",
                        "z_m",
                        "coord_frame",
                        "provenance",
                        "invalid_depth",
                    ],
                )
                writer.writeheader()
                for fi, rec in sorted(labels.items()):
                    for name in LANDMARKS:
                        if name not in rec:
                            continue
                        pt = rec[name]
                        writer.writerow(
                            {
                                "frame": fi,
                                "marker": name,
                                "u_px": pt["u_px"],
                                "v_px": pt["v_px"],
                                "x_m": "" if pt["x_m"] is None else f"{pt['x_m']:.6f}",
                                "y_m": "" if pt["y_m"] is None else f"{pt['y_m']:.6f}",
                                "z_m": "" if pt["z_m"] is None else f"{pt['z_m']:.6f}",
                                "coord_frame": "camera_3d_metre",
                                "provenance": "manual_click_aligned_depth",
                                "invalid_depth": pt["invalid_depth"],
                            }
                        )
            meta = {
                "bag": str(args.bag),
                "n_labeled_frames": len(labels),
                "landmarks": LANDMARKS,
                "y_up_transform": "Y_osim = -Y_cam applied only when building Pose2Sim TRC",
            }
            (args.out / "labeled_toes_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print("saved", csv_path)

    source.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
