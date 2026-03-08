#!/usr/bin/env python3
import argparse
import colorsys
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import yaml


@dataclass
class Record:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    fields: List[str]


class Tracklet:
    def __init__(self, track_id: int) -> None:
        self.track_id = track_id
        self.frames: List[int] = []
        self.centers: List[Tuple[float, float]] = []
        self.sizes: List[Tuple[float, float]] = []

    def add(self, frame: int, center: Tuple[float, float], size: Tuple[float, float]) -> None:
        self.frames.append(frame)
        self.centers.append(center)
        self.sizes.append(size)

    def sort_by_frame(self) -> None:
        if len(self.frames) <= 1:
            return
        order = sorted(range(len(self.frames)), key=self.frames.__getitem__)
        self.frames = [self.frames[i] for i in order]
        self.centers = [self.centers[i] for i in order]
        self.sizes = [self.sizes[i] for i in order]

    def merge_from(self, other: "Tracklet") -> None:
        self.frames.extend(other.frames)
        self.centers.extend(other.centers)
        self.sizes.extend(other.sizes)

    @property
    def start_frame(self) -> int:
        return self.frames[0]

    @property
    def end_frame(self) -> int:
        return self.frames[-1]

    @property
    def start_center(self) -> Tuple[float, float]:
        return self.centers[0]

    @property
    def end_center(self) -> Tuple[float, float]:
        return self.centers[-1]

    def velocity(self, window: int, at_end: bool) -> Optional[Tuple[float, float]]:
        if len(self.frames) < 2:
            return None
        window = max(2, int(window))
        if at_end:
            start_index = max(1, len(self.frames) - window + 1)
            indices = range(start_index, len(self.frames))
        else:
            end_index = min(len(self.frames), window)
            indices = range(1, end_index)

        vxs = []
        vys = []
        for idx in indices:
            dt = self.frames[idx] - self.frames[idx - 1]
            if dt <= 0:
                continue
            dx = self.centers[idx][0] - self.centers[idx - 1][0]
            dy = self.centers[idx][1] - self.centers[idx - 1][1]
            vxs.append(dx / dt)
            vys.append(dy / dt)

        if not vxs:
            return None
        return (sum(vxs) / len(vxs), sum(vys) / len(vys))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process MOT results to stitch tracklets that disappear/appear mid-frame "
            "with consistent motion."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing tracking outputs (e.g. data/.../tracking/sort).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file to infer frame size (expects data.shape or data_writer.shape).",
    )
    parser.add_argument("--frame-width", type=int, default=None, help="Override frame width.")
    parser.add_argument("--frame-height", type=int, default=None, help="Override frame height.")
    parser.add_argument(
        "--border-margin",
        type=float,
        default=60.0,
        help="Margin (px) defining the border area where entries/exits are allowed.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=30,
        help="Max frame gap allowed between the old end and new start.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=80.0,
        help="Max distance (px) between predicted old position and new start.",
    )
    parser.add_argument(
        "--max-angle-deg",
        type=float,
        default=45.0,
        help="Max angle (deg) between old/new velocity directions.",
    )
    parser.add_argument(
        "--max-speed-ratio",
        type=float,
        default=3.0,
        help="Max ratio between old/new speeds (<=0 disables).",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=0.25,
        help="Min speed (px/frame) to enforce direction checks.",
    )
    parser.add_argument(
        "--velocity-window",
        type=int,
        default=3,
        help="Number of points to estimate start/end velocity.",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Render post-processed tracks into img_draw_post directories.",
    )
    parser.add_argument(
        "--draw-output-name",
        default="img_draw_post",
        help="Folder name for rendered images inside each sequence output.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Override image root directory (uses config data_loader.data_dir_prefix).",
    )
    parser.add_argument(
        "--image-postfix",
        default=None,
        help="Override image postfix directory (uses config data_loader.data_dir_postfix).",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help=(
            "Process only one sequence name (e.g. seq22). "
            "When unset, process all canonical files at <input-root>/<seq>/thermal/<seq>_thermal.txt."
        ),
    )
    return parser.parse_args()


def load_shape_from_config(config_path: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not config_path:
        return None, None
    if not os.path.isfile(config_path):
        return None, None
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        return None, None
    for key in ("data", "data_writer"):
        shape = cfg.get(key, {}).get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            height = int(shape[0])
            width = int(shape[1])
            return width, height
    return None, None


def load_data_paths_from_config(config_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not config_path:
        return None, None
    if not os.path.isfile(config_path):
        return None, None
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        return None, None
    data_loader = cfg.get("data_loader", {})
    prefix = data_loader.get("data_dir_prefix")
    postfix = data_loader.get("data_dir_postfix")
    return prefix, postfix


def near_border(center: Tuple[float, float], width: int, height: int, margin: float) -> bool:
    x, y = center
    return (
        x <= margin
        or y <= margin
        or x >= (width - margin)
        or y >= (height - margin)
    )


def angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> Optional[float]:
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    denom = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])
    if denom <= 0:
        return None
    cos_val = max(-1.0, min(1.0, dot / denom))
    return math.degrees(math.acos(cos_val))


def resolve_id(track_id: int, remap: Dict[int, int]) -> int:
    while track_id in remap and remap[track_id] != track_id:
        track_id = remap[track_id]
    return track_id


def load_records(file_path: str) -> Tuple[List[Record], Dict[int, Tracklet]]:
    records: List[Record] = []
    tracklets: Dict[int, Tracklet] = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 6:
                continue
            try:
                frame = int(float(fields[0]))
                track_id = int(float(fields[1]))
                x = float(fields[2])
                y = float(fields[3])
                w = float(fields[4])
                h = float(fields[5])
            except ValueError:
                continue

            record = Record(
                frame=frame,
                track_id=track_id,
                x=x,
                y=y,
                w=w,
                h=h,
                fields=fields,
            )
            records.append(record)

            center = (x + w / 2.0, y + h / 2.0)
            size = (w, h)
            tracklet = tracklets.setdefault(track_id, Tracklet(track_id))
            tracklet.add(frame=frame, center=center, size=size)

    for trk in tracklets.values():
        trk.sort_by_frame()

    return records, tracklets


def stitch_tracklets(
    tracklets: Dict[int, Tracklet],
    width: int,
    height: int,
    border_margin: float,
    max_gap: int,
    max_distance: float,
    max_angle_deg: float,
    max_speed_ratio: float,
    min_speed: float,
    velocity_window: int,
) -> Tuple[Dict[int, int], int]:
    remap: Dict[int, int] = {}
    merges = 0
    tracklet_ids = sorted(tracklets.keys(), key=lambda tid: tracklets[tid].start_frame)

    for new_id in tracklet_ids:
        if new_id not in tracklets:
            continue
        new_trk = tracklets[new_id]
        if near_border(new_trk.start_center, width, height, border_margin):
            continue

        v_new = new_trk.velocity(window=velocity_window, at_end=False)
        if v_new is None:
            continue
        speed_new = math.hypot(v_new[0], v_new[1])

        best_old_id = None
        best_dist = None

        for old_id, old_trk in list(tracklets.items()):
            if old_id == new_id:
                continue
            if old_trk.end_frame >= new_trk.start_frame:
                continue
            gap = new_trk.start_frame - old_trk.end_frame
            if gap < 1 or gap > max_gap:
                continue
            if near_border(old_trk.end_center, width, height, border_margin):
                continue

            v_old = old_trk.velocity(window=velocity_window, at_end=True)
            if v_old is None:
                continue
            speed_old = math.hypot(v_old[0], v_old[1])

            pred_x = old_trk.end_center[0] + v_old[0] * gap
            pred_y = old_trk.end_center[1] + v_old[1] * gap
            dist = math.hypot(pred_x - new_trk.start_center[0], pred_y - new_trk.start_center[1])
            if dist > max_distance:
                continue

            if speed_old >= min_speed and speed_new >= min_speed:
                angle = angle_between(v_old, v_new)
                if angle is None or angle > max_angle_deg:
                    continue
                if max_speed_ratio > 0:
                    ratio = max(speed_old, speed_new) / max(1e-6, min(speed_old, speed_new))
                    if ratio > max_speed_ratio:
                        continue

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_old_id = old_id

        if best_old_id is not None:
            remap[new_id] = best_old_id
            tracklets[best_old_id].merge_from(new_trk)
            tracklets.pop(new_id, None)
            merges += 1

    return remap, merges


def write_records(file_path: str, records: List[Record], remap: Dict[int, int]) -> None:
    if not remap:
        return
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w") as f:
        for record in records:
            new_id = resolve_id(record.track_id, remap)
            if new_id != record.track_id:
                record.track_id = int(new_id)
                record.fields[1] = str(int(new_id))
            f.write(",".join(record.fields) + "\n")
    os.replace(tmp_path, file_path)


def iter_mot_files(input_root: str, sequence: Optional[str] = None) -> List[str]:
    mot_files: List[str] = []
    for root, _, files in os.walk(input_root):
        for name in files:
            if name.endswith("_thermal.txt"):
                file_path = os.path.join(root, name)
                thermal_dir = os.path.dirname(file_path)
                seq_dir = os.path.dirname(thermal_dir)
                seq_name = os.path.basename(seq_dir)

                # Only accept canonical per-sequence tracker outputs:
                # <input-root>/<seq>/thermal/<seq>_thermal.txt
                if os.path.basename(thermal_dir) != "thermal":
                    continue
                if name != f"{seq_name}_thermal.txt":
                    continue
                if sequence is not None and seq_name != sequence:
                    continue
                mot_files.append(file_path)
    return sorted(set(mot_files))


def filename_sort_key(name: str):
    stem = os.path.splitext(os.path.basename(name))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def create_unique_color_uchar(tag: int, hue_step: float = 0.41) -> Tuple[int, int, int]:
    h = (tag * hue_step) % 1
    v = 1.0 - (int(tag * hue_step) % 4) / 5.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, v)
    return (int(255 * r), int(255 * g), int(255 * b))


def draw_box(img, bbox: Tuple[float, float, float, float], color, label: Optional[str]) -> None:
    x1, y1, x2, y2 = bbox
    tl = max(1, round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
    c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)


def draw_tracks(
    records: List[Record],
    image_dir: str,
    output_dir: str,
) -> None:
    images = [name for name in os.listdir(image_dir) if not name.startswith(".")]
    if not images:
        return
    images = sorted(images, key=filename_sort_key)
    os.makedirs(output_dir, exist_ok=True)

    frame_to_records: Dict[int, List[Record]] = {}
    for record in records:
        frame_to_records.setdefault(record.frame, []).append(record)

    for frame_index, image_name in enumerate(images, start=1):
        img_path = os.path.join(image_dir, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        for record in frame_to_records.get(frame_index, []):
            x1 = record.x
            y1 = record.y
            x2 = x1 + record.w
            y2 = y1 + record.h
            color = create_unique_color_uchar(int(record.track_id) % 1000)
            draw_box(img, (x1, y1, x2, y2), color, str(int(record.track_id)))
        out_path = os.path.join(output_dir, image_name)
        cv2.imwrite(out_path, img)


def main() -> None:
    args = parse_args()

    width, height = load_shape_from_config(args.config)
    if args.frame_width is not None:
        width = args.frame_width
    if args.frame_height is not None:
        height = args.frame_height

    if width is None or height is None:
        raise SystemExit("Frame width/height not set. Provide --config or --frame-width/--frame-height.")

    mot_files = iter_mot_files(args.input_root, sequence=args.sequence)
    if not mot_files:
        if args.sequence is not None:
            print(f"No MOT files found for sequence '{args.sequence}' under: {args.input_root}")
        else:
            print(f"No MOT files found under: {args.input_root}")
        return

    image_root, image_postfix = load_data_paths_from_config(args.config)
    if args.image_root is not None:
        image_root = args.image_root
    if args.image_postfix is not None:
        image_postfix = args.image_postfix

    if args.draw and (not image_root or image_postfix is None):
        raise SystemExit("Image paths not set. Provide --config or --image-root/--image-postfix.")

    total_merges = 0
    for file_path in mot_files:
        records, tracklets = load_records(file_path)
        if not records:
            continue
        remap, merges = stitch_tracklets(
            tracklets=tracklets,
            width=width,
            height=height,
            border_margin=args.border_margin,
            max_gap=args.max_gap,
            max_distance=args.max_distance,
            max_angle_deg=args.max_angle_deg,
            max_speed_ratio=args.max_speed_ratio,
            min_speed=args.min_speed,
            velocity_window=args.velocity_window,
        )
        if merges > 0:
            write_records(file_path, records, remap)
        total_merges += merges
        print(f"{os.path.basename(file_path)}: stitched {merges} tracklets")

        if args.draw:
            seq_root = os.path.dirname(os.path.dirname(file_path))
            seq_name = os.path.basename(seq_root)
            image_dir = os.path.join(image_root, seq_name, image_postfix)
            output_dir = os.path.join(os.path.dirname(file_path), args.draw_output_name)
            if os.path.isdir(image_dir):
                draw_tracks(records, image_dir, output_dir)
            else:
                print(f"Warning: missing image dir for {seq_name}: {image_dir}")

    print(f"Done. Total stitched tracklets: {total_merges}")


if __name__ == "__main__":
    main()
