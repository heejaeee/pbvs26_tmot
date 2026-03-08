#!/usr/bin/env python3
"""Evaluate TMOT SORT-style submissions against val COCO annotations.

This script is dependency-light and uses only the Python standard library.
It expects:
- ground truth in:   <gt-val-dir>/seqXX/annotations.json
- predictions in:    <tracker-dir>/seqXX_thermal.txt

Metrics reported:
- MOTA
- IDF1
- FP / FN / IDSW
- matches
- mean IoU over matched boxes (reported as MOTP-IoU)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


BBox = Tuple[float, float, float, float]
FrameDet = Tuple[int, BBox]


@dataclass
class Row:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class Metrics:
    gt_total: int = 0
    pred_total: int = 0
    fp: int = 0
    fn: int = 0
    idsw: int = 0
    matches: int = 0
    idtp: int = 0
    idfp: int = 0
    idfn: int = 0
    match_iou_sum: float = 0.0

    @property
    def mota(self) -> float:
        if self.gt_total <= 0:
            return 0.0
        return 1.0 - float(self.fp + self.fn + self.idsw) / float(self.gt_total)

    @property
    def idf1(self) -> float:
        denom = 2 * self.idtp + self.idfp + self.idfn
        if denom <= 0:
            return 0.0
        return float(2 * self.idtp) / float(denom)

    @property
    def motp_iou(self) -> float:
        if self.matches <= 0:
            return 0.0
        return self.match_iou_sum / float(self.matches)

    def add(self, other: "Metrics") -> "Metrics":
        self.gt_total += other.gt_total
        self.pred_total += other.pred_total
        self.fp += other.fp
        self.fn += other.fn
        self.idsw += other.idsw
        self.matches += other.matches
        self.idtp += other.idtp
        self.idfp += other.idfp
        self.idfn += other.idfn
        self.match_iou_sum += other.match_iou_sum
        return self

    def to_dict(self) -> Dict[str, object]:
        return {
            "gt_total": self.gt_total,
            "pred_total": self.pred_total,
            "fp": self.fp,
            "fn": self.fn,
            "idsw": self.idsw,
            "matches": self.matches,
            "idtp": self.idtp,
            "idfp": self.idfp,
            "idfn": self.idfn,
            "match_iou_sum": self.match_iou_sum,
            "mota": self.mota,
            "idf1": self.idf1,
            "motp_iou": self.motp_iou,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SORT-style TMOT submission against val annotations."
    )
    parser.add_argument(
        "--tracker-dir",
        required=True,
        help="Directory containing seqXX_thermal.txt result files.",
    )
    parser.add_argument(
        "--gt-val-dir",
        required=True,
        help="Directory containing seqXX/annotations.json ground truth files.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for frame matching (default: 0.5).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full report as JSON.",
    )
    return parser.parse_args()


def filename_sort_key(name: str) -> Tuple[int, object]:
    stem = os.path.splitext(os.path.basename(name))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def sequence_sort_key(name: str) -> Tuple[int, object]:
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return (0, int(digits))
    return (1, name)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def hungarian_min_cost(cost: List[List[float]]) -> List[int]:
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0]) if cost[0] else 0
    if m == 0:
        return [-1] * n
    if n > m:
        raise ValueError("hungarian_min_cost requires n <= m")

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    out = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            out[p[j] - 1] = j - 1
    return out


def max_weight_assignment(weight: List[List[float]]) -> Tuple[List[int], float]:
    n = len(weight)
    m = len(weight[0]) if n > 0 else 0
    if n == 0 or m == 0:
        return [], 0.0

    size = max(n, m)
    padded = [[0.0] * size for _ in range(size)]
    max_w = 0.0
    for i in range(n):
        for j in range(m):
            w = float(weight[i][j])
            padded[i][j] = w
            if w > max_w:
                max_w = w

    cost = [[max_w - padded[i][j] for j in range(size)] for i in range(size)]
    assign = hungarian_min_cost(cost)

    matched_weight = 0.0
    for i in range(n):
        j = assign[i] if i < len(assign) else -1
        if 0 <= j < m:
            matched_weight += float(weight[i][j])
    return assign[:n], matched_weight


def parse_rows(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = [part.strip() for part in line.split(",")]
            if len(fields) < 6:
                continue
            try:
                rows.append(
                    Row(
                        frame=int(float(fields[0])),
                        track_id=int(float(fields[1])),
                        x=float(fields[2]),
                        y=float(fields[3]),
                        w=float(fields[4]),
                        h=float(fields[5]),
                    )
                )
            except ValueError:
                continue
    return rows


def build_pred_by_frame(rows: Iterable[Row]) -> Dict[int, List[FrameDet]]:
    pred_by_frame: Dict[int, List[FrameDet]] = {}
    for row in rows:
        pred_by_frame.setdefault(row.frame, []).append(
            (row.track_id, (row.x, row.y, row.w, row.h))
        )
    return pred_by_frame


def load_gt_annotations(gt_json_path: str) -> Tuple[int, Dict[int, List[FrameDet]]]:
    with open(gt_json_path, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    images_sorted = sorted(images, key=lambda image: filename_sort_key(image.get("file_name", "")))
    image_id_to_frame = {image["id"]: idx + 1 for idx, image in enumerate(images_sorted)}
    num_frames = len(images_sorted)

    gt_by_frame: Dict[int, List[FrameDet]] = {}
    for ann in data.get("annotations", []):
        if int(ann.get("category_id", 1)) != 1:
            continue
        image_id = ann.get("image_id")
        if image_id not in image_id_to_frame:
            continue
        bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if len(bbox) != 4:
            continue
        frame = image_id_to_frame[image_id]
        track_id = int(ann.get("track_id", ann.get("id", -1)))
        gt_by_frame.setdefault(frame, []).append(
            (
                track_id,
                (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            )
        )
    return num_frames, gt_by_frame


def discover_gt_sequences(gt_val_dir: str) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    for name in os.listdir(gt_val_dir):
        seq_dir = os.path.join(gt_val_dir, name)
        if not os.path.isdir(seq_dir):
            continue
        flat_json = os.path.join(seq_dir, "annotations.json")
        coco_json = os.path.join(seq_dir, "thermal", "COCO", "annotations.json")
        if os.path.isfile(flat_json):
            sequences[name] = flat_json
        elif os.path.isfile(coco_json):
            sequences[name] = coco_json
    return dict(sorted(sequences.items(), key=lambda kv: sequence_sort_key(kv[0])))


def discover_tracker_files(tracker_dir: str) -> Dict[str, str]:
    tracker_files: Dict[str, str] = {}
    for name in os.listdir(tracker_dir):
        if not name.endswith(".txt"):
            continue
        seq = os.path.splitext(name)[0]
        if seq.endswith("_thermal"):
            seq = seq[: -len("_thermal")]
        tracker_files[seq] = os.path.join(tracker_dir, name)
    return dict(sorted(tracker_files.items(), key=lambda kv: sequence_sort_key(kv[0])))


def match_frame(
    preds: Sequence[FrameDet],
    gts: Sequence[FrameDet],
    iou_thresh: float,
) -> List[Tuple[int, int, float]]:
    if not preds or not gts:
        return []

    if len(preds) <= len(gts):
        rows = preds
        cols = gts
        row_is_pred = True
    else:
        rows = gts
        cols = preds
        row_is_pred = False

    weight: List[List[float]] = []
    for _, row_box in rows:
        weight_row: List[float] = []
        for _, col_box in cols:
            ov = bbox_iou(row_box, col_box)
            weight_row.append(ov if ov >= iou_thresh else 0.0)
        weight.append(weight_row)

    assign, _ = max_weight_assignment(weight)
    matches: List[Tuple[int, int, float]] = []
    for row_idx, col_idx in enumerate(assign):
        if not (0 <= col_idx < len(cols)):
            continue
        ov = weight[row_idx][col_idx]
        if ov < iou_thresh:
            continue
        if row_is_pred:
            matches.append((row_idx, col_idx, ov))
        else:
            matches.append((col_idx, row_idx, ov))
    return matches


def evaluate_sequence(
    pred_by_frame: Dict[int, List[FrameDet]],
    gt_by_frame: Dict[int, List[FrameDet]],
    num_frames: int,
    iou_thresh: float,
) -> Metrics:
    metrics = Metrics()
    last_pred_for_gt: Dict[int, int] = {}
    pair_counts: Dict[Tuple[int, int], int] = {}
    gt_ids: Set[int] = set()
    pred_ids: Set[int] = set()

    for frame in range(1, num_frames + 1):
        preds = pred_by_frame.get(frame, [])
        gts = gt_by_frame.get(frame, [])

        metrics.gt_total += len(gts)
        metrics.pred_total += len(preds)
        for gt_id, _ in gts:
            gt_ids.add(gt_id)
        for pred_id, _ in preds:
            pred_ids.add(pred_id)

        matches = match_frame(preds, gts, iou_thresh)
        used_pred = set()
        used_gt = set()
        for pred_idx, gt_idx, ov in matches:
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
            pred_id = preds[pred_idx][0]
            gt_id = gts[gt_idx][0]

            prev_pred = last_pred_for_gt.get(gt_id)
            if prev_pred is not None and prev_pred != pred_id:
                metrics.idsw += 1
            last_pred_for_gt[gt_id] = pred_id

            metrics.matches += 1
            metrics.match_iou_sum += ov
            pair_key = (gt_id, pred_id)
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        metrics.fp += len(preds) - len(used_pred)
        metrics.fn += len(gts) - len(used_gt)

    gt_list = sorted(gt_ids)
    pred_list = sorted(pred_ids)
    if gt_list and pred_list:
        gt_pos = {gt_id: idx for idx, gt_id in enumerate(gt_list)}
        pred_pos = {pred_id: idx for idx, pred_id in enumerate(pred_list)}
        weight = [[0.0 for _ in pred_list] for _ in gt_list]
        for (gt_id, pred_id), count in pair_counts.items():
            weight[gt_pos[gt_id]][pred_pos[pred_id]] = float(count)
        _, matched = max_weight_assignment(weight)
        metrics.idtp = int(round(matched))
    else:
        metrics.idtp = 0

    metrics.idfp = max(0, metrics.pred_total - metrics.idtp)
    metrics.idfn = max(0, metrics.gt_total - metrics.idtp)
    return metrics


def render_table(rows: List[Dict[str, object]]) -> str:
    headers = ["sequence", "GT", "Pred", "Match", "FP", "FN", "IDSW", "MOTA", "IDF1", "MOTP-IoU"]
    data_rows = []
    for row in rows:
        data_rows.append(
            [
                str(row["sequence"]),
                str(row["gt_total"]),
                str(row["pred_total"]),
                str(row["matches"]),
                str(row["fp"]),
                str(row["fn"]),
                str(row["idsw"]),
                f"{100.0 * float(row['mota']):.4f}",
                f"{100.0 * float(row['idf1']):.4f}",
                f"{100.0 * float(row['motp_iou']):.4f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in data_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    lines = []
    lines.append("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    lines.append("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in data_rows:
        lines.append("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    tracker_dir = os.path.abspath(args.tracker_dir)
    gt_val_dir = os.path.abspath(args.gt_val_dir)

    if not os.path.isdir(tracker_dir):
        raise FileNotFoundError(f"Missing tracker dir: {tracker_dir}")
    if not os.path.isdir(gt_val_dir):
        raise FileNotFoundError(f"Missing GT val dir: {gt_val_dir}")

    gt_sequences = discover_gt_sequences(gt_val_dir)
    tracker_files = discover_tracker_files(tracker_dir)
    if not gt_sequences:
        raise FileNotFoundError(f"No seq*/annotations.json files found in: {gt_val_dir}")
    if not tracker_files:
        raise FileNotFoundError(f"No tracker .txt files found in: {tracker_dir}")

    rows: List[Dict[str, object]] = []
    overall = Metrics()

    gt_names = set(gt_sequences)
    pred_names = set(tracker_files)
    missing_predictions = sorted(gt_names - pred_names, key=sequence_sort_key)
    extra_predictions = sorted(pred_names - gt_names, key=sequence_sort_key)

    for seq_name in sorted(gt_sequences, key=sequence_sort_key):
        gt_json_path = gt_sequences[seq_name]
        pred_path = tracker_files.get(seq_name)

        num_frames, gt_by_frame = load_gt_annotations(gt_json_path)
        pred_rows = parse_rows(pred_path) if pred_path else []
        pred_by_frame = build_pred_by_frame(pred_rows)
        seq_metrics = evaluate_sequence(pred_by_frame, gt_by_frame, num_frames, args.iou_thresh)
        overall.add(seq_metrics)

        rows.append(
            {
                "sequence": seq_name,
                **seq_metrics.to_dict(),
                "num_frames": num_frames,
                "gt_json_path": gt_json_path,
                "pred_path": pred_path,
            }
        )

    overall_row = {"sequence": "OVERALL", **overall.to_dict()}
    rows.append(overall_row)

    print(f"Tracker dir: {tracker_dir}")
    print(f"GT val dir:  {gt_val_dir}")
    print(f"IoU thr:     {args.iou_thresh:.2f}")
    if missing_predictions:
        print(f"Missing prediction files for: {', '.join(missing_predictions)}")
    if extra_predictions:
        print(f"Extra prediction files ignored: {', '.join(extra_predictions)}")
    print()
    print(render_table(rows))

    if args.output_json:
        report = {
            "tracker_dir": tracker_dir,
            "gt_val_dir": gt_val_dir,
            "iou_thresh": args.iou_thresh,
            "missing_predictions": missing_predictions,
            "extra_predictions": extra_predictions,
            "per_sequence": rows[:-1],
            "overall": overall_row,
        }
        output_json = os.path.abspath(args.output_json)
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        print()
        print(f"Saved JSON report to: {output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
