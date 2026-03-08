#!/usr/bin/env python3
import argparse
import datetime as dt
import glob
import json
import os
import sys
from collections import OrderedDict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MOTA/MOTP/IDF1 on TMOT train split by converting COCO GT to MOTChallenge."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="tmot_dataset_challenge/tmot_dataset_challenge",
        help="Root of tmot_dataset_challenge containing annotations/ and images/.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to evaluate (default: train).",
    )
    parser.add_argument(
        "--tracker-dir",
        required=True,
        help="Directory that contains tracker result .txt files.",
    )
    parser.add_argument(
        "--keep-reversed",
        action="store_true",
        help="Keep reversed frame indices from tracker outputs (no un-reverse step).",
    )
    parser.add_argument(
        "--work-dir",
        default="data/tmot_dataset/eval_mot",
        help="Directory to store generated MOTChallenge files.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5).",
    )
    return parser.parse_args()


def filename_sort_key(name: str):
    stem = os.path.splitext(os.path.basename(name))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def convert_coco_to_mot(gt_json_path: str, out_txt_path: str) -> int:
    with open(gt_json_path, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    images_sorted = sorted(images, key=lambda img: filename_sort_key(img.get("file_name", "")))
    image_id_to_frame = {img["id"]: idx + 1 for idx, img in enumerate(images_sorted)}

    rows = []
    for ann in data.get("annotations", []):
        if ann.get("category_id", 1) != 1:
            continue
        frame_id = image_id_to_frame.get(ann.get("image_id"))
        if frame_id is None:
            continue
        track_id = ann.get("track_id", ann.get("id"))
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        rows.append((frame_id, track_id, x, y, w, h, 1, 1, 1, 1))

    rows.sort(key=lambda r: (r[0], r[1]))
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w") as f:
        for row in rows:
            f.write(
                "{},{},{},{},{},{},{},{},{},{}\n".format(
                    int(row[0]),
                    int(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    int(row[6]),
                    int(row[7]),
                    int(row[8]),
                    int(row[9]),
                )
            )
    return len(rows)


def build_gt(dataset_root: str, split: str, gt_root: str) -> int:
    ann_root = os.path.join(dataset_root, "annotations", split)
    if not os.path.isdir(ann_root):
        raise FileNotFoundError(f"Missing annotations dir: {ann_root}")

    seq_dirs = sorted(
        d for d in os.listdir(ann_root) if os.path.isdir(os.path.join(ann_root, d))
    )
    total = 0
    for seq in seq_dirs:
        gt_json = os.path.join(ann_root, seq, "thermal", "COCO", "annotations.json")
        if not os.path.isfile(gt_json):
            continue
        gt_out = os.path.join(gt_root, seq, "gt", "gt.txt")
        total += convert_coco_to_mot(gt_json, gt_out)
    return total


def unreverse_tracker_rows(rows):
    max_frame = 0
    max_id = 0
    for parts in rows:
        if len(parts) < 2:
            continue
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        max_frame = max(max_frame, frame)
        max_id = max(max_id, track_id)

    for parts in rows:
        if len(parts) < 2:
            continue
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        parts[0] = str(max_frame - frame + 1)
        parts[1] = str(max_id - track_id + 1)


def prepare_trackers(tracker_dir: str, tracker_root: str, keep_reversed: bool) -> int:
    if not os.path.isdir(tracker_dir):
        raise FileNotFoundError(f"Missing tracker dir: {tracker_dir}")

    files = []
    for root, _, filenames in os.walk(tracker_dir):
        for name in filenames:
            if name.endswith(".txt") and not name.startswith("eval"):
                files.append(os.path.join(root, name))

    if not files:
        raise FileNotFoundError(f"No tracker .txt files found in: {tracker_dir}")

    os.makedirs(tracker_root, exist_ok=True)
    written = 0
    for src in files:
        base = os.path.splitext(os.path.basename(src))[0]
        if base.endswith("_thermal"):
            seq_name = base[: -len("_thermal")]
        else:
            seq_name = base
        dst = os.path.join(tracker_root, f"{seq_name}.txt")
        if os.path.exists(dst):
            continue
        with open(src, "r") as f_src:
            rows = [line.strip().split(",") for line in f_src if line.strip()]
        if not keep_reversed:
            unreverse_tracker_rows(rows)
        with open(dst, "w") as f_dst:
            for parts in rows:
                f_dst.write(",".join(parts) + "\n")
        written += 1
    return written


def compare_dataframes(gts, ts, iou_thresh: float):
    import motmetrics as mm

    accs = []
    names = []
    for name, tsacc in ts.items():
        if name in gts:
            accs.append(mm.utils.compare_to_groundtruth(gts[name], tsacc, "iou", distth=iou_thresh))
            names.append(name)
    return accs, names


def run_eval(gt_root: str, tracker_root: str, iou_thresh: float):
    import motmetrics as mm

    gtfiles = glob.glob(os.path.join(gt_root, "*/gt/gt.txt"))
    tsfiles = glob.glob(os.path.join(tracker_root, "*.txt"))

    gts = OrderedDict(
        [
            (os.path.basename(os.path.dirname(os.path.dirname(f))), mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1))
            for f in gtfiles
        ]
    )
    ts = OrderedDict(
        [(os.path.splitext(os.path.basename(f))[0], mm.io.loadtxt(f, fmt="mot15-2D")) for f in tsfiles]
    )

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gts, ts, iou_thresh)

    metrics = ["mota", "motp", "idf1"]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))


def main() -> int:
    args = parse_args()
    dataset_root = os.path.abspath(args.dataset_root)
    tracker_dir = os.path.abspath(args.tracker_dir)

    run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(os.path.abspath(args.work_dir), f"run_{run_tag}")
    gt_root = os.path.join(work_dir, "gt")
    tracker_root = os.path.join(work_dir, "trackers")

    print(f"Dataset root: {dataset_root}")
    print(f"Tracker dir:  {tracker_dir}")
    print(f"Work dir:     {work_dir}")

    total_gt = build_gt(dataset_root, args.split, gt_root)
    total_ts = prepare_trackers(tracker_dir, tracker_root, args.keep_reversed)

    if total_gt == 0:
        print("No ground-truth annotations found. Check your dataset root and split.")
        return 1
    if total_ts == 0:
        print("No tracker files found. Check your tracker dir.")
        return 1

    run_eval(gt_root, tracker_root, args.iou_thresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
