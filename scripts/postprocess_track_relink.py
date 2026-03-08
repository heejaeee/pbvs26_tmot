#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


LEARNED_SCORE_THRESHOLD = 0.95
LEARNED_EDGE_START_MARGIN = 60.0
LEARNED_FEATURE_MEANS = [
    0.0,
    0.5440677966101695,
    0.2675149850756249,
    0.22737288135593223,
    0.07440677966101694,
    0.26593220338983053,
    0.5044067796610169,
    0.27356943945939216,
    0.4535762385112652,
    0.7227175653037892,
    1.1602542372881355,
    1.1787288135593221,
    0.1694915254237288,
    0.2711864406779661,
    -0.248158974089037,
    0.31571812484251655,
    -0.5638770989315535,
    -0.01694915254237288,
    0.8491105604257948,
    0.16648408852837807,
    0.22737288135593223,
    0.2848637639804861,
    0.39979310162209125,
    1.2816439662131665,
    0.6822785944171526,
    0.8687763817402185,
    -0.05084745762711865,
    0.7127118644067797,
    1.0250646257875324,
    0.4962950563843088,
    0.6250063260091684,
    0.0,
    0.6842009685230024,
    0.8731635121829247,
    0.5308733467085861,
    0.5334361579405306,
    -0.05084745762711865,
    0.8050242130750606,
    0.8078093906503948,
    0.5463174435758564,
    0.5087563220489051,
    0.01694915254237288,
    0.8882905120616985,
    0.7320516430719443,
    0.5454525923938062,
    0.47188758367057493,
    0.01694915254237288,
    0.8648933179692219,
]
LEARNED_FEATURE_STDS = [
    1.0,
    0.30777110264414204,
    0.1729390085287399,
    0.1748062328566587,
    0.11696320143364987,
    0.2824786495977402,
    0.386622705294573,
    0.2649371896285382,
    0.34400520768664916,
    1.3065481545893316,
    0.9326846956017658,
    0.9025930034893945,
    0.37518648510472297,
    0.44457310334250855,
    1.0056452684139745,
    0.8324705411436939,
    1.6787372683242268,
    0.9998573527967882,
    1.5540679539823592,
    0.23200727089128437,
    0.1748062328566587,
    0.25905392281371303,
    0.7948799410674372,
    2.5455276458802456,
    1.207933182238842,
    1.5650302590176137,
    0.7230765595831031,
    1.5483624417396267,
    1.8017554363691781,
    0.6743327379801324,
    1.0198890510741647,
    0.8829851544156683,
    1.32718273748583,
    1.5090899045573887,
    0.8656866004750128,
    0.9734626001851923,
    0.9099037478962406,
    1.73605900379616,
    1.3105648871827305,
    0.9955845423727998,
    0.9860132830744281,
    0.9998573527967882,
    1.9225800986569699,
    1.120553712313925,
    0.985214894549658,
    0.9234210418788252,
    0.9998573527967882,
    1.839555717995651,
]
LEARNED_FEATURE_WEIGHTS = [
    -5.515453183467339,
    -2.06108581903843,
    -1.4289950905803293,
    -0.9779570718869017,
    -0.9183835769123564,
    0.05021277011413361,
    -1.6200938768223792,
    0.294441489232011,
    -0.09880572355538107,
    0.30288617438476667,
    0.8859535958349384,
    1.1462531100769433,
    -2.091661236862404,
    -0.383200141351045,
    -1.5626223028266784,
    2.3439174030881547,
    -2.0984140762045382,
    0.5785090999174101,
    -0.22528358334312026,
    0.6171640764181865,
    -0.9779570718869017,
    0.1243722181021984,
    -0.9588904929535855,
    -0.48135841644200544,
    0.6445948158838992,
    0.4588556316098174,
    -1.1025075975767158,
    -1.205210985500397,
    0.044547082036925874,
    -0.08792923764439343,
    0.9145908913773007,
    1.6628452032393937,
    -0.10943235379445293,
    0.052501626093553834,
    -0.7261819313718042,
    1.0434069930255938,
    -0.08854252633794112,
    0.2829093001035641,
    -0.5950184696728538,
    -1.3323058798266534,
    -0.16781737676224925,
    2.9525806037449596,
    -0.4025018058296725,
    -0.8129313990247498,
    -1.830128812567763,
    -0.34504972111833904,
    -0.42647760507110727,
    -0.5145475689307628,
]


@dataclass
class Record:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    fields: List[str]

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + 0.5 * self.w, self.y + 0.5 * self.h)


class Tracklet:
    def __init__(self, track_id: int) -> None:
        self.track_id = int(track_id)
        self.records: List[Record] = []

    def add(self, record: Record) -> None:
        self.records.append(record)

    def finalize(self) -> None:
        self.records.sort(key=lambda item: item.frame)

    def extend_from(self, other: "Tracklet") -> None:
        self.records.extend(other.records)
        self.records.sort(key=lambda item: item.frame)

    @property
    def length(self) -> int:
        return len(self.records)

    @property
    def start_frame(self) -> int:
        return self.records[0].frame

    @property
    def end_frame(self) -> int:
        return self.records[-1].frame

    @property
    def start_record(self) -> Record:
        return self.records[0]

    @property
    def end_record(self) -> Record:
        return self.records[-1]

    def velocity(self, window: int, at_end: bool) -> Optional[Tuple[float, float]]:
        if len(self.records) < 2:
            return None
        window = max(2, int(window))
        seq = self.records[-window:] if at_end else self.records[:window]
        if len(seq) < 2:
            return None

        vxs: List[float] = []
        vys: List[float] = []
        for idx in range(1, len(seq)):
            prev = seq[idx - 1]
            cur = seq[idx]
            dt = cur.frame - prev.frame
            if dt <= 0:
                continue
            prev_center = prev.center
            cur_center = cur.center
            vxs.append((cur_center[0] - prev_center[0]) / dt)
            vys.append((cur_center[1] - prev_center[1]) / dt)
        if not vxs:
            return None
        return (sum(vxs) / len(vxs), sum(vys) / len(vys))


@dataclass(frozen=True)
class RelinkConfig:
    border_margin: float = 25.0
    max_gap: int = 60
    max_distance: float = 120.0
    min_new_frames: int = 2
    min_old_frames: int = 2
    require_old_not_edge: bool = False
    score_threshold: float = LEARNED_SCORE_THRESHOLD
    max_rounds: int = 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "border_margin": self.border_margin,
            "max_gap": self.max_gap,
            "max_distance": self.max_distance,
            "min_new_frames": self.min_new_frames,
            "min_old_frames": self.min_old_frames,
            "require_old_not_edge": self.require_old_not_edge,
            "score_threshold": self.score_threshold,
            "max_rounds": self.max_rounds,
        }


@dataclass(frozen=True)
class Candidate:
    old_id: int
    new_id: int
    mode_rank: int
    cost: float
    gap: int
    old_end_frame: int
    new_start_frame: int

    def sort_key(self) -> Tuple[int, float, int, int, int]:
        return (
            self.mode_rank,
            self.cost,
            self.gap,
            self.new_start_frame,
            self.old_end_frame,
        )


@dataclass
class TrackerFile:
    sequence: str
    src_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relink fragmented MOT tracklets after stitching."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help=(
            "Tracker root. Supports either canonical files under "
            "<root>/<seq>/thermal/<seq>_thermal.txt or flat submission folders."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file used to infer frame size from data.shape or data_writer.shape.",
    )
    parser.add_argument("--frame-width", type=int, default=None, help="Override frame width.")
    parser.add_argument("--frame-height", type=int, default=None, help="Override frame height.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional flat output folder for relinked seq*_thermal.txt files.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the discovered tracker files in place.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Process only one sequence name, such as seq22.",
    )
    parser.add_argument("--border-margin", type=float, default=25.0)
    parser.add_argument("--max-gap", type=int, default=60)
    parser.add_argument("--max-distance", type=float, default=120.0)
    parser.add_argument("--min-new-frames", type=int, default=2)
    parser.add_argument("--min-old-frames", type=int, default=2)
    parser.add_argument(
        "--allow-old-edge",
        action="store_true",
        help="Allow candidate predecessor tracklets that disappear near the border.",
    )
    parser.add_argument("--score-threshold", type=float, default=LEARNED_SCORE_THRESHOLD)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument(
        "--report-json",
        default="track_relink_report.json",
        help="Report filename written to the output folder or input root.",
    )
    return parser.parse_args()


def load_shape_from_config(config_path: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not config_path or not os.path.isfile(config_path):
        return None, None
    with open(config_path, "r") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        return None, None
    for key in ("data", "data_writer"):
        section = cfg.get(key, {})
        shape = section.get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
    return None, None


def sequence_sort_key(name: str) -> Tuple[int, object]:
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return (0, int(digits))
    return (1, name)


def discover_tracker_files(input_root: str, sequence: Optional[str]) -> List[TrackerFile]:
    files: Dict[str, str] = {}

    if os.path.isdir(input_root):
        for name in os.listdir(input_root):
            if not name.endswith("_thermal.txt"):
                continue
            seq_name = os.path.splitext(name)[0]
            seq_name = seq_name[: -len("_thermal")] if seq_name.endswith("_thermal") else seq_name
            if sequence is not None and seq_name != sequence:
                continue
            files[seq_name] = os.path.join(input_root, name)

    for root, _, names in os.walk(input_root):
        for name in names:
            if not name.endswith("_thermal.txt"):
                continue
            file_path = os.path.join(root, name)
            thermal_dir = os.path.dirname(file_path)
            if os.path.basename(thermal_dir) != "thermal":
                continue
            seq_dir = os.path.dirname(thermal_dir)
            seq_name = os.path.basename(seq_dir)
            if name != f"{seq_name}_thermal.txt":
                continue
            if sequence is not None and seq_name != sequence:
                continue
            files[seq_name] = file_path

    return [
        TrackerFile(sequence=seq_name, src_path=path)
        for seq_name, path in sorted(files.items(), key=lambda item: sequence_sort_key(item[0]))
    ]


def parse_records(path: str) -> List[Record]:
    records: List[Record] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fields = [part.strip() for part in line.split(",")]
            if len(fields) < 6:
                continue
            try:
                record = Record(
                    frame=int(float(fields[0])),
                    track_id=int(float(fields[1])),
                    x=float(fields[2]),
                    y=float(fields[3]),
                    w=float(fields[4]),
                    h=float(fields[5]),
                    fields=fields,
                )
            except ValueError:
                continue
            records.append(record)
    return records


def build_tracklets(records: Iterable[Record]) -> Dict[int, Tracklet]:
    tracklets: Dict[int, Tracklet] = {}
    for record in records:
        tracklets.setdefault(record.track_id, Tracklet(record.track_id)).add(record)
    for tracklet in tracklets.values():
        tracklet.finalize()
    return tracklets


def resolve_id(track_id: int, remap: Dict[int, int]) -> int:
    current = int(track_id)
    while current in remap and remap[current] != current:
        current = int(remap[current])
    return current


def euclidean(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def sigmoid(x: float) -> float:
    x = max(min(x, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-x))


def near_edge(record: Record, width: int, height: int, margin: float) -> bool:
    return (
        record.x <= margin
        or record.y <= margin
        or (record.x + record.w) >= (width - margin)
        or (record.y + record.h) >= (height - margin)
    )


def border_dist_x(record: Record, width: int) -> float:
    center_x, _ = record.center
    return min(center_x, float(width) - center_x)


def size_ratio(record_a: Record, record_b: Record) -> float:
    aw = max(record_a.w, 1e-6)
    ah = max(record_a.h, 1e-6)
    bw = max(record_b.w, 1e-6)
    bh = max(record_b.h, 1e-6)
    return max(max(aw / bw, bw / aw), max(ah / bh, bh / ah))


def better_candidate(left: Candidate, right: Optional[Candidate]) -> bool:
    return right is None or left.sort_key() < right.sort_key()


def overall_velocity(tracklet: Tracklet) -> Tuple[float, float]:
    if len(tracklet.records) < 2:
        return (0.0, 0.0)
    start = tracklet.start_record
    end = tracklet.end_record
    dt = max(1, end.frame - start.frame)
    start_center = start.center
    end_center = end.center
    return (
        (end_center[0] - start_center[0]) / float(dt),
        (end_center[1] - start_center[1]) / float(dt),
    )


def signed_unit(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def learned_link_score(
    old_tracklet: Tracklet,
    new_tracklet: Tracklet,
    width: int,
    height: int,
) -> float:
    old_end = old_tracklet.end_record
    new_start = new_tracklet.start_record
    old_end_center = old_end.center
    new_start_center = new_start.center
    gap = new_start.frame - old_end.frame
    if gap < 1:
        return 0.0

    old_len = max(1, old_tracklet.length)
    new_len = max(1, new_tracklet.length)
    size_gate = size_ratio(old_end, new_start)
    min_len = float(min(old_len, new_len))
    max_len = float(max(old_len, new_len))
    overall_old = overall_velocity(old_tracklet)
    overall_new = overall_velocity(new_tracklet)

    features = [
        1.0,
        float(gap) / 50.0,
        euclidean(old_end_center, new_start_center) / 200.0,
        abs(old_end_center[0] - new_start_center[0]) / 200.0,
        abs(old_end_center[1] - new_start_center[1]) / 200.0,
        float(old_len) / 200.0,
        float(new_len) / 200.0,
        min_len / max_len,
        math.log(max(size_gate, 1e-6)) / 2.0,
        size_gate / 5.0,
        border_dist_x(old_end, width) / 200.0,
        border_dist_x(new_start, width) / 200.0,
        float(near_edge(old_end, width, height, 25.0)),
        float(near_edge(new_start, width, height, LEARNED_EDGE_START_MARGIN)),
        overall_old[0] / 5.0,
        overall_new[0] / 5.0,
        (overall_old[0] - overall_new[0]) / 5.0,
        signed_unit(overall_old[0]) * signed_unit(overall_new[0]),
        abs(overall_old[0] - overall_new[0]) / 5.0,
        abs(overall_old[1] - overall_new[1]) / 5.0,
        abs(border_dist_x(old_end, width) - border_dist_x(new_start, width)) / 200.0,
        float(gap) * euclidean(old_end_center, new_start_center) / 5000.0,
        abs(old_end_center[0] - new_start_center[0]) / max(1.0, float(gap)) / 10.0,
    ]

    for window in (2, 3, 5, 10, 20):
        old_velocity = old_tracklet.velocity(window=min(window, old_len), at_end=True) or (0.0, 0.0)
        new_velocity = new_tracklet.velocity(window=min(window, new_len), at_end=False) or (0.0, 0.0)
        pred_old = (
            old_end_center[0] + old_velocity[0] * float(gap),
            old_end_center[1] + old_velocity[1] * float(gap),
        )
        pred_new = (
            new_start_center[0] - new_velocity[0] * float(gap),
            new_start_center[1] - new_velocity[1] * float(gap),
        )
        features.extend(
            [
                euclidean(pred_old, new_start_center) / 200.0,
                euclidean(pred_new, old_end_center) / 200.0,
                euclidean(old_velocity, new_velocity) / 10.0,
                signed_unit(old_velocity[0]) * signed_unit(new_velocity[0]),
                abs(old_velocity[0] - new_velocity[0]) / 5.0,
            ]
        )

    z = 0.0
    for idx, value in enumerate(features):
        if idx == 0:
            norm_value = value
        else:
            norm_value = (value - LEARNED_FEATURE_MEANS[idx]) / LEARNED_FEATURE_STDS[idx]
        z += LEARNED_FEATURE_WEIGHTS[idx] * norm_value
    return sigmoid(z)


def make_candidate(
    old_id: int,
    old_tracklet: Tracklet,
    new_id: int,
    new_tracklet: Tracklet,
    cfg: RelinkConfig,
    width: int,
    height: int,
) -> Optional[Candidate]:
    if old_id == new_id:
        return None
    if old_tracklet.end_frame >= new_tracklet.start_frame:
        return None
    if old_tracklet.length < cfg.min_old_frames or new_tracklet.length < cfg.min_new_frames:
        return None
    if cfg.require_old_not_edge and near_edge(old_tracklet.end_record, width, height, cfg.border_margin):
        return None

    gap = new_tracklet.start_frame - old_tracklet.end_frame
    if gap < 1 or gap > cfg.max_gap:
        return None

    if euclidean(old_tracklet.end_record.center, new_tracklet.start_record.center) > cfg.max_distance:
        return None

    score = learned_link_score(old_tracklet, new_tracklet, width=width, height=height)
    if score < cfg.score_threshold:
        return None

    return Candidate(
        old_id=old_id,
        new_id=new_id,
        mode_rank=0,
        cost=1.0 - score,
        gap=gap,
        old_end_frame=old_tracklet.end_frame,
        new_start_frame=new_tracklet.start_frame,
    )


def collect_candidates(
    tracklets: Dict[int, Tracklet],
    cfg: RelinkConfig,
    width: int,
    height: int,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    tracklet_items = list(tracklets.items())
    for new_id, new_tracklet in sorted(tracklet_items, key=lambda item: item[1].start_frame):
        for old_id, old_tracklet in tracklet_items:
            candidate = make_candidate(
                old_id=old_id,
                old_tracklet=old_tracklet,
                new_id=new_id,
                new_tracklet=new_tracklet,
                cfg=cfg,
                width=width,
                height=height,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def select_matches(candidates: List[Candidate]) -> List[Candidate]:
    if not candidates:
        return []

    best_for_new: Dict[int, Candidate] = {}
    best_for_old: Dict[int, Candidate] = {}
    for candidate in candidates:
        if better_candidate(candidate, best_for_new.get(candidate.new_id)):
            best_for_new[candidate.new_id] = candidate
        if better_candidate(candidate, best_for_old.get(candidate.old_id)):
            best_for_old[candidate.old_id] = candidate

    selected: List[Candidate] = []
    for candidate in candidates:
        if best_for_new.get(candidate.new_id) != candidate:
            continue
        if best_for_old.get(candidate.old_id) != candidate:
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.sort_key())


def relink_tracklets(
    tracklets: Dict[int, Tracklet],
    cfg: RelinkConfig,
    width: int,
    height: int,
) -> Tuple[Dict[int, int], int]:
    remap: Dict[int, int] = {}
    merges = 0

    for _ in range(max(1, cfg.max_rounds)):
        candidates = collect_candidates(tracklets, cfg, width, height)
        matches = select_matches(candidates)
        if not matches:
            break

        applied = 0
        for match in matches:
            if match.old_id not in tracklets or match.new_id not in tracklets:
                continue
            old_tracklet = tracklets[match.old_id]
            new_tracklet = tracklets[match.new_id]
            if old_tracklet.end_frame >= new_tracklet.start_frame:
                continue
            remap[match.new_id] = match.old_id
            old_tracklet.extend_from(new_tracklet)
            tracklets.pop(match.new_id, None)
            applied += 1

        merges += applied
        if applied == 0:
            break

    return remap, merges


def write_records(file_path: str, records: List[Record], remap: Dict[int, int]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w") as handle:
        for record in records:
            new_id = resolve_id(record.track_id, remap)
            fields = list(record.fields)
            fields[1] = str(int(new_id))
            handle.write(",".join(fields) + "\n")
    os.replace(tmp_path, file_path)


def build_config(args: argparse.Namespace) -> RelinkConfig:
    return RelinkConfig(
        border_margin=args.border_margin,
        max_gap=args.max_gap,
        max_distance=args.max_distance,
        min_new_frames=args.min_new_frames,
        min_old_frames=args.min_old_frames,
        require_old_not_edge=(not args.allow_old_edge),
        score_threshold=args.score_threshold,
        max_rounds=args.max_rounds,
    )


def main() -> None:
    args = parse_args()
    input_root = os.path.abspath(args.input_root)
    if not os.path.isdir(input_root):
        raise SystemExit(f"Missing input root: {input_root}")
    if args.in_place and args.output_dir is not None:
        raise SystemExit("--in-place and --output-dir are mutually exclusive.")

    width, height = load_shape_from_config(args.config)
    if args.frame_width is not None:
        width = int(args.frame_width)
    if args.frame_height is not None:
        height = int(args.frame_height)
    if width is None or height is None:
        raise SystemExit("Frame width/height not set. Provide --config or --frame-width/--frame-height.")

    tracker_files = discover_tracker_files(input_root, sequence=args.sequence)
    if not tracker_files:
        raise SystemExit(f"No tracker files found under: {input_root}")

    output_dir = None
    if not args.in_place:
        output_dir = (
            os.path.abspath(args.output_dir)
            if args.output_dir is not None
            else os.path.abspath(f"{input_root}_relinked")
        )
        os.makedirs(output_dir, exist_ok=True)

    cfg = build_config(args)
    total_merges = 0
    report_sequences: List[Dict[str, object]] = []

    for tracker_file in tracker_files:
        records = parse_records(tracker_file.src_path)
        if not records:
            continue
        tracklets = build_tracklets(records)
        remap, merges = relink_tracklets(tracklets, cfg, width=width, height=height)
        total_merges += merges

        dst_path = (
            tracker_file.src_path
            if args.in_place
            else os.path.join(output_dir, f"{tracker_file.sequence}_thermal.txt")
        )
        write_records(dst_path, records, remap)
        report_sequences.append(
            {
                "sequence": tracker_file.sequence,
                "source": tracker_file.src_path,
                "output": dst_path,
                "merges": merges,
                "remapped_ids": len(remap),
            }
        )
        print(
            f"{tracker_file.sequence}: wrote {os.path.basename(dst_path)} "
            f"(merges={merges}, remapped_ids={len(remap)})"
        )

    report_root = input_root if args.in_place else output_dir
    report_path = os.path.join(report_root, args.report_json)
    report = {
        "input_root": input_root,
        "output_root": report_root,
        "in_place": args.in_place,
        "config_used": cfg.to_dict(),
        "total_sequences": len(report_sequences),
        "total_merges": total_merges,
        "sequences": report_sequences,
    }
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"Done. Relinked tracks written to: {report_root}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
