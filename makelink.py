
#!/usr/bin/env python3
"""Create a small symlinked ERA5 old-data subset for Pangu training.

The script does not copy large data files. It creates a directory that looks
like the original old-data root:

    sampled-old-data/
      metadata.json -> source/metadata.json
      static -> source/static
      stats -> source/stats
      data/... -> symlinks to selected source files

Selected start times are expanded with a label lead time, usually +24 hours, so
the datapipe can still build input/output pairs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path


TIME_PATTERNS = [
    # 1980010100, 1980-01-01-00, 1980_01_01_00, 1980/01/01/00
    re.compile(
        r"(?P<year>19[0-9]{2}|20[0-9]{2})[-_/]?"
        r"(?P<month>0[1-9]|1[0-2])[-_/]?"
        r"(?P<day>0[1-9]|[12][0-9]|3[01])[-_/]?"
        r"(?P<hour>00|06|12|18)"
    ),
    # year=1980/month=01/day=01/hour=00 style
    re.compile(
        r"year[=/_-]?(?P<year>19[0-9]{2}|20[0-9]{2}).*?"
        r"month[=/_-]?(?P<month>0?[1-9]|1[0-2]).*?"
        r"day[=/_-]?(?P<day>0?[1-9]|[12][0-9]|3[01]).*?"
        r"hour[=/_-]?(?P<hour>00|06|12|18)"
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-root", required=True, help="Original ERA5 old-data root")
    parser.add_argument("--output-root", required=True, help="Symlink subset root to create")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(1980, 2000)))
    parser.add_argument("--months", nargs="+", type=int, default=[1, 3, 4, 7, 10, 12])
    parser.add_argument("--days", nargs="+", type=int, default=[1, 2, 15, 16])
    parser.add_argument("--hours", nargs="+", type=int, default=[0, 6, 12, 18])
    parser.add_argument("--lead-hours", type=int, default=24)
    parser.add_argument("--data-subdir", default="data")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-links", action="store_true")
    return parser.parse_args()


def extract_time(path: Path) -> dt.datetime | None:
    text = path.as_posix()
    for pattern in TIME_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            return dt.datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                int(match.group("hour")),
            )
        except ValueError:
            return None
    return None


def wanted_start_times(args: argparse.Namespace) -> set[dt.datetime]:
    times = set()
    for year in args.years:
        for month in args.months:
            for day in args.days:
                for hour in args.hours:
                    try:
                        times.add(dt.datetime(year, month, day, hour))
                    except ValueError:
                        continue
    return times


def link_path(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    os.symlink(src, dst, target_is_directory=src.is_dir())


def link_small_root_entries(source_root: Path, output_root: Path, overwrite: bool, dry_run: bool) -> None:
    for name in ("metadata.json", "static", "stats"):
        src = source_root / name
        dst = output_root / name
        if not src.exists():
            print(f"warn: missing {src}")
            continue
        print(f"link {dst} -> {src}")
        if not dry_run:
            link_path(src, dst, overwrite)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    source_data = source_root / args.data_subdir
    output_data = output_root / args.data_subdir

    if not source_data.exists():
        raise FileNotFoundError(f"data directory not found: {source_data}")

    start_times = wanted_start_times(args)
    label_delta = dt.timedelta(hours=args.lead_hours)
    required_times = set(start_times)
    required_times.update(t + label_delta for t in start_times)

    print(f"source_root={source_root}")
    print(f"output_root={output_root}")
    print(f"start_times={len(start_times)}")
    print(f"required_times_with_labels={len(required_times)}")
    print("scanning source data...")

    matched: dict[dt.datetime, list[Path]] = {}
    scanned = 0
    for path in source_data.rglob("*"):
        if not path.is_file():
            continue
        scanned += 1
        time_key = extract_time(path.relative_to(source_root))
        if time_key in required_times:
            matched.setdefault(time_key, []).append(path)

    selected_files = []
    for files in matched.values():
        selected_files.extend(files)

    missing_start = sorted(t for t in start_times if t not in matched)
    missing_label = sorted(t + label_delta for t in start_times if t + label_delta not in matched)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "data_subdir": args.data_subdir,
        "years": args.years,
        "months": args.months,
        "days": args.days,
        "hours": args.hours,
        "lead_hours": args.lead_hours,
        "scanned_files": scanned,
        "start_times": len(start_times),
        "required_times_with_labels": len(required_times),
        "matched_times": len(matched),
        "selected_files": len(selected_files),
        "missing_start_times": [t.isoformat() for t in missing_start[:50]],
        "missing_label_times": [t.isoformat() for t in missing_label[:50]],
        "missing_start_count": len(missing_start),
        "missing_label_count": len(missing_label),
        "dry_run": args.dry_run,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    link_small_root_entries(source_root, output_root, args.overwrite_links, args.dry_run)
    for src in selected_files:
        rel = src.relative_to(source_root)
        dst = output_root / rel
        link_path(src, dst, args.overwrite_links)

    (output_root / "sample_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"created {len(selected_files)} data symlinks under {output_data}")
    print(f"manifest: {output_root / 'sample_manifest.json'}")


if __name__ == "__main__":
    main()


