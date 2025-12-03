#!/usr/bin/env python3
"""
Combine numbered `js_sample_…_[N].json` and `wasm_sample_…_[N].json` files.

Usage
-----
python combine_samples.py /path/to/input_dir /path/to/output_dir
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List



FILE_RE = re.compile(
    r"^(?P<prefix>(?:js|wasm))_(?P<basename>.+?)_(?P<count>\d+)\.json$"
)


def group_sample_files(input_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    Scan `input_dir` and return a dict:

        {
          "js":   {"blabla": [Path(...), Path(...), …], "foo": […]},
          "wasm": {"something": […], …}
        }

    Keys inside the inner dict are the base names that appear between the
    prefix and the trailing count.
    """
    groups: Dict[str, Dict[str, List[Path]]] = {"js": {}, "wasm": {}}

    for path in input_dir.glob("*.json"):
        match = FILE_RE.match(path.name)
        if not match:
            continue  # skip anything that doesn't fit the pattern

        kind = match["prefix"].split("_", 1)[0]  # "js" or "wasm"
        base = match["basename"]

        groups.setdefault(kind, {}).setdefault(base, []).append(path)

    return groups


def combine_files(file_list: List[Path]) -> List:
    """Load each JSON file and return a single list with their contents."""
    combined: List = []
    for f in sorted(file_list):
        with f.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        combined.append(data)
    return combined

def write_combined_file(kind: str, base: str, data: List, out_root: Path) -> None:
    """Write the combined list to the correct output directory."""
    out_dir = out_root / kind  # <out>/js or <out>/wasm
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{kind}_{base}.json"
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)

    print(f"✓ Wrote {out_path.relative_to(out_root)} ({len(data)} parts)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine sample JSON files.")
    parser.add_argument("input_dir", type=Path, help="Directory containing the numbered sample files")
    parser.add_argument("output_dir", type=Path, help="Directory where combined files are written")
    args = parser.parse_args()

    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        parser.error(f"Input directory {input_dir} does not exist or is not a directory")

    grouped = group_sample_files(input_dir)

    if not any(grouped[kind] for kind in ("js", "wasm")):
        print("No matching sample files were found — nothing to do.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for kind in ("js", "wasm"):
        for base, paths in grouped[kind].items():
            combined_data = combine_files(paths)
            write_combined_file(kind, base, combined_data, output_dir)


if __name__ == "__main__":
    main()
