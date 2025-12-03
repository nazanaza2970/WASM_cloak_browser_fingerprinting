#!/usr/bin/env python3
"""
Compare code size between JSON files in `js/` and `wasm/`.

The JSON files look like:
[
  {
    "some key": {
      "label": int,
      "content": "…JavaScript code…"
    }
  },
  ...
]

Files are paired by the part **after** the first underscore:
    js_foo.json   ↔︎  wasm_foo.json
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

def extract_js_content_bytes(json_path: Path) -> int:
    """Return total bytes of all `"content"` strings in the JSON file."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return sum(len(entry[key]["content"].encode("utf-8"))
               for entry in data
               for key in entry)

def build_basename_map(folder: Path, prefix: str) -> Dict[str, Path]:
    """
    Map each base name (part after first '_') to its full Path.
    Only considers *.json files that start with the given prefix.
    """
    mapping = {}
    for p in folder.glob(f"{prefix}_*.json"):
        base = p.stem.split("_", 1)[1]  # e.g. "foo" from "js_foo"
        mapping[base] = p
    return mapping

def compare_folders(js_dir: str | Path, wasm_dir: str | Path) -> None:
    js_dir, wasm_dir = Path(js_dir), Path(wasm_dir)

    js_map   = build_basename_map(js_dir,   "js")
    wasm_map = build_basename_map(wasm_dir, "wasm")

    common   = sorted(set(js_map) & set(wasm_map))
    only_js  = sorted(set(js_map) - set(wasm_map))
    only_wasm= sorted(set(wasm_map) - set(js_map))

    if not common:
        print("⚠️  No matching basenames between folders.")
        return

    results: List[Dict[str, float | str]] = []
    js_total_bytes = wasm_total_bytes = 0

    for base in common:
        js_bytes   = extract_js_content_bytes(js_map[base])
        wasm_bytes = extract_js_content_bytes(wasm_map[base])

        diff   = wasm_bytes - js_bytes
        pct    = (diff / js_bytes * 100) if js_bytes else float("inf")

        results.append({
            "base": base,
            "js_bytes": js_bytes,
            "wasm_bytes": wasm_bytes,
            "diff": diff,
            "pct": pct
        })

        js_total_bytes   += js_bytes
        wasm_total_bytes += wasm_bytes

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"{'Base name':<20} {'JS (B)':>10} {'WASM (B)':>12} {'Diff':>10} {'% Diff':>8}")
    for r in results:
        print(f"{r['base']:<20} {r['js_bytes']:>10} {r['wasm_bytes']:>12} "
              f"{r['diff']:>10} {r['pct']:>7.2f}%")

    total_diff = wasm_total_bytes - js_total_bytes
    total_pct  = (total_diff / js_total_bytes * 100) if js_total_bytes else float("inf")

    print("\nTotals:")
    print(f"{'JS total bytes':<20} {js_total_bytes}")
    print(f"{'WASM total bytes':<20} {wasm_total_bytes}")
    print(f"{'Total diff (B)':<20} {total_diff}")
    print(f"{'Total % diff':<20} {total_pct:.2f}%")

    # List unmatched files (if any)
    if only_js or only_wasm:
        print("\nUnmatched basenames:")
        if only_js:
            print("  Present only in js/:   ", ", ".join(only_js))
        if only_wasm:
            print("  Present only in wasm/: ", ", ".join(only_wasm))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compare JS vs WASM JSON content sizes")
    ap.add_argument("js_dir",   help="Path to js/ folder")
    ap.add_argument("wasm_dir", help="Path to wasm/ folder")
    args = ap.parse_args()

    compare_folders(args.js_dir, args.wasm_dir)
