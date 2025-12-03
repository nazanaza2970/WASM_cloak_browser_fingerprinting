# import ijson
# import json
# import os
# import random
# from pathlib import Path
# from collections import defaultdict
# from datetime import datetime


# def reservoir_sample(stream, k, rand=random.random):
#     sample = []
#     for i, item in enumerate(stream, 1):
#         if i <= k:
#             sample.append(item)
#         else:
#             j = int(rand() * i)
#             if j < k:
#                 sample[j] = item
#     if len(sample) < k:
#         raise ValueError(f"Stream contained only {len(sample)} items (needed {k}).")
#     return sample


# def category_reservoir_sample(stream, category_field, category_targets, per_category, rand=random.random):
#     """
#     Perform streaming reservoir sampling from the stream, grouped by categories.

#     Returns a dict: category -> list of sampled items.
#     """
#     samples = {cat: [] for cat in category_targets}
#     counts = defaultdict(int)

#     for item in stream:
#         categories = item.get(category_field, [])
#         if not isinstance(categories, list):
#             continue

#         for cat in categories:
#             if cat in category_targets:
#                 counts[cat] += 1
#                 current = samples[cat]
#                 if len(current) < per_category:
#                     current.append(item)
#                 else:
#                     j = int(rand() * counts[cat])
#                     if j < per_category:
#                         current[j] = item
#                 break  # use only the first matching category

#     # Verify that we got enough samples for each category
#     for cat in category_targets:
#         if len(samples[cat]) < per_category:
#             raise ValueError(f"Only found {len(samples[cat])} items for category '{cat}' (needed {per_category}).")
#     return samples


# def sample_two_sources(
#     file1: str,
#     file2: str,
#     output_dir: str,
#     count: int,
#     n1: int = 200,
#     n2: int = 800,
#     *,
#     shuffle_output: bool = True,
#     prefix: str = "sample"
# ):
#     file1 = Path(file1)
#     file2 = Path(file2)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     category_targets = [
#         "AudioContext Fingerprinting",
#         "WebRTC Fingerprinting",
#         "Canvas Font Fingerprinting",
#         "Canvas Fingerprinting",
#     ]
#     per_category = n2 // len(category_targets)

#     for rep in range(1, count + 1):
#         # Sample from file1 (uniformly)
#         with file1.open("r") as f1:
#             stream1 = ijson.items(f1, "item")
#             sample1 = reservoir_sample(stream1, n1)

#         # Sample from file2 by category
#         with file2.open("r") as f2:
#             stream2 = ijson.items(f2, "item")
#             cat_samples = category_reservoir_sample(
#                 stream2,
#                 category_field="categories",
#                 category_targets=category_targets,
#                 per_category=per_category
#             )
#             sample2 = sum(cat_samples.values(), [])

#         combined = sample1 + sample2
#         if shuffle_output:
#             random.shuffle(combined)

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         outfile = output_dir / f"{prefix}_{rep:02d}_{timestamp}.json"
#         with outfile.open("w") as out:
#             json.dump(combined, out, indent=2)

#         print(f"✓ wrote {outfile} ({len(combined)} items: {n1} + {n2})")


#!/usr/bin/env python3
"""
sample_json_cli.py

Stream‑sample items from two large JSON files without loading them fully into
memory. Produces combined datasets of exactly N₁ + N₂ items, where N₂ is split
evenly across specified categories in the second file.

Example
-------
python sample_json_cli.py first.json second.json output 5 \
        --n1 200 --n2 800 \
        --categories "AudioContext Fingerprinting" "WebRTC Fingerprinting" \
        "Canvas Font Fingerprinting" "Canvas Fingerprinting" \
        --seed 42
"""

import argparse
import json
from pathlib import Path
import random
from datetime import datetime
from collections import defaultdict
import ijson
import os


# ---------------------------------------------------------------------------
# Streaming sampling helpers
# ---------------------------------------------------------------------------

def reservoir_sample(stream, k, rand):
    """Return *k* uniformly random items from *stream* using reservoir sampling."""
    sample = []
    for i, item in enumerate(stream, 1):
        if i <= k:
            sample.append(item)
        else:
            j = int(rand() * i)
            if j < k:
                sample[j] = item
    # if len(sample) < k:
    #     raise ValueError(f"Stream ended with {len(sample)} items (expected {k}).")
    return sample


def category_reservoir_sample(stream, category_field, categories, k, rand):
    """Reservoir‑sample exactly *k* items, *k/len(categories)* per category."""
    per_cat = k // len(categories)
    samples = {c: [] for c in categories}
    counts = defaultdict(int)

    for item in stream:
        cats = item.get(category_field, [])
        if not isinstance(cats, list):
            continue
        for cat in cats:
            if cat in categories:
                counts[cat] += 1
                bucket = samples[cat]
                if len(bucket) < per_cat:
                    bucket.append(item)
                else:
                    j = int(rand() * counts[cat])
                    if j < per_cat:
                        bucket[j] = item
                break  # stop after first matching category

    for cat in categories:
        if len(samples[cat]) < per_cat:
            raise ValueError(
                f"Found only {len(samples[cat])} records for category '{cat}' (need {per_cat})."
            )

    combined = []
    for cat in categories:
        combined.extend(samples[cat])
    return combined


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------

def sample_two_sources(
    file1: Path,
    file2: Path,
    output_dir: Path,
    count: int,
    n1: int,
    n2: int,
    categories,
    category_field: str,
    prefix: str,
    seed,
):
    rand = random.Random(seed) if seed is not None else random
    if os.path.isdir(output_dir):
        print("Directory exists")
    else:
        print("Directory does not exist")


    for rep in range(1, count + 1):
        # ---- sample from file1 -----------------------------------------------------
        with file1.open("r", encoding="utf‑8") as f1:
            sample1 = reservoir_sample(ijson.items(f1, "item"), n1, rand.random)

        # ---- sample from file2 -----------------------------------------------------
        with file2.open("r", encoding="utf‑8") as f2:
            sample2 = category_reservoir_sample(
                ijson.items(f2, "item"),
                category_field=category_field,
                categories=categories,
                k=n2,
                rand=rand.random,
            )

        # ---- combine and write output --------------------------------------------
        combined = sample1 + sample2
        rand.shuffle(combined)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = output_dir / f"{prefix}_{rep:02d}_{timestamp}.json"
        with outfile.open("w", encoding="utf‑8") as out:
            json.dump(combined, out, indent=2)
        print(f"✓ Wrote {outfile} ({len(combined)} items)")


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream‑sample two large JSON files into smaller combined datasets."
    )
    parser.add_argument("file1", help="Path to first JSON file")
    parser.add_argument("file2", help="Path to second JSON file")
    parser.add_argument("output_dir", help="Directory where output samples are saved")
    parser.add_argument("count", type=int, help="Number of sample files to create")

    parser.add_argument("--n1", type=int, default=100, help="Items from file1 (default 200)")
    parser.add_argument(
        "--n2", type=int, default=400, help="Total items from file2 (default 800)"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "AudioContext Fingerprinting",
            "WebRTC Fingerprinting",
            "Canvas Font Fingerprinting",
            "Canvas Fingerprinting",
        ],
        help="Category labels to sample from in file2 (space‑separated list)",
    )
    parser.add_argument(
        "--category-field",
        default="categories",
        help="Name of array field holding categories in file2 (default 'categories')",
    )
    parser.add_argument(
        "--prefix", default="sample", help="Filename prefix for outputs (default 'sample')"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible sampling"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.n2 % len(args.categories) != 0:
        raise SystemExit(
            f"Error: n2 ({args.n2}) must be divisible by the number of categories ({len(args.categories)})."
        )

    sample_two_sources(
        Path(args.file1),
        Path(args.file2),
        Path(args.output_dir),
        args.count,
        args.n1,
        args.n2,
        args.categories,
        args.category_field,
        args.prefix,
        args.seed,
    )


if __name__ == "__main__":
    main()
# This code is designed to be run as a script, so it will not execute if imported as a module.
# It provides a command-line interface for sampling JSON files.