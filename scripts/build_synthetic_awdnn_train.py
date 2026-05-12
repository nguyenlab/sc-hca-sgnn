#!/usr/bin/env python3
"""
Build the AWDNN training fragment file from our synthetic train split.

Selection mirrors awdnn_baseline.py / reep_baseline.py:
  positives = vulnerability_type in {reentrancy, cross-function}
  negatives = vulnerability_type == clean, balanced via random.sample(seed=42)
                                            so |neg| == |pos|

Output: a single fragment file in AWDNN's expected format
        (see external/awdnn/preprocess_sol_to_fragments.py).
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEPARATOR = "-" * 40
REENTRANCY_TYPES = {"reentrancy", "cross-function"}


def _is_actual_reentrancy(c):
    vt = c.get("vulnerability_type")
    if vt == "reentrancy":
        return True
    if vt != "cross-function":
        return False
    sol = ROOT / c["path"]
    js = sol.with_suffix(".json")
    if not js.exists():
        return False
    try:
        with open(js) as fh:
            md = json.load(fh)
        tn = md.get("template_name", "") or ""
        return tn.startswith("reentrancy_")
    except Exception:
        return False


def write_fragment(fh, basename, source_text, label):
    fh.write(f"{basename}\n")
    for raw in source_text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            stripped = "// " + stripped
        fh.write(f"{stripped}\n")
    fh.write(f"{label}\n")
    fh.write(f"{SEPARATOR}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build AWDNN training fragments from the synthetic train split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for balanced negative sampling")
    parser.add_argument("--output", default="/tmp/awdnn-synth-train/contracts_synth_train.txt")
    parser.add_argument("--manifest", default="/tmp/awdnn-synth-train/manifest.json")
    args = parser.parse_args()

    idx_path = ROOT / "data" / "synthetic" / "function_level_dataset" / "dataset_index.json"
    sm_path = ROOT / "data" / "synthetic-split" / "split_metadata.json"

    with open(idx_path) as f:
        contracts = json.load(f)["contracts"]
    with open(sm_path) as f:
        train_set = set(json.load(f)["splits"]["train"]["templates"])

    def graph_name(c):
        return Path(c["path"]).stem + ".json"

    train_contracts = [c for c in contracts if graph_name(c) in train_set]
    positives = [c for c in train_contracts if _is_actual_reentrancy(c)]
    negatives = [c for c in train_contracts if c["vulnerability_type"] == "clean"]

    rng = random.Random(args.seed)
    negatives = rng.sample(negatives, min(len(positives), len(negatives)))

    print(f"train split: {len(train_contracts)} contracts")
    print(f"  positives ({', '.join(REENTRANCY_TYPES)}): {len(positives)}")
    print(f"  negatives (clean, balanced @ seed={args.seed}): {len(negatives)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for c in positives:
            sol = ROOT / c["path"]
            text = sol.read_text(encoding="utf-8", errors="replace")
            write_fragment(fh, sol.name, text, 1)
            manifest.append({"name": sol.name, "label": 1,
                             "vulnerability_type": c["vulnerability_type"],
                             "source": str(sol)})
        for c in negatives:
            sol = ROOT / c["path"]
            text = sol.read_text(encoding="utf-8", errors="replace")
            write_fragment(fh, sol.name, text, 0)
            manifest.append({"name": sol.name, "label": 0,
                             "vulnerability_type": c["vulnerability_type"],
                             "source": str(sol)})

    print(f"  wrote {len(manifest)} fragments -> {out_path}")
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
