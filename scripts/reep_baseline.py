#!/usr/bin/env python3
"""
ReEP baseline evaluation: run the ReEP reentrancy detector on the synthetic
and real-world datasets and compute function-level precision / recall / F1.

Output: results/baselines/reep_{synthetic,realworld}.json

Usage examples:
  # Smoke check — 5 contracts from real-world positives only
  python scripts/reep_baseline.py --dataset realworld --limit 5 --positives-only

  # Full synthetic run (≈35 min at -j 8)
  python scripts/reep_baseline.py --dataset synthetic -j 8

  # Full real-world run
  python scripts/reep_baseline.py --dataset realworld -j 8
"""
import argparse
import csv
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "baselines"

REENTRANCY_TYPES = {"reentrancy", "cross-function"}


def _is_actual_reentrancy(c):
    """A contract is actual-reentrancy iff it has vulnerability_type=='reentrancy',
    OR it has vulnerability_type=='cross-function' AND its sidecar metadata's
    template_name begins with 'reentrancy_'. The 'cross-function' label in the
    master dataset_index is a generic umbrella covering 7 sub-vulnerability
    classes (lock, tod, dos, access, state, timestamp, reentrancy); only the
    reentrancy_* templates are actual reentrancy.
    """
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

# Docker image built by external/reep/Dockerfile (smoke tag used here;
# override with --image for a separately tagged production build).
DEFAULT_IMAGE = "reep:smoke"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_samples(split: str, seed: int, limit: int | None, positives_only: bool,
                 split_filter: str | None = None,
                 max_positives: int | None = None) -> list[dict]:
    """Return a list of sample dicts with keys:
       sol_path, graph_path, contract_label (0/1), vulnerability_type

    split_filter: when split=='synthetic', restrict to one of the
                  synthetic-split subsets {train, test, dev}.
    max_positives: cap reentrancy positives to N (sampled by seed) before the
                   balanced-negative draw. Lets us produce a 31+31 slice on
                   synthetic for parity with the real-world 31+31.
    """
    index_path = DATA_ROOT / split / "function_level_dataset" / "dataset_index.json"
    graphs_dir = DATA_ROOT / split / "function_level_dataset" / "function_graphs"

    with open(index_path) as f:
        index = json.load(f)
    contracts = index["contracts"]

    if split_filter is not None:
        sm_path = DATA_ROOT / "synthetic-split" / "split_metadata.json"
        with open(sm_path) as f:
            sm = json.load(f)
        if split_filter not in sm["splits"]:
            raise ValueError(f"unknown split_filter={split_filter!r}; "
                             f"available: {list(sm['splits'].keys())}")
        allowed = set(sm["splits"][split_filter]["templates"])

        def _graph_name(c):
            from pathlib import Path as _P
            return _P(c["path"]).stem + ".json"

        contracts = [c for c in contracts if _graph_name(c) in allowed]

    positives = [c for c in contracts if _is_actual_reentrancy(c)]
    negatives = [c for c in contracts if c["vulnerability_type"] == "clean"]

    rng = random.Random(seed)
    if max_positives is not None and len(positives) > max_positives:
        positives = rng.sample(positives, max_positives)
    if not positives_only:
        negatives = rng.sample(negatives, min(len(positives), len(negatives)))
    else:
        negatives = []

    selected = positives + negatives

    if limit is not None:
        rng.shuffle(selected)
        selected = selected[:limit]

    samples = []
    for c in selected:
        sol_path = ROOT / c["path"]
        # graph file is named after the sol basename
        graph_name = Path(c["path"]).stem + ".json"
        graph_path = graphs_dir / graph_name

        if not sol_path.exists():
            print(f"  WARN: missing sol file {sol_path}", file=sys.stderr)
            continue
        if not graph_path.exists():
            print(f"  WARN: missing graph file {graph_path}", file=sys.stderr)
            continue

        samples.append({
            "sol_path": str(sol_path),
            "graph_path": str(graph_path),
            "contract_label": 1 if c["vulnerability_type"] in REENTRANCY_TYPES else 0,
            "vulnerability_type": c["vulnerability_type"],
        })

    return samples


# ---------------------------------------------------------------------------
# Solc version selection
# ---------------------------------------------------------------------------

# Ordered list of solc versions installed in the Docker image (must stay in
# sync with the Dockerfile's solc-select install step).
_INSTALLED_SOLC = [
    (0, 4, 24), (0, 4, 25), (0, 4, 26),
    (0, 5, 0),  (0, 5, 16),
    (0, 6, 0),  (0, 6, 12),
    (0, 7, 0),  (0, 7, 6),
    (0, 8, 0),  (0, 8, 17),
]

_PRAGMA_RE = re.compile(
    r'pragma\s+solidity\s+(?P<op>[^\d]*)(?P<ver>\d+\.\d+(?:\.\d+)?)'
)


def _select_solc_version(sol_text: str) -> str:
    """Return the best installed solc version string for this contract's pragma."""
    m = _PRAGMA_RE.search(sol_text)
    if not m:
        return "0.4.25"  # fallback

    op = m.group("op").strip()   # e.g. "^", ">=", "", ">"
    ver_str = m.group("ver")
    parts = tuple(int(x) for x in ver_str.split("."))
    # Pad to 3-tuple
    if len(parts) == 2:
        parts = parts + (0,)

    major, minor = parts[0], parts[1]

    # For caret/range constraints: pick highest installed minor within the
    # major.minor series that is >= the specified version.
    # For exact/no-op: pick smallest installed >= specified (forward-compat
    # within minor is fine since Slither uses --solc-disable-warnings).
    candidates = [
        v for v in _INSTALLED_SOLC
        if v[0] == major and (
            v >= parts if op in ("", "=") else
            v >= parts   # for ^, >=, > all just pick >= and compatible major
        )
    ]
    if not candidates:
        # Fall back to highest overall installed
        candidates = _INSTALLED_SOLC

    # Prefer highest within same major series for ^ / >=; exact for no op.
    if op in ("^", "~"):
        # Same major.minor constraint
        same_minor = [v for v in candidates if v[0] == major and v[1] == minor]
        chosen = max(same_minor) if same_minor else max(candidates)
    else:
        chosen = min(candidates)

    return f"{chosen[0]}.{chosen[1]}.{chosen[2]}"


# ---------------------------------------------------------------------------
# ReEP runner (one contract)
# ---------------------------------------------------------------------------

def _run_one(args_tuple) -> dict:
    """Top-level (picklable) worker for ProcessPoolExecutor."""
    return run_reep_one(*args_tuple)


def run_reep_one(sol_path: str, graph_path: str, contract_label: int,
                 vulnerability_type: str, image: str, timeout_s: int) -> dict:
    """Run ReEP on a single .sol file, return per-function predictions + labels."""
    sol_path = Path(sol_path)
    graph_path = Path(graph_path)

    with open(graph_path) as f:
        graph = json.load(f)

    functions = graph["functions"]
    function_labels = graph["function_labels"]
    sol_basename = sol_path.name

    with open(sol_path) as f:
        sol_text = f.read()
    solc_ver = _select_solc_version(sol_text)

    record = {
        "sol": str(sol_path),
        "vulnerability_type": vulnerability_type,
        "contract_label": contract_label,
        "solc_ver": solc_ver,
        "reep_is_bug": None,
        "reep_function": None,
        "reep_contract": None,
        "elapsed_s": None,
        "timed_out": False,
        "error": None,
        "function_predictions": [],  # [{name, label, pred}]
    }

    with tempfile.TemporaryDirectory() as work_dir:
        dest = os.path.join(work_dir, sol_basename)
        csv_out = os.path.join(work_dir, sol_basename + ".csv")
        import shutil
        shutil.copy(sol_path, dest)

        # Override entrypoint to set solc version before calling ReEP.
        reep_cmd = (
            f"solc-select use {solc_ver} && "
            f"python /opt/reep-src/ReEP_tool/ReEP.py "
            f"-f /work/{sol_basename} -r -a -o /work/{sol_basename}.csv"
        )
        cmd = [
            "docker", "run", "--rm",
            "--entrypoint", "bash",
            "-v", f"{work_dir}:/work",
            image,
            "-c", reep_cmd,
        ]

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            record["elapsed_s"] = time.time() - t0
        except subprocess.TimeoutExpired:
            record["elapsed_s"] = time.time() - t0
            record["timed_out"] = True
            record["error"] = "timeout"
            record["reep_is_bug"] = False
        except Exception as exc:
            record["elapsed_s"] = time.time() - t0
            record["error"] = str(exc)
            record["reep_is_bug"] = False

        if not record["timed_out"] and record["error"] is None:
            # Parse CSV output
            try:
                with open(csv_out, newline="") as cf:
                    rows = list(csv.reader(cf))
                if rows:
                    row = rows[-1]
                    # format: filename, is_bug, ttime, Function_name, Contract_name, RE
                    if len(row) >= 5:
                        record["reep_is_bug"] = row[1].strip() == "True"
                        record["reep_function"] = row[3].strip()
                        record["reep_contract"] = row[4].strip()
                    else:
                        record["reep_is_bug"] = False
                        record["error"] = "malformed csv row"
                else:
                    record["reep_is_bug"] = False
                    record["error"] = "empty csv"
            except FileNotFoundError:
                record["reep_is_bug"] = False
                record["error"] = "no csv output"

    # Build per-function predictions.
    # ReEP flags at most one function per run (-a mode). Match by stripping
    # the parameter signature from ReEP's reported Function_name.
    reep_fn_base = None
    if record["reep_function"]:
        reep_fn_base = record["reep_function"].split("(")[0].strip()

    for fn, lbl in zip(functions, function_labels):
        our_name = fn["name"]
        if record["reep_is_bug"] and reep_fn_base and our_name == reep_fn_base:
            pred = 1
        else:
            pred = 0
        record["function_predictions"].append({
            "name": our_name,
            "label": lbl,
            "pred": pred,
        })

    return record


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(records: list[dict]) -> dict:
    all_labels, all_preds = [], []
    for rec in records:
        for fp in rec["function_predictions"]:
            all_labels.append(fp["label"])
            all_preds.append(fp["pred"])

    if not all_labels:
        return {}

    # Contract-level metrics (is any function in contract flagged?)
    contract_labels, contract_preds = [], []
    for rec in records:
        contract_labels.append(rec["contract_label"])
        flagged = any(fp["pred"] == 1 for fp in rec["function_predictions"])
        contract_preds.append(1 if flagged else 0)

    def safe_auc(labels, preds):
        try:
            return roc_auc_score(labels, preds)
        except Exception:
            return None

    return {
        "function_level": {
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "accuracy": accuracy_score(all_labels, all_preds),
            "auc": safe_auc(all_labels, all_preds),
            "n_samples": len(all_labels),
            "n_positive": sum(all_labels),
            "n_predicted": sum(all_preds),
        },
        "contract_level": {
            "precision": precision_score(contract_labels, contract_preds, zero_division=0),
            "recall": recall_score(contract_labels, contract_preds, zero_division=0),
            "f1": f1_score(contract_labels, contract_preds, zero_division=0),
            "accuracy": accuracy_score(contract_labels, contract_preds),
            "auc": safe_auc(contract_labels, contract_preds),
            "n_contracts": len(contract_labels),
            "n_positive_contracts": sum(contract_labels),
            "n_predicted_contracts": sum(contract_preds),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run ReEP reentrancy baseline")
    parser.add_argument("--dataset", choices=["synthetic", "realworld"], required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image tag")
    parser.add_argument("--timeout", type=int, default=120, help="Per-sample timeout (s)")
    parser.add_argument("-j", "--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Cap total samples")
    parser.add_argument("--positives-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-filter", choices=["train", "test", "dev"],
                        default=None,
                        help="When --dataset synthetic, restrict to a "
                             "synthetic-split subset.")
    parser.add_argument("--max-positives", type=int, default=None,
                        help="Cap reentrancy positives to N; balanced "
                             "negatives drawn afterward (for 31+31 slices).")
    parser.add_argument("--output", default=None, help="Override output JSON path")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else RESULTS_DIR / f"reep_{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} samples …")
    samples = load_samples(args.dataset, args.seed, args.limit, args.positives_only,
                           split_filter=args.split_filter,
                           max_positives=args.max_positives)
    pos = sum(1 for s in samples if s["contract_label"] == 1)
    neg = len(samples) - pos
    print(f"  {len(samples)} contracts: {pos} positive ({', '.join(REENTRANCY_TYPES)}), {neg} negative")

    # Build task args
    tasks = [
        (s["sol_path"], s["graph_path"], s["contract_label"],
         s["vulnerability_type"], args.image, args.timeout)
        for s in samples
    ]

    records = []
    n_timeout = 0
    n_error = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                rec = fut.result()
            except Exception as exc:
                idx = futures[fut]
                print(f"  [WORKER ERROR] sample {idx}: {exc}", file=sys.stderr)
                n_error += 1
                continue
            records.append(rec)
            if rec["timed_out"]:
                n_timeout += 1
            if rec["error"] and not rec["timed_out"]:
                n_error += 1
            elapsed = time.time() - t_start
            rate = done / elapsed
            eta = (len(tasks) - done) / rate if rate > 0 else 0
            is_bug = "BUG" if rec["reep_is_bug"] else "    "
            fn = (rec["reep_function"] or "")[:30]
            print(f"  [{done}/{len(tasks)}] {is_bug} {Path(rec['sol']).name[:40]:<40} "
                  f"fn={fn:<30} t={rec['elapsed_s']:.1f}s  ETA={eta:.0f}s")

    total_elapsed = time.time() - t_start
    metrics = compute_metrics(records)

    output = {
        "dataset": args.dataset,
        "split_filter": args.split_filter,
        "max_positives": args.max_positives,
        "image": args.image,
        "timeout_s": args.timeout,
        "seed": args.seed,
        "n_contracts": len(records),
        "n_timeout": n_timeout,
        "n_error": n_error,
        "total_elapsed_s": total_elapsed,
        "metrics": metrics,
        "records": records,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Timeouts: {n_timeout}/{len(records)}  Errors: {n_error}/{len(records)}")
    print(f"Total elapsed: {total_elapsed/60:.1f} min")
    print()
    print("=== Function-level metrics ===")
    m = metrics.get("function_level", {})
    print(f"  Precision: {m.get('precision', 0):.4f}")
    print(f"  Recall:    {m.get('recall', 0):.4f}")
    print(f"  F1:        {m.get('f1', 0):.4f}")
    print(f"  AUC:       {m.get('auc') or 'n/a'}")
    print()
    print("=== Contract-level metrics ===")
    m2 = metrics.get("contract_level", {})
    print(f"  Precision: {m2.get('precision', 0):.4f}")
    print(f"  Recall:    {m2.get('recall', 0):.4f}")
    print(f"  F1:        {m2.get('f1', 0):.4f}")


if __name__ == "__main__":
    main()
