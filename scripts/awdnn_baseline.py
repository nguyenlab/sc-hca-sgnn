#!/usr/bin/env python3
"""
AWDNN baseline evaluation: cross-dataset evaluation of the AWDNN reentrancy
detector (Osei, Huang, Ma 2025) on the TOSEM R2 real-world and synthetic
test slices.

AWDNN is contract-level only; metrics are reported at the contract level.

Because AWDNN's architecture is highly seed-dependent (authors' own logs cover
an F1 range of 0.843--0.875 on their own benchmark), we run N seeds per
configuration and report mean + stdev + per-seed records.

Configurations:
  --train-corpus authors   : /opt/awdnn-src/contracts_re.txt (built into image)
  --train-corpus synthetic : host fragment file produced by
                             scripts/build_synthetic_awdnn_train.py

  --test-datasets realworld synthetic
                          : run inference on each named dataset, one JSON
                            output per (corpus, dataset) pair.

For each seed we train once (or reuse --reuse-ckpt-dir) and then run
inference on every requested test dataset, mirroring the train-once-infer-many
pattern needed to compare a single model against multiple held-out test sets.
"""
import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "baselines"

REENTRANCY_TYPES = {"reentrancy", "cross-function"}
SEPARATOR = "-" * 40
DEFAULT_IMAGE = "awdnn:smoke"


def _is_actual_reentrancy(c):
    """See reep_baseline._is_actual_reentrancy: filters cross-function entries
    by sidecar template_name to keep only reentrancy_* templates."""
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

TRAIN_CORPUS_PATHS = {
    # In-container path (built into the Docker image)
    "authors": "/opt/awdnn-src/contracts_re.txt",
    # Host path produced by build_synthetic_awdnn_train.py
    "synthetic": "/tmp/awdnn-synth-train/contracts_synth_train.txt",
    # Host path produced by build_awdnn_extended_slices.py
    "synthetic-ext": "/tmp/awdnn-synth-train-ext/contracts.txt",
}

# Extended-slice test manifest (built by build_awdnn_extended_slices.py).
# When --test-datasets includes "synthetic-ext", load_samples reads from this
# manifest instead of the function-level dataset_index.
SYNTH_EXT_TEST_MANIFEST = "/tmp/awdnn-synth-test-ext/manifest.json"

# Function-fragment test files (built by build_awdnn_frag_test_slices.py).
# When --test-datasets includes "synthetic-frag" or "realworld-frag", the
# pre-built fragment file is copied directly into the work dir rather than
# being assembled from whole-contract .sol files.
FRAG_TEST_PATHS = {
    "synthetic-frag": "/tmp/awdnn-test-frag-synthetic.txt",
    "realworld-frag": "/tmp/awdnn-test-frag-realworld.txt",
}
FRAG_TEST_MANIFESTS = {
    "synthetic-frag": "/tmp/awdnn-test-frag-synthetic-manifest.json",
    "realworld-frag": "/tmp/awdnn-test-frag-realworld-manifest.json",
}


def load_samples(dataset: str, seed: int, limit: int | None,
                 split_filter: str | None = None) -> list[dict]:
    """Mirror reep_baseline.py / awdnn earlier behavior. For dataset=='synthetic'
    + split_filter='test' we evaluate on the held-out reentrancy slice.

    For dataset=='synthetic-ext' we read from an external manifest produced by
    build_awdnn_extended_slices.py (200+200 contracts drawn directly from
    sc-ast-injector/data/injected_sc/, restricted to test-only source addresses).

    For dataset in {'synthetic-frag', 'realworld-frag'} we read from the
    manifest produced by build_awdnn_frag_test_slices.py (function-level
    fragments extracted via injection metadata sidecars).
    """
    if dataset in FRAG_TEST_MANIFESTS:
        manifest_path = Path(FRAG_TEST_MANIFESTS[dataset])
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"--test-datasets {dataset} requires {manifest_path}; "
                "run scripts/build_awdnn_frag_test_slices.py first."
            )
        with open(manifest_path) as fh:
            entries = json.load(fh)
        samples = [{"sol_path": e["source"],
                    "contract_label": int(e["label"]),
                    "vulnerability_type": e["vulnerability_type"]}
                   for e in entries
                   if Path(e["source"]).exists()]
        return samples

    if dataset == "synthetic-ext":
        manifest_path = Path(SYNTH_EXT_TEST_MANIFEST)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"--test-datasets synthetic-ext requires {manifest_path}; "
                "run scripts/build_awdnn_extended_slices.py first."
            )
        with open(manifest_path) as fh:
            entries = json.load(fh)
        samples = [{"sol_path": e["source"],
                    "contract_label": int(e["label"]),
                    "vulnerability_type": e["vulnerability_type"]}
                   for e in entries
                   if Path(e["source"]).exists()]
        return samples

    index_path = DATA_ROOT / dataset / "function_level_dataset" / "dataset_index.json"
    with open(index_path) as fh:
        index = json.load(fh)
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
            return Path(c["path"]).stem + ".json"

        contracts = [c for c in contracts if _graph_name(c) in allowed]

    positives = [c for c in contracts if _is_actual_reentrancy(c)]
    negatives = [c for c in contracts if c["vulnerability_type"] == "clean"]
    rng = random.Random(seed)
    negatives = rng.sample(negatives, min(len(positives), len(negatives)))
    selected = positives + negatives

    if limit is not None:
        rng.shuffle(selected)
        selected = selected[:limit]

    samples = []
    for c in selected:
        sol_path = ROOT / c["path"]
        if not sol_path.exists():
            print(f"  WARN: missing sol file {sol_path}", file=sys.stderr)
            continue
        samples.append({
            "sol_path": str(sol_path),
            "contract_label": (1 if c["vulnerability_type"]
                               in REENTRANCY_TYPES else 0),
            "vulnerability_type": c["vulnerability_type"],
        })
    return samples


def write_fragment_file(samples, out_path):
    with open(out_path, "w", encoding="utf-8") as fh:
        for s in samples:
            sol = Path(s["sol_path"])
            text = sol.read_text(encoding="utf-8", errors="replace")
            fh.write(sol.name + "\n")
            for raw in text.splitlines():
                stripped = raw.strip()
                if not stripped:
                    continue
                if stripped.isdigit():
                    stripped = "// " + stripped
                fh.write(stripped + "\n")
            fh.write(f"{s['contract_label']}\n")
            fh.write(SEPARATOR + "\n")


# ---------------------------------------------------------------------------
# Docker invocations
# ---------------------------------------------------------------------------

def _docker_volumes(work_dir: str, train_corpus_host: str | None):
    """Mount work_dir at /work; if a host training-corpus path is given (i.e.
    --train-corpus synthetic), mount its parent so the file is visible.
    Returns the list of -v args."""
    vols = ["-v", f"{work_dir}:/work"]
    if train_corpus_host is not None:
        host_dir = str(Path(train_corpus_host).parent)
        vols += ["-v", f"{host_dir}:{host_dir}"]
    return vols


def run_train_one(image, work_dir, train_file_in_container, ckpt_dir_name,
                  seed, train_corpus_host, timeout_s):
    """Train AWDNN for a single seed; checkpoint written to /work/<ckpt_dir_name>."""
    cmd = [
        "docker", "run", "--rm",
        *_docker_volumes(work_dir, train_corpus_host),
        image,
        "--mode", "train",
        "--train-file", train_file_in_container,
        "--checkpoint-dir", f"/work/{ckpt_dir_name}",
        "--seed", str(seed),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"awdnn train seed={seed} failed (rc={proc.returncode}):\n"
            f"stderr: {proc.stderr[-2000:]}"
        )
    return elapsed


def run_infer_one(image, work_dir, test_file_in_container, ckpt_dir_path_container,
                  out_name, seed, timeout_s, extra_volumes=None):
    """Run inference using a specific checkpoint inside the container."""
    out_host = Path(work_dir) / out_name
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{work_dir}:/work",
        *(extra_volumes or []),
        image,
        "--mode", "infer",
        "--test-file", test_file_in_container,
        "--checkpoint-dir", ckpt_dir_path_container,
        "--output", f"/work/{out_name}",
        "--seed", str(seed),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"awdnn infer seed={seed} failed (rc={proc.returncode}):\n"
            f"stderr: {proc.stderr[-2000:]}"
        )
    with open(out_host) as fh:
        data = json.load(fh)
    data["_elapsed_s"] = elapsed
    data["seed"] = seed
    return data


def aggregate(per_seed_results, samples):
    name_to_sample = {Path(s["sol_path"]).name: s for s in samples}

    per_seed = []
    for r in per_seed_results:
        per_seed.append({
            "seed": r.get("seed"),
            "precision": r.get("precision"),
            "recall": r.get("recall"),
            "f1": r.get("f1"),
            "tp": r.get("tp"), "fp": r.get("fp"),
            "fn": r.get("fn"), "tn": r.get("tn"),
            "elapsed_s": r.get("_elapsed_s"),
        })

    def _safe_stat(fn, xs):
        xs = [x for x in xs if x is not None]
        return fn(xs) if xs else None

    f1s = [r["f1"] for r in per_seed]
    ps = [r["precision"] for r in per_seed]
    rs = [r["recall"] for r in per_seed]

    mean_metrics = {
        "precision_mean": _safe_stat(statistics.mean, ps),
        "precision_stdev": (_safe_stat(statistics.stdev, ps)
                            if len([p for p in ps if p is not None]) > 1 else 0.0),
        "recall_mean": _safe_stat(statistics.mean, rs),
        "recall_stdev": (_safe_stat(statistics.stdev, rs)
                         if len([r for r in rs if r is not None]) > 1 else 0.0),
        "f1_mean": _safe_stat(statistics.mean, f1s),
        "f1_stdev": (_safe_stat(statistics.stdev, f1s)
                     if len([f for f in f1s if f is not None]) > 1 else 0.0),
        "f1_min": min([f for f in f1s if f is not None], default=None),
        "f1_max": max([f for f in f1s if f is not None], default=None),
    }

    per_sample = {}
    for r in per_seed_results:
        seed = r.get("seed")
        for rec in r.get("records", []):
            name = rec["name"]
            if name not in per_sample:
                s = name_to_sample.get(name, {})
                per_sample[name] = {
                    "name": name,
                    "contract_label": rec["label"],
                    "vulnerability_type": s.get("vulnerability_type"),
                    "predictions_by_seed": {},
                }
            per_sample[name]["predictions_by_seed"][str(seed)] = {
                "pred": rec["pred"],
                "proba_bug": rec["proba_bug"],
            }

    return {
        "metrics": {"contract_level_per_seed": per_seed,
                    "contract_level_summary": mean_metrics},
        "records": list(per_sample.values()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run AWDNN cross-dataset baseline (multi-corpus, multi-test)")
    parser.add_argument("--test-datasets", nargs="+",
                        choices=["realworld", "synthetic", "synthetic-ext",
                                 "synthetic-frag", "realworld-frag"],
                        default=["realworld"],
                        help="Run inference on each named test dataset. "
                             "'synthetic-ext' loads from the extended manifest "
                             "produced by build_awdnn_extended_slices.py.")
    parser.add_argument("--train-corpus", choices=list(TRAIN_CORPUS_PATHS),
                        default="authors",
                        help="Which fragment file to train AWDNN on. "
                             "'synthetic-ext' loads from the extended corpus "
                             "produced by build_awdnn_extended_slices.py.")
    parser.add_argument("--reuse-ckpt-dir", default=None,
                        help="Host path to a directory holding ckpt_s{seed} "
                             "subdirectories from a prior run. If set, skip "
                             "training and load each seed's checkpoint from "
                             "this directory. Bind-mounted into the container.")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1, 2, 3, 4, 42])
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Seed for balanced negative draw (matches reep)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR),
                        help="Output dir; one JSON per (corpus,dataset) pair")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Resolve training corpus paths.
    # 'authors' is in-container (always available); host-path corpora must
    # already exist on disk.
    label_map = {
        "authors": "authors-trained",
        "synthetic": "synth-trained",
        "synthetic-ext": "synth-ext-trained",
    }
    if args.train_corpus in ("synthetic", "synthetic-ext"):
        train_corpus_host = TRAIN_CORPUS_PATHS[args.train_corpus]
        if not Path(train_corpus_host).exists():
            sys.exit(f"--train-corpus {args.train_corpus} but "
                     f"{train_corpus_host} not found; run the corresponding "
                     "build script first.")
    else:
        train_corpus_host = None
    train_corpus_label = label_map[args.train_corpus]
    train_file_in_container = TRAIN_CORPUS_PATHS[args.train_corpus]

    # Build test fragment files for every requested dataset.
    work_dir = Path(tempfile.mkdtemp(prefix="awdnn-baseline-"))
    print(f"work dir: {work_dir}")
    test_set_meta = {}  # dataset -> {fragment_file, samples, n_pos, n_neg}
    for ds in args.test_datasets:
        sf = "test" if ds == "synthetic" else None
        samples = load_samples(ds, args.sample_seed, args.limit,
                               split_filter=sf)
        # Sanitize the dataset name for use in filenames (replace - with _)
        safe_name = ds.replace("-", "_")
        pos = sum(1 for s in samples if s["contract_label"] == 1)
        neg = len(samples) - pos
        frag = work_dir / f"{safe_name}.txt"
        if ds in FRAG_TEST_PATHS:
            # Pre-built function-fragment file: copy directly, don't re-extract
            import shutil
            shutil.copy(FRAG_TEST_PATHS[ds], frag)
        else:
            write_fragment_file(samples, frag)
        print(f"  {ds}: {len(samples)} contracts ({pos}+{neg}) -> {frag}")
        test_set_meta[ds] = {
            "fragment_file": frag,
            "samples": samples,
            "n_pos": pos, "n_neg": neg,
        }

    # Per-seed: train (or reuse), then infer on every test dataset.
    per_seed_results = {ds: [] for ds in args.test_datasets}
    train_elapsed = {}
    t_start = time.time()
    for i, seed in enumerate(args.seeds):
        print(f"\n=== seed {seed} ({i+1}/{len(args.seeds)}) ===")

        if args.reuse_ckpt_dir:
            ckpt_dir_host = Path(args.reuse_ckpt_dir) / f"ckpt_s{seed}"
            if not ckpt_dir_host.exists():
                print(f"  [WARN] no ckpt at {ckpt_dir_host}, skipping",
                      file=sys.stderr)
                continue
            # Mount the parent of the ckpt dir into the container
            ckpt_extra_vol = ["-v",
                              f"{ckpt_dir_host.parent}:{ckpt_dir_host.parent}"]
            ckpt_dir_in_container = str(ckpt_dir_host)
            print(f"  reusing checkpoint {ckpt_dir_host}")
        else:
            ckpt_dir_name = f"ckpt_s{seed}"
            ckpt_dir_host = work_dir / ckpt_dir_name
            if ckpt_dir_host.exists():
                shutil.rmtree(ckpt_dir_host)
            try:
                t = run_train_one(args.image, str(work_dir),
                                  train_file_in_container, ckpt_dir_name, seed,
                                  train_corpus_host, args.timeout)
                train_elapsed[seed] = t
                print(f"  train: {t:.1f}s")
            except Exception as exc:
                print(f"  [ERROR] train seed {seed}: {exc}", file=sys.stderr)
                continue
            ckpt_extra_vol = []
            ckpt_dir_in_container = f"/work/{ckpt_dir_name}"

        for ds in args.test_datasets:
            meta = test_set_meta[ds]
            test_in_container = f"/work/{meta['fragment_file'].name}"
            try:
                r = run_infer_one(args.image, str(work_dir),
                                  test_in_container, ckpt_dir_in_container,
                                  f"infer_{ds}_s{seed}.json", seed,
                                  args.timeout, extra_volumes=ckpt_extra_vol)
                per_seed_results[ds].append(r)
                print(f"  infer {ds}: F1={r.get('f1', 0):.4f} "
                      f"P={r.get('precision', 0):.4f} "
                      f"R={r.get('recall', 0):.4f} "
                      f"t={r['_elapsed_s']:.1f}s")
            except Exception as exc:
                print(f"  [ERROR] infer {ds} seed {seed}: {exc}",
                      file=sys.stderr)

    total = time.time() - t_start

    # Write one JSON per (train_corpus, test_dataset) pair.
    for ds in args.test_datasets:
        agg = aggregate(per_seed_results[ds], test_set_meta[ds]["samples"])
        ds_safe = ds.replace("-", "_")
        out_path = (Path(args.output_dir) /
                    f"awdnn_{ds_safe}_{train_corpus_label.replace('-', '_')}.json")
        output = {
            "test_dataset": ds,
            "train_corpus": train_corpus_label,
            "train_corpus_path": train_file_in_container,
            "image": args.image,
            "seeds": args.seeds,
            "sample_seed": args.sample_seed,
            "n_contracts": len(test_set_meta[ds]["samples"]),
            "n_positive": test_set_meta[ds]["n_pos"],
            "n_negative": test_set_meta[ds]["n_neg"],
            "hyperparameters": ("AWDNN driver defaults (authors' logged "
                                "config): lr=1e-2, batch_size=64, "
                                "vec_length=100, epochs=10, attn_units=8, "
                                "Adam optimizer"),
            "train_elapsed_s_per_seed": train_elapsed,
            "total_elapsed_s": total,
            **agg,
        }
        with open(out_path, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"\nResults saved: {out_path}")
        s = agg["metrics"]["contract_level_summary"]
        if s.get("f1_mean") is not None:
            print(f"  F1: {s['f1_mean']:.4f} ± {s['f1_stdev']:.4f} "
                  f"[{s['f1_min']:.4f}, {s['f1_max']:.4f}]")
            print(f"  P:  {s['precision_mean']:.4f} ± {s['precision_stdev']:.4f}")
            print(f"  R:  {s['recall_mean']:.4f} ± {s['recall_stdev']:.4f}")

    print(f"\nWork dir retained: {work_dir}")


if __name__ == "__main__":
    main()
