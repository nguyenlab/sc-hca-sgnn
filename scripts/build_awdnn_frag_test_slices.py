#!/usr/bin/env python3
"""
Build function-level AWDNN fragment test files from dataset_index.json,
using the same extraction logic as build_awdnn_extended_slices.py.

For reentrancy contracts: extract the enclosing function of the injected
vulnerable region (via the per-contract JSON sidecar).
For clean contracts: extract a random function.

Outputs:
  /tmp/awdnn-test-frag-synthetic.txt   — 15 re + 15 clean fragments
  /tmp/awdnn-test-frag-realworld.txt   — 31 re + 31 clean fragments
  /tmp/awdnn-test-frag-synthetic-manifest.json
  /tmp/awdnn-test-frag-realworld-manifest.json
"""
import json
import random
import re
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
    js = (ROOT / c["path"]).with_suffix(".json")
    if not js.exists():
        return False
    try:
        md = json.load(open(js))
        return md.get("template_name", "").startswith("reentrancy_")
    except Exception:
        return False


def _balance_braces(text, start, end):
    n = len(text)
    fn_idx = text.rfind("function ", 0, max(start, 0))
    open_idx = text.find("{", fn_idx) if fn_idx >= 0 else -1
    if open_idx < 0 or open_idx > end:
        return max(0, start), min(n, end)
    depth = 0
    close_idx = open_idx
    for i in range(open_idx, n):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    return max(0, fn_idx), min(n, close_idx + 1)


def extract_injection_function(sol_path):
    sol = Path(sol_path)
    text = sol.read_text(encoding="utf-8", errors="replace")
    js = sol.with_suffix(".json")
    if not js.exists():
        return text
    try:
        md = json.load(open(js))
    except Exception:
        return text
    regions = md.get("injected_regions", [])
    vc = [r for r in regions if r.get("component") == "vulnerable_code"]
    if not vc:
        return text
    start = vc[0]["start_byte"]
    end = vc[0]["end_byte"]
    s, e = _balance_braces(text, start, end)
    return text[s:e]


_FUNC_RE = re.compile(r"\bfunction\s+\w+\s*\(", re.MULTILINE)


def extract_random_function(sol_path, rng):
    sol = Path(sol_path)
    text = sol.read_text(encoding="utf-8", errors="replace")
    matches = list(_FUNC_RE.finditer(text))
    if not matches:
        return text
    m = rng.choice(matches)
    fn_start = m.start()
    open_idx = text.find("{", fn_start)
    if open_idx < 0:
        return text
    depth, close_idx = 0, open_idx
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    return text[fn_start:close_idx + 1]


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


def load_index_contracts(dataset, split_filter=None, seed=42):
    index_path = ROOT / "data" / dataset / "function_level_dataset" / "dataset_index.json"
    idx = json.load(open(index_path))
    contracts = idx["contracts"]

    if split_filter is not None:
        sm_path = ROOT / "data" / "synthetic-split" / "split_metadata.json"
        sm = json.load(open(sm_path))
        allowed = set(sm["splits"][split_filter]["templates"])
        def graph_name(c):
            return Path(c["path"]).stem + ".json"
        contracts = [c for c in contracts if graph_name(c) in allowed]

    positives = [c for c in contracts if _is_actual_reentrancy(c)]
    negatives = [c for c in contracts if c["vulnerability_type"] == "clean"]
    rng = random.Random(seed)
    negatives = rng.sample(negatives, min(len(positives), len(negatives)))
    return positives, negatives


def build_frag_slice(dataset, split_filter, out_txt, out_manifest, seed=42):
    positives, negatives = load_index_contracts(dataset, split_filter, seed)
    rng_clean = random.Random(seed + 1)

    manifest = []
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(out_txt, "w", encoding="utf-8") as fh:
        for c in positives:
            sol = ROOT / c["path"]
            text = extract_injection_function(sol)
            write_fragment(fh, sol.name, text, 1)
            manifest.append({"name": sol.name, "label": 1,
                             "vulnerability_type": c["vulnerability_type"],
                             "source": str(sol),
                             "fragment_bytes": len(text)})
        for c in negatives:
            sol = ROOT / c["path"]
            text = extract_random_function(sol, rng_clean)
            write_fragment(fh, sol.name, text, 0)
            manifest.append({"name": sol.name, "label": 0,
                             "vulnerability_type": "clean",
                             "source": str(sol),
                             "fragment_bytes": len(text)})

    with open(out_manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)

    n_re = sum(1 for e in manifest if e["label"] == 1)
    n_cl = sum(1 for e in manifest if e["label"] == 0)
    sizes = [e["fragment_bytes"] for e in manifest]
    sizes.sort()
    print(f"[{dataset}] {n_re} re + {n_cl} clean  "
          f"median={sizes[len(sizes)//2]}B  max={sizes[-1]}B  -> {out_txt}")


if __name__ == "__main__":
    build_frag_slice(
        dataset="synthetic",
        split_filter="test",
        out_txt="/tmp/awdnn-test-frag-synthetic.txt",
        out_manifest="/tmp/awdnn-test-frag-synthetic-manifest.json",
    )
    build_frag_slice(
        dataset="realworld",
        split_filter=None,
        out_txt="/tmp/awdnn-test-frag-realworld.txt",
        out_manifest="/tmp/awdnn-test-frag-realworld-manifest.json",
    )
