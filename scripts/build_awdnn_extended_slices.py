#!/usr/bin/env python3
"""
Build expanded AWDNN train + test slices from the original sc-ast-injector
pool, restricted to source-address discipline of synthetic-split:

  - train slice: contracts whose source ∈ train-only (i.e., in train split,
                 not in test, not in dev). Sample N_TRAIN reentrancy + N_TRAIN
                 clean from this pool.
  - test slice:  contracts whose source ∈ test-only. Sample N_TEST reentrancy
                 + N_TEST clean.

Each fragment we emit is a *function-level slice*, not a whole contract: this
matches the format of the authors' `contracts_re.txt` (393 hand-curated
function fragments) and works around AWDNN's hardcoded 100-token input window.
For reentrancy positives, the slice is the enclosing function of the injected
vulnerable-code region (located via the per-contract injection metadata
sidecar). For clean negatives, we pick a random function from the contract.

Outputs:
  /tmp/awdnn-synth-train-ext/contracts.txt      — train fragment file
  /tmp/awdnn-synth-train-ext/manifest.json      — train provenance
  /tmp/awdnn-synth-test-ext/contracts.txt       — test fragment file
  /tmp/awdnn-synth-test-ext/manifest.json       — test provenance
"""
import argparse
import json
import random
import re
from pathlib import Path

INJ = Path('/home/minhnn/jaist-smb/jaist-s2505/s2520419/workspace/overleaf/'
           'tosem-response-letter-final/sc-ast-injector/data/injected_sc')
WILD = Path('/home/minhnn/jaist-smb/jaist-s2505/s2520419/workspace/overleaf/'
            'tosem-response-letter-final/sc-ast-injector/data/'
            'smartbugs-wild-clean-contracts')
ROOT = Path(__file__).resolve().parent.parent
SEPARATOR = "-" * 40


def src_addr(name):
    return name.replace('.json', '').replace('.sol', '').split('_')[0]


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


def _balance_braces(text, start, end):
    """Expand the [start, end) byte range to a balanced-brace block by walking
    forward to the closing brace and backward to the opening brace + the
    `function NAME(...)` signature. Returns (new_start, new_end) clipped to
    [0, len(text)]."""
    n = len(text)
    # Walk backward from start to find a 'function ' keyword.
    fn_idx = text.rfind("function ", 0, max(start, 0))
    open_idx = text.find("{", fn_idx) if fn_idx >= 0 else -1
    if open_idx < 0 or open_idx > end:
        return max(0, start), min(n, end)
    # Walk forward from open_idx counting braces.
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
    """For a reentrancy-injected contract, return the source-text slice
    corresponding to the function that contains the injection. Falls back to
    the whole contract if metadata is missing."""
    sol = Path(sol_path)
    text = sol.read_text(encoding='utf-8', errors='replace')
    js = sol.with_suffix('.json')
    if not js.exists():
        return text
    try:
        md = json.load(open(js))
    except Exception:
        return text
    regions = md.get('injected_regions', [])
    vc = [r for r in regions if r.get('component') == 'vulnerable_code']
    if not vc:
        return text
    start = vc[0]['start_byte']
    end = vc[0]['end_byte']
    s, e = _balance_braces(text, start, end)
    return text[s:e]


_FUNC_RE = re.compile(r"\bfunction\s+\w+\s*\(", re.MULTILINE)


def extract_random_function(sol_path, rng):
    """For a clean contract, pick a random function and return its source slice.
    Falls back to the whole contract if no function is found."""
    sol = Path(sol_path)
    text = sol.read_text(encoding='utf-8', errors='replace')
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


def build_slice(label, sources_set, n_re, n_clean, seed, out_dir):
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    re_pool = [f for f in (INJ / 'point').glob('*reentrancy*.sol')
               if src_addr(f.name) in sources_set]
    clean_pool = [WILD / f"{a}.sol" for a in sorted(sources_set)
                  if (WILD / f"{a}.sol").exists()]

    n_re_actual = min(n_re, len(re_pool))
    n_clean_actual = min(n_clean, len(clean_pool))
    re_chosen = rng.sample(re_pool, n_re_actual)
    clean_chosen = rng.sample(clean_pool, n_clean_actual)

    print(f"  [{label}] reentrancy pool={len(re_pool)} sampled={n_re_actual}")
    print(f"  [{label}] clean pool={len(clean_pool)} sampled={n_clean_actual}")

    frag_path = out_dir / "contracts.txt"
    manifest = []
    sizes = []
    rng_clean = random.Random(seed + 1)
    with open(frag_path, "w", encoding="utf-8") as fh:
        for sol in re_chosen:
            text = extract_injection_function(sol)
            sizes.append(len(text))
            write_fragment(fh, sol.name, text, 1)
            manifest.append({"name": sol.name, "label": 1,
                             "vulnerability_type": "reentrancy",
                             "source": str(sol),
                             "src_addr": src_addr(sol.name),
                             "fragment_bytes": len(text)})
        for sol in clean_chosen:
            text = extract_random_function(sol, rng_clean)
            sizes.append(len(text))
            write_fragment(fh, sol.name, text, 0)
            manifest.append({"name": sol.name, "label": 0,
                             "vulnerability_type": "clean",
                             "source": str(sol),
                             "src_addr": src_addr(sol.name),
                             "fragment_bytes": len(text)})
    sizes.sort()
    print(f"  [{label}] fragment-byte median={sizes[len(sizes)//2]}  "
          f"mean={sum(sizes)/len(sizes):.0f}  min={sizes[0]}  max={sizes[-1]}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  [{label}] wrote {len(manifest)} fragments -> {frag_path}")
    print(f"  [{label}] manifest -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-re", type=int, default=1000)
    parser.add_argument("--n-train-clean", type=int, default=1000)
    parser.add_argument("--n-test-re", type=int, default=200)
    parser.add_argument("--n-test-clean", type=int, default=200)
    args = parser.parse_args()

    sm = json.load(open(ROOT / "data/synthetic-split/split_metadata.json"))
    train_srcs = set(src_addr(t) for t in sm["splits"]["train"]["templates"])
    test_srcs  = set(src_addr(t) for t in sm["splits"]["test"]["templates"])
    dev_srcs   = set(src_addr(t) for t in sm["splits"]["dev"]["templates"])
    train_only = train_srcs - test_srcs - dev_srcs
    test_only  = test_srcs  - train_srcs - dev_srcs

    print(f"train-only sources: {len(train_only)}, "
          f"test-only sources: {len(test_only)}")

    build_slice("train", train_only, args.n_train_re, args.n_train_clean,
                args.seed, "/tmp/awdnn-synth-train-ext")
    build_slice("test",  test_only,  args.n_test_re,  args.n_test_clean,
                args.seed, "/tmp/awdnn-synth-test-ext")


if __name__ == "__main__":
    main()
