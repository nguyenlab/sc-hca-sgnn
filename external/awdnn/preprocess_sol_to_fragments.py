#!/usr/bin/env python3
# Convert a list of .sol files + contract-level labels into the AWDNN
# "fragment file" format expected by config/fragment_vectorizer.py:
#
#   BASENAME.sol
#   <stripped source line 1>
#   <stripped source line 2>
#   ...
#   0|1
#   ----------------------------------------
#   BASENAME2.sol
#   ...
#
# One fragment = one .sol file. Blank lines are dropped to match upstream's
# parse_file behavior (AWDNNA.py skips blanks anyway, but dropping them here
# keeps the output file smaller and less confusing to diff against).

import argparse
import json
from pathlib import Path


SEPARATOR = "-" * 40


def write_fragment(fh, basename, source_text, label):
    fh.write(f"{basename}\n")
    for raw in source_text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        # Guard against source lines that are bare digits (would confuse the
        # upstream parse_file's label detector). If encountered, prefix with a
        # comment marker so the tokenizer sees a distinct token.
        if stripped.isdigit():
            stripped = "// " + stripped
        fh.write(f"{stripped}\n")
    fh.write(f"{label}\n")
    fh.write(f"{SEPARATOR}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .sol files to AWDNN fragment file format",
    )
    parser.add_argument(
        "--positive-dir", required=True,
        help="Directory of reentrancy-vulnerable .sol files (label=1)",
    )
    parser.add_argument(
        "--negative-dir", required=True,
        help="Directory of clean .sol files (label=0)",
    )
    parser.add_argument(
        "--output", required=True, help="Output fragment file path",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Optional JSON manifest listing (basename, label, source_path)",
    )
    args = parser.parse_args()

    positives = sorted(Path(args.positive_dir).glob("*.sol"))
    negatives = sorted(Path(args.negative_dir).glob("*.sol"))

    print(f"  positives: {len(positives)} in {args.positive_dir}")
    print(f"  negatives: {len(negatives)} in {args.negative_dir}")

    manifest = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for src in positives:
            text = src.read_text(encoding="utf-8", errors="replace")
            write_fragment(fh, src.name, text, 1)
            manifest.append({"name": src.name, "label": 1,
                             "source": str(src)})
        for src in negatives:
            text = src.read_text(encoding="utf-8", errors="replace")
            write_fragment(fh, src.name, text, 0)
            manifest.append({"name": src.name, "label": 0,
                             "source": str(src)})

    print(f"  wrote {len(manifest)} fragments -> {args.output}")

    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"  manifest  -> {args.manifest}")


if __name__ == "__main__":
    main()
