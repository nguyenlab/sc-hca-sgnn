# AWDNN baseline reproduction

Reproducible build + driver for AWDNN (Osei, Huang, Ma; J. Syst. Software 223,
112361; 2025). Used as an additional reentrancy-specialized baseline for the
TOSEM R2 revision (alongside ReEP, in `../reep/`).

## Upstream repo

- **URL:** https://github.com/sbanning/AWDNN (author's GitHub handle is
  `sbanning`, initials of Samuel Banning — surname-based searches will miss
  it).
- **Pinned commit:** `b1c6dc3be13043dd89b799d597452bc288d54e8f`.

## README / code / paper / logs discrepancies

The upstream repo has a number of internal inconsistencies that are relevant
when interpreting AWDNN's reported numbers:

| | Paper (§ 4.2, Tables 4–5) | `arg_parser.py` defaults | Authors' shipped logs (`evaluations/wdnna/reentrancy/smartoutput_*.log`) |
|-|-|-|-|
| Optimizer | SGD | Adam (SGD line commented out in `widennet_att.py:80-81`) | Adam |
| Learning rate | 1e-4 | 1e-3 | **1e-2** |
| Batch size | 32 | 16 | **64** |
| Vector length | 150 | 150 | **100** |
| Epochs | (unspecified) | 20 | **10** |
| Attention units | 64 | 8 (hardcoded in `widennet_att.py:72`) | 8 |

Reproduction F1 on the authors' own `contracts_re.txt` (5 logged runs):
0.843, 0.875, 0.867, 0.857, 0.847. **Mean = 0.858; max = 0.875.** The paper
reports F1 = 0.950, leaving a ~9 F1-point unreproduced gap.

README also misnames three files (main entry is `AWDNNA.py` not `AWDNN.py`;
model is `config/models/widennet_att.py`, not `wdnn_att.py`; attention is
`config/attention/Custom_Attention.py` with a capital C).

`contracts_re.txt` contains 393 fragments (200 clean, 193 reentrancy), not
the full Messi-Q/Smart-Contract-Dataset — it is a curated subset used by the
authors for their experiments.

## Layout

- `Dockerfile` — base on `tensorflow/tensorflow:2.9.0`, pin deps, clone
  upstream, copy our driver.
- `awdnn_driver.py` — four modes:
  - `reproduce`: train + test via internal 80/20 split (mirrors AWDNNA.py
    with configurable hyperparameters).
  - `train`: train on `--train-file`, persist Word2Vec embeddings + Keras
    weights to `--checkpoint-dir`.
  - `infer`: load checkpoint, predict on `--test-file`.
  - `cross`: train → infer in one invocation. Used for cross-dataset
    evaluation (train on `contracts_re.txt`, infer on our SmartBugs 31+31).
- `preprocess_sol_to_fragments.py` — convert a directory of `.sol` files
  (positive/negative subdirs) into the AWDNN fragment file format.
- `smoke_test.sh` — build + quick sanity runs.

## Build

```
docker build -t awdnn:smoke -f Dockerfile .
```

## Run (examples)

Reproduce the authors' logs on `contracts_re.txt`:

```
docker run --rm -v $PWD:/work awdnn:smoke \
    --mode reproduce \
    --train-file /opt/awdnn-src/contracts_re.txt \
    --output /work/reproduce_metrics.json
```

Cross-dataset: train on `contracts_re.txt`, evaluate on our SmartBugs 62:

```
# 1) preprocess SmartBugs
python preprocess_sol_to_fragments.py \
    --positive-dir ../../data/realworld/sc-source/vulnerable/reentrancy \
    --negative-dir ../../data/realworld/sc-source/clean \
    --output smartbugs_re.txt

# 2) train + infer (Docker)
docker run --rm -v $PWD:/work awdnn:smoke \
    --mode cross \
    --train-file /opt/awdnn-src/contracts_re.txt \
    --test-file /work/smartbugs_re.txt \
    --checkpoint-dir /work/ckpt \
    --output /work/crosseval_metrics.json
```
