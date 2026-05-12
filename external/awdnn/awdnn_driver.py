#!/usr/bin/env python3
# AWDNN driver: adds train / infer / cross / reproduce modes on top of the
# upstream model (sbanning/AWDNN @ b1c6dc3). The upstream AWDNNA.py hardcodes
# train+test in one pass via an internal 80/20 split and does not persist
# weights, which blocks cross-dataset evaluation. This driver re-implements
# the same model architecture (verbatim from config/models/widennet_att.py)
# with save/load + four explicit modes.
#
# Hyperparameter defaults reproduce the authors' own logged runs
# (evaluations/wdnna/reentrancy/smartoutput_*.log): lr=1e-2, vec_length=100,
# batch_size=64, epochs=10, Adam. The paper claims lr=1e-4, batch=32, SGD,
# attn=64, vec=150 — override via flags if you want to try reproducing the
# paper's numbers instead of the authors' logs.

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    Normalization,
)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------------
# Fragment parsing (mirrors AWDNNA.py:parse_file, but collects the filename
# header line and emits it alongside each fragment for downstream matching).
# ---------------------------------------------------------------------------

SEPARATOR_MARKER = "-" * 40


def parse_fragments(filename):
    """Yield (name, fragment_lines, label) tuples from an AWDNN fragment file.

    The file format is:
        NAME.sol
        <source line>
        <source line>
        ...
        0|1
        ----------------------------------------
    Blank lines are skipped (same as AWDNNA.py).
    """
    with open(filename, "r", encoding="utf8") as fh:
        name = None
        frag = []
        label = 0
        first_nonblank = True
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            # Match separator as a whole line, not a substring. Upstream's
            # `"-" * 40 in line` check mis-fires on source comments like
            # "// ---------- Section ----------" that contain 40+ dashes.
            if stripped == SEPARATOR_MARKER and frag:
                yield name, frag, label
                name = None
                frag = []
                label = 0
                first_nonblank = True
                continue
            # Heuristic used by AWDNNA.py: a line whose first token is a digit
            # is treated as the label (if we already have a non-empty fragment)
            first_tok = stripped.split()[0]
            if first_tok.isdigit() and frag:
                if stripped.isdigit():
                    label = int(stripped)
                else:
                    frag.append(stripped)
            else:
                if first_nonblank:
                    # First non-blank line of a fragment is the filename header
                    name = stripped
                    first_nonblank = False
                else:
                    frag.append(stripped)
        if frag:
            yield name, frag, label


# ---------------------------------------------------------------------------
# Fragment vectorizer — save/load wrapper around Word2Vec (CBOW).
# Token/fragment logic is a verbatim port of config/fragment_vectorizer.py.
# ---------------------------------------------------------------------------

_OP3 = {"<<=", ">>="}
_OP2 = {
    "->", "++", "--", "!~", "<<", ">>", "<=", ">=",
    "==", "!=", "&&", "||", "+=", "-=", "*=", "/=",
    "%=", "&=", "^=", "|=",
}
_OP1 = {
    "(", ")", "[", "]", ".", "+", "-", "*", "&", "/",
    "%", "<", ">", "^", "|", "=", ",", "?", ":", ";", "{", "}",
}


def _tokenize_line(line):
    tmp, w = [], []
    i = 0
    while i < len(line):
        if line[i] == " ":
            tmp.append("".join(w))
            tmp.append(line[i])
            w = []
            i += 1
        elif line[i : i + 3] in _OP3:
            tmp.append("".join(w))
            tmp.append(line[i : i + 3])
            w = []
            i += 3
        elif line[i : i + 2] in _OP2:
            tmp.append("".join(w))
            tmp.append(line[i : i + 2])
            w = []
            i += 2
        elif line[i] in _OP1:
            tmp.append("".join(w))
            tmp.append(line[i])
            w = []
            i += 1
        else:
            w.append(line[i])
            i += 1
    if w:
        tmp.append("".join(w))
    tmp = [t for t in tmp if t not in ("", " ")]
    return tmp


def tokenize_fragment(fragment_lines):
    """Return (tokens, backwards_slice_flag).

    backwards_slice is True if any line contains a token matching /function\\d+/
    (ported from config/fragment_vectorizer.py).
    """
    import re

    fn_re = re.compile(r"function(\d)+")
    tokens = []
    backwards = False
    for line in fragment_lines:
        line_tokens = _tokenize_line(line)
        tokens += line_tokens
        if any(fn_re.match(t) for t in line_tokens):
            backwards = True
    return tokens, backwards


def fit_word2vec(all_fragments_tokens, vec_length):
    """CBOW Word2Vec, min_count=1 (upstream defaults)."""
    model = Word2Vec(
        all_fragments_tokens, min_count=1, vector_size=vec_length, sg=0
    )
    return model.wv


def vectorize_fragment(tokens, backwards, embeddings, vec_length):
    """Produce a (100, vec_length) matrix. OOV tokens → zero vector."""
    vectors = np.zeros(shape=(100, vec_length), dtype=np.float32)
    n = min(len(tokens), 100)
    if backwards:
        for i in range(n):
            tok = tokens[len(tokens) - 1 - i]
            if tok in embeddings:
                vectors[100 - 1 - i] = embeddings[tok]
    else:
        for i in range(n):
            tok = tokens[i]
            if tok in embeddings:
                vectors[i] = embeddings[tok]
    return vectors


# ---------------------------------------------------------------------------
# Model — verbatim architecture from config/models/widennet_att.py, with
# optimizer selectable and attention units configurable (upstream hardcodes
# `units=8` despite the paper claiming 64).
# ---------------------------------------------------------------------------


class CustomAttention(Layer):
    """Ported from config/attention/Custom_Attention.py (drops the
    module-load-time argparse dependency)."""

    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.feature_dim = input_shape[-1]
        self.W_q = self.add_weight(
            name="W_q", shape=(self.feature_dim, self.units),
            initializer="glorot_uniform", trainable=True,
        )
        self.W_k = self.add_weight(
            name="W_k", shape=(self.feature_dim, self.units),
            initializer="glorot_uniform", trainable=True,
        )
        self.W_v = self.add_weight(
            name="W_v", shape=(self.feature_dim, self.units),
            initializer="glorot_uniform", trainable=True,
        )
        self.dropout = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, **_):
        q = tf.matmul(inputs, tf.expand_dims(self.W_q, axis=0))
        k = tf.matmul(inputs, tf.expand_dims(self.W_k, axis=0))
        v = tf.matmul(inputs, tf.expand_dims(self.W_v, axis=0))
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.nn.softmax(attn)
        return tf.matmul(attn, v)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "dropout_rate": self.dropout_rate})
        return cfg


def build_awdnn_model(wide_shape, deep_shape, lr, optimizer_name,
                      attention_units, dropout_rate):
    """Port of WIDENNET_Attention.build_model with configurable optimizer +
    attention units."""
    input_wide = Input(shape=wide_shape)
    input_deep = Input(shape=deep_shape)

    wide = Normalization()(input_wide)
    deep = Normalization()(input_deep)

    h1 = Dense(192, activation="relu")(deep)
    h2 = Dense(320, activation="relu")(h1)
    attn = CustomAttention(units=attention_units,
                           dropout_rate=dropout_rate)(h2)

    merged = Concatenate(axis=-1)([wide, attn])
    flat = Flatten()(merged)
    out = Dense(2, activation="softmax")(flat)

    model = Model(inputs=[input_wide, input_deep], outputs=out)

    if optimizer_name.lower() == "sgd":
        opt = SGD(lr)
    else:
        opt = Adam(lr)
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Helpers for the pipeline
# ---------------------------------------------------------------------------


def prepare_vectors(fragments, embeddings, vec_length):
    """fragments: list of (name, tokens, backwards, label).
    Return (names, X, y) where X has shape (N, 100, vec_length)."""
    names, X_list, y_list = [], [], []
    for name, tokens, backwards, label in fragments:
        vec = vectorize_fragment(tokens, backwards, embeddings, vec_length)
        names.append(name)
        X_list.append(vec)
        y_list.append(label)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return names, X, y


def tokenize_all(filename):
    """Return list of (name, tokens, backwards, label) for a fragment file."""
    out = []
    for name, frag_lines, label in parse_fragments(filename):
        toks, back = tokenize_fragment(frag_lines)
        out.append((name, toks, back, label))
    return out


def split_wide_deep(X):
    # Upstream widennet_att.py does x_train[:, :50] = wide, x_train[:, 50:] = deep.
    return X[:, :50], X[:, 50:]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_reproduce(args):
    """Mirror the upstream AWDNNA.py behavior (train+test with internal
    80/20 split) but with configurable hyperparameters and deterministic
    seeding. Writes metrics JSON."""
    tokenized = tokenize_all(args.train_file)
    print(f"  parsed {len(tokenized)} fragments from {args.train_file}")

    # Train Word2Vec on all fragments (matches upstream).
    all_tokens = [t[1] for t in tokenized]
    embeddings = fit_word2vec(all_tokens, args.vec_length)
    print(f"  trained Word2Vec: {len(embeddings.index_to_key)} tokens")

    _, X, y = prepare_vectors(tokenized, embeddings, args.vec_length)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed,
    )
    x_train_wide, x_train_deep = split_wide_deep(x_train)
    x_test_wide, x_test_deep = split_wide_deep(x_test)

    y_train_c = to_categorical(y_train, num_classes=2)
    y_test_c = to_categorical(y_test, num_classes=2)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=y)
    cw = {i: w for i, w in enumerate(class_weights)}

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    model = build_awdnn_model(
        wide_shape=x_train_wide.shape[1:], deep_shape=x_train_deep.shape[1:],
        lr=args.lr, optimizer_name=args.optimizer,
        attention_units=args.attention_units, dropout_rate=args.dropout,
    )
    # Input order in upstream .fit call is [deep, wide], matching build_model
    # being called with [input_wide, input_deep] in its inputs list. We match
    # the upstream call order exactly.
    model.fit(
        [x_train_deep, x_train_wide], y_train_c,
        epochs=args.epochs, class_weight=cw, verbose=1,
        batch_size=args.batch_size,
    )

    loss, acc = model.evaluate(
        [x_test_deep, x_test_wide], y_test_c,
        batch_size=args.batch_size, verbose=0,
    )
    preds = model.predict(
        [x_test_deep, x_test_wide],
        batch_size=args.batch_size, verbose=0,
    ).round()
    y_true = np.argmax(y_test_c, axis=1)
    y_pred = np.argmax(preds, axis=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    metrics = {
        "mode": "reproduce",
        "seed": args.seed,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "accuracy": float(acc),
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "fpr": float(fpr), "fnr": float(fnr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "hyperparameters": {
            "lr": args.lr, "batch_size": args.batch_size,
            "epochs": args.epochs, "vec_length": args.vec_length,
            "attention_units": args.attention_units,
            "optimizer": args.optimizer, "dropout": args.dropout,
        },
    }
    _dump_metrics(metrics, args.output)
    return metrics


def _save_checkpoint(checkpoint_dir, embeddings, model, config):
    ckpt = Path(checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    embeddings.save(str(ckpt / "word2vec.kv"))
    model.save_weights(str(ckpt / "awdnn_weights.weights.h5"))
    with open(ckpt / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)


def _load_checkpoint(checkpoint_dir):
    ckpt = Path(checkpoint_dir)
    embeddings = KeyedVectors.load(str(ckpt / "word2vec.kv"))
    with open(ckpt / "config.json") as fh:
        config = json.load(fh)
    return embeddings, config


def mode_train(args):
    tokenized = tokenize_all(args.train_file)
    print(f"  parsed {len(tokenized)} fragments from {args.train_file}")
    all_tokens = [t[1] for t in tokenized]
    embeddings = fit_word2vec(all_tokens, args.vec_length)
    print(f"  trained Word2Vec: {len(embeddings.index_to_key)} tokens")

    _, X, y = prepare_vectors(tokenized, embeddings, args.vec_length)
    x_wide, x_deep = split_wide_deep(X)
    y_c = to_categorical(y, num_classes=2)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=y)
    cw = {i: w for i, w in enumerate(class_weights)}

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    model = build_awdnn_model(
        wide_shape=x_wide.shape[1:], deep_shape=x_deep.shape[1:],
        lr=args.lr, optimizer_name=args.optimizer,
        attention_units=args.attention_units, dropout_rate=args.dropout,
    )
    model.fit(
        [x_deep, x_wide], y_c,
        epochs=args.epochs, class_weight=cw, verbose=1,
        batch_size=args.batch_size,
    )

    config = {
        "vec_length": args.vec_length,
        "attention_units": args.attention_units,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "n_train": int(len(y)),
        "seed": args.seed,
    }
    _save_checkpoint(args.checkpoint_dir, embeddings, model, config)
    print(f"  checkpoint saved to {args.checkpoint_dir}")
    metrics = {"mode": "train", **config}
    _dump_metrics(metrics, args.output)
    return metrics


def mode_infer(args):
    embeddings, config = _load_checkpoint(args.checkpoint_dir)
    vec_length = config["vec_length"]

    tokenized = tokenize_all(args.test_file)
    print(f"  parsed {len(tokenized)} fragments from {args.test_file}")

    names, X, y = prepare_vectors(tokenized, embeddings, vec_length)
    x_wide, x_deep = split_wide_deep(X)
    y_c = to_categorical(y, num_classes=2)

    model = build_awdnn_model(
        wide_shape=x_wide.shape[1:], deep_shape=x_deep.shape[1:],
        lr=config["lr"], optimizer_name=config["optimizer"],
        attention_units=config["attention_units"],
        dropout_rate=config["dropout"],
    )
    # Build variables by calling the model once
    _ = model.predict([x_deep[:1], x_wide[:1]], verbose=0)
    model.load_weights(str(Path(args.checkpoint_dir)
                           / "awdnn_weights.weights.h5"))

    preds = model.predict(
        [x_deep, x_wide], batch_size=config["batch_size"], verbose=0,
    )
    y_pred = np.argmax(preds.round(), axis=1)
    y_true = np.argmax(y_c, axis=1)

    # Write per-contract predictions (parallel to the fragment file)
    records = []
    for name, yt, yp, proba in zip(names, y_true, y_pred, preds):
        records.append({
            "name": name,
            "label": int(yt),
            "pred": int(yp),
            "proba_clean": float(proba[0]),
            "proba_bug": float(proba[1]),
        })

    # Metrics if labels are present (label != -1 convention would be useful
    # but fragment format doesn't carry "unlabeled"; if all labels are 0 we
    # still compute metrics and let the caller interpret).
    if len(set(y_true.tolist())) >= 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        metrics = {
            "mode": "infer", "n": int(len(y_true)),
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "fpr": float(fpr), "fnr": float(fnr),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
    else:
        metrics = {"mode": "infer", "n": int(len(y_true)),
                   "note": "single-class test set — no P/R/F1"}

    metrics["records"] = records
    _dump_metrics(metrics, args.output)
    return metrics


def mode_cross(args):
    # Train on --train-file, then infer on --test-file in one invocation.
    # Saves checkpoint; rewrites the test metrics into args.output.
    train_args = argparse.Namespace(**vars(args))
    train_args.output = None  # we will write at infer step
    mode_train(train_args)
    # Reuse the same checkpoint for inference
    infer_args = argparse.Namespace(**vars(args))
    return mode_infer(infer_args)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _dump_metrics(metrics, output):
    if output is None:
        print(json.dumps(metrics, indent=2))
        return
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  metrics -> {output}")
    # Summary to stdout
    summary_keys = ("mode", "accuracy", "precision", "recall", "f1",
                    "n", "n_train", "n_test")
    print({k: metrics[k] for k in summary_keys if k in metrics})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="AWDNN driver with train / infer / cross / reproduce modes",
    )
    parser.add_argument("--mode", choices=["train", "infer", "cross",
                                            "reproduce"], required=True)
    parser.add_argument("--train-file", help="Fragment file for training "
                                             "(reproduce, train, cross)")
    parser.add_argument("--test-file", help="Fragment file for inference "
                                            "(infer, cross)")
    parser.add_argument("--checkpoint-dir", help="Directory for Word2Vec + "
                                                  "Keras weights")
    parser.add_argument("--output", help="JSON metrics output path")
    parser.add_argument("--seed", type=int, default=42)
    # Hyperparameters — defaults match the authors' logged runs, NOT the
    # paper. Override for paper-spec runs.
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--vec-length", type=int, default=100)
    parser.add_argument("--attention-units", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--optimizer", choices=["adam", "sgd"],
                        default="adam")

    args = parser.parse_args()

    # Validate required args per mode
    if args.mode in ("reproduce", "train", "cross") and not args.train_file:
        parser.error(f"--train-file is required for mode {args.mode}")
    if args.mode in ("infer", "cross") and not args.test_file:
        parser.error(f"--test-file is required for mode {args.mode}")
    if args.mode in ("train", "infer", "cross") and not args.checkpoint_dir:
        parser.error(f"--checkpoint-dir is required for mode {args.mode}")

    dispatch = {
        "reproduce": mode_reproduce,
        "train": mode_train,
        "infer": mode_infer,
        "cross": mode_cross,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
