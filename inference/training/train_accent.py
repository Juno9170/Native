"""Train the accent classifier on extracted SAA deviation features.

Input: /training/features.jsonl (from extract_features.py) — per-sample
deviation-token sequences + accent labels. Output: /training/checkpoints/
accent.pt — {state_dict, vocab, labels, config, strength} consumed by
app/accent.py at runtime.

Labels: accents with >= MIN_COUNT samples (plus native "english", capped
at ENGLISH_CAP to limit class imbalance); rarer accents are dropped.
Training: class-weighted cross-entropy with label smoothing, stratified
split, sequence-crop and token-dropout augmentation (in-app inputs vary
in length), cosine LR decay, early stop on validation macro-F1. Prints
per-class F1, top-3 accuracy, and the native-vs-non-native binary
accuracy (the product's most important distinction).

Only DEVIATION tokens (S:/D:/I:) are used — M: match tokens encode the
passage, not the accent (see accent.deviation_only). Global stats
(mean_dev, ins/del/sub fractions) bypass the transformer and feed the
head directly.

Strength calibration: s = mean_dev + 0.5*(frac_ins+frac_del). Buckets are
set from the observed distributions — mild up to the native speakers'
90th percentile, strong from the non-native median upward — and stored in
the checkpoint for the runtime classifier.

Run inside the inference container:
  docker compose exec -T inference python //training/train_accent.py
"""

import json
import math
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/app")
from app.accent import (  # noqa: E402
    AccentNet,
    MAX_LEN,
    PAD,
    deviation_only,
    encode,
    scalar_features,
)

FEATURES = "/training/features.jsonl"
OUT_DIR = "/training/checkpoints"
SEED = 42

MIN_COUNT = 15  # minimum samples for an accent to become a class
ENGLISH_CAP = 250  # native speakers dominate otherwise (579 vs ~30/class)
VOCAB_MIN_FREQ = 2
VOCAB_MAX = 3000

EPOCHS = 150
BATCH = 32
LR = 1e-3
LABEL_SMOOTHING = 0.1
PATIENCE = 15
CROP_MIN = 0.6  # augmentation: keep a random contiguous 60-100% window
TOK_DROPOUT = 0.05


def load_samples():
    rows = [json.loads(line) for line in open(FEATURES)]
    counts = Counter(r["accent"] for r in rows)
    keep = {a for a, c in counts.items() if a == "english" or c >= MIN_COUNT}
    rows = [r for r in rows if r["accent"] in keep]
    rng = random.Random(SEED)
    eng = [r for r in rows if r["accent"] == "english"]
    if len(eng) > ENGLISH_CAP:
        keep_eng = set(id(r) for r in rng.sample(eng, ENGLISH_CAP))
        rows = [r for r in rows if r["accent"] != "english" or id(r) in keep_eng]
    labels = sorted({r["accent"] for r in rows})
    return rows, labels


def build_vocab(rows):
    freq = Counter(t for r in rows for t in deviation_only(r["tokens"]))
    vocab = {"<pad>": PAD, "<unk>": 1}
    for tok, c in freq.most_common(VOCAB_MAX):
        if c >= VOCAB_MIN_FREQ:
            vocab[tok] = len(vocab)
    return vocab


def augment(ids, rng):
    """Random contiguous crop + token dropout -> fresh list each epoch."""
    n = len(ids)
    if n > 8:
        w = rng.randint(max(4, int(CROP_MIN * n)), n)
        start = rng.randint(0, n - w)
        ids = ids[start : start + w]
    return [t for t in ids if rng.random() > TOK_DROPOUT] or [PAD]


def collate(batch_ids, batch_scalars):
    n = max(len(b) for b in batch_ids)
    out = torch.full((len(batch_ids), n), PAD, dtype=torch.long)
    for i, b in enumerate(batch_ids):
        out[i, : len(b)] = torch.tensor(b)
    return out, torch.tensor(batch_scalars, dtype=torch.float32)


def run_epoch(model, data, scalars, y, rng, opt=None, class_w=None):
    train = opt is not None
    model.train() if train else model.eval()
    order = list(range(len(data)))
    rng.shuffle(order)
    total_loss, preds, gold = 0.0, [], []
    for i in range(0, len(order), BATCH):
        idx = order[i : i + BATCH]
        ids = [augment(data[j], rng) if train else data[j] for j in idx]
        x, sc = collate(ids, [scalars[j] for j in idx])
        yb = torch.tensor([y[j] for j in idx])
        with torch.set_grad_enabled(train):
            logits = model(x, sc)
            loss = nn.functional.cross_entropy(
                logits, yb, weight=class_w, label_smoothing=LABEL_SMOOTHING
            )
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        total_loss += float(loss) * len(idx)
        preds.extend(logits.argmax(-1).tolist())
        gold.extend(yb.tolist())
    return total_loss / len(data), preds, gold


@torch.no_grad()
def all_logits(model, data, scalars):
    """Logits in DATA order (unshuffled) — for top-3 and binary metrics."""
    model.eval()
    outs = []
    for i in range(0, len(data), BATCH):
        x, sc = collate(data[i : i + BATCH], scalars[i : i + BATCH])
        outs.append(model(x, sc))
    return torch.cat(outs)


def main() -> None:
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    rows, labels = load_samples()
    print(f"samples={len(rows)} classes={len(labels)}")
    print("classes:", ", ".join(f"{l}({sum(1 for r in rows if r['accent']==l)})" for l in labels))

    vocab = build_vocab(rows)
    print(f"vocab={len(vocab)}")
    data = [encode(r["tokens"], vocab) for r in rows]
    scalars = [
        scalar_features(r["tokens"], r["mean_dev"], r["frac_ins"], r["frac_del"])
        for r in rows
    ]
    y = [labels.index(r["accent"]) for r in rows]
    eng = labels.index("english")

    tr, te, str_, ste, ytr, yte = train_test_split(
        data, scalars, y, test_size=0.15, random_state=SEED, stratify=y
    )

    # Class weights: inverse sqrt frequency (softer than 1/n — the tiny
    # classes still matter but native english doesn't get crushed).
    freq = Counter(ytr)
    w = torch.tensor([1.0 / math.sqrt(freq[c]) for c in range(len(labels))])
    w = w / w.mean()

    model = AccentNet(len(vocab), len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_f1, best_state, best_epoch, bad = 0.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        loss, _, _ = run_epoch(model, tr, str_, ytr, rng, opt=opt, class_w=w)
        sched.step()
        _, preds, gold = run_epoch(model, te, ste, yte, rng)
        f1 = f1_score(gold, preds, average="macro")
        acc = float(np.mean(np.array(gold) == np.array(preds)))
        if epoch % 5 == 0 or epoch < 10:
            print(f"epoch {epoch:3d} loss={loss:.4f} val_acc={acc:.3f} val_macroF1={f1:.3f}", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop (best macroF1={best_f1:.3f} @ epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    logits = all_logits(model, te, ste)
    preds = logits.argmax(-1).tolist()
    print("\n" + classification_report(yte, preds, target_names=labels, digits=3))
    top3 = float(
        np.mean([g in t for g, t in zip(yte, logits.topk(min(3, len(labels)), dim=-1).indices.tolist())])
    )
    print(f"top-3 accuracy: {top3:.3f}")
    bin_gold = [g != eng for g in yte]
    bin_pred = [p != eng for p in preds]
    print(f"native-vs-non-native accuracy: {np.mean(np.array(bin_gold) == np.array(bin_pred)):.3f}")

    # Strength calibration from the FULL training set (not the split).
    s_native = np.array(
        [r["mean_dev"] + 0.5 * (r["frac_ins"] + r["frac_del"]) for r in rows if r["accent"] == "english"]
    )
    s_nonnative = np.array(
        [r["mean_dev"] + 0.5 * (r["frac_ins"] + r["frac_del"]) for r in rows if r["accent"] != "english"]
    )
    mild_max = float(np.percentile(s_native, 90))
    strong_min = float(np.percentile(s_nonnative, 50))
    print(f"strength: native p50={np.percentile(s_native,50):.3f} p90={mild_max:.3f} | "
          f"non-native p50={strong_min:.3f} p90={np.percentile(s_nonnative,90):.3f}")
    if strong_min <= mild_max:
        strong_min = mild_max + 0.05  # degenerate overlap: keep bands ordered

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "vocab": vocab,
            "labels": labels,
            "config": {},
            "strength": {"mild_max": mild_max, "strong_min": strong_min},
        },
        f"{OUT_DIR}/accent.pt",
    )
    print(f"saved {OUT_DIR}/accent.pt (val macroF1={best_f1:.3f} @ epoch {best_epoch})")


if __name__ == "__main__":
    main()
