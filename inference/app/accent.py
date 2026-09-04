"""Accent classification from phoneme-deviation sequences.

The classifier consumes the same material the scorer produces: the DP
alignment between expected IPA (espeak, from the user's text) and actual
IPA (wav2vec2, from their speech), rendered as a deviation-token sequence:

  M:<p>     matched phoneme
  S:<e>><a> substitution (expected e, heard a)
  D:<p>     deleted expected phoneme
  I:<p>     inserted spoken phoneme

This is deliberately text-independent: the model learns systematic
deviation PATTERNS (e.g. S:θ>s, D:h, I:ə), not the passage, so a model
trained on the fixed Speech Accent Archive paragraph transfers to
arbitrary user-typed text.

Model: tiny transformer encoder over a deviation-token vocabulary (~1k),
mean-pooled, linear head over accent classes. Trained offline by
inference/training/train_accent.py; the checkpoint (vocab + labels +
weights + strength calibration) lives at /training/checkpoints/accent.pt.
When no checkpoint is present the classifier is unavailable and the
endpoint reports 503.
"""

import os
import threading

import torch
import torch.nn as nn

CHECKPOINT_PATH = os.environ.get("ACCENT_CHECKPOINT", "/training/checkpoints/accent.pt")

PAD, UNK = 0, 1
MAX_LEN = 512


class AccentNet(nn.Module):
    """Embedding + 2-layer transformer encoder + mean-pool; the pooled
    vector is concatenated with global scalar stats before the head."""

    N_SCALARS = 4  # mean_dev, frac_ins, frac_del, frac_sub — see scalar_features

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model + self.N_SCALARS)
        self.head = nn.Linear(d_model + self.N_SCALARS, n_classes)

    def forward(self, ids: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        # ids: (B, T) padded with PAD; scalars: (B, N_SCALARS). Returns
        # class logits (B, C).
        mask = ids == PAD  # True = ignore
        pos = torch.arange(ids.size(1), device=ids.device).clamp(max=MAX_LEN - 1)
        x = self.embed(ids) + self.pos(pos)
        x = self.encoder(x, src_key_padding_mask=mask)
        # Masked mean pool.
        keep = (~mask).unsqueeze(-1).float()
        x = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        x = torch.cat([x, scalars], dim=1)
        return self.head(self.dropout(self.norm(x)))


def pairs_to_tokens(pairs, expected, actual, alts, positional_cost) -> list[str]:
    """Alignment pairs -> deviation tokens. positional_cost(e, a_tok) is
    scoring._positional_cost; passed in so training and runtime share the
    exact format produced by extract_features.py."""
    tokens = []
    for e, a in pairs:
        if e is not None and a is not None:
            if positional_cost(e, expected, alts, actual[a]) == 0.0:
                tokens.append(f"M:{expected[e]}")
            else:
                tokens.append(f"S:{expected[e]}>{actual[a]}")
        elif e is not None:
            tokens.append(f"D:{expected[e]}")
        else:
            tokens.append(f"I:{actual[a]}")
    return tokens


def deviation_only(tokens: list[str]) -> list[str]:
    """Drop M: (match) tokens: identical across classes, they encode the
    passage, not the accent, and drown out the S/D/I deviation signal."""
    return [t for t in tokens if not t.startswith("M:")]


def scalar_features(
    tokens: list[str], mean_dev: float, frac_ins: float, frac_del: float
) -> list[float]:
    """Global sequence stats fed straight into the classifier head:
    [mean_dev, frac_ins, frac_del, frac_sub]. Identical at train time
    (train_accent.py) and runtime (Classifier)."""
    n = max(len(tokens), 1)
    frac_sub = sum(1 for t in tokens if t.startswith("S:")) / n
    return [mean_dev, frac_ins, frac_del, frac_sub]


def encode(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(t, UNK) for t in deviation_only(tokens)[:MAX_LEN]]
    return ids or [PAD]


class Classifier:
    """Loaded checkpoint; classify() runs on CPU in <5 ms."""

    def __init__(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.vocab: dict[str, int] = ckpt["vocab"]
        self.labels: list[str] = ckpt["labels"]
        self.strength: dict = ckpt["strength"]
        cfg = ckpt["config"]
        self.model = AccentNet(len(self.vocab), len(self.labels), **cfg)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def classify_tokens(
        self, tokens: list[str], mean_dev: float, frac_ins: float, frac_del: float
    ) -> dict:
        ids = torch.tensor([encode(tokens, self.vocab)])
        scalars = torch.tensor([scalar_features(tokens, mean_dev, frac_ins, frac_del)])
        probs = torch.softmax(self.model(ids, scalars), dim=-1)[0]
        top = probs.topk(min(3, len(self.labels)))
        alts = [
            {"accent": self.labels[i], "p": round(float(probs[i]), 4)}
            for i in top.indices
        ]
        # Strength scalar: substitution cost plus gap rate, bucketed against
        # the training-time native/non-native distributions.
        s = mean_dev + 0.5 * (frac_ins + frac_del)
        mild_max, strong_min = self.strength["mild_max"], self.strength["strong_min"]
        level = "mild" if s <= mild_max else "strong" if s >= strong_min else "moderate"
        # 0..1 position within the calibrated band, for the UI.
        span = max(strong_min - mild_max, 1e-6)
        score = min(max((s - mild_max) / span, 0.0), 1.0)
        return {
            "accent": alts[0]["accent"],
            "confidence": alts[0]["p"],
            "alternatives": alts,
            "strength": {"level": level, "score": round(score, 3)},
        }


_lock = threading.Lock()
_classifier: Classifier | None = None


def get_classifier() -> Classifier | None:
    """Lazy-load the checkpoint; None when no trained model exists yet."""
    global _classifier
    with _lock:
        if _classifier is None and os.path.exists(CHECKPOINT_PATH):
            _classifier = Classifier(CHECKPOINT_PATH)
    return _classifier
