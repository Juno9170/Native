"""Pronunciation scoring: expected IPA (espeak-ng, en-us / General American)
vs. actual IPA from the wav2vec2 espeak model.

Normalization — applied identically to BOTH expected and actual IPA before
any comparison (see normalize_ipa):
  - Unicode NFC
  - strip primary/secondary stress marks (U+02C8, U+02CC)
  - strip tie bars (U+0361, U+035C) and syllable breaks (.)
  - collapse whitespace
Rationale: the wav2vec2 espeak model omits some diacritics (notably stress),
so both sides are reduced to the same unstressed segment inventory. Anything
we strip from one side must be stripped from the other.

Distance: panphon FeatureEditDistance at the phoneme-token level. Whole
utterances are compared via a DP alignment whose substitution cost is the
panphon feature edit distance between the two aligned tokens (indel cost 1).
Scores are normalized to 0..1: score = 1 - dist / max(n_expected, n_actual, 1),
clamped at 0, where 1 means identical.
"""

import re
import unicodedata

import panphon.distance
from phonemizer import phonemize
from phonemizer.separator import Separator

_dist = panphon.distance.Distance()

# phone="|" makes the espeak backend emit one "|"-separated token per phoneme
# (phonemizer requires word/phone separators to differ, so not a plain space),
# which gives us clean tokens for alignment and token counts.
_phone_sep = Separator(word=" ", syllable="", phone="|")

_STRIP_RE = re.compile(r"[ˈˌ͜͡.]")
_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")


def normalize_ipa(s: str) -> str:
    """Strip stress marks / tie bars / syllable breaks; collapse whitespace."""
    s = unicodedata.normalize("NFC", s)
    s = _STRIP_RE.sub("", s)
    return " ".join(s.split())


def _phonemize_word(word: str) -> list[str]:
    """Expected phoneme tokens for a single word, normalized."""
    out = phonemize(
        word, language="en-us", backend="espeak", separator=_phone_sep, strip=True
    )
    out = normalize_ipa(out)
    return [tok for chunk in out.split() for tok in chunk.split("|") if tok]


def _subst_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return float(_dist.feature_edit_distance(a, b))


def _align(expected: list[str], actual: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein-style DP alignment of phoneme token sequences.

    Substitution cost is panphon feature edit distance; insertion/deletion
    cost is 1. Returns a list of (expected_idx | None, actual_idx | None)
    pairs in sequence order.
    """
    n, m = len(expected), len(actual)
    cost = [[_subst_cost(expected[i], actual[j]) for j in range(m)] for i in range(n)]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + cost[i - 1][j - 1],
            )
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + cost[i - 1][j - 1]:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1.0:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def score(text: str, actual_ipa: str) -> dict:
    """Score actual IPA against the expected IPA for `text`.

    Raises ValueError if the text contains no scoreable words.
    """
    words = _WORD_RE.findall(text)
    if not words:
        raise ValueError("text contains no scoreable words")

    word_tokens = [_phonemize_word(w) for w in words]
    expected_tokens = [t for toks in word_tokens for t in toks]
    actual_tokens = normalize_ipa(actual_ipa).split()

    pairs = _align(expected_tokens, actual_tokens)

    # Expected-token index -> owning word index.
    owner: list[int] = []
    for wi, toks in enumerate(word_tokens):
        owner.extend([wi] * len(toks))

    total_dist = 0.0
    word_dist = [0.0] * len(words)
    word_actual: list[list[str]] = [[] for _ in words]
    for e, a in pairs:
        if e is None:
            # Inserted actual phoneme: penalized in the overall score but not
            # attributed to any word (word boundaries are ambiguous there).
            total_dist += 1.0
            continue
        wi = owner[e]
        if a is None:
            c = 1.0
        else:
            c = _subst_cost(expected_tokens[e], actual_tokens[a])
            word_actual[wi].append(actual_tokens[a])
        total_dist += c
        word_dist[wi] += c

    denom = max(len(expected_tokens), len(actual_tokens), 1)
    overall = max(0.0, 1.0 - total_dist / denom)

    out_words = []
    for w, toks, d, acts in zip(words, word_tokens, word_dist, word_actual):
        # No aligned actual phonemes -> empty ipa and the deletion cost alone
        # drives the word score to 0 (expected-length penalty).
        wdenom = max(len(toks), len(acts), 1)
        ws = max(0.0, 1.0 - d / wdenom)
        out_words.append(
            {
                "word": w,
                "expected": "".join(toks),
                "ipa": "".join(acts),
                "score": round(ws, 4),
            }
        )

    return {
        "expectedIpa": " ".join("".join(t) for t in word_tokens),
        "score": round(overall, 4),
        "words": out_words,
    }
