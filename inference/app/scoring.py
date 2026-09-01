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
Scores are normalized to 0..1: score = 1 - dist / region, clamped at 0, where
1 means identical.

Partial runs: expected tokens past the last aligned one are unreached words
(stopped early) and their deletions are free; `region` is max(tokens up to
the last match, actual tokens, 1). Skipped words inside the reached region
still cost. If nothing matched at all, the overall score is 0.
"""

import re
import unicodedata
from functools import lru_cache

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

# Common English function words: espeak word-alone phonemization returns the
# CITATION form ("a" -> eɪ, "to" -> tuː), but connected speech almost always
# uses the REDUCED form (ə, tə). Both are correct; accept either at equal
# cost, otherwise perfectly natural unstressed readings score near zero.
# Values are alternate IPA token lists; only alternates with the same token
# count as the word's canonical form are used (position-wise mapping).
_FUNCTION_WORD_ALTS: dict[str, list[list[str]]] = {
    "a": [["ə"]],
    "an": [["ə", "n"]],
    "the": [["ð", "iː"]],
    "to": [["t", "ə"]],
    "of": [["ə", "v"]],
    "and": [["ə", "n", "d"]],
    "you": [["j", "ə"]],
    "are": [["ə", "ɹ"]],
    "for": [["f", "ə"]],
    "or": [["ə", "ɹ"]],
    "your": [["j", "ə"]],
    "was": [["w", "ə", "z"]],
}


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
    # Memoized: the phoneme inventory is ~40 symbols, so the DP cost matrix
    # (n*m cells) only ever needs ~1600 real panphon calls.
    return _subst_cost_cached(a, b)


@lru_cache(maxsize=None)
def _subst_cost_cached(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return float(_dist.feature_edit_distance(a, b))


def _positional_cost(
    i: int, expected: list[str], alts: dict[int, list[str]], actual_tok: str
) -> float:
    """Substitution cost at expected position i: min over the canonical token
    and any registered function-word alternates at that position."""
    c = _subst_cost(expected[i], actual_tok)
    for alt in alts.get(i, ()):
        c = min(c, _subst_cost(alt, actual_tok))
        if c == 0.0:
            break
    return c


def _align(
    expected: list[str], actual: list[str], alts: dict[int, list[str]] | None = None
) -> list[tuple[int | None, int | None]]:
    """Levenshtein-style DP alignment of phoneme token sequences.

    Substitution cost is panphon feature edit distance (minimized over
    function-word alternates in `alts`); insertion/deletion cost is 1.
    Returns a list of (expected_idx | None, actual_idx | None) pairs in
    sequence order.
    """
    alts = alts or {}
    n, m = len(expected), len(actual)
    cost = [[_positional_cost(i, expected, alts, actual[j]) for j in range(m)] for i in range(n)]
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


def _expected_for(text: str) -> tuple[list[str], list[list[str]], list[str], dict[int, list[str]]]:
    """Expected side for a passage: words, per-word tokens, flat tokens, and
    positional function-word alternates. Cached (LRU, 32 passages) because the
    reader's /align endpoint and the final /score share it.
    """
    return _expected_cached(text)


@lru_cache(maxsize=32)
def _expected_cached(text: str):
    words = _WORD_RE.findall(text)
    word_tokens = [_phonemize_word(w) for w in words]
    expected_tokens = [t for toks in word_tokens for t in toks]

    # Register function-word alternates position-wise (only when the
    # alternate's token count matches the canonical form's).
    alts: dict[int, list[str]] = {}
    base = 0
    for w, toks in zip(words, word_tokens):
        for alt in _FUNCTION_WORD_ALTS.get(w.lower(), ()):
            if len(alt) == len(toks):
                for p, tok in enumerate(alt):
                    alts.setdefault(base + p, []).append(tok)
        base += len(toks)
    return words, word_tokens, expected_tokens, alts


def _fit_prefix(expected: list[str], actual: list[str], alts: dict[int, list[str]]) -> int:
    """Fitting (semi-global) alignment: the actual sequence is fully consumed
    against a PREFIX of the expected sequence; trailing expected tokens are
    free. Returns the prefix length k (in tokens). Ties resolve to the
    smallest k — reading starts at the beginning of the passage, unlike
    scoring, where the global alignment may tie-break late.
    """
    n, m = len(expected), len(actual)
    if m == 0:
        return 0
    prev = [float(j) for j in range(m + 1)]
    best_k, best_v = 0, prev[m]
    for i in range(1, n + 1):
        cur = [float(i)] + [0.0] * m
        ei = expected[i - 1]
        ei_alts = alts.get(i - 1, ())
        for j in range(1, m + 1):
            a = actual[j - 1]
            c = _subst_cost(ei, a)
            for alt in ei_alts:
                c2 = _subst_cost(alt, a)
                if c2 < c:
                    c = c2
            cur[j] = min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + c)
        if cur[m] < best_v:
            best_v, best_k = cur[m], i
        prev = cur
    return best_k


def align_progress(text: str, actual_ipa: str) -> int:
    """0-based index of the last expected word matched by the speech so far;
    -1 if nothing has matched yet. Used by the scrolling reader.

    Raises ValueError if the text contains no scoreable words.
    """
    words, word_tokens, expected_tokens, alts = _expected_for(text)
    if not words:
        raise ValueError("text contains no scoreable words")
    actual_tokens = normalize_ipa(actual_ipa).split()

    k = _fit_prefix(expected_tokens, actual_tokens, alts)
    if k == 0:
        return -1

    owner: list[int] = []
    for wi, toks in enumerate(word_tokens):
        owner.extend([wi] * len(toks))
    return owner[k - 1]


def score(text: str, actual_ipa: str) -> dict:
    """Score actual IPA against the expected IPA for `text`.

    Raises ValueError if the text contains no scoreable words.
    """
    words, word_tokens, expected_tokens, alts = _expected_for(text)
    if not words:
        raise ValueError("text contains no scoreable words")
    actual_tokens = normalize_ipa(actual_ipa).split()

    pairs = _align(expected_tokens, actual_tokens, alts)

    # Partial runs: expected tokens past the last matched one are words the
    # user never reached (stopped early) — their deletions are free. Skipped
    # words *within* the reached region still cost. The score denominator is
    # the compared region, not the full passage.
    last_matched = -1
    for e, a in pairs:
        if e is not None and a is not None:
            last_matched = max(last_matched, e)

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
        if a is None and e > last_matched:
            continue  # unreached word, free
        wi = owner[e]
        if a is None:
            c = 1.0
        else:
            c = _positional_cost(e, expected_tokens, alts, actual_tokens[a])
            word_actual[wi].append(actual_tokens[a])
        total_dist += c
        word_dist[wi] += c

    denom = max(last_matched + 1, len(actual_tokens), 1)
    # Nothing matched at all (silence, noise) -> 0, not a free 1.0.
    overall = 0.0 if last_matched < 0 else max(0.0, 1.0 - total_dist / denom)

    out_words = []
    for w, toks, d, acts in zip(words, word_tokens, word_dist, word_actual):
        # No aligned actual phonemes -> word not said (skipped mid-run or
        # unreached): empty ipa, score 0. The UI styles empty ipa as
        # "not attempted".
        if not acts:
            ws = 0.0
        else:
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
