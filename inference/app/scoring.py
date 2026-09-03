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

Partial runs: a word counts as said only if enough of its phonemes matched
well (>= min(tokens, 2) matches at cost <= 0.5). Reading ends at the first
run of >=3 consecutive unsaid words — expected tokens past that cutoff are
free (stopped early), while skipped words inside the reached region still
cost. If nothing was reached at all, the overall score is 0.
"""

import re
import unicodedata
from functools import lru_cache

import panphon.distance
import panphon.featuretable
from phonemizer import phonemize
from phonemizer.separator import Separator

_dist = panphon.distance.Distance()
_ft = panphon.featuretable.FeatureTable()


@lru_cache(maxsize=None)
def _is_vocalic(tok: str) -> bool:
    """True if every phone in the token is syllabic (a vowel/diphthong).
    Used to reject vowel soup: hums, breaths, and mic noise transcribe as
    near-pure vowel runs, while real speech is consonant-rich."""
    phones = _ft.ipa_segs(tok)
    if not phones:
        return False
    return all(
        (p := _ft.fts(phone)) and p.match({"syl": 1}) for phone in phones
    )

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
        # Tie-break toward the EARLIEST consistent match: on ties prefer
        # deleting expected tokens (moving up without consuming actual), so
        # with repeated content the actual speech anchors to the first
        # occurrence, not the last.
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1.0:
            pairs.append((i - 1, None))
            i -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + cost[i - 1][j - 1]:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
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


def locate_window(
    text: str, actual_ipa: str, from_word: int, span: int = 25,
    max_word: int | None = None,
) -> int:
    """Reader position from a short trailing audio window (a "peek").

    Unlike align_progress (prefix/fitting, for the cumulative transcript), the
    window contains only the LAST few words spoken, so it is aligned LOCALLY
    against the band of expected words [from_word, from_word + span): the
    matched region may start anywhere in the band, and the answer is the END
    of the best-matching region. Returns the 0-based word index, or -1 when
    the match is too poor to trust (silence/noise hallucinations).

    max_word is a hard cap on the answer. A reader realistically skips at
    most 1-2 words between updates, so a match ending further ahead is always
    a duplicate-word coincidence, never intentional — such regions are
    excluded from candidacy outright (not just penalized).

    Raises ValueError if the text contains no scoreable words.
    """
    words, word_tokens, expected_tokens, alts = _expected_for(text)
    if not words:
        raise ValueError("text contains no scoreable words")
    actual_tokens = normalize_ipa(actual_ipa).split()
    if len(actual_tokens) < 3:
        return -1
    # Vowel-soup guard: humming/breathing/mic noise transcribes as near-pure
    # vowels and would otherwise schwa-match its way forward through the text.
    vocalic = sum(1 for t in actual_tokens if _is_vocalic(t))
    if vocalic / len(actual_tokens) > 0.7:
        return -1

    from_word = max(0, min(from_word, len(words) - 1))
    lo = sum(len(t) for t in word_tokens[:from_word])
    hi = sum(len(t) for t in word_tokens[: min(len(words), from_word + span)])
    if hi <= lo:
        return -1

    # Local alignment over expected[lo:hi]: dp rows = expected tokens, cols =
    # actual tokens; cur[0] = 0 lets the matched region start anywhere in the
    # band; the answer is the best final column over band rows, ties toward
    # the earlier end. A short window's few phonemes can cheaply match MANY
    # places in the band, so region selection adds a small drift penalty per
    # token of distance from the band start (0.02/token ≈ one phoneme
    # mismatch per 5 words) — enough to anchor the reader near the current
    # position, far too small to override a genuine match. Rows past max_word
    # are still computed (the match may legitimately START there... no — the
    # region END is the answer, so rows past the cap are simply never
    # candidates).
    owner: list[int] = []
    for wi, toks in enumerate(word_tokens):
        owner.extend([wi] * len(toks))

    m = len(actual_tokens)
    prev = [float(j) for j in range(m + 1)]  # virtual row before the band
    best_i, best_sel, best_raw = lo, float("inf"), float("inf")
    for i in range(lo + 1, hi + 1):
        cur = [0.0] + [0.0] * m
        ei = expected_tokens[i - 1]
        ei_alts = alts.get(i - 1, ())
        for j in range(1, m + 1):
            a = actual_tokens[j - 1]
            c = _subst_cost(ei, a)
            for alt in ei_alts:
                c2 = _subst_cost(alt, a)
                if c2 < c:
                    c = c2
            cur[j] = min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + c)
        if max_word is not None and owner[i - 1] > max_word:
            prev = cur
            continue  # region ends past the skip cap: not a candidate
        sel = cur[m] + 0.02 * (i - lo)
        if sel < best_sel:
            best_sel, best_i, best_raw = sel, i, cur[m]
        prev = cur

    # Reject noise: measured on this pipeline — TTS speech windows align at
    # ~0.15-0.19 cost/token, clean synthetic speech ~0.02-0.12, random-phoneme
    # noise ~0.26. Real (denoised) white noise transcribes to empty and never
    # reaches here; hums/breaths are caught by the vowel-soup guard above.
    # Threshold errs toward accepting accented speech over rejecting noise.
    if best_i <= lo or best_raw > 0.30 * m:
        return -1

    return owner[best_i - 1]


def score(text: str, actual_ipa: str) -> dict:
    """Score actual IPA against the expected IPA for `text`.

    Raises ValueError if the text contains no scoreable words.
    """
    words, word_tokens, expected_tokens, alts = _expected_for(text)
    if not words:
        raise ValueError("text contains no scoreable words")
    actual_tokens = normalize_ipa(actual_ipa).split()

    pairs = _align(expected_tokens, actual_tokens, alts)

    # Expected-token index -> owning word index.
    owner: list[int] = []
    for wi, toks in enumerate(word_tokens):
        owner.extend([wi] * len(toks))

    # Run-end detection. "last matched token" alone is unreliable on long
    # passages: lenient substitution costs spuriously match a few tokens near
    # the end, marking unread words as reached. Instead, a word is UNSAID
    # when almost none of its expected phonemes matched well, and reading is
    # over at the first unsaid word followed by a (>=3-word) suffix with at
    # most one said word in it — a reader realistically skips 1-2 words, so
    # a long quiet tail means "stopped here". Expected tokens past the cutoff
    # are free (stopped early); unsaid words within the reached region still
    # cost (skipped mid-run).
    good = [0] * len(words)  # good (low-cost) matches per word
    for e, a in pairs:
        if e is not None and a is not None and _positional_cost(
            e, expected_tokens, alts, actual_tokens[a]
        ) <= 0.5:
            good[owner[e]] += 1

    def said(wi: int) -> bool:
        return good[wi] >= min(len(word_tokens[wi]), 2)

    said_mask = [said(wi) for wi in range(len(words))]
    # Reading is over at the first unsaid word whose suffix (>=3 words long)
    # contains at most one said word — a reader realistically skips 1-2 words,
    # so a long unsaid tail means "stopped here", and one said word in it is
    # tolerated as a spurious late match. Fallback for short tails: cutoff
    # right after the last said word.
    cutoff = len(words)
    for c in range(len(words) - 2):
        if not said_mask[c] and sum(said_mask[c:]) <= 1:
            cutoff = c
            break
    if cutoff == len(words) and not all(said_mask):
        last_said = max((wi for wi, s in enumerate(said_mask) if s), default=-1)
        cutoff = last_said + 1
    # Token index of the end of the last reached word (-1 if none reached).
    last_tok = sum(len(t) for t in word_tokens[:cutoff]) - 1

    total_dist = 0.0
    word_dist = [0.0] * len(words)
    word_actual: list[list[str]] = [[] for _ in words]
    for e, a in pairs:
        if e is None:
            # Inserted actual phoneme: penalized in the overall score but not
            # attributed to any word (word boundaries are ambiguous there).
            total_dist += 1.0
            continue
        if a is None and e > last_tok:
            continue  # past the cutoff: unreached word, free
        wi = owner[e]
        if a is None:
            c = 1.0
        else:
            c = _positional_cost(e, expected_tokens, alts, actual_tokens[a])
            word_actual[wi].append(actual_tokens[a])
        total_dist += c
        word_dist[wi] += c

    # Words past the cutoff are "not said": no attribution, score 0.
    for wi in range(cutoff, len(words)):
        word_actual[wi] = []
        word_dist[wi] = 0.0

    denom = max(last_tok + 1, len(actual_tokens), 1)
    # Nothing reached at all (silence, noise) -> 0, not a free 1.0.
    overall = 0.0 if last_tok < 0 else max(0.0, 1.0 - total_dist / denom)

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
