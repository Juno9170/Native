"""Pronunciation scoring: expected IPA (espeak-ng, en-us / General American)
vs. actual IPA from the wav2vec2 espeak model.

Normalization — applied identically to BOTH expected and actual IPA before
any comparison (see normalize_ipa):
  - Unicode NFC
  - strip primary/secondary stress marks (U+02C8, U+02CC)
  - strip tie bars (U+0361, U+035C) and syllable breaks (.)
  - split r-colored vowels ("ɔːɹ" -> "ɔː ɹ") — espeak emits them as one
    phone token, the wav2vec2 model usually as two
  - collapse whitespace
Rationale: the wav2vec2 espeak model omits some diacritics (notably stress),
so both sides are reduced to the same unstressed segment inventory. Anything
we strip from one side must be stripped from the other.

Distance: panphon FeatureEditDistance at the phoneme-token level. Whole
utterances are compared via a DP alignment whose substitution cost is the
panphon feature edit distance between the two aligned tokens (indel cost 1),
thresholded at 0.5 — a worse substitution costs more than a gap, so the DP
never smears speech thinly across unread words (see _dp_subst). Scores are
normalized to 0..1: score = 1 - dist / region, clamped at 0, where 1 means
identical.

Partial runs: scoring is two-pass. First a fitting alignment locates the
frontier (how far the reader got); the full alignment then runs ONLY within
that reached region — a global alignment over the whole passage would smear
the speech thinly across unread words. A word counts as said only if enough
of its phonemes matched well (>= min(tokens, 2) matches at cost <= 0.5), and
a frontier walk (skip tolerance <=2 words) trims pass-1 overshoot. Expected
tokens past the frontier are free (stopped early); skipped words inside the
reached region still cost. If nothing was reached at all, the score is 0.
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

# R-colored vowels: espeak emits "ɔːɹ" as ONE phone token, while the
# wav2vec2 model usually splits it ("ɔː" "ɹ") — and occasionally doesn't.
# The distance between the merged and split forms is large enough that the
# DP prefers gapping entirely, which is how genuinely-read words like "or"
# ended up with zero attributed phonemes. Split the digraph on BOTH sides.
_R_COLORED_RE = re.compile(r"([aeiouɑæɜəʌɔɪʊɛːˑ]+)ɹ")


def normalize_ipa(s: str) -> str:
    """Strip stress marks / tie bars / syllable breaks; split r-colored
    vowels; collapse whitespace."""
    s = unicodedata.normalize("NFC", s)
    s = _STRIP_RE.sub("", s)
    s = _R_COLORED_RE.sub(r"\1 ɹ", s)
    return " ".join(s.split())

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


def _phonemize_word(word: str) -> list[str]:
    """Expected phoneme tokens for a single word, normalized."""
    out = phonemize(
        word, language="en-us", backend="espeak", separator=_phone_sep, strip=True
    )
    out = normalize_ipa(out)
    return [tok for chunk in out.split() for tok in chunk.split("|") if tok]


def _phonemize_words(words: list[str]) -> list[list[str]]:
    """Expected phoneme tokens per word from ONE espeak call for the whole
    passage. Per-word calls cost ~25 ms each — 4+ s for a 200-word passage,
    which stalled the first reader update while the text cache was cold.
    Falls back to per-word calls if espeak's word chunking doesn't line up
    1:1 with the word list.
    """
    out = phonemize(
        " ".join(words),
        language="en-us",
        backend="espeak",
        separator=_phone_sep,
        strip=True,
    )
    chunks = normalize_ipa(out).split(" ")
    if len(chunks) != len(words):
        return [_phonemize_word(w) for w in words]
    return [[tok for tok in chunk.split("|") if tok] for chunk in chunks]


def _subst_cost(a: str, b: str) -> float:
    # Memoized: the phoneme inventory is ~40 symbols, so the DP cost matrix
    # (n*m cells) only ever needs ~1600 real panphon calls.
    return _subst_cost_cached(a, b)


@lru_cache(maxsize=None)
def _subst_cost_cached(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return float(_dist.feature_edit_distance(a, b))


# The alignment DPs use ASYMMETRIC, THRESHOLDED costs — not raw Levenshtein:
#
#   substitution  raw panphon cost when <= 0.5, else 2.0 + cost
#   deletion      0.25 per expected token (skipped reference phoneme)
#   insertion     1.00 per actual token (extra spoken phoneme)
#
# Why: raw costs make two failure modes cheaper than the truth.
#   1. Unrelated phonemes are often only ~0.2-0.5 apart in panphon space
#      (voiced plosives d~b ≈ 0.1!), always cheaper than a 2.0 delete+insert
#      pair — so the DP substitutes garbage rather than gapping, smearing the
#      transcript thinly across words nobody read (a real "dog" aligning onto
#      "above" three words later). Deleting an expected token must be cheaper
#      than a lucky-but-wrong substitution, hence 0.25.
#   2. Any substitution beyond the "good match" bar (0.5, the same threshold
#      the scoring side uses) is not a pronunciation variant, it's a gap —
#      priced above delete+insert (1.25) so the DP never takes it.
# Good matches (0.0-0.25) still always win, so read regions align densely.
# Word scores still use the raw graded cost; only the DP search sees this.
_DP_SUBST_OK = 0.5
_DP_DEL = 0.25
_DP_INS = 1.0

# Banded DP width for _fit_prefix: the speech/reference diagonal can drift
# by at most this many tokens (insertions, stutters) before cells off-band
# could matter. 64 tokens ≈ 18 words of skew — far beyond real readings.
_FIT_BAND = 64


def _dp_subst(c: float) -> float:
    return c if c <= _DP_SUBST_OK else _DP_DEL + _DP_INS + c


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
    expected: list[str],
    actual: list[str],
    alts: dict[int, list[str]] | None = None,
    del_cost: float = _DP_DEL,
) -> list[tuple[int | None, int | None]]:
    """Levenshtein-style DP alignment of phoneme token sequences.

    Substitution cost is panphon feature edit distance (minimized over
    function-word alternates in `alts`, thresholded — see _dp_subst);
    insertion cost is _DP_INS, deletion cost is del_cost. Cheap deletion
    (the default) is right for SEARCHING (frontier/window finding over mostly
    unread text); scoring the reached region uses del_cost=1.0 instead, so
    short function words aren't gapped away — their phonemes bleed into
    neighbors in connected speech, and with cheap deletion the DP would
    rather delete "the" (0.5) than attribute the neighboring tokens to it.
    Returns a list of (expected_idx | None, actual_idx | None) pairs in
    sequence order.
    """
    alts = alts or {}
    n, m = len(expected), len(actual)
    cost = [[_dp_subst(_positional_cost(i, expected, alts, actual[j])) for j in range(m)] for i in range(n)]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * del_cost
    for j in range(1, m + 1):
        dp[0][j] = j * _DP_INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j] + del_cost,
                dp[i][j - 1] + _DP_INS,
                dp[i - 1][j - 1] + cost[i - 1][j - 1],
            )
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        # Tie-break toward the EARLIEST consistent match: on ties prefer
        # deleting expected tokens (moving up without consuming actual), so
        # with repeated content the actual speech anchors to the first
        # occurrence, not the last.
        if i > 0 and dp[i][j] == dp[i - 1][j] + del_cost:
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
    word_tokens = _phonemize_words(words)
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

    The DP is banded (_FIT_BAND tokens around the diagonal): speech tracks
    the reference roughly token-for-token, so off-band cells can never be on
    the optimal path, and a full n*m DP on a 200-word passage is ~0.5M cells
    of pure Python per chunk — seconds of latency the reader feels.
    """
    n, m = len(expected), len(actual)
    if m == 0:
        return 0
    inf = float("inf")
    prev = [float(j) * _DP_INS for j in range(m + 1)]
    best_k, best_v = 0, prev[m]
    for i in range(1, n + 1):
        lo = max(1, i - _FIT_BAND)
        hi = min(m, i + _FIT_BAND)
        cur = [inf] * (m + 1)
        cur[0] = float(i) * _DP_DEL
        ei = expected[i - 1]
        ei_alts = alts.get(i - 1, ())
        for j in range(lo, hi + 1):
            a = actual[j - 1]
            c = _subst_cost(ei, a)
            for alt in ei_alts:
                c2 = _subst_cost(alt, a)
                if c2 < c:
                    c = c2
            b = prev[j] + _DP_DEL
            v = cur[j - 1] + _DP_INS
            if v < b:
                b = v
            v = prev[j - 1] + _dp_subst(c)
            if v < b:
                b = v
            cur[j] = b
        if cur[m] < best_v:
            best_v, best_k = cur[m], i
        prev = cur
    return best_k


def _word_progress(word_tokens: list[list[str]], k: int) -> float:
    """Fractional reader progress from k consumed expected tokens:
    word index + fraction of that word's phonemes reached, so the strip can
    slide continuously instead of stepping whole words. k is 1-based (tokens
    consumed); word 5 fully consumed -> 6.0 (reading word 6 next).
    """
    start = 0
    for wi, toks in enumerate(word_tokens):
        end = start + len(toks)
        if k <= end:
            return wi + (k - start) / len(toks)
        start = end
    return float(len(word_tokens))


def align_progress(
    text: str, actual_ipa: str, max_word: int | None = None
) -> float:
    """Fractional progress through the passage: word index + fraction of that
    word's phonemes matched by the speech so far; -1 if nothing has matched
    yet. Used by the scrolling reader.

    max_word hard-caps the answer: expected tokens past that word are removed
    before fitting. Chunks arrive every ~2.5 s, so the position can physically
    advance only so far between them — uncapped prefix fitting lets transcript
    noise cheap-match its way into words nobody has read yet.

    Raises ValueError if the text contains no scoreable words.
    """
    words, word_tokens, expected_tokens, alts = _expected_for(text)
    if not words:
        raise ValueError("text contains no scoreable words")
    actual_tokens = normalize_ipa(actual_ipa).split()

    if max_word is not None and 0 <= max_word < len(words) - 1:
        end = sum(len(t) for t in word_tokens[: max_word + 1])
        expected_tokens = expected_tokens[:end]

    k = _fit_prefix(expected_tokens, actual_tokens, alts)
    if k == 0:
        return -1

    return _word_progress(word_tokens, k)


def locate_window(
    text: str, actual_ipa: str, from_word: int, span: int = 14,
    max_word: int | None = None,
) -> float:
    """Reader position from a short trailing audio window (a "peek").

    Unlike align_progress (prefix/fitting, for the cumulative transcript), the
    window contains only the LAST few words spoken, so it is aligned LOCALLY
    against the band of expected words [from_word, from_word + span): the
    matched region may start anywhere in the band, and the answer is the END
    of the best-matching region as FRACTIONAL progress (word index + fraction
    of that word's phonemes reached), or -1 when the match is too poor to
    trust (silence/noise hallucinations).

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
    # band; the answer is the END of the best-matching region, ties toward
    # the earlier end. Leading actual tokens cost full insertions, so the
    # band must be wide enough to contain the whole window — the backend
    # starts the band ~6 words behind the strip (a 2.5 s window covers at
    # most ~12 words even at 290 wpm). A short window's few phonemes can
    # cheaply match MANY places in the band, so region selection adds a
    # drift penalty per token of distance from the band start (0.05/token ≈
    # one phoneme mismatch per 2 words); the dominant force is tail
    # consumption (1.0/inserted token), which anchors the answer at the true
    # end of the speech.
    owner: list[int] = []
    for wi, toks in enumerate(word_tokens):
        owner.extend([wi] * len(toks))

    m = len(actual_tokens)
    prev = [float(j) * _DP_INS for j in range(m + 1)]  # virtual row before the band
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
            cur[j] = min(
                prev[j] + _DP_DEL,
                cur[j - 1] + _DP_INS,
                prev[j - 1] + _dp_subst(c),
            )
        if max_word is not None and owner[i - 1] > max_word:
            prev = cur
            continue  # region ends past the skip cap: not a candidate
        sel = cur[m] + 0.05 * (i - lo)
        if sel < best_sel:
            best_sel, best_i, best_raw = sel, i, cur[m]
        prev = cur

    # Reject noise: genuine speech windows align at ~0.15-0.45 cost/token
    # (clean slow speech ~0.15; fast, garbled TTS ~0.45), while true noise
    # must insert (~1.0) or garbage-substitute (>1.25) every token — a wide
    # margin either way. Real (denoised) white noise transcribes to empty and
    # never reaches here; hums/breaths are caught by the vowel-soup guard.
    if best_i <= lo or best_raw > 0.60 * m:
        return -1

    return _word_progress(word_tokens, best_i)


def score(text: str, actual_ipa: str) -> dict:
    """Score actual IPA against the expected IPA for `text`.

    Raises ValueError if the text contains no scoreable words.
    """
    words, word_tokens, expected_tokens, alts = _expected_for(text)
    if not words:
        raise ValueError("text contains no scoreable words")
    actual_tokens = normalize_ipa(actual_ipa).split()

    # Expected-token index -> owning word index.
    owner: list[int] = []
    for wi, toks in enumerate(word_tokens):
        owner.extend([wi] * len(toks))

    # Pass 1 — find the frontier: fit the transcript against a PREFIX of the
    # passage. A global alignment over the whole passage smears the speech
    # thinly across unread words (every actual phoneme must be consumed, and
    # a lucky ~0.3 substitution always beats inserting it), which both marks
    # unread words as read and strips attribution from words that WERE read.
    # Restricting the alignment to the reached region keeps it dense.
    k = _fit_prefix(expected_tokens, actual_tokens, alts)
    fit_cutoff = 0 if k == 0 else owner[k - 1] + 1
    end_tok = sum(len(t) for t in word_tokens[:fit_cutoff])

    # Pass 2 — global alignment within the reached region only, with FULL
    # deletion cost: there is no unread tail to smear into anymore, and
    # expensive deletion keeps short function words ("the", "of", "or")
    # attributed — connected speech bleeds their phonemes into neighbors, and
    # cheap deletion would gap them to "not said" even though they were read.
    pairs = (
        _align(expected_tokens[:end_tok], actual_tokens, alts, del_cost=1.0)
        if end_tok
        else []
    )

    # Frontier walk: a word is SAID when enough of its phonemes matched well,
    # and reading is sequential — a said word extends the reached region only
    # within skipping distance (<=2 words) of the frontier. This trims any
    # overshoot from pass 1 (e.g. trailing noise lucky-matching a word or two
    # past where the reader actually stopped). Expected tokens past the
    # frontier are free (stopped early); unsaid words within the reached
    # region still cost (skipped mid-run).
    good = [0] * len(words)  # good (low-cost) matches per word
    for e, a in pairs:
        if e is not None and a is not None and _positional_cost(
            e, expected_tokens, alts, actual_tokens[a]
        ) <= 0.5:
            good[owner[e]] += 1

    def said(wi: int) -> bool:
        return good[wi] >= min(len(word_tokens[wi]), 2)

    frontier = -1
    for wi in range(fit_cutoff):
        if said(wi) and wi <= frontier + 3:
            frontier = wi
    cutoff = frontier + 1
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
        if e > last_tok:
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
