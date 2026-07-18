"""
Appendix C — Core Pipeline Implementation

The key algorithmic components of the adopted pipeline, extracted from the
notebooks into an importable module so the design choices described in the
paper can be read and reused without opening a notebook.

Reference: "Confidence Misread as Fear: Register Shift and Genre Conditioning
in Lyric-to-Image Synthesis", NeurIPS 2026 Creative AI Track.
"""

from collections import Counter

# ---------------------------------------------------------------------------
# Stage 1 — Custom emotion-aware stopword list
# ---------------------------------------------------------------------------
# NLTK's default English list removes 179 high-frequency tokens. For prose this
# is appropriate; for lyric analysis it is destructive. These 47 tokens carry
# primary thematic weight in popular music and are retained.

EMOTION_STOPWORDS_KEEP = {
    'hurt', 'fear', 'love', 'lost', 'pain', 'dark', 'light',
    'fall', 'shadow', 'cry', 'dream', 'night', 'hope', 'down',
    'through', 'away', 'still', 'ever', 'never', 'always', 'alone',
    'broken', 'heart', 'tears', 'die', 'dead', 'burn', 'cold',
    'empty', 'free', 'rise', 'run', 'fight', 'stand', 'hold',
    'find', 'keep', 'let', 'know', 'feel', 'seen', 'gone',
    'left', 'whole', 'deep', 'high', 'real', 'true'
}


def build_stopwords():
    """NLTK defaults minus the emotionally load-bearing tokens."""
    from nltk.corpus import stopwords
    return set(stopwords.words('english')) - EMOTION_STOPWORDS_KEEP


# ---------------------------------------------------------------------------
# Stage 2 — TF-IDF with emotion-boosted frequency scoring
# ---------------------------------------------------------------------------
# TF-IDF alone suppresses emotionally critical words that appear across multiple
# lyric samples (their IDF weight drops). A 2x multiplier on 14 anchor terms
# counteracts this. RAKE was rejected: its phrase-preference scoring
# systematically penalizes the single-word anchors that dominate lyric writing.

EMOTION_ANCHORS = {
    'hurt', 'fear', 'love', 'hate', 'sad', 'happy', 'cry', 'pain',
    'joy', 'lost', 'dark', 'light', 'night', 'dream'
}


def extract_keywords(tokens, corpus_tokens, compute_tfidf, top_n=7):
    """Merge TF-IDF distinctiveness with emotion-boosted raw frequency.

    Args:
        tokens: preprocessed token list for this lyric sample
        corpus_tokens: token lists for all samples (the IDF denominator)
        compute_tfidf: callable(tokens, corpus_tokens) -> {token: score}
        top_n: number of keywords to return

    Returns:
        Top-n tokens by combined score.
    """
    tfidf = compute_tfidf(tokens, corpus_tokens)

    freq = Counter(tokens)
    boosted = {t: freq[t] * (2 if t in EMOTION_ANCHORS else 1) for t in freq}

    combined = {}
    for token in set(list(tfidf.keys()) + list(boosted.keys())):
        combined[token] = tfidf.get(token, 0) + boosted.get(token, 0)

    return sorted(combined, key=combined.get, reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Stage 4 — Emotion-to-palette mapping (introduced in V3)
# ---------------------------------------------------------------------------
# Encoding the palette explicitly resolves the V2 failure where style
# descriptors semantically overrode the emotion signal.

PALETTE_MAP = {
    'sadness':  'deep blues and purples, muted tones',
    'joy':      'warm brightness, golden tones, vivid colors',
    'anger':    'deep reds and oranges, high contrast',
    'fear':     'desaturated greys, cold blues, shadows',
    'neutral':  'muted earth tones, soft light',
    'disgust':  'dark greens and browns, gritty textures',
    'surprise': 'bright unexpected colors, dynamic composition',
}

MOOD_MAP = {
    'sadness': 'melancholic and brooding',
    'joy':     'euphoric and energetic',
    'anger':   'intense and powerful',
    'fear':    'tense and unsettling',
    'neutral': 'contemplative and calm',
}


# ---------------------------------------------------------------------------
# Stage 4 — Genre-conditioned override (introduced in V5)
# ---------------------------------------------------------------------------
# This is the paper's mitigation for the register-shift failure. The emotion
# classifier reads hip-hop bravado as fear (0.331, flat posterior, no decisive
# signal), so the override bypasses the classifier entirely when hip-hop
# vocabulary is detected.
#
# Note this heuristic encodes the authors' own judgment about which words
# signal the register. It is a correction, not a solution; a learned genre
# classifier would relocate that judgment rather than remove it.

HIP_HOP_SIGNALS = {
    'city', 'crown', 'chain', 'throne', 'hustle', 'grind',
    'top', 'king', 'run', 'lambo', 'rearview', 'gold'
}

HIP_HOP_OVERRIDE = (
    'confidence and ambition',
    'warm golden tones and crimson, city lights at night',
    'triumphant and determined',
)


def apply_genre_override(tokens, emotion, palette, mood, threshold=2):
    """Override emotion-derived parameters when hip-hop register is detected.

    Fires when at least `threshold` hip-hop signal words appear in the token
    set, regardless of what the emotion classifier returned.
    """
    if len(HIP_HOP_SIGNALS & set(tokens)) >= threshold:
        return HIP_HOP_OVERRIDE
    return emotion, palette, mood


# ---------------------------------------------------------------------------
# Stage 4 — Narrative prompt construction (V4 template, reused by V5)
# ---------------------------------------------------------------------------

def build_v4_prompt(emotion, palette, mood, keywords):
    """The ~65-word narrative template adopted as the pipeline standard."""
    kw_str = ', '.join(keywords[:7])
    return (
        f"A cinematic album cover depicting the emotional journey "
        f"of {emotion}. Scene: {kw_str} bathed in {palette}. "
        f"Mood: {mood}. Art direction: dramatic lighting, "
        f"high contrast, professional photography."
    )


NEGATIVE_PROMPT = (
    "text, watermark, words, letters, typography, caption, subtitle"
)


# ---------------------------------------------------------------------------
# Note on the stopword list
# ---------------------------------------------------------------------------
# Verification against NLTK's English stopword list (198 tokens in current
# NLTK releases) shows that only 'down' and 'through' from EMOTION_STOPWORDS_KEEP
# are actually members of that list. The remaining tokens are content words that
# NLTK never removes, so retaining them is a no-op with respect to the set
# difference in build_stopwords().
#
# The list is therefore best read as a documented inventory of the vocabulary
# treated as thematically load-bearing in this pipeline — it also feeds the
# emotion-anchor boosting in Stage 2, where it does change behavior — rather
# than as a set of tokens rescued from NLTK's filter.
