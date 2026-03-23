"""
Rule-based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words (including slang and emojis)
  - Compute a numeric score with negation handling
  - Convert that score into a mood label
"""

import re
from typing import List, Tuple

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


# Words that flip the sentiment of the immediately following token.
_NEGATORS = {
    "not", "never", "cant", "dont", "isnt", "wasnt", "wont",
    "wouldnt", "shouldnt", "cannot", "hardly", "barely", "no",
}

# Internet slang with clear positive or negative meaning.
_SLANG_POSITIVE = {
    "fire", "lit", "sick", "goat", "lowkey", "bussin", "blessed",
    "valid", "dope", "vibing", "w",
}
_SLANG_NEGATIVE = {
    "mid", "sus", "cringe", "cap", "trash", "wack", "rip",
}

# Emoji signals.
_EMOJI_POSITIVE = {"🔥", "😊", "😄", "🥰", "👍", "✨", "🎉", "😍", "🙌", "💯", "😅"}
_EMOJI_NEGATIVE = {"😢", "😭", "😡", "💔", "😤", "🙁", "😞", "😰"}


class MoodAnalyzer:
    """
    A rule-based mood classifier with negation handling, slang, and emoji support.
    """

    def __init__(
        self,
        positive_words: List[str] = None,
        negative_words: List[str] = None,
    ) -> None:
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for O(1) lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens.

        Steps:
          1. Strip whitespace and lowercase.
          2. Remove apostrophes so contractions become negator-matchable tokens
             (can't → cant, don't → dont, won't → wont).
          3. Replace ASCII punctuation with spaces, preserving unicode
             characters like emojis.
          4. Split on whitespace and filter empty strings.
        """
        cleaned = text.strip().lower()
        # Remove apostrophes for contraction normalization
        cleaned = cleaned.replace("'", "")
        # Replace ASCII punctuation with spaces (emojis are unicode, not affected)
        cleaned = re.sub(r'[!"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', cleaned)
        tokens = [t for t in cleaned.split() if t]
        return tokens

    # ------------------------------------------------------------------
    # Internal scoring helper
    # ------------------------------------------------------------------

    def _score_details(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Internal helper. Returns (score, positive_hits, negative_hits).

        Checks each token against:
          - POSITIVE_WORDS / NEGATIVE_WORDS from dataset.py
          - Internet slang (_SLANG_POSITIVE / _SLANG_NEGATIVE)
          - Emoji signals (_EMOJI_POSITIVE / _EMOJI_NEGATIVE)
          - Negation: if the previous token is in _NEGATORS, the signal is flipped.
        """
        tokens = self.preprocess(text)
        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []

        for i, token in enumerate(tokens):
            negated = i > 0 and tokens[i - 1] in _NEGATORS

            is_positive = (
                token in self.positive_words
                or token in _SLANG_POSITIVE
                or token in _EMOJI_POSITIVE
            )
            is_negative = (
                token in self.negative_words
                or token in _SLANG_NEGATIVE
                or token in _EMOJI_NEGATIVE
            )

            if is_positive:
                if negated:
                    score -= 1
                    negative_hits.append(f"not {token}")
                else:
                    score += 1
                    positive_hits.append(token)
            elif is_negative:
                if negated:
                    score += 1
                    positive_hits.append(f"not {token}")
                else:
                    score -= 1
                    negative_hits.append(token)

        return score, positive_hits, negative_hits

    # ------------------------------------------------------------------
    # Scoring logic
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric mood score for the given text.

        Positive signals increase the score; negative signals decrease it.
        Negation flips the signal of the immediately following sentiment word.
        Returns the net integer score.
        """
        score, _, _ = self._score_details(text)
        return score

    # ------------------------------------------------------------------
    # Label prediction
    # ------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score and signal counts into a mood label.

        Rules:
          - Both positive AND negative hits present → "mixed"
          - Net score > 0 (only positive hits)        → "positive"
          - Net score < 0 (only negative hits)        → "negative"
          - No mood signals detected                   → "neutral"
        """
        score, positive_hits, negative_hits = self._score_details(text)

        if positive_hits and negative_hits:
            return "mixed"
        elif score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        return "neutral"

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short human-readable explanation of the model's decision.

        Shows the net score and which tokens contributed positively or negatively,
        including negated tokens (e.g., "not happy" recorded as a negative hit).

        Example:
          'Score = -1 (positive: [], negative: ["not happy"])'
        """
        score, positive_hits, negative_hits = self._score_details(text)
        return (
            f"Score = {score} "
            f"(positive: {positive_hits if positive_hits else []}, "
            f"negative: {negative_hits if negative_hits else []})"
        )
