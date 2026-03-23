# Shared data for the Mood Machine lab.
#
# This file defines:
#   - POSITIVE_WORDS: starter list of positive words
#   - NEGATIVE_WORDS: starter list of negative words
#   - SAMPLE_POSTS: short example posts for evaluation and training
#   - TRUE_LABELS: human labels for each post in SAMPLE_POSTS

# Word lists — expanded to include common emotions, slang roots, and adjectives

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "proud",
    "hopeful",
    "wonderful",
    "brilliant",
    "nice",
    "okay",
    "accomplished",
    "better",
    "glad",
    "fantastic",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "exhausted",
    "failed",
    "miserable",
    "frustrated",
    "overwhelmed",
    "horrible",
]

# Labeled dataset.
# Each post in SAMPLE_POSTS must have exactly one matching entry in TRUE_LABELS.
# Quick sanity check: len(SAMPLE_POSTS) == len(TRUE_LABELS)

SAMPLE_POSTS = [
    # --- Original 6 posts ---
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",

    # --- 10 new posts: slang, emojis, negation, sarcasm, mixed emotions ---

    # Slang + emoji (positive)
    "This is absolutely fire 🔥",

    # Sarcasm — model will misclassify this as positive (intentional failure case)
    "I love getting stuck in traffic",

    # Negation of a negative word (positive)
    "Not bad at all honestly",

    # Mixed: negative exhaustion + positive achievement
    "I'm exhausted but really proud of what I accomplished",

    # Internet slang (positive)
    "This food is lowkey bussin",

    # Stacked negatives (negative)
    "I really hate how tired and stressed I always feel",

    # Mixed: stress + optimism
    "kinda stressed out but everything will be okay",

    # Slang + emoji (positive)
    "The concert was sick 🔥",

    # Ambiguous "fine" — no clear signal (neutral)
    "I'm fine with it",

    # Mixed: failure + relief emoji
    "Just failed my exam but at least it's over 😅",
]

TRUE_LABELS = [
    # Original 6
    "positive",
    "negative",
    "mixed",
    "neutral",
    "positive",
    "negative",

    # New 10
    "positive",   # fire 🔥
    "negative",   # sarcasm — model predicted positive (documented failure)
    "positive",   # not bad
    "mixed",      # exhausted but proud
    "positive",   # lowkey bussin
    "negative",   # hate + tired + stressed
    "mixed",      # stressed + okay
    "positive",   # sick 🔥
    "neutral",    # I'm fine with it
    "mixed",      # failed + 😅
]

# Sanity check (runs on import)
assert len(SAMPLE_POSTS) == len(TRUE_LABELS), (
    f"Mismatch: {len(SAMPLE_POSTS)} posts but {len(TRUE_LABELS)} labels. "
    "Every post must have exactly one label."
)
