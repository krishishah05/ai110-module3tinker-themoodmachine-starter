# Mood Machine: Model Card

## Model Overview

The Mood Machine is a text-based sentiment classifier that predicts one of four mood labels — **positive**, **negative**, **neutral**, or **mixed** — for short social-media-style posts.

Two implementations were built and compared:

| | Rule-Based (`mood_analyzer.py`) | ML (`ml_experiments.py`) |
|---|---|---|
| **Method** | Hand-crafted word lists + scoring rules | Bag-of-words + Logistic Regression |
| **Transparency** | Fully interpretable via `explain()` | Black-box (weights not inspected) |
| **Accuracy (training data)** | 0.94 (15/16) | 1.00 (16/16)* |

*ML accuracy is measured on training data only — it is artificially inflated. See Evaluation section.

---

## Data

### Dataset Summary
- **Total examples**: 16 labeled posts (`SAMPLE_POSTS` in `dataset.py`)
- **Label distribution**: positive (6), negative (4), mixed (4), neutral (2)

### How the Dataset Was Built
The original 6 posts were extended to 16 by adding posts that test specific linguistic phenomena:

| Post | Label | Why it was added |
|---|---|---|
| "This is absolutely fire 🔥" | positive | Slang + emoji signal |
| "I love getting stuck in traffic" | negative | Sarcasm — intentional failure case |
| "Not bad at all honestly" | positive | Negation of a negative word |
| "I'm exhausted but really proud of what I accomplished" | mixed | Mixed negative + positive emotions |
| "This food is lowkey bussin" | positive | Internet slang only (no standard words) |
| "I really hate how tired and stressed I always feel" | negative | Stacked negative signals |
| "kinda stressed out but everything will be okay" | mixed | Stress + optimism together |
| "The concert was sick 🔥" | positive | Slang + emoji (sick = awesome) |
| "I'm fine with it" | neutral | Ambiguous "fine" — no clear signal |
| "Just failed my exam but at least it's over 😅" | mixed | Failure (negative) + relief emoji |

### Labeling Process
Labels were assigned manually. The **mixed** label was used when a post contained detectably both positive and negative signals. The **neutral** label was used when no clear sentiment signal existed. Some labels are inherently subjective — "I'm fine" could reasonably be labeled neutral, mixed, or negative depending on context.

### Dataset Limitations
- Only 16 total examples — far too few for reliable generalization.
- The neutral class has only 2 examples and is underrepresented.
- All posts are written in informal American English. Other dialects, languages, and cultural expressions are not represented.
- Sarcasm is present in only 1 example and is labeled negative, even though the model cannot detect it.

---

## Rule-Based Approach

### How It Works

**Step 1 — Preprocess**
- Lowercase the text
- Remove apostrophes so contractions become matchable (can't → cant, don't → dont)
- Strip ASCII punctuation with spaces; unicode emoji characters are preserved
- Tokenize on whitespace

**Step 2 — Score**
Each token is checked against four signal sources:
1. `POSITIVE_WORDS` / `NEGATIVE_WORDS` (expanded word lists in `dataset.py`)
2. `_SLANG_POSITIVE` / `_SLANG_NEGATIVE` (e.g., "fire", "bussin", "mid", "cringe")
3. `_EMOJI_POSITIVE` / `_EMOJI_NEGATIVE` (e.g., 🔥, 😊, 😢)
4. **Negation handling**: if the previous token is a negator (not, never, cant, dont, etc.), the signal is flipped. "not happy" counts as a negative hit; "not bad" counts as a positive hit.

**Step 3 — Predict**
- Both positive AND negative hits present → `"mixed"`
- Net score > 0, no negative hits → `"positive"`
- Net score < 0, no positive hits → `"negative"`
- No mood signals detected → `"neutral"`

### Example Explanations from `explain()`
```
"Feeling tired but kind of hopeful"
  → mixed  Score = 0 (positive: ['hopeful'], negative: ['tired'])

"I am not happy about this"
  → negative  Score = -1 (positive: [], negative: ['not happy'])

"The concert was sick 🔥"
  → positive  Score = 2 (positive: ['sick', '🔥'], negative: [])

"I love getting stuck in traffic"
  → positive  Score = 1 (positive: ['love'], negative: [])  ← WRONG (sarcasm)
```

---

## Machine Learning Approach

### How It Works
- **Feature extraction**: `CountVectorizer` — converts each post into a vector of raw word counts (bag-of-words)
- **Model**: `LogisticRegression` with `max_iter=1000`
- **Training**: Fit on all 16 `SAMPLE_POSTS` / `TRUE_LABELS`
- **Evaluation**: Evaluated on the same 16 posts used for training

### Important Caveat
The ML model achieved **100% accuracy on its training data**. This does not mean it is a better model — it means it memorized 16 examples. If given a new post it has never seen, it will likely fail because:
- Its vocabulary is limited to words in the 16 training posts
- It has never seen new slang, emojis, or phrasing
- Class imbalance (only 2 neutral posts) biases the decision boundary

The ML model correctly classified "I love getting stuck in traffic" as **negative**, but only because that exact post is in its training data. It learned nothing about sarcasm — it just memorized the label.

---

## Evaluation

### Rule-Based Results

**Accuracy: 0.94 (15/16)**

| Post | Predicted | True | Correct? |
|---|---|---|---|
| "I love this class so much" | positive | positive | ✓ |
| "Today was a terrible day" | negative | negative | ✓ |
| "Feeling tired but kind of hopeful" | mixed | mixed | ✓ |
| "This is fine" | neutral | neutral | ✓ |
| "So excited for the weekend" | positive | positive | ✓ |
| "I am not happy about this" | negative | negative | ✓ |
| "This is absolutely fire 🔥" | positive | positive | ✓ |
| **"I love getting stuck in traffic"** | **positive** | **negative** | **✗** |
| "Not bad at all honestly" | positive | positive | ✓ |
| "I'm exhausted but really proud..." | mixed | mixed | ✓ |
| "This food is lowkey bussin" | positive | positive | ✓ |
| "I really hate how tired and stressed..." | negative | negative | ✓ |
| "kinda stressed out but...okay" | mixed | mixed | ✓ |
| "The concert was sick 🔥" | positive | positive | ✓ |
| "I'm fine with it" | neutral | neutral | ✓ |
| "Just failed my exam but...😅" | mixed | mixed | ✓ |

### ML Results

**Accuracy: 1.00 (16/16)** — training accuracy only, not generalizable.

### Comparison

| | Rule-Based | ML |
|---|---|---|
| Accuracy (training set) | 0.94 | 1.00* |
| Handles new/unseen posts | Partially (via word lists) | No |
| Handles sarcasm | No | No (memorized, not learned) |
| Explainable decisions | Yes | No |
| Sensitive to data changes | Yes (word lists) | Very (only 16 examples) |

---

## Limitations

### Sarcasm
**Example**: "I love getting stuck in traffic" → predicted `positive`, true label `negative`.

The rule-based model sees "love" and scores +1. It has no understanding of context, irony, or tone. Sarcasm detection requires contextual understanding that keyword matching fundamentally cannot provide.

### Negation Scope
The negation handler only looks at the **immediately preceding token**. Multi-token negations fail silently:
- "I don't think this is good" → "good" is two tokens away from "don't" → not negated → scored as positive (wrong)
- "This is anything but great" → "but" is not a negator → "great" scored as positive (wrong)

### Rare Slang and Cultural Terms
The slang vocabulary is limited to ~10 common terms. New slang like "rizz", "delulu", "it's giving", or "no cap" produces a score of 0 and is classified as neutral regardless of context.

### Small Dataset
With 16 examples, every accuracy number is noise. A single mislabeled post changes accuracy by 6.25%. Neither model is validated at this scale.

### Ambiguous Labels
"I'm fine" and "This is fine" are labeled neutral, but could easily be read as passive-aggressive negatives. These judgment calls are invisible to both models but shape every evaluation result.

---

## Ethical Considerations

### Language and Cultural Bias
The dataset and word lists were written in informal American English. The model may:
- Miss sentiment expressed through other dialects, non-native phrasing, or cultural idioms
- Misinterpret code-switching or multilingual posts
- Perform better for users whose language matches the training vocabulary

### Risk in Sensitive Contexts
If deployed on real user posts (e.g., mental health apps, crisis lines), the model's failure to detect distress masked by sarcasm or polite phrasing ("I'm fine 🙂") could have real consequences. **This model should not be used in any safety-critical or mental health context.**

### Subjectivity of Labels
Sentiment is not objective. Two people may label "I'm exhausted but proud" differently. The labels in this dataset reflect one annotator's judgment and are not a ground truth.

---

## Improvement Ideas

- **More data**: A minimum of 500+ diverse, independently labeled examples across all four classes.
- **TF-IDF**: Replace raw bag-of-words counts with TF-IDF weighting to reduce noise.
- **Negation scope**: Extend negation to multi-token spans (e.g., look ahead 3 tokens).
- **Sarcasm detection**: Requires contextual models (e.g., BERT, RoBERTa) or explicit sarcasm-annotated data.
- **Emoji lexicon**: Use a full emoji sentiment dictionary (e.g., from VADER) rather than a hand-picked set.
- **Cross-validation**: Evaluate the ML model on a held-out test split, not training data.
- **Confidence scores**: Return a probability distribution over labels, not just the argmax.
