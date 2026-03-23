# The Mood Machine

Submitted by: Krishi Shah

## Overview

The Mood Machine is a sentiment analyzer that classifies short social-media-style posts into one of four mood labels: **positive**, **negative**, **neutral**, or **mixed**.

Two approaches were built and compared:

- **Rule-Based Classifier** (`mood_analyzer.py`) — hand-crafted word lists, negation detection, internet slang, and emoji signals.
- **ML Classifier** (`ml_experiments.py`) — bag-of-words features with Logistic Regression (scikit-learn).

Built as part of the AI110 Foundations of AI Engineering course (Module 3 Tinker: The Mood Machine).

## Project Structure

| File | Purpose |
|---|---|
| `dataset.py` | Word lists and 16 labeled sample posts |
| `mood_analyzer.py` | Rule-based classifier with negation, slang, and emoji support |
| `main.py` | Evaluates and demos the rule-based model |
| `ml_experiments.py` | Trains and evaluates a Logistic Regression classifier |
| `model_card.md` | Full documentation of both models, results, and limitations |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the rule-based model (evaluation + interactive mode):
```bash
python main.py
```

Run the ML model:
```bash
python ml_experiments.py
```

## Results

| Model | Accuracy |
|---|---|
| Rule-Based | **0.94** (15/16) |
| ML — Logistic Regression | 1.00* (16/16) |

*ML accuracy is measured on training data only and is artificially inflated due to memorization. See `model_card.md` for full analysis.

### Known Failure: Sarcasm

The rule-based model misclassifies `"I love getting stuck in traffic"` as **positive** (true label: **negative**). The word "love" scores +1 and the model has no mechanism for detecting irony. This is a documented limitation — see the model card for details.

## Highlights

- **Negation handling**: "not happy" → negative; "not bad" → positive
- **Slang signals**: "fire", "sick", "bussin", "lowkey" → positive; "mid", "cringe" → negative
- **Emoji signals**: 🔥 😊 😅 → positive; 😢 😡 💔 → negative
- **Mixed detection**: posts with both positive and negative hits → "mixed" label
- **Explainability**: every prediction includes a score breakdown via `explain()`

## License

For educational use as part of CodePath AI110.
