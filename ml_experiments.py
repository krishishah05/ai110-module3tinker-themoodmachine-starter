"""
Simple ML experiments for the Mood Machine lab.

This file uses scikit-learn to train a tiny text classifier on the same
SAMPLE_POSTS and TRUE_LABELS used by the rule-based model.
"""

from typing import List, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from dataset import SAMPLE_POSTS, TRUE_LABELS


def train_ml_model(
    texts: List[str],
    labels: List[str],
) -> Tuple[CountVectorizer, LogisticRegression]:
    """
    Train a simple text classifier using bag-of-words features
    and logistic regression.
    """
    if len(texts) != len(labels):
        raise ValueError(
            "texts and labels must be the same length. "
            "Check SAMPLE_POSTS and TRUE_LABELS in dataset.py."
        )

    if not texts:
        raise ValueError("No training data provided. Add examples in dataset.py.")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return vectorizer, model


def evaluate_on_dataset(
    texts: List[str],
    labels: List[str],
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> float:
    """
    Evaluate the trained model on a labeled dataset.

    Prints each text with its predicted and true label,
    then returns overall accuracy.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must be the same length.")

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    print("=== ML Model Evaluation on Dataset ===\n")
    correct = 0
    for text, true_label, pred_label in zip(texts, labels, preds):
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"
        print(f'  {status} "{text}"')
        print(f'    predicted={pred_label}, true={true_label}\n')

    accuracy = accuracy_score(labels, preds)
    print(f"ML model accuracy on this dataset: {accuracy:.2f} ({correct}/{len(texts)})")
    print("Note: This is training accuracy (same data used for training and evaluation).")
    print("It is artificially inflated and does not reflect real-world performance.\n")
    return accuracy


def predict_single_text(
    text: str,
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> str:
    """
    Predict the mood label for a single text string using the trained ML model.
    """
    X = vectorizer.transform([text])
    return model.predict(X)[0]


def run_interactive_loop(
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> None:
    """
    Let the user type their own sentences and see the ML model's predicted label.

    Type 'quit' or press Enter on an empty line to exit.
    """
    print("\n=== Interactive Mood Machine (ML model) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the ML Mood Machine.")
            break

        label = predict_single_text(user_input, vectorizer, model)
        print(f"ML model: {label}\n")


if __name__ == "__main__":
    print("Training an ML model on SAMPLE_POSTS and TRUE_LABELS from dataset.py...")
    print("Make sure you have added enough labeled examples before running this.\n")

    vectorizer, model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)

    evaluate_on_dataset(SAMPLE_POSTS, TRUE_LABELS, vectorizer, model)

    run_interactive_loop(vectorizer, model)

    print("\nTip: Compare these predictions with the rule-based model")
    print("by running `python main.py`. Notice where they fail in")
    print("similar ways and where they fail in different ways.")
