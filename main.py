"""
Entry point for the Mood Machine rule-based mood analyzer.
"""

from typing import List

from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS


def evaluate_rule_based(posts: List[str], labels: List[str]) -> float:
    """
    Evaluate the rule-based MoodAnalyzer on a labeled dataset.

    Prints each text with its predicted label, true label, and explanation,
    then returns the overall accuracy as a float between 0 and 1.
    """
    analyzer = MoodAnalyzer()
    correct = 0
    total = len(posts)

    print("=== Rule-Based Evaluation on SAMPLE_POSTS ===\n")
    for text, true_label in zip(posts, labels):
        predicted_label = analyzer.predict_label(text)
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        reason = analyzer.explain(text)
        print(f'  {status} "{text}"')
        print(f'    predicted={predicted_label}, true={true_label}')
        print(f'    {reason}\n')

    if total == 0:
        print("No labeled examples to evaluate.")
        return 0.0

    accuracy = correct / total
    print(f"Rule-based accuracy on SAMPLE_POSTS: {accuracy:.2f} ({correct}/{total})")
    return accuracy


def run_batch_demo() -> None:
    """
    Run the MoodAnalyzer on the sample posts and print predictions with explanations.
    """
    analyzer = MoodAnalyzer()
    print("\n=== Batch Demo on SAMPLE_POSTS (rule-based) ===\n")
    for text in SAMPLE_POSTS:
        label = analyzer.predict_label(text)
        reason = analyzer.explain(text)
        print(f'  "{text}"')
        print(f'    → {label}  ({reason})\n')


def run_interactive_loop() -> None:
    """
    Let the user type their own sentences and see the predicted mood.

    Type 'quit' or press Enter on an empty line to exit.
    """
    analyzer = MoodAnalyzer()
    print("\n=== Interactive Mood Machine (rule-based) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the Mood Machine.")
            break

        label = analyzer.predict_label(user_input)
        reason = analyzer.explain(user_input)
        print(f"Model: {label}  ({reason})\n")


if __name__ == "__main__":
    evaluate_rule_based(SAMPLE_POSTS, TRUE_LABELS)

    run_batch_demo()

    run_interactive_loop()

    print("\nTip: After you explore the rule-based model here,")
    print("run `python ml_experiments.py` to try a simple ML-based model")
    print("trained on the same SAMPLE_POSTS and TRUE_LABELS.")
