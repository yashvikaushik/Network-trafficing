from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from ai.src.models.lstm import TemporalLSTMClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "option_a_binary"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lstm_option_a_exp2"

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def load_data(path: Path):
    data = np.load(path)
    return (
        data["X"].astype(np.float32),
        data["y"].astype(np.int64),
    )


def load_model():
    checkpoint = torch.load(
        OUTPUT_DIR / "model.pt",
        map_location=DEVICE,
    )

    model = TemporalLSTMClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


@torch.no_grad()
def get_probabilities(model, X):
    tensor = torch.from_numpy(X).to(DEVICE)

    probabilities = []

    batch_size = 256

    for start in range(0, len(tensor), batch_size):
        batch = tensor[start:start + batch_size]

        logits = model(batch)

        probs = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        probabilities.append(
            probs.cpu().numpy()
        )

    return np.concatenate(probabilities)


def evaluate_threshold(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "attack_precision": precision_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "attack_recall": recall_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "attack_f1": f1_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "benign_recall": recall_score(
            y_true,
            predictions,
            pos_label=0,
            zero_division=0,
        ),
    }


def main():
    print("=" * 70)
    print("SIH 2026 - LSTM Threshold Analysis")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print()

    X_train, y_train = load_data(
        DATA_DIR / "train.npz"
    )

    X_val, y_val = load_data(
        DATA_DIR / "val.npz"
    )

    X_test, y_test = load_data(
        DATA_DIR / "test.npz"
    )

    with open(
        OUTPUT_DIR / "scaler.pkl",
        "rb",
    ) as handle:
        scaler = pickle.load(handle)

    def scale(X):
        shape = X.shape

        X_scaled = scaler.transform(
            X.reshape(-1, X.shape[-1])
        )

        return X_scaled.reshape(shape).astype(
            np.float32
        )

    # IMPORTANT:
    # This scaler was fitted only on training data.
    X_val = scale(X_val)
    X_test = scale(X_test)

    model = load_model()

    print("Getting validation probabilities...")
    val_probabilities = get_probabilities(
        model,
        X_val,
    )

    print("Getting test probabilities...")
    test_probabilities = get_probabilities(
        model,
        X_test,
    )

    thresholds = np.arange(
        0.20,
        0.71,
        0.05,
    )

    validation_results = []

    for threshold in thresholds:
        result = evaluate_threshold(
            y_val,
            val_probabilities,
            float(threshold),
        )

        validation_results.append(result)

    # Select threshold using validation macro-F1.
    best = max(
        validation_results,
        key=lambda item: item["macro_f1"],
    )

    best_threshold = best["threshold"]

    # Also report the validation threshold that gives
    # at least 50% attack recall, if one exists.
    recall_candidates = [
        item
        for item in validation_results
        if item["attack_recall"] >= 0.50
    ]

    best_recall_operating_point = None

    if recall_candidates:
        best_recall_operating_point = max(
            recall_candidates,
            key=lambda item: item["attack_precision"],
        )

    # Apply the selected validation threshold ONCE to test.
    test_result = evaluate_threshold(
        y_test,
        test_probabilities,
        best_threshold,
    )

    print()
    print("VALIDATION THRESHOLD ANALYSIS")
    print("-" * 70)

    print(
        f"{'Threshold':>10} "
        f"{'Attack Prec':>12} "
        f"{'Attack Rec':>12} "
        f"{'Attack F1':>11} "
        f"{'Macro F1':>10}"
    )

    for result in validation_results:
        print(
            f"{result['threshold']:>10.2f} "
            f"{result['attack_precision']:>12.4f} "
            f"{result['attack_recall']:>12.4f} "
            f"{result['attack_f1']:>11.4f} "
            f"{result['macro_f1']:>10.4f}"
        )

    print()
    print("=" * 70)
    print("SELECTED THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold selected using validation macro-F1: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation macro-F1: "
        f"{best['macro_f1']:.4f}"
    )

    print(
        f"Validation attack recall: "
        f"{best['attack_recall']:.4f}"
    )

    print(
        f"Validation attack precision: "
        f"{best['attack_precision']:.4f}"
    )

    if best_recall_operating_point:
        print()
        print(
            "Best validation operating point with "
            "attack recall >= 50%:"
        )
        print(
            f"  Threshold: "
            f"{best_recall_operating_point['threshold']:.2f}"
        )
        print(
            f"  Attack recall: "
            f"{best_recall_operating_point['attack_recall']:.4f}"
        )
        print(
            f"  Attack precision: "
            f"{best_recall_operating_point['attack_precision']:.4f}"
        )

    print()
    print("=" * 70)
    print("FINAL TEST RESULT AT SELECTED THRESHOLD")
    print("=" * 70)

    for key, value in test_result.items():
        print(f"{key}: {value:.4f}")

    results = {
        "selection_method": (
            "Threshold selected on validation macro-F1; "
            "test evaluated only after threshold selection."
        ),
        "selected_threshold": best_threshold,
        "validation_results": validation_results,
        "selected_validation_result": best,
        "best_validation_point_with_attack_recall_at_least_0.50": (
            best_recall_operating_point
        ),
        "test_result_at_selected_threshold": test_result,
    }

    output_path = (
        OUTPUT_DIR /
        "threshold_analysis.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
