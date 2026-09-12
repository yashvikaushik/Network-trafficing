from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai.src.models.lstm import TemporalLSTMClassifier


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "option_a_binary"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lstm_option_a"

TRAIN_PATH = DATA_DIR / "train.npz"
VAL_PATH = DATA_DIR / "val.npz"
TEST_PATH = DATA_DIR / "test.npz"

SEED = 42
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 60
PATIENCE = 10


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_split(path: Path):
    data = np.load(path)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    target_state_id = data["target_state_id"].astype(np.int64)

    if X.ndim != 3:
        raise ValueError(f"{path}: expected 3D X, got {X.shape}")

    if X.shape[1] != 10:
        raise ValueError(f"{path}: expected sequence length 10, got {X.shape[1]}")

    if X.shape[2] != 21:
        raise ValueError(f"{path}: expected 21 features, got {X.shape[2]}")

    if len(X) != len(y) or len(y) != len(target_state_id):
        raise ValueError(f"{path}: X/y/target_state_id lengths do not match")

    if not np.isfinite(X).all():
        raise ValueError(f"{path}: X contains non-finite values")

    return X, y, target_state_id


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()

    # Fit on training data ONLY.
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    return scaler


def apply_scaler(
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    original_shape = X.shape

    transformed = scaler.transform(
        X.reshape(-1, original_shape[-1])
    )

    return transformed.reshape(original_shape).astype(np.float32)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }

    # Attack recall = sensitivity for class 1.
    metrics["attack_recall"] = metrics["recall"]

    # Benign recall = sensitivity for class 0.
    metrics["benign_recall"] = float(
        recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics["false_positive_rate"] = float(
        fp / (fp + tn) if (fp + tn) else 0.0
    )

    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    metrics["true_positive"] = int(tp)

    if probabilities is not None:
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, probabilities)
            )
        except ValueError:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = float(
                average_precision_score(y_true, probabilities)
            )
        except ValueError:
            metrics["pr_auc"] = None

    return metrics


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()

    all_y = []
    all_pred = []
    all_prob = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)

        logits = model(X_batch)

        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = (probabilities >= 0.5).long()

        all_y.append(y_batch.numpy())
        all_pred.append(predictions.cpu().numpy())
        all_prob.append(probabilities.cpu().numpy())

    return (
        np.concatenate(all_y),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def save_training_curve(history: dict) -> None:
    plt.figure(figsize=(9, 5))

    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_curve.png",
        dpi=160,
    )

    plt.close()


def save_confusion_matrix(cm: np.ndarray) -> None:
    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title("LSTM Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks([0, 1], ["BENIGN", "ATTACK"])
    plt.yticks([0, 1], ["BENIGN", "ATTACK"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=160,
    )

    plt.close()


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------

def main() -> None:
    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print("=" * 70)
    print("SIH 2026 - Temporal LSTM Training")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    X_train, y_train, ids_train = load_split(TRAIN_PATH)
    X_val, y_val, ids_val = load_split(VAL_PATH)
    X_test, y_test, ids_test = load_split(TEST_PATH)

    print("Dataset:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val:   {X_val.shape}, {y_val.shape}")
    print(f"  Test:  {X_test.shape}, {y_test.shape}")
    print()

    print("Class distribution:")
    print(
        f"  Train: BENIGN={(y_train == 0).sum()}, "
        f"ATTACK={(y_train == 1).sum()}"
    )
    print(
        f"  Val:   BENIGN={(y_val == 0).sum()}, "
        f"ATTACK={(y_val == 1).sum()}"
    )
    print(
        f"  Test:  BENIGN={(y_test == 0).sum()}, "
        f"ATTACK={(y_test == 1).sum()}"
    )
    print()

    # ---------------------------------------------------------------
    # Scaling
    # ---------------------------------------------------------------

    scaler = fit_scaler(X_train)

    X_train = apply_scaler(scaler, X_train)
    X_val = apply_scaler(scaler, X_val)
    X_test = apply_scaler(scaler, X_test)

    with open(
        OUTPUT_DIR / "scaler.pkl",
        "wb",
    ) as handle:
        pickle.dump(scaler, handle)

    # ---------------------------------------------------------------
    # PyTorch datasets
    # ---------------------------------------------------------------

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = TemporalLSTMClassifier(
        input_size=21,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=2,
        dropout=DROPOUT,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Model parameters: {parameter_count:,}")
    print()

    # ---------------------------------------------------------------
    # Class-weighted loss
    # ---------------------------------------------------------------

    class_counts = np.bincount(
        y_train,
        minlength=2,
    ).astype(np.float32)

    weights = class_counts.sum() / (
        2.0 * class_counts
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    best_model_path = OUTPUT_DIR / "model.pt"

    print("Training:")
    print("-" * 70)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        sample_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = len(y_batch)

            running_loss += (
                loss.item() * batch_size
            )

            sample_count += batch_size

        train_loss = running_loss / sample_count

        # Validation
        model.eval()

        val_running_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)

                loss = criterion(
                    logits,
                    y_batch,
                )

                batch_size = len(y_batch)

                val_running_loss += (
                    loss.item() * batch_size
                )

                val_count += batch_size

        val_loss = val_running_loss / val_count

        val_y, val_pred, val_prob = predict(
            model,
            val_loader,
            device,
        )

        val_f1 = f1_score(
            val_y,
            val_pred,
            average="macro",
            zero_division=0,
        )

        history["train_loss"].append(
            float(train_loss)
        )

        history["val_loss"].append(
            float(val_loss)
        )

        history["val_macro_f1"].append(
            float(val_f1)
        )

        improved = val_f1 > best_val_f1

        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": 21,
                    "sequence_length": 10,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "num_classes": 2,
                    "best_val_macro_f1": best_val_f1,
                    "best_epoch": best_epoch,
                },
                best_model_path,
            )

        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
            + ("  *BEST*" if improved else "")
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}."
            )
            break

    # ---------------------------------------------------------------
    # Save history
    # ---------------------------------------------------------------

    (OUTPUT_DIR / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    save_training_curve(history)

    # ---------------------------------------------------------------
    # Load BEST model
    # ---------------------------------------------------------------

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ---------------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------------

    test_y, test_pred, test_prob = predict(
        model,
        test_loader,
        device,
    )

    test_metrics = calculate_metrics(
        test_y,
        test_pred,
        test_prob,
    )

    cm = confusion_matrix(
        test_y,
        test_pred,
        labels=[0, 1],
    )

    save_confusion_matrix(cm)

    report = classification_report(
        test_y,
        test_pred,
        target_names=["BENIGN", "ATTACK"],
        digits=4,
        zero_division=0,
    )

    (OUTPUT_DIR / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            metrics_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    config = {
        "seed": SEED,
        "input_features": 21,
        "sequence_length": 10,
        "target": {
            "0": "BENIGN",
            "1": "ATTACK",
        },
        "model": {
            "type": "PyTorch LSTM classifier",
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "parameter_count": parameter_count,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "loss": "class-weighted cross entropy",
            "class_weights": weights.tolist(),
        },
        "device": str(device),
        "data": {
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "temporal_policy": {
            "random_split": False,
            "chronological_day_split": True,
            "scaler_fit_on_train_only": True,
            "test_used_for_model_selection": False,
        },
        "limitation": (
            "CICFlowMeter source timestamps retain the documented "
            "12-hour/no-AM-PM ambiguity."
        ),
    }

    (OUTPUT_DIR / "training_config.json").write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "label_mapping.json").write_text(
        json.dumps(
            {
                "BENIGN": 0,
                "ATTACK": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Final output
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    print()
    print("Confusion matrix:")
    print(cm)

    print()
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation macro-F1: "
        f"{best_val_f1:.4f}"
    )

    print()
    print("Saved artifacts:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
