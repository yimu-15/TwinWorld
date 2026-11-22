
from pathlib import Path
import json

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "p2b"
OUT = ROOT / "outputs" / "p2c1"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261113
HORIZONS = [0, 5, 10, 20, 35, 50]
VIEWS = ["A_only", "B_only", "A_plus_B"]

manifest = json.loads(
    (DATA / "manifest.json").read_text(encoding="utf-8")
)
HZ = float(manifest["sampling_hz"])


def load_split(split):
    # Intentionally load only train and val.
    with np.load(
        DATA / f"{split}_observations.npz",
        allow_pickle=False
    ) as f:
        passive = f["passive_xy"].copy()
        active = f["active_xy"].copy()

    with np.load(
        DATA / f"{split}_private.npz",
        allow_pickle=False
    ) as f:
        labels = f["mechanism"].copy()

    assert passive.shape == active.shape
    assert passive.ndim == 3
    assert passive.shape[-1] == 2
    assert len(labels) == len(active)
    assert active.shape[1] == 51

    return passive, active, labels


def get_features(passive, active, horizon, view):
    # Estimate pre-intervention velocity only
    # from public, noisy passive observations.
    t_passive = np.arange(passive.shape[1]) / HZ
    centered_t = t_passive - t_passive.mean()

    centered_p = (
        passive - passive.mean(axis=1, keepdims=True)
    )

    velocities = np.sum(
        centered_p * centered_t[None, :, None],
        axis=1
    ) / np.sum(centered_t ** 2)

    # Elapsed time after intervention.
    elapsed = np.arange(horizon + 1) / HZ

    # Displacement relative to passive extrapolation.
    response = (
        active[:, :horizon + 1, :]
        - active[:, :1, :]
        - velocities[:, None, :] *
          elapsed[None, :, None]
    )

    if view == "A_only":
        selected = response[:, :, 0]
    elif view == "B_only":
        selected = response[:, :, 1]
    else:
        selected = response

    return selected.reshape(len(active), -1)


def main():
    p_train, a_train, y_train = load_split("train")
    p_val, a_val, y_val = load_split("val")

    print("Train episodes:", len(y_train))
    print("Val episodes:", len(y_val))
    print("Classes:", np.unique(y_train).tolist())

    results = []

    print(
        "\nview          horizon   val_acc"
        "   macro_f1   elastic_recall"
    )

    for view in VIEWS:
        for h in HORIZONS:
            x_train = get_features(
                p_train, a_train, h, view
            )
            x_val = get_features(
                p_val, a_val, h, view
            )

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    random_state=SEED
                )
            )

            model.fit(x_train, y_train)
            pred = model.predict(x_val)

            acc = float(
                accuracy_score(y_val, pred)
            )
            macro_f1 = float(
                f1_score(
                    y_val, pred,
                    average="macro",
                    zero_division=0
                )
            )

            classes = model.named_steps[
                "logisticregression"
            ].classes_

            recalls = recall_score(
                y_val, pred,
                labels=classes,
                average=None,
                zero_division=0
            )

            per_class = {
                str(c): float(r)
                for c, r in zip(classes, recalls)
            }

            elastic_recall = next(
                (
                    v for k, v in per_class.items()
                    if k.lower() in ("elastic", "2")
                ),
                None
            )

            record = {
                "view": view,
                "horizon_frames": h,
                "horizon_seconds": h / HZ,
                "val_accuracy": acc,
                "macro_f1": macro_f1,
                "per_class_recall": per_class,
                "elastic_recall": elastic_recall
            }
            results.append(record)

            er = (
                f"{elastic_recall:.3f}"
                if elastic_recall is not None
                else "n/a"
            )

            print(
                f"{view:<13} "
                f"{h:>3}       "
                f"{acc:.3f}     "
                f"{macro_f1:.3f}      {er}"
            )

    output = {
        "protocol": "P2-C1 horizon validation study",
        "seed": SEED,
        "data_used": ["train", "val"],
        "test_accessed": False,
        "model": "StandardScaler + LogisticRegression(C=1)",
        "sampling_hz": HZ,
        "results": results,
        "note": (
            "Supervised diagnostic study only. "
            "Not active experiment selection or "
            "visual concept revision."
        )
    }

    path = OUT / "p2c1_horizon_val.json"
    path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8"
    )

    # Zero-window observations contain no response
    # and must remain at the 3-class chance level.
    for r in results:
        if r["horizon_frames"] == 0:
            assert abs(
                r["val_accuracy"] - 1 / 3
            ) < 1e-8, "Zero-window audit failed"

    print("\nP2-C1 validation audit PASS")
    print("Saved:", path)


if __name__ == "__main__":
    main()
