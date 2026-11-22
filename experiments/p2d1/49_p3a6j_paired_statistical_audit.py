
"""
P3-A6J: Paired Statistical Audit

Inputs:
    P3-A6I frozen fresh report / private audit

Read-only:
    No fitting, simulation, retuning or file changes.

Statistical unit:
    Base scene (48), paired across mechanisms.

Bootstrap:
    Resample base scenes, retaining the five
    represented mechanisms within each scene.

Report gap bands separately. Do NOT count
shared base scenes as independent observations.
"""

from pathlib import Path
from collections import Counter
import json
import numpy as np
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results" / "p3a6i"

REPORT = ROOT / "p3a6i_fresh_report.json"
AUDIT = ROOT / "p3a6i_fresh_audit_PRIVATE.json"

KNOWN = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
)

BANDS = (
    "original",
    "above_original_range",
)

BOOTSTRAPS = 20000
SEED = 2026122901


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def correct(row, method):
    return (
        row[f"{method}_outcome_PRIVATE"]
        == "correct"
    )


def summarize_transitions(rows, label):
    wins = sum(
        not correct(r, "baseline")
        and correct(r, "candidate")
        for r in rows
    )

    losses = sum(
        correct(r, "baseline")
        and not correct(r, "candidate")
        for r in rows
    )

    both_correct = sum(
        correct(r, "baseline")
        and correct(r, "candidate")
        for r in rows
    )

    both_incorrect = (
        len(rows) - wins - losses - both_correct
    )

    discordant = wins + losses

    pvalue = (
        binomtest(
            min(wins, losses),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
        if discordant
        else 1.0
    )

    baseline = sum(
        correct(r, "baseline")
        for r in rows
    )

    candidate = sum(
        correct(r, "candidate")
        for r in rows
    )

    print("\n", label, "| N =", len(rows))
    print(
        "  Correct: ",
        baseline, "->", candidate
    )
    print(
        "  Incorrect -> correct:", wins
    )
    print(
        "  Correct -> incorrect:", losses
    )
    print(
        "  Both correct:", both_correct
    )
    print(
        "  Both incorrect:", both_incorrect
    )
    print(
        "  Exact McNemar p:",
        round(float(pvalue), 8)
    )

    for method in ("baseline", "candidate"):
        counts = Counter(
            r[f"{method}_outcome_PRIVATE"]
            for r in rows
        )
        print(
            " ", method, "outcomes:",
            dict(counts)
        )

    return {
        "wins": wins,
        "losses": losses,
        "p": float(pvalue),
    }


def scene_bootstrap(index, ids, band):
    """
    One base scene contains all five represented
    mechanisms. Bootstrap samples whole scenes.

    This CI describes variation under the
    generator's sampled scene distribution.
    """
    differences = []

    for sid in ids:
        group = [
            index[(sid, mode)]
            for mode in KNOWN
        ]

        old_count = sum(
            correct(r, "baseline")
            for r in group
        )

        new_count = sum(
            correct(r, "candidate")
            for r in group
        )

        differences.append(
            (new_count - old_count) / len(KNOWN)
        )

    differences = np.asarray(
        differences, dtype=float
    )

    rng = np.random.default_rng(
        SEED + (
            0 if band == "original" else 1
        )
    )

    selections = rng.integers(
        low=0,
        high=len(ids),
        size=(BOOTSTRAPS, len(ids)),
    )

    bootstrap = np.mean(
        differences[selections],
        axis=1
    )

    low, high = np.quantile(
        bootstrap,
        [0.025, 0.975],
    )

    print("\nSCENE-LEVEL BOOTSTRAP")
    print(
        "  Accuracy gain:",
        round(
            100 * float(np.mean(differences)),
            4,
        ),
        "percentage points"
    )
    print(
        "  95% percentile CI:",
        (
            round(100 * float(low), 4),
            round(100 * float(high), 4),
        ),
        "percentage points"
    )
    print(
        "  Positive scene contributions:",
        int(np.sum(differences > 0)),
        "/", len(ids)
    )
    print(
        "  Negative scene contributions:",
        int(np.sum(differences < 0)),
        "/", len(ids)
    )


def main():
    report = load(REPORT)
    audit = load(AUDIT)

    if (
        report["smoke"]
        or report["test_seed"] != 20261228
        or audit["smoke"]
    ):
        raise RuntimeError(
            "Expected frozen fresh P3-A6I results"
        )

    if report["thresholds"] != audit["thresholds"]:
        raise RuntimeError(
            "Threshold metadata mismatch"
        )

    for band in BANDS:
        rows = audit["audit_PRIVATE"][band]

        if len(rows) != 48 * 6:
            raise RuntimeError(
                f"Unexpected episode count in {band}"
            )

        idx = {}

        for row in rows:
            key = (
                int(row["scene_id"]),
                row["truth_PRIVATE"],
            )

            if key in idx:
                raise RuntimeError(
                    f"Duplicate episode: {key}"
                )

            idx[key] = row

        ids = sorted({
            int(r["scene_id"])
            for r in rows
        })

        if len(ids) != 48:
            raise RuntimeError(
                "Expected 48 unique base scenes"
            )

        for sid in ids:
            for mode in (
                *KNOWN,
                "saturating",
            ):
                if (sid, mode) not in idx:
                    raise RuntimeError(
                        f"Missing scene/mode: {sid}, {mode}"
                    )

        print("\n" + "=" * 72)
        print("P3-A6J BAND:", band)
        print("=" * 72)

        known_rows = [
            idx[(sid, mode)]
            for sid in ids
            for mode in KNOWN
        ]

        summarize_transitions(
            known_rows,
            "ALL REPRESENTED LAWS",
        )

        for mode in KNOWN:
            subset = [
                idx[(sid, mode)]
                for sid in ids
            ]

            summarize_transitions(
                subset,
                mode,
            )

        ood = [
            idx[(sid, "saturating")]
            for sid in ids
        ]

        old_rejected = sum(
            r["baseline_outcome_PRIVATE"]
            == "correct_reject"
            for r in ood
        )

        new_rejected = sum(
            r["candidate_outcome_PRIVATE"]
            == "correct_reject"
            for r in ood
        )

        print(
            "\nSATURATING REJECTION:",
            old_rejected, "->", new_rejected,
            "out of", len(ood)
        )

        scene_bootstrap(
            idx, ids, band
        )

    print(
        "\nP3-A6J PAIRED STATISTICAL "
        "AUDIT PASS"
    )

    print(
        "Read-only audit; no thresholds "
        "or policies changed."
    )


if __name__ == "__main__":
    main()
