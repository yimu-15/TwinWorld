
"""
P3-A3C: Selective-risk and failure attribution.

Read-only analysis of P3-A3B.
No MuJoCo, no refitting, no threshold changes.

Outputs:
- Per-mechanism decision confusion
- Coverage and accepted-only E4 risk
- Rejected-sample fallback prediction diagnostics
- Scene-level paired bootstrap differences
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results" / "p3a3b"

AUDIT = SOURCE / "p3a3b_development_audit_PRIVATE.json"
REPORT = SOURCE / "p3a3b_development_report.json"

SEED = 20261124

ARMS = (
    "pulse_1p5",
    "pulse_2p25",
    "sweep_2p25",
)

TESTS = (
    "pulse_2p75",
    "oscillation_2p25",
)

KNOWN = (
    "independent",
    "rigid",
    "elastic",
)

IN_LIBRARY = {
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

OUT_OF_LIBRARY = "saturating"

METHODS = (
    "known_refit",
    "unrestricted_refit",
    "gated_revision",
)


def error_mm(row, method, test=None):
    scores = row["methods"][method]["E4_RMS_m"]

    if test is not None:
        return float(scores[test]) * 1000.0

    return float(np.mean([
        scores[t] for t in TESTS
    ])) * 1000.0


def scene_ci(rows, delta_fn, seed, n_boot=3000):
    """
    Paired scene-cluster bootstrap.
    Lower differences mean better predictions.
    """
    if not rows:
        return None

    groups = defaultdict(list)

    for row in rows:
        groups[row["scene_id"]].append(
            float(delta_fn(row))
        )

    values = np.asarray([
        np.mean(groups[k])
        for k in sorted(groups)
    ])

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        len(values),
        size=(n_boot, len(values)),
    )

    draws = values[indices].mean(axis=1)

    return {
        "mean_delta_mm": round(
            float(values.mean()), 4
        ),
        "scene_CI95_mm": [
            round(float(v), 4)
            for v in np.quantile(
                draws, [0.025, 0.975]
            )
        ],
        "scenes": len(values),
    }


def show_split(split, rows):
    print("\n" + "=" * 70)
    print("SPLIT:", split)
    print("TOTAL EPISODES:", len(rows))
    print("=" * 70)

    for arm in ARMS:
        arm_rows = [
            r for r in rows
            if r["arm"] == arm
        ]

        print("\nARM:", arm)

        truth_groups = {
            truth: [
                r for r in arm_rows
                if r["truth_PRIVATE_EVAL_ONLY"]
                == truth
            ]
            for truth in (
                KNOWN
                + tuple(IN_LIBRARY)
                + (OUT_OF_LIBRARY,)
            )
        }

        print("\nPer-mechanism gated decisions:")

        for truth, subset in truth_groups.items():
            if not subset:
                continue

            counts = Counter(
                (
                    r["methods"]["gated_revision"]["decision"],
                    r["methods"]["gated_revision"]["kind"]
                )
                for r in subset
            )

            print(
                f"  {truth:12s}",
                f"N={len(subset):2d}",
                dict(counts)
            )

        known = [
            r for r in arm_rows
            if r["truth_PRIVATE_EVAL_ONLY"] in KNOWN
        ]

        in_library = [
            r for r in arm_rows
            if r["truth_PRIVATE_EVAL_ONLY"]
            in IN_LIBRARY
        ]

        out_library = truth_groups[OUT_OF_LIBRARY]

        print("\nCoverage and selective prediction risk:")

        for method in METHODS:
            accepted = [
                r for r in arm_rows
                if r["methods"][method]["decision"]
                != "inconclusive"
            ]

            rejected = [
                r for r in arm_rows
                if r["methods"][method]["decision"]
                == "inconclusive"
            ]

            false_edit = sum(
                r["methods"][method]["decision"] == "edit"
                for r in known
            )

            correct_edit = sum(
                (
                    r["methods"][method]["decision"]
                    == "edit"
                )
                and (
                    r["methods"][method]["kind"]
                    == IN_LIBRARY[
                        r["truth_PRIVATE_EVAL_ONLY"]
                    ]
                )
                for r in in_library
            )

            print(f"\n  METHOD: {method}")

            print(
                "    coverage:",
                f"{len(accepted)}/{len(arm_rows)}"
            )

            print(
                "    known false edit:",
                f"{false_edit}/{len(known)}"
            )

            if in_library:
                print(
                    "    correct in-library edit:",
                    f"{correct_edit}/{len(in_library)}"
                )

            if accepted:
                risk = float(np.mean([
                    error_mm(r, method)
                    for r in accepted
                ]))

                matched_known_risk = float(np.mean([
                    error_mm(r, "known_refit")
                    for r in accepted
                ]))

                print(
                    "    accepted-only E4 RMS(mm):",
                    round(risk, 4)
                )

                print(
                    "    known-refit RMS on SAME "
                    "accepted episodes(mm):",
                    round(matched_known_risk, 4)
                )

            if rejected:
                fallback = float(np.mean([
                    error_mm(r, method)
                    for r in rejected
                ]))

                print(
                    "    abstained fallback RMS(mm):",
                    round(fallback, 4)
                )

                print(
                    "    WARNING: fallback is not "
                    "an accepted prediction."
                )

        print("\nPaired E4 comparisons:")

        if in_library:
            for method in (
                "gated_revision",
                "unrestricted_refit"
            ):
                result = scene_ci(
                    in_library,
                    lambda r: (
                        error_mm(r, method)
                        - error_mm(r, "known_refit")
                    ),
                    SEED + 1
                )

                print(
                    "  In-library",
                    method,
                    "minus Known-Refit:",
                    result
                )

        if known:
            result = scene_ci(
                known,
                lambda r: (
                    error_mm(r, "gated_revision")
                    - error_mm(r, "unrestricted_refit")
                ),
                SEED + 2
            )

            print(
                "  Known Gated minus Unrestricted:",
                result
            )

        if out_library:
            result = scene_ci(
                out_library,
                lambda r: (
                    error_mm(r, "gated_revision")
                    - error_mm(r, "known_refit")
                ),
                SEED + 3
            )

            print(
                "  Out-of-library Gated minus Known:",
                result
            )


def main():
    if not AUDIT.exists():
        raise FileNotFoundError(AUDIT)

    if not REPORT.exists():
        raise FileNotFoundError(REPORT)

    audit = json.loads(
        AUDIT.read_text(encoding="utf-8")
    )

    report = json.loads(
        REPORT.read_text(encoding="utf-8")
    )

    for split in ("development", "mass_shift"):
        rows = audit[split]

        n_modes = 6 if split == "development" else 3

        expected = (
            report["scene_counts"][split]
            * n_modes
            * len(ARMS)
        )

        assert len(rows) == expected, (
            split, len(rows), expected
        )

        keys = {
            (
                r["scene_id"],
                r["truth_PRIVATE_EVAL_ONLY"],
                r["arm"]
            )
            for r in rows
        }

        assert len(keys) == len(rows)

        for row in rows:
            assert all(
                method in row["methods"]
                for method in METHODS
            )

            assert all(
                np.isfinite(error_mm(row, method))
                for method in METHODS
            )

        show_split(split, rows)

    print("\nP3-A3C READ-ONLY SELECTIVE AUDIT PASS")
    print("No new MuJoCo simulations.")
    print("No threshold tuning.")
    print("No files modified.")


if __name__ == "__main__":
    main()
