"""
P3-A8C3A-0 — Counterfactual Action-Ranking Validity Audit

Goal:
Test whether pre-action structural disagreement predicts
the actual preference between High and Reverse-High.

Inputs:
    Frozen P3-A8C2B PRIVATE audit JSON.

Methods:
    old_disagreement
    affine_disagreement

Metrics:
    Pairwise action-ranking concordance
    Spearman rank correlation
    Scene-cluster bootstrap confidence intervals
    Disagreement-margin quartile analysis
    Action costs and preference coverage

IMPORTANT:
    Read-only retrospective analysis.
    No new MuJoCo simulation.
    No model refitting.
    No calibration changes.
    No E4.
    No future trajectory used for online planning.
    Ground truth is used ONLY for offline evaluation.

This is a diagnostic experiment, not an independent
confirmation of a newly selected policy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent

SOURCE = (
    HERE / "results" / "p3a8c2b"
    / "p3a8c2b_fresh_audit_PRIVATE.json"
)

OUT = HERE / "results" / "p3a8c3a0"

BANDS = (
    "original",
    "above_original_range",
)

KNOWN = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
)

TRUTH = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

METHODS = ("old", "affine")

BOOTSTRAP_SEED = 20281114

N_BOOTSTRAP = 2000


def load_source():

    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing source: {SOURCE}"
        )

    with SOURCE.open(
        "r", encoding="utf-8"
    ) as f:
        data = json.load(f)

    if set(data) != set(BANDS):
        raise RuntimeError(
            "Unexpected audit bands"
        )

    return data


def preference(high, reverse, mode):
    """
    Offline diagnostic preference.

    +1: High preferred
    -1: Reverse preferred
     0: No strict preference

    Truth is used ONLY here, never in
    pre-action disagreement predictions.
    """

    high = set(high)
    reverse = set(reverse)

    if mode == "saturating":

        high_safe = len(high) != 1
        reverse_safe = len(reverse) != 1

        if high_safe != reverse_safe:
            return 1 if high_safe else -1

        return 0

    truth = TRUTH[mode]

    h_contains = truth in high
    r_contains = truth in reverse

    if h_contains != r_contains:
        return 1 if h_contains else -1

    if h_contains and r_contains:

        if len(high) < len(reverse):
            return 1

        if len(reverse) < len(high):
            return -1

        return 0

    # Both actions exclude the true mechanism.
    # A false singleton is penalized.

    h_false = len(high) == 1
    r_false = len(reverse) == 1

    if h_false != r_false:
        return -1 if h_false else 1

    return 0


def prepare_rows(source_rows, method):

    rows = []

    for row in source_rows:

        if not row["first_set"]:
            continue

        if method not in row["scores_PRIVATE"]:
            continue

        score = row["scores_PRIVATE"][method]

        high_score = float(score["high"])

        reverse_score = float(
            score["reverse_high"]
        )

        if not np.isfinite(
            [high_score, reverse_score]
        ).all():
            raise RuntimeError(
                "Nonfinite planner disagreement"
            )

        margin = high_score - reverse_score

        # Reproduce C2B tie-break:
        # High is selected on exact ties.

        selected = 1 if margin >= 0 else -1

        truth_mode = row["truth_PRIVATE"]

        actual = preference(
            row["final_sets"]["fixed_high"],
            row["final_sets"]["fixed_reverse"],
            truth_mode,
        )

        costs = row["costs_J"]

        selected_cost = (
            costs["fixed_high"]
            if selected == 1
            else costs["fixed_reverse"]
        )

        rows.append({
            "scene_id": int(row["scene_id"]),
            "mode_PRIVATE": truth_mode,
            "band": row["band"],
            "method": method,
            "high_score": high_score,
            "reverse_score": reverse_score,
            "margin": float(margin),
            "abs_margin": float(abs(margin)),
            "selected": int(selected),
            "preference_PRIVATE": int(actual),
            "concordant_PRIVATE": (
                bool(selected == actual)
                if actual != 0 else None
            ),
            "selected_cost_J": float(
                selected_cost
            ),
            "fixed_high_cost_J": float(
                costs["fixed_high"]
            ),
            "fixed_reverse_cost_J": float(
                costs["fixed_reverse"]
            ),
        })

    return rows


def safe_spearman(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return None

    if np.unique(x).size < 2:
        return None

    if np.unique(y).size < 2:
        return None

    result = spearmanr(x, y)

    rho = float(result.statistic)

    return rho if np.isfinite(rho) else None


def calculate(rows):

    comparable = [
        r for r in rows
        if r["preference_PRIVATE"] != 0
    ]

    n = len(rows)
    k = len(comparable)

    if n == 0:
        return {
            "n": 0,
            "comparable": 0,
            "preference_coverage": None,
            "concordance": None,
            "spearman_rho": None,
        }

    concordance = (
        float(np.mean([
            r["concordant_PRIVATE"]
            for r in comparable
        ]))
        if comparable else None
    )

    # Correlation calculated on comparable
    # pairs only, to avoid no-preference ties
    # dominating the result.

    rho = safe_spearman(
        [r["margin"] for r in comparable],
        [
            r["preference_PRIVATE"]
            for r in comparable
        ],
    )

    return {
        "n": n,
        "comparable": k,
        "preference_coverage": float(k / n),
        "concordance": concordance,
        "spearman_rho": rho,
        "high_preferred": sum(
            r["preference_PRIVATE"] == 1
            for r in comparable
        ),
        "reverse_preferred": sum(
            r["preference_PRIVATE"] == -1
            for r in comparable
        ),
        "selected_high": sum(
            r["selected"] == 1
            for r in rows
        ),
        "selected_reverse": sum(
            r["selected"] == -1
            for r in rows
        ),
        "mean_selected_cost_J": float(
            np.mean([
                r["selected_cost_J"]
                for r in rows
            ])
        ),
        "mean_fixed_high_cost_J": float(
            np.mean([
                r["fixed_high_cost_J"]
                for r in rows
            ])
        ),
        "mean_fixed_reverse_cost_J": float(
            np.mean([
                r["fixed_reverse_cost_J"]
                for r in rows
            ])
        ),
    }


def bootstrap(rows, seed, n_bootstrap):

    groups = {}

    for row in rows:
        groups.setdefault(
            row["scene_id"], []
        ).append(row)

    scene_ids = sorted(groups)

    if len(scene_ids) < 10:
        return {
            "status": "insufficient_clusters"
        }

    rng = np.random.default_rng(seed)

    concordances = []
    correlations = []

    for _ in range(n_bootstrap):

        sampled_ids = rng.choice(
            scene_ids,
            size=len(scene_ids),
            replace=True,
        )

        sampled_rows = []

        for sid in sampled_ids:
            sampled_rows.extend(groups[sid])

        metrics = calculate(sampled_rows)

        c = metrics["concordance"]
        r = metrics["spearman_rho"]

        if c is not None:
            concordances.append(c)

        if r is not None:
            correlations.append(r)

    def ci(values):

        if not values:
            return None

        return [
            float(np.quantile(
                values, 0.025
            )),
            float(np.quantile(
                values, 0.975
            )),
        ]

    return {
        "cluster_unit": "base_scene_id",
        "n_clusters": len(scene_ids),
        "repetitions": n_bootstrap,
        "concordance_95CI": ci(
            concordances
        ),
        "spearman_95CI": ci(
            correlations
        ),
        "valid_concordance_draws": len(
            concordances
        ),
        "valid_spearman_draws": len(
            correlations
        ),
    }


def margin_quartiles(rows):

    comparable = [
        r for r in rows
        if r["preference_PRIVATE"] != 0
    ]

    if len(comparable) < 8:
        return []

    # Stable deterministic sorting.
    ordered = sorted(
        comparable,
        key=lambda r: (
            r["abs_margin"],
            r["scene_id"],
            r["mode_PRIVATE"],
        ),
    )

    groups = np.array_split(
        np.arange(len(ordered)), 4
    )

    results = []

    for qid, indices in enumerate(
        groups, start=1
    ):

        subset = [
            ordered[int(i)]
            for i in indices
        ]

        accuracy = float(np.mean([
            r["concordant_PRIVATE"]
            for r in subset
        ]))

        results.append({
            "quartile": qid,
            "n": len(subset),
            "min_abs_margin": float(
                min(r["abs_margin"] for r in subset)
            ),
            "max_abs_margin": float(
                max(r["abs_margin"] for r in subset)
            ),
            "concordance": accuracy,
        })

    return results


def evaluate_band(data, band, smoke):

    output = {}

    subsets = {
        "known": [
            row for row in data[band]
            if row["truth_PRIVATE"] in KNOWN
        ],
        "unknown": [
            row for row in data[band]
            if row["truth_PRIVATE"]
            == "saturating"
        ],
    }

    for subset_name, subset in subsets.items():

        output[subset_name] = {}

        for method in METHODS:

            rows = prepare_rows(
                subset, method
            )

            if smoke:
                rows = [
                    r for r in rows
                    if r["scene_id"] < 4
                ]

            metrics = calculate(rows)

            boot = (
                None if smoke
                else bootstrap(
                    rows,
                    BOOTSTRAP_SEED
                    + (0 if band == "original"
                       else 10000)
                    + (0 if method == "old"
                       else 1000),
                    N_BOOTSTRAP,
                )
            )

            quartiles = margin_quartiles(rows)

            result = {
                "metrics": metrics,
                "bootstrap": boot,
                "margin_quartiles": quartiles,
            }

            output[subset_name][method] = result

            print(
                f"\n[{band}]"
                f" [{subset_name}] [{method}]"
            )

            print(
                "N:", metrics["n"],
                "| Comparable:",
                metrics["comparable"],
            )

            print(
                "Coverage:",
                metrics["preference_coverage"]
            )

            print(
                "Concordance:",
                metrics["concordance"]
            )

            print(
                "Spearman:",
                metrics["spearman_rho"]
            )

            print(
                "High/Reverse preferred:",
                metrics.get("high_preferred"),
                metrics.get("reverse_preferred"),
            )

            print(
                "Bootstrap:", boot
            )

            print(
                "Margin quartiles:", quartiles
            )

    return output


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    data = load_source()

    print("=" * 72)
    print("P3-A8C3A-0 ACTION-RANKING AUDIT")
    print("=" * 72)

    print("Source:", SOURCE)
    print("Smoke:", args.smoke)
    print("No simulation.")
    print("No E4.")
    print("Frozen historical policy outputs.")

    results = {}

    for band in BANDS:

        results[band] = evaluate_band(
            data,
            band,
            args.smoke,
        )

    report = {
        "protocol":
            "P3-A8C3A-0 action-ranking validity",
        "smoke": args.smoke,
        "source": str(SOURCE),
        "prediction_methods": list(METHODS),
        "preference_definition":
            "Truth-preservation, then set size; "
            "false singleton avoidance",
        "results": results,
        "new_simulations": 0,
        "new_model_fits": 0,
        "historical_thresholds_changed": False,
        "E4_accessed": False,
        "limitations": [
            "Retrospective diagnostic",
            "Preferences use private truth",
            "Preference is not a complete utility",
            "No independent confirmation",
            "No online policy modification",
            "Shared scenes across mechanisms",
            "Cost excluded from preference",
            "No guarantee of decision calibration",
        ],
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report_path = (
        OUT /
        f"p3a8c3a0_{suffix}_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nP3-A8C3A-0 COMPLETE")
    print("Saved:", report_path)


if __name__ == "__main__":
    main()