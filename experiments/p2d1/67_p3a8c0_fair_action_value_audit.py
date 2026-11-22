
"""
P3-A8C0 — Fair Action-Value Audit

Read-only audit of formal P3-A8B2 records.

Questions:
1. What if Always-High stops when the
   initial hypothesis set is already empty?
2. Does Affine improve actual identifiability?
3. Does disagreement ranking predict the
   true value of High versus Reverse?

No simulator execution.
No threshold fitting or adjustment.
Truth labels used only for PRIVATE audit.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SOURCE = RESULTS / "p3a8b2"
OUT = RESULTS / "p3a8c0"

REPORT = SOURCE / "p3a8b2_fresh_report.json"
AUDIT = SOURCE / "p3a8b2_fresh_audit_PRIVATE.json"

BANDS = ("original", "above_original_range")

KNOWN = (
    "independent", "rigid",
    "elastic", "deadzone", "nonlinear"
)

MODES = KNOWN + ("saturating",)

TRUE_LAW = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}


def read(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def count(rows, method, outcome):
    return sum(
        r["outcomes_PRIVATE"][method] == outcome
        for r in rows
    )


def audit_band(rows, reference):
    if len(rows) != 216:
        raise RuntimeError(
            "Expected 216 formal cases"
        )

    seen = set()

    for r in rows:
        key = (
            r["scene_id"],
            r["truth_PRIVATE"]
        )

        if key in seen:
            raise RuntimeError(
                "Duplicate scene-mechanism pair"
            )

        seen.add(key)

    if len(set(
        r["scene_id"] for r in rows
    )) != 36:
        raise RuntimeError(
            "Expected 36 base scenes"
        )

    known = [
        r for r in rows
        if r["truth_PRIVATE"] in KNOWN
    ]

    unknown = [
        r for r in rows
        if r["truth_PRIVATE"] == "saturating"
    ]

    if len(known) != 180 or len(unknown) != 36:
        raise RuntimeError(
            "Incomplete mechanism matrix"
        )

    eligible = [
        r for r in rows
        if len(r["initial_set"]) > 0
    ]

    # The fair comparator performs High
    # only when the initial set is nonempty.
    fair_cost = 0.0

    for r in rows:
        if not r["initial_set"]:
            if r["final_sets"]["always_high"]:
                raise RuntimeError(
                    "Empty-set intersection violation"
                )

            fair_cost += r["costs_J"]["mid_only"]

        else:
            fair_cost += r["costs_J"]["always_high"]

    high_cost = sum(
        r["costs_J"]["always_high"]
        for r in rows
    )

    affine_cost = sum(
        r["costs_J"]["affine_planner"]
        for r in rows
    )

    old_cost = sum(
        r["costs_J"]["old_planner"]
        for r in rows
    )

    changes = [
        r for r in eligible
        if r["selected_actions"]["old_planner"]
        != r["selected_actions"]["affine_planner"]
    ]

    actual_best = {
        id(r): max(
            r["actual_A_scores_PRIVATE"],
            key=r["actual_A_scores_PRIVATE"].get,
        )
        for r in eligible
    }

    old_match = sum(
        r["selected_actions"]["old_planner"]
        == actual_best[id(r)]
        for r in eligible
    )

    affine_match = sum(
        r["selected_actions"]["affine_planner"]
        == actual_best[id(r)]
        for r in eligible
    )

    # Compare policy outcomes, not just
    # model-disagreement scores.
    high_only_correct = sum(
        r["outcomes_PRIVATE"]["always_high"]
        == "correct_singleton"
        and r["outcomes_PRIVATE"]["always_reverse"]
        != "correct_singleton"
        for r in known
    )

    reverse_only_correct = sum(
        r["outcomes_PRIVATE"]["always_reverse"]
        == "correct_singleton"
        and r["outcomes_PRIVATE"]["always_high"]
        != "correct_singleton"
        for r in known
    )

    high_only_reject = sum(
        r["outcomes_PRIVATE"]["always_high"]
        == "unknown_empty"
        and r["outcomes_PRIVATE"]["always_reverse"]
        != "unknown_empty"
        for r in unknown
    )

    reverse_only_reject = sum(
        r["outcomes_PRIVATE"]["always_reverse"]
        == "unknown_empty"
        and r["outcomes_PRIVATE"]["always_high"]
        != "unknown_empty"
        for r in unknown
    )

    # Exact paired comparison of true-law
    # membership in the two final sets.
    same_true_membership = sum(
        (
            TRUE_LAW[r["truth_PRIVATE"]]
            in r["final_sets"]["always_high"]
        ) == (
            TRUE_LAW[r["truth_PRIVATE"]]
            in r["final_sets"]["affine_planner"]
        )
        for r in known
    )

    same_correct_unique = sum(
        (
            r["outcomes_PRIVATE"]["always_high"]
            == "correct_singleton"
        ) == (
            r["outcomes_PRIVATE"]["affine_planner"]
            == "correct_singleton"
        )
        for r in known
    )

    result = {
        "n": len(rows),
        "eligible": len(eligible),
        "initial_empty": len(rows) - len(eligible),

        "fair_high_if_nonempty_cost_J": fair_cost,
        "always_high_cost_J": high_cost,
        "old_planner_cost_J": old_cost,
        "affine_planner_cost_J": affine_cost,

        "affine_saving_vs_fair_high_J":
            fair_cost - affine_cost,

        "affine_saving_vs_fair_high_fraction":
            (fair_cost - affine_cost) / fair_cost,

        "old_to_affine_changes": len(changes),

        "old_to_affine_direction": dict(Counter(
            r["selected_actions"]["old_planner"]
            + " -> "
            + r["selected_actions"]["affine_planner"]
            for r in changes
        )),

        "actual_A_high_best": sum(
            actual_best[id(r)] == "high"
            for r in eligible
        ),

        "old_actual_A_ranking_match": old_match,
        "affine_actual_A_ranking_match":
            affine_match,

        "known_high_only_correct":
            high_only_correct,

        "known_reverse_only_correct":
            reverse_only_correct,

        "unknown_high_only_empty":
            high_only_reject,

        "unknown_reverse_only_empty":
            reverse_only_reject,

        "known_same_correct_unique":
            same_correct_unique,

        "known_same_true_membership":
            same_true_membership,

        "known_affine_correct":
            count(
                known, "affine_planner",
                "correct_singleton"
            ),

        "known_high_correct":
            count(
                known, "always_high",
                "correct_singleton"
            ),

        "unknown_affine_empty":
            count(
                unknown, "affine_planner",
                "unknown_empty"
            ),

        "unknown_high_empty":
            count(
                unknown, "always_high",
                "unknown_empty"
            ),
    }

    # Cross-check input summary before
    # accepting our derived statistics.
    expected = reference["policies"]

    assert result["known_affine_correct"] == (
        expected["affine_planner"][
            "known_correct_singleton"
        ]
    )

    assert result["known_high_correct"] == (
        expected["always_high"][
            "known_correct_singleton"
        ]
    )

    assert result["unknown_affine_empty"] == (
        expected["affine_planner"]["unknown_empty"]
    )

    assert result["unknown_high_empty"] == (
        expected["always_high"]["unknown_empty"]
    )

    private_changes = [
        {
            "scene_id": r["scene_id"],
            "truth_PRIVATE": r["truth_PRIVATE"],
            "old_action":
                r["selected_actions"]["old_planner"],
            "affine_action":
                r["selected_actions"]["affine_planner"],
            "old_outcome_PRIVATE":
                r["outcomes_PRIVATE"]["old_planner"],
            "affine_outcome_PRIVATE":
                r["outcomes_PRIVATE"]["affine_planner"],
        }
        for r in changes
    ]

    return result, private_changes


def main():
    report = read(REPORT)
    audit = read(AUDIT)

    if report["smoke"]:
        raise RuntimeError(
            "Expected formal experiment data"
        )

    summaries = {}
    private = {}

    for band in BANDS:
        summaries[band], private[band] = (
            audit_band(
                audit[band],
                report["summary"][band]
            )
        )

        s = summaries[band]

        print("\nBAND:", band)
        print(" Eligible:", s["eligible"])
        print(
            " Fair High cost:",
            round(
                s["fair_high_if_nonempty_cost_J"],
                3
            ),
        )
        print(
            " Affine cost:",
            round(
                s["affine_planner_cost_J"], 3
            ),
        )
        print(
            " Affine saving vs fair High:",
            round(
                s[
                    "affine_saving_vs_fair_high_fraction"
                ] * 100, 3
            ),
            "%",
        )
        print(
            " Action changes:",
            s["old_to_affine_changes"]
        )
        print(
            " Ranking matches (old/affine):",
            s["old_actual_A_ranking_match"],
            "/",
            s["affine_actual_A_ranking_match"],
        )
        print(
            " Known High-only / Reverse-only:",
            s["known_high_only_correct"],
            "/",
            s["known_reverse_only_correct"],
        )
        print(
            " Unknown High-only / Reverse-only:",
            s["unknown_high_only_empty"],
            "/",
            s["unknown_reverse_only_empty"],
        )

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    public_path = OUT / (
        "p3a8c0_fair_action_value_report.json"
    )

    private_path = OUT / (
        "p3a8c0_changes_PRIVATE.json"
    )

    public = {
        "protocol":
            "P3-A8C0 read-only fair baseline audit",
        "source_sha256": {
            REPORT.name: digest(REPORT),
            AUDIT.name: digest(AUDIT),
        },
        "summary": summaries,
        "new_simulations": 0,
        "new_thresholds": 0,
        "E4_accessed_by_this_script": False,
        "limitations": [
            "Retrospective analysis of A8B2",
            "Not independent test replication",
            "Only current set-intersection semantics",
            "Private labels only for offline audit",
            "Two bands share base scenes",
            "Mechanical work is a cost proxy",
        ],
    }

    public_path.write_text(
        json.dumps(
            public, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    private_path.write_text(
        json.dumps(
            private, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\nP3-A8C0 AUDIT COMPLETE")
    print("Saved:", public_path)
    print("Saved:", private_path)


if __name__ == "__main__":
    main()
