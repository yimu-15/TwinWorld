
"""
P3-A4B: Hypothesis-Aware Intervention Replay.

Policy:
- High-confidence Rigid -> Stop
- Independent-like -> Weak Pulse
- Other evidence -> Strong Pulse

Reads existing P3-A4A counterfactual records.
No MuJoCo and no model refitting.

IMPORTANT:
The rule was designed after examining P3-A4A.
Results are exploratory, NOT independent validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results" / "p3a4a"
OUT = HERE / "results" / "p3a4b"

SEED = 20261127

KNOWN = {
    "independent",
    "rigid",
    "elastic",
}

NEW = {
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

TESTS = (
    "pulse_2p75",
    "oscillation_2p25",
)

BASELINES = (
    "always_weak",
    "always_strong",
    "stop_or_strong",
)

RIGID_BIC_CUTOFF = -10.0


def choose_action(evidence, early_rms_cut):
    """
    Nonprivileged decision using E1/E2 evidence.

    No true mechanism, E3 outcome or E4 outcome.
    """
    kind = evidence["known_kind"]
    bic = float(evidence["bic_advantage"])
    rms = float(evidence["known_fit_rms_m"])

    if (
        kind == "rigid"
        and bic <= RIGID_BIC_CUTOFF
        and rms <= early_rms_cut
    ):
        return "none"

    if kind == "independent":
        return "pulse_1p5"

    return "pulse_2p25"


def check_policy():
    threshold = 0.009

    evidence = {
        "known_kind": "rigid",
        "bic_advantage": -20.0,
        "known_fit_rms_m": 0.005,
    }

    assert choose_action(
        evidence, threshold
    ) == "none"

    evidence["known_kind"] = "independent"

    assert choose_action(
        evidence, threshold
    ) == "pulse_1p5"

    evidence["known_kind"] = "elastic"

    assert choose_action(
        evidence, threshold
    ) == "pulse_2p25"

    evidence["known_kind"] = "rigid"
    evidence["bic_advantage"] = -2.0

    assert choose_action(
        evidence, threshold
    ) == "pulse_2p25"

    print(
        "P3-A4B ACTION POLICY SELF-TEST PASS"
    )


def rms_mm(policy, test=None):
    errors = policy["E4_RMS_m"]

    if test is not None:
        return 1000.0 * float(errors[test])

    return 1000.0 * float(
        np.mean([
            errors[t]
            for t in TESTS
        ])
    )


def ci_by_scene(
    rows,
    fn,
    seed,
    n_boot=3000,
):
    """
    Cluster bootstrap over independent scenes.
    """
    if not rows:
        return None

    groups = defaultdict(list)

    for row in rows:
        groups[row["scene_id"]].append(
            float(fn(row))
        )

    values = np.asarray([
        np.mean(groups[k])
        for k in sorted(groups)
    ])

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )

    draws = values[indices].mean(axis=1)

    return {
        "mean": float(values.mean()),
        "ci95_scene": [
            float(v)
            for v in np.quantile(
                draws,
                [0.025, 0.975],
            )
        ],
        "scenes": len(values),
        "episodes": len(rows),
    }


def replay(rows, early_cut):
    """
    Select action BEFORE accessing stored
    counterfactual outcomes.

    Offline evaluation uses only the chosen
    action's result for the adaptive policy.
    """
    outputs = []

    action_sources = {
        "none": "early_gate",
        "pulse_1p5": "always_weak",
        "pulse_2p25": "always_strong",
    }

    for row in rows:

        action = choose_action(
            row["early_evidence"],
            early_cut,
        )

        source = action_sources[action]

        selected = row["policies"][source]

        assert selected["E3_action"] == action

        outputs.append({
            "scene_id": row["scene_id"],

            "truth_PRIVATE_EVAL_ONLY":
                row["truth_PRIVATE_EVAL_ONLY"],

            "early_known_kind":
                row["early_evidence"]["known_kind"],

            "action": action,

            "selected": selected,

            "baselines": {
                name: row["policies"][name]
                for name in BASELINES
            },
        })

    return outputs


def summarize(rows, seed):
    known = [
        r for r in rows
        if r["truth_PRIVATE_EVAL_ONLY"]
        in KNOWN
    ]

    new = [
        r for r in rows
        if r["truth_PRIVATE_EVAL_ONLY"]
        in NEW
    ]

    out_library = [
        r for r in rows
        if r["truth_PRIVATE_EVAL_ONLY"]
        == "saturating"
    ]

    stopped = [
        r for r in rows
        if r["action"] == "none"
    ]

    accepted = [
        r for r in rows
        if r["selected"]["decision"]
        != "inconclusive"
    ]

    report = {
        "n": len(rows),

        "actions": dict(Counter(
            r["action"]
            for r in rows
        )),

        "stopped_truth_PRIVATE": dict(Counter(
            r["truth_PRIVATE_EVAL_ONLY"]
            for r in stopped
        )),

        "stopped_wrong_structure_PRIVATE": sum(
            r["truth_PRIVATE_EVAL_ONLY"]
            != r["selected"]["kind"]
            for r in stopped
        ),

        "decisions": dict(Counter(
            r["selected"]["decision"]
            for r in rows
        )),

        "known_false_edit": [
            sum(
                r["selected"]["decision"] == "edit"
                for r in known
            ),
            len(known),
        ],

        "correct_in_library": [
            sum(
                r["selected"]["decision"] == "edit"
                and r["selected"]["kind"]
                == NEW[
                    r["truth_PRIVATE_EVAL_ONLY"]
                ]
                for r in new
            ),
            len(new),
        ],

        "out_library_inconclusive": [
            sum(
                r["selected"]["decision"]
                == "inconclusive"
                for r in out_library
            ),
            len(out_library),
        ],

        "coverage": [
            len(accepted),
            len(rows),
        ],

        "mean_E3_work_J": float(np.mean([
            r["selected"]["E3_work_proxy_J"]
            for r in rows
        ])),

        "E4_all_fallback_diagnostic_mm": {
            test: float(np.mean([
                rms_mm(r["selected"], test)
                for r in rows
            ]))
            for test in TESTS
        },

        "E4_accepted_only_mm": (
            float(np.mean([
                rms_mm(r["selected"])
                for r in accepted
            ]))
            if accepted else None
        ),

        "always_strong_on_same_accepted_mm": (
            float(np.mean([
                rms_mm(
                    r["baselines"]["always_strong"]
                )
                for r in accepted
            ]))
            if accepted else None
        ),

        "paired_vs_always_strong": {},
    }

    comparisons = {
        "E3_work_J": lambda r: (
            r["selected"]["E3_work_proxy_J"]
            -
            r["baselines"]["always_strong"]
            ["E3_work_proxy_J"]
        ),

        "E4_both_mm": lambda r: (
            rms_mm(r["selected"])
            -
            rms_mm(
                r["baselines"]["always_strong"]
            )
        ),

        "E4_pulse_mm": lambda r: (
            rms_mm(r["selected"], TESTS[0])
            -
            rms_mm(
                r["baselines"]["always_strong"],
                TESTS[0],
            )
        ),

        "E4_oscillation_mm": lambda r: (
            rms_mm(r["selected"], TESTS[1])
            -
            rms_mm(
                r["baselines"]["always_strong"],
                TESTS[1],
            )
        ),
    }

    for i, (name, fn) in enumerate(
        comparisons.items()
    ):
        report["paired_vs_always_strong"][
            name
        ] = ci_by_scene(
            rows,
            fn,
            seed + i,
        )

    return report


def main():
    check_policy()

    audit_path = (
        SOURCE /
        "p3a4a_development_audit_PRIVATE.json"
    )

    report_path = (
        SOURCE /
        "p3a4a_development_report.json"
    )

    if (
        not audit_path.exists()
        or not report_path.exists()
    ):
        raise FileNotFoundError(
            "Missing P3-A4A formal results: "
            + str(SOURCE)
        )

    audit = json.loads(
        audit_path.read_text(
            encoding="utf-8"
        )
    )

    previous = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    early_cut = float(
        previous["calibration_known_only"]
        ["early"]["known_fit_rms_m"]
    )

    output = {
        "study": (
            "P3-A4B hypothesis-aware "
            "intervention replay"
        ),

        "development_only": True,
        "post_hoc_rule": True,

        "early_known_rms_cut_m": early_cut,

        "limitations": [
            "The rule was designed after "
            "examining P3-A4A results.",

            "This is retrospective replay, "
            "not independent validation.",

            "Actions use E1/E2 evidence only.",

            "Counterfactual trajectories were "
            "generated by P3-A4A.",

            "Saturating is excluded from "
            "the model library.",

            "Inconclusive E4 predictions "
            "are fallback diagnostics.",

            "Prediction uses observed E4 A.",

            "No learned policy and no RGB input.",
        ],

        "splits": {},
    }

    for i, split in enumerate(
        ("development", "mass_shift")
    ):

        rows = audit[split]

        n_modes = (
            6 if split == "development"
            else 3
        )

        expected = (
            previous["scene_counts"][split]
            * n_modes
        )

        if len(rows) != expected:
            raise ValueError(
                f"{split}: expected {expected}, "
                f"got {len(rows)}"
            )

        keys = {
            (
                r["scene_id"],
                r["truth_PRIVATE_EVAL_ONLY"],
            )
            for r in rows
        }

        if len(keys) != len(rows):
            raise ValueError(
                f"Duplicate episodes in {split}"
            )

        selected = replay(
            rows,
            early_cut,
        )

        output["splits"][split] = summarize(
            selected,
            SEED + i * 100,
        )

        stats = output["splits"][split]

        print("\n" + "=" * 65)
        print("SPLIT:", split)
        print("=" * 65)

        print("Actions:", stats["actions"])

        print(
            "Stopped truths (PRIVATE):",
            stats["stopped_truth_PRIVATE"],
        )

        print(
            "Wrong stops (PRIVATE):",
            stats["stopped_wrong_structure_PRIVATE"],
        )

        print(
            "Known false edits:",
            stats["known_false_edit"],
        )

        print(
            "Correct new edits:",
            stats["correct_in_library"],
        )

        print(
            "Saturating inconclusive:",
            stats["out_library_inconclusive"],
        )

        print(
            "Coverage:",
            stats["coverage"],
        )

        print(
            "Mean E3 work (J):",
            round(
                stats["mean_E3_work_J"],
                4,
            ),
        )

        print(
            "E4 diagnostic RMS (mm):",
            {
                k: round(v, 4)
                for k, v in stats[
                    "E4_all_fallback_diagnostic_mm"
                ].items()
            },
        )

        print("\nPaired vs AlwaysStrong:")

        for name, comparison in stats[
            "paired_vs_always_strong"
        ].items():
            print(
                " ",
                name,
                comparison,
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        OUT /
        "p3a4b_replay_development_report.json"
    )

    result_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A4B RETROSPECTIVE REPLAY PASS"
    )

    print(
        "No MuJoCo, no training, "
        "no original data modified."
    )

    print(
        "POST-HOC DEVELOPMENT ONLY; "
        "NOT FINAL VALIDATION."
    )

    print("Saved:", result_path)


if __name__ == "__main__":
    main()
