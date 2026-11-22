
from pathlib import Path
from runpy import run_path
from collections import Counter, defaultdict
import argparse
import json

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a3a"

SEED = 20261122

KNOWN = (
    "independent",
    "rigid",
    "elastic"
)

IN_LIBRARY = {
    "deadzone": "deadzone",
    "nonlinear": "cubic"
}

HELD_OUT = "saturating"

MODES = KNOWN + tuple(IN_LIBRARY) + (HELD_OUT,)

ARMS = (
    "pulse_1p5",
    "pulse_2p25",
    "sweep_2p25"
)

TESTS = (
    "pulse_2p75",
    "oscillation_2p25"
)

MIN_MARGIN = 2.0


def fit_pair(api, probes, nfev):
    known = api["fit_library"](
        probes, KNOWN, nfev
    )[0]

    edit = api["fit_library"](
        probes,
        ("deadzone", "cubic"),
        nfev
    )[0]

    return {
        "known": known,
        "edit": edit,
        "margin": float(
            known["bic"] - edit["bic"]
        )
    }


def make_dataset(
    sim,
    api,
    profiles,
    n,
    modes,
    seed,
    nfev,
    shifted=False,
    evaluation=False
):
    srng = np.random.default_rng(seed)
    nrng = np.random.default_rng(seed + 100000)

    commands = {
        "e1": profiles["pulse"] * 0.25,
        "e2": profiles["pulse"] * 0.50,

        "pulse_1p5":
            profiles["pulse"] * 1.50,

        "pulse_2p25":
            profiles["pulse"] * 2.25,

        "sweep_2p25":
            profiles["sweep"] * 2.25,

        "pulse_2p75":
            profiles["pulse"] * 2.75,

        "oscillation_2p25":
            profiles["oscillation"] * 2.25
    }

    names = (
        ("e1", "e2")
        + ARMS
        + (TESTS if evaluation else ())
    )

    rows = []
    passive_max = 0.0
    a_max = {k: 0.0 for k in names}

    for sid in range(n):
        scene = sim["sample_scene"](srng)

        if shifted:
            scene["mass_a"] = float(
                srng.uniform(1.8, 2.3)
            )
            scene["mass_b"] = float(
                srng.uniform(1.8, 2.3)
            )

        model = sim["make_model"](scene)

        first_passive = {}
        first_a = {}

        for mode in modes:
            probes = {}
            costs = {}

            for name in names:
                passive, active, work, impulse = (
                    sim["simulate"](
                        model,
                        scene,
                        mode,
                        commands[name]
                    )
                )

                if name in first_passive:
                    passive_max = max(
                        passive_max,
                        float(np.max(np.abs(
                            passive
                            - first_passive[name]
                        )))
                    )

                    a_max[name] = max(
                        a_max[name],
                        float(np.max(np.abs(
                            active[:, 0]
                            - first_a[name]
                        )))
                    )
                else:
                    first_passive[name] = passive
                    first_a[name] = active[:, 0].copy()

                p, a = api["noisy_observation"](
                    passive,
                    active,
                    nrng
                )

                probes[name] = (
                    api["RevisionProbe"](p, a)
                )

                costs[name] = {
                    "work_J": float(work),
                    "impulse_Ns": float(impulse)
                }

            early = fit_pair(
                api,
                [probes["e1"], probes["e2"]],
                nfev
            )

            arms = {
                arm: fit_pair(
                    api,
                    [
                        probes["e1"],
                        probes["e2"],
                        probes[arm]
                    ],
                    nfev
                )
                for arm in ARMS
            }

            rows.append({
                "scene_id": sid,
                "truth_PRIVATE": mode,
                "early": early,
                "arms": arms,
                "costs": costs,
                "tests_PRIVATE": {
                    k: probes[k]
                    for k in TESTS
                } if evaluation else {}
            })

        if passive_max > 1e-8:
            raise RuntimeError(
                f"Passive mismatch: {passive_max}"
            )

    return rows, {
        "passive_max_m": passive_max,
        "A_gap_max_m_by_action": a_max
    }


def calibrate(rows):
    cuts = {}

    for arm in ("early",) + ARMS:
        margin = [
            (
                r["early"]["margin"]
                if arm == "early"
                else r["arms"][arm]["margin"]
            )
            for r in rows
        ]

        residual = [
            (
                r["early"]["known"]["fit_rms_m"]
                if arm == "early"
                else r["arms"][arm]["known"]["fit_rms_m"]
            )
            for r in rows
        ]

        cuts[arm] = {
            "margin": max(
                MIN_MARGIN,
                float(np.quantile(
                    margin, 0.95
                ))
            ),
            "known_fit_rms_m": float(
                np.quantile(
                    residual, 0.95
                )
            )
        }

    return cuts


def decide(fits, cut):
    choice = (
        "edit"
        if fits["margin"] > cut["margin"]
        else "known"
    )

    selected = fits[choice]

    if (
        selected["fit_rms_m"]
        > cut["known_fit_rms_m"]
    ):
        return "inconclusive", fits["known"]

    return (
        "edit" if choice == "edit"
        else "keep"
    ), selected


def cluster_stat(rows, fn, seed, repetitions=1500):
    groups = defaultdict(list)

    for r in rows:
        groups[r["scene_id"]].append(
            float(fn(r))
        )

    if not groups:
        return None

    vals = np.array([
        np.mean(groups[k])
        for k in sorted(groups)
    ])

    rng = np.random.default_rng(seed)

    sample = vals[
        rng.integers(
            len(vals),
            size=(repetitions, len(vals))
        )
    ].mean(axis=1)

    return {
        "mean": float(np.mean(vals)),
        "ci95": list(map(
            float,
            np.quantile(
                sample,
                [0.025, 0.975]
            )
        )),
        "n_scenes": len(vals),
        "n_episodes": len(rows)
    }


def evaluate(rows, cuts, api, seed):
    flat = []

    for r in rows:
        record = {
            "scene_id": r["scene_id"],
            "truth_PRIVATE_EVAL_ONLY":
                r["truth_PRIVATE"],
            "methods": {}
        }

        for arm in ("early",) + ARMS:
            fits = (
                r["early"]
                if arm == "early"
                else r["arms"][arm]
            )

            decision, chosen = decide(
                fits,
                cuts[arm]
            )

            errors = {
                test: float(
                    api["prediction_rms"](
                        r["tests_PRIVATE"][test],
                        chosen
                    )
                )
                for test in TESTS
            }

            record["methods"][arm] = {
                "decision": decision,
                "chosen_kind": chosen["kind"],
                "margin_BIC": fits["margin"],

                "work_E3_J": (
                    0.0 if arm == "early"
                    else r["costs"][arm]["work_J"]
                ),

                "E4_m": errors
            }

        flat.append(record)

    summary = {}

    for j, arm in enumerate(
        ("early",) + ARMS
    ):
        get = lambda r: r["methods"][arm]

        known = [
            r for r in flat
            if r["truth_PRIVATE_EVAL_ONLY"]
            in KNOWN
        ]

        edits = [
            r for r in flat
            if r["truth_PRIVATE_EVAL_ONLY"]
            in IN_LIBRARY
        ]

        held = [
            r for r in flat
            if r["truth_PRIVATE_EVAL_ONLY"]
            == HELD_OUT
        ]

        summary[arm] = {
            "decisions": dict(Counter(
                get(r)["decision"]
                for r in flat
            )),

            "known_false_edit": cluster_stat(
                known,
                lambda r:
                    get(r)["decision"] == "edit",
                seed + j * 10 + 1
            ),

            "known_inconclusive": cluster_stat(
                known,
                lambda r:
                    get(r)["decision"]
                    == "inconclusive",
                seed + j * 10 + 2
            ),

            "in_library_correct_edit": cluster_stat(
                edits,
                lambda r: (
                    get(r)["decision"] == "edit"
                    and get(r)["chosen_kind"]
                    == IN_LIBRARY[
                        r["truth_PRIVATE_EVAL_ONLY"]
                    ]
                ),
                seed + j * 10 + 3
            ),

            "in_library_wrong_edit": cluster_stat(
                edits,
                lambda r: (
                    get(r)["decision"] == "edit"
                    and get(r)["chosen_kind"]
                    != IN_LIBRARY[
                        r["truth_PRIVATE_EVAL_ONLY"]
                    ]
                ),
                seed + j * 10 + 4
            ),

            "out_of_library_forced_edit": cluster_stat(
                held,
                lambda r:
                    get(r)["decision"] == "edit",
                seed + j * 10 + 5
            ),

            "out_of_library_inconclusive": cluster_stat(
                held,
                lambda r:
                    get(r)["decision"]
                    == "inconclusive",
                seed + j * 10 + 6
            ),

            "E3_mean_abs_work_J": float(
                np.mean([
                    get(r)["work_E3_J"]
                    for r in flat
                ])
            ),

            "E4_mean_rms_mm": {
                test: 1000 * float(np.mean([
                    get(r)["E4_m"][test]
                    for r in flat
                ]))
                for test in TESTS
            }
        }

        for kind in MODES:
            kindrows = [
                r for r in flat
                if r["truth_PRIVATE_EVAL_ONLY"]
                == kind
            ]

            if kindrows:
                summary[arm].setdefault(
                    "E4_by_truth_mm", {}
                )[kind] = {
                    test: 1000 * float(np.mean([
                        get(r)["E4_m"][test]
                        for r in kindrows
                    ]))
                    for test in TESTS
                }

    for arm in ARMS:
        summary[arm][
            "E4_paired_minus_early_mm"
        ] = cluster_stat(
            flat,
            lambda r: 1000 * np.mean([
                (
                    r["methods"][arm]
                    ["E4_m"][test]
                    -
                    r["methods"]["early"]
                    ["E4_m"][test]
                )
                for test in TESTS
            ]),
            seed + 100 + ARMS.index(arm)
        )

    return summary, flat


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    parser.add_argument(
        "--self-test-only",
        action="store_true"
    )

    args = parser.parse_args()

    # Pure decision tests
    fake = {
        "margin": 10,
        "known": {"fit_rms_m": 0.005},
        "edit": {"fit_rms_m": 0.003}
    }

    cut = {
        "margin": 5,
        "known_fit_rms_m": 0.010
    }

    assert decide(fake, cut)[0] == "edit"

    fake["margin"] = 1
    assert decide(fake, cut)[0] == "keep"

    print(
        "P3-A3A decision self-test PASS "
        "(not MuJoCo validation)"
    )

    if args.self_test_only:
        return

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    original_force = (
        sim["simulate"]
        .__globals__["internal_force"]
    )

    # Add a simulator-only held-out mechanism.
    # No modification of the original source file.
    def patched_force(mode, r, vr, scene):
        if mode == HELD_OUT:
            linear = (
                -scene["k"] * r
                -scene["c"] * vr
            )

            cap = (
                0.4
                + 0.3 * scene["mass_b"]
            )

            return float(np.clip(
                linear, -cap, cap
            ))

        return original_force(
            mode, r, vr, scene
        )

    sim["simulate"].__globals__[
        "internal_force"
    ] = patched_force

    profiles = sim["make_profiles"]()

    if args.smoke:
        ncal, ndev, nshift, nfev = (
            2, 2, 2, 12
        )
    else:
        ncal, ndev, nshift, nfev = (
            12, 18, 18, 40
        )

    cal, ca = make_dataset(
        sim, api, profiles,
        ncal, KNOWN,
        SEED + 11, nfev
    )

    dev, da = make_dataset(
        sim, api, profiles,
        ndev, MODES,
        SEED + 22, nfev,
        evaluation=True
    )

    shift, sa = make_dataset(
        sim, api, profiles,
        nshift, KNOWN,
        SEED + 33, nfev,
        shifted=True,
        evaluation=True
    )

    cuts = calibrate(cal)

    dev_stats, dev_audit = evaluate(
        dev, cuts, api, SEED + 100
    )

    shift_stats, shift_audit = evaluate(
        shift, cuts, api, SEED + 200
    )

    physics_groups = (ca, da, sa)

    action_names = set().union(*[
        p["A_gap_max_m_by_action"].keys()
        for p in physics_groups
    ])

    report = {
        "protocol": (
            "P3-A3A under-excited E1/E2; "
            "E3 model refit"
        ),

        "development_only": True,
        "seed": SEED,
        "calibration_known_only": True,

        "counts": {
            "calibration": ncal,
            "development": ndev,
            "mass_shift": nshift
        },

        "E1_E2": [
            "pulse_0p25",
            "pulse_0p50"
        ],

        "E3_arms": list(ARMS),
        "E4_holdout": list(TESTS),

        "in_library_new": IN_LIBRARY,

        "out_of_library_truth_PRIVATE":
            HELD_OUT,

        "thresholds": cuts,
        "development": dev_stats,
        "mass_shift": shift_stats,

        "physics": {
            "max_passive_gap_m": max(
                p["passive_max_m"]
                for p in physics_groups
            ),

            "max_A_mode_gap_m_by_action": {
                k: max(
                    p["A_gap_max_m_by_action"].get(
                        k, 0.0
                    )
                    for p in physics_groups
                )
                for k in sorted(action_names)
            }
        },

        "limitations": [
            "Saturating force is held out "
            "from the permitted model library.",

            "E3 is fixed per arm; "
            "this is NOT adaptive action selection.",

            "Weak E1/E2 was chosen after "
            "earlier development audits.",

            "BIC uses correlated trajectory "
            "samples and is a heuristic.",

            "E4 B predictions condition on "
            "observed E4 A.",

            "Development only, "
            "not a final independent test.",

            "Inconclusive is based on fitting "
            "residual and may fail on unseen laws."
        ]
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    suffix = (
        "smoke"
        if args.smoke
        else "development"
    )

    report_path = (
        OUT / f"p3a3a_{suffix}_report.json"
    )

    audit_path = (
        OUT /
        f"p3a3a_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    audit_path.write_text(
        json.dumps(
            {
                "development": dev_audit,
                "mass_shift": shift_audit
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        f"P3-A3A {suffix} | "
        f"cal/dev/shift="
        f"{ncal}/{ndev}/{nshift}"
    )

    for group, stats in (
        ("development", dev_stats),
        ("mass_shift", shift_stats)
    ):
        print(group)

        for arm, m in stats.items():
            correct = m[
                "in_library_correct_edit"
            ]

            forced = m[
                "out_of_library_forced_edit"
            ]

            print(
                f"  {arm:17s} "
                f"cost={m['E3_mean_abs_work_J']:.3f}J",
                f"false_edit="
                f"{m['known_false_edit']['mean']:.3f}",
                f"correct_in_library="
                f"{None if correct is None else round(correct['mean'], 3)}",
                f"out_library_forced="
                f"{None if forced is None else round(forced['mean'], 3)}",
                "E4mm=",
                {
                    k: round(v, 3)
                    for k, v in m[
                        "E4_mean_rms_mm"
                    ].items()
                }
            )

    print(
        "P3-A3A DEVELOPMENT DIAGNOSTICS COMPLETE "
        "(not concept discovery)"
    )

    print("Saved:", report_path)


if __name__ == "__main__":
    main()
