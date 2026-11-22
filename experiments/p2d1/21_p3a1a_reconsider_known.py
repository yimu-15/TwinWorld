
from pathlib import Path
from runpy import run_path
import argparse
import json

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a1a"
OUT.mkdir(parents=True, exist_ok=True)

previous = run_path(str(
    HERE / "20_p3a0_parameter_refit.py"
))
sim = run_path(str(
    HERE / "16_p2d1b_probe_diversity.py"
))

Probe = previous["Probe"]
fit_candidates = previous["fit_candidates"]
add_noise = previous["noisy_observation"]

KNOWN = ("independent", "rigid", "elastic")
ALL = KNOWN + ("deadzone", "nonlinear")

SEED = 20261118
SIGMA = 0.005
PULSE = sim["make_profiles"]()["pulse"]

# Identification, challenge, held-out verification
AMPS = (1.0, 2.0, 1.5)


def joint_fit(probes, allowed):
    n = sum(len(p.y_b) for p in probes)
    best = None

    for kind in allowed:

        if kind == "elastic":
            best_opt = None

            for start in ([3.0, 1.0], [12.0, 2.0]):
                opt = least_squares(
                    lambda theta: np.concatenate([
                        (
                            p.predict("elastic", theta)
                            - p.y_b
                        ) / SIGMA
                        for p in probes
                    ]),
                    start,
                    bounds=([0.01, 0.001], [70.0, 15.0]),
                    max_nfev=70
                )

                if (
                    best_opt is None
                    or np.sum(opt.fun**2)
                    < np.sum(best_opt.fun**2)
                ):
                    best_opt = opt

            params = best_opt.x.tolist()
            loss = float(np.sum(best_opt.fun**2))
            complexity = 2

        else:
            params = None
            loss = float(sum(
                np.sum(
                    (
                        (
                            p.predict(kind)
                            - p.y_b
                        ) / SIGMA
                    ) ** 2
                )
                for p in probes
            ))
            complexity = 0

        bic = loss + complexity * np.log(n)

        if best is None or bic < best["bic"]:
            best = {
                "kind": kind,
                "params": params,
                "bic": float(bic)
            }

    return best


def heldout_error(fit, probe):
    prediction = probe.predict(
        fit["kind"], fit["params"]
    )

    return float(np.sqrt(np.mean(
        (prediction - probe.y_b) ** 2
    )))


def make_split(count, modes, seed, shift=False):
    scene_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 50000)

    rows = []

    for scene_id in range(count):
        scene = sim["sample_scene"](scene_rng)

        if shift:
            scene["mass_a"] = float(
                scene_rng.uniform(1.8, 2.3)
            )
            scene["mass_b"] = float(
                scene_rng.uniform(1.8, 2.3)
            )

        for mode in modes:
            model = sim["make_model"](scene)
            probes = []

            for amp in AMPS:
                passive, active, *_ = sim["simulate"](
                    model, scene, mode, PULSE * amp
                )

                p, a = add_noise(
                    passive, active, noise_rng
                )

                probes.append(Probe(p, a))

            # Initial choice from 1x evidence only
            first, _ = fit_candidates(probes[0])

            # Both methods can use 1x and 2x for fitting.
            fixed = joint_fit(
                probes[:2], [first]
            )

            reopened = joint_fit(
                probes[:2], KNOWN
            )

            # Third experiment is held out.
            rows.append({
                "scene_id": scene_id,
                "mechanism_PRIVATE_EVAL_ONLY": mode,
                "initial_kind": first,
                "fixed_kind": fixed["kind"],
                "reopened_kind": reopened["kind"],
                "fixed_validation_rms_m":
                    heldout_error(fixed, probes[2]),
                "reopened_validation_rms_m":
                    heldout_error(reopened, probes[2])
            })

    return rows


def cluster_rate(rows, fn, seed):
    by_scene = {}

    for row in rows:
        by_scene.setdefault(
            row["scene_id"], []
        ).append(float(fn(row)))

    data = np.array([
        np.mean(v) for v in by_scene.values()
    ])

    rng = np.random.default_rng(seed)

    draws = rng.choice(
        data,
        size=(2000, len(data)),
        replace=True
    ).mean(axis=1)

    return {
        "rate": float(np.mean(data)),
        "scene_bootstrap_CI95":
            np.quantile(
                draws, [0.025, 0.975]
            ).tolist(),
        "scenes": len(data),
        "episodes": len(rows)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true"
    )
    args = parser.parse_args()

    ncal, ndev, nshift = (
        (2, 2, 2) if args.smoke
        else (12, 18, 18)
    )

    print("P3-A1A known hypothesis reconsideration")

    cal = make_split(
        ncal, KNOWN, SEED + 11
    )
    dev = make_split(
        ndev, ALL, SEED + 22
    )
    shift = make_split(
        nshift, KNOWN, SEED + 33, shift=True
    )

    results = {}

    for method in ("fixed", "reopened"):
        key = method + "_validation_rms_m"

        threshold = float(np.quantile(
            [r[key] for r in cal], 0.95
        ))

        known = [
            r for r in dev
            if r["mechanism_PRIVATE_EVAL_ONLY"] in KNOWN
        ]
        unknown = [
            r for r in dev
            if r["mechanism_PRIVATE_EVAL_ONLY"] not in KNOWN
        ]

        results[method] = {
            "calibration_q95_m": threshold,

            "dev_known_false_rejection":
                cluster_rate(
                    known,
                    lambda r: r[key] > threshold,
                    SEED + 1
                ),

            "mass_shift_false_rejection":
                cluster_rate(
                    shift,
                    lambda r: r[key] > threshold,
                    SEED + 2
                ),

            "unknown_rejection":
                cluster_rate(
                    unknown,
                    lambda r: r[key] > threshold,
                    SEED + 3
                ),

            "known_structure_selection_accuracy":
                float(np.mean([
                    r[method + "_kind"]
                    == r["mechanism_PRIVATE_EVAL_ONLY"]
                    for r in known
                ]))
        }

    report = {
        "protocol":
            "P3-A1A known-hypothesis reconsideration",
        "development_only": True,
        "three_amplitudes": list(AMPS),
        "seed": SEED,
        "known_laws": list(KNOWN),
        "scene_groups": {
            "cal_known": ncal,
            "dev_all": ndev,
            "mass_shift_known": nshift
        },
        "results": results,
        "note": (
            "1x and 2x for model selection/refit; "
            "1.5x held out. No concept creation."
        )
    }

    suffix = (
        "smoke" if args.smoke
        else "development"
    )

    output = OUT / f"p3a1a_{suffix}_report.json"
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    private = OUT / (
        f"p3a1a_{suffix}_audit_PRIVATE.json"
    )
    private.write_text(
        json.dumps({
            "calibration": cal,
            "development": dev,
            "mass_shift": shift
        }, indent=2),
        encoding="utf-8"
    )

    for method, res in results.items():
        print(
            method,
            "threshold_mm",
            round(
                res["calibration_q95_m"] * 1000,
                2
            )
        )

        for name in (
            "dev_known_false_rejection",
            "mass_shift_false_rejection",
            "unknown_rejection"
        ):
            print(
                " ", name,
                round(res[name]["rate"], 3)
            )

        print(
            " known structure selection accuracy:",
            round(
                res["known_structure_selection_accuracy"],
                3
            )
        )

    print("P3-A1A DEVELOPMENT AUDIT COMPLETED")
    print("Saved:", output)


if __name__ == "__main__":
    main()
