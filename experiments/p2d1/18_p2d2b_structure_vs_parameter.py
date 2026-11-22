r"""TwinWorld P2-D2B: distinguish known-parameter shift from unknown mechanisms.

Development-only experiment, NOT a confirmatory test or concept revision.
Dependencies: Giant Python 3.11, numpy, mujoco, scikit-learn, matplotlib
Run from project root:
.\Giant\Scripts\python.exe .\experiments\p2d1\18_p2d2b_structure_vs_parameter.py

Uses functions from 16_p2d1b_probe_diversity.py unchanged, but generates FRESH
independent physical scenes with disjoint seeds. No access to old P2-C0 test.
"""
from __future__ import annotations

from pathlib import Path
from runpy import run_path
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SIM_SCRIPT = ROOT / "experiments" / "p2d1" / "16_p2d1b_probe_diversity.py"
OUT = ROOT / "experiments" / "p2d1" / "results" / "p2d2b"
OUT.mkdir(parents=True, exist_ok=True)

sim_module = run_path(str(SIM_SCRIPT))
make_model = sim_module["make_model"]
simulate = sim_module["simulate"]
make_profiles = sim_module["make_profiles"]
MODES = list(sim_module["MODES"])
COMMAND = make_profiles()["pulse"]

KNOWN = (0, 1, 2)
UNKNOWN = (3, 4)
N_TRAIN_SCENES = 60
N_CAL_SCENES = 30
N_ID_SCENES = 36
N_SHIFT_SCENES = 36
N_AUGMENT = 3
NOISE_STD = 0.005
SAMPLING_DT = 0.02
SEEDS = {"train": 42001, "cal": 42002, "id_dev": 42003,
         "mass_shift_dev": 42004, "sensor": 42005}
SCORES = ("max_softmax_novelty", "knn_feature_distance")
VIEWS = ("A_only", "B_only", "A_plus_B")


def sample_scene(rng: np.random.Generator, mass_shift=False):
    # Structural-unknown data is sampled from exactly the same parameter support
    # as ID known data. Mass-shift data differs only in mass ranges.
    lo, hi = ((1.7, 2.3) if mass_shift else (0.8, 1.6))
    return {
        "mass_a": float(rng.uniform(lo, hi)),
        "mass_b": float(rng.uniform(lo, hi)),
        "velocity": float(rng.uniform(0.05, 0.25)),
        "k": float(rng.uniform(5.0, 16.0)),
        "c": float(rng.uniform(0.5, 2.2)),
        "gap": float(rng.uniform(0.015, 0.12)),
        "k3": float(rng.uniform(100.0, 850.0)),
    }


def generate_split(split, n_scenes, selected_modes, mass_shift=False):
    rng = np.random.default_rng(SEEDS[split])
    records = []
    max_passive_diff = 0.0
    for scene_id in range(n_scenes):
        params = sample_scene(rng, mass_shift=mass_shift)
        model = make_model(params)
        passive_by_mechanism = []
        for mode_index in selected_modes:
            mode = MODES[mode_index]
            passive, active, _, _ = simulate(model, params, mode, COMMAND)
            assert passive.shape == (51, 2) and active.shape == (51, 2)
            assert np.isfinite(passive).all() and np.isfinite(active).all()
            passive_by_mechanism.append(passive)
            records.append({
                "scene_id": int(scene_id),
                "mode": int(mode_index),
                "passive": passive,
                "active": active,
            })
        passive_by_mechanism = np.stack(passive_by_mechanism)
        max_passive_diff = max(max_passive_diff, float(np.max(np.abs(
            passive_by_mechanism - passive_by_mechanism[:1]))))
    if max_passive_diff >= 1e-8:
        raise AssertionError(f"{split}: passive equivalence failed: {max_passive_diff}")
    return records, max_passive_diff


def noisy_features(record, view, rng):
    passive = record["passive"] + rng.normal(0, NOISE_STD, (51, 2))
    active = record["active"] + rng.normal(0, NOISE_STD, (51, 2))
    t = np.arange(51, dtype=float) * SAMPLING_DT
    tc = t - t.mean()
    slopes = (tc[:, None] * (passive - passive.mean(axis=0))).sum(axis=0) / (tc * tc).sum()
    # Residual relative to pre-intervention motion; only accessible observations.
    response = active - passive[-1][None, :] - t[:, None] * slopes[None, :]
    if view == "A_only":
        return response[:, 0].astype(np.float64)
    if view == "B_only":
        return response[:, 1].astype(np.float64)
    return response.reshape(-1).astype(np.float64)


def featurize(records, view, rng, augment=1):
    x, y, group = [], [], []
    for rec in records:
        for _ in range(augment):
            x.append(noisy_features(rec, view, rng))
            y.append(rec["mode"])
            group.append(rec["scene_id"])
    return np.asarray(x), np.asarray(y, dtype=int), np.asarray(group, dtype=int)


def wilson_interval(k, n, z=1.96):
    if n <= 0:
        return [None, None]
    p = k / n
    denom = 1 + z*z/n
    midpoint = (p + z*z/(2*n)) / denom
    delta = (z / denom) * np.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return [float(max(0, midpoint - delta)), float(min(1, midpoint + delta))]


def assessment(scores, threshold, labels, desired_modes, reject_if_high=True):
    mask = np.isin(labels, desired_modes)
    rejected = scores[mask] > threshold if reject_if_high else scores[mask] < threshold
    n, k = len(rejected), int(rejected.sum())
    return {"rejection_rate": float(k / n), "rejected": k, "n": n,
            "wilson_95_ci": wilson_interval(k, n)}


def main():
    print("P2-D2B fresh-scene structure vs parameter shift audit")
    train, gtr = generate_split("train", N_TRAIN_SCENES, KNOWN)
    cal, gcal = generate_split("cal", N_CAL_SCENES, KNOWN)
    id_dev, gid = generate_split("id_dev", N_ID_SCENES, tuple(range(5)))
    shifted, gshift = generate_split("mass_shift_dev", N_SHIFT_SCENES, KNOWN, mass_shift=True)
    print("scene groups: train/cal/id-dev/mass-shift-dev =",
          N_TRAIN_SCENES, N_CAL_SCENES, N_ID_SCENES, N_SHIFT_SCENES)
    print("passive max diffs:", gtr, gcal, gid, gshift)

    report = {
        "protocol": "P2-D2B independent development: structure vs parameter shift",
        "development_only": True,
        "simulator": "MuJoCo via P2-D1B unmodified simulator functions",
        "seeds": SEEDS,
        "scene_counts": {"train": N_TRAIN_SCENES, "cal": N_CAL_SCENES,
                         "id_dev": N_ID_SCENES, "mass_shift_dev": N_SHIFT_SCENES},
        "sample_counts": {"train": len(train), "cal": len(cal),
                          "id_dev": len(id_dev), "mass_shift_dev": len(shifted)},
        "mechanisms": MODES,
        "known": [MODES[i] for i in KNOWN],
        "unseen_structure": [MODES[i] for i in UNKNOWN],
        "actions": ["pulse"],
        "public_observation": "noisy passive and active x/y positions at 50Hz",
        "noise_std_m": NOISE_STD,
        "mass_ranges_kg": {"known_id": [0.8, 1.6], "mass_shift": [1.7, 2.3]},
        "passive_max_gap_m": {"train": gtr, "cal": gcal, "id_dev": gid,
                              "mass_shift_dev": gshift},
        "notes": [
            "Scores and thresholds use known train/cal only; id-dev and mass-shift-dev are analysis sets.",
            "No previously analyzed P2-C0 test samples are loaded.",
            "Structural unknowns are deadzone/nonlinear; they share parameter sampling ranges with ID.",
            "Shifted known environments alter masses, not mechanism structure; sample sizes are scene groups.",
            "Model selection or protocol changes after seeing these dev results require a new held-out confirmatory set.",
            "Known mechanisms and unknown physical mechanisms can be observationally indistinguishable under this probe.",
            "This is rejection baseline evaluation, not structural theory revision or RGB visual reasoning."
        ],
        "results": {},
    }

    for idx, view in enumerate(VIEWS):
        rng = np.random.default_rng(SEEDS["sensor"] + idx * 11)
        xtr, ytr, _ = featurize(train, view, rng, N_AUGMENT)
        xcal, ycal, _ = featurize(cal, view, rng)
        xid, yid, _ = featurize(id_dev, view, rng)
        xshift, yshift, _ = featurize(shifted, view, rng)
        assert set(ytr) == set(KNOWN)
        scaler = StandardScaler().fit(xtr)
        ztr, zcal, zid, zshift = (scaler.transform(x) for x in (xtr, xcal, xid, xshift))
        classifier = LogisticRegression(C=1.0, max_iter=4000,
                                        random_state=SEEDS["train"]).fit(ztr, ytr)
        neighbors = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(ztr)

        def known_predictions(z):
            return classifier.predict(z)

        def novelty(z, method):
            if method == "max_softmax_novelty":
                return 1.0 - classifier.predict_proba(z).max(axis=1)
            return neighbors.kneighbors(z, return_distance=True)[0].mean(axis=1)

        view_report = {
            "known_id_accuracy": float(accuracy_score(yid[np.isin(yid, KNOWN)],
                                                      known_predictions(zid)[np.isin(yid, KNOWN)])),
            "mass_shift_known_accuracy": float(accuracy_score(yshift, known_predictions(zshift))),
            "methods": {}
        }
        for method in SCORES:
            s_cal = novelty(zcal, method)
            # Calibration uses known-only validation. No unknowns are used to set a threshold.
            threshold = float(np.quantile(s_cal, .95))
            s_id = novelty(zid, method)
            s_shift = novelty(zshift, method)
            known_id_mask = np.isin(yid, KNOWN)
            unknown_mask = np.isin(yid, UNKNOWN)
            auroc_id = float(roc_auc_score(unknown_mask.astype(int), s_id))
            auroc_shift = float(roc_auc_score(
                np.r_[np.zeros(len(s_shift)), np.ones(unknown_mask.sum())],
                np.r_[s_shift, s_id[unknown_mask]]))
            method_report = {
                "calibration_threshold_known_95th_percentile": threshold,
                "id_known_rejection": assessment(s_id, threshold, yid, KNOWN),
                "mass_shift_known_rejection": assessment(s_shift, threshold, yshift, KNOWN),
                "unknown_structural_rejection": assessment(s_id, threshold, yid, UNKNOWN),
                "unknown_deadzone_rejection": assessment(s_id, threshold, yid, (3,)),
                "unknown_nonlinear_rejection": assessment(s_id, threshold, yid, (4,)),
                "unknown_vs_id_known_AUROC": auroc_id,
                "unknown_vs_mass_shift_known_AUROC": auroc_shift,
            }
            view_report["methods"][method] = method_report
            print(
                f"{view:8s} {method:24s} "
                f"ID-known-FR={method_report['id_known_rejection']['rejection_rate']:.3f} "
                f"known-mass-shift-FR={method_report['mass_shift_known_rejection']['rejection_rate']:.3f} "
                f"unknown-reject={method_report['unknown_structural_rejection']['rejection_rate']:.3f} "
                f"AUC(unknown/ID)={auroc_id:.3f} "
                f"AUC(unknown/shift)={auroc_shift:.3f}")
        report["results"][view] = view_report

    out = OUT / "p2d2b_structure_vs_parameter_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("P2-D2B DEVELOPMENT DIAGNOSTICS COMPLETE")
    print("Saved:", out)


if __name__ == "__main__":
    main()
