"""Read-only export of TwinWorld figure data V2.

Does NOT rerun experiments, retrain models, modify originals, or touch E4.
Only READS original npz/report JSONs and WRITES new files under paper_figure_data_v2/.
"""
import json, os, hashlib, csv, glob

ROOT = r"D:\OnShouldersGiants\TwinWorld_P0"
OUT = os.path.join(ROOT, "experiments", "p2d1", "results", "paper_figure_data_v2")

SUBDIRS = [
    "F1_trajectories", "F2_method_architecture", "F3_passive_active",
    "F4_physical_prior", "F5_amplitude", "F6_classical_baselines",
    "S1_fit_vs_selection", "S2_open_world", "S3_solver_audit",
    "S4_adaptive_action", "S5_prior_mismatch", "source_manifest",
]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dirs():
    os.makedirs(OUT, exist_ok=True)
    for s in SUBDIRS:
        os.makedirs(os.path.join(OUT, s), exist_ok=True)

def dump_npz_to_csv(npz_path, out_dir, prefix):
    """Read npz and dump each array to a CSV + a meta.json. Pure raw export."""
    import numpy as np
    d = np.load(npz_path, allow_pickle=True)
    meta = {"source": npz_path, "keys": {}}
    for k in d.files:
        a = d[k]
        shape = list(a.shape)
        meta["keys"][k] = {"shape": shape, "dtype": str(a.dtype)}
        flat = a.reshape(-1)
        csv_path = os.path.join(out_dir, f"{prefix}__{k}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"index", f"{k}"])
            for i, v in enumerate(flat):
                if isinstance(v, (int, float, complex, bool)):
                    w.writerow([i, v])
                else:
                    w.writerow([i, str(v)])
        meta["keys"][k]["csv"] = os.path.basename(csv_path)
        meta["keys"][k]["sha256"] = sha256_file(csv_path)
    with open(os.path.join(out_dir, f"{prefix}__meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def copy_report(src, dst_dir, dst_name, evidence, exp_id):
    if not os.path.exists(src):
        return {"missing": True, "source": src}
    with open(src, "rb") as f:
        data = f.read()
    dst = os.path.join(dst_dir, dst_name)
    with open(dst, "wb") as f:
        f.write(data)
    return {
        "missing": False, "source": src, "dst": os.path.relpath(dst, OUT),
        "sha256": sha256_file(src), "evidence": evidence, "exp_id": exp_id,
    }

def main():
    ensure_dirs()
    provenance = {"exported_files": []}

    # ---- F1 trajectories: dump npz -> csv ----
    f1 = os.path.join(OUT, "F1_trajectories")
    npz_specs = [
        ("data/train_public.npz", "P0_train_public"),
        ("data/val_public.npz", "P0_val_public"),
        ("data/test_public.npz", "P0_test_public"),
        ("data/train_private_labels.npz", "P0_train_labels"),
        ("data/val_private_labels.npz", "P0_val_labels"),
        ("data/test_private_labels.npz", "P0_test_labels"),
        ("outputs/p1b/p1b_audit_only.npz", "P1B_audit"),
        ("outputs/p2a/p2a_audit_only.npz", "P2A_audit"),
        ("outputs/p2b/train_observations.npz", "P2B_train_obs"),
        ("outputs/p2b/val_observations.npz", "P2B_val_obs"),
        ("outputs/p2b/test_observations.npz", "P2B_test_obs"),
        ("outputs/p2b/train_private.npz", "P2B_train_private"),
        ("outputs/p2b/val_private.npz", "P2B_val_private"),
        ("outputs/p2b/test_private.npz", "P2B_test_private"),
        ("outputs/p2d0/p2d0_audit_only.npz", "P2D0_audit"),
        ("experiments/p2d1/results/p2d1a_audit_only.npz", "P2D1A_audit"),
        ("experiments/p2d1/results/p2d1b/p2d1b_audit_only.npz", "P2D1B_audit"),
    ]
    for rel, prefix in npz_specs:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            dump_npz_to_csv(p, f1, prefix)
            provenance["exported_files"].append({
                "figure": "F1", "file": prefix, "source": rel,
                "type": "trajectory_export", "evidence": "raw",
            })

    # ---- F3..F6 / S1..S5: verbatim copy report JSON ----
    report_specs = [
        # (relative source, dest dir, dest name, evidence class, exp_id)
        ("outputs/p0_baselines.json", "F3_passive_active", "P0_baselines.json", "formal", "03"),
        ("outputs/p1b/p1b_report.json", "F3_passive_active", "P1B_report.json", "formal", "06"),
        ("outputs/p1c/p1c_report.json", "F3_passive_active", "P1C_report.json", "formal", "07"),
        ("outputs/p1c2/p1c2_report.json", "F3_passive_active", "P1C2_report.json", "audit", "08"),
        ("outputs/p1c3/p1c3_report.json", "F3_passive_active", "P1C3_report.json", "formal", "09"),
        ("outputs/p2a/p2a_report.json", "F3_passive_active", "P2A_report.json", "smoke", "10"),
        ("outputs/p2b/audit.json", "F3_passive_active", "P2B_audit.json", "formal", "11"),
        ("outputs/p2b/manifest.json", "F3_passive_active", "P2B_manifest.json", "formal", "11"),
        ("outputs/p2c0/p2c0_baselines.json", "F3_passive_active", "P2C0_baselines.json", "formal", "12"),
        ("outputs/p2c1/p2c1_horizon_val.json", "F3_passive_active", "P2C1_horizon.json", "dev", "13"),
        ("outputs/p2d0/p2d0_report.json", "F3_passive_active", "P2D0_report.json", "audit", "14"),

        ("experiments/p2d1/results/p4m2b2b/p4m2b2b_fresh_report.json", "F4_physical_prior", "79_p4m2b2b_report.json", "formal", "79"),
        ("experiments/p2d1/results/p4m2c2e/p4m2c2e_fresh_report.json", "F4_physical_prior", "86_p4m2c2e_report.json", "formal", "86"),

        ("experiments/p2d1/results/p4m2d1/p4m2d1_fresh_report.json", "F5_amplitude", "88_p4m2d1_report.json", "formal", "88"),
        ("experiments/p2d1/results/p4m2d2b/p4m2d2b_v2_fresh_report.json", "F5_amplitude", "90_p4m2d2b_v2_report.json", "formal", "90"),

        ("experiments/p2d1/results/p4b2b/p4b2b_fresh_report.json", "F6_classical_baselines", "106_p4b2b_report.json", "formal", "106"),
        ("experiments/p2d1/results/p4b3c/p4b3c_fresh_report.json", "F6_classical_baselines", "110_p4b3c_report.json", "formal", "110"),

        ("experiments/p2d1/results/p4g2b/p4g2b_closure_report.json", "S1_fit_vs_selection", "102_p4g2b_report.json", "audit", "102"),

        ("experiments/p2d1/results/p3a6h/p3a6h_fresh_report.json", "S2_open_world", "47_p3a6h_report.json", "formal", "47"),
        ("experiments/p2d1/results/p3a6i/p3a6i_fresh_report.json", "S2_open_world", "48_p3a6i_report.json", "formal", "48"),
        ("experiments/p2d1/results/p3a6k/p3a6k_fresh_report.json", "S2_open_world", "50_p3a6k_report.json", "formal", "50"),

        ("experiments/p2d1/results/p4s1a/p4s1a_diagnostic_report.json", "S3_solver_audit", "91_p4s1a_report.json", "diagnostic", "91"),
        ("experiments/p2d1/results/p4s1b/p4s1b_fresh_report.json", "S3_solver_audit", "92_p4s1b_report.json", "formal", "92"),
        ("experiments/p2d1/results/p4s1c/p4s1c_solver_closure_report.json", "S3_solver_audit", "93_p4s1c_report.json", "audit", "93"),

        ("experiments/p2d1/results/p3a8c2b/p3a8c2b_fresh_report.json", "S4_adaptive_action", "71_p3a8c2b_report.json", "formal", "71"),

        ("experiments/p2d1/results/p4m2c0/p4m2c0_dev_report.json", "S5_prior_mismatch", "80_p4m2c0_report.json", "dev", "80"),
        ("experiments/p2d1/results/p4m2c1/p4m2c1_dev_report.json", "S5_prior_mismatch", "81_p4m2c1_report.json", "audit", "81"),
    ]
    for rel, dsub, dname, evidence, exp_id in report_specs:
        r = copy_report(os.path.join(ROOT, rel), os.path.join(OUT, dsub), dname, evidence, exp_id)
        r["figure"] = dsub.split("_")[0]
        r["type"] = "verbatim_report_copy"
        provenance["exported_files"].append(r)

    # ---- source manifest SHA256 ----
    manifest = {"sha256": {}, "notes": "Computed read-only over original source files referenced by this export."}
    src_paths = [os.path.join(ROOT, rel) for rel, *_ in npz_specs] + \
                [os.path.join(ROOT, rel) for rel, *_ in report_specs] + \
                [os.path.join(ROOT, "scripts/twincore.py"),
                 os.path.join(ROOT, "docs/PROTOCOL_V1.md")]
    for p in src_paths:
        if os.path.exists(p):
            manifest["sha256"][os.path.relpath(p, ROOT)] = sha256_file(p)
        else:
            manifest["sha256"][os.path.relpath(p, ROOT)] = "MISSING"
    with open(os.path.join(OUT, "source_manifest", "sha256_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(OUT, "source_manifest", "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    print("EXPORT_DONE")

if __name__ == "__main__":
    main()
