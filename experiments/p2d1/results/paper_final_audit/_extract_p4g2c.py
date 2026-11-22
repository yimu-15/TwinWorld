"""Extract P4-G2C (#103) per-case data + verify fit-improvement vs selection reversal.

Reads 24 checkpoints; for each case pulls (MID_PLUS_FULL): winner + correct +
true_law_rms_mm + relative_margin for both arms (PAIRED_IDEAL_FIXED vs SHIFTED_REFIT).
Verifies reconstructed counts match the report primary/secondary endpoints, and
quantifies the reversal: SHIFTED_REFIT lowers true-law RMS (better fit) but
selects fewer mechanisms correctly.
"""
import csv, json
from pathlib import Path

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0\experiments\p2d1\results\p4g2c")

rows = []
for i in range(24):
    d = json.loads((ROOT / "checkpoints" / f"scene_{i:02d}.json").read_text(encoding="utf-8"))
    rows.extend(d["rows"])

out = []
fixed_corr = refit_corr = 0
elastic_n = 0
elastic_refit_better_fit = 0
elastic_refit_worse_sel = 0
elastic_refit_better_fit_and_worse_sel = 0
for r in rows:
    true = r["true_law_PRIVATE"]
    dec = r["decisions"]
    fw = dec["PAIRED_IDEAL_FIXED"]["MID_PLUS_FULL"]["winner"]
    rw = dec["SHIFTED_REFIT"]["MID_PLUS_FULL"]["winner"]
    diag = r["post_selection_diagnostics_PRIVATE"]
    fd = diag["PAIRED_IDEAL_FIXED"]["MID_PLUS_FULL"]
    rd = diag["SHIFTED_REFIT"]["MID_PLUS_FULL"]
    f_corr = 1 if fw == true else 0
    r_corr = 1 if rw == true else 0
    f_rms = fd["true_law_rms_mm"]
    r_rms = rd["true_law_rms_mm"]
    fixed_corr += f_corr; refit_corr += r_corr
    if true == "elastic":
        elastic_n += 1
        refit_better_fit = r_rms < f_rms
        refit_worse_sel = r_corr < f_corr
        if refit_better_fit: elastic_refit_better_fit += 1
        if refit_worse_sel: elastic_refit_worse_sel += 1
        if refit_better_fit and refit_worse_sel: elastic_refit_better_fit_and_worse_sel += 1
    out.append({
        "scene_id": r["scene_id"],
        "true_law": true,
        "mode": r.get("mode_PRIVATE"),
        "condition": r.get("condition_PRIVATE"),
        "FIXED_winner": fw,
        "REFIT_winner": rw,
        "FIXED_correct": f_corr,
        "REFIT_correct": r_corr,
        "FIXED_true_law_rms_mm": round(f_rms, 6),
        "REFIT_true_law_rms_mm": round(r_rms, 6),
        "REFIT_better_fit": 1 if r_rms < f_rms else 0,
        "REFIT_worse_selection": 1 if r_corr < f_corr else 0,
    })

with open(ROOT / "p4g2c_384case_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)

print("P4-G2C cases:", len(out))
print("  FIXED correct (MID_PLUS_FULL):", fixed_corr, "(report a_correct=347)")
print("  REFIT correct (MID_PLUS_FULL):", refit_corr, "(report b_correct=327)")
print("  elastic n:", elastic_n)
print("  elastic: refit better-fit cases:", elastic_refit_better_fit, "of", elastic_n)
print("  elastic: refit worse-selection cases:", elastic_refit_worse_sel, "of", elastic_n)
print("  elastic: BOTH better-fit AND worse-selection (reversal):", elastic_refit_better_fit_and_worse_sel, "of", elastic_n)
