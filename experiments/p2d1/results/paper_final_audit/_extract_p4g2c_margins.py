"""Enhance P4-G2C (#103) per-case CSV with full competition-margin scores.

Per case, for both arms (PAIRED_IDEAL_FIXED vs SHIFTED_REFIT, MID_PLUS_FULL):
winner, correct, per-candidate scores (elastic/deadzone/cubic), true-law RMS,
and relative margin. Verifies FIXED=347 / REFIT=327 and elastic 19/96 reversal.
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
elastic_reversal = 0
for r in rows:
    true = r["true_law_PRIVATE"]
    def arm_info(arm):
        dec = r["decisions"][arm]["MID_PLUS_FULL"]
        diag = r["post_selection_diagnostics_PRIVATE"][arm]["MID_PLUS_FULL"]
        s = dec.get("scores_mm_PRIVATE", {})
        return dec.get("winner"), diag.get("correct"), \
               s.get("elastic"), s.get("deadzone"), s.get("cubic"), \
               diag.get("true_law_rms_mm"), diag.get("relative_margin")
    fw, fc, fse, fsd, fsc, frms, fmarg = arm_info("PAIRED_IDEAL_FIXED")
    rw, rc, rse, rsd, rsc, rrms, rmarg = arm_info("SHIFTED_REFIT")
    fixed_corr += 1 if fc else 0
    refit_corr += 1 if rc else 0
    better_fit = 1 if rrms < frms else 0
    worse_sel = 1 if (rc is False and fc is True) else 0
    if true == "elastic" and better_fit and worse_sel:
        elastic_reversal += 1
    out.append({
        "scene_id": r["scene_id"], "true_law": true,
        "mode": r.get("mode_PRIVATE"), "condition": r.get("condition_PRIVATE"),
        "FIXED_winner": fw, "REFIT_winner": rw,
        "FIXED_correct": 1 if fc else 0, "REFIT_correct": 1 if rc else 0,
        "FIXED_score_elastic_mm": round(fse, 6) if fse is not None else None,
        "FIXED_score_deadzone_mm": round(fsd, 6) if fsd is not None else None,
        "FIXED_score_cubic_mm": round(fsc, 6) if fsc is not None else None,
        "REFIT_score_elastic_mm": round(rse, 6) if rse is not None else None,
        "REFIT_score_deadzone_mm": round(rsd, 6) if rsd is not None else None,
        "REFIT_score_cubic_mm": round(rsc, 6) if rsc is not None else None,
        "FIXED_true_law_rms_mm": round(frms, 6) if frms is not None else None,
        "REFIT_true_law_rms_mm": round(rrms, 6) if rrms is not None else None,
        "FIXED_relative_margin": round(fmarg, 6) if fmarg is not None else None,
        "REFIT_relative_margin": round(rmarg, 6) if rmarg is not None else None,
        "REFIT_better_fit": better_fit, "REFIT_worse_selection": worse_sel,
    })

fields = list(out[0].keys())
with open(ROOT / "p4g2c_384case_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader(); w.writerows(out)

print("P4-G2C cases:", len(out))
print("FIXED correct:", fixed_corr, "(report 347) | REFIT correct:", refit_corr, "(report 327)")
print("elastic reversal (better fit + worse selection):", elastic_reversal, "(expected 19)")
print("columns:", len(fields))
