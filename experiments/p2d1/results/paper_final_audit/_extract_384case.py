"""Extract 384-case per-case results from P4-B2B (#106) and P4-B3C (#110) checkpoints.

The report JSONs hold only aggregate counts; per-case decisions (winner law per
method, correctness, per-case call logs) live in checkpoints/scene_*.json `rows`.
This read-only script aggregates them into one CSV per experiment and verifies
the reconstructed correct counts match the report primary endpoints.
"""
import csv
import json
from pathlib import Path

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0\experiments\p2d1\results")

def load_scenes(dir_name):
    ck = ROOT / dir_name / "checkpoints"
    rows = []
    for i in range(24):
        p = ck / f"scene_{i:02d}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.extend(d["rows"])
    return rows

# ---------- P4-B3C (#110): TRF_CAP vs SLSQP_CAP, budget 200, WOO, MID_PLUS_FULL ----------
rows_b3c = load_scenes("p4b3c")
out_b3c = []
trf_corr = slsqp_corr = 0
trf_calls = slsqp_calls = 0
for r in rows_b3c:
    dec = r.get("decisions_PRIVATE", {})
    d200 = dec.get("200", {})
    trf_w = d200.get("TRF_CAP", {}).get("MID_PLUS_FULL", {}).get("winner")
    slsqp_w = d200.get("SLSQP_CAP", {}).get("MID_PLUS_FULL", {}).get("winner")
    true = r["true_law_PRIVATE"]
    trf_c = 1 if trf_w == true else 0
    slsqp_c = 1 if slsqp_w == true else 0
    trf_corr += trf_c
    slsqp_corr += slsqp_c
    # per-case total calls at budget 200 (sum over domain x law)
    fl = r.get("fit_logs_PRIVATE", {}).get("200", {})
    def sum_calls(solver):
        s = 0
        for dom, laws in fl.get(solver, {}).items():
            for law, rec in laws.items():
                s += rec.get("calls", 0)
        return s
    tc = sum_calls("TRF_CAP")
    sc = sum_calls("SLSQP_CAP")
    trf_calls += tc
    slsqp_calls += sc
    out_b3c.append({
        "scene_id": r["scene_id"],
        "band": r.get("band_PRIVATE"),
        "mode": r.get("mode_PRIVATE"),
        "condition": r.get("condition_PRIVATE"),
        "true_law": true,
        "TRF_CAP_winner_b200_WOO_FULL": trf_w,
        "SLSQP_CAP_winner_b200_WOO_FULL": slsqp_w,
        "TRF_correct": trf_c,
        "SLSQP_correct": slsqp_c,
        "TRF_calls_b200": tc,
        "SLSQP_calls_b200": sc,
    })

with open(ROOT / "p4b3c" / "p4b3c_384case_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_b3c[0].keys()))
    w.writeheader()
    w.writerows(out_b3c)

print("P4-B3C cases:", len(out_b3c))
print("  TRF correct (b200 WOO FULL):", trf_corr, "(report a_correct=321)")
print("  SLSQP correct (b200 WOO FULL):", slsqp_corr, "(report b_correct=324)")
print("  TRF per-case calls sum:", trf_calls, "| SLSQP per-case calls sum:", slsqp_calls)

# ---------- P4-B2B (#106): EQERR_SG / TRF_OUT / SLSQP, WOO, MID_PLUS_FULL ----------
rows_b2b = load_scenes("p4b2b")
out_b2b = []
eq_corr = trf_corr = slsqp_corr = 0
for r in rows_b2b:
    dec = r.get("decisions", {})
    eq_w = dec.get("EQERR_SG", {}).get("WOO", {}).get("MID_PLUS_FULL", {}).get("winner")
    trf_w = dec.get("TRF_OUT", {}).get("WOO", {}).get("MID_PLUS_FULL", {}).get("winner")
    slsqp_w = dec.get("SLSQP_MATCHED5_REFERENCE", {}).get("WOO", {}).get("MID_PLUS_FULL", {}).get("winner")
    true = r["true_law_PRIVATE"]
    eq_c = 1 if eq_w == true else 0
    trf_c = 1 if trf_w == true else 0
    slsqp_c = 1 if slsqp_w == true else 0
    eq_corr += eq_c; trf_corr += trf_c; slsqp_corr += slsqp_c
    out_b2b.append({
        "scene_id": r["scene_id"],
        "mode": r.get("mode_PRIVATE"),
        "condition": r.get("condition_PRIVATE"),
        "true_law": true,
        "EQERR_SG_winner_WOO_FULL": eq_w,
        "TRF_OUT_winner_WOO_FULL": trf_w,
        "SLSQP_winner_WOO_FULL": slsqp_w,
        "EQERR_correct": eq_c,
        "TRF_correct": trf_c,
        "SLSQP_correct": slsqp_c,
    })

with open(ROOT / "p4b2b" / "p4b2b_384case_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_b2b[0].keys()))
    w.writeheader()
    w.writerows(out_b2b)

print("P4-B2B cases:", len(out_b2b))
print("  EQERR correct (WOO FULL):", eq_corr, "(report a_correct=313)")
print("  TRF correct (WOO FULL):", trf_corr, "(report 328)")
print("  SLSQP correct (WOO FULL):", slsqp_corr, "(report 328)")
