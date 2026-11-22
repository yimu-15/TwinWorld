"""Extract P4-B3C (#110) 384-case records for BOTH Cap 100 and Cap 200.

Regenerates p4b3c_384case_results.csv with per-case winner/correct/calls at both
hard budgets. Verifies reconstructed correct counts and call sums against
p4b3c_fresh_report.json (primary/secondary + compute_cost actual_204vector_calls).
"""
import csv, json
from pathlib import Path

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0\experiments\p2d1\results\p4b3c")
REPORT = json.loads((ROOT / "p4b3c_fresh_report.json").read_text(encoding="utf-8"))

rows = []
for i in range(24):
    d = json.loads((ROOT / "checkpoints" / f"scene_{i:02d}.json").read_text(encoding="utf-8"))
    rows.extend(d["rows"])

def sum_calls(fl, budget, solver):
    s = 0
    for dom, laws in fl.get(budget, {}).get(solver, {}).items():
        for law, rec in laws.items():
            s += rec.get("calls", 0)
    return s

out = []
agg = {"b100": {"TRF": 0, "SLSQP": 0}, "b200": {"TRF": 0, "SLSQP": 0}}
calls = {"b100": {"TRF": 0, "SLSQP": 0}, "b200": {"TRF": 0, "SLSQP": 0}}
for r in rows:
    dec = r.get("decisions_PRIVATE", {})
    true = r["true_law_PRIVATE"]
    rec = {
        "scene_id": r["scene_id"],
        "band": r.get("band_PRIVATE"),
        "mode": r.get("mode_PRIVATE"),
        "condition": r.get("condition_PRIVATE"),
        "true_law": true,
    }
    for b in ("100", "200"):
        fl = r.get("fit_logs_PRIVATE", {})
        tw = dec.get(b, {}).get("TRF_CAP", {}).get("MID_PLUS_FULL", {}).get("winner")
        sw = dec.get(b, {}).get("SLSQP_CAP", {}).get("MID_PLUS_FULL", {}).get("winner")
        tc = 1 if tw == true else 0
        sc = 1 if sw == true else 0
        tcall = sum_calls(fl, b, "TRF_CAP")
        scall = sum_calls(fl, b, "SLSQP_CAP")
        agg[f"b{b}"]["TRF"] += tc; agg[f"b{b}"]["SLSQP"] += sc
        calls[f"b{b}"]["TRF"] += tcall; calls[f"b{b}"]["SLSQP"] += scall
        rec.update({
            f"TRF_winner_b{b}": tw, f"SLSQP_winner_b{b}": sw,
            f"TRF_correct_b{b}": tc, f"SLSQP_correct_b{b}": sc,
            f"TRF_calls_b{b}": tcall, f"SLSQP_calls_b{b}": scall,
        })
    out.append(rec)

fields = list(out[0].keys())
with open(ROOT / "p4b3c_384case_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader(); w.writerows(out)

print("P4-B3C cases:", len(out))
print("Cap100  TRF correct:", agg["b100"]["TRF"], "(report 320) | SLSQP:", agg["b100"]["SLSQP"], "(report 338)")
print("Cap200  TRF correct:", agg["b200"]["TRF"], "(report 321) | SLSQP:", agg["b200"]["SLSQP"], "(report 324)")
print("Cap100  TRF calls sum:", calls["b100"]["TRF"], "(report 114350) | SLSQP:", calls["b100"]["SLSQP"], "(report 111199)")
print("Cap200  TRF calls sum:", calls["b200"]["TRF"], "(report 189743) | SLSQP:", calls["b200"]["SLSQP"], "(report 193082)")
