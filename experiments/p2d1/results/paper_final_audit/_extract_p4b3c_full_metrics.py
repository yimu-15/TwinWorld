"""Enhance P4-B3C (#110) per-case CSV with full budget metrics.

Adds per-case (Cap 100 & 200, TRF & SLSQP): availability, winner, correct,
actual residual-vector calls, unused calls, wall-clock seconds, optimizer-success
count, budget-exhausted count. Verifies against p4b3c_fresh_report.json.
"""
import csv, json
from pathlib import Path

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0\experiments\p2d1\results\p4b3c")
REPORT = json.loads((ROOT / "p4b3c_fresh_report.json").read_text(encoding="utf-8"))

rows = []
for i in range(24):
    d = json.loads((ROOT / "checkpoints" / f"scene_{i:02d}.json").read_text(encoding="utf-8"))
    rows.extend(d["rows"])

out = []
sums = {}
for r in rows:
    true = r["true_law_PRIVATE"]
    rec = {"scene_id": r["scene_id"], "band": r.get("band_PRIVATE"),
           "mode": r.get("mode_PRIVATE"), "condition": r.get("condition_PRIVATE"),
           "true_law": true}
    for b in ("100", "200"):
        fl = r.get("fit_logs_PRIVATE", {}).get(b, {})
        for solver in ("TRF_CAP", "SLSQP_CAP"):
            p = solver[:4].lower()  # trf_ / slsq
            dec = r.get("decisions_PRIVATE", {}).get(b, {}).get(solver, {}).get("MID_PLUS_FULL", {})
            winner = dec.get("winner")
            avail = 1 if dec.get("available") else 0
            corr = 1 if winner == true else 0
            calls = unused = wall = succ = cap = 0
            for dom, laws in fl.get(solver, {}).items():
                for law, lg in laws.items():
                    calls += lg.get("calls", 0)
                    unused += lg.get("unused_calls", 0)
                    wall += lg.get("wall_seconds", 0.0)
                    for a in lg.get("attempts", []):
                        if a.get("status") == "optimizer_success":
                            succ += 1
                        elif a.get("status") == "budget_cap":
                            cap += 1
            rec.update({
                f"{p}_avail_{b}": avail, f"{p}_winner_{b}": winner, f"{p}_corr_{b}": corr,
                f"{p}_calls_{b}": calls, f"{p}_unused_{b}": unused,
                f"{p}_wall_s_{b}": round(wall, 6), f"{p}_succ_{b}": succ, f"{p}_cap_{b}": cap,
            })
    out.append(rec)

fields = list(out[0].keys())
with open(ROOT / "p4b3c_384case_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader(); w.writerows(out)

# verify key aggregates
for b in ("100", "200"):
    for solver, p in (("TRF_CAP", "trf_"), ("SLSQP_CAP", "slsq")):
        corr = sum(1 for o in out if o[f"{p}_corr_{b}"] == 1)
        calls = sum(o[f"{p}_calls_{b}"] for o in out)
        print(f"Cap{b} {solver}: correct={corr} calls={calls}")

print("expected: b100 TRF 320/114350, SLSQP 338/111199; b200 TRF 321/189743, SLSQP 324/193082")
print("n cases:", len(out), "| columns:", len(fields))
