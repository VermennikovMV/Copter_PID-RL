"""Recompute development metrics from saved traces; preserve original tables."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with (args.directory / "results.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    changed_rows = 0
    for row in rows:
        previous_penalty = float(row["engineering_failure_penalized_rmse_m"])
        previous_rmse = float(row["engineering_rmse_m"]) if row["engineering_rmse_m"] else None
        trace = args.directory / f"{row['controller']}_{row['scenario']}_{row['mission']}_{row['seed']}.npz"
        with np.load(trace) as data:
            errors = data["error"][data["time"] > 2.0 + 1e-9]
        horizon = 30.0 if row["mission"] == "hover" else 40.0
        sample_count = round((horizon - 2.0) / 0.02)
        missing_samples = max(0, sample_count - len(errors))
        row["engineering_rmse_m"] = float(np.sqrt(np.mean(errors**2))) if len(errors) else None
        row["engineering_failure_penalized_rmse_m"] = float(np.sqrt((np.sum(errors**2) + 16 * missing_samples) / sample_count))
        if ((previous_rmse is None) != (row["engineering_rmse_m"] is None)
                or abs(previous_penalty - row["engineering_failure_penalized_rmse_m"]) > 1e-10):
            changed_rows += 1
    with (args.directory / "results_recomputed.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    keys = sorted({(r["controller"], r["scenario"], r["mission"]) for r in rows})
    for controller, scenario, mission in keys:
        selected = [r for r in rows if (r["controller"], r["scenario"], r["mission"]) == (controller, scenario, mission)]
        completed = [float(r["route_completion_s"]) for r in selected if r["route_completion_s"]]
        summary.append({"controller": controller, "scenario": scenario, "mission": mission,
                        "n": len(selected), "survived": sum(r["survived"] == "True" for r in selected),
                        "penalized_rmse_m": float(np.mean([r["engineering_failure_penalized_rmse_m"] for r in selected])),
                        "completed": len(completed), "completion_s": float(np.mean(completed)) if completed else None})
    output = {"summary": summary,
              "metric_recomputation_note": "No observed RMSE exists after warmup for a failure before 2s: null, never zero. Failure padding covers only evaluation samples after warmup. Original results.csv and traces preserved.",
              "rows_corrected": changed_rows,
              "all_flights": len(rows), "survived": sum(r["survived"] == "True" for r in rows)}
    (args.directory / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
