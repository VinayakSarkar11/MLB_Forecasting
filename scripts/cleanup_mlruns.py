"""Prune MLflow experiment history.

Strategy:
  - Active experiments (model2_midab_cascade, model1_plate_discipline):
      keep the 3 most recent runs, delete the rest.
  - Superseded experiments (earlier model2/model1 iterations):
      keep 1 run for reference, delete the rest.
  - Abandoned experiments (test/failed experiments):
      delete entirely.

Run:
    python scripts/cleanup_mlruns.py          # dry run — shows what would be deleted
    python scripts/cleanup_mlruns.py --apply  # actually delete
    python scripts/cleanup_mlruns.py --apply --gc  # delete + free disk space
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient

KEEP_ACTIVE      = 3   # runs to keep for live experiments
KEEP_SUPERSEDED  = 1   # runs to keep for old-but-reference experiments

# Experiments we still care about — keep KEEP_ACTIVE most recent runs
ACTIVE = {
    "model2_midab_cascade",
    "model1_plate_discipline",
}

# Superseded by the current approach — keep 1 run for audit trail
SUPERSEDED = {
    "model2_pa_cascade",
    "model2_pa_outcomes",
    "cascade_model1",
}

# Nothing worth keeping — delete all runs
ABANDONED = {
    "pitch_type_location_test",
    "swing_prediction_test",
    "model1_plate_discipline_3class_test",
}


def runs_to_delete(client: MlflowClient, exp_id: str, keep: int) -> list:
    runs = client.search_runs(exp_id, order_by=["start_time DESC"])
    return [r.info.run_id for r in runs[keep:]]


def main(apply: bool, gc: bool) -> None:
    client = MlflowClient()
    total_marked = 0

    for exp in client.search_experiments():
        name = exp.name
        exp_id = exp.experiment_id

        if name in ACTIVE:
            to_delete = runs_to_delete(client, exp_id, keep=KEEP_ACTIVE)
            keep_n = KEEP_ACTIVE
        elif name in SUPERSEDED:
            to_delete = runs_to_delete(client, exp_id, keep=KEEP_SUPERSEDED)
            keep_n = KEEP_SUPERSEDED
        elif name in ABANDONED:
            to_delete = runs_to_delete(client, exp_id, keep=0)
            keep_n = 0
        else:
            continue  # Default or unknown — leave alone

        all_runs = client.search_runs(exp_id)
        print(f"\n{name}: {len(all_runs)} runs → keep {keep_n}, delete {len(to_delete)}")

        for run_id in to_delete:
            total_marked += 1
            if apply:
                client.delete_run(run_id)
                print(f"  deleted {run_id[:8]}")
            else:
                print(f"  [dry run] would delete {run_id[:8]}")

    print(f"\n{'Deleted' if apply else 'Would delete'} {total_marked} runs.")

    if apply and gc:
        print("\nRunning mlflow gc to free disk space...")
        import subprocess
        result = subprocess.run(
            ["mlflow", "gc", "--backend-store-uri", "mlruns"],
            capture_output=True, text=True,
        )
        print(result.stdout or result.stderr or "done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete runs (default: dry run)")
    parser.add_argument("--gc",    action="store_true",
                        help="Run mlflow gc after deletion to free disk space")
    args = parser.parse_args()

    if not args.apply:
        print("DRY RUN — pass --apply to actually delete\n")

    main(apply=args.apply, gc=args.gc)
