"""Champion-challenger model evaluation.

Compares two metrics JSON files (champion vs challenger) and decides
whether to promote the challenger based on a configurable MAPE threshold.

This logic mirrors what the Airflow DAG will run after every training job:
  1. Load champion metrics (current production model)
  2. Load challenger metrics (newly trained model)
  3. Promote challenger if it beats champion by >= MIN_IMPROVEMENT_PCT
  4. Write a promotion decision JSON that downstream steps can branch on

Run locally:
    python src/models/evaluate.py \
        --champion models/lgbm_metrics.json \
        --challenger models/prophet_metrics.json \
        --output models/promotion_decision.json

Design notes
------------
* We use MAPE as the primary metric because it is scale-invariant and
  directly interpretable by non-technical finance stakeholders.
* MIN_IMPROVEMENT_PCT = 2.0 means the challenger must be at least 2
  percentage-points better in MAPE, not just noise-level better.
  This guards against promoting models whose improvement would vanish
  in the next week's retrain.
* The function returns a dict so it can be called from the Airflow
  PythonOperator without subprocess overhead.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Challenger must beat champion by at least this many MAPE percentage points
MIN_IMPROVEMENT_PCT = 2.0


def compute_mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, ignoring zero actuals."""
    import numpy as np
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def evaluate_champion_challenger(
    champion_metrics_path: str,
    challenger_metrics_path: str,
    output_path: str,
    min_improvement_pct: float = MIN_IMPROVEMENT_PCT,
) -> dict:
    """Compare champion and challenger; write and return promotion decision.

    Parameters
    ----------
    champion_metrics_path : Path to champion metrics JSON
    challenger_metrics_path : Path to challenger metrics JSON
    output_path : Where to write the promotion decision JSON
    min_improvement_pct : Minimum MAPE improvement required to promote

    Returns
    -------
    dict with keys: promote (bool), champion_mape, challenger_mape,
                    delta_mape, reason
    """
    champion = load_metrics(champion_metrics_path)
    challenger = load_metrics(challenger_metrics_path)

    champion_mape = champion["mape"]
    challenger_mape = challenger["mape"]
    delta_mape = champion_mape - challenger_mape  # positive = challenger is better

    champion_model = champion.get("model", "unknown")
    challenger_model = challenger.get("model", "unknown")

    log.info(f"Champion  ({champion_model}):  MAPE={champion_mape:.2f}%")
    log.info(f"Challenger ({challenger_model}): MAPE={challenger_mape:.2f}%")
    log.info(f"Delta MAPE: {delta_mape:+.2f}pp  (threshold: {min_improvement_pct}pp)")

    if delta_mape >= min_improvement_pct:
        promote = True
        reason = (
            f"Challenger {challenger_model} beats champion {champion_model} "
            f"by {delta_mape:.2f}pp MAPE (threshold {min_improvement_pct}pp). PROMOTE."
        )
    elif delta_mape > 0:
        promote = False
        reason = (
            f"Challenger {challenger_model} is marginally better by {delta_mape:.2f}pp "
            f"MAPE but does not meet the {min_improvement_pct}pp threshold. RETAIN CHAMPION."
        )
    else:
        promote = False
        reason = (
            f"Challenger {challenger_model} is worse than champion {champion_model} "
            f"by {abs(delta_mape):.2f}pp MAPE. RETAIN CHAMPION."
        )

    log.info(reason)

    # Secondary metric check: challenger must not regress badly on RMSE
    # even if MAPE appears better (handles edge cases with near-zero actuals)
    secondary_warning = None
    if promote:
        champion_rmse = champion.get("rmse", 0)
        challenger_rmse = challenger.get("rmse", 0)
        if challenger_rmse > champion_rmse * 1.10:
            secondary_warning = (
                f"WARNING: Challenger RMSE ({challenger_rmse:.1f}) is >10% worse "
                f"than champion ({champion_rmse:.1f}) despite better MAPE. "
                "Review before deploying."
            )
            log.warning(secondary_warning)

    decision = {
        "promote": promote,
        "champion_model": champion_model,
        "challenger_model": challenger_model,
        "champion_mape": round(champion_mape, 4),
        "challenger_mape": round(challenger_mape, 4),
        "delta_mape_pp": round(delta_mape, 4),
        "min_improvement_threshold_pp": min_improvement_pct,
        "champion_mae": champion.get("mae"),
        "challenger_mae": challenger.get("mae"),
        "champion_rmse": champion.get("rmse"),
        "challenger_rmse": challenger.get("rmse"),
        "secondary_warning": secondary_warning,
        "reason": reason,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(decision, f, indent=2)
    log.info(f"Promotion decision saved to {output_path}")

    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Champion-challenger model evaluation")
    parser.add_argument("--champion", required=True, help="Path to champion metrics JSON")
    parser.add_argument("--challenger", required=True, help="Path to challenger metrics JSON")
    parser.add_argument(
        "--output",
        default="models/promotion_decision.json",
        help="Path to write promotion decision JSON",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=MIN_IMPROVEMENT_PCT,
        help=f"Minimum MAPE improvement in pp to promote (default {MIN_IMPROVEMENT_PCT})",
    )
    args = parser.parse_args()

    decision = evaluate_champion_challenger(
        champion_metrics_path=args.champion,
        challenger_metrics_path=args.challenger,
        output_path=args.output,
        min_improvement_pct=args.min_improvement,
    )

    print("\n=== Champion-Challenger Evaluation ===")
    for k, v in decision.items():
        print(f"  {k}: {v}")
