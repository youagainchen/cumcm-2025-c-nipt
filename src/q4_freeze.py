# -*- coding: utf-8 -*-
"""Freeze the final Question 4 model after grouped validation."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from q4_model import coefficients, fit_model, fit_subtype_models, load_events, predict_proba, predict_subtypes


FINAL_FEATURE_SET = "z_quality"
FINAL_MODEL_NAME = "logit_z_quality"
FINAL_TARGET = "label"


def validation_threshold(output_dir: Path) -> float:
    path = output_dir / "q4_threshold_policy.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"validation threshold table not found: {path}; run python src/q4_validate.py first"
        )
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = {"model", "threshold"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"invalid validation threshold table {path}: missing columns {missing}")
    values = pd.to_numeric(
        table.loc[table["model"] == FINAL_MODEL_NAME, "threshold"], errors="coerce"
    ).dropna()
    values = values[np.isfinite(values) & values.between(0.0, 1.0)]
    if not len(values):
        raise ValueError(f"no valid threshold for {FINAL_MODEL_NAME!r} in {path}")
    return float(values.median())


def run(data_path=None, output_dir=None):
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = validation_threshold(output_dir)
    events = load_events(Path(data_path) if data_path else None)
    final_model = fit_model(events, feature_set=FINAL_FEATURE_SET, target=FINAL_TARGET)
    subtype_models = fit_subtype_models(events, feature_set=FINAL_FEATURE_SET)
    probability = predict_proba(final_model, events)
    predictions = events[["event_id", "mother_id", "draw_idx", "label", "label_T13", "label_T18", "label_T21"]].copy()
    predictions["p_abnormal"] = probability
    predictions["final_threshold"] = threshold
    predictions["final_prediction"] = (probability >= threshold).astype(int)
    predictions = predictions.merge(predict_subtypes(subtype_models, events), on="event_id", validate="one_to_one")
    predictions.to_csv(output_dir / "q4_final_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat([coefficients(final_model), *[coefficients(model) for model in subtype_models.values()]], ignore_index=True).to_csv(
        output_dir / "q4_final_coefficients.csv", index=False, encoding="utf-8-sig")
    validation_summary = {}
    comparison_path = output_dir / "q4_model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path, encoding="utf-8-sig")
        row = comparison.loc[comparison["model"] == FINAL_MODEL_NAME]
        if len(row):
            validation_summary = {key: value for key, value in row.iloc[0].to_dict().items()
                                  if not (isinstance(value, float) and not np.isfinite(value))}
    metadata = {
        "model_name": FINAL_MODEL_NAME,
        "feature_set": FINAL_FEATURE_SET,
        "target": FINAL_TARGET,
        "threshold": threshold,
        "threshold_policy": "median of repeated grouped-CV training-fold thresholds; sensitivity priority 90%",
        "data_source": "data/processed/female_clean_event.csv" if data_path is None else str(data_path),
        "label_definition": "AB non-empty = abnormal in the official blood-draw event table",
        "event_count": int(len(events)),
        "mother_count": int(events["mother_id"].nunique()),
        "validation_summary": validation_summary,
    }
    (output_dir / "q4_final_model_meta.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2, default=float), encoding="utf-8")
    with open(output_dir / "q4_final_model.pkl", "wb") as handle:
        pickle.dump({"model": final_model, "subtype_models": subtype_models, "metadata": metadata}, handle)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze the final Question 4 model")
    parser.add_argument("--data", type=Path, default=None,
                        help="official event-level female_clean_event.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.data, args.output_dir)
