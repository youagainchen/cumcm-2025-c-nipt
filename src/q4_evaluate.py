# -*- coding: utf-8 -*-
"""Grouped out-of-subject evaluation for Question 4 (auxiliary).

Status: AUXILIARY. The official paper-facing evaluation is q4_validate.py,
which uses repeated grouped CV (5 seeds x 5 folds) instead of a single split.
A single split is not usable for ranking here: swapping the seed alone moves
the "all" feature set's OOF PR-AUC over 0.416-0.517, while the candidates
differ by ~0.005. What this script still adds is per-subtype (T13/T18/T21)
models, which q4_validate.py does not fit separately.

All outputs therefore carry the q4_singlesplit_ prefix so they cannot
overwrite the official tables (both scripts previously wrote
q4_bootstrap_ci.csv with different column schemas).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from q4_model import FEATURE_SETS, SUBTYPES, aggregate_events, fit_model, load_rows, predict_proba


def choose_threshold(y, probability, minimum_sensitivity=0.90):
    y, probability = np.asarray(y, int), np.asarray(probability, float)
    choices = []
    for threshold in np.unique(np.r_[0.0, probability, 1.0]):
        prediction = probability >= threshold
        tp = ((prediction == 1) & (y == 1)).sum()
        fp = ((prediction == 1) & (y == 0)).sum()
        tn = ((prediction == 0) & (y == 0)).sum()
        fn = ((prediction == 0) & (y == 1)).sum()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        choices.append((threshold, sensitivity, specificity))
    eligible = [x for x in choices if x[1] >= minimum_sensitivity]
    if eligible:
        return max(eligible, key=lambda x: (x[2], x[0]))[0], "sensitivity_at_least_90"
    return max(choices, key=lambda x: (x[1], x[2], x[0]))[0], "maximum_sensitivity_boundary"


def metrics(y, probability, prediction):
    y, probability, prediction = np.asarray(y, int), np.asarray(probability, float), np.asarray(prediction, int)
    tp = int(((prediction == 1) & (y == 1)).sum()); fp = int(((prediction == 1) & (y == 0)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum()); fn = int(((prediction == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else np.nan; specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan; npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if ppv + sensitivity else np.nan
    try: pr_auc = float(average_precision_score(y, probability))
    except ValueError: pr_auc = np.nan
    try: roc_auc = float(roc_auc_score(y, probability))
    except ValueError: roc_auc = np.nan
    return {"n": len(y), "positive": int(y.sum()), "predicted_positive": int(prediction.sum()), "tp": tp,
            "fp": fp, "tn": tn, "fn": fn, "sensitivity": sensitivity, "specificity": specificity,
            "ppv": ppv, "npv": npv, "f1": f1, "pr_auc": pr_auc, "roc_auc": roc_auc,
            "brier": float(brier_score_loss(y, probability))}


def grouped_folds(events, y, requested, seed):
    groups = events.mother_id.astype(str).to_numpy()
    maximum = min(requested, int(y.sum()), int((1 - y).sum()))
    for count in range(maximum, 1, -1):
        splitter = StratifiedGroupKFold(count, shuffle=True, random_state=seed)
        folds = list(splitter.split(events, y, groups))
        if all(np.unique(y[train]).size == 2 and np.unique(y[valid]).size == 2 for train, valid in folds):
            return folds
    raise ValueError(f"cannot construct grouped folds with both classes for {requested} splits")


def evaluate_candidate(events, feature_set, target, model_name, folds):
    y = pd.to_numeric(events[target], errors="coerce").fillna(0).astype(int).to_numpy()
    predictions, thresholds = [], []
    for fold, (train, valid) in enumerate(folds):
        model = fit_model(events.iloc[train], feature_set, target)
        threshold, policy = choose_threshold(y[train], predict_proba(model, events.iloc[train]))
        probabilities = predict_proba(model, events.iloc[valid])
        thresholds.append({"model": model_name, "target": target, "feature_set": feature_set, "fold": fold,
                           "threshold": threshold, "policy": policy, "train_n": len(train),
                           "train_positive": int(y[train].sum())})
        for index, probability in zip(valid, probabilities):
            predictions.append({"model": model_name, "target": target, "feature_set": feature_set, "fold": fold,
                                "event_id": events.iloc[index].event_id, "mother_id": events.iloc[index].mother_id,
                                "label": int(y[index]), "probability": float(probability), "threshold": float(threshold),
                                "prediction": int(probability >= threshold)})
    return pd.DataFrame(predictions), pd.DataFrame(thresholds)


def metric_table(oof):
    records = []
    keys = ["model", "target", "feature_set"]
    for key, part in oof.groupby(keys + ["fold"]):
        records.append(dict(zip(keys + ["scope", "fold"], list(key[:3]) + ["fold", key[3]])) | metrics(part.label, part.probability, part.prediction))
    for key, part in oof.groupby(keys):
        records.append(dict(zip(keys + ["scope", "fold"], list(key) + ["oof_fold_threshold", "all"])) | metrics(part.label, part.probability, part.prediction))
        threshold = float(part.threshold.median())
        records.append(dict(zip(keys + ["scope", "fold", "threshold"], list(key) + ["oof_median_threshold", "all", threshold])) |
                        metrics(part.label, part.probability, part.probability.to_numpy() >= threshold))
    return pd.DataFrame(records)


def bootstrap_ci(oof, repetitions, seed):
    rng = np.random.default_rng(seed); records = []
    for key, part in oof.groupby(["model", "target", "feature_set"]):
        mothers = part.mother_id.unique(); by_mother = {m: g for m, g in part.groupby("mother_id")}
        for repetition in range(repetitions):
            sample = pd.concat([by_mother[m] for m in rng.choice(mothers, len(mothers), replace=True)], ignore_index=True)
            values = metrics(sample.label, sample.probability, sample.prediction)
            for metric in ("sensitivity", "specificity", "ppv", "npv", "f1", "pr_auc", "roc_auc", "brier"):
                records.append(dict(zip(["model", "target", "feature_set", "repetition", "metric"], list(key) + [repetition, metric])) | {"value": values[metric]})
    draws = pd.DataFrame(records)
    return draws.groupby(["model", "target", "feature_set", "metric"], as_index=False).value.agg(
        mean="mean", lower=lambda x: x.quantile(0.025), upper=lambda x: x.quantile(0.975))


def run(data_path=None, output_dir=None, splits=5, bootstrap=500, seed=2026):
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "outputs"; output_dir.mkdir(exist_ok=True, parents=True)
    events = aggregate_events(load_rows(Path(data_path) if data_path else None))
    candidates = [(name, name, "label") for name in FEATURE_SETS] + [(f"subtype_{s}", "z_quality", f"label_{s}") for s in SUBTYPES]
    parts, threshold_parts = zip(*(evaluate_candidate(
        events, candidate[1], candidate[2], candidate[0],
        grouped_folds(events, pd.to_numeric(events[candidate[2]], errors="coerce").fillna(0).astype(int).to_numpy(), splits, seed)
    ) for candidate in candidates))
    oof, thresholds = pd.concat(parts, ignore_index=True), pd.concat(threshold_parts, ignore_index=True)
    summary = thresholds.groupby(["model", "target", "feature_set"], as_index=False).agg(threshold=("threshold", "median"), train_n=("train_n", "sum"), train_positive=("train_positive", "sum"))
    summary["fold"], summary["policy"] = "median", "median_of_training_fold_thresholds"; thresholds = pd.concat([thresholds, summary[thresholds.columns]], ignore_index=True)
    z = events[["z13", "z18", "z21", "zx"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    rules = {"z_gt_3_any": z.gt(3).any(axis=1), "abs_z_gt_3_any": z.abs().gt(3).any(axis=1)}
    for column in z:
        rules[f"{column}_gt_3"] = z[column].gt(3)
        rules[f"abs_{column}_gt_3"] = z[column].abs().gt(3)
    rule_parts = []
    for name, prediction in rules.items():
        part = events[["event_id", "mother_id", "label"]].copy()
        part[["model", "target", "feature_set", "fold"]] = [f"rule_{name}", "label", "rule", "all"]
        part["probability"], part["threshold"], part["prediction"] = prediction.astype(float), 0.5, prediction.astype(int)
        rule_parts.append(part[["model", "target", "feature_set", "fold", "event_id", "mother_id", "label", "probability", "threshold", "prediction"]])
    all_oof = pd.concat([oof, *rule_parts], ignore_index=True)
    all_oof.to_csv(output_dir / "q4_singlesplit_oof_predictions.csv", index=False, encoding="utf-8-sig"); thresholds.to_csv(output_dir / "q4_singlesplit_thresholds.csv", index=False, encoding="utf-8-sig")
    metric_table(all_oof).to_csv(output_dir / "q4_singlesplit_cv_metrics.csv", index=False, encoding="utf-8-sig"); bootstrap_ci(oof, bootstrap, seed).to_csv(output_dir / "q4_singlesplit_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    return {"event_count": len(events), "mother_count": events.mother_id.nunique(), "model_count": len(candidates), "splits": splits}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path); parser.add_argument("--output-dir", type=Path); parser.add_argument("--splits", type=int, default=5); parser.add_argument("--bootstrap", type=int, default=500); parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); print(run(args.data, args.output_dir, args.splits, args.bootstrap, args.seed))
