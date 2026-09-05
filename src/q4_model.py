# -*- coding: utf-8 -*-
"""Question 4: female-fetus abnormality classification.

The unit of analysis is one blood-draw event, loaded from the event table
produced by ``clean.py``. The row-level table is used only for audits and the
explicit wrong-unit sensitivity analysis. AB is the only source of the event
label. This module deliberately returns probabilities;
sample-out-of-subject threshold selection belongs to the evaluation layer.

Evaluation layer: q4_validate.py is the single official pipeline (repeated
grouped CV over 5 seeds x 5 folds, continuous rule scores, noise-aware model
comparison, thresholds, sensitivity, error strata, per-subtype models and
cluster bootstrap). The former q4_evaluate.py has been merged into it.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ROW_DATA_CANDIDATES = (
    ROOT / "data" / "processed" / "female_clean.csv",
    ROOT / "data" / "female_clean.csv",
)
EVENT_DATA_CANDIDATES = (
    ROOT / "data" / "processed" / "female_clean_event.csv",
    ROOT / "data" / "female_clean_event.csv",
)
OUT = ROOT / "outputs"

Z_FEATURES = ["z13", "z18", "z21", "zx"]
QUALITY_FEATURES = [
    "x_conc", "gc", "gc13", "gc18", "gc21", "reads_raw", "uniq_reads",
    "map_ratio", "dup_ratio", "filt_ratio",
]
PERSON_FEATURES = ["bmi", "week", "age"]
FEATURE_SETS = {
    "z": Z_FEATURES,
    "z_quality": Z_FEATURES + QUALITY_FEATURES,
    "all": Z_FEATURES + QUALITY_FEATURES + PERSON_FEATURES,
}
SUBTYPES = ("T13", "T18", "T21")
LABELS = ("label", "label_T13", "label_T18", "label_T21")


def _resolve_data_path(path: Path | None, candidates: tuple[Path, ...], name: str) -> Path:
    if path is not None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"{name} not found: {source}")
        return source
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found under data/ or data/processed/; run python src/clean.py first")


def resolve_row_data_path(path: Path | None = None) -> Path:
    return _resolve_data_path(path, ROW_DATA_CANDIDATES, "female_clean.csv")


def resolve_event_data_path(path: Path | None = None) -> Path:
    return _resolve_data_path(path, EVENT_DATA_CANDIDATES, "female_clean_event.csv")


# Backward-compatible name for callers that explicitly need the row table.
resolve_data_path = resolve_row_data_path


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_rows(path: Path | None = None) -> pd.DataFrame:
    source = resolve_row_data_path(path)
    frame = pd.read_csv(source, encoding="utf-8-sig")
    required = {"mother_id", "draw_idx", "sample_id", *LABELS, *Z_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"female data is missing columns: {missing}")
    numeric_columns = set(FEATURE_SETS["all"]) | set(LABELS) | {
        "draw_idx", "flag_gc", "flag_map_ratio", "flag_dup_ratio", "flag_filt_ratio", "flag_any"
    }
    frame = _numeric(frame, numeric_columns)
    frame["mother_id"] = frame["mother_id"].astype(str)
    frame["draw_idx"] = frame["draw_idx"].fillna(frame["sample_id"])
    return frame


def load_events(path: Path | None = None) -> pd.DataFrame:
    """Load and normalize the official event-level table from ``clean.py``."""
    source = resolve_event_data_path(path)
    frame = pd.read_csv(source, encoding="utf-8-sig")
    if "week" not in frame.columns and "week_mean" in frame.columns:
        frame["week"] = frame["week_mean"]
    if "replicate_count" not in frame.columns and "n_reps" in frame.columns:
        frame["replicate_count"] = frame["n_reps"]
    if "is_tech_repeat" not in frame.columns and "replicate_count" in frame.columns:
        frame["is_tech_repeat"] = (
            pd.to_numeric(frame["replicate_count"], errors="coerce").fillna(1) > 1
        ).astype(int)
    required = {"event_id", "mother_id", "draw_idx", *LABELS, *Z_FEATURES,
                "week", "replicate_count", "is_tech_repeat"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"female event data is missing columns: {missing}")
    numeric_columns = set(FEATURE_SETS["all"]) | set(LABELS) | {
        "draw_idx", "replicate_count", "n_reps", "is_tech_repeat",
        "flag_gc", "flag_map_ratio", "flag_dup_ratio", "flag_filt_ratio", "flag_any",
    }
    frame = _numeric(frame, numeric_columns)
    frame["mother_id"] = frame["mother_id"].astype(str)
    if frame["event_id"].duplicated().any() or frame[["mother_id", "draw_idx"]].duplicated().any():
        raise ValueError("female event data must contain one row per (mother_id, draw_idx)")
    return frame.sort_values(["mother_id", "week", "draw_idx"], kind="stable").reset_index(drop=True)


def _audit_row(category: str, metric: str, value, detail: str = "") -> dict:
    return {"category": category, "metric": metric, "value": value, "detail": detail}


def audit_data(rows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = [
        _audit_row("size", "row_count", len(rows)), _audit_row("size", "event_count", len(events)),
        _audit_row("size", "mother_count", events["mother_id"].nunique()),
        _audit_row("size", "technical_repeat_rows", int((rows.groupby(["mother_id", "draw_idx"]).size() - 1).clip(lower=0).sum())),
        _audit_row("size", "technical_repeat_events", int((events["replicate_count"] > 1).sum())),
    ]
    for column in LABELS:
        records.append(_audit_row("label", f"{column}_positive_events", int(events[column].sum())))
        disagreement = rows.groupby(["mother_id", "draw_idx"])[column].agg(["min", "max"])
        records.append(_audit_row("label", f"{column}_replicate_disagreement_events",
                                  int((disagreement["min"] != disagreement["max"]).sum())))
    for column in FEATURE_SETS["all"]:
        if column in rows:
            missing = int(rows[column].isna().sum())
            records.append(_audit_row("missing", column, missing, f"row_rate={missing / len(rows):.6f}"))
    for column in ["flag_gc", "flag_map_ratio", "flag_dup_ratio", "flag_filt_ratio", "flag_any"]:
        if column in events:
            records.append(_audit_row("qc", f"{column}_events", int(events[column].sum())))
    numeric = events[FEATURE_SETS["all"]].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().sort_values(ascending=False)
    if not pairs.empty:
        records.append(_audit_row("correlation", "largest_absolute_pairwise_correlation",
                                  float(pairs.iloc[0]), f"pair={pairs.index[0][0]}::{pairs.index[0][1]}"))
    return pd.DataFrame(records, columns=["category", "metric", "value", "detail"])


def _rule_metrics(name: str, prediction: pd.Series, truth: pd.Series) -> dict:
    prediction = prediction.astype(int).to_numpy()
    truth = truth.astype(int).to_numpy()
    tp = int(((prediction == 1) & (truth == 1)).sum())
    fp = int(((prediction == 1) & (truth == 0)).sum())
    tn = int(((prediction == 0) & (truth == 0)).sum())
    fn = int(((prediction == 0) & (truth == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if ppv + sensitivity else np.nan
    try:
        pr_auc = average_precision_score(truth, prediction)
    except ValueError:
        pr_auc = np.nan
    try:
        roc_auc = roc_auc_score(truth, prediction)
    except ValueError:
        roc_auc = np.nan
    return {"rule": name, "n": len(truth), "positive_truth": int(truth.sum()),
            "predicted_positive": int(prediction.sum()), "tp": tp, "fp": fp,
            "tn": tn, "fn": fn, "sensitivity": sensitivity, "specificity": specificity,
            "ppv": ppv, "npv": npv, "f1": f1, "pr_auc": pr_auc,
            "roc_auc": roc_auc, "brier": brier_score_loss(truth, prediction)}


def rule_baseline(events: pd.DataFrame) -> pd.DataFrame:
    z = events[Z_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    rules = {"z_gt_3_any": z.gt(3).any(axis=1), "abs_z_gt_3_any": z.abs().gt(3).any(axis=1)}
    for column in Z_FEATURES:
        rules[f"{column}_gt_3"] = z[column].gt(3)
        rules[f"abs_{column}_gt_3"] = z[column].abs().gt(3)
    return pd.DataFrame([_rule_metrics(name, prediction, events["label"])
                         for name, prediction in rules.items()])


def group_difference(events: pd.DataFrame) -> pd.DataFrame:
    """Compare candidate indicators between AB-normal and AB-abnormal events."""
    records = []
    for feature in FEATURE_SETS["all"]:
        values = pd.to_numeric(events[feature], errors="coerce")
        normal = values[events["label"] == 0].dropna().to_numpy(float)
        abnormal = values[events["label"] == 1].dropna().to_numpy(float)
        if len(normal) < 2 or len(abnormal) < 2:
            continue
        pooled_scale = np.sqrt(((len(normal) - 1) * np.var(normal, ddof=1)
                                + (len(abnormal) - 1) * np.var(abnormal, ddof=1))
                               / (len(normal) + len(abnormal) - 2))
        records.append({"feature": feature, "normal_n": len(normal), "abnormal_n": len(abnormal),
                        "normal_mean": float(np.mean(normal)), "abnormal_mean": float(np.mean(abnormal)),
                        "normal_median": float(np.median(normal)), "abnormal_median": float(np.median(abnormal)),
                        "mean_difference_abnormal_minus_normal": float(np.mean(abnormal) - np.mean(normal)),
                        "standardized_mean_difference": float((np.mean(abnormal) - np.mean(normal)) / pooled_scale)
                        if pooled_scale > 0 else np.nan,
                        "welch_t_p": float(ttest_ind(abnormal, normal, equal_var=False).pvalue),
                        "mann_whitney_p": float(mannwhitneyu(abnormal, normal, alternative="two-sided").pvalue),
                        "missing_n": int(values.isna().sum())})
    result = pd.DataFrame(records)
    for p_column, q_column in [("welch_t_p", "welch_t_q"), ("mann_whitney_p", "mann_whitney_q")]:
        order = np.argsort(result[p_column].to_numpy(float))
        q_values = np.empty(len(result), dtype=float)
        ranked = result[p_column].to_numpy(float)[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
        q_values[order] = adjusted
        result[q_column] = q_values
    return result


@dataclass
class FittedQ4Model:
    feature_set: str
    features: list[str]
    target: str
    pipeline: Pipeline
    C: float
    class_weight: str | None


def _check_features(events: pd.DataFrame, features: list[str]) -> None:
    missing = sorted(set(features) - set(events.columns))
    if missing:
        raise ValueError(f"event data is missing model features: {missing}")


def fit_model(events: pd.DataFrame, feature_set: str = "z", target: str = "label",
              C: float = 1.0, class_weight: str | None = None) -> FittedQ4Model:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set {feature_set!r}; choose {sorted(FEATURE_SETS)}")
    if target not in events.columns:
        raise ValueError(f"target column not found: {target}")
    features = list(FEATURE_SETS[feature_set])
    _check_features(events, features)
    x = events[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(events[target], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        raise ValueError(f"target {target} has fewer than two classes")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        # 不写 penalty="l2"：它本就是默认值，而 sklearn 1.8 起该参数已弃用
        # （1.10 移除），显式传入只会刷 FutureWarning，行为没有任何差别。
        ("classifier", LogisticRegression(C=C, class_weight=class_weight,
                                            solver="lbfgs", max_iter=5000, random_state=2026)),
    ])
    pipeline.fit(x, y)
    return FittedQ4Model(feature_set, features, target, pipeline, C, class_weight)


def predict_proba(model: FittedQ4Model, events: pd.DataFrame) -> np.ndarray:
    _check_features(events, model.features)
    x = events[model.features].apply(pd.to_numeric, errors="coerce")
    return model.pipeline.predict_proba(x)[:, 1]


def fit_subtype_models(events: pd.DataFrame, feature_set: str = "z_quality",
                       C: float = 1.0) -> dict[str, FittedQ4Model]:
    return {subtype: fit_model(events, feature_set=feature_set, target=f"label_{subtype}", C=C)
            for subtype in SUBTYPES}


def predict_subtypes(models: dict[str, FittedQ4Model], events: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"event_id": events["event_id"].to_numpy()}
                          if "event_id" in events else {"row_index": events.index})
    for subtype in SUBTYPES:
        if subtype not in models:
            raise KeyError(f"missing subtype model: {subtype}")
        result[f"p_{subtype}"] = predict_proba(models[subtype], events)
    return result


def coefficients(model: FittedQ4Model) -> pd.DataFrame:
    classifier = model.pipeline.named_steps["classifier"]
    imputer = model.pipeline.named_steps["imputer"]
    scaler = model.pipeline.named_steps["scaler"]
    rows = [{"model": model.feature_set, "target": model.target, "term": "Intercept",
             "coefficient_standardized": float(classifier.intercept_[0]),
             "odds_ratio": float(np.exp(np.clip(classifier.intercept_[0], -700, 700))),
             "imputer_median": np.nan, "scaler_mean": np.nan, "scaler_scale": np.nan}]
    for index, feature in enumerate(model.features):
        coefficient = float(classifier.coef_[0, index])
        rows.append({"model": model.feature_set, "target": model.target, "term": feature,
                     "coefficient_standardized": coefficient,
                     "odds_ratio": float(np.exp(np.clip(coefficient, -700, 700))),
                     "imputer_median": float(imputer.statistics_[index]),
                     "scaler_mean": float(scaler.mean_[index]),
                     "scaler_scale": float(scaler.scale_[index])})
    return pd.DataFrame(rows)


def model_specs(models: dict[str, FittedQ4Model], events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, model in models.items():
        rows.append({
            "model": model_name,
            "target": model.target,
            "features": "+".join(model.features),
            "feature_count": len(model.features),
            "C": model.C,
            "class_weight": model.class_weight or "none",
            "event_count": len(events),
            "positive_count": int(pd.to_numeric(events[model.target], errors="coerce").fillna(0).sum()),
        })
    return pd.DataFrame(rows)


def run(data_path: Path | None = None, output_dir: Path | None = None,
        row_data_path: Path | None = None) -> dict:
    output_dir = Path(output_dir) if output_dir is not None else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(data_path)
    rows = load_rows(row_data_path)
    event_columns = ["event_id", "mother_id", "draw_idx", "sample_id", "week", "age", "height", "weight", "bmi",
                     *FEATURE_SETS["z_quality"], "n_reps", "replicate_count", "is_tech_repeat", *LABELS,
                     "flag_gc", "flag_map_ratio", "flag_dup_ratio", "flag_filt_ratio", "flag_any"]
    event_columns += [column for column in ("aneuploidy_raw", "week_raw", "draw_date") if column in events]
    event_columns = list(dict.fromkeys(column for column in event_columns if column in events))
    # 注意：不要写 data/processed/female_clean_event.csv —— 那是 clean.py(丙) 的产物。
    # 本脚本此前会用不同的列集合覆盖它，谁后跑谁生效，属于静默的数据源冲突。
    # clean.py v3.2 起事件级文件已自带全部测量列，本表仅作对账副本另存到 outputs/。
    events[event_columns].to_csv(output_dir / "q4_event_table.csv", index=False, encoding="utf-8-sig")
    audit_data(rows, events).to_csv(output_dir / "q4_data_audit.csv", index=False, encoding="utf-8-sig")
    group_difference(events).to_csv(output_dir / "q4_group_difference.csv", index=False, encoding="utf-8-sig")
    rule_baseline(events).to_csv(output_dir / "q4_rule_baseline.csv", index=False, encoding="utf-8-sig")
    models = {name: fit_model(events, feature_set=name) for name in FEATURE_SETS}
    subtype_models = fit_subtype_models(events)
    predictions = events[["event_id", "mother_id", "draw_idx", "label", *LABELS[1:]]].copy()
    for name, model in models.items():
        predictions[f"p_{name}"] = predict_proba(model, events)
    predictions = predictions.merge(predict_subtypes(subtype_models, events), on="event_id", validate="one_to_one")
    predictions.to_csv(output_dir / "q4_event_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat([coefficients(model) for model in [*models.values(), *subtype_models.values()]], ignore_index=True).to_csv(
        output_dir / "q4_coefficients.csv", index=False, encoding="utf-8-sig")
    model_specs({**models, **{f"subtype_{key}": value for key, value in subtype_models.items()}}, events).to_csv(
        output_dir / "q4_model_candidates.csv", index=False, encoding="utf-8-sig"
    )
    source_path = resolve_event_data_path(data_path)
    try:
        source_display = source_path.relative_to(ROOT).as_posix()
    except ValueError:
        source_display = str(source_path)
    metadata = {"source": source_display, "row_count": int(len(rows)),
                "event_count": int(len(events)), "mother_count": int(events["mother_id"].nunique()),
                "feature_sets": FEATURE_SETS, "subtype_targets": list(SUBTYPES),
                "label_definition": "AB-derived event label supplied by female_clean_event.csv",
                "threshold_policy": "probability only; threshold is selected out of subject in q4_validate.py",
                "default_preview_model": "z_quality"}
    (output_dir / "q4_model_meta.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    with open(output_dir / "q4_model.pkl", "wb") as handle:
        pickle.dump({"models": models, "subtype_models": subtype_models, "metadata": metadata}, handle)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Question 4 female abnormality models")
    parser.add_argument("--data", type=Path, default=None,
                        help="official event-level female_clean_event.csv")
    parser.add_argument("--row-data", type=Path, default=None,
                        help="row-level female_clean.csv, used only for audits")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.data, args.output_dir, args.row_data), ensure_ascii=False, indent=2))
