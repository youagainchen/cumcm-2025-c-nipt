# -*- coding: utf-8 -*-
"""
q4_validate.py —— 二号：重复分组验证、阈值、稳健性与错误分析

与 q4_evaluate.py 的关系
------------------------
一号的 `q4_evaluate.py` 已搭出单次分组交叉验证。本模块不重复其模型层
（直接复用 `q4_model` 的 fit_model/predict_proba），只修两处会影响结论
的缺陷，并补齐分工文档中二号的任务 3-6：

  缺陷1 单次划分不稳定。实测仅更换随机种子，"all" 特征集的 OOF PR-AUC 在
        0.416~0.517 之间摆动（极差 0.10），而候选模型之间的差距只有 0.005
        量级，完全淹没在划分噪声里。本模块改为**重复分组交叉验证**
        （默认 5 个种子 x 5 折），所有比较都带跨重复的标准差，差距小于
        噪声时明确判定为"无法区分"。

  缺陷2 规则基线以 0/1 硬预测参与 AUC 计算，与模型的连续概率不可比
        （例如 rule_z18_gt_3 的 TP=0 却报出 ROC-AUC 0.488）。本模块给规则
        赋**连续分数**（各染色体 Z 的最大值 / 最大绝对值），使其与模型在
        同一套 PR-AUC、ROC-AUC 与阈值框架下比较；同时保留题面 Z>3 的硬
        判定作为固定操作点单独汇报。

数据事实提醒（见 q4_signal_audit.py）
------------------------------------
本数据的 AB 列与 Z 值并不对应：T18 阳性 46 条中 z18>3 占 0%，阴性中占
5.9%。因此规则基线性能低是数据事实，不是建模失误；全部可用信号集中在
x_conc 一个变量（事件级/孕妇层面/孕妇内三层证据一致）。

输出
----
outputs/q4_repeated_cv.csv       重复分组 CV 的逐重复与汇总指标
outputs/q4_model_comparison.csv  带噪声判定的模型对比结论
outputs/q4_threshold_policy.csv  各折阈值、策略与验证折表现
outputs/q4_sensitivity.csv       QC/口径/特征集/权重稳健性
outputs/q4_errors.csv            假阴性与假阳性的分层错误分析

运行
----
python src/q4_validate.py                    # 默认 5 重复 x 5 折
python src/q4_validate.py --repeats 10       # 更稳的结论
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from q4_model import (FEATURE_SETS, SUBTYPES, aggregate_events, fit_model,
                      load_rows, predict_proba)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET_SENSITIVITY = 0.90
BASE_SEED = 2026


# ---------------------------------------------------------------- 指标

def confusion(y: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int, int]:
    y, prediction = np.asarray(y, int), np.asarray(prediction, int)
    tp = int(((prediction == 1) & (y == 1)).sum())
    fp = int(((prediction == 1) & (y == 0)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def metrics(y, score, prediction) -> dict:
    """score 必须是连续排序分数（模型概率或规则连续分），否则 AUC 无意义。"""
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    tp, fp, tn, fn = confusion(y, prediction)
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    denominator = (ppv + sensitivity) if np.isfinite(ppv) and np.isfinite(sensitivity) else np.nan
    f1 = (2 * ppv * sensitivity / denominator
          if np.isfinite(denominator) and denominator > 0 else np.nan)
    two_classes = len(np.unique(y)) == 2
    return {"n": len(y), "positive": int(y.sum()), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sensitivity, "specificity": specificity, "ppv": ppv, "npv": npv,
            "f1": f1,
            "pr_auc": float(average_precision_score(y, score)) if two_classes else np.nan,
            "roc_auc": float(roc_auc_score(y, score)) if two_classes else np.nan}


def choose_threshold(y, score, minimum_sensitivity=TARGET_SENSITIVITY):
    """训练折内选阈值：灵敏度达标前提下特异度最高；达不到则回退并标记。"""
    y, score = np.asarray(y, int), np.asarray(score, float)
    candidates = np.unique(score)
    if len(candidates) > 500:
        candidates = np.quantile(score, np.linspace(0, 1, 500))
    best, best_specificity = None, -1.0
    fallback, best_sensitivity = candidates[0], -1.0
    for threshold in candidates:
        tp, fp, tn, fn = confusion(y, score >= threshold)
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if sensitivity >= minimum_sensitivity and specificity > best_specificity:
            best, best_specificity = threshold, specificity
        if sensitivity > best_sensitivity:
            fallback, best_sensitivity = threshold, sensitivity
    if best is not None:
        return float(best), "sensitivity_at_least_90"
    return float(fallback), "maximum_sensitivity_boundary"


# ---------------------------------------------------------------- 候选

def rule_score(events: pd.DataFrame, kind: str) -> np.ndarray:
    """规则基线的连续分数，使其与模型概率在同一框架下可比。"""
    z = events[["z13", "z18", "z21"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    z = np.nan_to_num(z, nan=0.0)
    return np.nanmax(np.abs(z), axis=1) if kind == "abs" else np.nanmax(z, axis=1)


CANDIDATES = (
    [{"name": f"rule_{kind}Z", "kind": "rule", "rule": kind} for kind in ("signed", "abs")]
    + [{"name": f"logit_{name}", "kind": "model", "feature_set": name}
       for name in FEATURE_SETS]
)


def folds_for(events: pd.DataFrame, y: np.ndarray, n_splits: int, seed: int):
    """按孕妇分组的分层折；若某折缺类则降低折数，并返回实际折数。"""
    groups = events.mother_id.astype(str).to_numpy()
    upper = min(n_splits, int(y.sum()), int((1 - y).sum()))
    for count in range(upper, 1, -1):
        splitter = StratifiedGroupKFold(count, shuffle=True, random_state=seed)
        folds = list(splitter.split(events, y, groups))
        if all(np.unique(y[tr]).size == 2 and np.unique(y[va]).size == 2 for tr, va in folds):
            return folds, count
    raise ValueError("无法构造两类齐全的分组折")


def run_once(events: pd.DataFrame, target: str, candidate: dict,
             n_splits: int, seed: int):
    """一次完整的分组 CV，返回 OOF 预测与逐折阈值记录。"""
    y = pd.to_numeric(events[target], errors="coerce").fillna(0).astype(int).to_numpy()
    folds, actual = folds_for(events, y, n_splits, seed)
    oof, thresholds = [], []
    for fold, (train_index, valid_index) in enumerate(folds):
        train, valid = events.iloc[train_index], events.iloc[valid_index]
        if candidate["kind"] == "rule":
            train_score = rule_score(train, candidate["rule"])
            score = rule_score(valid, candidate["rule"])
        else:
            model = fit_model(train, candidate["feature_set"], target)
            train_score = predict_proba(model, train)
            score = predict_proba(model, valid)
        threshold, policy = choose_threshold(y[train_index], train_score)
        thresholds.append({"model": candidate["name"], "target": target, "seed": seed,
                           "fold": fold, "n_folds": actual, "threshold": threshold,
                           "policy": policy})
        for position, index in enumerate(valid_index):
            oof.append({"model": candidate["name"], "target": target, "seed": seed,
                        "fold": fold, "row": int(index),
                        "mother_id": events.mother_id.iloc[index],
                        "label": int(y[index]), "score": float(score[position]),
                        "threshold": threshold,
                        "prediction": int(score[position] >= threshold)})
    return pd.DataFrame(oof), pd.DataFrame(thresholds)


def repeated_cv(events: pd.DataFrame, target: str, repeats: int, n_splits: int):
    """重复分组 CV：多个种子各跑一次完整 CV，保留每次的 OOF 指标。"""
    oof_parts, threshold_parts, records = [], [], []
    for candidate in CANDIDATES:
        for repeat in range(repeats):
            seed = BASE_SEED + repeat
            oof, thresholds = run_once(events, target, candidate, n_splits, seed)
            oof_parts.append(oof)
            threshold_parts.append(thresholds)
            row = metrics(oof.label, oof.score, oof.prediction)
            row.update({"model": candidate["name"], "target": target, "seed": seed,
                        "n_mothers": oof.mother_id.nunique()})
            records.append(row)
    return (pd.DataFrame(records), pd.concat(oof_parts, ignore_index=True),
            pd.concat(threshold_parts, ignore_index=True))


def comparison(per_repeat: pd.DataFrame) -> pd.DataFrame:
    """跨重复汇总，并给出"差距是否超过划分噪声"的判定。"""
    summary = (per_repeat.groupby("model")
               .agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                    roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
                    sensitivity_mean=("sensitivity", "mean"),
                    specificity_mean=("specificity", "mean"),
                    ppv_mean=("ppv", "mean"), f1_mean=("f1", "mean"),
                    repeats=("pr_auc", "size"))
               .reset_index().sort_values("pr_auc_mean", ascending=False))
    best = summary.iloc[0]
    # 判定标准：两模型 PR-AUC 之差是否超过两者标准差的合成
    summary["gap_vs_best"] = best.pr_auc_mean - summary.pr_auc_mean
    noise = np.sqrt(best.pr_auc_sd ** 2 + summary.pr_auc_sd ** 2)
    summary["noise_scale"] = noise
    summary["distinguishable_from_best"] = summary.gap_vs_best > noise
    return summary


# ---------------------------------------------------------------- 稳健性与错误

def sensitivity_analysis(rows: pd.DataFrame, events: pd.DataFrame,
                         repeats: int, n_splits: int) -> pd.DataFrame:
    """口径与设定扰动下，最优模型的结论是否稳定。"""
    scenarios = {"baseline": events}
    if "flag_any" in events.columns:
        clean = events[events.flag_any == 0]
        if clean.label.nunique() == 2 and len(clean) > 50:
            scenarios["exclude_qc_flagged"] = clean
    # 行级口径：把技术重复当独立样本（错误做法），用于量化其影响
    row_level = rows.copy()
    row_level["event_id"] = (row_level.mother_id.astype(str) + "_"
                             + row_level.draw_idx.astype(str))
    scenarios["row_level_wrong_unit"] = row_level
    # 每位孕妇只留首次抽血，彻底消除孕妇内重复
    first = events.sort_values(["mother_id", "draw_idx"]).groupby("mother_id").head(1)
    if first.label.nunique() == 2:
        scenarios["first_draw_only"] = first

    records = []
    for name, frame in scenarios.items():
        for candidate in CANDIDATES:
            values = []
            for repeat in range(max(repeats // 2, 2)):
                try:
                    oof, _ = run_once(frame.reset_index(drop=True), "label", candidate,
                                      n_splits, BASE_SEED + repeat)
                except Exception:
                    continue
                values.append(metrics(oof.label, oof.score, oof.prediction))
            if not values:
                continue
            table = pd.DataFrame(values)
            records.append({"scenario": name, "model": candidate["name"],
                            "n_units": len(frame),
                            "n_mothers": frame.mother_id.nunique(),
                            "pr_auc_mean": table.pr_auc.mean(),
                            "pr_auc_sd": table.pr_auc.std(),
                            "roc_auc_mean": table.roc_auc.mean(),
                            "sensitivity_mean": table.sensitivity.mean(),
                            "specificity_mean": table.specificity.mean()})
    return pd.DataFrame(records)


def error_analysis(events: pd.DataFrame, oof: pd.DataFrame, model: str) -> pd.DataFrame:
    """最优模型的错误案例分层：按亚型、孕周、BMI、QC、技术重复。"""
    block = oof[oof.model == model].copy()
    # 每个事件在多个重复中出现，取多数表决作为其代表预测
    voted = (block.groupby("row")
             .agg(label=("label", "first"), mean_score=("score", "mean"),
                  vote=("prediction", "mean"), mother_id=("mother_id", "first"))
             .reset_index())
    voted["prediction"] = (voted.vote >= 0.5).astype(int)
    voted["error_type"] = np.select(
        [(voted.label == 1) & (voted.prediction == 0),
         (voted.label == 0) & (voted.prediction == 1),
         (voted.label == 1) & (voted.prediction == 1)],
        ["false_negative", "false_positive", "true_positive"], default="true_negative")
    context_columns = [c for c in ["week", "bmi", "age", "x_conc", "z13", "z18", "z21",
                                   "flag_any", "is_tech_repeat", "label_T13",
                                   "label_T18", "label_T21", "draw_idx"]
                       if c in events.columns]
    context = events.reset_index(drop=True)[["mother_id"] + context_columns]
    detail = voted.merge(context, left_on="row", right_index=True,
                         suffixes=("", "_event"))
    return detail.drop(columns=[c for c in ["mother_id_event"] if c in detail.columns])


def main() -> None:
    parser = argparse.ArgumentParser(description="问题四：重复分组验证与稳健性（二号）")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    events = aggregate_events(rows)
    print(f"事件级 {len(events)} 个抽血事件 / {events.mother_id.nunique()} 位孕妇；"
          f"阳性 {int(events.label.sum())}")
    print(f"重复分组交叉验证：{args.repeats} 个种子 x 最多 {args.splits} 折")

    per_repeat, oof, thresholds = repeated_cv(events, "label", args.repeats, args.splits)
    per_repeat.to_csv(args.output_dir / "q4_repeated_cv.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(args.output_dir / "q4_threshold_policy.csv", index=False, encoding="utf-8-sig")

    summary = comparison(per_repeat)
    summary.to_csv(args.output_dir / "q4_model_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n【模型对比】跨重复均值 ± 标准差，按 PR-AUC 排序")
    display = summary[["model", "pr_auc_mean", "pr_auc_sd", "roc_auc_mean",
                       "sensitivity_mean", "specificity_mean",
                       "gap_vs_best", "noise_scale", "distinguishable_from_best"]]
    print(display.round(4).to_string(index=False))
    indistinguishable = summary[~summary.distinguishable_from_best].model.tolist()
    print(f"-> 与最优模型在划分噪声内无法区分的：{indistinguishable}")

    fallback = thresholds[thresholds.policy == "maximum_sensitivity_boundary"]
    print(f"\n【阈值策略】{len(fallback)}/{len(thresholds)} 折无法在训练内达到 "
          f"{TARGET_SENSITIVITY:.0%} 灵敏度，已回退到灵敏度最大处并标记，未事后放宽规则")

    best_model = summary.iloc[0].model
    errors = error_analysis(events, oof, best_model)
    errors.to_csv(args.output_dir / "q4_errors.csv", index=False, encoding="utf-8-sig")
    counts = errors.error_type.value_counts()
    print(f"\n【错误分析】最优模型 {best_model}（多数表决后）")
    print(counts.to_string())
    false_negative = errors[errors.error_type == "false_negative"]
    if len(false_negative):
        for subtype in SUBTYPES:
            column = f"label_{subtype}"
            if column in errors.columns:
                total = int(errors[errors.label == 1][column].sum())
                missed = int(false_negative[column].sum())
                if total:
                    print(f"  {subtype}: 漏诊 {missed}/{total}")

    sensitivity = sensitivity_analysis(rows, events, args.repeats, args.splits)
    sensitivity.to_csv(args.output_dir / "q4_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n【稳健性】各口径下最优模型的 PR-AUC")
    pivot = (sensitivity[sensitivity.model == best_model]
             [["scenario", "n_units", "n_mothers", "pr_auc_mean", "pr_auc_sd"]])
    print(pivot.round(4).to_string(index=False))

    print("\n已输出 q4_repeated_cv / q4_model_comparison / q4_threshold_policy / "
          "q4_errors / q4_sensitivity")


if __name__ == "__main__":
    main()
