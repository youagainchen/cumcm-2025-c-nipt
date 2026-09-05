# -*- coding: utf-8 -*-
"""
q4_signal_audit.py —— 二号：AB 判定的信号来源审计

为什么需要这一步
----------------
一号的 `q4_model.group_difference()` 已给出事件级的组间差异与 FDR 校正。
但事件级差异不足以判断信号是否真实：同一位孕妇贡献多个事件，若某些孕妇
恰好既标阳、其测序批次又系统性偏移，事件级检验会把"孕妇/批次差异"误读成
"AB 相关信号"。二号的职责是回答"报出来的性能有多可信"，因此必须补两层：

    事件级      554 个抽血事件直接比较（与一号口径一致，用于对账）
    孕妇层面    同一孕妇聚合成一条再比较，消除"事件多的孕妇权重大"的影响
    孕妇内      只用标签在不同次检测间发生变化的孕妇做组内对照 —— 同一个人
                自己跟自己比，孕妇层面的一切固定混杂（体质、批次、送检机构）
                都被差分掉，是判断"事件级信号是否真实"的最强证据

结论摘要（当前数据）
--------------------
1. Z 值与 AB 标签在**原始数据中就不对应**：T18 阳性的 46 条记录里 z18>3 占
   0%，而阴性记录中占 5.9%；T13 阳性的 z13 中位数反而低于阴性。已回到
   data/raw/附件.xlsx 逐行核对，确认非清洗错误。因此"Z>3 判阳"这一临床
   规则在本数据上近乎随机，论文中不能假设 Z 值是 AB 的直接依据。
2. 全部可用信号集中在 `x_conc` 一个变量，且三层证据一致（含孕妇内对照），
   说明它是真实的事件级信号，而非个体或批次混杂。

输出
----
outputs/q4_signal_audit.csv     三层 AUC 与 p 值
outputs/q4_zscore_check.csv     各染色体 Z 值与对应亚型标签的一致性核查

运行
----
python src/q4_signal_audit.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from q4_model import FEATURE_SETS, SUBTYPES, load_events, load_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# NIPT 常用的判阳参考线；此处只用于核查"标签是否与 Z 值一致"，不作为模型阈值
Z_ALERT = 3.0


def _auc(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U 换算的 AUC 与双侧 p；样本过少时返回 NaN。"""
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if len(positive) < 3 or len(negative) < 3:
        return np.nan, np.nan
    u, p = mannwhitneyu(positive, negative, alternative="two-sided")
    return float(u / (len(positive) * len(negative))), float(p)


def signal_audit(events: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """三层信号审计。events 为事件级，rows 为行级（用于孕妇内对照）。"""
    changing = rows.groupby("mother_id").label.nunique()
    within = rows[rows.mother_id.isin(changing[changing > 1].index)]
    by_mother = events.groupby("mother_id").agg(
        label=("label", "max"),
        **{feature: (feature, "mean") for feature in FEATURE_SETS["all"]})

    records = []
    for feature in FEATURE_SETS["all"]:
        event_auc, event_p = _auc(
            pd.to_numeric(events.loc[events.label == 1, feature], errors="coerce").to_numpy(float),
            pd.to_numeric(events.loc[events.label == 0, feature], errors="coerce").to_numpy(float))
        mother_auc, mother_p = _auc(
            by_mother.loc[by_mother.label == 1, feature].to_numpy(float),
            by_mother.loc[by_mother.label == 0, feature].to_numpy(float))
        within_auc, within_p = _auc(
            pd.to_numeric(within.loc[within.label == 1, feature], errors="coerce").to_numpy(float),
            pd.to_numeric(within.loc[within.label == 0, feature], errors="coerce").to_numpy(float))
        records.append({
            "feature": feature,
            "event_auc": event_auc, "event_p": event_p,
            "mother_auc": mother_auc, "mother_p": mother_p,
            "within_mother_auc": within_auc, "within_mother_p": within_p,
            "consistent_direction": bool(
                np.isfinite(event_auc) and np.isfinite(within_auc)
                and (event_auc - 0.5) * (within_auc - 0.5) > 0),
        })
    table = pd.DataFrame(records)
    table["event_signal"] = (table.event_auc - 0.5).abs()
    table = table.sort_values("event_signal", ascending=False).drop(columns="event_signal")
    table.insert(0, "n_mothers_with_changing_label",
                 int((changing > 1).sum()))
    return table


def zscore_check(rows: pd.DataFrame) -> pd.DataFrame:
    """核查各染色体 Z 值与对应亚型标签是否一致。

    真实 NIPT 中 T18 判阳的直接依据是 z18 超过判定线。若阳性组的超线比例
    不高于阴性组，说明本数据的 AB 列并非由这些 Z 值按常规规则生成，必须在
    论文中说明，否则"用 Z 值复现 AB"的建模前提就是错的。
    """
    records = []
    for subtype in SUBTYPES:
        column = f"z{subtype[1:]}"
        label = f"label_{subtype}"
        if column not in rows.columns or label not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        positive = values[rows[label] == 1].dropna()
        negative = values[rows[label] == 0].dropna()
        records.append({
            "subtype": subtype, "z_column": column,
            "n_positive": len(positive), "n_negative": len(negative),
            "positive_median": float(positive.median()),
            "negative_median": float(negative.median()),
            "positive_share_over_alert": float((positive > Z_ALERT).mean()),
            "negative_share_over_alert": float((negative > Z_ALERT).mean()),
            "positive_max": float(positive.max()), "negative_max": float(negative.max()),
            "label_matches_zscore_rule": bool(
                (positive > Z_ALERT).mean() > (negative > Z_ALERT).mean()),
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="问题四信号来源审计（二号）")
    parser.add_argument("--data", type=Path, default=None,
                        help="official event-level female_clean_event.csv")
    parser.add_argument("--row-data", type=Path, default=None,
                        help="row-level female_clean.csv, used only for row-level audits")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(args.data)
    rows = load_rows(args.row_data)
    print(f"事件级 {len(events)} 个抽血事件 / {events.mother_id.nunique()} 位孕妇；"
          f"阳性 {int(events.label.sum())}")

    check = zscore_check(rows)
    check.to_csv(args.output_dir / "q4_zscore_check.csv", index=False, encoding="utf-8-sig")
    print("\n【Z 值与亚型标签一致性核查】")
    print(check[["subtype", "n_positive", "positive_median", "negative_median",
                 "positive_share_over_alert", "negative_share_over_alert",
                 "label_matches_zscore_rule"]].round(4).to_string(index=False))
    broken = check[~check.label_matches_zscore_rule]
    if len(broken):
        print(f"-> {broken.subtype.tolist()} 的阳性组超线比例不高于阴性组，"
              f"AB 列并非按 Z>{Z_ALERT:g} 的常规规则生成；"
              f"论文中不得假设 Z 值是 AB 的直接依据。")

    audit = signal_audit(events, rows)
    audit.to_csv(args.output_dir / "q4_signal_audit.csv", index=False, encoding="utf-8-sig")
    print("\n【三层信号审计】前 6 个特征（孕妇内一列为最强证据）")
    print(audit.drop(columns="n_mothers_with_changing_label")
          .head(6).round(4).to_string(index=False))

    strong = audit[(audit.event_p < 0.01) & ((audit.event_auc - 0.5).abs() > 0.10)]
    print(f"\n事件级达到 p<0.01 且 |AUC-0.5|>0.10 的特征：{strong.feature.tolist()}")
    robust = strong[(strong.within_mother_p < 0.05) & strong.consistent_direction]
    print(f"其中在孕妇内对照下仍显著且方向一致的：{robust.feature.tolist()}")
    print("-> 只有后者可被认定为真实的事件级信号；其余特征的组间差异不能排除"
          "孕妇层面或批次混杂。")
    print(f"\n已输出 {args.output_dir / 'q4_signal_audit.csv'} 与 "
          f"{args.output_dir / 'q4_zscore_check.csv'}")


if __name__ == "__main__":
    main()
