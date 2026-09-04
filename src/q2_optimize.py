# -*- coding: utf-8 -*-
"""问题二：以期望风险最小为目标的连续 BMI 动态分组。

不是对 BMI 做无监督聚类，而是在排序后的孕妇上求全局最优连续分割：

    min  sum_g min_t sum_{i in g} E[R_i(t)]

约束每组至少 ``min_group_size`` 人，且切点只能放在不同 BMI 取值之间。
分别支持问题一瞬时达标概率与区间删失生存模型两种概率口径。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from q2_risk import ExpectedRisk, T_MAX, T_MIN


ROOT = Path(__file__).resolve().parent.parent
EVENT = ROOT / "data" / "processed" / "male_clean_event.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_baseline_bmi() -> pd.DataFrame:
    d = pd.read_csv(EVENT, encoding="utf-8-sig").sort_values(["mother_id", "week_mean"])
    m = d.groupby("mother_id", as_index=False).agg(bmi=("bmi", "first"))
    return m.dropna().sort_values(["bmi", "mother_id"]).reset_index(drop=True)


def risk_matrix(er: ExpectedRisk, bmi: np.ndarray, grid_step: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(T_MIN, T_MAX + 1e-9, grid_step)
    values = np.empty((len(bmi), len(times)))
    for i, b in enumerate(bmi):
        values[i] = [er.expected_risk(t, b) for t in times]
    if not np.all(np.isfinite(values)):
        raise RuntimeError("期望风险矩阵含非有限值，拒绝继续分组。")
    return times, values


def admissible_positions(bmi: np.ndarray) -> np.ndarray:
    """返回允许成为组边界的排序位置；相同 BMI 不能拆到两个组。"""
    change = np.flatnonzero(np.diff(bmi) > 1e-10) + 1
    return np.r_[0, change, len(bmi)].astype(int)


def optimize_all_k(
    bmi: np.ndarray,
    times: np.ndarray,
    risks: np.ndarray,
    max_k: int = 6,
    min_group_size: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(bmi)
    positions = admissible_positions(bmi)
    pos_set = set(positions.tolist())
    prefix = np.vstack([np.zeros(len(times)), np.cumsum(risks, axis=0)])

    # 区间代价使用全体样本风险总和；最后除以 n 得总体平均风险。
    cost: dict[tuple[int, int], tuple[float, int]] = {}
    for i in positions[:-1]:
        for j in positions[positions > i]:
            if j - i < min_group_size:
                continue
            totals = prefix[j] - prefix[i]
            ti = int(np.argmin(totals))
            cost[(int(i), int(j))] = (float(totals[ti]), ti)

    inf = float("inf")
    dp = np.full((max_k + 1, n + 1), inf)
    prev = np.full((max_k + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, max_k + 1):
        for j in positions:
            if j < k * min_group_size:
                continue
            for i in positions[positions < j]:
                if i not in pos_set or i < (k - 1) * min_group_size:
                    continue
                if (i, j) not in cost or not np.isfinite(dp[k - 1, i]):
                    continue
                candidate = dp[k - 1, i] + cost[(i, j)][0]
                if candidate < dp[k, j]:
                    dp[k, j] = candidate
                    prev[k, j] = i

    summaries, groups = [], []
    risk_one = dp[1, n] / n
    previous = None
    for k in range(1, max_k + 1):
        if not np.isfinite(dp[k, n]):
            continue
        bounds = [n]
        j = n
        for kk in range(k, 0, -1):
            j = int(prev[kk, j])
            if j < 0:
                raise RuntimeError(f"无法回溯 k={k} 的最优分组。")
            bounds.append(j)
        bounds = bounds[::-1]
        avg = dp[k, n] / n
        gain = np.nan if previous is None else previous - avg
        summaries.append(
            dict(k=k, avg_risk=avg, reduction_vs_k1=risk_one - avg,
                 relative_reduction=(risk_one - avg) / risk_one,
                 marginal_gain=gain)
        )
        previous = avg
        for g, (i, j) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
            _, ti = cost[(i, j)]
            lower = float(bmi[i]) if g == 1 else float((bmi[i - 1] + bmi[i]) / 2)
            upper = float(bmi[j - 1]) if g == k else float((bmi[j - 1] + bmi[j]) / 2)
            groups.append(
                dict(k=k, group=g, lower=lower, upper=upper, left_closed=True,
                     right_closed=(g == k), n=j - i, bmi_min=float(bmi[i]),
                     bmi_max=float(bmi[j - 1]), best_week=float(times[ti]),
                     avg_risk=float(cost[(i, j)][0] / (j - i)))
            )
    return pd.DataFrame(summaries), pd.DataFrame(groups)


def select_elbow(summary: pd.DataFrame) -> int:
    """用风险-k曲线到首尾弦线的最大垂距给出可复现的拐点建议。"""
    if len(summary) <= 2:
        return int(summary.k.iloc[-1])
    x = (summary.k - summary.k.min()) / (summary.k.max() - summary.k.min())
    y = (summary.avg_risk - summary.avg_risk.min()) / (
        summary.avg_risk.max() - summary.avg_risk.min() + 1e-15
    )
    distance = (1 - x) - y
    return int(summary.loc[distance.idxmax(), "k"])


def run_path(name: str, er: ExpectedRisk, max_k: int, min_group_size: int,
             grid_step: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    mothers = load_baseline_bmi()
    bmi = mothers.bmi.to_numpy(float)
    times, risks = risk_matrix(er, bmi, grid_step)
    summary, groups = optimize_all_k(bmi, times, risks, max_k, min_group_size)
    summary.insert(0, "path", name)
    groups.insert(0, "path", name)
    elbow = select_elbow(summary)
    summary["elbow_recommended"] = summary.k.eq(elbow)
    return summary, groups, elbow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=["q1", "survival", "both"], default="both")
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--min-group-size", type=int, default=15)
    parser.add_argument("--grid-step", type=float, default=0.05)
    args = parser.parse_args()

    paths: list[tuple[str, ExpectedRisk]] = []
    if args.path in ("q1", "both"):
        paths.append(("q1_instant", ExpectedRisk()))
    if args.path in ("survival", "both"):
        from q2_survival import load_prob_fn
        paths.append(("survival_aft", ExpectedRisk(prob_fn=load_prob_fn())))

    all_s, all_g, elbows = [], [], {}
    for name, er in paths:
        print(f"正在计算 {name} 风险矩阵和动态规划……")
        summary, groups, elbow = run_path(
            name, er, args.max_k, args.min_group_size, args.grid_step
        )
        all_s.append(summary)
        all_g.append(groups)
        elbows[name] = elbow
        print("\n", name, "风险-组数：")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"拐点建议 k={elbow}，对应分组：")
        print(groups[groups.k == elbow].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary = pd.concat(all_s, ignore_index=True)
    groups = pd.concat(all_g, ignore_index=True)
    summary.to_csv(OUT / "q2_k_curve.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(OUT / "q2_groups_all_k.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "q2_optimize_meta.json", "w", encoding="utf-8") as f:
        json.dump(dict(elbows=elbows, max_k=args.max_k,
                       min_group_size=args.min_group_size, grid_step=args.grid_step),
                  f, ensure_ascii=False, indent=2)
    print("\n已输出 outputs/q2_k_curve.csv、q2_groups_all_k.csv、q2_optimize_meta.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"问题二分组优化失败：{exc}", file=sys.stderr)
        raise
