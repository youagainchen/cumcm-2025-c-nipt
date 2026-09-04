# -*- coding: utf-8 -*-
"""问题二最终方案：等权集成风险下，用显式判据选定组数并给出可执行分组。

组数 k 不再硬编码，而是由 select_k() 依次施加三条可复现的准则：

  C1 收益捕获率：分组带来的风险下降有天花板（本数据全部可达收益仅占
     基线风险的 1.39%）。要求该 k 已捕获总可达收益的 CAPTURE_FRAC 以上，
     即"再多分组也榨不出多少了"。注意不能写成"边际收益足够小"——那样
     选出的是 k>=4，而它们恰恰倒在 C2 上，逻辑自相矛盾。
  C2 组间差异可辨：相邻组最优时点间隔必须 >= MIN_WEEK_GAP 周。若两组给
     出的建议时点只差几天，分组在临床上不产生实际区别，属于无效切分。
  C3 组样本量下限：每组 >= min_group_size（沿用 q2_optimize 的约束）。

三条都满足的最大 k 里取最小者，得到"组数不宜太多也不宜太少"的解。
本数据下判据给出 k=3；作为对照，纯肘部法给 k=2，而 k>=4 会在 C2 上失败
（第1、2组时点仅差 0.75 周 ≈ 5 天）。

可执行边界在数学最优割点附近取整，代价记录在 regret_vs_exact 列。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from q2_optimize import load_baseline_bmi, optimize_all_k, risk_matrix
from q2_risk import ExpectedRisk
from q2_survival import load_prob_fn


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# 判据参数
CAPTURE_FRAC = 0.90    # C1：至少捕获总可达收益的 90%
MIN_WEEK_GAP = 1.0     # C2：相邻组最优时点至少相差 1 周，否则分组无操作意义
MIN_GROUP_SIZE = 15    # C3

# 可执行边界：在 k 选定后于数学最优割点附近取整
OPERATIONAL_CUTS_BY_K = {
    2: (31.5,),
    3: (31.5, 37.0),
    4: (30.0, 32.0, 37.0),
}


def select_k(summary: pd.DataFrame, groups: pd.DataFrame,
             capture_frac: float = CAPTURE_FRAC,
             min_week_gap: float = MIN_WEEK_GAP) -> tuple[int, pd.DataFrame]:
    """按 C1/C2/C3 选组数，返回 (k, 判据明细表)。"""
    base = float(summary.avg_risk.iloc[0])
    total_gain = base - float(summary.avg_risk.min())
    rows = []
    for k in summary.k:
        g = groups[groups.k == k].sort_values("group")
        gain = base - float(summary.loc[summary.k == k, "avg_risk"].iloc[0])
        captured = gain / total_gain if total_gain > 0 else np.nan
        gaps = np.diff(g.best_week.to_numpy(float)) if len(g) > 1 else np.array([np.inf])
        rows.append({
            "k": int(k),
            "avg_risk": float(summary.loc[summary.k == k, "avg_risk"].iloc[0]),
            "captured_frac": captured,
            "marginal_gain": float(summary.loc[summary.k == k, "marginal_gain"].iloc[0]),
            "C1_收益捕获": bool(k > 1 and captured >= capture_frac),
            "min_week_gap": float(gaps.min()),
            "C2_组间可辨": bool(gaps.min() >= min_week_gap),
            "min_group_n": int(g.n.min()),
            "C3_样本量": bool(g.n.min() >= MIN_GROUP_SIZE),
        })
    table = pd.DataFrame(rows)
    table["全部满足"] = table.C1_收益捕获 & table.C2_组间可辨 & table.C3_样本量
    ok = table[table["全部满足"]]
    if ok.empty:
        raise RuntimeError("没有任何 k 同时满足三条判据，需重新审视判据参数")
    k_star = int(ok.k.iloc[0])          # 满足全部判据的最小 k
    table["selected"] = table.k.eq(k_star)
    return k_star, table


def build_operational_groups(bmi: np.ndarray, times: np.ndarray,
                             combined_risk: np.ndarray, exact_avg_risk: float,
                             k: int) -> pd.DataFrame:
    cuts = OPERATIONAL_CUTS_BY_K[k]
    edges = (float(bmi.min()), *cuts, float(bmi.max()))
    rows: list[dict] = []
    total_cost = 0.0
    for group, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), 1):
        if group < k:
            mask = (bmi >= lower) & (bmi < upper)
            interval = f"[{lower:.2f}, {upper:g})"
        else:
            mask = (bmi >= lower) & (bmi <= upper)
            interval = f"[{lower:g}, {upper:.2f}]"
        cost = combined_risk[mask].sum(axis=0)
        best_idx = int(np.argmin(cost))
        total_cost += float(cost[best_idx])
        rows.append({
            "group": group,
            "lower": lower,
            "upper": upper,
            "interval": interval,
            "n": int(mask.sum()),
            "bmi_min": float(bmi[mask].min()),
            "bmi_max": float(bmi[mask].max()),
            "best_week": float(times[best_idx]),
            "avg_risk": float(cost[best_idx] / mask.sum()),
        })
    final = pd.DataFrame(rows)
    total_avg_risk = total_cost / len(bmi)
    final["total_avg_risk"] = total_avg_risk
    final["regret_vs_exact"] = total_avg_risk - exact_avg_risk
    return final


def main() -> None:
    mothers = load_baseline_bmi()
    bmi = mothers.bmi.to_numpy(float)

    print("计算最终模型的两个风险输入组件……")
    times, risk_instant = risk_matrix(ExpectedRisk(), bmi, 0.05)
    times_censored, risk_censored = risk_matrix(
        ExpectedRisk(prob_fn=load_prob_fn()), bmi, 0.05
    )
    if not np.allclose(times, times_censored):
        raise RuntimeError("两个风险输入组件的候选检测时点网格不一致。")

    combined = 0.5 * (risk_instant + risk_censored)
    summary, groups = optimize_all_k(
        bmi, times, combined, max_k=6, min_group_size=MIN_GROUP_SIZE
    )

    k_star, criteria = select_k(summary, groups)
    summary["recommended"] = summary.k.eq(k_star)

    exact = groups[groups.k == k_star].copy()
    exact_avg_risk = float(summary.loc[summary.k == k_star, "avg_risk"].iloc[0])
    final_groups = build_operational_groups(
        bmi, times, combined, exact_avg_risk, k_star
    )

    summary.to_csv(OUT / "q2_final_k_curve.csv", index=False, encoding="utf-8-sig")
    criteria.to_csv(OUT / "q2_k_criteria.csv", index=False, encoding="utf-8-sig")
    exact.to_csv(OUT / "q2_exact_groups.csv", index=False, encoding="utf-8-sig")
    final_groups.to_csv(OUT / "q2_final_groups.csv", index=False, encoding="utf-8-sig")

    print("\n最终模型的风险-组数曲线：")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n组数判据明细（C1 收益递减 / C2 组间可辨 / C3 样本量）：")
    print(criteria.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n-> 判据选定 k = {k_star}")
    print(f"\n精确{k_star}组数学最优解（用于审计）：")
    print(exact.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n采用整数边界{OPERATIONAL_CUTS_BY_K[k_star]}的可执行{k_star}组方案：")
    print(final_groups.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        "\n已输出 q2_final_k_curve.csv、q2_k_criteria.csv、"
        "q2_exact_groups.csv、q2_final_groups.csv"
    )


if __name__ == "__main__":
    main()
