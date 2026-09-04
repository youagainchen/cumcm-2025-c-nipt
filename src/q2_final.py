# -*- coding: utf-8 -*-
"""问题二最终方案：等权集成风险下固定采用4个连续BMI组。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from q2_optimize import load_baseline_bmi, optimize_all_k, risk_matrix
from q2_risk import ExpectedRisk
from q2_survival import load_prob_fn


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FINAL_K = 4
OPERATIONAL_CUTS = (30.0, 32.0, 37.0)


def build_operational_groups(
    bmi: np.ndarray,
    times: np.ndarray,
    combined_risk: np.ndarray,
    exact_avg_risk: float,
) -> pd.DataFrame:
    edges = (float(bmi.min()), *OPERATIONAL_CUTS, float(bmi.max()))
    rows: list[dict] = []
    total_cost = 0.0
    for group, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), 1):
        if group < FINAL_K:
            mask = (bmi >= lower) & (bmi < upper)
            interval = f"[{lower:.2f}, {upper:.0f})"
        else:
            mask = (bmi >= lower) & (bmi <= upper)
            interval = f"[{lower:.0f}, {upper:.2f}]"
        cost = combined_risk[mask].sum(axis=0)
        best_idx = int(np.argmin(cost))
        total_cost += float(cost[best_idx])
        rows.append(
            {
                "group": group,
                "lower": lower,
                "upper": upper,
                "interval": interval,
                "n": int(mask.sum()),
                "bmi_min": float(bmi[mask].min()),
                "bmi_max": float(bmi[mask].max()),
                "best_week": float(times[best_idx]),
                "avg_risk": float(cost[best_idx] / mask.sum()),
            }
        )
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
        bmi, times, combined, max_k=6, min_group_size=15
    )
    summary["recommended"] = summary.k.eq(FINAL_K)

    exact = groups[groups.k == FINAL_K].copy()
    exact_avg_risk = float(summary.loc[summary.k == FINAL_K, "avg_risk"].iloc[0])
    final_groups = build_operational_groups(
        bmi, times, combined, exact_avg_risk
    )

    summary.to_csv(OUT / "q2_final_k_curve.csv", index=False, encoding="utf-8-sig")
    exact.to_csv(OUT / "q2_exact_4groups.csv", index=False, encoding="utf-8-sig")
    final_groups.to_csv(OUT / "q2_final_groups.csv", index=False, encoding="utf-8-sig")

    print("\n最终模型的风险-组数曲线：")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n精确4组数学最优解（用于审计）：")
    print(exact.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n采用整数边界30、32、37的可执行4组方案：")
    print(final_groups.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        "\n已输出 q2_final_k_curve.csv、q2_exact_4groups.csv、"
        "q2_final_groups.csv"
    )


if __name__ == "__main__":
    main()
