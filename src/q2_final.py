# -*- coding: utf-8 -*-
"""问题二汇总：两条概率路径的等权集成与模型权重敏感性。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from q2_optimize import load_baseline_bmi, optimize_all_k, risk_matrix, select_elbow
from q2_risk import ExpectedRisk
from q2_survival import load_prob_fn


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def k2_profile(bmi: np.ndarray, times: np.ndarray, risks: np.ndarray,
               min_group_size: int = 15) -> pd.DataFrame:
    prefix = np.vstack([np.zeros(len(times)), np.cumsum(risks, axis=0)])
    rows = []
    for cut in np.flatnonzero(np.diff(bmi) > 1e-10) + 1:
        if cut < min_group_size or len(bmi) - cut < min_group_size:
            continue
        left = prefix[cut]
        right = prefix[-1] - prefix[cut]
        il, ir = int(np.argmin(left)), int(np.argmin(right))
        rows.append(dict(
            cut=(bmi[cut - 1] + bmi[cut]) / 2, n_left=cut, n_right=len(bmi) - cut,
            week_left=times[il], week_right=times[ir], avg_risk=(left[il] + right[ir]) / len(bmi),
        ))
    ans = pd.DataFrame(rows)
    ans["regret"] = ans.avg_risk - ans.avg_risk.min()
    ans["within_0p1pct"] = ans.avg_risk <= ans.avg_risk.min() * 1.001
    return ans


def main() -> None:
    mothers = load_baseline_bmi()
    bmi = mothers.bmi.to_numpy(float)
    print("计算问题一瞬时概率路径……")
    times, risk_q1 = risk_matrix(ExpectedRisk(), bmi, 0.05)
    print("计算区间删失AFT路径……")
    times_s, risk_surv = risk_matrix(ExpectedRisk(prob_fn=load_prob_fn()), bmi, 0.05)
    if not np.allclose(times, times_s):
        raise RuntimeError("两条路径的候选检测时点网格不一致。")

    all_s, all_g = [], []
    for w_surv in [0.00, 0.25, 0.50, 0.75, 1.00]:
        combined = (1 - w_surv) * risk_q1 + w_surv * risk_surv
        summary, groups = optimize_all_k(
            bmi, times, combined, max_k=6, min_group_size=15
        )
        elbow = select_elbow(summary)
        summary.insert(0, "survival_weight", w_surv)
        summary["elbow_recommended"] = summary.k.eq(elbow)
        groups.insert(0, "survival_weight", w_surv)
        all_s.append(summary)
        all_g.append(groups)

    summary = pd.concat(all_s, ignore_index=True)
    groups = pd.concat(all_g, ignore_index=True)
    exact = groups[(groups.survival_weight == 0.50) & (groups.k == 2)].copy()
    weight_sens = groups[groups.k == 2].copy()

    combined = 0.5 * (risk_q1 + risk_surv)
    profile = k2_profile(bmi, times, combined)
    operational = profile.iloc[(profile.cut - 32.0).abs().argmin()]
    final_groups = pd.DataFrame([
        dict(group=1, lower=float(bmi.min()), upper=32.0, interval="[20.70, 32)",
             n=int((bmi < 32).sum()), best_week=operational.week_left),
        dict(group=2, lower=32.0, upper=float(bmi.max()), interval="[32, 46.88]",
             n=int((bmi >= 32).sum()), best_week=operational.week_right),
    ])
    final_groups["total_avg_risk"] = operational.avg_risk
    final_groups["regret_vs_exact"] = operational.regret

    summary.to_csv(OUT / "q2_final_k_curve.csv", index=False, encoding="utf-8-sig")
    weight_sens.to_csv(OUT / "q2_model_weight_sensitivity.csv", index=False,
                       encoding="utf-8-sig")
    profile.to_csv(OUT / "q2_boundary_profile.csv", index=False, encoding="utf-8-sig")
    final_groups.to_csv(OUT / "q2_final_groups.csv", index=False, encoding="utf-8-sig")

    print("\n等权集成的风险-组数曲线：")
    print(summary[summary.survival_weight == 0.50].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"
    ))
    print("\n等权目标的精确数学最优解（用于审计，不直接当临床切点）：")
    print(exact.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    near = profile[profile.within_0p1pct]
    print(f"\n风险距最优不超过0.1%的切点范围：{near.cut.min():.2f}～{near.cut.max():.2f}")
    print("推荐可执行方案（整数BMI=32切分）：")
    print(final_groups.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n模型权重敏感性（各权重下k=2）：")
    print(weight_sens.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n已输出 q2_final_k_curve.csv、q2_model_weight_sensitivity.csv、"
          "q2_boundary_profile.csv、q2_final_groups.csv")


if __name__ == "__main__":
    main()
