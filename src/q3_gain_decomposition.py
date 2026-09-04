# -*- coding: utf-8 -*-
"""问题三收益分解：把"问题三优于问题二"拆成时点贡献与边界贡献。

动机
----
直接比较两问的总平均风险（问题二 1.910156、问题三 2.000668）是无意义的：
两问使用不同的达标概率模型，风险数值不同尺度。把问题二方案放到问题三的
风险口径下重新评分（2.030496）虽然可比，但把整个差值归因于"多因素建模"
仍然是错的——差值里绝大部分来自"在新概率模型下重新选时点"，与是否引入
身高体重无关。

本脚本在同一风险口径（问题三多因素期望风险）下评三个方案：

    A  问题二边界 + 问题二时点     即问题二原方案
    B  问题二边界 + 重优化时点     只换时点，边界不动
    C  问题三边界 + 重优化时点     问题三正式方案

于是
    时点贡献 = A - B
    边界贡献 = B - C
    总收益   = A - C

输出
----
outputs/q3_gain_decomposition.csv

运行
----
python src/q3_gain_decomposition.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from q3_optimize import (build_risk_matrix, load_instances,
                         load_probability_interface)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# 问题二正式方案（outputs/q2_final_groups.csv）
Q2_CUTS = (31.5, 37.0)
Q2_TIMES = (12.35, 14.20, 17.90)


def read_q3_cuts() -> tuple[float, ...]:
    """从问题三正式输出读整数边界，避免与 q3_optimize 的结果脱节。"""
    path = OUT / "q3_groups.csv"
    if not path.exists():
        raise SystemExit("未找到 outputs/q3_groups.csv，请先运行 python src/q3_optimize.py")
    groups = pd.read_csv(path, encoding="utf-8-sig").sort_values("group")
    return tuple(float(x) for x in groups["upper"].to_numpy()[:-1])


def evaluate(bmi: np.ndarray, times: np.ndarray, risks: np.ndarray,
             cuts: tuple[float, ...], fixed_times: tuple[float, ...] | None,
             plan: str, note: str) -> tuple[float, list[dict]]:
    """给定边界评分；fixed_times 为 None 时对每组重新取风险最小的时点。"""
    edges = (float(bmi.min()) - 1e-9, *cuts, float(bmi.max()) + 1e-9)
    total, rows = 0.0, []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (bmi >= lower) & (bmi < upper)
        if not mask.any():
            raise RuntimeError(f"方案 {plan} 的第 {index+1} 组为空")
        cost = risks[mask].sum(axis=0)
        if fixed_times is None:
            position = int(np.argmin(cost))
        else:
            position = int(np.argmin(np.abs(times - fixed_times[index])))
        total += float(cost[position])
        rows.append({
            "plan": plan, "note": note, "group": index + 1,
            "lower": float(max(lower, bmi.min())),
            "upper": float(min(upper, bmi.max())),
            "n": int(mask.sum()),
            "best_week": float(times[position]),
            "group_avg_risk": float(cost[position] / mask.sum()),
        })
    average = total / len(bmi)
    for row in rows:
        row["total_avg_risk"] = average
    return average, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="问题三收益分解")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--module", default=None)
    args = parser.parse_args()

    people = load_instances()
    prob_fn, source = load_probability_interface(args.module)
    print(f"概率接口：{source}；孕妇数 {len(people)}")
    times, risks = build_risk_matrix(people, prob_fn, grid_step=args.grid_step)
    bmi = people.bmi.to_numpy(float)
    q3_cuts = read_q3_cuts()

    plans = [
        ("A", Q2_CUTS, Q2_TIMES, "问题二边界+问题二时点（原方案）"),
        ("B", Q2_CUTS, None, "问题二边界+按问题三模型重优化时点"),
        ("C", q3_cuts, None, "问题三边界+重优化时点（正式方案）"),
    ]
    averages, all_rows = {}, []
    for plan, cuts, fixed, note in plans:
        average, rows = evaluate(bmi, times, risks, cuts, fixed, plan, note)
        averages[plan] = average
        all_rows.extend(rows)
        print(f"{plan} {note:32s} 平均风险={average:.6f}")

    total_gain = averages["A"] - averages["C"]
    time_gain = averages["A"] - averages["B"]
    cut_gain = averages["B"] - averages["C"]
    summary = pd.DataFrame([
        {"component": "总收益 A-C", "absolute": total_gain,
         "share_of_total": 1.0, "relative_to_A": total_gain / averages["A"]},
        {"component": "时点贡献 A-B", "absolute": time_gain,
         "share_of_total": time_gain / total_gain,
         "relative_to_A": time_gain / averages["A"]},
        {"component": "边界贡献 B-C", "absolute": cut_gain,
         "share_of_total": cut_gain / total_gain,
         "relative_to_A": cut_gain / averages["A"]},
    ])

    detail = pd.DataFrame(all_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "q3_gain_decomposition.csv", "w", encoding="utf-8-sig",
              newline="") as handle:
        handle.write("# 方案明细（统一在问题三多因素期望风险口径下评分）\n")
        detail.to_csv(handle, index=False)
        handle.write("\n# 收益分解\n")
        summary.to_csv(handle, index=False)

    print("\n收益分解：")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\n时点贡献占总收益 {time_gain/total_gain*100:.1f}%，"
          f"边界贡献占 {cut_gain/total_gain*100:.1f}%")
    print("已输出 outputs/q3_gain_decomposition.csv")


if __name__ == "__main__":
    main()
