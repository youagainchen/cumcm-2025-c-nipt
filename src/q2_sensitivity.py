# -*- coding: utf-8 -*-
"""问题二：风险参数、检测误差和 AFT 参数不确定性传播。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from q2_optimize import load_baseline_bmi, optimize_all_k, risk_matrix
from q2_risk import ExpectedRisk, RiskParams
from q2_survival import load_prob_fn


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
COEF = OUT / "q1_coef.npy"
SURV = OUT / "q2_survival.npy"


def final_k() -> int:
    """采用 q2_final.py 判据选定的组数；未跑过则回退到当前定案值 3。"""
    plan = OUT / "q2_final_groups.csv"
    if plan.exists():
        return int(len(pd.read_csv(plan, encoding="utf-8-sig")))
    return 3


def final_k_result(label: str, instant: ExpectedRisk, censored: ExpectedRisk,
                   grid_step: float = 0.10) -> list[dict]:
    k = final_k()
    bmi = load_baseline_bmi().bmi.to_numpy(float)
    times, values_instant = risk_matrix(instant, bmi, grid_step)
    times_censored, values_censored = risk_matrix(censored, bmi, grid_step)
    if not np.allclose(times, times_censored):
        raise RuntimeError("两个风险输入组件的候选检测时点网格不一致。")
    values = 0.5 * (values_instant + values_censored)
    summary, groups = optimize_all_k(
        bmi, times, values, max_k=k, min_group_size=15
    )
    s = summary.loc[summary.k == k].iloc[0]
    out = []
    for row in groups[groups.k == k].itertuples():
        out.append(dict(
            scenario=label, group=row.group, lower=row.lower, upper=row.upper,
            n=row.n, best_week=row.best_week, group_risk=row.avg_risk,
            total_avg_risk=s.avg_risk, relative_reduction=s.relative_reduction,
        ))
    return out


def risk_and_measurement_sensitivity() -> pd.DataFrame:
    base = RiskParams()
    parameter_scenarios: list[tuple[str, RiskParams]] = [
        ("风险梯度低(r_mid=2)", replace(base, r_mid=2.0)),
        ("正式设定", base),
        ("风险梯度高(r_mid=5)", replace(base, r_mid=5.0)),
        ("低失联(q=0.05)", replace(base, q_dropout=0.05)),
        ("高失联(q=0.15)", replace(base, q_dropout=0.15)),
        ("短复检间隔(3.14周)", replace(base, retest_gap=3.14)),
        ("长复检间隔(4.14周)", replace(base, retest_gap=4.14)),
    ]

    aft_prob = load_prob_fn()
    rows = []
    for label, params in parameter_scenarios:
        print(f"敏感性场景：{label}")
        rows.extend(final_k_result(
            label,
            ExpectedRisk(params=params),
            ExpectedRisk(prob_fn=aft_prob, params=params),
        ))

    coef = np.load(COEF, allow_pickle=True).item()
    latent = dict(coef)
    latent["s2e"] = max(coef["s2e"] - coef["s2_tech"], 1e-9)
    elevated = dict(coef)
    elevated["s2e"] = coef["s2e"] + coef["s2_tech"]
    for label, coef_variant in [
        ("去除测序内误差(潜在值)", latent),
        ("测序内方差加倍", elevated),
    ]:
        print(f"敏感性场景：{label}")
        rows.extend(final_k_result(
            label,
            ExpectedRisk(coef=coef_variant, params=base),
            ExpectedRisk(prob_fn=aft_prob, params=base),
        ))
    return pd.DataFrame(rows)


def _nearest_psd(cov: np.ndarray) -> np.ndarray:
    cov = (cov + cov.T) / 2
    val, vec = np.linalg.eigh(cov)
    return (vec * np.maximum(val, 1e-12)) @ vec.T


def aft_parameter_uncertainty(draws: int = 1000, seed: int = 2025) -> tuple[pd.DataFrame, pd.DataFrame]:
    """传播 LogNormal AFT 的联合参数协方差到达标时间和固定方案风险。"""
    s = np.load(SURV, allow_pickle=True).item()
    if s["model"] != "LogNormal":
        raise RuntimeError("当前参数传播公式针对AIC最优的LogNormal AFT模型。")
    labels = [tuple(x) for x in s["param_labels"]]
    index = {label: i for i, label in enumerate(labels)}
    need = [("mu_", "bmi_c"), ("mu_", "Intercept"), ("sigma_", "Intercept")]
    if not all(x in index for x in need):
        raise RuntimeError(f"AFT参数字段不完整：{labels}")

    rng = np.random.default_rng(seed)
    theta = rng.multivariate_normal(
        np.asarray(s["param_values"], float),
        _nearest_psd(np.asarray(s["param_cov"], float)), size=draws,
    )
    mothers = load_baseline_bmi()
    plan_path = OUT / "q2_final_groups.csv"
    if not plan_path.exists():
        raise RuntimeError("未找到最终分组方案，请先运行 q2_final.py。")
    plan = pd.read_csv(plan_path, encoding="utf-8-sig")
    if len(plan) < 2:
        raise RuntimeError("q2_final_groups.csv 的分组数异常。")

    b_ref = [30.0, 35.0, 40.0]
    draw_rows, risk_rows = [], []
    params = RiskParams()
    for draw_id, th in enumerate(theta):
        beta = th[index[("mu_", "bmi_c")]]
        intercept = th[index[("mu_", "Intercept")]]
        sigma = np.exp(th[index[("sigma_", "Intercept")]])

        for b in b_ref:
            mu = intercept + beta * (b - s["mu_b"])
            draw_rows.append(dict(
                draw=draw_id, bmi=b,
                t80=np.exp(mu + sigma * st.norm.ppf(0.80)),
                t90=np.exp(mu + sigma * st.norm.ppf(0.90)),
            ))

        def prob_fn(week, bmi_baseline, thr=0.04, prev_week=None):
            if not np.isclose(thr, 0.04):
                raise ValueError("AFT仅针对4%阈值。")
            week = np.asarray(week, float)
            bmi = np.asarray(bmi_baseline, float)
            mu = intercept + beta * (bmi - s["mu_b"])
            now = st.norm.cdf((np.log(np.maximum(week, 1e-9)) - mu) / sigma)
            if prev_week is None:
                return now
            prev = st.norm.cdf((np.log(np.maximum(prev_week, 1e-9)) - mu) / sigma)
            return np.clip((now - prev) / np.maximum(1 - prev, 1e-12), 0, 1)

        instant = ExpectedRisk(params=params)
        censored = ExpectedRisk(prob_fn=prob_fn, params=params)
        total, count = 0.0, 0
        for row in plan.itertuples():
            if row.group < len(plan):
                subset = mothers[(mothers.bmi >= row.lower) & (mothers.bmi < row.upper)]
            else:
                subset = mothers[(mothers.bmi >= row.lower) & (mothers.bmi <= row.upper)]
            total += sum(
                0.5 * instant.expected_risk(row.best_week, b)
                + 0.5 * censored.expected_risk(row.best_week, b)
                for b in subset.bmi
            )
            count += len(subset)
        risk_rows.append(dict(draw=draw_id, plan_avg_risk=total / count))

    draws_df = pd.DataFrame(draw_rows)
    risk_df = pd.DataFrame(risk_rows)
    summary_rows = []
    for b, g in draws_df.groupby("bmi"):
        for metric in ("t80", "t90"):
            q = g[metric].quantile([0.025, 0.5, 0.975])
            summary_rows.append(dict(quantity=metric, bmi=b, q025=q.iloc[0],
                                     median=q.iloc[1], q975=q.iloc[2]))
    q = risk_df.plan_avg_risk.quantile([0.025, 0.5, 0.975])
    summary_rows.append(dict(quantity="fixed_plan_avg_risk", bmi=np.nan,
                             q025=q.iloc[0], median=q.iloc[1], q975=q.iloc[2]))
    return pd.DataFrame(summary_rows), pd.concat(
        [draws_df, risk_df.assign(bmi=np.nan, t80=np.nan, t90=np.nan)], ignore_index=True
    )


def main() -> None:
    sens = risk_and_measurement_sensitivity()
    sens.to_csv(OUT / "q2_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n风险参数与测量误差敏感性：")
    print(sens.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary, draws = aft_parameter_uncertainty()
    summary.to_csv(OUT / "q2_uncertainty_summary.csv", index=False, encoding="utf-8-sig")
    draws.to_csv(OUT / "q2_uncertainty_draws.csv", index=False, encoding="utf-8-sig")
    print("\nAFT参数不确定性传播（1000次联合参数抽样）：")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n已输出 q2_sensitivity.csv、q2_uncertainty_summary.csv、q2_uncertainty_draws.csv")


if __name__ == "__main__":
    main()
