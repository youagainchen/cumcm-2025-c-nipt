# -*- coding: utf-8 -*-
"""问题三（二号）：最终分组对风险参数和概率接口变体的敏感性。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import expit

from q2_risk import RiskParams
from q3_optimize import (
    build_risk_matrix,
    exact_scan,
    load_instances,
    load_probability_interface,
    mock_prob_qualified,
    optimize_integer_boundaries,
    validate_probability_interface,
)


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "outputs" / "q3_model.npy"


def _nearest_psd(covariance: np.ndarray) -> np.ndarray:
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, 1e-12)) @ vectors.T


def probability_from_parameters(payload: dict, theta: np.ndarray):
    """由一组AFT参数构造统一概率接口，不修改一号模型文件。"""
    labels = [tuple(x) for x in payload["param_labels"]]
    params = dict(zip(labels, np.asarray(theta, float)))
    model = payload["model"]
    features = payload["features"]
    stats = payload["feature_stats"]
    location_name = {
        "Weibull": "lambda_", "LogNormal": "mu_", "LogLogistic": "alpha_"
    }[model]

    def cdf(week, bmi, height, weight, age):
        raw = {"bmi": bmi, "height": height, "weight": weight, "age": age}
        z = {
            name + "_z": (np.asarray(raw[name], float) - stats[name][0]) / stats[name][1]
            for name in ("bmi", "height", "weight", "age")
        }
        if "bmi_hw_resid" in features:
            coeff = stats["bmi_hw_resid_coefficients"]
            z["bmi_hw_resid"] = z["bmi_z"] - (
                coeff[0] + coeff[1] * z["height_z"] + coeff[2] * z["weight_z"]
            )
        location = np.asarray(params[(location_name, "Intercept")], float)
        for feature in features:
            location = location + params[(location_name, feature)] * z[feature]
        log_week = np.log(np.maximum(np.asarray(week, float), 1e-9))
        if model == "LogNormal":
            sigma = np.exp(params[("sigma_", "Intercept")])
            return st.norm.cdf((log_week - location) / sigma)
        if model == "Weibull":
            rho = np.exp(params[("rho_", "Intercept")])
            return 1 - np.exp(-np.exp(rho * (log_week - location)))
        beta = np.exp(params[("beta_", "Intercept")])
        return expit(beta * (log_week - location))

    def prob_qualified(week, bmi_baseline, height, weight, age,
                       thr=0.04, prev_week=None):
        if not np.isclose(thr, 0.04):
            raise ValueError("AFT参数抽样仅支持4%阈值。")
        now = np.asarray(cdf(week, bmi_baseline, height, weight, age), float)
        if prev_week is None:
            return now
        previous = np.asarray(cdf(prev_week, bmi_baseline, height, weight, age), float)
        return np.clip((now - previous) / np.maximum(1 - previous, 1e-12), 0, 1)

    return prob_qualified


def parameter_variants(draws: int, seed: int) -> dict[str, object]:
    if draws <= 0:
        return {}
    if not MODEL_PATH.exists():
        raise RuntimeError("缺少 outputs/q3_model.npy，无法传播参数不确定性。")
    payload = np.load(MODEL_PATH, allow_pickle=True).item()
    rng = np.random.default_rng(seed)
    theta = rng.multivariate_normal(
        np.asarray(payload["param_values"], float),
        _nearest_psd(np.asarray(payload["param_cov"], float)),
        size=draws,
    )
    return {
        f"参数抽样{index:03d}": probability_from_parameters(payload, draw)
        for index, draw in enumerate(theta, 1)
    }


def one_scenario(
    label,
    prob_fn,
    params: RiskParams,
    people: pd.DataFrame,
    grid_step: float,
    kind: str,
) -> dict:
    times, risks = build_risk_matrix(people, prob_fn, params, grid_step)
    summary, exact_groups, criteria, k_star = exact_scan(
        people, times, risks, allow_unselected=True
    )
    if k_star is None:
        failures = []
        for row in criteria.itertuples():
            failed = []
            if not row.C1_收益捕获:
                failed.append("C1")
            if not row.C2_组间可辨:
                failed.append("C2")
            if not row.C3_样本量:
                failed.append("C3")
            failures.append(f"k={row.k}:{'/'.join(failed) or '-'}")
        return {
            "scenario_type": kind,
            "scenario": label,
            "status": "no_feasible_k",
            "criterion_failures": ";".join(failures),
            "k": np.nan,
            "integer_cuts": "",
            "group_n": "",
            "best_weeks": "",
            "avg_risk": np.nan,
            "regret_vs_exact": np.nan,
            "min_group_n": np.nan,
        }
    exact_avg = float(summary.loc[summary.k == k_star, "avg_risk"].iloc[0])
    groups, cuts, avg_risk = optimize_integer_boundaries(
        people, times, risks, k_star, exact_avg
    )
    return {
        "scenario_type": kind,
        "scenario": label,
        "status": "ok",
        "criterion_failures": "",
        "k": k_star,
        "integer_cuts": "|".join(str(x) for x in cuts),
        "group_n": "|".join(str(int(x)) for x in groups.n),
        "best_weeks": "|".join(f"{x:.2f}" for x in groups.best_week),
        "recommended_times": "|".join(groups.recommended_time.astype(str)),
        "avg_risk": avg_risk,
        "rounded_avg_risk": float(groups.rounded_total_avg_risk.iloc[0]),
        "regret_vs_exact": avg_risk - exact_avg,
        "min_group_n": int(groups.n.min()),
    }


def optional_probability_variants(module_name: str | None) -> dict:
    if not module_name:
        for candidate in ("q3_model", "q3_survival"):
            try:
                module = importlib.import_module(candidate)
                break
            except Exception:
                module = None
    else:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            module = None
    if module is None:
        return {}
    factory = getattr(module, "probability_variants", None)
    if not callable(factory):
        return {}
    variants = factory()
    if not isinstance(variants, dict) or not all(callable(v) for v in variants.values()):
        raise RuntimeError("probability_variants() 必须返回 {名称: 可调用概率函数}。")
    return variants


def run(prob_fn, source: str, mock: bool, module_name: str | None,
        grid_step: float = 0.05, parameter_draws: int = 0,
        seed: int = 2026) -> pd.DataFrame:
    people = load_instances()
    validate_probability_interface(prob_fn, people)
    base = RiskParams()
    parameter_scenarios = [
        ("正式设定", base),
        ("风险梯度低(r_mid=2)", replace(base, r_mid=2.0)),
        ("风险梯度高(r_mid=5)", replace(base, r_mid=5.0)),
        ("低失联(q=0.05)", replace(base, q_dropout=0.05)),
        ("高失联(q=0.15)", replace(base, q_dropout=0.15)),
        ("短复检间隔(3.14周)", replace(base, retest_gap=3.14)),
        ("长复检间隔(4.14周)", replace(base, retest_gap=4.14)),
    ]
    rows = []
    for label, params in parameter_scenarios:
        print(f"风险参数敏感性：{label}")
        rows.append(one_scenario(
            label, prob_fn, params, people, grid_step, "risk_parameter"
        ))

    if not mock:
        for label, variant in optional_probability_variants(module_name).items():
            print(f"概率模型/检测误差敏感性：{label}")
            validate_probability_interface(variant, people)
            rows.append(one_scenario(
                label, variant, base, people, grid_step, "probability_variant"
            ))
        for label, variant in parameter_variants(parameter_draws, seed).items():
            print(f"AFT参数不确定性重优化：{label}")
            validate_probability_interface(variant, people)
            rows.append(one_scenario(
                label, variant, base, people, grid_step,
                "parameter_covariance_draw",
            ))

    result = pd.DataFrame(rows)
    if not mock:
        formal_path = ROOT / "outputs" / "q3_groups.csv"
        if formal_path.exists():
            formal = pd.read_csv(formal_path, encoding="utf-8-sig")
            baseline = result[result.scenario.eq("正式设定")].iloc[0]
            expected_cuts = "|".join(
                f"{float(x):g}" for x in formal.upper.iloc[:-1]
            )
            expected_weeks = "|".join(f"{float(x):.2f}" for x in formal.best_week)
            if (
                baseline.status != "ok"
                or int(baseline.k) != len(formal)
                or baseline.integer_cuts != expected_cuts
                or baseline.best_weeks != expected_weeks
            ):
                raise RuntimeError("敏感性基准情景未精确复现问题三正式方案。")
    prefix = "q3_mock" if mock else "q3"
    result.to_csv(
        ROOT / "outputs" / f"{prefix}_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    draws = result[result.scenario_type.eq("parameter_covariance_draw")]
    if len(draws):
        stability = (
            draws.groupby(["status", "k", "integer_cuts"], dropna=False)
            .size().rename("count").reset_index()
        )
        stability["fraction"] = stability["count"] / len(draws)
        stability["total_draws"] = len(draws)
        stability.to_csv(
            ROOT / "outputs" / "q3_boundary_stability.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"概率接口：{source}")
    print(result.to_string(index=False))
    print(f"已输出 outputs/{prefix}_sensitivity.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument(
        "--parameter-draws", type=int, default=100,
        help="AFT参数协方差蒙特卡洛重优化次数；完整分析建议至少100次",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.mock:
        prob_fn, source = mock_prob_qualified, "mock（仅联调）"
    else:
        prob_fn, source = load_probability_interface(args.module)
    run(
        prob_fn, source, args.mock, args.module, args.grid_step,
        args.parameter_draws, args.seed,
    )


if __name__ == "__main__":
    main()
