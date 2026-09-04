# -*- coding: utf-8 -*-
"""问题三（二号）：最终分组对风险参数和概率接口变体的敏感性。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

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
        "avg_risk": avg_risk,
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
        grid_step: float = 0.10) -> pd.DataFrame:
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

    result = pd.DataFrame(rows)
    prefix = "q3_mock" if mock else "q3"
    result.to_csv(
        ROOT / "outputs" / f"{prefix}_sensitivity.csv",
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
    parser.add_argument("--grid-step", type=float, default=0.10)
    args = parser.parse_args()
    if args.mock:
        prob_fn, source = mock_prob_qualified, "mock（仅联调）"
    else:
        prob_fn, source = load_probability_interface(args.module)
    run(prob_fn, source, args.mock, args.module, args.grid_step)


if __name__ == "__main__":
    main()
