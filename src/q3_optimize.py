# -*- coding: utf-8 -*-
"""问题三（二号）：基于多因素达标概率的BMI连续分组与检测时点优化。

正式运行会自动加载一号提供的 q3_model.prob_qualified 或
q3_survival.prob_qualified。接口未到位前可用 --mock 验证整条优化链，
模拟结果只写入 q3_mock_*，不会冒充正式结论。
"""
from __future__ import annotations

import argparse
import importlib
import itertools
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from q2_final import CAPTURE_FRAC, MIN_WEEK_GAP
from q2_optimize import optimize_all_k
from q2_risk import RiskParams, T_MAX, T_MIN, risk_curve


ROOT = Path(__file__).resolve().parent.parent
EVENT = ROOT / "data" / "processed" / "male_clean_event.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
MIN_GROUP_SIZE = 15
MAX_K = 6

ProbabilityFn = Callable[..., float | np.ndarray]


def load_instances() -> pd.DataFrame:
    """每位孕妇保留首次检测时已经可知的多因素信息。"""
    data = pd.read_csv(EVENT, encoding="utf-8-sig")
    data = data.sort_values(["mother_id", "week_mean", "visit_idx"])
    people = data.groupby("mother_id", as_index=False).agg(
        bmi=("bmi", "first"),
        height=("height", "first"),
        weight=("weight", "first"),
        age=("age", "first"),
    )
    required = ["mother_id", "bmi", "height", "weight", "age"]
    people = people.dropna(subset=required).copy()
    people = people.sort_values(["bmi", "mother_id"]).reset_index(drop=True)
    if people.mother_id.duplicated().any():
        raise RuntimeError("孕妇ID重复，无法保证每位孕妇只分配一次。")
    return people


def mock_prob_qualified(
    week,
    bmi_baseline,
    height,
    weight,
    age,
    thr=0.04,
    prev_week=None,
):
    """仅用于T2联调的平滑模拟概率，不作为问题三模型或论文结果。"""
    if not np.isclose(thr, 0.04):
        raise ValueError("模拟接口仅支持4%阈值。")
    week = np.asarray(week, float)
    bmi = np.asarray(bmi_baseline, float)
    height = np.asarray(height, float)
    weight = np.asarray(weight, float)
    age = np.asarray(age, float)
    # 体重只通过相对 BMI*身高² 的偏差进入，避免简单重复BMI信息。
    expected_weight = bmi * (height / 100.0) ** 2
    linear = (
        -0.25
        + 0.33 * (week - 12.0)
        - 0.105 * (bmi - 30.0)
        + 0.018 * (height - 165.0)
        - 0.015 * (weight - expected_weight)
        - 0.018 * (age - 30.0)
    )
    now = 1.0 / (1.0 + np.exp(-linear))
    if prev_week is None:
        return float(now) if now.ndim == 0 else now
    prev_week = np.asarray(prev_week, float)
    previous_linear = linear - 0.33 * (week - prev_week)
    previous = 1.0 / (1.0 + np.exp(-previous_linear))
    conditional = np.clip((now - previous) / np.maximum(1.0 - previous, 1e-12), 0, 1)
    return float(conditional) if conditional.ndim == 0 else conditional


def load_probability_interface(module_name: str | None = None) -> tuple[ProbabilityFn, str]:
    candidates = [module_name] if module_name else ["q3_model", "q3_survival"]
    errors = []
    for name in candidates:
        if not name:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        fn = getattr(module, "prob_qualified", None)
        if callable(fn):
            return fn, f"{name}.prob_qualified"
        factory = getattr(module, "load_prob_fn", None)
        if callable(factory):
            try:
                fn = factory()
            except Exception as exc:
                errors.append(f"{name}.load_prob_fn(): {exc}")
                continue
            if callable(fn):
                return fn, f"{name}.load_prob_fn()"
        errors.append(f"{name}: 缺少 prob_qualified 或 load_prob_fn")
    detail = "；".join(errors) if errors else "未找到候选模块"
    raise RuntimeError(
        "尚未取得一号的多因素概率接口或模型产物。请提供 src/q3_model.py 或 "
        f"src/q3_survival.py；详情：{detail}"
    )


def validate_probability_interface(prob_fn: ProbabilityFn, people: pd.DataFrame) -> None:
    """在全部个体和递归使用区间内验证累计概率与条件概率。"""
    weeks = np.arange(T_MIN, RiskParams().horizon + 1e-9, 0.5)
    for row in people.itertuples():
        cumulative = []
        for week in weeks:
            p = float(np.asarray(prob_fn(
                week, row.bmi, row.height, row.weight, row.age,
                thr=0.04, prev_week=None,
            )))
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise RuntimeError(
                    f"概率接口返回非法值：mother={row.mother_id}, p={p}, week={week}"
                )
            cumulative.append(p)
        cumulative = np.asarray(cumulative)
        if np.any(np.diff(cumulative) < -1e-10):
            raise RuntimeError(f"累计达标概率随孕周下降：mother={row.mother_id}")

        for previous, current in ((12.0, 16.0), (16.0, 20.0), (20.0, 24.0)):
            p_previous = float(np.asarray(prob_fn(
                previous, row.bmi, row.height, row.weight, row.age,
                thr=0.04, prev_week=None,
            )))
            p_current = float(np.asarray(prob_fn(
                current, row.bmi, row.height, row.weight, row.age,
                thr=0.04, prev_week=None,
            )))
            conditional = float(np.asarray(prob_fn(
                current, row.bmi, row.height, row.weight, row.age,
                thr=0.04, prev_week=previous,
            )))
            expected = np.clip(
                (p_current - p_previous) / max(1.0 - p_previous, 1e-12), 0, 1
            )
            if (
                not np.isfinite(conditional)
                or not 0 <= conditional <= 1
                or not np.isclose(conditional, expected, atol=1e-8, rtol=1e-6)
            ):
                raise RuntimeError(
                    "条件概率与累计概率不一致："
                    f"mother={row.mother_id}, prev={previous}, week={current}, "
                    f"returned={conditional}, expected={expected}"
                )


def expected_risk(
    first_week: float,
    row,
    prob_fn: ProbabilityFn,
    params: RiskParams,
) -> float:
    """保持个体协变量不变，只在复检递归中推进孕周。"""
    reach = 1.0
    total = 0.0
    for attempt in range(params.max_retest + 1):
        week = float(first_week + attempt * params.retest_gap)
        if week > params.horizon:
            break
        prev_week = None if attempt == 0 else week - params.retest_gap
        p = float(np.asarray(prob_fn(
            week, row.bmi, row.height, row.weight, row.age,
            thr=0.04, prev_week=prev_week,
        )))
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise RuntimeError(
                f"概率非法：mother={row.mother_id}, week={week}, p={p}"
            )
        total += reach * p * float(risk_curve(week, params))
        failed = reach * (1.0 - p)
        total += failed * params.q_dropout * params.r_dropout
        reach = failed * (1.0 - params.q_dropout)
        if reach < 1e-6:
            break
    end_week = min(
        first_week + params.max_retest * params.retest_gap,
        params.horizon,
    )
    total += reach * float(risk_curve(end_week, params))
    return float(total)


def build_risk_matrix(
    people: pd.DataFrame,
    prob_fn: ProbabilityFn,
    params: RiskParams | None = None,
    grid_step: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    params = params or RiskParams()
    times = np.arange(T_MIN, T_MAX + 1e-9, grid_step)
    shape = (len(people), len(times))
    base_week = np.broadcast_to(times[None, :], shape)
    bmi = people.bmi.to_numpy(float)[:, None]
    height = people.height.to_numpy(float)[:, None]
    weight = people.weight.to_numpy(float)[:, None]
    age = people.age.to_numpy(float)[:, None]
    reach = np.ones(shape)
    values = np.zeros(shape)
    try:
        for attempt in range(params.max_retest + 1):
            week = base_week + attempt * params.retest_gap
            active = week <= params.horizon
            previous = None if attempt == 0 else week - params.retest_gap
            probability = np.asarray(prob_fn(
                week, bmi, height, weight, age,
                thr=0.04, prev_week=previous,
            ), float)
            probability = np.broadcast_to(probability, shape)
            if np.any(active & (
                ~np.isfinite(probability) | (probability < 0) | (probability > 1)
            )):
                raise RuntimeError("向量化概率接口返回非法值。")
            success = np.where(active, reach * probability, 0.0)
            values += success * risk_curve(week, params)
            failed = np.where(active, reach * (1.0 - probability), 0.0)
            values += failed * params.q_dropout * params.r_dropout
            reach = np.where(active, failed * (1.0 - params.q_dropout), reach)
        end_week = np.minimum(
            base_week + params.max_retest * params.retest_gap,
            params.horizon,
        )
        values += reach * risk_curve(end_week, params)
    except (TypeError, ValueError, IndexError):
        # 兼容只接受标量的外部概率接口；正式接口的非法概率仍由下方检查拒绝。
        values = np.empty(shape)
        for i, row in enumerate(people.itertuples()):
            values[i] = [expected_risk(t, row, prob_fn, params) for t in times]
    if not np.all(np.isfinite(values)):
        raise RuntimeError("问题三风险矩阵含非有限值。")
    return times, values


def exact_scan(
    people: pd.DataFrame,
    times: np.ndarray,
    risks: np.ndarray,
    allow_unselected: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int | None]:
    summary, groups = optimize_all_k(
        people.bmi.to_numpy(float),
        times,
        risks,
        max_k=MAX_K,
        min_group_size=MIN_GROUP_SIZE,
    )
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
            "captured_frac": captured,
            "min_week_gap": float(gaps.min()),
            "min_group_n": int(g.n.min()),
            "C1_收益捕获": bool(k > 1 and captured >= CAPTURE_FRAC),
            "C2_组间可辨": bool(gaps.min() >= MIN_WEEK_GAP),
            "C3_样本量": bool(g.n.min() >= MIN_GROUP_SIZE),
        })
    criteria = pd.DataFrame(rows)
    criteria["全部满足"] = (
        criteria.C1_收益捕获 & criteria.C2_组间可辨 & criteria.C3_样本量
    )
    feasible = criteria[criteria["全部满足"]]
    if feasible.empty:
        if not allow_unselected:
            raise RuntimeError("没有任何 k 同时满足三条判据，需重新审视判据参数")
        k_star = None
    else:
        k_star = int(feasible.k.iloc[0])
    criteria["selected"] = criteria.k.eq(k_star) if k_star is not None else False
    summary = summary.merge(
        criteria[["k", "captured_frac", "min_week_gap", "min_group_n",
                  "C1_收益捕获", "C2_组间可辨", "C3_样本量", "selected"]],
        on="k",
        how="left",
    )
    return summary, groups, criteria, k_star


def optimize_integer_boundaries(
    people: pd.DataFrame,
    times: np.ndarray,
    risks: np.ndarray,
    k: int,
    exact_avg_risk: float,
) -> tuple[pd.DataFrame, tuple[int, ...], float]:
    """枚举全部整数BMI边界组合，返回风险最小且满足样本量约束的方案。"""
    bmi = people.bmi.to_numpy(float)
    prefix = np.vstack([np.zeros(len(times)), np.cumsum(risks, axis=0)])
    candidate_positions: dict[int, int] = {}
    for cut in range(int(np.ceil(bmi.min())), int(np.floor(bmi.max())) + 1):
        position = int(np.searchsorted(bmi, cut, side="left"))
        if MIN_GROUP_SIZE <= position <= len(bmi) - MIN_GROUP_SIZE:
            candidate_positions[cut] = position

    best = None
    for cuts in itertools.combinations(candidate_positions, k - 1):
        positions = [0, *(candidate_positions[c] for c in cuts), len(bmi)]
        if any(b - a < MIN_GROUP_SIZE for a, b in zip(positions[:-1], positions[1:])):
            continue
        total = 0.0
        time_indices = []
        for left, right in zip(positions[:-1], positions[1:]):
            costs = prefix[right] - prefix[left]
            idx = int(np.argmin(costs))
            total += float(costs[idx])
            time_indices.append(idx)
        if best is None or total < best[0]:
            best = (total, cuts, positions, time_indices)
    if best is None:
        raise RuntimeError(f"不存在满足每组至少{MIN_GROUP_SIZE}人的{k}组整数边界。")

    total, cuts, positions, time_indices = best
    edges = [float(bmi.min()), *(float(x) for x in cuts), float(bmi.max())]
    rows = []
    rounded_total = 0.0
    for group, (left, right, lo, hi, ti) in enumerate(
        zip(positions[:-1], positions[1:], edges[:-1], edges[1:], time_indices), 1
    ):
        best_week = float(times[ti])
        rounded_days = int(np.rint(best_week * 7))
        rounded_week = rounded_days / 7.0
        rounded_cost = float(np.interp(
            rounded_week, times, prefix[right] - prefix[left]
        ))
        rounded_total += rounded_cost
        rows.append({
            "group": group,
            "lower": lo,
            "upper": hi,
            "left_closed": True,
            "right_closed": group == k,
            "interval": (
                f"[{lo:.2f}, {hi:g})" if group < k else f"[{lo:g}, {hi:.2f}]"
            ),
            "n": right - left,
            "bmi_min": float(bmi[left]),
            "bmi_max": float(bmi[right - 1]),
            "best_week": best_week,
            "recommended_week": rounded_week,
            "recommended_time": f"{rounded_days // 7}周+{rounded_days % 7}天",
            "avg_risk": float((prefix[right] - prefix[left])[ti] / (right - left)),
            "rounded_avg_risk": rounded_cost / (right - left),
        })
    groups = pd.DataFrame(rows)
    total_avg = total / len(bmi)
    groups["total_avg_risk"] = total_avg
    groups["regret_vs_exact"] = total_avg - exact_avg_risk
    groups["rounded_total_avg_risk"] = rounded_total / len(bmi)
    groups["time_rounding_regret"] = rounded_total / len(bmi) - total_avg
    return groups, tuple(cuts), total_avg


def assign_instances(
    people: pd.DataFrame,
    groups: pd.DataFrame,
    prob_fn: ProbabilityFn,
    params: RiskParams,
) -> pd.DataFrame:
    assigned = people.copy()
    cuts = groups.upper.iloc[:-1].to_numpy(float)
    assigned["group"] = np.searchsorted(cuts, assigned.bmi.to_numpy(float), side="right") + 1
    week_map = groups.set_index("group").best_week
    recommended_week_map = groups.set_index("group").recommended_week
    recommended_time_map = groups.set_index("group").recommended_time
    assigned["best_week"] = assigned.group.map(week_map)
    assigned["recommended_week"] = assigned.group.map(recommended_week_map)
    assigned["recommended_time"] = assigned.group.map(recommended_time_map)
    probabilities, risks = [], []
    recommended_probabilities, recommended_risks = [], []
    for row in assigned.itertuples():
        probabilities.append(float(np.asarray(prob_fn(
            row.best_week, row.bmi, row.height, row.weight, row.age,
            thr=0.04, prev_week=None,
        ))))
        risks.append(expected_risk(row.best_week, row, prob_fn, params))
        recommended_probabilities.append(float(np.asarray(prob_fn(
            row.recommended_week, row.bmi, row.height, row.weight, row.age,
            thr=0.04, prev_week=None,
        ))))
        recommended_risks.append(
            expected_risk(row.recommended_week, row, prob_fn, params)
        )
    assigned["qualified_probability"] = probabilities
    assigned["expected_risk"] = risks
    assigned["recommended_qualified_probability"] = recommended_probabilities
    assigned["recommended_expected_risk"] = recommended_risks
    return assigned


def compare_with_q2(
    people: pd.DataFrame,
    times: np.ndarray,
    risks: np.ndarray,
    q3_groups: pd.DataFrame,
) -> pd.DataFrame:
    """在同一套问题三个体风险下公平比较问题二与问题三方案。"""
    q2_path = OUT / "q2_final_groups.csv"
    q3 = q3_groups[[
        "group", "interval", "n", "recommended_week", "rounded_avg_risk",
        "rounded_total_avg_risk",
    ]].rename(columns={
        "recommended_week": "best_week",
        "rounded_avg_risk": "avg_risk",
        "rounded_total_avg_risk": "total_avg_risk",
    }).copy()
    q3.insert(0, "model", "q3_multifactor_operational")
    q3.insert(1, "metric_basis", "q3_multifactor_expected_risk")
    if not q2_path.exists():
        return q3
    q2_plan = pd.read_csv(q2_path, encoding="utf-8-sig").sort_values("group")
    bmi = people.bmi.to_numpy(float)
    coverage = np.zeros(len(people), dtype=int)
    rows = []
    total = 0.0
    for row in q2_plan.itertuples():
        if row.group < len(q2_plan):
            mask = (bmi >= row.lower) & (bmi < row.upper)
        else:
            mask = (bmi >= row.lower) & (bmi <= row.upper)
        if not mask.any():
            raise RuntimeError(f"问题二第{row.group}组在问题三样本中为空。")
        coverage += mask.astype(int)
        idx = int(np.argmin(np.abs(times - float(row.best_week))))
        group_total = float(risks[mask, idx].sum())
        total += group_total
        rows.append({
            "model": "q2_official_fixed",
            "metric_basis": "q3_multifactor_expected_risk",
            "group": int(row.group),
            "interval": row.interval,
            "n": int(mask.sum()),
            "best_week": float(row.best_week),
            "avg_risk": group_total / int(mask.sum()),
        })
    if not np.all(coverage == 1):
        raise RuntimeError("问题二方案未在问题三样本上实现完整且唯一覆盖。")
    q2 = pd.DataFrame(rows)
    q2["total_avg_risk"] = total / len(people)
    return pd.concat([q2, q3], ignore_index=True)


def validate_outputs(
    people: pd.DataFrame,
    summary: pd.DataFrame,
    groups: pd.DataFrame,
    instances: pd.DataFrame,
) -> None:
    if len(instances) != len(people) or instances.mother_id.nunique() != len(people):
        raise RuntimeError("每位孕妇未被恰好分配一次。")
    if int(groups.n.sum()) != len(people) or int(groups.n.min()) < MIN_GROUP_SIZE:
        raise RuntimeError("分组人数不满足覆盖或最小样本量约束。")
    if np.any(np.diff(summary.avg_risk.to_numpy(float)) > 1e-10):
        raise RuntimeError("风险随组数增加而上升。")
    if not instances.qualified_probability.between(0, 1).all():
        raise RuntimeError("个体达标概率不在[0,1]内。")
    if not instances.recommended_qualified_probability.between(0, 1).all():
        raise RuntimeError("整天推荐时点的个体达标概率不在[0,1]内。")
    if not np.allclose(
        groups.upper.iloc[:-1].to_numpy(float),
        groups.lower.iloc[1:].to_numpy(float),
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError("BMI分组不连续。")
    expected_counts = groups.set_index("group").n.astype(int).sort_index()
    actual_counts = instances.groupby("group").size().astype(int).sort_index()
    if not expected_counts.equals(actual_counts):
        raise RuntimeError("个体分配人数与分组汇总表不一致。")
    if instances.groupby("bmi").group.nunique().max() != 1:
        raise RuntimeError("相同BMI被拆分到不同组。")
    if np.any(np.diff(groups.best_week.to_numpy(float)) < -1e-10):
        raise RuntimeError("BMI升高时推荐孕周出现逆序。")
    operational_means = (
        instances.groupby("group").recommended_expected_risk.mean()
        .reindex(groups.group).to_numpy(float)
    )
    if not np.allclose(operational_means, groups.rounded_avg_risk, atol=1e-8):
        raise RuntimeError("整天推荐时点的个体风险与分组汇总不一致。")


def run(
    prob_fn: ProbabilityFn,
    source: str,
    mock: bool = False,
    grid_step: float = 0.05,
) -> dict:
    people = load_instances()
    validate_probability_interface(prob_fn, people)
    params = RiskParams()
    times, risks = build_risk_matrix(people, prob_fn, params, grid_step)
    summary, exact_groups, criteria, k_star = exact_scan(people, times, risks)
    exact = exact_groups[exact_groups.k == k_star].copy()
    exact_avg = float(summary.loc[summary.k == k_star, "avg_risk"].iloc[0])
    groups, cuts, total_avg = optimize_integer_boundaries(
        people, times, risks, k_star, exact_avg
    )
    instances = assign_instances(people, groups, prob_fn, params)
    comparison = compare_with_q2(people, times, risks, groups)
    validate_outputs(people, summary, groups, instances)

    prefix = "q3_mock" if mock else "q3"
    summary.to_csv(OUT / f"{prefix}_k_curve.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(OUT / f"{prefix}_groups.csv", index=False, encoding="utf-8-sig")
    instances.to_csv(OUT / f"{prefix}_instances.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / f"{prefix}_vs_q2.csv", index=False, encoding="utf-8-sig")
    exact.to_csv(OUT / f"{prefix}_exact_groups.csv", index=False, encoding="utf-8-sig")

    print(f"概率接口：{source}")
    print(f"孕妇数：{len(people)}；判据选择 k={k_star}；整数边界={cuts}")
    print(groups.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"整数化平均风险={total_avg:.6f}；相对精确解增量={total_avg-exact_avg:.6g}")
    print(f"已输出 outputs/{prefix}_groups.csv 等5个文件。")
    return {
        "people": people,
        "summary": summary,
        "groups": groups,
        "instances": instances,
        "comparison": comparison,
        "k": k_star,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="一号概率接口模块名，默认自动查找q3_model/q3_survival")
    parser.add_argument("--mock", action="store_true", help="仅联调，输出q3_mock_*")
    parser.add_argument("--grid-step", type=float, default=0.05)
    args = parser.parse_args()
    if args.mock:
        prob_fn, source = mock_prob_qualified, "mock（仅联调）"
    else:
        prob_fn, source = load_probability_interface(args.module)
    run(prob_fn, source, mock=args.mock, grid_step=args.grid_step)


if __name__ == "__main__":
    main()
