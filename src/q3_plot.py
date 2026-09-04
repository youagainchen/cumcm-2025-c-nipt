# -*- coding: utf-8 -*-
"""问题三最终论文图表：仅呈现正式多因素AFT与分组优化结果。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns

from q2_risk import RiskParams
from q3_model import load_prob_fn
from q3_optimize import expected_risk, load_instances


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures" / "q3_v1"
FIG.mkdir(parents=True, exist_ok=True)

FEATURE_LABELS = {
    "bmi_age": "BMI+年龄",
    "height_weight_age": "身高+体重+年龄",
    "bmi_hw_resid_age": "BMI残差+年龄",
    "all_factors": "全变量（共线）",
}


def setup() -> None:
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        name = "DejaVu Sans"
    sns.set_theme(style="whitegrid", font=name)
    plt.rcParams.update({
        "font.family": name,
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved figures/q3_v1/{stem}.(png|pdf)")


def check_inputs() -> None:
    required = [
        "q3_model.npy", "q3_aft_candidates.csv", "q3_aft_grouped_cv.csv",
        "q3_aft_calibration.csv", "q3_groups.csv", "q3_instances.csv",
        "q3_k_curve.csv", "q3_sensitivity.csv", "q3_boundary_stability.csv",
        "q3_vs_q2.csv",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    if missing:
        raise SystemExit(f"缺少问题三正式结果：{missing}")


def plot_model_validation() -> None:
    aft = pd.read_csv(OUT / "q3_aft_candidates.csv", encoding="utf-8-sig")
    cv = pd.read_csv(OUT / "q3_aft_grouped_cv.csv", encoding="utf-8-sig")
    calibration = pd.read_csv(OUT / "q3_aft_calibration.csv", encoding="utf-8-sig")
    aft["label"] = aft.feature_model.map(FEATURE_LABELS) + " · " + aft.distribution
    aft = aft.sort_values("delta_aic", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    colors = [
        "#bdbdbd" if "BMI残差" in label else
        "#b2182b" if label == "身高+体重+年龄 · LogNormal" else "#4c78a8"
        for label in aft.label
    ]
    axes[0].barh(aft.label, aft.delta_aic, color=colors)
    axes[0].axvline(2, color="#d55e00", ls="--", lw=1.5, label="ΔAIC=2")
    axes[0].set(xlabel="相对最小AIC的差值", title="A. 候选AFT模型比较")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].text(
        0.98, 0.98, "BMI残差仅保留0.42%独立方差\n作为共线性诊断，不用于正式模型",
        transform=axes[0].transAxes, ha="right", va="top", fontsize=8,
        color="#555555",
    )

    valid = cv[cv.valid.astype(str).str.lower().eq("true")].copy()
    cv_summary = valid.groupby("distribution", as_index=False).agg(
        mean_log_likelihood=("mean_log_likelihood", "mean"),
        mean_auc=("first_detection_auc", "mean"),
        mean_brier=("first_detection_brier", "mean"),
    ).sort_values("mean_log_likelihood", ascending=False)
    x = np.arange(len(cv_summary))
    axes[1].scatter(
        x, cv_summary.mean_log_likelihood, s=85, zorder=3,
        color=["#b2182b" if item == "LogNormal" else "#72a0c1"
               for item in cv_summary.distribution],
    )
    axes[1].plot(x, cv_summary.mean_log_likelihood, color="#bbbbbb", lw=1)
    for xpos, row in zip(x, cv_summary.itertuples()):
        axes[1].text(
            xpos, row.mean_log_likelihood + 0.0012,
            f"AUC={row.mean_auc:.2f}\nBrier={row.mean_brier:.2f}",
            ha="center", va="bottom", fontsize=8,
        )
    axes[1].set_xticks(x, cv_summary.distribution)
    axes[1].set(
        xlabel="分布", ylabel="五折平均区间对数似然（越高越好）",
        title="B. 按孕妇五折验证",
    )
    axes[1].set_ylim(cv_summary.mean_log_likelihood.min() - 0.004,
                     cv_summary.mean_log_likelihood.max() + 0.010)

    axes[2].plot([0, 1], [0, 1], color="#555555", ls="--", label="理想校准")
    for distribution, group in calibration.groupby("distribution"):
        group = group.copy()
        group["aggregate_bin"] = pd.qcut(
            group.mean_predicted, q=5, duplicates="drop"
        )
        rows = []
        for _, bin_group in group.groupby("aggregate_bin", observed=True):
            rows.append({
                "predicted": np.average(bin_group.mean_predicted, weights=bin_group.n),
                "observed": np.average(bin_group.mean_observed, weights=bin_group.n),
            })
        grouped = pd.DataFrame(rows).sort_values("predicted")
        axes[2].plot(grouped.predicted, grouped.observed, marker="o", lw=1.8,
                     label=distribution)
    axes[2].set(
        xlim=(0.45, 1.0), ylim=(0.45, 1.0),
        xlabel="预测首次达标概率", ylabel="实际首次达标率",
        title="C. 五折分箱校准",
    )
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("问题三正式概率模型：候选比较与样本外验证", fontsize=15,
                 fontweight="bold")
    save(fig, "q3_01_model_validation")


def representative_people(people: pd.DataFrame) -> pd.DataFrame:
    targets = people.bmi.quantile([0.10, 0.45, 0.75, 0.95]).to_numpy()
    indices = [(people.bmi - target).abs().idxmin() for target in targets]
    return people.loc[indices].drop_duplicates("mother_id").sort_values("bmi")


def plot_probability_and_risk() -> None:
    people = representative_people(load_instances())
    prob_fn = load_prob_fn()
    params = RiskParams()
    weeks = np.linspace(11, 25, 281)
    colors = sns.color_palette("viridis", len(people))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for row, color in zip(people.itertuples(), colors):
        label = f"BMI {row.bmi:.1f}（{row.height:.0f}cm/{row.weight:.0f}kg/{row.age:.0f}岁）"
        probability = np.asarray([
            prob_fn(t, row.bmi, row.height, row.weight, row.age) for t in weeks
        ], float)
        risks = np.asarray([expected_risk(t, row, prob_fn, params) for t in weeks])
        axes[0].plot(weeks, probability, color=color, lw=2, label=label)
        idx = int(np.argmin(risks))
        axes[1].plot(weeks, risks, color=color, lw=2, label=label)
        axes[1].scatter(weeks[idx], risks[idx], color=color, s=35, zorder=3)
    axes[0].axhline(0.8, color="#777777", ls="--", lw=1)
    axes[0].set(
        xlabel="孕周（周）", ylabel="累计达到4%的概率",
        title="A. 代表性个体达标概率", ylim=(0, 1.02),
    )
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set(
        xlabel="首次检测孕周", ylabel="期望风险",
        title="B. 早测失败与晚测延误的权衡",
    )
    fig.suptitle("多因素达标概率进入复检递推后的个体决策", fontsize=15,
                 fontweight="bold")
    fig.text(0.5, 0.01, "圆点为各代表性个体的风险最小时点；同一孕妇复检时仅推进孕周。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, "q3_02_probability_risk")


def plot_group_optimization() -> None:
    curve = pd.read_csv(OUT / "q3_k_curve.csv", encoding="utf-8-sig")
    groups = pd.read_csv(OUT / "q3_groups.csv", encoding="utf-8-sig")
    people = load_instances()
    k_star = len(groups)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    axes[0, 0].plot(curve.k, curve.relative_reduction * 100, marker="o",
                    color="#1f4e79", lw=2.4)
    axes[0, 0].axvline(k_star, color="#b2182b", ls="--", label=f"采用k={k_star}")
    axes[0, 0].set(xlabel="组数 k", ylabel="相对不分组风险降幅（%）",
                   title="A. 分组收益")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(curve.k, curve.captured_frac * 100, marker="o",
                    color="#4c78a8", label="收益捕获率")
    axes[0, 1].axhline(90, color="#4c78a8", ls="--", lw=1, label="C1：90%")
    gap_axis = axes[0, 1].twinx()
    finite_gap = curve.min_week_gap.replace([np.inf, -np.inf], np.nan)
    gap_axis.plot(curve.k, finite_gap, marker="s", color="#d55e00",
                  label="最小时点间隔")
    gap_axis.axhline(1, color="#d55e00", ls="--", lw=1, label="C2：1周")
    axes[0, 1].set(xlabel="组数 k", ylabel="收益捕获率（%）",
                   title="B. 显式组数判据", ylim=(0, 105))
    gap_axis.set_ylabel("相邻组最小时点间隔（周）")
    lines = axes[0, 1].lines + gap_axis.lines
    axes[0, 1].legend(lines, [line.get_label() for line in lines],
                      frameon=False, fontsize=8, loc="center right")

    bars = axes[1, 0].bar(groups.group.astype(str), groups.recommended_week,
                          color=sns.color_palette("Blues", len(groups) + 2)[1:-1])
    for bar, row in zip(bars, groups.itertuples()):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2, row.recommended_week + 0.18,
            f"{row.recommended_time}\nn={row.n}", ha="center", fontsize=9,
        )
    axes[1, 0].set(xlabel="BMI组", ylabel="可执行检测孕周",
                   title="C. 各组推荐时点", ylim=(0, 22))

    bins = np.linspace(people.bmi.min(), people.bmi.max(), 24)
    palette = sns.color_palette("crest", len(groups))
    for row, color in zip(groups.itertuples(), palette):
        if row.group < len(groups):
            subset = people[(people.bmi >= row.lower) & (people.bmi < row.upper)]
        else:
            subset = people[(people.bmi >= row.lower) & (people.bmi <= row.upper)]
        axes[1, 1].hist(subset.bmi, bins=bins, color=color, alpha=0.85,
                        label=f"{row.interval}，n={row.n}")
    for cut in groups.upper.iloc[:-1]:
        axes[1, 1].axvline(cut, color="#333333", ls="--", lw=1.2)
    axes[1, 1].set(xlabel="基线BMI", ylabel="孕妇数", title="D. 最终连续分组")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle("问题三最终分组：组数、整数边界与检测时点", fontsize=15,
                 fontweight="bold")
    save(fig, "q3_03_group_optimization")


def plot_sensitivity() -> None:
    sensitivity = pd.read_csv(OUT / "q3_sensitivity.csv", encoding="utf-8-sig")
    stability = pd.read_csv(OUT / "q3_boundary_stability.csv", encoding="utf-8-sig")
    risk = sensitivity[sensitivity.scenario_type.eq("risk_parameter")].copy()
    draws = sensitivity[sensitivity.scenario_type.eq("parameter_covariance_draw")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    y = np.arange(len(risk))[::-1]
    ok = risk.status.eq("ok")
    axes[0].scatter(risk.loc[ok, "k"], y[ok], color="#4c78a8", s=65)
    axes[0].scatter(np.full((~ok).sum(), 1.5), y[~ok], marker="x",
                    color="#b2182b", s=70, label="无可行k")
    axes[0].set_yticks(y, risk.scenario)
    axes[0].set(xlabel="判据选定组数 k", title="A. 风险参数敏感性",
                xticks=[1, 2, 3, 4, 5, 6], xlim=(1, 6))
    axes[0].legend(frameon=False, fontsize=8)

    status = pd.Series({
        "4组": ((draws.status == "ok") & (draws.k == 4)).sum(),
        "3组": ((draws.status == "ok") & (draws.k == 3)).sum(),
        "其他组数": ((draws.status == "ok") & (~draws.k.isin([3, 4]))).sum(),
        "无可行k": (draws.status == "no_feasible_k").sum(),
    })
    bars = axes[1].bar(status.index, status.values,
                       color=["#4c78a8", "#72a0c1", "#e69f00", "#b2182b"])
    for bar, value in zip(bars, status.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1,
                     f"{value}%", ha="center")
    axes[1].set(ylabel="100次参数抽样中的次数", title="B. 组数稳定性", ylim=(0, 58))

    top = stability.sort_values("fraction", ascending=False).head(7).copy()
    top["scheme"] = np.where(
        top.status.eq("no_feasible_k"), "无可行k",
        "k=" + top.k.fillna(0).astype(int).astype(str) + "，边界 " + top.integer_cuts.fillna(""),
    )
    top = top.sort_values("fraction")
    axes[2].barh(top.scheme, top.fraction * 100,
                 color=["#b2182b" if x == "无可行k" else "#4c78a8" for x in top.scheme])
    axes[2].set(xlabel="出现频率（%）", title="C. 高频整数边界方案")

    fig.suptitle("问题三方案的敏感性与参数不确定性", fontsize=15,
                 fontweight="bold")
    save(fig, "q3_04_sensitivity_stability")


def plot_q2_comparison() -> None:
    comparison = pd.read_csv(OUT / "q3_vs_q2.csv", encoding="utf-8-sig")
    names = {
        "q2_official_fixed": "问题二原方案",
        "q3_multifactor_operational": "问题三多因素方案",
    }
    comparison["label"] = comparison.model.map(names)
    totals = comparison.groupby("label", as_index=False).total_avg_risk.first()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"问题二原方案": "#999999", "问题三多因素方案": "#1f77b4"}
    for y, (label, group) in enumerate(comparison.groupby("label", sort=False)):
        palette = sns.light_palette(colors[label], n_colors=len(group) + 2)[1:-1]
        for index, (row, color) in enumerate(zip(group.itertuples(), palette)):
            lower, upper = [float(x) for x in row.interval.strip("[]() ").replace(",", " ").split()]
            axes[0].plot([lower, upper], [y, y], lw=10, alpha=0.75,
                         color=color, solid_capstyle="butt")
            axes[0].text((lower + upper) / 2, y + 0.10,
                         f"{row.best_week:.2f}周", ha="center", fontsize=8)
            if index < len(group) - 1:
                axes[0].vlines(upper, y - 0.10, y + 0.10, color="#333333", lw=1.2)
    axes[0].set_yticks(range(len(names)), list(names.values()))
    axes[0].set(xlabel="BMI区间（线段）与推荐孕周（数字）",
                title="A. 分组边界与时点变化", ylim=(-0.35, 1.4))

    q2 = float(totals.loc[totals.label == "问题二原方案", "total_avg_risk"].iloc[0])
    q3 = float(totals.loc[totals.label == "问题三多因素方案", "total_avg_risk"].iloc[0])
    y = np.arange(len(totals))
    axes[1].hlines(y, totals.total_avg_risk.min() - 0.006,
                   totals.total_avg_risk, color="#cccccc", lw=2)
    axes[1].scatter(totals.total_avg_risk, y, s=100,
                    color=[colors[item] for item in totals.label], zorder=3)
    for ypos, value in zip(y, totals.total_avg_risk):
        axes[1].text(value + 0.0015, ypos, f"{value:.4f}", va="center")
    axes[1].set_yticks(y, totals.label)
    axes[1].set(
        xlabel="同一问题三口径下的总体平均风险",
        title=f"B. 公平风险比较（相对下降{(q2-q3)/q2*100:.2f}%）",
        xlim=(totals.total_avg_risk.min() - 0.008,
              totals.total_avg_risk.max() + 0.016),
    )

    fig.suptitle("问题二与问题三方案对照", fontsize=15, fontweight="bold")
    save(fig, "q3_05_q2_comparison")


def main() -> None:
    setup()
    check_inputs()
    for pattern in ("*.png", "*.pdf"):
        for old in FIG.glob(pattern):
            old.unlink()
    plot_model_validation()
    plot_probability_and_risk()
    plot_group_optimization()
    plot_sensitivity()
    plot_q2_comparison()
    print(f"完成：{len(list(FIG.glob('*.png')))}张PNG，"
          f"{len(list(FIG.glob('*.pdf')))}张PDF")


if __name__ == "__main__":
    main()
