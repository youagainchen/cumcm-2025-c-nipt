# -*- coding: utf-8 -*-
"""问题二最终模型论文图表；只呈现正式等权集成风险模型。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns

from q2_risk import ExpectedRisk, risk_curve
from q2_survival import load_prob_fn


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
EVENT = ROOT / "data" / "processed" / "male_clean_event.csv"
FIG = ROOT / "figures" / "q2_v1"
FIG.mkdir(parents=True, exist_ok=True)


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
    plt.rcParams.update({"font.family": name, "axes.unicode_minus": False,
                         "savefig.dpi": 300, "axes.titleweight": "bold"})


def save(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved figures/q2_v1/{stem}.(png|pdf)")


def check_inputs() -> None:
    required = ["q2_survival.npy", "q2_final_k_curve.csv", "q2_final_groups.csv",
                "q2_sensitivity.csv", "q2_uncertainty_summary.csv"]
    missing = [name for name in required if not (OUT / name).exists()]
    if missing:
        raise SystemExit(f"缺少问题二结果：{missing}")


def final_risk_curve(weeks: np.ndarray, bmi: float, instant: ExpectedRisk,
                     censored: ExpectedRisk) -> np.ndarray:
    """正式目标：两个风险分量各占50%，只返回集成后的期望风险。"""
    return np.array([
        0.5 * instant.expected_risk(float(t), bmi)
        + 0.5 * censored.expected_risk(float(t), bmi)
        for t in weeks
    ])


def plot_risk_mechanism() -> None:
    instant = ExpectedRisk()
    censored = ExpectedRisk(prob_fn=load_prob_fn())
    week = np.linspace(11, 29, 361)
    decision = np.linspace(11, 25, 281)
    bmis = [25, 30, 35, 40]
    colors = sns.color_palette("viridis", 4)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axes[0].plot(week, risk_curve(week), color="#b2182b", lw=2.5)
    axes[0].axvline(12, color="#777777", ls="--")
    axes[0].axvline(27, color="#777777", ls="--")
    axes[0].set(xlabel="孕周（周）", ylabel="延误风险 R(t)",
                title="A. 连续分段延误风险")

    for bmi, color in zip(bmis, colors):
        risks = final_risk_curve(decision, bmi, instant, censored)
        axes[1].plot(decision, risks, color=color, lw=2, label=f"BMI={bmi}")
        idx = int(np.argmin(risks))
        axes[1].scatter(decision[idx], risks[idx], color=color, s=38, zorder=3)
    axes[1].set(xlabel="首次检测孕周", ylabel="最终集成期望风险",
                title="B. BMI与最佳检测时点")
    axes[1].legend(frameon=False)

    fig.suptitle("最终风险模型：早测失败与晚测延误的权衡",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01,
             "正式目标为等权集成期望风险；重测间隔3.7周，失败后失联概率9.4%；圆点为风险最小时点。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, "q2_01_risk_mechanism")


def plot_optimization() -> None:
    summary = pd.read_csv(OUT / "q2_final_k_curve.csv", encoding="utf-8-sig")
    final_groups = pd.read_csv(OUT / "q2_final_groups.csv", encoding="utf-8-sig")
    data = pd.read_csv(EVENT, encoding="utf-8-sig").sort_values(["mother_id", "week_mean"])
    bmi = data.groupby("mother_id").bmi.first()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    axes[0].plot(summary.k, summary.relative_reduction * 100, marker="o",
                 lw=2.5, color="#1f4e79")
    k_final = len(final_groups)
    axes[0].axvline(k_final, color="#b2182b", ls="--", label=f"采用 k={k_final}")
    axes[0].set(xlabel="组数 k", ylabel="相对不分组风险降幅（%）",
                title="A. 最终模型的组数收益")
    axes[0].legend(frameon=False)

    bars = axes[1].bar(final_groups.group.astype(str), final_groups.best_week,
                       color=sns.color_palette("Blues", 5)[1:])
    for bar, row in zip(bars, final_groups.itertuples()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, row.best_week + 0.15,
                     f"{row.best_week:.2f}周\nn={row.n}", ha="center", fontsize=8)
    axes[1].set(xlabel="BMI组", ylabel="推荐检测孕周",
                title="B. 各组最佳检测时点", ylim=(0, 20))

    bins = np.linspace(bmi.min(), bmi.max(), 22)
    colors = ["#4c78a8", "#72a0c1", "#e69f00", "#b2182b"]
    for row, color in zip(final_groups.itertuples(), colors):
        if row.group < len(final_groups):
            subset = bmi[(bmi >= row.lower) & (bmi < row.upper)]
        else:
            subset = bmi[(bmi >= row.lower) & (bmi <= row.upper)]
        axes[2].hist(subset, bins=bins, color=color, alpha=0.8,
                     label=f"组{row.group} n={row.n}")
    for cut in final_groups.upper.to_numpy()[:-1]:
        axes[2].axvline(cut, color="#333333", ls="--", lw=1)
    axes[2].set(xlabel="基线BMI", ylabel="孕妇数", title="C. 最终可执行分组")
    axes[2].legend(frameon=False)

    fig.suptitle("最终BMI分组：组数、边界与检测时点",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01,
             f"最终采用{len(final_groups)}组，整数BMI边界为"
             f"{'、'.join(f'{c:g}' for c in final_groups.upper.to_numpy()[:-1])}；"
             "组数由收益捕获率、组间时点可辨性与组样本量三条判据选定，"
             "各组时点由集成期望风险最小化得到。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, "q2_02_group_optimization")


def plot_uncertainty() -> None:
    sensitivity = pd.read_csv(OUT / "q2_sensitivity.csv", encoding="utf-8-sig")
    uncertainty = pd.read_csv(OUT / "q2_uncertainty_summary.csv", encoding="utf-8-sig")
    k_final = int(sensitivity.group.max())
    high = sensitivity[sensitivity.group == k_final].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    y = np.arange(len(high))[::-1]
    axes[0].scatter(high.best_week, y, color="#4c78a8", s=50)
    ref_week = float(high.loc[high.scenario == "正式设定", "best_week"].iloc[0])
    axes[0].axvline(ref_week, color="#b2182b", ls="--",
                    label=f"正式设定{ref_week:.1f}周")
    axes[0].set_yticks(y, high.scenario)
    axes[0].set(xlabel=f"最高BMI组(第{k_final}组)最佳孕周",
                title="A. 风险参数与检测误差敏感性")
    axes[0].legend(frameon=False, fontsize=8)

    selected = uncertainty[uncertainty.quantity.isin(["t80", "t90"])].copy()
    for metric, marker, color, offset in [
        ("t80", "o", "#4c78a8", -0.18),
        ("t90", "s", "#d55e00", 0.18),
    ]:
        group = selected[selected.quantity == metric]
        axes[1].errorbar(
            group.bmi + offset, group["median"],
            yerr=[group["median"] - group.q025, group.q975 - group["median"]],
            fmt=marker, color=color, capsize=4, lw=2, label=metric.upper()
        )
    axes[1].axhspan(25, 29, color="#777777", alpha=0.12, label="25周后稀疏区")
    axes[1].set(xlabel="BMI", ylabel="达标孕周（中位数及95%区间）",
                title="B. 最终模型参数不确定性传播")
    axes[1].legend(frameon=False)

    fig.suptitle("最终方案的敏感性与不确定性",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "q2_03_uncertainty")


def main() -> None:
    setup()
    check_inputs()
    for pattern in ("*.png", "*.pdf"):
        for old in FIG.glob(pattern):
            old.unlink()
    plot_risk_mechanism()
    plot_optimization()
    plot_uncertainty()
    print(f"完成：{len(list(FIG.glob('*.png')))}张PNG，"
          f"{len(list(FIG.glob('*.pdf')))}张PDF")


if __name__ == "__main__":
    main()
