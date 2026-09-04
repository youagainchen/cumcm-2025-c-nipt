# -*- coding: utf-8 -*-
"""问题二四张组合图；先依次运行 q2_survival/optimize/sensitivity/final。"""
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


def inputs():
    required = ["q2_survival.npy", "q2_k_curve.csv", "q2_final_k_curve.csv",
                "q2_boundary_profile.csv", "q2_sensitivity.csv",
                "q2_model_weight_sensitivity.csv", "q2_uncertainty_summary.csv"]
    missing = [x for x in required if not (OUT / x).exists()]
    if missing:
        raise SystemExit(f"缺少问题二结果：{missing}")


def plot_survival_evidence() -> None:
    d = pd.read_csv(EVENT, encoding="utf-8-sig").sort_values(["mother_id", "week_mean"])
    m = d.groupby("mother_id", as_index=False).agg(censored=("censored", "first"))
    s = np.load(OUT / "q2_survival.npy", allow_pickle=True).item()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    counts = m.censored.value_counts().reindex(["left", "none", "right"])
    labels = ["左删失\n首次已达标", "区间删失\n未达标→达标", "右删失\n始终未达标"]
    bars = axes[0].bar(labels, counts, color=["#4c78a8", "#e69f00", "#b2182b"])
    for bar, v in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 3, f"{v}\n({v/len(m):.1%})",
                     ha="center", fontsize=9)
    axes[0].set(ylabel="孕妇数", title="A. 首次达标时间的删失结构")

    for b, color in zip([25, 30, 35, 40], sns.color_palette("viridis", 4)):
        bi = np.argmin(np.abs(s["grid_bmi"] - b))
        axes[1].plot(s["grid_week"], s["prob_matrix"][bi], lw=2.2,
                     color=color, label=f"BMI={b}")
    axes[1].set(xlabel="孕周（周）", ylabel="P(T≤t)", ylim=(0, 1.02),
                title="B. LogNormal AFT累计达标概率")
    axes[1].legend(frameon=False)

    er = ExpectedRisk()
    week = np.linspace(11, 25, 281)
    for b, color in zip([30, 35, 40], ["#4c78a8", "#e69f00", "#009e73"]):
        q1 = np.asarray(er.prob_qualified(week, b), float)
        bi = np.argmin(np.abs(s["grid_bmi"] - b))
        axes[2].plot(week, q1, color=color, lw=2, label=f"Q1瞬时 BMI={b}")
        axes[2].plot(s["grid_week"], s["prob_matrix"][bi], color=color,
                     ls="--", lw=1.7, label=f"AFT累计 BMI={b}")
    axes[2].set(xlabel="孕周（周）", ylabel="达标概率", ylim=(0.35, 1.01),
                title="C. 两条概率路径的口径差异")
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("区间删失结构与达标概率模型", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, "AFT按AIC选择LogNormal；首次达标后5.9%的后续检测再次低于4%，两种概率口径不完全等价。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(FIG / "q2_01_survival_evidence.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "q2_01_survival_evidence.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_risk_mechanism() -> None:
    er = ExpectedRisk()
    week = np.linspace(11, 29, 361)
    decision = np.linspace(11, 25, 281)
    bmis = [25, 30, 35, 40]
    colors = sns.color_palette("viridis", 4)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    axes[0].plot(week, risk_curve(week), color="#b2182b", lw=2.5)
    axes[0].axvline(12, color="#777777", ls="--")
    axes[0].axvline(27, color="#777777", ls="--")
    axes[0].set(xlabel="孕周（周）", ylabel="R(t)", title="A. 延误风险函数")

    for b, color in zip(bmis, colors):
        axes[1].plot(decision, er.prob_qualified(decision, b), color=color,
                     lw=2, label=f"BMI={b}")
        risks = np.array([er.expected_risk(t, b) for t in decision])
        axes[2].plot(decision, risks, color=color, lw=2, label=f"BMI={b}")
        i = np.argmin(risks)
        axes[2].scatter(decision[i], risks[i], color=color, s=35, zorder=3)
    axes[1].set(xlabel="首次检测孕周", ylabel="P(Y≥4%)", ylim=(0.35, 1.01),
                title="B. BMI影响单次达标概率")
    axes[1].legend(frameon=False)
    axes[2].set(xlabel="首次检测孕周", ylabel="递归期望风险", title="C. 复检后的期望风险")
    axes[2].legend(frameon=False)
    fig.suptitle("早测失败与晚测延误的风险权衡", fontsize=15, fontweight="bold")
    fig.text(0.5, .01, "重测间隔3.7周；失败后失联概率9.4%；圆点为各BMI的风险最小时点。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(FIG / "q2_02_risk_mechanism.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "q2_02_risk_mechanism.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_optimization() -> None:
    k0 = pd.read_csv(OUT / "q2_k_curve.csv", encoding="utf-8-sig")
    kf = pd.read_csv(OUT / "q2_final_k_curve.csv", encoding="utf-8-sig")
    bp = pd.read_csv(OUT / "q2_boundary_profile.csv", encoding="utf-8-sig")
    d = pd.read_csv(EVENT, encoding="utf-8-sig").sort_values(["mother_id", "week_mean"])
    bmi = d.groupby("mother_id").bmi.first()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for name, g in k0.groupby("path"):
        axes[0].plot(g.k, g.relative_reduction * 100, marker="o", lw=2, label=name)
    ens = kf[kf.survival_weight == .5]
    axes[0].plot(ens.k, ens.relative_reduction * 100, marker="s", lw=2.5,
                 color="#1f4e79", label="等权集成")
    axes[0].axvline(2, color="#b2182b", ls="--", label="拐点 k=2")
    axes[0].set(xlabel="组数k", ylabel="相对不分组风险降幅（%）", title="A. 组数收益递减")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(bp.cut, bp.avg_risk, color="#4c78a8", lw=2)
    near = bp[bp.within_0p1pct]
    axes[1].axvspan(near.cut.min(), near.cut.max(), color="#4c78a8", alpha=.16,
                    label="距最优≤0.1%")
    axes[1].axvline(32, color="#b2182b", ls="--", label="采用BMI=32")
    axes[1].set(xlabel="两组切点BMI", ylabel="等权平均风险", title="B. 边界风险剖面")
    axes[1].legend(frameon=False)

    bins = np.linspace(bmi.min(), bmi.max(), 22)
    axes[2].hist(bmi[bmi < 32], bins=bins, color="#4c78a8", alpha=.8, label="组1 n=152")
    axes[2].hist(bmi[bmi >= 32], bins=bins, color="#e69f00", alpha=.8, label="组2 n=115")
    axes[2].axvline(32, color="#b2182b", ls="--")
    axes[2].set(xlabel="基线BMI", ylabel="孕妇数", title="C. 最终可执行分组")
    axes[2].legend(frameon=False)
    fig.suptitle("组数、边界与最终BMI分组", fontsize=15, fontweight="bold")
    fig.text(.5, .01, "最终方案：[20.70,32)于12.45周检测；[32,46.88]于14.25周检测。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(FIG / "q2_03_group_optimization.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "q2_03_group_optimization.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_uncertainty() -> None:
    sens = pd.read_csv(OUT / "q2_sensitivity.csv", encoding="utf-8-sig")
    weight = pd.read_csv(OUT / "q2_model_weight_sensitivity.csv", encoding="utf-8-sig")
    unc = pd.read_csv(OUT / "q2_uncertainty_summary.csv", encoding="utf-8-sig")
    high = sens[sens.group == 2].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    y = np.arange(len(high))[::-1]
    axes[0].scatter(high.best_week, y, color="#4c78a8", s=50)
    axes[0].axvline(14.6, color="#b2182b", ls="--", label="主设定14.6周")
    axes[0].set_yticks(y, high.scenario)
    axes[0].set(xlabel="高BMI组最佳孕周", title="A. 风险参数与检测误差")
    axes[0].legend(frameon=False, fontsize=8)

    w2 = weight[weight.group == 2]
    axes[1].plot(w2.survival_weight, w2.lower, marker="o", label="最优BMI切点")
    axr = axes[1].twinx()
    axr.plot(w2.survival_weight, w2.best_week, marker="s", color="#e69f00",
             label="高BMI组孕周")
    axes[1].set(xlabel="生存模型权重", ylabel="精确最优BMI切点", title="B. 模型权重敏感性")
    axr.set_ylabel("高BMI组最佳孕周", color="#e69f00")
    lines = axes[1].lines + axr.lines
    axes[1].legend(lines, [x.get_label() for x in lines], frameon=False, fontsize=8)

    u = unc[unc.quantity.isin(["t80", "t90"])].copy()
    for metric, marker, color, off in [("t80", "o", "#4c78a8", -.18),
                                        ("t90", "s", "#d55e00", .18)]:
        g = u[u.quantity == metric]
        axes[2].errorbar(g.bmi + off, g["median"],
                         yerr=[g["median"] - g.q025, g.q975 - g["median"]],
                         fmt=marker, color=color, capsize=4, lw=2, label=metric.upper())
    axes[2].axhspan(25, 29, color="#777777", alpha=.12, label="25周后稀疏区")
    axes[2].set(xlabel="BMI", ylabel="达标孕周（中位数及95%区间）",
                title="C. AFT参数不确定性传播")
    axes[2].legend(frameon=False)
    fig.suptitle("问题二敏感性与不确定性", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(FIG / "q2_04_uncertainty.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "q2_04_uncertainty.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    setup()
    inputs()
    for pattern in ("*.png", "*.pdf"):
        for old in FIG.glob(pattern):
            old.unlink()
    plot_survival_evidence()
    plot_risk_mechanism()
    plot_optimization()
    plot_uncertainty()
    print(f"完成：{len(list(FIG.glob('*.png')))}张PNG，{len(list(FIG.glob('*.pdf')))}张PDF")


if __name__ == "__main__":
    main()
