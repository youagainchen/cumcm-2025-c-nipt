# -*- coding: utf-8 -*-
"""
q4_plot.py —— 二号：问题四建模结果图（不写论文正文）

依赖 q4_validate.py 与 q4_signal_audit.py 的正式输出，只作图不重算指标，
保证图与 CSV 数字同源。

图清单（对应分工文档二号任务7）
  q4_01_signal_audit      信号来源审计：三层 AUC + Z 值与标签不一致的证据
  q4_02_discrimination    ROC 与 PR 曲线（OOF，含重复带）
  q4_03_calibration       概率校准曲线 + 预测分布
  q4_04_confusion_errors  混淆矩阵 + 分层灵敏度
  q4_05_robustness        模型对比（带噪声判定）+ 稳健性 + Bootstrap 区间

运行
  python src/q4_validate.py && python src/q4_signal_audit.py && python src/q4_plot.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures" / "q4_v1"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "logit_z_quality": "Logistic Z+质量",
    "logit_all": "Logistic 全特征",
    "logit_z": "Logistic 仅Z值",
    "forest_all_depth3": "随机森林(深度3)",
    "rule_absZ": "规则 max|Z|",
    "rule_signedZ": "规则 maxZ",
}
PRIMARY = "logit_z_quality"


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


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved figures/q4_v1/{stem}.(png|pdf)")


def check_inputs() -> None:
    required = ["q4_oof_repeated.csv", "q4_model_comparison.csv", "q4_bootstrap_ci.csv",
                "q4_errors.csv", "q4_error_strata.csv", "q4_sensitivity.csv",
                "q4_signal_audit.csv", "q4_zscore_check.csv"]
    missing = [name for name in required if not (OUT / name).exists()]
    if missing:
        raise SystemExit(f"缺少问题四结果：{missing}\n"
                         f"请先运行 python src/q4_validate.py 与 python src/q4_signal_audit.py")


def label_of(model: str) -> str:
    return MODEL_LABELS.get(model, model)


# ---------------------------------------------------------------- 图1

def plot_signal_audit() -> None:
    audit = pd.read_csv(OUT / "q4_signal_audit.csv", encoding="utf-8-sig")
    check = pd.read_csv(OUT / "q4_zscore_check.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # A. 三层 AUC：把 AUC 换算成"偏离随机的幅度"，方向统一
    top = audit.reindex(audit.event_auc.sub(0.5).abs().sort_values(ascending=False).index).head(8)
    y = np.arange(len(top))[::-1]
    for offset, column, color, name in [
        (-0.26, "event_auc", "#4c78a8", "事件级"),
        (0.0, "mother_auc", "#72a0c1", "孕妇层面"),
        (0.26, "within_mother_auc", "#b2182b", "孕妇内（最强证据）"),
    ]:
        axes[0].barh(y + offset, top[column] - 0.5, height=0.24, left=0.5,
                     color=color, label=name)
    axes[0].axvline(0.5, color="#333333", lw=1.2)
    axes[0].set_yticks(y, top.feature)
    axes[0].set(xlabel="AUC（0.5 为随机）", title="A. 信号来源：三层 AUC 对照")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")

    # B. Z 值与标签一致性：阳性/阴性组超过判定线的比例
    width = 0.36
    x = np.arange(len(check))
    axes[1].bar(x - width / 2, check.positive_share_over_alert * 100, width,
                color="#b2182b", label="该亚型阳性组")
    axes[1].bar(x + width / 2, check.negative_share_over_alert * 100, width,
                color="#9aa7b1", label="阴性组")
    for index, row in check.iterrows():
        axes[1].text(index - width / 2, row.positive_share_over_alert * 100 + 0.15,
                     f"{row.positive_share_over_alert*100:.1f}%", ha="center", fontsize=9)
    axes[1].set_xticks(x, [f"{row.subtype}\n(n={int(row.n_positive)})"
                           for _, row in check.iterrows()])
    axes[1].set(ylabel="Z 值超过判定线 3 的比例 (%)",
                title="B. AB 标签与 Z 值不对应")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("问题四信号来源审计：Z 值不构成 AB 判定依据，信号集中于 X 染色体浓度",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, -0.02,
             "孕妇内对照只用标签在不同次检测间变化的 43 位孕妇，可差分掉个体与批次混杂；"
             "T18 阳性组无一例 z18>3，而阴性组占 5.9%。",
             ha="center", fontsize=9)
    save(fig, "q4_01_signal_audit")


# ---------------------------------------------------------------- 图2

def plot_discrimination() -> None:
    oof = pd.read_csv(OUT / "q4_oof_repeated.csv", encoding="utf-8-sig")
    comparison = pd.read_csv(OUT / "q4_model_comparison.csv", encoding="utf-8-sig")
    order = comparison.sort_values("pr_auc_mean", ascending=False).model.tolist()
    palette = dict(zip(order, sns.color_palette("tab10", len(order))))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    prevalence = oof.groupby("model").label.mean().mean()
    for model in order:
        block = oof[oof.model == model]
        # 每个重复画一条淡线，显示划分带来的波动；均值曲线加粗
        for seed, part in block.groupby("seed"):
            fpr, tpr, _ = roc_curve(part.label, part.score)
            axes[0].plot(fpr, tpr, color=palette[model], alpha=0.18, lw=0.9)
            precision, recall, _ = precision_recall_curve(part.label, part.score)
            axes[1].plot(recall, precision, color=palette[model], alpha=0.18, lw=0.9)
        fpr, tpr, _ = roc_curve(block.label, block.score)
        row = comparison[comparison.model == model].iloc[0]
        axes[0].plot(fpr, tpr, color=palette[model], lw=2.2,
                     label=f"{label_of(model)}  AUC={row.roc_auc_mean:.3f}")
        precision, recall, _ = precision_recall_curve(block.label, block.score)
        axes[1].plot(recall, precision, color=palette[model], lw=2.2,
                     label=f"{label_of(model)}  AP={row.pr_auc_mean:.3f}±{row.pr_auc_sd:.3f}")

    axes[0].plot([0, 1], [0, 1], ls="--", color="#666666", lw=1)
    axes[0].set(xlabel="假阳性率", ylabel="灵敏度", title="A. ROC 曲线（细线为各次重复）")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].axhline(prevalence, ls="--", color="#666666", lw=1)
    axes[1].text(0.02, prevalence + 0.01, f"随机基线 {prevalence:.3f}", fontsize=8)
    axes[1].set(xlabel="召回率（灵敏度）", ylabel="精确率（PPV）",
                title="B. PR 曲线：类别不平衡下更能反映实际表现")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("问题四判别能力：5 次重复 × 5 折按孕妇分组交叉验证",
                 fontsize=14, fontweight="bold")
    save(fig, "q4_02_discrimination")


# ---------------------------------------------------------------- 图3

def plot_calibration() -> None:
    oof = pd.read_csv(OUT / "q4_oof_repeated.csv", encoding="utf-8-sig")
    models = [m for m in ("logit_z_quality", "logit_all", "forest_all_depth3")
              if m in set(oof.model)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for model, color in zip(models, sns.color_palette("tab10", len(models))):
        block = oof[(oof.model == model) & (oof.seed == oof.seed.min())]
        bins = pd.qcut(block.score, 8, duplicates="drop")
        grouped = block.groupby(bins, observed=True).agg(
            predicted=("score", "mean"), observed=("label", "mean"), n=("label", "size"))
        # Wilson 区间，避免小样本分箱给出 0 宽度误差棒
        z, n, p = 1.96, grouped.n.to_numpy(float), grouped.observed.to_numpy(float)
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        axes[0].errorbar(grouped.predicted, grouped.observed,
                         yerr=[np.clip(p - (centre - half), 0, None),
                               np.clip((centre + half) - p, 0, None)],
                         marker="o", lw=1.6, capsize=3, color=color,
                         label=label_of(model))
    axes[0].plot([0, 1], [0, 1], ls="--", color="#666666", lw=1)
    axes[0].set(xlabel="预测概率（分箱均值）", ylabel="实际阳性比例",
                title="A. 概率校准（误差棒为 Wilson 95% 区间）")
    axes[0].legend(frameon=False, fontsize=9)

    block = oof[(oof.model == PRIMARY) & (oof.seed == oof.seed.min())]
    for label, color, name in [(0, "#9aa7b1", "AB 正常"), (1, "#b2182b", "AB 异常")]:
        sns.kdeplot(block[block.label == label].score, ax=axes[1], fill=True,
                    alpha=0.45, color=color, label=name, cut=0)
    threshold = float(block.threshold.median())
    axes[1].axvline(threshold, color="#333333", ls="--", lw=1.4,
                    label=f"阈值中位数 {threshold:.3f}")
    axes[1].set(xlabel="预测概率", ylabel="密度",
                title=f"B. {label_of(PRIMARY)} 的预测分布与阈值")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("问题四概率校准与阈值位置", fontsize=14, fontweight="bold")
    fig.text(0.5, -0.02,
             "阈值按“训练折内灵敏度≥90% 前提下特异度最高”选取，验证折不参与选择。",
             ha="center", fontsize=9)
    save(fig, "q4_03_calibration")


# ---------------------------------------------------------------- 图4

def plot_confusion_errors() -> None:
    errors = pd.read_csv(OUT / "q4_errors.csv", encoding="utf-8-sig")
    strata = pd.read_csv(OUT / "q4_error_strata.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    overall = strata[strata.stratum == "全体"].iloc[0]
    matrix = np.array([[overall.tn, overall.fp], [overall.fn, overall.tp]], float)
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["预测正常", "预测异常"],
                yticklabels=["实际正常", "实际异常"], annot_kws={"size": 14})
    axes[0].set_title(f"A. 混淆矩阵（{label_of(PRIMARY)}，多数表决）")
    axes[0].text(1.0, 2.32,
                 f"灵敏度 {overall.sensitivity:.3f}   特异度 {overall.specificity:.3f}   "
                 f"PPV {overall.ppv:.3f}",
                 ha="center", fontsize=10)

    subset = strata[strata.stratum != "全体"].copy()
    subset = subset[subset.sensitivity.notna()]
    y = np.arange(len(subset))[::-1]
    colors = ["#b2182b" if s.startswith("亚型") else "#4c78a8" for s in subset.stratum]
    axes[1].barh(y, subset.sensitivity, color=colors, height=0.62)
    axes[1].axvline(overall.sensitivity, color="#333333", ls="--", lw=1.2,
                    label=f"整体 {overall.sensitivity:.3f}")
    for position, (_, row) in zip(y, subset.iterrows()):
        axes[1].text(row.sensitivity + 0.012, position,
                     f"{row.sensitivity:.3f} (n={int(row.n)})", va="center", fontsize=8)
    axes[1].set_yticks(y, subset.stratum, fontsize=8)
    axes[1].set(xlim=(0, 1.22), xlabel="灵敏度", title="B. 分层灵敏度（红色为亚型）")
    # 图例放在上方图外，避免与最下方一行的数值标注重叠
    axes[1].legend(frameon=False, fontsize=9, loc="lower left",
                   bbox_to_anchor=(0.0, 1.01), ncol=1)

    fig.suptitle("问题四错误结构：高灵敏度以大量假阳为代价，T21 漏诊最严重",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, -0.02,
             f"假阳 {int(overall.fp)} 例对真阳 {int(overall.tp)} 例，PPV 仅 "
             f"{overall.ppv:.1%}；T21 仅 13 例阳性，灵敏度显著低于 T13 与 T18。",
             ha="center", fontsize=9)
    save(fig, "q4_04_confusion_errors")


# ---------------------------------------------------------------- 图5

def plot_robustness() -> None:
    comparison = pd.read_csv(OUT / "q4_model_comparison.csv", encoding="utf-8-sig")
    boot = pd.read_csv(OUT / "q4_bootstrap_ci.csv", encoding="utf-8-sig")
    sensitivity = pd.read_csv(OUT / "q4_sensitivity.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # A. 模型对比：均值±标准差，并标出与最优不可区分者
    table = comparison.sort_values("pr_auc_mean")
    y = np.arange(len(table))
    colors = ["#b2182b" if not flag else "#9aa7b1"
              for flag in table.distinguishable_from_best]
    axes[0].barh(y, table.pr_auc_mean, xerr=table.pr_auc_sd, height=0.6,
                 color=colors, capsize=3)
    axes[0].set_yticks(y, [label_of(m) for m in table.model], fontsize=9)
    axes[0].set(xlabel="PR-AUC（跨重复均值±标准差）",
                title="A. 模型对比：红色与最优不可区分")
    best = table.iloc[-1]
    axes[0].axvline(best.pr_auc_mean, color="#333333", ls="--", lw=1)

    # B. Bootstrap 区间
    pr = boot[boot.metric == "pr_auc"].sort_values("median")
    y = np.arange(len(pr))
    axes[1].errorbar(pr["median"], y,
                     xerr=[pr["median"] - pr.ci_low, pr.ci_high - pr["median"]],
                     fmt="o", color="#4c78a8", capsize=4, lw=1.6)
    axes[1].set_yticks(y, [label_of(m) for m in pr.model], fontsize=9)
    axes[1].set(xlabel="PR-AUC", title="B. 以孕妇为簇的 Bootstrap 95% 区间")

    # C. 稳健性
    focus = sensitivity[sensitivity.scenario.isin(
        ["baseline", "exclude_qc_flagged", "row_level_wrong_unit", "first_draw_only"])]
    focus = focus[focus.model == PRIMARY].copy()
    extra = sensitivity[sensitivity.scenario.isin(
        ["drop_bmi", "drop_person_features", "x_conc_only"])]
    focus = pd.concat([focus, extra], ignore_index=True).sort_values("pr_auc_mean")
    y = np.arange(len(focus))
    axes[2].barh(y, focus.pr_auc_mean, xerr=focus.pr_auc_sd, height=0.6,
                 color="#72a0c1", capsize=3)
    axes[2].set_yticks(y, focus.scenario, fontsize=8)
    axes[2].set(xlabel="PR-AUC", title="C. 口径与特征稳健性")
    baseline = focus[focus.scenario == "baseline"]
    if len(baseline):
        axes[2].axvline(float(baseline.pr_auc_mean.iloc[0]), color="#b2182b",
                        ls="--", lw=1.2, label="baseline")
        axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("问题四稳健性：特征集之间不可区分，性能主要由单一变量支撑",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, -0.02,
             "去掉 BMI 或全部个体变量后性能几乎不变；仅用 x_conc 一个变量即可达到约八成性能；"
             "每位孕妇只保留首次抽血时性能明显下降。",
             ha="center", fontsize=9)
    save(fig, "q4_05_robustness")


def main() -> None:
    setup()
    check_inputs()
    plot_signal_audit()
    plot_discrimination()
    plot_calibration()
    plot_confusion_errors()
    plot_robustness()
    print("完成：5 张 PNG + 5 张 PDF")


if __name__ == "__main__":
    main()
