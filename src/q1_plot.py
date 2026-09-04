# -*- coding: utf-8 -*-
"""问题一 v4 论文图表生成脚本。

数据来源：
  data/processed/male_clean_event.csv  主分析事件级数据
  outputs/q1_out.txt                  完整精度统计结果

输出：figures/q1_v4/*.png 和 *.pdf

运行：python src/q1_plot.py

说明：模型曲线、残差和分组交叉验证由本脚本重新拟合；1000 次 Bootstrap
区间等耗时结果使用 q1_model.py 的正式完整精度输出，避免仅为画图重复长时间计算。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import patsy
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent.parent
EVENT = ROOT / "data" / "processed" / "male_clean_event.csv"
ROWLV = ROOT / "data" / "processed" / "male_clean.csv"
LOG_PATH = ROOT / "outputs" / "q1_out.txt"
FIG_DIR = ROOT / "figures" / "q1_v4"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WEEK_TERM = "cr(week_c, df=3, constraints='center')"
FINAL_F = f"resp ~ {WEEK_TERM} + bmi_between_c + bmi_within"

# 这些是 outputs/q1_out.txt 中完整精度运行的正式汇总数字。最终模型重新拟合后
# 会核对关键系数；如果数据更新造成不一致，脚本会中止，防止画出新旧混合图。
OFFICIAL = {
    "between": -0.00458,
    "within": 0.00279,
    "between_wald": (-0.00677, -0.00239),
    "within_wald": (-0.00084, 0.00642),
    "between_boot": (-0.00675, -0.00219),
    "within_boot": (-0.00001, 0.01003),
    "tech_var": 0.000174,
    "tech_ci": (0.000126, 0.000257),
    "var_mother": 0.002850,
    "var_draw": 0.000431,
    "var_tech": 0.000172,
}

ROBUSTNESS = pd.DataFrame(
    [
        ("最终模型\n事件级", -0.00458),
        ("行级\n三层嵌套", -0.00457),
        ("行级朴素\n未拆技术重复", -0.00454),
        ("剔除QC异常", -0.00459),
        ("剔除时序异常", -0.00456),
    ],
    columns=["setting", "coef"],
)


def setup_style() -> None:
    """选择可用中文字体并设置统一论文风格。"""
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    chosen = None
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            chosen = name
            break
        except ValueError:
            continue
    if chosen is None:
        warnings.warn("未找到中文字体，图中文字可能显示为方框。")
        chosen = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": chosen,
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )
    sns.set_theme(style="whitegrid", font=chosen)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {png.relative_to(ROOT)} and {pdf.relative_to(ROOT)}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (EVENT, ROWLV, LOG_PATH):
        if not path.exists():
            raise SystemExit(f"缺少 {path.relative_to(ROOT)}，请先运行 clean.py 和 q1_model.py")
    d = pd.read_csv(EVENT, encoding="utf-8-sig").rename(
        columns={"week_mean": "week", "y_conc_mean": "y_conc"}
    )
    d = d.dropna(subset=["mother_id", "week", "bmi", "y_conc"]).copy()
    d = d.sort_values(["mother_id", "week", "visit_idx"]).reset_index(drop=True)
    d["resp"] = np.sqrt(d.y_conc)
    d["week_c"] = d.week - d.week.mean()
    d["bmi_baseline"] = d.groupby("mother_id").bmi.transform("first")
    d["bmi_between_c"] = d.bmi_baseline - d.bmi_baseline.mean()
    d["bmi_within"] = d.bmi - d.bmi_baseline
    d["age_c"] = d.age - d.age.mean()
    row = pd.read_csv(ROWLV, encoding="utf-8-sig")
    row = row.dropna(subset=["mother_id", "week", "bmi", "y_conc"]).copy()
    return d, row


def fit_final(d: pd.DataFrame):
    """重拟合最终模型，优先 lbfgs，必要时 Powell。"""
    last = None
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = smf.mixedlm(
                    FINAL_F, d, groups=d.mother_id, re_formula="~week_c"
                ).fit(reml=True, method=method, maxiter=500, disp=False)
            last = result
            if bool(result.converged) and np.all(np.isfinite(result.fe_params)):
                return result
        except Exception:
            continue
    raise RuntimeError(f"最终模型绘图重拟合失败: {last}")


def check_official(final, d: pd.DataFrame) -> None:
    """防止数据更新后继续使用不匹配的正式统计结果。"""
    if len(d) != 1021 or d.mother_id.nunique() != 267:
        raise RuntimeError("数据规模已变化，请先重跑 q1_model.py 并更新绘图汇总常数。")
    got = np.array(
        [final.fe_params["bmi_between_c"], final.fe_params["bmi_within"]], float
    )
    expected = np.array([OFFICIAL["between"], OFFICIAL["within"]])
    if not np.allclose(got, expected, atol=5e-5):
        raise RuntimeError(
            f"重拟合系数 {got} 与正式日志 {expected} 不一致，请先更新正式模型结果。"
        )


def plot_data_correlation(d: pd.DataFrame) -> None:
    """把数据覆盖、描述性相关和聚类区间合并成一张四联图。"""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0, 0]
    for _, g in d.groupby("mother_id"):
        ax.plot(g.week, g.y_conc * 100, color="#7f8c8d", alpha=0.11, lw=0.6)
    ax.scatter(d.week, d.y_conc * 100, s=9, color="#4c78a8", alpha=0.12, edgecolors="none")
    bins = np.arange(11, 30.01, 1)
    d["week_bin_plot"] = pd.cut(d.week, bins=bins, include_lowest=True)
    med = d.groupby("week_bin_plot", observed=True).agg(
        week=("week", "median"), y=("y_conc", "median")
    )
    ax.plot(med.week, med.y * 100, color="#d55e00", marker="o", ms=4, lw=2,
            label="孕周分箱中位数")
    ax.axhline(4, color="#b2182b", ls="--", lw=1.3, label="4%达标线")
    ax.axvspan(25, 29, color="#999999", alpha=0.14, label="25周后稀疏区")
    ax.set(xlim=(11, 29), xlabel="孕周（周）", ylabel="Y染色体浓度（%）",
           title="A. 孕妇纵向检测轨迹")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    edges = np.arange(11, 30.01, 2)
    labels = [f"{int(a)}–{int(b)}" for a, b in zip(edges[:-1], edges[1:])]
    cats = pd.cut(d.week, bins=edges, right=False, include_lowest=True, labels=labels)
    coverage = d.groupby(cats, observed=False).agg(
        事件数=("mother_id", "size"), 孕妇数=("mother_id", "nunique")
    )
    x = np.arange(len(coverage))
    width = 0.38
    ax.bar(x - width / 2, coverage["事件数"], width, color="#4c78a8", label="事件数")
    ax.bar(x + width / 2, coverage["孕妇数"], width, color="#f2a541", label="孕妇数")
    for xx, val in zip(x - width / 2, coverage["事件数"]):
        ax.text(xx, val + 4, str(int(val)), ha="center", va="bottom", fontsize=7)
    for xx, val in zip(x + width / 2, coverage["孕妇数"]):
        ax.text(xx, val + 10, str(int(val)), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, coverage.index, rotation=45, ha="right")
    ax.set(xlabel="孕周区间（周）", ylabel="数量", title="B. 各孕周区间数据覆盖")
    ax.legend(frameon=False)
    cols = ["week", "bmi", "age", "height", "weight", "y_conc"]
    names = ["孕周", "BMI", "年龄", "身高", "体重", "Y浓度"]
    corr = d[cols].corr(method="spearman")
    ax = axes[1, 0]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                xticklabels=names, yticklabels=names, square=True,
                cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title("C. 事件级Spearman相关（描述性）")

    boot_week = cluster_boot_corr(d, "week")
    boot_bmi = cluster_boot_corr(d, "bmi")
    vals = [np.mean(boot_week), np.mean(boot_bmi), 0.5920, -0.2618]
    cis = [np.percentile(boot_week, [2.5, 97.5]),
           np.percentile(boot_bmi, [2.5, 97.5]), None, None]
    labels_corr = ["总体孕周–Y", "总体BMI–Y", "孕妇内孕周–Y", "孕妇间孕周–Y"]
    colors = ["#4c78a8", "#d55e00", "#009e73", "#b2182b"]
    ycorr = np.arange(4)[::-1]
    ax = axes[1, 1]
    ax.axvline(0, color="#444444", ls="--", lw=1)
    for i, (val, ci, color) in enumerate(zip(vals, cis, colors)):
        if ci is None:
            ax.scatter(val, ycorr[i], color=color, s=55, zorder=3)
            ax.text(val + 0.025, ycorr[i], f"r={val:+.3f}", va="center", fontsize=8)
        else:
            ax.errorbar(val, ycorr[i], xerr=[[val-ci[0]], [ci[1]-val]], fmt="o",
                        color=color, capsize=4, lw=2)
            ax.text(ci[1] + 0.025, ycorr[i], f"95%CI [{ci[0]:+.3f},{ci[1]:+.3f}]",
                    va="center", fontsize=8)
    ax.set_yticks(ycorr, labels_corr)
    ax.set(xlim=(-0.36, 0.72), xlabel="相关系数",
           title="D. 聚类Bootstrap及组内/组间相关")
    fig.suptitle("问题一：数据覆盖与相关特性", fontsize=15, fontweight="bold")
    save_figure(fig, "q1_01_data_correlation")


def prediction_grid(final, d: pd.DataFrame):
    week = np.linspace(11, 29, 361)
    grid = pd.DataFrame(
        {"week_c": week - d.week.mean(), "bmi_between_c": 0.0, "bmi_within": 0.0}
    )
    # statsmodels 不同版本保存 patsy 元数据的位置不同。直接从与正式模型完全
    # 相同的公式重建设计信息，避免依赖内部 PandasData 属性。
    _, x_train = patsy.dmatrices(FINAL_F, d, return_type="dataframe")
    design_info = x_train.design_info
    Xg = patsy.build_design_matrices([design_info], grid, return_type="dataframe")[0]
    fe_names = list(final.fe_params.index)
    X = Xg[fe_names].to_numpy()
    beta = final.fe_params[fe_names].to_numpy()
    cov = final.cov_params().loc[fe_names, fe_names].to_numpy()
    pred = X @ beta
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, cov, X), 0))
    return week, pred, se


def plot_final_nonlinear_model(d: pd.DataFrame, final) -> None:
    week, pred, se = prediction_grid(final, d)
    lo, hi = pred - 1.96 * se, pred + 1.96 * se

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.scatter(d.week, d.y_conc * 100, s=11, alpha=0.10, color="#7f8c8d",
               edgecolors="none", label="抽血事件")
    ax.fill_between(week, np.maximum(lo, 0) ** 2 * 100, np.maximum(hi, 0) ** 2 * 100,
                    color="#4c78a8", alpha=0.18, label="95%点态置信带")
    ax.plot(week, pred**2 * 100, color="#1f4e79", lw=2.7,
            label="最终非线性混合效应模型")
    ax.axhline(4, color="#b2182b", ls="--", lw=1.2, label="4%达标线")
    ax.axvspan(16, 20, color="#56b4e9", alpha=0.08)
    ax.axvspan(25, 29, color="#777777", alpha=0.14)
    ax.text(18, 13.6, "中段放缓", ha="center", color="#2c7fb8")
    ax.text(27, ax.get_ylim()[1] * 0.82, "稀疏区\n25周后n=11", ha="center", color="#555555")
    ax.set(xlim=(11, 29), xlabel="孕周（周）", ylabel="Y染色体浓度（%）",
           title="最终非线性混合效应模型的孕周效应")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "q1_02_nonlinear_relationship")


def plot_model_evidence() -> None:
    """仅展示最终模型的BMI推断和数据口径稳健性。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))

    ax = axes[0]
    labels = ["基线BMI\n（between）", "孕期内BMI变化\n（within）"]
    est = np.array([OFFICIAL["between"], OFFICIAL["within"]])
    wald = np.array([OFFICIAL["between_wald"], OFFICIAL["within_wald"]])
    boot = np.array([OFFICIAL["between_boot"], OFFICIAL["within_boot"]])
    y = np.arange(2)[::-1]
    ax.axvline(0, color="#444444", lw=1, ls="--")
    for i in range(2):
        ax.errorbar(est[i], y[i] + 0.08,
                    xerr=[[est[i] - wald[i, 0]], [wald[i, 1] - est[i]]],
                    fmt="o", color="#4c78a8", capsize=4, lw=2,
                    label="Wald 95%CI" if i == 0 else None)
        ax.errorbar(est[i], y[i] - 0.08,
                    xerr=[[est[i] - boot[i, 0]], [boot[i, 1] - est[i]]],
                    fmt="s", color="#d55e00", capsize=4, lw=2,
                    label="孕妇级Bootstrap 95%CI" if i == 0 else None)
    ax.set_yticks(y, labels)
    ax.set(xlabel="回归系数（sqrt(Y)尺度）", title="A. 最终模型的BMI效应及区间",
           ylim=(-0.45, 1.45))
    ax.text(0.98, 0.97, "BMI拆分 ML-LRT\np=6.33e-4", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    ax = axes[1]
    robust = ROBUSTNESS.iloc[::-1].reset_index(drop=True)
    ax.axvspan(OFFICIAL["between_boot"][0], OFFICIAL["between_boot"][1],
               color="#4c78a8", alpha=0.14, label="最终模型Bootstrap 95%CI")
    ax.axvline(0, color="#444444", ls="--", lw=1)
    ax.scatter(robust.coef, np.arange(len(robust)), s=55, color="#1f4e79", zorder=3)
    ax.set_yticks(np.arange(len(robust)), robust.setting)
    ax.set(xlabel="基线BMI系数（sqrt(Y)尺度）", title="B. 最终模型的稳健性")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("最终非线性模型的BMI效应与稳健性", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01,
             "基线BMI效应在不同数据口径下均约为−0.0045；孕期内BMI效应区间跨0。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIG_DIR / "q1_03_model_evidence.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "q1_03_model_evidence.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved figures/q1_v4/q1_03_model_evidence.(png|pdf)")


def plot_diagnostics(d: pd.DataFrame, final) -> None:
    fitted = np.asarray(final.fittedvalues)
    resid = np.asarray(final.resid)
    zres = (resid - resid.mean()) / resid.std(ddof=1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.scatter(fitted, resid, s=13, alpha=0.35, color="#4c78a8", edgecolors="none")
    low = sm.nonparametric.lowess(resid, fitted, frac=0.35, return_sorted=True)
    ax.plot(low[:, 0], low[:, 1], color="#d55e00", lw=2)
    ax.axhline(0, color="#444444", ls="--", lw=1)
    ax.set(xlabel="条件拟合值（sqrt尺度）", ylabel="条件残差", title="A. 残差 vs 拟合值")

    ax = axes[0, 1]
    st.probplot(zres, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor("#4c78a8")
    ax.get_lines()[0].set_markeredgecolor("#4c78a8")
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color("#d55e00")
    ax.set_title("B. 标准化残差Q-Q图")

    ax = axes[1, 0]
    scale_loc = np.sqrt(np.abs(zres))
    ax.scatter(fitted, scale_loc, s=13, alpha=0.35, color="#4c78a8", edgecolors="none")
    low = sm.nonparametric.lowess(scale_loc, fitted, frac=0.35, return_sorted=True)
    ax.plot(low[:, 0], low[:, 1], color="#d55e00", lw=2)
    ax.set(xlabel="条件拟合值（sqrt尺度）", ylabel="sqrt(|标准化残差|)",
           title="C. Scale–Location")

    ax = axes[1, 1]
    outlier = np.abs(zres) > 3
    ax.scatter(d.week[~outlier], zres[~outlier], s=13, alpha=0.30,
               color="#4c78a8", edgecolors="none", label="一般事件")
    ax.scatter(d.week[outlier], zres[outlier], s=28, color="#b2182b", label="|标准化残差|>3")
    ax.axhline(0, color="#444444", ls="--", lw=1)
    ax.axhline(3, color="#b2182b", ls=":", lw=1)
    ax.axhline(-3, color="#b2182b", ls=":", lw=1)
    ax.axvspan(25, 29, color="#777777", alpha=0.12)
    ax.set(xlim=(11, 29), xlabel="孕周（周）", ylabel="标准化残差",
           title="D. 标准化残差 vs 孕周")
    ax.legend(frameon=False)

    fig.suptitle("最终混合效应模型诊断", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01,
             "Shapiro–Wilk p<0.0001；Breusch–Pagan p<0.0001；红点为18个大残差事件（1.76%）。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(FIG_DIR / "q1_04_diagnostics.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "q1_04_diagnostics.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved figures/q1_v4/q1_04_diagnostics.(png|pdf)")


def grouped_cv_predictions(d: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    ids = rng.permutation(d.mother_id.unique())
    folds = np.array_split(ids, 5)
    frames = []
    for fold_no, test_ids in enumerate(folds, 1):
        tr = d[~d.mother_id.isin(test_ids)].copy()
        te = d[d.mother_id.isin(test_ids)].copy()
        for frame in (tr, te):
            frame["week_c"] = frame.week - tr.week.mean()
            frame["bmi_between_c"] = frame.bmi_baseline - tr.bmi_baseline.mean()
        result = None
        for method in ("lbfgs", "powell"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidate = smf.mixedlm(
                        FINAL_F, tr, groups=tr.mother_id, re_formula="~week_c"
                    ).fit(reml=False, method=method, maxiter=300, disp=False)
                if bool(candidate.converged):
                    result = candidate
                    break
            except Exception:
                continue
        if result is None:
            raise RuntimeError(f"第{fold_no}折混合模型拟合失败")
        pred_resp = np.asarray(result.predict(te), float)
        pred_y = pred_resp**2 + float(result.scale)
        frames.append(
            pd.DataFrame(
                {
                    "fold": fold_no,
                    "observed": te.y_conc.to_numpy(),
                    "predicted": pred_y,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def plot_grouped_cv(d: pd.DataFrame) -> None:
    pred = grouped_cv_predictions(d)
    fold_metrics = []
    for fold, g in pred.groupby("fold"):
        sse = np.sum((g.observed - g.predicted) ** 2)
        sst = np.sum((g.observed - g.observed.mean()) ** 2)
        fold_metrics.append((fold, 1 - sse / sst))
    fold_metrics = pd.DataFrame(fold_metrics, columns=["fold", "R2"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    ax = axes[0]
    palette = sns.color_palette("Blues", 6)[1:]
    for fold, g in pred.groupby("fold"):
        ax.scatter(g.observed * 100, g.predicted * 100, s=12, alpha=0.30,
                   color=palette[fold - 1], label=f"第{fold}折", edgecolors="none")
    lim = [0, max(pred.observed.max(), pred.predicted.max()) * 100 * 1.03]
    ax.plot(lim, lim, color="#b2182b", ls="--", lw=1.4, label="理想45°线")
    ax.set(xlim=lim, ylim=lim, xlabel="实测Y浓度（%）", ylabel="预测Y浓度（%）",
           title="A. 最终模型：新孕妇预测值 vs 实测值")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    colors = ["#b2182b" if v < 0 else "#4c78a8" for v in fold_metrics.R2]
    ax.bar(fold_metrics.fold.astype(str), fold_metrics.R2, color=colors)
    ax.axhline(0, color="#444444", lw=1)
    for i, v in enumerate(fold_metrics.R2):
        ax.text(i, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                color="white" if v < 0 else "#222222")
    ax.set(xlabel="测试折", ylabel="折内R²", title="B. 最终模型各折预测R²")
    ax.text(0.5, 0.97, "总体R²=-0.03986", transform=ax.transAxes, ha="center", va="top")
    fig.suptitle("最终非线性模型按孕妇分组的5折交叉验证", fontsize=15, fontweight="bold")
    save_figure(fig, "q1_05_grouped_cv")


def cluster_boot_corr(d: pd.DataFrame, x: str, B: int = 1000) -> np.ndarray:
    # 与 q1_model.py 的正式结果使用相同随机种子，确保图中区间逐位一致。
    rng = np.random.default_rng(0)
    groups = {m: g[[x, "y_conc"]].to_numpy() for m, g in d.groupby("mother_id")}
    ids = np.array(list(groups))
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(ids, len(ids), replace=True)
        arr = np.vstack([groups[m] for m in pick])
        out[b] = st.pearsonr(arr[:, 0], arr[:, 1]).statistic
    return out


def main() -> None:
    setup_style()
    # 目录仅保留最终模型的五张必要图；决策图归入问题二。
    for pattern in ("*.png", "*.pdf"):
        for old in FIG_DIR.glob(pattern):
            old.unlink()
    d, _ = load_data()
    final = fit_final(d)
    check_official(final, d)
    plot_data_correlation(d.copy())
    plot_final_nonlinear_model(d, final)
    plot_model_evidence()
    plot_diagnostics(d, final)
    plot_grouped_cv(d)
    print(f"完成：共生成 {len(list(FIG_DIR.glob('*.png')))} 张PNG和"
          f" {len(list(FIG_DIR.glob('*.pdf')))} 张PDF。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"绘图失败：{exc}", file=sys.stderr)
        raise
