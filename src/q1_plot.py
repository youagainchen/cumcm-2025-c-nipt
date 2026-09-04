# -*- coding: utf-8 -*-
"""问题一 v4 论文图表生成脚本。

数据来源：
  data/processed/male_clean_event.csv  主分析事件级数据
  outputs/q1_coef.npy                 正式问题二接口
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
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent.parent
EVENT = ROOT / "data" / "processed" / "male_clean_event.csv"
ROWLV = ROOT / "data" / "processed" / "male_clean.csv"
COEF_PATH = ROOT / "outputs" / "q1_coef.npy"
LOG_PATH = ROOT / "outputs" / "q1_out.txt"
FIG_DIR = ROOT / "figures" / "q1_v4"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WEEK_TERM = "cr(week_c, df=3, constraints='center')"
FINAL_F = f"resp ~ {WEEK_TERM} + bmi_between_c + bmi_within"

# 这些是 outputs/q1_out.txt 中完整精度运行的正式汇总数字。主模型重新拟合后
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

MODEL_COMPARISON = pd.DataFrame(
    [
        ("线性孕周", -3619.07, -3589.50),
        ("线性+基线BMI", -3632.61, -3598.11),
        ("样条df3+基线BMI", -3695.11, -3650.76),
        ("二次+基线BMI", -3672.99, -3633.56),
        ("最终模型", -3695.40, -3646.12),
        ("最终+交互", -3694.68, -3640.47),
        ("最终+年龄", -3695.53, -3641.32),
        ("身高体重替代", -3697.42, -3643.21),
        ("样条df4", -3695.33, -3641.11),
        ("样条df5", -3694.14, -3634.99),
        ("样条df6", -3692.19, -3628.12),
    ],
    columns=["model", "AIC", "BIC"],
)

CV_OFFICIAL = pd.DataFrame(
    [
        ("中心化样条", -0.03986, 0.03407, 0.02774),
        ("样条+年龄", -0.03080, 0.03392, 0.02771),
        ("样条+身高体重", -0.03251, 0.03395, 0.02757),
        ("线性", -0.03831, 0.03404, 0.02777),
    ],
    columns=["model", "R2_Y", "RMSE_Y", "MAE_Y"],
)

ROBUSTNESS = pd.DataFrame(
    [
        ("主模型\n事件级", -0.00458),
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
    for path in (EVENT, ROWLV, COEF_PATH, LOG_PATH):
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
    """防止数据更新后继续使用旧日志中的区间和比较结果。"""
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


def plot_trajectory_coverage(d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.65, 1]})
    ax = axes[0]
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

    ax = axes[1]
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
    fig.suptitle("问题一：原始纵向轨迹与孕周覆盖", fontsize=15, fontweight="bold")
    save_figure(fig, "q1_01_trajectory_coverage")


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


def plot_nonlinear_comparison(d: pd.DataFrame, final) -> None:
    week, pred, se = prediction_grid(final, d)
    lo, hi = pred - 1.96 * se, pred + 1.96 * se

    Xlin = sm.add_constant(d.week)
    lin = sm.OLS(d.resp, Xlin).fit()
    pred_lin = lin.predict(sm.add_constant(pd.Series(week, name="week")))
    Xq = sm.add_constant(np.column_stack([d.week, d.week**2]))
    quad = sm.OLS(d.resp, Xq).fit()
    pred_quad = quad.predict(sm.add_constant(np.column_stack([week, week**2])))
    low = sm.nonparametric.lowess(d.resp, d.week, frac=0.35, return_sorted=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.scatter(d.week, d.y_conc * 100, s=11, alpha=0.10, color="#7f8c8d",
               edgecolors="none", label="抽血事件")
    ax.plot(week, pred_lin**2 * 100, color="#999999", ls="-.", lw=1.5, label="线性")
    ax.plot(week, pred_quad**2 * 100, color="#e69f00", ls="--", lw=1.8, label="二次")
    ax.plot(low[:, 0], low[:, 1] ** 2 * 100, color="#009e73", ls=":", lw=2.2,
            label="LOWESS")
    ax.fill_between(week, np.maximum(lo, 0) ** 2 * 100, np.maximum(hi, 0) ** 2 * 100,
                    color="#4c78a8", alpha=0.18, label="最终样条95%点态CI")
    ax.plot(week, pred**2 * 100, color="#1f4e79", lw=2.7, label="中心化自然样条(df=3)")
    ax.axhline(4, color="#b2182b", ls="--", lw=1.2, label="4%达标线")
    ax.axvspan(16, 20, color="#56b4e9", alpha=0.08)
    ax.axvspan(25, 29, color="#777777", alpha=0.14)
    ax.text(18, 13.6, "中段放缓", ha="center", color="#2c7fb8")
    ax.text(27, ax.get_ylim()[1] * 0.82, "稀疏区\n25周后n=11", ha="center", color="#555555")
    ax.text(0.02, 0.97, "ΔAIC（二次−样条）=22.12", transform=ax.transAxes,
            ha="left", va="top", bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
    ax.set(xlim=(11, 29), xlabel="孕周（周）", ylabel="Y染色体浓度（%）",
           title="孕周非线性：线性、二次、LOWESS与最终样条比较")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "q1_02_nonlinear_comparison")


def plot_bmi_forest() -> None:
    labels = ["基线BMI（between）", "孕期内BMI变化（within）"]
    est = np.array([OFFICIAL["between"], OFFICIAL["within"]])
    wald = np.array([OFFICIAL["between_wald"], OFFICIAL["within_wald"]])
    boot = np.array([OFFICIAL["between_boot"], OFFICIAL["within_boot"]])
    y = np.arange(2)[::-1]

    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.axvline(0, color="#444444", lw=1, ls="--")
    for i in range(2):
        ax.errorbar(est[i], y[i] + 0.09,
                    xerr=[[est[i] - wald[i, 0]], [wald[i, 1] - est[i]]],
                    fmt="o", color="#4c78a8", capsize=4, lw=2,
                    label="Wald 95%CI" if i == 0 else None)
        ax.errorbar(est[i], y[i] - 0.09,
                    xerr=[[est[i] - boot[i, 0]], [boot[i, 1] - est[i]]],
                    fmt="s", color="#d55e00", capsize=4, lw=2,
                    label="孕妇级Bootstrap 95%CI" if i == 0 else None)
    ax.set_yticks(y, labels)
    ax.set(xlabel="回归系数（sqrt(Y)尺度）", title="BMI组间效应与组内效应")
    ax.text(0.98, 0.94, "BMI拆分 ML-LRT p=6.33e-4", transform=ax.transAxes,
            ha="right", va="top")
    ax.text(OFFICIAL["between"], y[0] + 0.20, "p=0.00004", ha="center", fontsize=9)
    ax.text(OFFICIAL["within"], y[1] + 0.20, "p=0.13163", ha="center", fontsize=9)
    ax.set_ylim(-0.35, 1.38)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(0.98, 0.43))
    save_figure(fig, "q1_03_bmi_effect_forest")


def plot_model_comparison() -> None:
    tab = MODEL_COMPARISON.sort_values("AIC", ascending=False).reset_index(drop=True)
    y = np.arange(len(tab))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, metric, color in zip(axes, ["AIC", "BIC"], ["#4c78a8", "#e69f00"]):
        colors = ["#1f4e79" if m == "最终模型" else color for m in tab.model]
        ax.scatter(tab[metric], y, c=colors, s=55, zorder=3)
        for yy, val in zip(y, tab[metric]):
            ax.text(val + 0.8, yy, f"{val:.2f}", va="center", fontsize=8)
        ax.set_xlabel(metric + "（越低越好）")
        ax.set_title(metric + "比较")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, tab.model)
    fig.suptitle("固定效应与样条复杂度比较", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, "最终模型综合考虑简约性、BMI解释、问题二接口与组级验证，非机械选择最低AIC。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    # 已手动调整布局，直接保存避免 helper 再次压缩脚注。
    fig.savefig(FIG_DIR / "q1_04_model_comparison.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "q1_04_model_comparison.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved figures/q1_v4/q1_04_model_comparison.(png|pdf)")


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
    fig.savefig(FIG_DIR / "q1_05_diagnostics.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "q1_05_diagnostics.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved figures/q1_v4/q1_05_diagnostics.(png|pdf)")


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

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    ax = axes[0]
    x = np.arange(len(CV_OFFICIAL))
    width = 0.36
    ax.bar(x - width / 2, CV_OFFICIAL.RMSE_Y, width, color="#4c78a8", label="RMSE")
    ax.bar(x + width / 2, CV_OFFICIAL.MAE_Y, width, color="#e69f00", label="MAE")
    ax.set_xticks(x, CV_OFFICIAL.model, rotation=30, ha="right")
    ax.set(ylabel="原始Y浓度误差", title="A. 候选模型预测误差")
    ax.legend(frameon=False)

    ax = axes[1]
    palette = sns.color_palette("Blues", 6)[1:]
    for fold, g in pred.groupby("fold"):
        ax.scatter(g.observed * 100, g.predicted * 100, s=12, alpha=0.30,
                   color=palette[fold - 1], label=f"第{fold}折", edgecolors="none")
    lim = [0, max(pred.observed.max(), pred.predicted.max()) * 100 * 1.03]
    ax.plot(lim, lim, color="#b2182b", ls="--", lw=1.4, label="理想45°线")
    ax.set(xlim=lim, ylim=lim, xlabel="实测Y浓度（%）", ylabel="预测Y浓度（%）",
           title="B. 新孕妇预测值 vs 实测值")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[2]
    colors = ["#b2182b" if v < 0 else "#4c78a8" for v in fold_metrics.R2]
    ax.bar(fold_metrics.fold.astype(str), fold_metrics.R2, color=colors)
    ax.axhline(0, color="#444444", lw=1)
    for i, v in enumerate(fold_metrics.R2):
        ax.text(i, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                color="white" if v < 0 else "#222222")
    ax.set(xlabel="测试折", ylabel="折内R²", title="C. 各折预测R²")
    ax.text(0.5, 0.97, "总体R²=-0.03986", transform=ax.transAxes, ha="center", va="top")
    fig.suptitle("按孕妇分组的5折交叉验证", fontsize=15, fontweight="bold")
    save_figure(fig, "q1_06_grouped_cv")


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


def plot_correlation(d: pd.DataFrame) -> None:
    cols = ["week", "bmi", "age", "height", "weight", "y_conc"]
    names = ["孕周", "BMI", "年龄", "身高", "体重", "Y浓度"]
    corr = d[cols].corr(method="spearman")
    boot_week = cluster_boot_corr(d, "week")
    boot_bmi = cluster_boot_corr(d, "bmi")
    vals = [np.mean(boot_week), np.mean(boot_bmi)]
    cis = [np.percentile(boot_week, [2.5, 97.5]), np.percentile(boot_bmi, [2.5, 97.5])]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), gridspec_kw={"width_ratios": [1.25, 1]})
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                xticklabels=names, yticklabels=names, square=True, cbar_kws={"shrink": 0.8},
                ax=axes[0])
    axes[0].set_title("A. 事件级Spearman相关（描述性）")
    y = np.array([1, 0])
    for i, (v, ci, color) in enumerate(zip(vals, cis, ["#4c78a8", "#d55e00"])):
        axes[1].errorbar(v, y[i], xerr=[[v - ci[0]], [ci[1] - v]], fmt="o",
                         color=color, capsize=5, lw=2.2)
        axes[1].text(ci[1] + 0.01, y[i], f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", va="center")
    axes[1].axvline(0, color="#444444", ls="--", lw=1)
    axes[1].set_yticks(y, ["孕周–Y浓度", "BMI–Y浓度"])
    axes[1].set(xlabel="Pearson相关系数", xlim=(-0.32, 0.30),
                title="B. 孕妇级Bootstrap 95%CI")
    fig.suptitle("问题一相关特性", fontsize=15, fontweight="bold")
    save_figure(fig, "q1_s01_correlation")


def plot_within_between(d: pd.DataFrame) -> None:
    dd = d[d.groupby("mother_id").mother_id.transform("size") >= 2].copy()
    dd["dw"] = dd.week - dd.groupby("mother_id").week.transform("mean")
    dd["dy"] = dd.y_conc - dd.groupby("mother_id").y_conc.transform("mean")
    bw = d.groupby("mother_id").agg(week=("week", "mean"), y=("y_conc", "mean"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(data=dd, x="dw", y="dy", scatter_kws={"s": 12, "alpha": 0.22},
                line_kws={"color": "#d55e00", "lw": 2}, ax=axes[0])
    axes[0].axhline(0, color="#777777", lw=0.8)
    axes[0].axvline(0, color="#777777", lw=0.8)
    axes[0].set(xlabel="孕周−该孕妇平均孕周", ylabel="Y浓度−该孕妇平均Y浓度",
                title="A. 孕妇内关系：r=+0.5920")
    sns.regplot(data=bw, x="week", y="y", scatter_kws={"s": 20, "alpha": 0.42},
                line_kws={"color": "#d55e00", "lw": 2}, ax=axes[1])
    axes[1].set(xlabel="孕妇平均孕周", ylabel="孕妇平均Y浓度",
                title="B. 孕妇间关系：r=−0.2618")
    fig.suptitle("孕周–Y浓度的组内与组间反号", fontsize=15, fontweight="bold")
    save_figure(fig, "q1_s02_within_between")


def plot_variance_components() -> None:
    labels = ["孕妇间", "抽血间", "测序内"]
    vals = np.array([OFFICIAL["var_mother"], OFFICIAL["var_draw"], OFFICIAL["var_tech"]])
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=["#4c78a8", "#e69f00", "#009e73"])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.00006, f"{val:.6f}",
                ha="center", va="bottom")
    lo, hi = OFFICIAL["tech_ci"]
    ax.errorbar(2, OFFICIAL["tech_var"],
                yerr=[[OFFICIAL["tech_var"] - lo], [hi - OFFICIAL["tech_var"]]],
                color="#222222", capsize=5, lw=1.8, zorder=4, label="直接池化95%CI")
    ax.set(ylabel="方差（sqrt(Y)尺度）", title="三层方差分解与技术误差")
    ax.legend(frameon=False)
    save_figure(fig, "q1_s03_variance_components")


def load_coef() -> dict:
    coef = np.load(COEF_PATH, allow_pickle=True).item()
    required = {"week_grid", "week_effect_grid", "week_min", "week_max", "b3", "b4",
                "mu_w", "mu_bb", "s2u0", "s2u1", "cov01", "s2e", "g0", "g1", "g_var"}
    missing = required.difference(coef)
    if missing:
        raise RuntimeError(f"q1_coef.npy 缺少字段: {sorted(missing)}")
    if not coef.get("converged", False):
        raise RuntimeError("q1_coef.npy 标记为未收敛，拒绝画预测图。")
    return coef


def mean_resp(week, bmi_baseline, C):
    week = np.asarray(week, float)
    bmi_baseline = np.asarray(bmi_baseline, float)
    eff = np.interp(week, C["week_grid"], C["week_effect_grid"], left=np.nan, right=np.nan)
    wc = week - C["mu_w"]
    within = C["g0"] + C["g1"] * wc
    return eff + C["b3"] * (bmi_baseline - C["mu_bb"]) + C["b4"] * within


def var_resp(week, C):
    wc = np.asarray(week, float) - C["mu_w"]
    return (C["s2u0"] + 2 * C["cov01"] * wc + C["s2u1"] * wc**2
            + C["s2e"] + C["b4"] ** 2 * C["g_var"])


def probability(week, bmi, C):
    mu = mean_resp(week, bmi, C)
    return st.norm.sf((np.sqrt(0.04) - mu) / np.sqrt(var_resp(week, C)))


def earliest_week(bmi: float, p: float, C) -> float:
    scan = np.asarray(C["week_grid"], float)
    vals = probability(scan, bmi, C) - p
    if vals[0] >= 0:
        return float(scan[0])
    crossing = np.where((vals[:-1] < 0) & (vals[1:] >= 0))[0]
    if not len(crossing):
        return np.nan
    i = int(crossing[0])
    return brentq(lambda w: float(probability(w, bmi, C) - p), scan[i], scan[i + 1])


def plot_icc(C) -> None:
    week = np.linspace(C["week_min"], C["week_max"], 361)
    wc = week - C["mu_w"]
    vre = C["s2u0"] + 2 * C["cov01"] * wc + C["s2u1"] * wc**2
    icc = vre / (vre + C["s2e"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(week, icc, color="#1f4e79", lw=2.5)
    ax.axvspan(25, 29, color="#777777", alpha=0.14, label="25周后稀疏区")
    for wk in [12, 16, 20, 24]:
        val = np.interp(wk, week, icc)
        ax.scatter(wk, val, color="#d55e00", zorder=3)
        ax.text(wk, val + 0.008, f"{val:.3f}", ha="center", fontsize=8)
    ax.set(xlim=(11, 29), ylim=(0.75, 0.94), xlabel="孕周（周）", ylabel="ICC",
           title="孕妇内相关系数随孕周变化")
    ax.legend(frameon=False)
    save_figure(fig, "q1_s04_icc_by_week")


def plot_robustness() -> None:
    tab = ROBUSTNESS.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axvspan(OFFICIAL["between_boot"][0], OFFICIAL["between_boot"][1],
               color="#4c78a8", alpha=0.14, label="主模型Bootstrap 95%CI")
    ax.axvline(0, color="#444444", ls="--", lw=1)
    ax.scatter(tab.coef, np.arange(len(tab)), s=60, color="#1f4e79", zorder=3)
    for yy, val in enumerate(tab.coef):
        ax.text(val + 0.00008, yy, f"{val:.5f}", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(tab)), tab.setting)
    ax.set(xlabel="基线BMI系数（sqrt(Y)尺度）", title="不同数据口径下的基线BMI效应")
    ax.legend(frameon=False)
    save_figure(fig, "q1_s05_robustness")


def plot_probability_surface(C) -> None:
    week = np.linspace(C["week_min"], C["week_max"], 361)
    bmi = np.linspace(20.7, 46.9, 270)
    W, B = np.meshgrid(week, bmi)
    P = probability(W, B, C)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    levels = np.linspace(0, 1, 21)
    cf = ax.contourf(W, B, P, levels=levels, cmap="viridis", extend="both")
    cs = ax.contour(W, B, P, levels=[0.5, 0.8, 0.9, 0.95], colors="white", linewidths=1.4)
    ax.clabel(cs, fmt=lambda x: f"P={x:.2f}", inline=True, fontsize=8)
    ax.axvspan(25, 29, color="white", alpha=0.18)
    ax.text(27, 45.5, "稀疏区", color="white", ha="center", fontweight="bold")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("P(Y≥4%)")
    ax.set(xlim=(11, 29), xlabel="孕周（周）", ylabel="基线BMI",
           title="Y染色体浓度达到4%的模型概率")
    fig.text(0.5, 0.01, "概率用于相对时点比较，尚未经过独立外部校准；25周后数据稀疏。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FIG_DIR / "q1_07_probability_surface.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "q1_07_probability_surface.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved figures/q1_v4/q1_07_probability_surface.(png|pdf)")


def plot_week_reach(C) -> None:
    bmi = np.linspace(20.7, 46.9, 160)
    probs = [0.50, 0.80, 0.90, 0.95]
    colors = ["#009e73", "#56b4e9", "#e69f00", "#d55e00"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for p, color in zip(probs, colors):
        reach = np.array([earliest_week(x, p, C) for x in bmi])
        at_lower = np.isfinite(reach) & (reach <= C["week_min"] + 1e-8)
        regular = np.isfinite(reach) & ~at_lower
        ax.plot(bmi[regular], reach[regular], color=color, lw=2.2, label=f"目标概率 {p:.2f}")
        ax.scatter(bmi[at_lower], reach[at_lower], facecolors="none", edgecolors=color,
                   s=18, linewidths=0.8)
    ax.axhspan(25, 29, color="#777777", alpha=0.12, label="25周后稀疏区")
    ax.axhline(11, color="#444444", ls=":", lw=1)
    ax.text(46.6, 11.15, "空心点：触及11周观测下界", ha="right", fontsize=8)
    ax.set(xlim=(20.7, 46.9), ylim=(10.7, 29.2), xlabel="基线BMI",
           ylabel="域内最早达标孕周", title="不同把握度下的最早达标孕周")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "q1_08_week_reach")


def main() -> None:
    setup_style()
    d, _ = load_data()
    final = fit_final(d)
    check_official(final, d)
    coef = load_coef()

    plot_trajectory_coverage(d.copy())
    plot_nonlinear_comparison(d, final)
    plot_bmi_forest()
    plot_model_comparison()
    plot_diagnostics(d, final)
    plot_grouped_cv(d)
    plot_correlation(d)
    plot_within_between(d)
    plot_variance_components()
    plot_icc(coef)
    plot_robustness()
    plot_probability_surface(coef)
    plot_week_reach(coef)
    print(f"完成：共生成 {len(list(FIG_DIR.glob('*.png')))} 张PNG和"
          f" {len(list(FIG_DIR.glob('*.pdf')))} 张PDF。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"绘图失败：{exc}", file=sys.stderr)
        raise
