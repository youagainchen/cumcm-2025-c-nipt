# -*- coding: utf-8 -*-
"""问题一：Y 浓度与孕周、BMI 等因素的 GAMM 分析。

本脚本使用 B-spline 展开实现广义加性混合模型：孕周和 BMI 用平滑项，
孕妇代码作为随机截距，以处理同一孕妇的重复检测。脚本优先读取事件级
数据，找不到事件级文件时回退到行级数据。

运行示例（数据准备好后）：
    python src/q1_gamm.py
    python src/q1_gamm.py --exclude-qc-flagged

输出：
    outputs/q1_gamm/fixed_effects.csv
    outputs/q1_gamm/model_summary.txt
    outputs/q1_gamm/model_meta.json
    figures/q1_gamm/week_bmi_effect.png
    figures/q1_gamm/residual_diagnostics.png
"""
from __future__ import annotations

import argparse
import json
import warnings
from math import erf
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from statsmodels.regression.mixed_linear_model import MixedLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "q1_gamm"
DEFAULT_FIG_DIR = ROOT / "figures" / "q1_gamm"

OPTIONAL_COVARIATES = [
    "age", "reads_raw", "map_ratio", "dup_ratio", "uniq_reads", "gc", "filt_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the Q1 GAMM model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--exclude-qc-flagged", action="store_true")
    parser.add_argument("--week-df", type=int, default=6)
    parser.add_argument("--bmi-df", type=int, default=5)
    return parser.parse_args()


def locate_data(data_dir: Path) -> tuple[Path, str, str, str]:
    candidates = [
        (data_dir / "male_clean_event.csv", "y_conc_mean", "week_mean", "event"),
        (data_dir / "male_clean.csv", "y_conc", "week", "row"),
        (data_dir / "male_min.csv", "y_conc", "week", "row"),
    ]
    for path, y_col, week_col, level in candidates:
        if path.exists():
            return path, y_col, week_col, level
    names = ", ".join(str(path.name) for path, *_ in candidates)
    raise FileNotFoundError(f"未找到男胎数据，请准备以下文件之一：{names}")


def read_model_data(data_dir: Path, exclude_qc_flagged: bool) -> tuple[pd.DataFrame, dict]:
    path, y_col, week_col, level = locate_data(data_dir)
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = ["mother_id", "bmi", y_col, week_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"数据缺少必要字段：{missing}")

    original_rows = len(data)
    data = data.copy()
    data["y_model"] = pd.to_numeric(data[y_col], errors="coerce")
    data["week_model"] = pd.to_numeric(data[week_col], errors="coerce")
    numeric_columns = ["bmi", "y_model", "week_model"] + [
        column for column in OPTIONAL_COVARIATES if column in data.columns
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    excluded_qc = 0
    if exclude_qc_flagged and "flag_any" in data.columns:
        excluded_qc = int(data["flag_any"].fillna(0).astype(bool).sum())
        data = data.loc[~data["flag_any"].fillna(0).astype(bool)].copy()

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.loc[
        data["mother_id"].notna()
        & data["bmi"].notna()
        & data["week_model"].notna()
        & data["y_model"].notna()
        & data["y_model"].between(0, 1)
    ].copy()

    candidate_covariates = [
        column for column in OPTIONAL_COVARIATES
        if column in data.columns and data[column].notna().mean() >= 0.90
    ]
    model_columns = ["mother_id", "bmi", "week_model", "y_model"] + candidate_covariates
    data = data[model_columns].dropna().copy()
    data["mother_id"] = data["mother_id"].astype(str)

    if data["mother_id"].nunique() < 2:
        raise ValueError("有效数据中的孕妇数量不足，无法拟合混合效应模型。")
    if len(data) < 30:
        raise ValueError("有效记录数不足 30 条，暂不建议拟合 GAMM。")

    meta = {
        "source": str(path),
        "data_level": level,
        "original_rows": original_rows,
        "qc_flagged_excluded": excluded_qc,
        "model_rows": int(len(data)),
        "mother_count": int(data["mother_id"].nunique()),
        "covariates": candidate_covariates,
        "dropped_for_missing_or_invalid": int(original_rows - len(data) - excluded_qc),
    }
    return data, meta


def make_formula(week_df: int, bmi_df: int, covariates: list[str]) -> str:
    terms = [
        "1",
        f"bs(week_model, df={week_df}, degree=3, include_intercept=False)",
        f"bs(bmi, df={bmi_df}, degree=3, include_intercept=False)",
    ]
    terms.extend(covariates)
    return " + ".join(terms)


def fit_gamm(data: pd.DataFrame, week_df: int, bmi_df: int) -> tuple[object, object, str]:
    formula = make_formula(week_df, bmi_df, [
        column for column in OPTIONAL_COVARIATES if column in data.columns
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design = dmatrix(formula, data, return_type="dataframe")
        model = MixedLM(
            data["y_model"],
            design,
            groups=data["mother_id"].to_numpy(),
            exog_re=pd.DataFrame({"mother_intercept": 1.0}, index=data.index),
        )
        try:
            result = model.fit(reml=True, method="lbfgs", maxiter=2000, disp=False)
        except Exception:
            result = model.fit(reml=True, method="powell", maxiter=2000, disp=False)
    return model, result, formula


def fixed_effect_table(model: object, result: object) -> pd.DataFrame:
    names = list(model.exog_names)
    estimates = np.asarray(result.fe_params)
    standard_errors = np.asarray(result.bse_fe)
    z_values = estimates / standard_errors
    p_values = 2 * (1 - _normal_cdf(np.abs(z_values)))
    return pd.DataFrame({
        "term": names,
        "estimate": estimates,
        "std_error": standard_errors,
        "z_value": z_values,
        "p_value": p_values,
        "ci_low": estimates - 1.96 * standard_errors,
        "ci_high": estimates + 1.96 * standard_errors,
    })


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(lambda value: erf(value / np.sqrt(2)))(values))


def build_prediction_frame(data: pd.DataFrame, week_values: np.ndarray) -> pd.DataFrame:
    prediction = pd.DataFrame({
        "week_model": week_values,
        "bmi": np.repeat(data["bmi"].median(), len(week_values)),
    })
    for column in OPTIONAL_COVARIATES:
        if column in data.columns:
            prediction[column] = data[column].median()
    return prediction


def predict_fixed(result: object, design_info: object, frame: pd.DataFrame) -> np.ndarray:
    design = build_design_matrices([design_info], frame, return_type="dataframe")[0]
    return np.asarray(result.predict(exog=design))


def save_effect_plot(
    data: pd.DataFrame,
    result: object,
    design_info: object,
    output_path: Path,
) -> None:
    week_values = np.linspace(data["week_model"].min(), data["week_model"].max(), 160)
    frame = build_prediction_frame(data, week_values)
    predictions = predict_fixed(result, design_info, frame)
    plt.figure(figsize=(8, 5))
    plt.scatter(data["week_model"], data["y_model"], s=12, alpha=0.22, color="#3B6EA8")
    plt.plot(week_values, predictions, color="#C44E52", linewidth=2.2)
    plt.xlabel("Gestational week")
    plt.ylabel("Y concentration")
    plt.title("GAMM fitted relationship between gestational week and Y concentration")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_diagnostic_plot(data: pd.DataFrame, result: object, design_info: object, output_path: Path) -> None:
    design = build_design_matrices([design_info], data, return_type="dataframe")[0]
    fitted = np.asarray(result.predict(exog=design))
    residual = data["y_model"].to_numpy() - fitted
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(fitted, residual, s=12, alpha=0.35, color="#3B6EA8")
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set_xlabel("Fitted value")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs fitted")
    axes[1].hist(residual, bins=25, color="#7A9E9F", edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


class FittedGAMM:
    """给问题二调用的固定效应预测接口。"""

    def __init__(self, result: object, design_info: object, medians: dict[str, float], bounds: dict[str, tuple[float, float]]):
        self.result = result
        self.design_info = design_info
        self.medians = medians
        self.bounds = bounds

    def predict(self, week: float | np.ndarray, bmi: float | np.ndarray, **kwargs: float) -> np.ndarray:
        week_array, bmi_array = np.broadcast_arrays(np.asarray(week, dtype=float), np.asarray(bmi, dtype=float))
        frame = pd.DataFrame({"week_model": week_array.ravel(), "bmi": bmi_array.ravel()})
        for column, value in self.medians.items():
            frame[column] = kwargs.get(column, value)
        frame["week_model"] = frame["week_model"].clip(*self.bounds["week_model"])
        frame["bmi"] = frame["bmi"].clip(*self.bounds["bmi"])
        design = build_design_matrices([self.design_info], frame, return_type="dataframe")[0]
        prediction = np.asarray(self.result.predict(exog=design))
        return prediction.reshape(week_array.shape)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    data, meta = read_model_data(args.data_dir, args.exclude_qc_flagged)
    model, result, formula = fit_gamm(data, args.week_df, args.bmi_df)
    design_info = dmatrix(formula, data, return_type="dataframe").design_info

    effects = fixed_effect_table(model, result)
    effects.to_csv(args.output_dir / "fixed_effects.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "model_summary.txt").write_text(result.summary().as_text(), encoding="utf-8")
    meta.update({
        "formula": formula,
        "week_df": args.week_df,
        "bmi_df": args.bmi_df,
        "random_effect": "mother_id random intercept",
        "aic": getattr(result, "aic", None),
        "bic": getattr(result, "bic", None),
        "converged": bool(getattr(result, "converged", False)),
    })
    (args.output_dir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    save_effect_plot(data, result, design_info, args.fig_dir / "week_bmi_effect.png")
    save_diagnostic_plot(data, result, design_info, args.fig_dir / "residual_diagnostics.png")
    print(f"GAMM fitted: rows={len(data)}, mothers={data['mother_id'].nunique()}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
