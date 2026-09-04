# -*- coding: utf-8 -*-
"""问题三：多因素达标时间模型与概率接口。

本脚本以孕妇为分析对象、以抽血事件为纵向观测，完成两条互补路径：

1. sqrt(Y) 混合效应模型：描述单次检测达到 4% 的概率；
2. 区间删失 AFT 模型：直接描述首次达到 4% 的潜在时间分布。

问题二优化器可调用 ``load_prob_fn`` 获得统一的五参数接口：
``prob_qualified(week, bmi, height, weight, age, thr=0.04, prev_week=None)``。
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import expit
from patsy import build_design_matrices, dmatrix
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.regression.mixed_linear_model import MixedLM


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "male_clean.csv"
OUT = ROOT / "outputs"
MODEL_PATH = OUT / "q3_model.npy"
Q1_S2_TECH = 0.000174
WEEK_MIN, WEEK_MAX = 11.0, 25.0
FEATURES = ("bmi", "height", "weight", "age")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三多因素达标时间模型")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--parameter-mc", "--bootstrap", dest="bootstrap", type=int, default=500,
                        help="参数蒙特卡洛抽样次数（保留--bootstrap兼容别名）")
    return parser.parse_args()


def numeric(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = data.copy()
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def load_events(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {"mother_id", "draw_idx", "week", "bmi", "y_conc", "age", "height", "weight"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"数据缺少字段：{sorted(missing)}")
    data = numeric(data, list(required - {"mother_id"}) +
                   [c for c in ["t_left", "t_right"] if c in data.columns])
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["mother_id", "draw_idx", "week", "bmi", "y_conc",
                               "age", "height", "weight"]).copy()
    data["mother_id"] = data["mother_id"].astype(str)
    data["y_conc"] = data["y_conc"].clip(0, 1)
    group_cols = ["mother_id", "draw_idx"]
    aggregations = {
        "week": "mean", "bmi": "first", "age": "first", "height": "first",
        "weight": "first", "y_conc": "mean",
    }
    events = data.groupby(group_cols, as_index=False).agg(aggregations)
    events["sqrt_y"] = np.sqrt(events["y_conc"])
    events = events.sort_values(["mother_id", "week", "draw_idx"]).reset_index(drop=True)
    subject = (events.sort_values(["mother_id", "week"])
               .groupby("mother_id", as_index=False)
               .first()[["mother_id", "bmi", "age", "height", "weight"]])
    first_observation = (events.sort_values(["mother_id", "week"])
                         .groupby("mother_id", as_index=False).first()
                         [["mother_id", "week", "y_conc"]]
                         .rename(columns={"week": "first_week"}))
    first_observation["first_attained"] = first_observation["y_conc"] >= 0.04
    subject = subject.merge(first_observation[["mother_id", "first_week", "first_attained"]],
                            on="mother_id", how="left")
    if {"t_left", "t_right", "censored"}.issubset(data.columns):
        marks = (data.sort_values(["mother_id", "week"])
                 .groupby("mother_id", as_index=False)
                 .first()[["mother_id", "t_left", "t_right", "censored"]])
        subject = subject.merge(marks, on="mother_id", how="left")
    else:
        subject[["t_left", "t_right", "censored"]] = np.nan
    return events, subject


def standardize(events: pd.DataFrame, subject: pd.DataFrame) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for column in FEATURES:
        mean = float(subject[column].mean())
        scale = float(subject[column].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"协变量 {column} 没有有效变异")
        stats[column] = (mean, scale)
        subject[column + "_z"] = (subject[column] - mean) / scale
        events[column + "_z"] = (events[column] - mean) / scale
    design = np.column_stack([
        np.ones(len(subject)), subject["height_z"], subject["weight_z"]
    ])
    residual_coefficients = np.linalg.lstsq(
        design, subject["bmi_z"].to_numpy(), rcond=None
    )[0]
    subject["bmi_hw_resid"] = subject["bmi_z"] - design @ residual_coefficients
    residual_map = subject.set_index("mother_id")["bmi_hw_resid"]
    events["bmi_hw_resid"] = events["mother_id"].map(residual_map)
    stats["bmi_hw_resid"] = (0.0, float(subject["bmi_hw_resid"].std(ddof=0)))
    stats["bmi_hw_resid_coefficients"] = tuple(float(x) for x in residual_coefficients)
    return stats


def candidate_features(subject: pd.DataFrame) -> dict[str, list[str]]:
    height_weight = ["height_z", "weight_z", "age_z"]
    return {
        "bmi_age": ["bmi_z", "age_z"],
        "height_weight_age": height_weight,
        "bmi_hw_resid_age": ["bmi_hw_resid", "age_z"],
        "all_factors": ["bmi_z", "height_z", "weight_z", "age_z"],
    }


def fit_mixed(events: pd.DataFrame, features: list[str]) -> tuple[object, object, object, str]:
    formula = "1 + bs(week, df=6, degree=3, include_intercept=False) + " + " + ".join(features)
    design = dmatrix(formula, events, return_type="dataframe")
    model = MixedLM(
        events["sqrt_y"], design, groups=events["mother_id"].to_numpy(),
        exog_re=np.ones((len(events), 1)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = None
        for method in ("lbfgs", "powell", "nm"):
            try:
                trial = model.fit(reml=False, method=method, maxiter=1000, disp=False)
            except Exception:
                continue
            if bool(getattr(trial, "converged", False)) and mixed_is_valid(model, trial):
                result = trial
                break
        if result is None:
            raise RuntimeError(f"混合模型未收敛：{features}")
    return model, result, design.design_info, formula


def mixed_is_valid(model: object, result: object) -> bool:
    try:
        exog = np.asarray(model.exog, float)
        if np.linalg.matrix_rank(exog) < exog.shape[1]:
            return False
        condition = np.linalg.cond(exog)
        if not np.isfinite(condition) or condition > 1e10:
            return False
        covariance = np.asarray(result.cov_params(), float)
        random_covariance = np.asarray(result.cov_re, float)
        if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(random_covariance)):
            return False
        if np.min(np.linalg.eigvalsh((covariance + covariance.T) / 2)) < -1e-8:
            return False
        if np.min(np.linalg.eigvalsh((random_covariance + random_covariance.T) / 2)) < -1e-10:
            return False
        if not np.isfinite(float(result.scale)) or float(result.scale) <= 0:
            return False
    except Exception:
        return False
    return True


def fit_mixed_candidates(events: pd.DataFrame, subject: pd.DataFrame) -> tuple[str, dict, pd.DataFrame]:
    specs = candidate_features(subject)
    fits: dict[str, dict] = {}
    rows = []
    for name, features in specs.items():
        try:
            model, result, design_info, formula = fit_mixed(events, features)
            rows.append({"model": name, "features": "+".join(features),
                         "log_likelihood": float(result.llf),
                         "aic": float(result.aic), "bic": float(result.bic),
                         "converged": True})
            fits[name] = {"model": model, "result": result,
                          "design_info": design_info, "features": features,
                          "formula": formula}
        except Exception as exc:
            rows.append({"model": name, "features": "+".join(features),
                         "log_likelihood": np.nan, "aic": np.nan, "bic": np.nan,
                         "converged": False, "error": str(exc)})
    table = pd.DataFrame(rows)
    valid = table.dropna(subset=["aic"])
    if valid.empty:
        raise RuntimeError("所有多因素混合模型均拟合失败")
    table["delta_aic"] = table["aic"] - valid["aic"].min()
    weights = np.where(np.isfinite(table["delta_aic"]),
                       np.exp(-0.5 * table["delta_aic"]), 0.0)
    table["akaike_weight"] = weights / weights.sum()
    best_name = str(valid.sort_values("aic").iloc[0]["model"])
    return best_name, fits, table


def bounds_for_aft(subject: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not {"t_left", "t_right", "censored"}.issubset(subject.columns):
        raise ValueError("数据没有问题二所需的 t_left/t_right/censored 字段")
    lower = subject["t_left"].to_numpy(float).copy()
    upper = subject["t_right"].to_numpy(float).copy()
    censored = subject["censored"].astype(str).to_numpy()
    lower[censored == "left"] = 0.0
    upper[censored == "right"] = np.inf
    lower = np.maximum(lower, 1e-6)
    if not np.all(np.isfinite(lower)):
        raise ValueError("AFT 下界存在无效值")
    return lower, upper


def fit_aft_candidates(subject: pd.DataFrame) -> tuple[str, str, object, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter

    lower, upper = bounds_for_aft(subject)
    specs = candidate_features(subject)
    distributions = {
        "Weibull": WeibullAFTFitter,
        "LogNormal": LogNormalAFTFitter,
        "LogLogistic": LogLogisticAFTFitter,
    }
    fits, rows = {}, []
    for spec_name, features in specs.items():
        frame = subject[features].copy()
        frame["lo"] = lower
        frame["hi"] = upper
        for dist_name, fitter in distributions.items():
            try:
                fit = fitter().fit_interval_censoring(
                    frame, lower_bound_col="lo", upper_bound_col="hi", show_progress=False
                )
                if not aft_is_valid(fit):
                    raise RuntimeError("参数或协方差矩阵无效")
                rows.append({"feature_model": spec_name, "distribution": dist_name,
                             "log_likelihood": float(fit.log_likelihood_),
                             "aic": float(fit.AIC_), "converged": True})
                fits[(spec_name, dist_name)] = fit
            except Exception as exc:
                rows.append({"feature_model": spec_name, "distribution": dist_name,
                             "log_likelihood": np.nan, "aic": np.nan,
                             "converged": False, "error": str(exc)})
    table = pd.DataFrame(rows)
    valid = table.dropna(subset=["aic"])
    if valid.empty:
        raise RuntimeError("所有区间删失 AFT 模型均拟合失败")
    table["delta_aic"] = table["aic"] - valid["aic"].min()
    weights = np.where(np.isfinite(table["delta_aic"]),
                       np.exp(-0.5 * table["delta_aic"]), 0.0)
    table["akaike_weight"] = weights / weights.sum()
    cv_table, calibration_table = grouped_aft_cv(subject, specs)
    multifactor = valid[valid["feature_model"] == "all_factors"]
    cv_scores = (cv_table[cv_table["valid"]]
                 .groupby("distribution", as_index=False)["mean_log_likelihood"].mean()
                 .sort_values("mean_log_likelihood", ascending=False))
    if not multifactor.empty and not cv_scores.empty:
        selected_distribution = str(cv_scores.iloc[0]["distribution"])
        chosen = multifactor[multifactor["distribution"] == selected_distribution].iloc[0]
    else:
        chosen = (multifactor.sort_values("aic").iloc[0]
                  if not multifactor.empty else valid.sort_values("aic").iloc[0])
    key = (str(chosen.feature_model), str(chosen.distribution))
    return key[0], key[1], fits[key], table, specs, cv_table, calibration_table


def aft_is_valid(fit: object) -> bool:
    try:
        params = np.asarray(fit.params_, float)
        covariance = np.asarray(fit.variance_matrix_, float)
        if not np.all(np.isfinite(params)) or not np.all(np.isfinite(covariance)):
            return False
        if np.min(np.linalg.eigvalsh((covariance + covariance.T) / 2)) < -1e-8:
            return False
        return np.isfinite(float(fit.log_likelihood_))
    except Exception:
        return False


def export_aft(best_spec: str, best_dist: str, fit: object,
               specs: dict[str, list[str]], subject: pd.DataFrame,
               stats: dict[str, tuple[float, float]], path: Path) -> dict:
    labels = [tuple(label) for label in fit.params_.index]
    values = fit.params_.to_numpy(float)
    covariance = fit.variance_matrix_.to_numpy(float)
    payload = {
        "model": best_dist, "feature_model": best_spec,
        "features": specs[best_spec], "param_labels": labels,
        "param_values": values, "param_cov": covariance,
        "feature_stats": stats, "week_min": WEEK_MIN, "week_max": WEEK_MAX,
        "threshold": 0.04, "mother_count": int(len(subject)),
        "censor_counts": subject["censored"].value_counts().to_dict(),
    }
    np.save(path, payload, allow_pickle=True)
    return payload


def aft_location(payload: dict, bmi, height, weight, age) -> tuple[np.ndarray, float]:
    features = payload["features"]
    raw = {"bmi": bmi, "height": height, "weight": weight, "age": age}
    stats = payload["feature_stats"]
    z = {column + "_z": (np.asarray(raw[column], float) - stats[column][0]) / stats[column][1]
         for column in FEATURES}
    if "bmi_hw_resid" in features:
        coefficients = payload["feature_stats"]["bmi_hw_resid_coefficients"]
        z["bmi_hw_resid"] = z["bmi_z"] - (
            coefficients[0] + coefficients[1] * z["height_z"] +
            coefficients[2] * z["weight_z"]
        )
    params = dict(zip([tuple(x) for x in payload["param_labels"]], payload["param_values"]))
    location_name = {"Weibull": "lambda_", "LogNormal": "mu_", "LogLogistic": "alpha_"}[payload["model"]]
    intercept = float(params[(location_name, "Intercept")])
    location = np.full(np.broadcast(z["bmi_z"], z["age_z"]).shape, intercept, dtype=float)
    for feature in features:
        location += float(params[(location_name, feature)]) * np.asarray(z[feature], float)
    if payload["model"] == "LogNormal":
        shape = float(np.exp(params[("sigma_", "Intercept")]))
    elif payload["model"] == "LogLogistic":
        shape = float(np.exp(params[("beta_", "Intercept")]))
    else:
        shape = float(np.exp(params[("rho_", "Intercept")]))
    return location, shape


def prob_fn_from_payload(payload: dict, param_values=None):
    payload = dict(payload)
    if param_values is not None:
        payload["param_values"] = np.asarray(param_values, float)
    model = payload["model"]

    def cdf(week, bmi_baseline, height, weight, age):
        location, sigma = aft_location(payload, bmi_baseline, height, weight, age)
        week = np.asarray(week, float)
        if model == "LogNormal":
            return st.norm.cdf((np.log(np.maximum(week, 1e-9)) - location) / sigma)
        if model == "Weibull":
            return 1 - np.exp(-np.power(np.maximum(week, 1e-9) / np.exp(location), sigma))
        return expit(sigma * (np.log(np.maximum(week, 1e-9)) - location))

    def prob_qualified(week, bmi_baseline, height, weight, age,
                       thr=0.04, prev_week=None):
        if not np.isclose(thr, 0.04):
            raise ValueError("问题三模型只针对 Y 浓度 4% 阈值拟合")
        now = np.asarray(cdf(week, bmi_baseline, height, weight, age), float)
        if prev_week is None:
            return now
        previous = np.asarray(cdf(prev_week, bmi_baseline, height, weight, age), float)
        return np.clip((now - previous) / np.maximum(1 - previous, 1e-12), 0, 1)

    return prob_qualified


def load_prob_fn(path: Path = MODEL_PATH):
    payload = np.load(path, allow_pickle=True).item()
    return prob_fn_from_payload(payload)


def mixed_probability(fit_info: dict, events: pd.DataFrame, stats: dict, feature_values: dict,
                      week: float, added_variance: float = 0.0) -> float:
    frame = pd.DataFrame({"week": [week], **feature_values})
    design = build_design_matrices([fit_info["design_info"]], frame, return_type="dataframe")[0]
    mean = float(np.asarray(fit_info["result"].predict(exog=design))[0])
    variance = float(fit_info["result"].scale + fit_info["result"].cov_re.iloc[0, 0] + added_variance)
    return float(st.norm.sf((np.sqrt(0.04) - mean) / np.sqrt(max(variance, 1e-12))))


def parameter_uncertainty(payload: dict, draws: int, subject: pd.DataFrame | None = None,
                          seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    covariance = (payload["param_cov"] + payload["param_cov"].T) / 2
    eigenvalues, vectors = np.linalg.eigh(covariance)
    covariance = (vectors * np.maximum(eigenvalues, 1e-12)) @ vectors.T
    theta = rng.multivariate_normal(payload["param_values"], covariance, size=draws)
    index = {tuple(label): i for i, label in enumerate(payload["param_labels"])}
    location_name = {"Weibull": "lambda_", "LogNormal": "mu_", "LogLogistic": "alpha_"}[payload["model"]]
    if subject is None:
        profiles = pd.DataFrame([{
            "bmi": payload["feature_stats"]["bmi"][0],
            "height": payload["feature_stats"]["height"][0],
            "weight": payload["feature_stats"]["weight"][0],
            "age": payload["feature_stats"]["age"][0],
        }])
    else:
        targets = (30.0, 35.0, 40.0)
        profiles = pd.DataFrame([
            subject.iloc[(subject["bmi"] - target).abs().argmin()][["bmi", "height", "weight", "age"]]
            for target in targets
        ]).reset_index(drop=True)
    rows = []
    for profile in profiles.to_dict("records"):
        z = {column + "_z": (float(profile[column]) - payload["feature_stats"][column][0]) /
             payload["feature_stats"][column][1] for column in FEATURES}
        if "bmi_hw_resid" in payload["features"]:
            coefficients = payload["feature_stats"]["bmi_hw_resid_coefficients"]
            z["bmi_hw_resid"] = z["bmi_z"] - (
                coefficients[0] + coefficients[1] * z["height_z"] + coefficients[2] * z["weight_z"]
            )
        for draw in theta:
            location = draw[index[(location_name, "Intercept")]]
            for feature in payload["features"]:
                location += draw[index[(location_name, feature)]] * z[feature]
            if payload["model"] == "LogNormal":
                shape = np.exp(draw[index[("sigma_", "Intercept")]])
                quantile = lambda p: np.exp(location + shape * st.norm.ppf(p))
            elif payload["model"] == "LogLogistic":
                shape = np.exp(draw[index[("beta_", "Intercept")]])
                quantile = lambda p: np.exp(location) * (p / (1 - p)) ** (1 / shape)
            else:
                shape = np.exp(draw[index[("rho_", "Intercept")]])
                quantile = lambda p: np.exp(location) * (-np.log1p(-p)) ** (1 / shape)
            rows.append({**profile, "t80": quantile(0.80), "t90": quantile(0.90)})
    result = pd.DataFrame(rows)
    summary = []
    for bmi, group in result.groupby("bmi"):
        for metric in ("t80", "t90"):
            values = group[metric].dropna()
            if len(values):
                q = values.quantile([0.025, 0.5, 0.975])
                profile = group.iloc[0]
                summary.append({"quantity": metric, "bmi": bmi,
                                "height": profile["height"], "weight": profile["weight"],
                                "age": profile["age"],
                                "q025": q.iloc[0], "median": q.iloc[1], "q975": q.iloc[2]})
    return pd.DataFrame(summary)


def grouped_aft_cv(subject: pd.DataFrame, specs: dict[str, list[str]], folds: int = 5,
                   seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter

    if len(subject) < folds:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    ids = subject["mother_id"].to_numpy()
    shuffled = rng.permutation(ids)
    rows = []
    calibration_rows = []
    lower, upper = bounds_for_aft(subject)
    features = specs["all_factors"]
    distributions = {
        "Weibull": WeibullAFTFitter,
        "LogNormal": LogNormalAFTFitter,
        "LogLogistic": LogLogisticAFTFitter,
    }
    for fold, validation_ids in enumerate(np.array_split(shuffled, folds), start=1):
        validation_mask = subject["mother_id"].isin(validation_ids).to_numpy()
        train_raw = subject.loc[~validation_mask, list(FEATURES)].copy()
        fold_stats = {
            column: (float(train_raw[column].mean()), float(train_raw[column].std(ddof=0)))
            for column in FEATURES
        }
        if any(not np.isfinite(scale) or scale <= 0 for _, scale in fold_stats.values()):
            raise RuntimeError(f"第{fold}折训练集协变量没有有效变异")
        train = pd.DataFrame({
            column + "_z": (train_raw[column] - fold_stats[column][0]) / fold_stats[column][1]
            for column in FEATURES
        }, index=train_raw.index)
        train["lo"] = lower[~validation_mask]
        train["hi"] = upper[~validation_mask]
        validation_raw = subject.loc[validation_mask, list(FEATURES)].copy()
        validation = pd.DataFrame({
            column + "_z": (validation_raw[column] - fold_stats[column][0]) / fold_stats[column][1]
            for column in FEATURES
        }, index=validation_raw.index)
        validation["first_week"] = subject.loc[validation_mask, "first_week"].to_numpy(float)
        validation["first_attained"] = subject.loc[validation_mask, "first_attained"].to_numpy(bool)
        validation_lower = lower[validation_mask]
        validation_upper = upper[validation_mask]
        for distribution, fitter in distributions.items():
            try:
                fit = fitter().fit_interval_censoring(
                    train, lower_bound_col="lo", upper_bound_col="hi", show_progress=False
                )
                if not aft_is_valid(fit):
                    raise RuntimeError("参数或协方差矩阵无效")
                log_likelihood = 0.0
                predicted = []
                observed = []
                for row, lo, hi in zip(validation[features].to_dict("records"),
                                       validation_lower, validation_upper):
                    frame = pd.DataFrame([row])
                    survival_lo = (float(fit.predict_survival_function(
                        frame, times=[lo]).iloc[0, 0]) if np.isfinite(lo) else 0.0)
                    survival_hi = (float(fit.predict_survival_function(
                        frame, times=[hi]).iloc[0, 0]) if np.isfinite(hi) else 0.0)
                    probability = (1 - survival_hi) if lo <= 1e-5 else (
                        survival_lo - survival_hi if np.isfinite(hi) else survival_lo
                    )
                    log_likelihood += np.log(max(probability, 1e-12))
                if {"first_week", "first_attained"}.issubset(validation.columns):
                    for row, week, attained in zip(
                            validation[features].to_dict("records"),
                            validation["first_week"], validation["first_attained"]):
                        frame = pd.DataFrame([row])
                        survival = float(fit.predict_survival_function(
                            frame, times=[float(week)]).iloc[0, 0])
                        predicted.append(1 - survival)
                        observed.append(float(attained))
                auc = (roc_auc_score(observed, predicted)
                       if len(set(observed)) == 2 else np.nan)
                brier = brier_score_loss(observed, predicted) if observed else np.nan
                if predicted:
                    calibration = pd.DataFrame({"predicted": predicted, "observed": observed})
                    calibration["bin"] = pd.qcut(
                        calibration["predicted"], q=5, duplicates="drop"
                    )
                    grouped = calibration.groupby("bin", observed=True)
                    for bin_name, bin_group in grouped:
                        calibration_rows.append({
                            "fold": fold, "distribution": distribution,
                            "bin": str(bin_name), "n": len(bin_group),
                            "mean_predicted": bin_group["predicted"].mean(),
                            "mean_observed": bin_group["observed"].mean(),
                        })
                rows.append({"fold": fold, "distribution": distribution,
                             "feature_model": "all_factors", "n_validation": len(validation),
                             "interval_log_likelihood": log_likelihood,
                             "mean_log_likelihood": log_likelihood / max(len(validation), 1),
                             "first_detection_auc": auc, "first_detection_brier": brier,
                             "valid": True})
            except Exception as exc:
                rows.append({"fold": fold, "distribution": distribution,
                             "feature_model": "all_factors", "n_validation": len(validation),
                             "interval_log_likelihood": np.nan, "mean_log_likelihood": np.nan,
                             "first_detection_auc": np.nan, "first_detection_brier": np.nan,
                             "valid": False, "error": str(exc)})
    return pd.DataFrame(rows), pd.DataFrame(calibration_rows)


def estimate_technical_time_sd(events: pd.DataFrame) -> float:
    slopes = []
    for _, group in events.sort_values(["mother_id", "week"]).groupby("mother_id"):
        weeks = group["week"].to_numpy(float)
        values = group["sqrt_y"].to_numpy(float)
        delta_week = np.diff(weeks)
        delta_value = np.diff(values)
        valid = np.isfinite(delta_week) & np.isfinite(delta_value) & (delta_week > 0)
        slopes.extend(np.abs(delta_value[valid] / delta_week[valid]))
    positive = np.asarray(slopes)[np.asarray(slopes) > 1e-8]
    slope = float(np.median(positive)) if len(positive) else 0.02
    return float(np.sqrt(Q1_S2_TECH) / slope)


def write_summary(args, events, subject, mixed_name, mixed_table, aft_name, aft_dist, aft_table,
                  payload, mixed_fit, stats, cv_table, calibration_table):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mixed_table.to_csv(args.output_dir / "q3_mixed_candidates.csv", index=False, encoding="utf-8-sig")
    aft_table.to_csv(args.output_dir / "q3_aft_candidates.csv", index=False, encoding="utf-8-sig")
    fixed = pd.DataFrame({"term": mixed_fit["result"].fe_params.index,
                          "estimate": np.asarray(mixed_fit["result"].fe_params),
                          "std_error": np.asarray(mixed_fit["result"].bse_fe)})
    fixed["z_value"] = fixed["estimate"] / fixed["std_error"]
    fixed["p_value"] = 2 * st.norm.sf(np.abs(fixed["z_value"]))
    fixed.to_csv(args.output_dir / "q3_mixed_fixed_effects.csv", index=False, encoding="utf-8-sig")
    uncertainty = parameter_uncertainty(payload, args.bootstrap, subject=subject)
    uncertainty.to_csv(args.output_dir / "q3_uncertainty_summary.csv", index=False, encoding="utf-8-sig")
    cv_table.to_csv(args.output_dir / "q3_aft_grouped_cv.csv", index=False, encoding="utf-8-sig")
    calibration_table.to_csv(args.output_dir / "q3_aft_calibration.csv", index=False, encoding="utf-8-sig")
    measurement_rows = []
    aft_prob_fn = prob_fn_from_payload(payload)
    technical_time_sd = estimate_technical_time_sd(events)
    subject_groups = subject.assign(
        bmi_group=pd.qcut(subject["bmi"], q=4, duplicates="drop")
    )
    for bmi_group, group in subject_groups.groupby("bmi_group", observed=True):
        for week in (12.0, 14.0, 16.0):
            for label in ("remove_technical", "baseline", "double_technical"):
                probabilities = []
                for _, row in group.iterrows():
                    base_probability = float(aft_prob_fn(
                        week, row.bmi, row.height, row.weight, row.age
                    ))
                    if label == "remove_technical":
                        probabilities.append(base_probability)
                    else:
                        jitter = technical_time_sd * (2.0 if label == "double_technical" else 1.0)
                        shifted = np.array([max(week - jitter, WEEK_MIN), week + jitter])
                        probabilities.append(float(np.mean(aft_prob_fn(
                            shifted, row.bmi, row.height, row.weight, row.age
                        ))))
                measurement_rows.append({"bmi_group": str(bmi_group), "bmi_mean": group["bmi"].mean(),
                    "n": len(group), "week": week, "scenario": label,
                    "qualified_probability": float(np.mean(probabilities))})
    pd.DataFrame(measurement_rows).to_csv(
        args.output_dir / "q3_measurement_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    meta = {
        "source": str(args.data), "event_count": int(len(events)),
        "mother_count": int(len(subject)), "mixed_selected": mixed_name,
        "mixed_formula": mixed_fit["formula"], "aft_selected": aft_name,
        "aft_distribution": aft_dist, "feature_stats": stats,
        "parameter_mc_draws": args.bootstrap, "technical_variance": Q1_S2_TECH,
        "technical_time_sd": technical_time_sd,
        "aft_selection": "all_factors distribution selected by grouped 5-fold mean interval log-likelihood",
        "aft_cv": "grouped 5-fold; all_factors + all candidate distributions",
    }
    (args.output_dir / "q3_model_summary.txt").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n\nAFT候选比较：\n" +
        aft_table.to_string(index=False) + "\n\n混合模型候选比较：\n" + mixed_table.to_string(index=False),
        encoding="utf-8"
    )
    (args.output_dir / "q3_model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events, subject = load_events(args.data)
    stats = standardize(events, subject)
    mixed_name, mixed_fits, mixed_table = fit_mixed_candidates(events, subject)
    aft_name, aft_dist, aft_fit, aft_table, specs, cv_table, calibration_table = fit_aft_candidates(subject)
    payload = export_aft(aft_name, aft_dist, aft_fit, specs, subject, stats, args.output_dir / "q3_model.npy")
    mixed_fit = mixed_fits[mixed_name]
    write_summary(args, events, subject, mixed_name, mixed_table, aft_name, aft_dist,
                  aft_table, payload, mixed_fit, stats, cv_table, calibration_table)
    print(f"问题三一号模型完成：{len(subject)} 位孕妇、{len(events)} 个抽血事件")
    print(f"混合模型最优：{mixed_name}; AFT最优：{aft_name}+{aft_dist}")
    print(f"概率接口模型已保存：{args.output_dir / 'q3_model.npy'}")


if __name__ == "__main__":
    main()
