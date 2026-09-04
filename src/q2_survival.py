# -*- coding: utf-8 -*-
"""
q2_survival.py —— 甲：问题二任务② 区间删失生存分析

为什么必须做这一步：
  问题一给出的是 P(Y(t) >= 4%)，即"在孕周 t 这一次抽血能测到达标"的
  瞬时概率，用的是纵向回归模型。本模块给出另一条独立路径——把"首次达标
  孕周 T"当作**区间删失的生存时间**直接建模，得到 P(T <= t | BMI)。

  两条路径的口径本来就不同（见 §口径差异检查），互相印证才有说服力：
    - 问题一路径：参数化纵向模型，靠 sqrt(Y) 的均值-方差结构外推概率；
    - 本模块路径：不对 Y 的轨迹形状作假设，只用"到哪一周为止达标了没有"
      这个事实，对左/区间/右删失做正确的似然处理。

删失结构（来自丙的 clean.py，事件级，按孕妇聚合后 n=267）：
    left  217 (81.3%)  首次抽血即达标 -> T <= t_right，下界取 0
    none   43 (16.1%)  观测到 未达标->达标 的转变 -> t_left < T <= t_right
    right   7 ( 2.6%)  全程未达标 -> T > t_left
  81% 是左删失这一点非常关键：数据里"首次达标孕周中位数 12.86 周"其实
  是"首次抽血孕周"的中位数，不是生物学上的达标时刻。任何直接把
  first_pass_week 当观测值做回归的做法都会系统性高估达标时间。

输出：
    outputs/q2_survival.txt   完整日志
    outputs/q2_survival.npy   选定模型的参数与预测网格，供 q2_risk 调用

用法（把本模块的概率函数接进风险模型，做两条路径的交叉验证）：
    from q2_risk import ExpectedRisk
    from q2_survival import load_prob_fn
    er = ExpectedRisk(prob_fn=load_prob_fn())
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
EVENT = ROOT / 'data' / 'processed' / 'male_clean_event.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEP = '=' * 78

# 左删失样本的达标时间下界。0 表示"不作任何生物学假设"；
# 敏感性分析里会换成 8 周（临床上 cfDNA 可检出的最早孕周附近）看结论是否变化。
LEFT_LOWER_DEFAULT = 0.0

LOCATION_PARAM = {
    'Weibull': 'lambda_',
    'LogNormal': 'mu_',
    'LogLogistic': 'alpha_',
}


class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, 'w', encoding='utf-8', newline='\n')

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def load_mother_level():
    d = pd.read_csv(EVENT, encoding='utf-8-sig').sort_values(['mother_id', 'week_mean'])
    m = d.groupby('mother_id').agg(
        t_left=('t_left', 'first'), t_right=('t_right', 'first'),
        censored=('censored', 'first'), bmi=('bmi', 'first'),
        age=('age', 'first'), height=('height', 'first'), weight=('weight', 'first'),
    ).reset_index()
    return d, m


def build_bounds(m, left_lower=LEFT_LOWER_DEFAULT):
    """转成 lifelines 的区间删失格式：[lower, upper]，右删失 upper=inf。"""
    lo = m.t_left.astype(float).copy()
    hi = m.t_right.astype(float).copy()
    lo[m.censored == 'left'] = left_lower
    hi[m.censored == 'right'] = np.inf
    return lo.values, hi.values


def fit_afts(df, bmi_col='bmi_c'):
    """拟合三种参数分布的 AFT 模型，按 AIC 比较。"""
    from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
    cands = {
        'Weibull': WeibullAFTFitter(),
        'LogNormal': LogNormalAFTFitter(),
        'LogLogistic': LogLogisticAFTFitter(),
    }
    fits, rec = {}, []
    for name, f in cands.items():
        try:
            # LogNormal 在数值实现中会对下界取 log；以极小正数表达理论下界0。
            fit_df = df.copy()
            fit_df['lo'] = fit_df.lo.clip(lower=1e-6)
            f.fit_interval_censoring(fit_df, lower_bound_col='lo', upper_bound_col='hi',
                                     show_progress=False)
            fits[name] = f
            coef = f.params_.get((LOCATION_PARAM[name], bmi_col), np.nan)
            rec.append([name, f.log_likelihood_, f.AIC_, coef])
        except Exception as e:
            rec.append([name, np.nan, np.nan, np.nan])
            print(f'  {name} 拟合失败: {type(e).__name__}: {e}')
    tab = pd.DataFrame(rec, columns=['分布', 'logLik', 'AIC', f'{bmi_col}系数'])
    return fits, tab


def main():
    sys.stdout = _Tee(sys.stdout, OUT / 'q2_survival.txt')
    d, m = load_mother_level()

    print(SEP)
    print('【任务②】区间删失生存分析：首次达标孕周 T ~ BMI')
    print(f'孕妇数 {len(m)}；删失结构 {m.censored.value_counts().to_dict()}')
    print(f'左删失占比 {(m.censored=="left").mean()*100:.1f}% —— 数据里的"首次达标孕周"')
    print('主要反映的是"首次抽血时间"，不是真实达标时刻，必须按删失处理。')

    # ---------- 口径差异检查：达标后会不会掉回 4% 以下 ----------
    print('\n' + SEP + '\n【口径差异检查】达标之后是否会掉回 4% 以下？')
    drop_back, n_after = 0, 0
    for mid, g in d.groupby('mother_id'):
        q = g.qual_mean.values
        if q.max() == 0:
            continue
        first = int(np.argmax(q == 1))
        after = q[first + 1:]
        n_after += len(after)
        drop_back += int((after == 0).sum())
    print(f'首次达标之后的检测共 {n_after} 次，其中掉回 4% 以下 {drop_back} 次'
          f'（{drop_back/max(n_after,1)*100:.1f}%）')
    if drop_back / max(n_after, 1) < 0.05:
        print('-> 掉回比例很低，"一旦达标基本保持达标"，生存模型的 P(T<=t) 与问题一的')
        print('   P(Y(t)>=4%) 可以互相印证；差异主要来自模型假设而非现象本身。')
    else:
        print('-> 掉回比例不低，两个口径不等价：生存模型的 P(T<=t) 会系统性高于')
        print('   "这一次抽血能测到达标"的概率。风险模型应以问题一的瞬时口径为准，')
        print('   本模块作为稳健性对照。')

    # ---------- 非参数：Turnbull 估计 ----------
    print('\n' + SEP + '\n【非参数估计】Turnbull（区间删失版 Kaplan-Meier）')
    from lifelines import KaplanMeierFitter
    lo, hi = build_bounds(m)
    kmf = KaplanMeierFitter()
    try:
        kmf.fit_interval_censoring(lo, hi)
        sf = kmf.survival_function_
        for t in [11, 12, 13, 14, 16, 20, 25]:
            idx = sf.index.get_indexer([t], method='nearest')[0]
            val = float(sf.iloc[idx].mean())
            print(f'  P(T <= {t:>2}周) ≈ {1-val:.3f}')
    except Exception as e:
        print(f'  Turnbull 估计失败（{type(e).__name__}），跳过，改用参数化模型：{e}')

    # ---------- 参数化 AFT ----------
    print('\n' + SEP + '\n【参数化 AFT 模型】T ~ BMI（区间删失似然）')
    MU_B = float(m.bmi.mean())
    df = pd.DataFrame({'lo': lo, 'hi': hi, 'bmi_c': m.bmi.values - MU_B})
    fits, tab = fit_afts(df)
    print(tab.round(4).to_string(index=False))
    if not fits:
        print('全部分布拟合失败，任务②无法给出参数化结果')
        return
    best_name = tab.dropna(subset=['AIC']).sort_values('AIC').iloc[0]['分布']
    best = fits[best_name]
    print(f'\n-> AIC 最优分布：{best_name}')
    print(best.summary[['coef', 'se(coef)', 'p', 'coef lower 95%', 'coef upper 95%']]
          .round(5).to_string())

    # ---------- 由生存模型导出达标概率 ----------
    print('\n' + SEP + '\n【与问题一路径对照】P(T<=t | BMI)  vs  问题一的 P(Y(t)>=4%)')
    from q2_risk import ExpectedRisk
    er_q1 = ExpectedRisk()
    grid_b = [22, 26, 30, 34, 38, 42]
    grid_t = [12, 14, 16, 20]
    print('  BMI  孕周 | 生存模型P(T<=t) | 问题一P(Y>=4%) | 差值')
    for b in grid_b:
        X = pd.DataFrame({'bmi_c': [b - MU_B]})
        for t in grid_t:
            s = float(best.predict_survival_function(X, times=[t]).values[0][0])
            p_surv = 1 - s
            p_q1 = float(np.atleast_1d(er_q1.prob_qualified(t, b))[0])
            print(f'  {b:>3}  {t:>3}  |     {p_surv:.3f}       |     {p_q1:.3f}      | {p_surv-p_q1:+.3f}')

    # ---------- 敏感性：左删失下界的假设 ----------
    print('\n' + SEP + '\n【敏感性】左删失下界取 0 vs 取 8 周')
    for lower in [0.0, 8.0]:
        lo2, hi2 = build_bounds(m, left_lower=lower)
        df2 = pd.DataFrame({'lo': lo2, 'hi': hi2, 'bmi_c': m.bmi.values - MU_B})
        f2, tab2 = fit_afts(df2)
        if best_name in f2:
            c = f2[best_name].params_.get((LOCATION_PARAM[best_name], 'bmi_c'), np.nan)
            aic = f2[best_name].AIC_
            print(f'  下界={lower:>4.1f}周: {best_name} 的 bmi_c 系数={c:+.5f}，AIC={aic:.2f}')

    # ---------- 导出供风险模型调用 ----------
    grid_week = np.arange(11.0, 25.0 + 1e-9, 0.05)
    grid_bmi = np.arange(20.0, 47.0 + 1e-9, 0.5)
    surv_mat = np.zeros((len(grid_bmi), len(grid_week)))
    for i, b in enumerate(grid_bmi):
        X = pd.DataFrame({'bmi_c': [b - MU_B]})
        s = best.predict_survival_function(X, times=grid_week).values[:, 0]
        surv_mat[i] = 1.0 - s
    np.save(OUT / 'q2_survival.npy',
            dict(model=best_name, mu_b=MU_B, grid_week=grid_week,
                 grid_bmi=grid_bmi, prob_matrix=surv_mat,
                 param_labels=list(best.params_.index),
                 param_values=best.params_.to_numpy(float),
                 param_cov=best.variance_matrix_.to_numpy(float)),
            allow_pickle=True)
    print(f'\n已保存 outputs/q2_survival.npy（{best_name} 模型的 P(T<=t|BMI) 网格）')
    print('乙做交叉验证时：ExpectedRisk(prob_fn=load_prob_fn()) 即可切换到这条路径。')


def load_prob_fn():
    """返回连续二维插值的达标概率函数。

    ``prev_week`` 不为空时返回在此前检测未达标条件下，本次首次达标的概率：
    P(prev<T<=week | T>prev)。风险递归必须使用这个条件概率，避免把累计分布
    F(week) 在多次复检中重复计算。
    """
    path = OUT / 'q2_survival.npy'
    if not path.exists():
        raise SystemExit('未找到 outputs/q2_survival.npy，请先运行 python src/q2_survival.py')
    S = np.load(path, allow_pickle=True).item()

    interp = RegularGridInterpolator(
        (S['grid_bmi'], S['grid_week']), S['prob_matrix'],
        bounds_error=False, fill_value=None,
    )

    def cdf(week, bmi_baseline):
        wk, bmi = np.broadcast_arrays(np.asarray(week, float),
                                      np.asarray(bmi_baseline, float))
        wk = np.clip(wk, S['grid_week'][0], S['grid_week'][-1])
        bmi = np.clip(bmi, S['grid_bmi'][0], S['grid_bmi'][-1])
        ans = interp(np.column_stack([bmi.ravel(), wk.ravel()])).reshape(wk.shape)
        return float(ans) if ans.ndim == 0 else ans

    def prob_fn(week, bmi_baseline, thr=0.04, prev_week=None):
        if not np.isclose(thr, 0.04):
            raise ValueError('生存模型仅针对题面4%阈值拟合，不能更换thr。')
        now = np.asarray(cdf(week, bmi_baseline), float)
        if prev_week is None:
            return float(now) if now.ndim == 0 else now
        prev = np.asarray(cdf(prev_week, bmi_baseline), float)
        conditional = np.clip((now - prev) / np.maximum(1.0 - prev, 1e-12), 0.0, 1.0)
        return float(conditional) if conditional.ndim == 0 else conditional

    return prob_fn


if __name__ == '__main__':
    main()
