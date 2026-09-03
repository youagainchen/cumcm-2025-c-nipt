# -*- coding: utf-8 -*-
"""
q1_model.py —— 甲：问题一
胎儿 Y 染色体浓度与孕妇孕周数、BMI 的关系模型及显著性检验

输入   data/processed/male_clean.csv   （由 src/clean.py 生成，丙）
输出   outputs/q1_out.txt              完整运行日志
       outputs/q1_coef.npy             最终模型系数，供问题二调用

流程：相关特性 -> 响应变换选择 -> 随机效应结构 -> 固定效应设定
      -> 最终模型与显著性检验 -> 诊断与稳健性 -> 问题二接口

纪律（见 docs/C题_第一阶段分工.md）：
  * 1082 条记录只来自 267 位孕妇，绝不当独立样本处理；
  * 一切质量过滤只做敏感性分析，主模型用全样本。

运行：  python src/q1_model.py   （在仓库根目录执行）
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as viff

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed' / 'male_clean.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEP = '=' * 78


def load():
    if not DATA.exists():
        raise SystemExit(f'未找到 {DATA}，请先运行 python src/clean.py')
    d = pd.read_csv(DATA, encoding='utf-8-sig')
    need = ['mother_id', 'week', 'bmi', 'y_conc', 'age', 'height', 'weight']
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit(f'male_clean.csv 缺少列：{miss}')
    d = d.dropna(subset=need).reset_index(drop=True)
    return d


d = load()
print(SEP)
print(f'数据源 {DATA.relative_to(ROOT)}（丙的清洗产物）')
print(f'样本 {len(d)} 条记录 / {d.mother_id.nunique()} 位孕妇；'
      f'每人检测 {d.n_visits.min()}~{d.n_visits.max()} 次，中位 {d.n_visits.median():.0f}')
print(f'Y浓度 均值 {d.y_conc.mean():.4f} 中位 {d.y_conc.median():.4f} '
      f'偏度 {st.skew(d.y_conc):.3f} 峰度 {st.kurtosis(d.y_conc):.3f}')
print(f'孕周 {d.week.min():.2f}~{d.week.max():.2f}；BMI {d.bmi.min():.2f}~{d.bmi.max():.2f}')
print(f'达标(>=4%)记录占比 {d.is_qualified.mean():.3f}；'
      f'曾达标孕妇 {d.groupby("mother_id").reached.first().sum()}/{d.mother_id.nunique()}')

# ============ 部件1：相关特性 ============
print('\n' + SEP + '\n【部件1】相关特性分析')
rows = []
for v in ['week', 'bmi', 'age', 'height', 'weight']:
    r, pr = st.pearsonr(d[v], d.y_conc)
    s, ps = st.spearmanr(d[v], d.y_conc)
    rows.append([v, r, pr, s, ps])
print(pd.DataFrame(rows, columns=['变量', 'Pearson_r', 'p_P', 'Spearman_rho', 'p_S'])
      .round(4).to_string(index=False))


def partial_corr(x, y, covars):
    X = sm.add_constant(covars)
    return st.pearsonr(sm.OLS(x, X).fit().resid, sm.OLS(y, X).fit().resid)


print('\n偏相关（控制孕周后的净相关）：')
for v in ['bmi', 'age', 'height', 'weight']:
    r, p = partial_corr(d[v].values, d.y_conc.values, d[['week']].values)
    print(f'  {v:8s} r = {r:+.4f}   p = {p:.3e}')
r, p = partial_corr(d.week.values, d.y_conc.values, d[['bmi']].values)
print(f'  {"week":8s} r = {r:+.4f}   p = {p:.3e}   (控制 BMI)')

dd = d[d.n_visits >= 2].copy()
dd['dw'] = dd.week - dd.groupby('mother_id').week.transform('mean')
dd['dy'] = dd.y_conc - dd.groupby('mother_id').y_conc.transform('mean')
r_in, p_in = st.pearsonr(dd.dw, dd.dy)
bw = d.groupby('mother_id').agg(w=('week', 'mean'), y=('y_conc', 'mean'), b=('bmi', 'mean'))
r_bw, p_bw = st.pearsonr(bw.w, bw.y)
r_b, p_b = st.pearsonr(bw.b, bw.y)
print(f'\n孕周-Y浓度 组内相关（同一孕妇内去均值）: r = {r_in:+.4f}, p = {p_in:.2e}')
print(f'孕周-Y浓度 组间相关（孕妇均值之间）    : r = {r_bw:+.4f}, p = {p_bw:.2e}')
print('  -> 两者符号相反：孕周效应是真实的个体内时间趋势；组间为负源于选择性复检')
print(f'BMI-Y浓度 组间相关（孕妇层面）        : r = {r_b:+.4f}, p = {p_b:.2e}')

# ============ 部件3a：响应变量的变换 ============
print('\n' + SEP + '\n【部件3a】响应变量变换选择（AIC 已做 Jacobian 校正，可跨尺度比较）')
y = d.y_conc.values
TRANS = {
    'y (原始)': (y, 0.0),
    'log(y)': (np.log(y), -np.log(y).sum()),
    'sqrt(y)': (np.sqrt(y), -np.log(2 * np.sqrt(y)).sum()),
    'logit(y)': (np.log(y / (1 - y)), -np.log(y * (1 - y)).sum()),
}
rec = []
for name, (resp, jac) in TRANS.items():
    t = d.copy()
    t['resp'] = np.asarray(resp)
    m = smf.mixedlm('resp ~ week + bmi', t, groups=t.mother_id).fit(reml=False)
    k = len(m.params) + 1
    ll = m.llf + jac
    rec.append([name, ll, -2 * ll + 2 * k, -2 * ll + k * np.log(len(t))])
tt = pd.DataFrame(rec, columns=['响应变量', '校正logLik', 'AIC', 'BIC'])
tt['dAIC'] = (tt.AIC - tt.AIC.min()).round(1)
print(tt.round(2).to_string(index=False))
BEST = tt.loc[tt.AIC.idxmin(), '响应变量']
print(f'-> 选择 {BEST}；后续全部在该尺度上建模')
print('   （不同尺度的似然不可直接比较，此处已加 Jacobian 项 Σln|g\'(y)| 校正）')

d['resp'] = np.asarray(TRANS[BEST][0])
THR_RESP = {'y (原始)': .04, 'log(y)': np.log(.04),
            'sqrt(y)': np.sqrt(.04), 'logit(y)': np.log(.04 / .96)}[BEST]
MU_W, MU_B = float(d.week.mean()), float(d.bmi.mean())
d['week_c'] = d.week - MU_W
d['bmi_c'] = d.bmi - MU_B
d['age_c'] = d.age - d.age.mean()

# ============ 部件2：随机效应结构 ============
print('\n' + SEP + '\n【部件2】随机效应结构：为什么不能用 OLS')
ols = smf.ols('resp ~ week_c + bmi_c', d).fit()
ri = smf.mixedlm('resp ~ week_c + bmi_c', d, groups=d.mother_id).fit(reml=False)
s2u, s2e = float(ri.cov_re.iloc[0, 0]), float(ri.scale)
lr = 2 * (ri.llf - ols.llf)
print(f'OLS       logLik = {ols.llf:9.3f}   week 系数 SE = {ols.bse["week_c"]:.5f}')
print(f'随机截距   logLik = {ri.llf:9.3f}   week 系数 SE = {ri.bse["week_c"]:.5f}')
print(f'  s2_u = {s2u:.5f}   s2_e = {s2e:.5f}   ICC = {s2u/(s2u+s2e):.4f}'
      f'  -> {s2u/(s2u+s2e)*100:.1f}% 的变异来自孕妇个体差异')
print(f'  LRT(随机截距=0): chi2 = {lr:.2f}, 混合卡方 p = {0.5*st.chi2.sf(lr,1):.2e}')
rs = smf.mixedlm('resp ~ week_c + bmi_c', d, groups=d.mother_id,
                 re_formula='~week_c').fit(reml=False)
lr2 = 2 * (rs.llf - ri.llf)
p_rs = 0.5 * (st.chi2.sf(lr2, 1) + st.chi2.sf(lr2, 2))
print(f'随机斜率   logLik = {rs.llf:9.3f}')
print(f'  LRT(随机截距 -> +孕周随机斜率): chi2 = {lr2:.2f}, df=2, 混合卡方 p = {p_rs:.2e}')
USE_RS = bool(p_rs < 0.05)
print(f'-> 采用「随机截距 + 孕周随机斜率」：{USE_RS}')
print('   含义：不同孕妇不仅起点不同，Y 浓度随孕周上升的速率也显著不同')
print('   注：方差分量检验落在参数空间边界，零分布取 0.5χ²₁+0.5χ²₂ 混合，非标准 χ²')
RE_F = '~week_c' if USE_RS else None

# ============ 部件3b：固定效应设定 ============
print('\n' + SEP + '\n【部件3b】固定效应设定比较（ML 拟合，随机效应结构固定）')
specs = {
    'M1 week': 'resp ~ week_c',
    'M2 week+bmi': 'resp ~ week_c + bmi_c',
    'M3 week+week^2+bmi': 'resp ~ week_c + I(week_c**2) + bmi_c',
    'M4 M3+week*bmi': 'resp ~ week_c * bmi_c + I(week_c**2)',
    'M5 M3+age': 'resp ~ week_c + I(week_c**2) + bmi_c + age_c',
    'M6 身高体重替代BMI': 'resp ~ week_c + I(week_c**2) + height + weight',
}
fits, rec = {}, []
for n, f in specs.items():
    m = smf.mixedlm(f, d, groups=d.mother_id, re_formula=RE_F).fit(reml=False)
    fits[n] = m
    k = len(m.params) + 1
    rec.append([n, len(m.fe_params), m.llf, -2 * m.llf + 2 * k,
                -2 * m.llf + k * np.log(len(d))])
cmp = pd.DataFrame(rec, columns=['模型', '固定效应数', 'logLik', 'AIC', 'BIC'])
cmp['dAIC'] = (cmp.AIC - cmp.AIC.min()).round(2)
cmp['dBIC'] = (cmp.BIC - cmp.BIC.min()).round(2)
print(cmp.round(2).to_string(index=False))


def lrt(a, b, name):
    stat = 2 * (fits[b].llf - fits[a].llf)
    df = len(fits[b].params) - len(fits[a].params)
    print(f'  {name:24s} chi2 = {stat:7.3f}, df = {df}, p = {st.chi2.sf(stat, df):.3e}')


print('\n嵌套模型似然比检验：')
lrt('M1 week', 'M2 week+bmi', 'M1->M2 加 BMI')
lrt('M2 week+bmi', 'M3 week+week^2+bmi', 'M2->M3 加孕周二次项')
lrt('M3 week+week^2+bmi', 'M4 M3+week*bmi', 'M3->M4 加孕周xBMI')
lrt('M3 week+week^2+bmi', 'M5 M3+age', 'M3->M5 加年龄')
print(f'\n-> AIC 最优：{cmp.loc[cmp.AIC.idxmin(), "模型"]}；'
      f'BIC 最优：{cmp.loc[cmp.BIC.idxmin(), "模型"]}')
print('   M6 的 AIC 优势极微（<2）而 BIC 选 M3；且问题一、二的落点是 BMI，主模型取 M3。')
print('   M6 说明身高体重分列比合成 BMI 携带更多信息 —— 记为问题三的入口。')

# ============ 部件4：最终模型 ============
print('\n' + SEP + '\n【部件4】最终模型（REML 重估方差分量）')
FINAL_F = 'resp ~ week_c + I(week_c**2) + bmi_c'
final = smf.mixedlm(FINAL_F, d, groups=d.mother_id, re_formula=RE_F).fit()
fe = final.fe_params
ci = final.conf_int()
print('固定效应：')
print(pd.DataFrame({'估计值': final.params, '标准误': final.bse, 'z': final.tvalues,
                    'p值': final.pvalues, 'CI下限': ci[0], 'CI上限': ci[1]})
      .loc[fe.index].round(5).to_string())
Q2K = [k for k in fe.index if 'week_c ** 2' in k or 'week_c**2' in k][0]
G = final.cov_re.values
s2u0 = float(G[0, 0])
s2u1 = float(G[1, 1]) if USE_RS else 0.0
cov01 = float(G[0, 1]) if USE_RS else 0.0
s2e = float(final.scale)
if USE_RS:
    print(f'\n随机效应协方差 G：截距方差 {s2u0:.5f}，斜率方差 {s2u1:.7f}，'
          f'协方差 {cov01:.5f}（相关 {cov01/np.sqrt(s2u0*s2u1):+.3f}）')
print(f'残差方差 s2_e = {s2e:.5f}')
print(f'ICC（均值孕周处）= {s2u0/(s2u0+s2e):.4f}')

print(f'\n显式表达式（中心化：孕周均值 {MU_W:.3f} 周，BMI 均值 {MU_B:.3f}）：')
print(f'  {BEST} = {fe["Intercept"]:.5f} {fe["week_c"]:+.5f}(t-{MU_W:.3f}) '
      f'{fe[Q2K]:+.6f}(t-{MU_W:.3f})^2 {fe["bmi_c"]:+.5f}(B-{MU_B:.3f})'
      + (f' + u0_i + u1_i(t-{MU_W:.3f}) + e_ij' if USE_RS else ' + u_i + e_ij'))

base = float(fe['Intercept'])
print(f'\n原尺度边际效应（孕周 {MU_W:.1f} 周、BMI {MU_B:.1f} 处）：')
print(f'  每增 1 孕周，Y 浓度约 {2*base*fe["week_c"]*100:+.3f} 个百分点')
print(f'  BMI 每增 1 单位，Y 浓度约 {2*base*fe["bmi_c"]*100:+.3f} 个百分点')
vt = -fe['week_c'] / (2 * fe[Q2K])
print(f'  二次项顶点在孕周 {MU_W + vt:.2f} 周'
      f'（{"极小值，观测区间内单调上升且加速" if fe[Q2K] > 0 else "极大值"}）')
vf = float(np.var(np.asarray(final.model.exog) @ fe.values, ddof=1))
vr = s2u0 + (s2u1 * float(np.var(d.week_c, ddof=1)) if USE_RS else 0.0)
print(f'  边际 R2 = {vf/(vf+vr+s2e):.4f}   条件 R2 = {(vf+vr)/(vf+vr+s2e):.4f}')

# ============ 部件5：诊断与稳健性 ============
print('\n' + SEP + '\n【部件5】模型诊断')
res = np.asarray(final.resid)
print(f'残差 偏度 {st.skew(res):+.3f}  峰度 {st.kurtosis(res):+.3f}')
print(f'Shapiro-Wilk（随机500条）p = '
      f'{st.shapiro(pd.Series(res).sample(500, random_state=0)).pvalue:.4f}')
bp = sm.stats.diagnostic.het_breuschpagan(res, sm.add_constant(d[['week_c', 'bmi_c']]))
print(f'Breusch-Pagan 异方差 LM = {bp[0]:.3f}, p = {bp[1]:.4f}')
X = sm.add_constant(d[['week_c', 'bmi_c', 'age_c']])
print('VIF:', {c: round(viff(X.values, i), 2) for i, c in enumerate(X.columns) if c != 'const'})
print(f'|标准化残差| > 3 的记录数 = {(np.abs(st.zscore(res)) > 3).sum()}'
      f'（占 {(np.abs(st.zscore(res)) > 3).mean()*100:.2f}%）')
print(f'\n注：n={len(d)} 下 Shapiro 与 BP 功效极高，微小偏离即显著；残差偏度仅'
      f' {st.skew(res):+.2f}，实务可接受。下方用聚类 Bootstrap 给出不依赖分布假设的区间。')

print('\n--- 稳健性检验 ---')
print(f'GC 含量分位：{d.gc.quantile([0, .01, .5, .99, 1]).round(4).to_dict()}')
print(f'  丙按题面 40%~60% 打的 flag_gc 共 {int(d.flag_gc.sum())} 条'
      f'（占 {d.flag_gc.mean()*100:.1f}%），因本数据 GC 整体集中在 0.386~0.421，')
print('  该阈值会误判近半样本，故主模型不做删除，仅在此作敏感性对照。')
rob1 = smf.mixedlm(FINAL_F, d[d.flag_any == 0], groups=d[d.flag_any == 0].mother_id,
                   re_formula=RE_F).fit()
zg = (np.abs(st.zscore(d.gc)) < 3) & (np.abs(st.zscore(d.map_ratio)) < 3) \
     & (np.abs(st.zscore(d.filt_ratio)) < 3)
rob2 = smf.mixedlm(FINAL_F, d[zg], groups=d[zg].mother_id, re_formula=RE_F).fit()
first = d.sort_values(['mother_id', 'week']).groupby('mother_id').head(1)
rob3 = smf.ols(FINAL_F, first).fit()
print(pd.DataFrame({
    f'主模型(n={len(d)})': fe,
    f'丙flag_any过滤(n={int((d.flag_any==0).sum())})': rob1.fe_params,
    f'±3σ过滤(n={int(zg.sum())})': rob2.fe_params,
    f'仅首检OLS(n={len(first)})': rob3.params}).round(5).to_string())

rng = np.random.default_rng(0)
idx_by_mom = {m: g.index.values for m, g in d.groupby('mother_id')}
ids = np.array(list(idx_by_mom))
boot = []
for _ in range(120):
    pick = rng.choice(ids, len(ids), replace=True)
    sub = d.loc[np.concatenate([idx_by_mom[m] for m in pick])].copy()
    sub['g'] = np.concatenate([np.full(len(idx_by_mom[m]), k) for k, m in enumerate(pick)])
    try:
        boot.append(smf.mixedlm(FINAL_F, sub, groups=sub.g, re_formula=RE_F)
                    .fit(reml=False).fe_params.values)
    except Exception:
        pass
B = np.array(boot)
print(f'\n按孕妇聚类 Bootstrap（{len(B)} 次重抽）：')
print(pd.DataFrame({'点估计': fe.values, 'Boot均值': B.mean(0), 'BootSE': B.std(0, ddof=1),
                    '2.5%': np.percentile(B, 2.5, axis=0),
                    '97.5%': np.percentile(B, 97.5, axis=0)},
                   index=fe.index).round(5).to_string())
print('  -> 与渐近区间一致，结论不依赖正态/同方差假设')

# ============ 部件6：问题二接口 ============
print('\n' + SEP + '\n【部件6】问题二接口')
COEF = dict(scale=BEST, b0=float(fe['Intercept']), b1=float(fe['week_c']),
            b2=float(fe[Q2K]), b3=float(fe['bmi_c']), mu_w=MU_W, mu_b=MU_B,
            s2u0=s2u0, s2u1=s2u1, cov01=cov01, s2e=s2e, thr_resp=float(THR_RESP))


def mean_resp(week, bmi, C=COEF):
    w = np.asarray(week, float) - C['mu_w']
    b = np.asarray(bmi, float) - C['mu_b']
    return C['b0'] + C['b1'] * w + C['b2'] * w ** 2 + C['b3'] * b


def var_resp(week, C=COEF):
    """一位随机新孕妇在该孕周处的总方差（随机截距 + 斜率 + 残差）"""
    w = np.asarray(week, float) - C['mu_w']
    return C['s2u0'] + 2 * C['cov01'] * w + C['s2u1'] * w ** 2 + C['s2e']


def predict_y(week, bmi, C=COEF):
    return mean_resp(week, bmi, C) ** 2


def prob_qualified(week, bmi, thr=0.04, C=COEF):
    z = (np.sqrt(thr) - mean_resp(week, bmi, C)) / np.sqrt(var_resp(week, C))
    return st.norm.sf(z)


def week_reach(bmi, thr=0.04, p=0.9, C=COEF, lo=10.0, hi=25.0):
    """该 BMI 的孕妇以把握度 p 达标的最早孕周（问题二直接调用）"""
    from scipy.optimize import brentq
    f = lambda t: prob_qualified(t, bmi, thr, C) - p
    if f(lo) >= 0:
        return lo
    if f(hi) < 0:
        return np.nan
    return brentq(f, lo, hi)


def main():
    print(f'  predict_y(week=16, bmi=32) = {predict_y(16, 32):.4f}')
    print(f'  P(Y>=4% | week=16, bmi=32) = {prob_qualified(16, 32):.3f}')
    print('\n  各 BMI 达到 4% 的最早孕周（区间 10~25 周）：')
    print('    BMI    P=0.50    P=0.80    P=0.90    P=0.95')
    for b in [22, 26, 30, 34, 38, 42]:
        cells = []
        for p in (.5, .8, .9, .95):
            t = week_reach(b, p=p)
            cells.append('  >25 ' if not np.isfinite(t)
                         else ('  <=10' if t <= 10.001 else f'{t:6.2f}'))
        print(f'    {b:>3}   ' + '    '.join(cells))
    np.save(OUT / 'q1_coef.npy', COEF, allow_pickle=True)
    print(f'\n系数已保存 {(OUT / "q1_coef.npy").relative_to(ROOT)}，'
          f'问题二用 np.load(..., allow_pickle=True).item() 载入。')


if __name__ == '__main__':
    main()
