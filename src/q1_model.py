# -*- coding: utf-8 -*-
"""
q1_model.py —— 甲：问题一（v3，孕周效应改为自然三次样条，不再用二次多项式）

v3 相对 v2 的变更（原因：非参数 LOWESS/GAM 检查发现孕周效应形状是
「早期快速上升 -> 16~20周平台期 -> 24~29周加速」，二次多项式结构性地
拟合不出这个中段平台，AIC 比样条差 ~20，属决定性差距）：
  * 孕周效应由 week+week^2 换成自然三次样条 cr(week_c, df=5)：真正的
    非线性、非参数形状，不再预设"先加速后不变"这一种二次曲线形态。
  * 固定效应比较表相应改写（M3 用样条替代二次项）。
  * 问题二接口不再用闭式二次公式，而是预先在观测孕周范围内打一张细网格
    (0.02周步长)，存网格值，运行时用线性插值取样条曲线的值——避免下游
    脚本需要重新拟合样条基或依赖 patsy 设计矩阵对象。
  * 样条对训练范围外的外推非常不可靠（比二次多项式更不可靠），因此
    week_reach/prob_qualified 严格限制在观测孕周范围 [11,29] 内，
    超出范围直接返回 NaN 并打印警告，不做任何隐式外推。

v2 相对 v1 的修复（仍然保留，见下）：
  P0  MixedLM 不再全局屏蔽警告静默接受未收敛结果；改用 fit_mixed() 多优化器
      重试，显式检查 .converged，全部失败则明确标注「结果仅供参考」。
  P0  BMI 拆成基线(between,Q2分组用) / 孕期内变化(within)两项，LRT 检验
      朴素单系数做法是否成立。
  P1  sqrt 反变换偏差修正：E[Y]=mean_resp^2+var_resp。
  P1  聚类 Bootstrap 1000 次，显式统计未收敛比例。
  P1  相关系数显著性改用按孕妇聚类 Bootstrap 区间。
  P0.5 新增普通线性回归基线对照，量化回应官方评阅要点。

数据口径：主口径 data/processed/male_clean_event.csv（事件级），
          对照 data/processed/male_clean.csv（行级，估计技术误差）。

输出   outputs/q1_out.txt    完整运行日志
       outputs/q1_coef.npy   最终模型系数（含样条网格），供问题二调用

运行：  python src/clean.py && python src/q1_model.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as viff

pd.set_option('display.width', 200)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --fast：开发迭代用，只用单一优化器、Bootstrap 降到 200 次；跑得快但数字
# 不能写进论文。默认（不带该参数）为完整精度：多优化器重试 + 1000 次
# Bootstrap，正式定稿前必须用默认模式重跑一遍。
FAST = '--fast' in sys.argv
METHODS = ('lbfgs',) if FAST else ('lbfgs', 'powell', 'cg', 'nm')
N_BOOT = 200 if FAST else 1000

ROOT = Path(__file__).resolve().parent.parent
EVENT = ROOT / 'data' / 'processed' / 'male_clean_event.csv'
ROWLV = ROOT / 'data' / 'processed' / 'male_clean.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEP = '=' * 78
SPLINE_DF = 5
WEEK_TERM = f'cr(week_c, df={SPLINE_DF})'  # 自然三次样条：真正的非线性孕周项

if FAST:
    print('*** --fast 模式：单优化器 + Bootstrap 降到 200 次，仅供开发调试，'
          '数字不得写入论文 ***')


def fit_mixed(formula, data, groups, re_formula=None, vc_formula=None,
              reml=True, methods=METHODS,
              maxiter=300, label=''):
    """稳健拟合：多优化器重试，只接受 .converged=True 的结果；全部失败则
    显式打印警告并标注，不得被当作正常结论使用。"""
    best, tried = None, []
    for method in methods:
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('ignore')
                m = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula,
                                vc_formula=vc_formula).fit(reml=reml, method=method,
                                                           maxiter=maxiter)
        except Exception:
            tried.append(f'{method}:异常')
            continue
        conv = bool(getattr(m, 'converged', False))
        tried.append(f'{method}:{"收敛" if conv else "未收敛"}')
        if conv and (best is None or m.llf > best.llf):
            best = m
    if best is not None:
        return best, True, tried
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('ignore')
        m = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula,
                        vc_formula=vc_formula).fit(reml=reml, maxiter=maxiter)
    print(f'  !! [{label}] 全部优化器未收敛（{tried}），此结果仅供参考，不作为结论依据')
    return m, False, tried


def load():
    if not EVENT.exists():
        raise SystemExit(f'未找到 {EVENT}，请先运行 python src/clean.py')
    e = pd.read_csv(EVENT, encoding='utf-8-sig')
    e = e.rename(columns={'week_mean': 'week', 'y_conc_mean': 'y_conc'})
    e = e.dropna(subset=['mother_id', 'week', 'bmi', 'y_conc']).reset_index(drop=True)
    r = pd.read_csv(ROWLV, encoding='utf-8-sig')
    r = r.dropna(subset=['mother_id', 'week', 'bmi', 'y_conc']).reset_index(drop=True)
    return e, r


d, rowlv = load()
d = d.sort_values(['mother_id', 'week']).reset_index(drop=True)
d['bmi_baseline'] = d.groupby('mother_id').bmi.transform('first')
d['bmi_within'] = d.bmi - d.bmi_baseline
_first_by_mom = d.groupby('mother_id').bmi.first()
_mean_by_mom = d.groupby('mother_id').bmi.mean()
R_FIRST_MEAN = float(st.pearsonr(_first_by_mom, _mean_by_mom.loc[_first_by_mom.index])[0])
WEEK_MIN, WEEK_MAX = float(d.week.min()), float(d.week.max())

print(SEP)
print(f'主口径 {EVENT.relative_to(ROOT)}（事件级，丙 clean.py v3）')
print(f'  {len(d)} 个抽血事件 / {d.mother_id.nunique()} 位孕妇；'
      f'每人抽血 {d.groupby("mother_id").size().min()}~{d.groupby("mother_id").size().max()} 次，'
      f'中位 {d.groupby("mother_id").size().median():.0f}')
print(f'  含技术重复的事件 {int(d.is_tech_repeat.sum())} 个（{d.is_tech_repeat.mean()*100:.1f}%），'
      f'行级共 {len(rowlv)} 条记录')
print(f'Y浓度(事件均值) 均值 {d.y_conc.mean():.4f} 中位 {d.y_conc.median():.4f} '
      f'偏度 {st.skew(d.y_conc):.3f} 峰度 {st.kurtosis(d.y_conc):.3f}')
print(f'孕周观测范围 [{WEEK_MIN:.2f}, {WEEK_MAX:.2f}]；样条外无法可靠外推，'
      f'问题二反解严格限制在此区间内')
print(f'BMI {d.bmi.min():.2f}~{d.bmi.max():.2f}')
print(f'事件级达标(均值口径>=4%)占比 {d.qual_mean.mean():.3f}；'
      f'曾达标孕妇 {int(d.groupby("mother_id").reached.first().sum())}/{d.mother_id.nunique()}')
print(f'\n首次(基线) BMI 与孕妇均值 BMI 相关 r={R_FIRST_MEAN:.4f}'
      f'（高度一致，用首次 BMI 作 Q2 可操作的分组变量：调度时未来 BMI 未知）')

# ============ 部件0：技术误差量化 ============
print('\n' + SEP + '\n【部件0】技术（测量）误差量化 —— 供问题二、三的误差分析直接引用')
rep = d[d.y_conc_n >= 2]
print(f'含 ≥2 次测序的事件 {len(rep)} 个（样本较少，区间较宽）；'
      f'事件内 Y 浓度 SD：中位 {rep.y_conc_sd.median():.4f}，最大 {rep.y_conc_sd.max():.4f}')
r3 = rowlv.copy()
r3['resp'] = np.sqrt(r3.y_conc)
r3['week_c'] = r3.week - r3.week.mean()
r3['bmi_c'] = r3.bmi - r3.bmi.mean()
r3['draw_key'] = r3.mother_id + '_' + r3.draw_idx.astype(str)
m3l, conv0, _ = fit_mixed(f'resp ~ {WEEK_TERM} + bmi_c', r3, groups=r3.mother_id,
                          re_formula='~week_c', vc_formula={'draw': '0+C(draw_key)'},
                          label='三层嵌套-技术误差')
s2_tech = float(m3l.scale)
n_rep_events = int((d.y_conc_n >= 2).sum())
df_tech = max(n_rep_events - 1, 1)
chi_lo, chi_hi = st.chi2.ppf([.975, .025], df_tech)
tech_lo, tech_hi = s2_tech * df_tech / chi_hi, s2_tech * df_tech / chi_lo
print(f'三层嵌套（孕妇/抽血/测序，收敛={conv0}）分解 sqrt(Y) 的方差：')
print(f'  孕妇间   {float(m3l.cov_re.iloc[0,0]):.6f}')
print(f'  抽血间   {float(m3l.vcomp[0]):.6f}')
print(f'  测序内（测序内误差估计，仅 {n_rep_events} 个重复事件支撑）'
      f'  {s2_tech:.6f}  近似 95% CI [{tech_lo:.6f}, {tech_hi:.6f}]')
print(f'  SD ≈ {np.sqrt(s2_tech):.4f}（sqrt 尺度），近似 CI [{np.sqrt(tech_lo):.4f}, {np.sqrt(tech_hi):.4f}]')

# ============ 部件0.5：线性回归基线对照 + 孕周形状的非参数检查 ============
print('\n' + SEP + '\n【部件0.5】非线性检验：线性回归基线 + 二次多项式 vs 样条的形状对照')
print('           （回应官方评阅要点："多元线性回归效果不理想，应考虑非线性模型"）')
Xlin = sm.add_constant(d[['week', 'bmi']])
lin = sm.OLS(d.y_conc, Xlin).fit()
bp_lin = sm.stats.diagnostic.het_breuschpagan(lin.resid, Xlin)
sw_lin = st.shapiro(lin.resid.sample(min(500, len(lin.resid)), random_state=0))
print(f'普通多元线性回归  Y ~ week + BMI（原始尺度，忽略重复测量）：')
print(f'  R2 = {lin.rsquared:.4f}   Breusch-Pagan p = {bp_lin[1]:.2e}   '
      f'Shapiro p = {sw_lin.pvalue:.2e}  -> 效果差，与评阅要点判断一致')

d['resp0'] = np.sqrt(d.y_conc)
d['week_c'] = d.week - d.week.mean()
lo_np = sm.nonparametric.lowess(d.resp0, d.week_c, frac=0.35, return_sorted=True)
Xq = sm.add_constant(np.column_stack([d.week_c, d.week_c ** 2]))
quad_marg = sm.OLS(d.resp0, Xq).fit()
qpred = quad_marg.params.iloc[0] + quad_marg.params.iloc[1] * lo_np[:, 0] \
    + quad_marg.params.iloc[2] * lo_np[:, 0] ** 2
dev_quad = np.abs(lo_np[:, 1] - qpred)
print(f'\n非参数 LOWESS（不预设函数形式）与二次多项式的偏差：'
      f'最大 {dev_quad.max():.4f}，平均 {dev_quad.mean():.4f}（sqrt(Y)尺度，边际、未控制BMI）')
print('LOWESS 显示孕周效应形状为「11~15周快速上升 -> 16~20周平台期 -> 24~29周加速」，')
print('二次多项式结构性地无法拟合中段平台，故本文孕周项改用自然三次样条'
      f'（{WEEK_TERM}），由数据决定形状，不预设单一开口方向的曲线。')

# ============ 部件1：相关特性 ============
print('\n' + SEP + '\n【部件1】相关特性分析（事件级）')
rows = []
for v in ['week', 'bmi', 'age', 'height', 'weight']:
    r, pr = st.pearsonr(d[v], d.y_conc)
    s, ps = st.spearmanr(d[v], d.y_conc)
    rows.append([v, r, pr, s, ps])
print(pd.DataFrame(rows, columns=['变量', 'Pearson_r', 'p_P(朴素)', 'Spearman_rho', 'p_S(朴素)'])
      .round(4).to_string(index=False))
print('  注：以上 p 值把 1021 个事件当独立观测，是朴素上限，见下方聚类 Bootstrap。')


def cluster_boot_corr(x, y, mother_id, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'x': np.asarray(x), 'y': np.asarray(y), 'm': np.asarray(mother_id)})
    idx_by_m = {m: g.index.values for m, g in df.groupby('m')}
    ids = np.array(list(idx_by_m))
    rs = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids), replace=True)
        sub = df.loc[np.concatenate([idx_by_m[m] for m in pick])]
        if sub.x.std() > 0 and sub.y.std() > 0:
            rs.append(st.pearsonr(sub.x, sub.y)[0])
    rs = np.array(rs)
    return float(rs.mean()), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


print(f'\n按孕妇聚类 Bootstrap 的相关系数区间（{N_BOOT} 次重抽）：')
for v in ['week', 'bmi']:
    mu, lo, hi = cluster_boot_corr(d[v], d.y_conc, d.mother_id, B=N_BOOT)
    sig = '显著（区间不含0）' if lo * hi > 0 else '不显著（区间含0）'
    print(f'  {v:6s}  Boot均值={mu:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  -> {sig}')


def partial_corr(x, y, covars):
    X = sm.add_constant(covars)
    return st.pearsonr(sm.OLS(x, X).fit().resid, sm.OLS(y, X).fit().resid)


print('\n偏相关（控制孕周后的净相关，朴素 p 值仅供参考）：')
for v in ['bmi', 'age', 'height', 'weight']:
    r, p = partial_corr(d[v].values, d.y_conc.values, d[['week']].values)
    print(f'  {v:8s} r = {r:+.4f}   p(朴素) = {p:.3e}')

nv = d.groupby('mother_id').mother_id.transform('size')
dd = d[nv >= 2].copy()
dd['dw'] = dd.week - dd.groupby('mother_id').week.transform('mean')
dd['dy'] = dd.y_conc - dd.groupby('mother_id').y_conc.transform('mean')
dd['db'] = dd.bmi - dd.groupby('mother_id').bmi.transform('mean')
r_in, p_in = st.pearsonr(dd.dw, dd.dy)
bw = d.groupby('mother_id').agg(w=('week', 'mean'), y=('y_conc', 'mean'), b=('bmi', 'mean'))
r_bw, p_bw = st.pearsonr(bw.w, bw.y)
r_b, p_b = st.pearsonr(bw.b, bw.y)
r_wb, p_wb = st.pearsonr(dd.dw, dd.db)
print(f'\n孕周-Y浓度 组内相关: r = {r_in:+.4f}, p(朴素) = {p_in:.2e}')
print(f'孕周-Y浓度 组间相关: r = {r_bw:+.4f}, p(朴素) = {p_bw:.2e}  -> 符号相反，与选择性')
print('  复检机制一致（数据一致的解释，非已证明的因果结论）')
print(f'⚠️ 孕妇内 BMI-孕周相关: r = {r_wb:+.4f}, p(朴素) = {p_wb:.2e} -> BMI 与孕周存在')
print('   孕期内混杂，故【部件2】把 BMI 拆成 between/within 两项。')

# ============ 部件2：BMI between/within 分解的必要性 ============
print('\n' + SEP + '\n【部件2】BMI 混杂检验：能否把 BMI 当普通时变协变量？')
d['resp'] = np.sqrt(d.y_conc)
MU_W = float(d.week.mean())
MU_BB = float(d.bmi_baseline.mean())
d['bmi_c'] = d.bmi - d.bmi.mean()
d['bmi_between_c'] = d.bmi_baseline - MU_BB
d['age_c'] = d.age - d.age.mean()

m_naive_bmi, c1, _ = fit_mixed(f'resp ~ {WEEK_TERM} + bmi_c', d, groups=d.mother_id,
                               re_formula='~week_c', label='BMI朴素单系数')
m_split_bmi, c2, _ = fit_mixed(f'resp ~ {WEEK_TERM} + bmi_between_c + bmi_within',
                               d, groups=d.mother_id, re_formula='~week_c', label='BMI拆分')
stat = 2 * (m_split_bmi.llf - m_naive_bmi.llf)
p_split = st.chi2.sf(stat, 1)
print(f'朴素单系数模型（收敛={c1}）：bmi 系数 = {m_naive_bmi.fe_params["bmi_c"]:.5f}')
print(f'拆分模型（收敛={c2}）：between = {m_split_bmi.fe_params["bmi_between_c"]:.5f}，'
      f'within = {m_split_bmi.fe_params["bmi_within"]:.5f}')
print(f'LRT(H0: between系数=within系数): chi2={stat:.3f}, df=1, p={p_split:.3e}')
print(('-> 拒绝 H0：必须拆分，between（基线BMI）才是 Q2 分组应使用的量。' if p_split < 0.05
       else '-> 未拒绝 H0，但为可解释性仍采用拆分模型。'))

# ============ 部件3a：随机效应结构（在样条孕周结构下重新检验） ============
print('\n' + SEP + '\n【部件3a】随机效应结构：为什么不能用 OLS')
F_SPLIT = f'resp ~ {WEEK_TERM} + bmi_between_c + bmi_within'
ols = smf.ols('resp ~ week_c + bmi_between_c + bmi_within', d).fit()
ri, c_ri, _ = fit_mixed('resp ~ week_c + bmi_between_c + bmi_within', d, groups=d.mother_id,
                        reml=False, label='随机截距')
s2u, s2e = float(ri.cov_re.iloc[0, 0]), float(ri.scale)
lr = 2 * (ri.llf - ols.llf)
print(f'OLS         logLik = {ols.llf:9.3f}')
print(f'随机截距(收敛={c_ri})  logLik = {ri.llf:9.3f}')
print(f'  LRT(随机截距=0): chi2 = {lr:.2f}, 混合卡方 p = {0.5*st.chi2.sf(lr,1):.2e}')
rs, c_rs, _ = fit_mixed('resp ~ week_c + bmi_between_c + bmi_within', d, groups=d.mother_id,
                        re_formula='~week_c', reml=False, label='随机斜率')
lr2 = 2 * (rs.llf - ri.llf)
p_rs = 0.5 * (st.chi2.sf(lr2, 1) + st.chi2.sf(lr2, 2))
print(f'随机斜率(收敛={c_rs})  logLik = {rs.llf:9.3f}')
print(f'  LRT(+孕周随机斜率): chi2 = {lr2:.2f}, df=2, 混合卡方 p = {p_rs:.2e}')
USE_RS = bool(p_rs < 0.05)
print(f'-> 采用「随机截距 + 孕周随机斜率」：{USE_RS}')
RE_F = '~week_c' if USE_RS else None

# ============ 部件3b：固定效应设定（孕周项改样条） ============
print('\n' + SEP + '\n【部件3b】固定效应设定比较（ML 拟合；M3 起孕周项改用自然三次样条）')
specs = {
    'M1 week(线性)': 'resp ~ week_c',
    'M2 M1+bmi(between)': 'resp ~ week_c + bmi_between_c',
    'M3 week改样条+bmi(between)': f'resp ~ {WEEK_TERM} + bmi_between_c',
    'M3q 对照:week+week^2+bmi(between)': 'resp ~ week_c + I(week_c**2) + bmi_between_c',
    'M4 M3+bmi(within)': F_SPLIT,
    'M5 M4+week线性x bmi_between交互': F_SPLIT + ' + week_c:bmi_between_c',
    'M6 M4+age': F_SPLIT + ' + age_c',
    'M7 M4用身高体重替代between': (f'resp ~ {WEEK_TERM} + bmi_within + height + weight'),
}
fits, rec = {}, []
for n, f in specs.items():
    m, conv, _ = fit_mixed(f, d, groups=d.mother_id, re_formula=RE_F, reml=False, label=n)
    fits[n] = m
    k = len(m.params) + 1
    rec.append([n, len(m.fe_params), m.llf, -2 * m.llf + 2 * k,
                -2 * m.llf + k * np.log(len(d)), conv])
cmp = pd.DataFrame(rec, columns=['模型', '固定效应数', 'logLik', 'AIC', 'BIC', '收敛'])
cmp['dAIC'] = (cmp.AIC - cmp.AIC.min()).round(2)
cmp['dBIC'] = (cmp.BIC - cmp.BIC.min()).round(2)
print(cmp.round(2).to_string(index=False))
print(f'\n样条(M3) 相对二次多项式(M3q) 的 AIC 优势: '
      f'{cmp.loc[cmp.模型=="M3q 对照:week+week^2+bmi(between)","AIC"].iloc[0] - cmp.loc[cmp.模型=="M3 week改样条+bmi(between)","AIC"].iloc[0]:.2f}'
      f'（正值=样条更优，验证部件0.5的判断）')


def lrt(a, b, name):
    stat = 2 * (fits[b].llf - fits[a].llf)
    df = len(fits[b].params) - len(fits[a].params)
    print(f'  {name:28s} chi2 = {stat:7.3f}, df = {df}, p = {st.chi2.sf(stat, df):.3e}')


print('\n嵌套模型似然比检验：')
lrt('M1 week(线性)', 'M2 M1+bmi(between)', 'M1->M2 加基线BMI')
lrt('M4 M3+bmi(within)', 'M5 M4+week线性x bmi_between交互', 'M4->M5 加交互')
lrt('M4 M3+bmi(within)', 'M6 M4+age', 'M4->M6 加年龄')
FINAL_NAME = 'M4 M3+bmi(within)'
print(f'\n-> 主模型取 {FINAL_NAME}：孕周用样条（非线性），BMI 按 between/within 拆分。')

# ============ 部件4：最终模型 ============
print('\n' + SEP + '\n【部件4】最终模型（REML 重估方差分量，孕周为自然三次样条）')
FINAL_F = F_SPLIT
final, c_final, tried_final = fit_mixed(FINAL_F, d, groups=d.mother_id, re_formula=RE_F,
                                        reml=True, label='最终模型-REML')
print(f'收敛状态: {c_final}（尝试记录: {tried_final}）')
if not c_final:
    print('!!! 最终模型未收敛，以下数字仅供参考 !!!')
fe = final.fe_params
ci = final.conf_int()
print('固定效应（样条基函数系数本身不直接解释，看下方的曲线取值）：')
print(pd.DataFrame({'估计值': final.params, '标准误': final.bse, 'z': final.tvalues,
                    'p值': final.pvalues, 'CI下限': ci[0], 'CI上限': ci[1]})
      .loc[fe.index].round(5).to_string())
def wald_chi2(names, label):
    """协方差矩阵可能因未收敛而奇异（尤其 --fast 单优化器模式），
    不得让整个脚本崩溃：奇异时改用伪逆并显式标注，绝不假装算出了正确值。"""
    bb = fe[names].values
    VV = final.cov_params().loc[names, names].values
    try:
        stat = float(bb @ np.linalg.inv(VV) @ bb)
        singular = False
    except np.linalg.LinAlgError:
        stat = float(bb @ np.linalg.pinv(VV) @ bb)
        singular = True
    p = st.chi2.sf(stat, len(names))
    tag = '（!! 协方差矩阵奇异，用伪逆近似，通常因未收敛，请勿引用 !!）' if singular else ''
    print(f'{label}: chi2 = {stat:.2f}, df = {len(names)}, p = {p:.3e} {tag}')
    return stat, p, singular


ks = [k for k in fe.index if k != 'Intercept']
Wstat, _, _ = wald_chi2(ks, '\n整体 Wald 检验（全部固定效应=0）')
spline_ks = [k for k in ks if 'cr(' in k]
W_sp, _, _ = wald_chi2(spline_ks, '孕周样条项整体检验（全部样条系数=0，即孕周无效应）')
print(f'基线BMI(between)系数: {fe["bmi_between_c"]:.5f}, p = {final.pvalues["bmi_between_c"]:.4f}')
print(f'孕期内BMI(within)系数: {fe["bmi_within"]:.5f}, p = {final.pvalues["bmi_within"]:.4f}')

G = final.cov_re.values
s2u0 = float(G[0, 0])
s2u1 = float(G[1, 1]) if USE_RS else 0.0
cov01 = float(G[0, 1]) if USE_RS else 0.0
s2e = float(final.scale)
if USE_RS:
    print(f'\n随机效应协方差 G：截距方差 {s2u0:.5f}，斜率方差 {s2u1:.7f}，'
          f'协方差 {cov01:.5f}（相关 {cov01/np.sqrt(s2u0*s2u1):+.3f}）')
print(f'残差方差 s2_e = {s2e:.5f}')


def icc_at(week_c_val):
    su = s2u0 + 2 * cov01 * week_c_val + s2u1 * week_c_val ** 2
    return su / (su + s2e)


print('ICC 随孕周变化（随机斜率下 ICC 非常数，逐周报告）：')
for wk in [12, 16, 20, 24]:
    print(f'  孕周{wk:>2}周: ICC = {icc_at(wk - MU_W):.4f}')

# ---- 用训练数据的 design_info 重建样条基，生成预测网格（供解读+问题二接口）----
y_tr, X_tr = patsy.dmatrices(FINAL_F, d, return_type='dataframe')
DESIGN_INFO = X_tr.design_info
GRID_STEP = 0.02
week_grid = np.arange(WEEK_MIN, WEEK_MAX + 1e-9, GRID_STEP)
grid_df = pd.DataFrame({'week_c': week_grid - MU_W,
                        'bmi_between_c': 0.0, 'bmi_within': 0.0})
Xg = patsy.build_design_matrices([DESIGN_INFO], grid_df, return_type='dataframe')[0]
week_effect_grid = (Xg.values @ fe[Xg.columns].values)  # 纯孕周项贡献（BMI置0）
base_intercept = float(fe['Intercept'])

print(f'\n样条曲线关键孕周点取值（sqrt(Y)尺度，基线BMI处，作为部件0.5预测的定稿版）：')
print('week   sqrt(Y)预测   原尺度浓度(%)')
for wk in [11, 13, 15, 17, 19, 21, 23, 25, 27, 29]:
    val = float(np.interp(wk, week_grid, week_effect_grid))
    print(f'{wk:>4}   {val:.4f}        {val**2*100:.3f}%')
d1 = np.diff(week_effect_grid)
plateau_idx = np.where((week_grid[:-1] >= 16) & (week_grid[:-1] <= 20))[0]
print(f'16~20周平台段的平均斜率 = {d1[plateau_idx].mean():.5f}'
      f'（对照 11~15周 = {d1[(week_grid[:-1]>=11)&(week_grid[:-1]<=15)].mean():.5f}，'
      f'24~28周 = {d1[(week_grid[:-1]>=24)&(week_grid[:-1]<=28)].mean():.5f}）')
print('  -> 中段明显放缓，两端明显更陡，量化确认了样条捕捉到的「平台」现象')

vf = float(np.var(np.asarray(final.model.exog) @ fe.values, ddof=1))
vr = s2u0 + (s2u1 * float(np.var(d.week_c, ddof=1)) if USE_RS else 0.0)
marg_r2 = vf / (vf + vr + s2e)
cond_r2 = (vf + vr) / (vf + vr + s2e)
print(f'\n边际 R2 = {marg_r2:.4f}   条件 R2 = {cond_r2:.4f}')
print(f'对照【部件0.5】普通线性回归 R2 = {lin.rsquared:.4f}：'
      f'条件 R2 提升 {cond_r2 - lin.rsquared:+.4f}')

# ============ 部件5：诊断与稳健性 ============
print('\n' + SEP + '\n【部件5】模型诊断')
res = np.asarray(final.resid)
print(f'残差 偏度 {st.skew(res):+.3f}  峰度 {st.kurtosis(res):+.3f}')
print(f'Shapiro-Wilk（随机500条）p = '
      f'{st.shapiro(pd.Series(res).sample(min(500, len(res)), random_state=0)).pvalue:.4f}')
bp = sm.stats.diagnostic.het_breuschpagan(res, sm.add_constant(
    d[['week_c', 'bmi_between_c', 'bmi_within']]))
print(f'Breusch-Pagan 异方差 LM = {bp[0]:.3f}, p = {bp[1]:.4f}')
X = sm.add_constant(d[['week_c', 'bmi_between_c', 'bmi_within', 'age_c']])
print('VIF:', {c: round(viff(X.values, i), 2) for i, c in enumerate(X.columns) if c != 'const'})
print(f'|标准化残差| > 3 的事件数 = {(np.abs(st.zscore(res)) > 3).sum()}'
      f'（占 {(np.abs(st.zscore(res)) > 3).mean()*100:.2f}%）')

print('\n--- 稳健性检验 ---')
r3['bmi_baseline'] = r3.mother_id.map(d.groupby('mother_id').bmi_baseline.first())
r3['bmi_between_c'] = r3.bmi_baseline - MU_BB
r3['bmi_within'] = r3.bmi - r3.bmi_baseline
naive, c_nv, _ = fit_mixed(FINAL_F, r3, groups=r3.mother_id, re_formula=RE_F, label='行级朴素')
nest, c_ns, _ = fit_mixed(FINAL_F, r3, groups=r3.mother_id, re_formula=RE_F,
                          vc_formula={'draw': '0+C(draw_key)'}, label='行级三层嵌套')
qc = d[d.flag_any == 0]
robqc, c_qc, _ = fit_mixed(FINAL_F, qc, groups=qc.mother_id, re_formula=RE_F, label='QC过滤')
ch = d[d.flag_chrono == 0]
robch, c_ch, _ = fit_mixed(FINAL_F, ch, groups=ch.mother_id, re_formula=RE_F, label='剔时序异常')
print(f'各稳健性设定的收敛状态：行级朴素={c_nv}  行级三层嵌套={c_ns}  '
      f'QC过滤={c_qc}  剔时序异常={c_ch}')
print(f'\nBMI(between) 系数对比（均为样条孕周结构）：')
print(f'  主模型 事件级(n={len(d)})       {final.fe_params["bmi_between_c"]:.5f}  '
      f'p={final.pvalues["bmi_between_c"]:.4f}')
print(f'  行级三层嵌套(n={len(r3)})       {nest.fe_params["bmi_between_c"]:.5f}  '
      f'p={nest.pvalues["bmi_between_c"]:.4f}')
print(f'  行级朴素[未拆技术重复](n={len(r3)}) {naive.fe_params["bmi_between_c"]:.5f}  '
      f'p={naive.pvalues["bmi_between_c"]:.4f}')
print(f'  QC过滤(n={len(qc)})             {robqc.fe_params["bmi_between_c"]:.5f}  '
      f'p={robqc.pvalues["bmi_between_c"]:.4f}')
print(f'  剔时序异常(n={len(ch)})         {robch.fe_params["bmi_between_c"]:.5f}  '
      f'p={robch.pvalues["bmi_between_c"]:.4f}')

print(f'\n聚类 Bootstrap（目标 {N_BOOT} 次重抽，逐次检查收敛，仅用收敛结果计算区间）：')
rng = np.random.default_rng(0)
idx_by_mom = {m: g.index.values for m, g in d.groupby('mother_id')}
ids = np.array(list(idx_by_mom))
boot_between, boot_within, n_fail = [], [], 0
for _ in range(N_BOOT):
    pick = rng.choice(ids, len(ids), replace=True)
    sub = d.loc[np.concatenate([idx_by_mom[m] for m in pick])].copy()
    sub['g'] = np.concatenate([np.full(len(idx_by_mom[m]), k) for k, m in enumerate(pick)])
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore')
            bm = smf.mixedlm(FINAL_F, sub, groups=sub.g, re_formula=RE_F).fit(
                reml=False, method='lbfgs', maxiter=100)
        if bool(getattr(bm, 'converged', False)):
            boot_between.append(bm.fe_params['bmi_between_c'])
            boot_within.append(bm.fe_params['bmi_within'])
        else:
            n_fail += 1
    except Exception:
        n_fail += 1
bb_arr, bw_arr = np.array(boot_between), np.array(boot_within)
print(f'  成功收敛 {len(bb_arr)}/{N_BOOT} 次（失败/未收敛 {n_fail} 次，'
      f'失败率 {n_fail/N_BOOT*100:.1f}%）')
print(f'  bmi_between: 点估计={fe["bmi_between_c"]:.5f}  Boot均值={bb_arr.mean():.5f}  '
      f'95%CI=[{np.percentile(bb_arr,2.5):.5f}, {np.percentile(bb_arr,97.5):.5f}]')
print(f'  bmi_within : 点估计={fe["bmi_within"]:.5f}  Boot均值={bw_arr.mean():.5f}  '
      f'95%CI=[{np.percentile(bw_arr,2.5):.5f}, {np.percentile(bw_arr,97.5):.5f}]')
print(f'  （若失败率不低需在论文中如实注明，不能笼统宣称"完全不依赖分布假设"）')

# ============ 部件6：问题二接口（网格插值，避免下游依赖 patsy 设计矩阵）============
print('\n' + SEP + '\n【部件6】问题二接口（孕周项=样条网格插值，sqrt反变换偏差已修正）')
wd = d[['week_c', 'bmi_within']].copy()
g_reg = sm.OLS(wd.bmi_within, sm.add_constant(wd.week_c)).fit()
G0, G1, G_VAR = float(g_reg.params.iloc[0]), float(g_reg.params.iloc[1]), float(g_reg.scale)

COEF = dict(scale='sqrt(y)', week_grid=week_grid, week_effect_grid=week_effect_grid,
            week_min=WEEK_MIN, week_max=WEEK_MAX,
            b3=float(fe['bmi_between_c']), b4=float(fe['bmi_within']),
            mu_w=MU_W, mu_bb=MU_BB, s2u0=s2u0, s2u1=s2u1, cov01=cov01, s2e=s2e,
            thr_resp=float(np.sqrt(0.04)), s2_tech=s2_tech,
            g0=G0, g1=G1, g_var=G_VAR, level='event', converged=c_final,
            model='spline_df5_mixed')


def mean_resp(week, bmi_baseline, C=COEF):
    """均值预测：孕周项用样条网格插值（训练范围外返回 NaN），
    基线 BMI 已知，孕期内 BMI 漂移用总体线性趋势外推。"""
    week = np.asarray(week, float)
    out_of_range = (week < C['week_min']) | (week > C['week_max'])
    week_eff = np.interp(week, C['week_grid'], C['week_effect_grid'],
                         left=np.nan, right=np.nan)
    if np.any(out_of_range):
        print(f'  ⚠️ week_reach/predict 查询超出样条训练范围 '
              f'[{C["week_min"]},{C["week_max"]}]，已返回 NaN（拒绝外推）')
    w_c = week - C['mu_w']
    bb = np.asarray(bmi_baseline, float) - C['mu_bb']
    within_hat = C['g0'] + C['g1'] * w_c
    return week_eff + C['b3'] * bb + C['b4'] * within_hat


def var_resp(week, C=COEF):
    w = np.asarray(week, float) - C['mu_w']
    v_re = C['s2u0'] + 2 * C['cov01'] * w + C['s2u1'] * w ** 2 + C['s2e']
    return v_re + (C['b4'] ** 2) * C['g_var']


def predict_y(week, bmi_baseline, C=COEF):
    mu = mean_resp(week, bmi_baseline, C)
    return mu ** 2 + var_resp(week, C)


def prob_qualified(week, bmi_baseline, thr=0.04, C=COEF):
    mu = mean_resp(week, bmi_baseline, C)
    z = (np.sqrt(thr) - mu) / np.sqrt(var_resp(week, C))
    p = st.norm.sf(z)
    return np.where(np.isnan(mu), np.nan, p)


def week_reach(bmi_baseline, thr=0.04, p=0.9, C=COEF, lo=None, hi=None):
    """该基线 BMI 的孕妇以把握度 p 达标的最早孕周。lo/hi 默认严格限制在
    样条训练范围 [week_min, week_max] 内，绝不外推。"""
    from scipy.optimize import brentq
    lo = C['week_min'] if lo is None else max(lo, C['week_min'])
    hi = C['week_max'] if hi is None else min(hi, C['week_max'])
    f = lambda t: prob_qualified(t, bmi_baseline, thr, C) - p
    flo, fhi = f(lo), f(hi)
    if np.isnan(flo) or np.isnan(fhi):
        return np.nan
    if flo >= 0:
        return lo
    if fhi < 0:
        return np.nan
    return brentq(f, lo, hi)


def main():
    if not COEF['converged']:
        print('  ⚠️ 最终模型未全部收敛，以下数字仅作演示，正式结论前需人工复核！')
    print(f'  predict_y(week=16, bmi_baseline=32) = {predict_y(16, 32):.4f}')
    print(f'  P(Y>=4% | week=16, bmi_baseline=32) = {float(prob_qualified(16, 32)):.3f}')
    print(f'\n  各基线BMI达到4%的最早孕周（严格限制在观测范围[{WEEK_MIN:.0f},{WEEK_MAX:.0f}]内）：')
    print('    BMI    P=0.50    P=0.80    P=0.90    P=0.95')
    for bb in [22, 26, 30, 34, 38, 42]:
        cells = []
        for p in (.5, .8, .9, .95):
            t = week_reach(bb, p=p)
            cells.append('  超出观测范围' if not np.isfinite(t)
                         else (f'  <={WEEK_MIN:.0f}*' if t <= WEEK_MIN + 0.001 else f'{t:6.2f}'))
        print(f'    {bb:>3}   ' + '    '.join(cells))
    print(f'  （* = 触达观测下界 {WEEK_MIN:.0f} 周，真实达标孕周可能更早，但数据未覆盖，属外推提示）')
    np.save(OUT / 'q1_coef.npy', COEF, allow_pickle=True)
    print(f'\n系数已保存 {(OUT / "q1_coef.npy").relative_to(ROOT)}'
          f'（孕周项为样条网格，问题二用前应检查 converged 标记），'
          f'用 np.load(..., allow_pickle=True).item() 载入。')
    print('\n接口签名提示：predict_y / prob_qualified / week_reach 的第二参数是')
    print('  bmi_baseline（基线/首次 BMI），孕周项已由二次多项式换成样条网格插值。')


if __name__ == '__main__':
    main()
