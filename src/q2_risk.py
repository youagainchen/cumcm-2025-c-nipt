# -*- coding: utf-8 -*-
"""
q2_risk.py —— 甲：问题二的风险函数与期望风险机制（乙的优化器直接调用本模块）

设计决策（2026-09-04 甲乙对齐后定稿，全部可在 RiskParams 里改，供敏感性分析）：

1. 风险函数 R(t) 用**连续分段线性**（不用阶跃）：
     t <= 12          : R = 1                      （题面"早期发现风险较低"）
     12 < t <= 27     : R = 1 + 2(t-12)/15         （13-27周风险高，27周处=3）
     t > 27           : R = 3 + 6(t-27)/3          （28周后极高，30周处=9）
   选连续而非阶跃的理由：阶跃会让最优时点全部挤在 12.0 周这个边界上，
   组间差异被人为抹平，且目标函数不可导、优化不稳。

   ⚠️ 风险梯度(r_mid)是本模型最敏感的参数，实测扫描结果：
       r_mid=10, 失联=30  -> 各BMI最优时点全是12.0周，分组完全退化
       r_mid=5,  失联=30  -> 跨度仅0.25周，仍近似退化
       r_mid=3,  失联=30  -> 12/12/12/12.25/14/15.5（当前选用，温和分化）
       r_mid=2,  失联=60  -> 高BMI被推到23-25周上界，临床上已错过干预窗口
   机制解释：r_mid 越大，"多等一周"的确定性代价越高，早测就越占优，
   BMI 差异被压平。分组有意义的前提是"早测失败的代价"与"晚测本身的
   代价"处于可比量级。该参数的取舍是价值判断，已在文档中如实说明，
   并把上表作为敏感性分析的主要内容。

2. "检测未达标"的代价用**重测递归展开**，而不是固定罚分：
     E(t) = p(t)*R(t) + [1-p(t)]*E(t+Δ)
   其中 p(t)=P(Y>=4% | t, 基线BMI) 来自问题一模型。
   这样题面"28周后风险极高"才真正被触发——观测数据里没有任何孕妇是
   28周后才首次达标（0人），晚期高风险只能通过"检测失败->重测->拖延"
   这条路径产生，固定罚分做不到这一点。

3. 重测间隔 Δ = 3.7 周，**取自本数据集实测**：115 次"未达标后再抽血"的
   实际间隔中位数为 3.71 周（IQR 3.14-4.14），不是拍脑袋的假设。

4. 目标函数：最小化期望风险（题面"孕妇潜在风险最小"字面口径）。

5. 检测时点搜索窗口 [11, 25] 周：下界取问题一模型有效范围起点（不外推），
   上界取题面明确的可检测上限。重测若越过 25 周，风险照 R(t) 晚期段计算
   ——这正是晚期高风险被触发的路径，不额外截断。

用法（乙）：
    from q2_risk import RiskParams, ExpectedRisk
    er = ExpectedRisk()                      # 默认载入 outputs/q1_coef.npy
    er.expected_risk(t=13.0, bmi_baseline=32)     # 单点期望风险
    er.best_time(bmi_baseline=32)                 # 该BMI的最优时点与风险
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path

import numpy as np
import scipy.stats as st

ROOT = Path(__file__).resolve().parent.parent
COEF_PATH = ROOT / 'outputs' / 'q1_coef.npy'

# 检测时点搜索窗口（周）
T_MIN, T_MAX = 11.0, 25.0


@dataclass
class RiskParams:
    """风险函数与重测机制的全部可调参数，敏感性分析时改这里即可。"""
    t_early: float = 12.0      # 早期窗口上界
    t_mid: float = 27.0        # 中期窗口上界
    t_late_ref: float = 30.0   # 晚期参考点
    r_early: float = 1.0       # 早期风险基准
    r_mid: float = 3.0         # 中期末端(27周)风险
    r_late: float = 9.0        # 晚期参考点(30周)风险
    retest_gap: float = 3.7    # 重测间隔Δ（数据实测中位数）
    max_retest: int = 8        # 递归最大重测次数（防止无限展开）
    horizon: float = 29.0      # 问题一模型有效上界，超出不再取达标概率
    q_dropout: float = 0.094   # 每次检测失败后"不再复检"的概率（数据实测：127次未达标中12次未回访）
    r_dropout: float = 30.0    # 失联（永远拿不到结论）的风险，默认与30周确诊等价


def risk_curve(t, p: RiskParams = RiskParams()):
    """连续分段线性风险函数 R(t)，t 可为标量或数组。"""
    t = np.asarray(t, dtype=float)
    slope_mid = (p.r_mid - p.r_early) / (p.t_mid - p.t_early)
    slope_late = (p.r_late - p.r_mid) / (p.t_late_ref - p.t_mid)
    return np.where(
        t <= p.t_early,
        p.r_early,
        np.where(
            t <= p.t_mid,
            p.r_early + slope_mid * (t - p.t_early),
            p.r_mid + slope_late * (t - p.t_mid),
        ),
    )


class ExpectedRisk:
    """把问题一的达标概率模型接进风险机制。

    默认使用问题一导出的参数化模型（outputs/q1_coef.npy）。
    甲的生存分析定稿后，只需传入 prob_fn=<新的达标概率函数>，
    乙的优化代码不用改任何其它地方——这是两条路径交叉验证的接口约定。
    """

    def __init__(self, coef=None, prob_fn=None, params: RiskParams = None):
        self.params = params or RiskParams()
        self.prob_fn = prob_fn
        self._prob_supports_prev = (
            prob_fn is not None and "prev_week" in inspect.signature(prob_fn).parameters
        )
        if prob_fn is None:
            if coef is None:
                if not COEF_PATH.exists():
                    raise SystemExit(f'未找到 {COEF_PATH}，请先运行 python src/q1_model.py')
                coef = np.load(COEF_PATH, allow_pickle=True).item()
            if not coef.get('converged', False):
                print('  ⚠️ q1_coef.npy 的 converged 标记为 False，问题二结果不可用于定稿')
            self.coef = coef

    # ---- 问题一参数化模型的达标概率（与 q1_model.prob_qualified 完全一致）----
    def _prob_q1(self, week, bmi_baseline, thr=0.04):
        C = self.coef
        week = np.asarray(week, float)
        TOL = 1e-6
        out = (week < C['week_min'] - TOL) | (week > C['week_max'] + TOL)
        wk = np.clip(week, C['week_min'], C['week_max'])
        eff = np.interp(wk, C['week_grid'], C['week_effect_grid'])
        w_c = week - C['mu_w']
        bb = np.asarray(bmi_baseline, float) - C['mu_bb']
        mu = eff + C['b3'] * bb + C['b4'] * (C['g0'] + C['g1'] * w_c)
        var = (C['s2u0'] + 2 * C['cov01'] * w_c + C['s2u1'] * w_c ** 2 + C['s2e']
               + C['b4'] ** 2 * C['g_var'])
        p = st.norm.sf((np.sqrt(thr) - mu) / np.sqrt(var))
        return np.where(out, np.nan, p)

    def prob_qualified(self, week, bmi_baseline, thr=0.04, prev_week=None):
        if self.prob_fn is not None:
            if self._prob_supports_prev:
                return self.prob_fn(week, bmi_baseline, thr, prev_week=prev_week)
            # 兼容只实现原三参数签名的外部概率接口。
            return self.prob_fn(week, bmi_baseline, thr)
        return self._prob_q1(week, bmi_baseline, thr)

    # ---- 核心：重测递归展开的期望风险 ----
    def expected_risk(self, t, bmi_baseline, thr=0.04):
        """带失联风险的重测递归：

            E(t) = p(t)·R(t) + [1-p(t)]·{ q·R_失联 + (1-q)·E(t+Δ) }

        即每次检测失败后，孕妇以概率 q 不再复检（永远拿不到结论，按
        R_失联 计最大风险），以概率 1-q 在 t+Δ 复检。
        q=0.094 取自本数据实测：127 次未达标检测中有 12 次之后该孕妇
        再未回访。

        没有这一项时，模型会退化成"所有 BMI 都在 12 周检测"——因为失败
        的唯一代价只是推迟 3.7 周，而 R(t) 在 12 周前是平的，早测永远
        最优，BMI 分组失去意义。加入失联风险后，高 BMI 孕妇早测的代价
        （达标率低→失败→可能失联）才被正确计入。

        递归终止：达到 max_retest 次，或时点越过模型有效上界 horizon。
        """
        p = self.params
        t = float(t)
        reach = 1.0     # 走到当前这一轮检测的概率（既未达标也未失联）
        total = 0.0
        for k in range(p.max_retest + 1):
            tk = t + k * p.retest_gap
            if tk > p.horizon:
                break
            prev = None if k == 0 else tk - p.retest_gap
            pk = float(np.atleast_1d(
                self.prob_qualified(tk, bmi_baseline, thr, prev_week=prev)
            )[0])
            if not np.isfinite(pk):
                break
            total += reach * pk * float(risk_curve(tk, p))          # 本轮成功
            fail = reach * (1.0 - pk)
            total += fail * p.q_dropout * p.r_dropout               # 本轮失败后失联
            reach = fail * (1.0 - p.q_dropout)                      # 继续复检
            if reach < 1e-6:
                break
        # 用尽重测次数/越过有效区间仍未确诊：按最后时点风险计（保守）
        t_end = min(t + p.max_retest * p.retest_gap, p.horizon)
        total += reach * float(risk_curve(t_end, p))
        return total

    def best_time(self, bmi_baseline, thr=0.04, grid_step=0.05):
        """在 [T_MIN, T_MAX] 网格上搜索使期望风险最小的检测时点。"""
        grid = np.arange(T_MIN, T_MAX + 1e-9, grid_step)
        risks = np.array([self.expected_risk(t, bmi_baseline, thr) for t in grid])
        i = int(np.argmin(risks))
        return float(grid[i]), float(risks[i])

    def group_risk(self, bmi_values, t, thr=0.04):
        """给定一组孕妇的基线BMI与统一检测时点t，返回其平均期望风险。
        乙的分组优化直接调用本函数作为目标函数的组内部分。"""
        return float(np.mean([self.expected_risk(t, b, thr) for b in np.atleast_1d(bmi_values)]))

    def group_best_time(self, bmi_values, thr=0.04, grid_step=0.05):
        """给定一组孕妇，求该组统一最优时点与对应平均期望风险。"""
        grid = np.arange(T_MIN, T_MAX + 1e-9, grid_step)
        risks = np.array([self.group_risk(bmi_values, t, thr) for t in grid])
        i = int(np.argmin(risks))
        return float(grid[i]), float(risks[i])


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    import pandas as pd

    er = ExpectedRisk()
    p = er.params
    print('=' * 70)
    print('风险函数 R(t) 取值检查（连续分段线性）')
    for t in [11, 12, 13, 16, 20, 24, 27, 28, 30]:
        print(f'  t={t:>2}周  R={float(risk_curve(t, p)):6.2f}')

    print('\n' + '=' * 70)
    print(f'期望风险与最优时点（重测间隔Δ={p.retest_gap}周，窗口[{T_MIN},{T_MAX}]周）')
    print('  BMI   最优时点   最小期望风险   该时点单次达标概率')
    for b in [22, 26, 30, 32, 34, 38, 42]:
        t_opt, r_opt = er.best_time(b)
        pq = float(np.atleast_1d(er.prob_qualified(t_opt, b))[0])
        print(f'  {b:>3}    {t_opt:6.2f}周     {r_opt:8.3f}        {pq:.3f}')

    print('\n' + '=' * 70)
    print('全体孕妇统一时点（不分组）的基准结果，供乙的分组优化做对照：')
    d = pd.read_csv(ROOT / 'data' / 'processed' / 'male_clean_event.csv', encoding='utf-8-sig')
    d = d.sort_values(['mother_id', 'week_mean'])
    bmi_base = d.groupby('mother_id').bmi.first().dropna().values
    t_all, r_all = er.group_best_time(bmi_base)
    print(f'  n={len(bmi_base)} 位孕妇，统一最优时点 = {t_all:.2f} 周，平均期望风险 = {r_all:.4f}')
    print('  （乙的分组方案必须显著优于这个"不分组"基准，否则分组没有意义）')
