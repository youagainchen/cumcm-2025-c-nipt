# 2025 高教社杯 C 题：NIPT 的时点选择与胎儿异常判定

本仓库用于 2025 年全国大学生数学建模竞赛 C 题的团队协作，包括数据清洗、统计建模、BMI 分组与最佳 NIPT 时点优化、女胎异常判定、可视化和论文材料。

## 研究任务

1. 分析男胎 Y 染色体浓度与孕周、BMI 等指标的关系并检验显著性。
2. 根据 BMI 分组，为男胎孕妇确定风险最小的 NIPT 时点，并分析检测误差。
3. 综合身高、体重、年龄、检测误差及达标比例，优化 BMI 分组和检测时点。
4. 根据 AB 列标签，建立女胎染色体异常判定方法并评估性能。

## 目录

```text
problem/          题面
data/raw/         原始附件（只读，不在原文件上修改）
data/processed/   清洗后的行级、抽血事件级数据
docs/             分工、数据字典与建模记录
src/              可复现的清洗、建模和评估代码
figures/          论文图表
outputs/          模型结果与中间输出
paper/            论文与格式参考
```

## 协作约定

- 原始数据只读；所有清洗结果写入 `data/processed/`。
- 训练集和验证集必须按孕妇代码分组，避免同一孕妇的数据泄漏。
- 同一次抽血的多次测序保留为技术重复，并与不同孕周的随访检测区分。
- 图表统一输出为 300 dpi PNG；正式结论必须能由脚本复现。
- 大型临时文件、缓存和个人环境配置不得提交。

详细第一阶段安排见 [`docs/C题_第一阶段分工.md`](docs/C题_第一阶段分工.md)。

## 阶段一数据就绪状态（丙）

- 数据管道：`python src/clean.py` 从 `data/raw/附件.xlsx` 复现全部清洗产物。
- 产物（`data/processed/`，被 .gitignore 忽略、不入库）：
  - 行级：`male_min.csv`、`male_clean.csv`（男胎）、`female_clean.csv`（女胎，AB 列标签）；
  - **事件级**（每 (孕妇,抽血) 一行，问题一/二推荐口径）：`male_clean_event.csv`、`female_clean_event.csv`
    —— 问题二首达/删失默认均值口径并附 `*_any` 对照，见 [docs/data_dictionary.md](docs/data_dictionary.md)。
- 文档：列命名映射、字段含义、纪律与**对照评阅要点的处理立场**见 [`docs/data_dictionary.md`](docs/data_dictionary.md)；数字画像（含技术重复量化）见 [`docs/data_report.md`](docs/data_report.md)。

## 阶段一问题一状态

- 建模脚本：`python src/q1_model.py`（依赖 `data/processed/male_clean.csv`，先跑 `src/clean.py`）。
- 产物（`outputs/`，不入库）：`q1_out.txt` 完整日志、`q1_coef.npy` 最终模型系数（问题二直接载入）。
- 建模思路与图表需求：[`docs/建模思路/问题一_建模思路.md`](docs/建模思路/问题一_建模思路.md)。
- 绘图脚本：`python src/q1_plot.py`，输出5组必要图到 `figures/q1_v4/`（PNG+PDF）。
- 当前本地模型：sqrt(Y) 的随机截距+孕周随机斜率混合模型；孕周采用中心化自然样条(df=3)，BMI 拆分为基线(between)与孕期内变化(within)。v4 完整精度已通过（1000/1000次聚类Bootstrap成功）；关联结论稳定，但按孕妇5折CV的原始Y尺度R²为负，不能把高条件R²当成新孕妇预测准确率。
- 问题一图表由本地 `src/q1_plot.py` 统一复现；图表规格见建模思路文档第9节。

## 问题二：BMI分组与最佳NIPT时点

- 风险与复检机制：`python src/q2_risk.py`。
- 区间删失AFT：`python src/q2_survival.py`。
- 连续BMI动态分组：`python src/q2_optimize.py`。
- 误差传播与敏感性：`python src/q2_sensitivity.py`。
- 最终集成风险模型和可执行方案：`python src/q2_final.py`。
- 组数由收益捕获率、组间时点可辨性和最小组人数三项显式判据共同选择；当前正式方案为3个连续BMI组，整数边界为31.5和37。
- 绘图脚本：`python src/q2_plot.py`，输出3组最终模型图到 `figures/q2_v1/`（PNG+PDF）。
- 完整说明：[`docs/建模思路/问题二_建模思路.md`](docs/建模思路/问题二_建模思路.md)。

## 问题三：多因素风险下的分组优化（二号部分）

- 核心优化：`python src/q3_optimize.py`，接入一号的多因素达标概率接口后，计算个体期望风险、扫描 `k=1..6`、动态规划连续BMI分组，并全局搜索可执行的整数边界。
- 正式点估计方案为**4组**，整数BMI边界30、32、36；整天推荐时点依次为13周+1天、14周+2天、15周+4天、19周+1天；总体平均风险2.000668。
- 组数由与问题二相同的三条判据选出（收益捕获率≥90%、组间时点间隔≥1周、每组≥15人）。问题二为3组、问题三为4组，差异源于两问使用的达标概率模型不同，判据规则一致，均未硬编码组数；两问的风险数值口径不同，不可直接比较。
- 正式接口的协变量规格为身高+体重+年龄，不含BMI（BMI=体重/身高²为恒等式，与二者代数冗余，VIF 238/107/377）；分组变量仍为基线BMI。
- 局限（已如实披露）：参数协方差抽样中仅48%支持4组、37%支持3组、15%无可行解，正式边界30|32|36仅在16%抽样中重现，应表述为点估计推荐方案而非稳健最优解。
- 敏感性分析：`python src/q3_sensitivity.py --parameter-draws 100 --seed 2026`，重新优化风险梯度、失联率、复检间隔、概率模型变体及AFT参数协方差抽样。
- 正式CSV结果位于`outputs/q3_*.csv`，其中边界稳定性汇总为`q3_boundary_stability.csv`。
- 一号接口未提交前可用 `--mock` 联调；联调结果只写入 `outputs/q3_mock_*`，不得作为正式结论。
- 完整建模思路：[`docs/建模思路/问题三_建模思路.md`](docs/建模思路/问题三_建模思路.md)；模型与优化细节分别保留在对应分工文档中。
- 绘图脚本：`python src/q3_plot.py`，输出5组必要组合图至`figures/q3_v1/`（PNG+PDF）。

## 问题四：女胎异常判定

- 数据与模型：`python src/q4_model.py`，直接读取 `data/processed/female_clean_event.csv`，以554个抽血事件、147位孕妇为正式建模单位；AB非空为事件级异常标签（66/554），AE列不参与建模。行级表仅用于审计和错误单位敏感性分析。
- 信号审计：`python src/q4_signal_audit.py`，分别在事件级、孕妇级和孕妇内检查特征信号，并单列行级 Z 值与 AB 标签的一致性核查。
- 样本外验证：`python src/q4_validate.py`，采用按孕妇分组的5种子×5折重复交叉验证；阈值只在训练折内按“灵敏度不低于90%时特异度最大”选择，亚型模型与总体模型共用折划分。
- 模型冻结：验证完成后运行 `python src/q4_freeze.py`，从验证生成的阈值表冻结最终模型；阈值表缺失或无有效主模型阈值时会直接报错。
- 主模型为L2正则化 Logistic（Z值+质量特征）。PR-AUC为0.484±0.037，ROC-AUC为0.778；多数票混淆矩阵为TP=56、FP=284、TN=204、FN=10。其与全特征Logistic的PR-AUC差0.0066，小于划分噪声0.0469，故选更简单模型。
- 结论边界：模型只复现本数据的AB判定，不能当作临床疾病诊断器；AB与常用Z值阈值明显脱节，且PPV仅0.165，外推必须谨慎。
- 绘图：`python src/q4_plot.py`，输出6组正式图至`figures/q4_v1/`（300 dpi PNG+PDF）。
- 隐私：`outputs/q4_final_predictions.csv` 及其他含 `mother_id` 的逐事件明细仅在本地生成，已从公开仓库跟踪范围排除。
- 完整建模思路：[`docs/建模思路/问题四_建模思路.md`](docs/建模思路/问题四_建模思路.md)；两人非论文分工见[`docs/C题_问题四两人分工.md`](docs/C题_问题四两人分工.md)。

完整复现顺序：`clean.py` → `q4_model.py` → `q4_signal_audit.py` → `q4_validate.py` → `q4_freeze.py` → `q4_plot.py`。
