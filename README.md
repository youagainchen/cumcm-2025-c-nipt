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
- 敏感性分析：`python src/q3_sensitivity.py`，重新优化风险梯度、失联率、复检间隔及一号提供的概率模型变体。
- 一号接口未提交前可用 `--mock` 联调；联调结果只写入 `outputs/q3_mock_*`，不得作为正式结论。
- 优化部分说明：[`docs/建模思路/问题三_优化部分.md`](docs/建模思路/问题三_优化部分.md)。
