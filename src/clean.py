# -*- coding: utf-8 -*-
"""
clean.py —— 丙：C 题数据管道（第一阶段：开局 -> 问题一完成）

按仓库布局约定读取原始附件并产出清洗数据（原始数据只读，产物可复现，不入库）：

  输入   data/raw/附件.xlsx
  输出   data/processed/male_min.csv       T3 快速版（10 列，先解锁甲乙建模）
         data/processed/male_clean.csv     男胎完整清洗版（47 列）
         data/processed/female_clean.csv   女胎完整清洗版（44 列）
         docs/data_report.md               数据画像报告（写论文可直接引用）

设计纪律（来自 docs/C题_第一阶段分工.md，勿违反）：
  * 只打 flag / 做标记，绝不删行 —— 删样本是后续敏感性分析的内容。
  * 女胎标签只能用 AB 列(aneuploidy_raw)。AE 列(fetal_health) 女胎全为「是」，不可作标签。
  * 男胎 1082 条记录只来自 267 位孕妇（重复测量），任何把记录当独立样本的做法都是错的。
  * 中文列名只在读入时出现一次；此后全组只碰英文短名。

运行：  python src/clean.py   （在仓库根目录执行即可，脚本自动定位）
输出 CSV 均为 UTF-8(带 BOM，便于 Excel 打开)。
注：.gitignore 会忽略 data/processed/*，清洗产物不提交、由本脚本复现，符合仓库约定。
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent

# 输入输出位置（仓库约定）
RAW_INPUT = ROOT / "data" / "raw" / "附件.xlsx"
if not RAW_INPUT.exists():                      # 兜底：旧布局
    RAW_INPUT = ROOT / "附件.xlsx"
OUT_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "docs" / "data_report.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 配置
Y_THRESHOLD = 0.04          # 达标阈值：Y 染色体浓度 >= 4%
GC_OK_RANGE = (0.40, 0.60)  # 题面附录1：正常 GC 含量 40%~60%
# 以下为“粗筛”cutoff，仅用于打 flag（不删行），后续敏感性分析可调整：
QC_CUT = {
    "map_ratio": (0.60, None),   # 比对到参考基因组比例过低可疑（数据普遍 ~0.80）
    "dup_ratio": (None, 0.30),   # 重复读段比例过高可疑（数据普遍 ~0.03）
    "filt_ratio": (None, 0.50),  # 被过滤读段占比过高可疑
}

# ---------------------------------------------------------------- 列字母 -> 英文短名
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["AA", "AB", "AC", "AD", "AE"]
MAP = {
    "A": "sample_id",   "B": "mother_id",    "C": "age",       "D": "height",
    "E": "weight",      "F": "lmp",          "G": "ivf",       "H": "draw_date",
    "I": "draw_idx",    "J": "week_raw",     "K": "bmi",       "L": "reads_raw",
    "M": "map_ratio",   "N": "dup_ratio",    "O": "uniq_reads","P": "gc",
    "Q": "z13",         "R": "z18",          "S": "z21",       "T": "zx",
    "U": "zy",          "V": "y_conc",       "W": "x_conc",    "X": "gc13",
    "Y": "gc18",        "Z": "gc21",         "AA": "filt_ratio","AB": "aneuploidy_raw",
    "AC": "gravidity",  "AD": "parity",      "AE": "fetal_health",
}
NUM_LETTERS = set("CDEKLMNOPQRSTUVWXYZ") | {"AA", "AC", "AD"}
NUM_COLS = {MAP[c] for c in NUM_LETTERS}
# 注意：B(mother_id) J(week_raw) F(lmp) G(ivf) H(draw_date) AB/AE 保持原文/类型化

# ---------------------------------------------------------------- 工具
def to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def clean_str(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s != "" else None


re_week_plus = re.compile(r"(\d+(?:\.\d+)?)w\+\s*(\d+)")
re_week_plain = re.compile(r"(\d+(?:\.\d+)?)w")


def parse_week(v):
    """'11w+6' -> 11 + 6/7 ≈ 11.857。容错：'13w'、空格、全角 ＋、W/周。
    无法解析返回 None。"""
    s = clean_str(v)
    if s is None:
        return None
    s = s.replace("＋", "+").replace("（", "(").replace("）", ")")
    s = s.replace("周", "w").replace("W", "w").replace(" ", "")
    m = re_week_plus.search(s)
    if m:
        return float(m.group(1)) + float(m.group(2)) / 7.0
    m = re_week_plain.search(s)
    if m:
        return float(m.group(1))
    return to_float(s)


def quantile(vals, p):
    n = len(vals)
    if n == 0:
        return None
    s = sorted(vals)
    idx = (n - 1) * p
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def fmt(x, nd=4):
    if x is None:
        return "—"
    t = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return t if t not in ("", "-") else "0"


def pct(n, d):
    return f"{n / d * 100:.1f}%" if d else "—"


# ---------------------------------------------------------------- 读取 + 命名/类型化
def read_sheet(sheet):
    it = sheet.iter_rows(values_only=True)
    header = next(it)  # 仅展示用；列定位一律用字母（女胎 U/V 表头为空）
    raw = []
    for r in it:
        raw.append({LETTERS[i]: (r[i] if i < len(r) else None) for i in range(len(LETTERS))})
    return header, raw


def to_record(row):
    rec = {}
    for L, name in MAP.items():
        v = row[L]
        rec[name] = to_float(v) if name in NUM_COLS else clean_str(v)
    rec["week_raw_s"] = clean_str(row["J"])
    rec["week"] = parse_week(rec["week_raw_s"])
    return rec


# ---------------------------------------------------------------- 派生
def add_derived(recs):
    """visit_idx / n_visits：按母体在表中的出现顺序编号（第几次检测/共几次）。"""
    seq, tot = Counter(), Counter()
    for r in recs:
        tot[r["mother_id"]] += 1
    for r in recs:
        seq[r["mother_id"]] += 1
        r["visit_idx"] = seq[r["mother_id"]]
        r["n_visits"] = tot[r["mother_id"]]


def add_qc_flags(recs):
    for r in recs:
        gc = r["gc"]
        r["flag_gc"] = 1 if (gc is not None
                             and not (GC_OK_RANGE[0] <= gc <= GC_OK_RANGE[1])) else 0
        for col, (lo, hi) in QC_CUT.items():
            v = r[col]
            bad = v is not None and ((lo is not None and v < lo)
                                     or (hi is not None and v > hi))
            r["flag_" + col] = 1 if bad else 0
        r["flag_any"] = 1 if (r["flag_gc"] or r["flag_map_ratio"]
                              or r["flag_dup_ratio"] or r["flag_filt_ratio"]) else 0


def add_male_q2_marks(male):
    """问题二预置：每位孕妇的「Y 浓度首次达标」区间与删失标记。

    - reached:            该孕妇是否至少一次达标
    - first_pass_week/visit: 首次达标的孕周与第几次检测
    - t_left/t_right:      首次达标落在 (t_left, t_right]
      * censored='none'   达标前观测到未达标行（达标落在两次检测之间）
      * censored='left'   首测即达标（真实达标 <= t_right，下界取 0）
      * censored='right'  全程未达标，右删失（真实达标 > 最后一次检测孕周）
    不改变行在表中的顺序（按母体分组排序只是为了让字段计算正确）。
    """
    groups = {}
    for r in male:
        groups.setdefault(r["mother_id"], []).append(r)
    for rows in groups.values():
        rows.sort(key=lambda x: (x["week"] if x["week"] is not None else -1,
                                 x["visit_idx"]))
        quals = [r for r in rows if r["is_qualified"]]
        reached = len(quals) > 0
        fp_week = quals[0]["week"] if reached else None
        fp_visit = quals[0]["visit_idx"] if reached else None
        last_week = max((r["week"] for r in rows if r["week"] is not None), default=None)
        if reached:
            pre_fail = [r for r in rows if not r["is_qualified"]
                        and r["week"] is not None and r["week"] < fp_week]
            if pre_fail:
                censored, t_left, t_right = "none", max(r["week"] for r in pre_fail), fp_week
            else:
                censored, t_left, t_right = "left", 0.0, fp_week
        else:
            censored, t_left, t_right = "right", last_week, None
        for r in rows:
            r["reached"] = reached
            r["first_pass_week"] = fp_week
            r["first_pass_visit"] = fp_visit
            r["t_left"] = t_left
            r["t_right"] = t_right
            r["censored"] = censored


def add_female_labels(recs):
    """女胎标签只能取自 aneuploidy_raw（AB 列）。"""
    for r in recs:
        ab = r["aneuploidy_raw"] or ""
        r["label"] = 1 if ab else 0
        r["label_T13"] = 1 if "T13" in ab else 0
        r["label_T18"] = 1 if "T18" in ab else 0
        r["label_T21"] = 1 if "T21" in ab else 0


def write_csv(path, recs, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)


# ---------------------------------------------------------------- 报告
def make_report(male, female):
    L = []
    def line(s=""):
        L.append(s)

    nm, nf = len(male), len(female)
    line("# 数据报告 data_report（由 src/clean.py 自动生成）")
    line("")
    line(f"- 数据来源：data/raw/附件.xlsx（男胎 {nm} 条 / 女胎 {nf} 条）")
    line("- 复现：`python src/clean.py`（产物在 data/processed/，不入库）")
    line(f"- 达标阈值：Y 浓度 ≥ {Y_THRESHOLD}（4%），女胎标签仅用 AB 列")

    # 一、基本画像
    line("")
    line("## 一、基本画像")
    for tag, recs in (("男胎", male), ("女胎", female)):
        mothers = {r["mother_id"] for r in recs}
        seen = set()
        per_mother = []
        for r in recs:
            if r["mother_id"] not in seen:
                seen.add(r["mother_id"])
                per_mother.append(r)
        vdist = Counter(r["n_visits"] for r in per_mother)
        weeks = [r["week"] for r in recs if r["week"] is not None]
        bmis = [r["bmi"] for r in recs if r["bmi"] is not None]
        ages = [r["age"] for r in recs if r["age"] is not None]
        line("")
        line(f"### {tag}")
        line(f"- 记录数 **{len(recs)}**，唯一孕妇 **{len(mothers)}** 人。")
        line(f"- 每人检测次数分布：{dict(sorted(vdist.items()))}"
             f"（同一孕妇多次检测=同一体的重复观测，不是独立样本！）")
        line(f"- 孕周 {fmt(min(weeks),1)}–{fmt(max(weeks),1)} 周（非空 {len(weeks)}/{len(recs)}）")
        line(f"- BMI {fmt(min(bmis))}–{fmt(max(bmis))}（非空 {len(bmis)}/{len(recs)}）")
        line(f"- 年龄 {fmt(min(ages))}–{fmt(max(ages))} 岁（非空 {len(ages)}/{len(recs)}）")
        if tag == "男胎":
            yc = [r["y_conc"] for r in recs if r["y_conc"] is not None]
            line(f"- Y 浓度 min={fmt(min(yc))} 中位={fmt(quantile(yc, 0.5))} max={fmt(max(yc))}"
                 f"（非空 {len(yc)}/{len(recs)}）")

    # 二、缺失
    line("")
    line("## 二、缺失值（仅列非全空项）")
    for tag, recs in (("男胎", male), ("女胎", female)):
        mt = [(name, sum(1 for r in recs if r.get(name) is None))
              for name in MAP.values()]
        mt = [(c, n) for c, n in mt if n]
        if not mt:
            line(f"**{tag}**：无缺失。")
            continue
        line(f"**{tag}**：")
        for c, n in mt:
            line(f"- `{c}`：缺失 {n}/{len(recs)}（{pct(n, len(recs))}）")

    # 三、BMI 核对
    line("")
    line("## 三、BMI 核对（用身高体重重算 vs 原 K 列）")
    for tag, recs in (("男胎", male), ("女胎", female)):
        n_both = n_mis = 0
        max_dev = 0.0
        for r in recs:
            h, w, b = r["height"], r["weight"], r["bmi"]
            if None in (h, w, b):
                continue
            n_both += 1
            d = abs(w / (h / 100.0) ** 2 - b)
            if d > 1e-6:
                n_mis += 1
                max_dev = max(max_dev, d)
        line(f"- {tag}：可核对 {n_both} 条；|重算−原值|>1e-6 的 **{n_mis}** 条，"
             f"最大偏差 {max_dev:.4f}。"
             + ("两者一致。" if n_mis == 0 else "不一致需讨论。"))

    # 四、QC 体检
    line("")
    line("## 四、测序质量体检（只打 flag，不删行）")
    line(f"- flag 阈值：GC 不在 {GC_OK_RANGE[0]}–{GC_OK_RANGE[1]}，或 "
         f"map_ratio<{QC_CUT['map_ratio'][0]}、dup_ratio>{QC_CUT['dup_ratio'][1]}、"
         f"filt_ratio>{QC_CUT['filt_ratio'][1]} 之一即为可疑（可调，供敏感性分析）。")
    for tag, recs in (("男胎", male), ("女胎", female)):
        line(f"### {tag}")
        for col in ("gc", "map_ratio", "dup_ratio", "uniq_reads", "filt_ratio", "reads_raw"):
            vec = [r[col] for r in recs]
            nn = [x for x in vec if x is not None]
            if not nn:
                continue
            q1, q3 = quantile(nn, 0.25), quantile(nn, 0.75)
            iqr = q3 - q1
            n_out = sum(1 for x in nn if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr)
            flag = sum(1 for r in recs if r.get("flag_" + col, 0))
            line(f"- `{col}`：min={fmt(min(nn))} 中位={fmt(quantile(nn, 0.5))} "
                 f"max={fmt(max(nn))}；IQR 离群 ≈{n_out}；按阈值打 flag {flag} 条。")
        gc_a = sum(1 for r in recs if r["flag_gc"])
        any_a = sum(1 for r in recs if r["flag_any"])
        line(f"- 汇总：GC 超标 {gc_a} 条；四项任一可疑 **{any_a}** 条"
             f"（占 {pct(any_a, len(recs))}）。")

    # 五、男胎问题二标记
    line("")
    line("## 五、男胎：Y 浓度达标标记（问题二输入）")
    mothers_m = {r["mother_id"] for r in male}
    first_rows = {}
    for r in male:
        first_rows.setdefault(r["mother_id"], r)
    n_reach = sum(1 for r in first_rows.values() if r["reached"])
    cens = Counter(r["censored"] for r in first_rows.values())
    line(f"- 曾达标的孕妇 **{n_reach}** / {len(mothers_m)} 人；删失：{dict(cens)}。")
    line("  - `none`：达标落在 (t_left, t_right]，即两次检测之间首次出现达标")
    line("  - `left`：首测即达标（真实达标 ≤ t_right，下界 0）")
    line("  - `right`：全程未达标，右删失（真实达标 > 最后一次检测孕周）")
    fp = sorted(r["first_pass_week"] for r in first_rows.values() if r["reached"])
    if fp:
        line(f"- 首次达标孕周（受检孕妇中）：{fmt(fp[0],1)}–{fmt(fp[-1],1)}，"
             f"中位 {fmt(quantile(fp, 0.5),1)}（n={len(fp)}）")

    # 六、女胎标签
    line("")
    line("## 六、女胎标签（仅用 AB 列 aneuploidy_raw）")
    line("- ⚠️ AE 列(fetal_health) 女胎表全部为「是」，不可作标签（否则为全常数、模型失效）。")
    ab_n = sum(1 for r in female if r["label"])
    line(f"- label（AB 非空=异常）：**{ab_n}** / {nf} 条异常。")
    for c in ("label_T13", "label_T18", "label_T21"):
        line(f"- `{c}`：{sum(1 for r in female if r[c])} 条。")
    vals = Counter(r["aneuploidy_raw"] for r in female if r["aneuploidy_raw"])
    line(f"- AB 取值分布：{dict(vals)}")

    # 七、孕周解析
    bad = [r for r in male + female if r["week"] is None]
    line("")
    line("## 七、孕周解析")
    line(f"- 无法解析 {len(bad)} 条（占 {pct(len(bad), len(male) + len(female))}）"
         + (f"，样例 {[r['week_raw_s'] for r in bad[:5]]}" if bad else "。"))

    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 主流程
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not RAW_INPUT.exists():
        raise SystemExit("未找到原始数据：" + str(RAW_INPUT))
    wb = openpyxl.load_workbook(RAW_INPUT, read_only=True, data_only=True)
    male_ws, female_ws = wb.worksheets[0], wb.worksheets[1]

    _, male_rows = read_sheet(male_ws)
    _, female_rows = read_sheet(female_ws)
    male = [to_record(r) for r in male_rows]
    female = [to_record(r) for r in female_rows]

    # 1) 男胎达标标记行
    for r in male:
        r["is_qualified"] = 1 if (r["y_conc"] is not None
                                  and r["y_conc"] >= Y_THRESHOLD) else 0

    # 2) 派生列（母体维）
    add_derived(male)
    add_derived(female)
    add_male_q2_marks(male)   # 依赖 visit_idx 与 is_qualified
    add_qc_flags(male)
    add_qc_flags(female)
    add_female_labels(female)

    # 3) 列集合
    base = [MAP[c] for c in LETTERS]  # 31 列（sample_id .. fetal_health）
    male_clean_cols = base + [
        "week", "week_raw_s", "visit_idx", "n_visits", "is_qualified",
        "reached", "first_pass_week", "first_pass_visit", "t_left", "t_right", "censored",
        "flag_gc", "flag_map_ratio", "flag_dup_ratio", "flag_filt_ratio", "flag_any",
    ]
    female_clean_cols = base + [
        "week", "week_raw_s", "visit_idx", "n_visits",
        "label", "label_T13", "label_T18", "label_T21",
        "flag_gc", "flag_map_ratio", "flag_dup_ratio", "flag_filt_ratio", "flag_any",
    ]
    male_min_cols = ["mother_id", "age", "height", "weight", "bmi", "week",
                     "y_conc", "draw_idx", "visit_idx", "n_visits"]

    # 4) 写数据
    write_csv(OUT_DIR / "male_min.csv", male, male_min_cols)
    write_csv(OUT_DIR / "male_clean.csv", male, male_clean_cols)
    write_csv(OUT_DIR / "female_clean.csv", female, female_clean_cols)
    REPORT_PATH.write_text(make_report(male, female), encoding="utf-8")

    print("OK -> data/processed/{male_min,male_clean,female_clean}.csv  +  docs/data_report.md")
    print(f"male rows={len(male)}  female rows={len(female)}")


if __name__ == "__main__":
    main()
