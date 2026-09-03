# -*- coding: utf-8 -*-
"""
clean.py —— 丙：C 题数据管道（第一阶段，v3，按队友质检意见修订）

按仓库布局约定读取原始附件并产出清洗数据（原始数据只读，产物可复现，不入库）：

  输入   data/raw/附件.xlsx
  输出   data/processed/male_min.csv             T3 快速版（含技术重复标记）
         data/processed/male_clean.csv           男胎完整清洗版（行级）
         data/processed/female_clean.csv         女胎完整清洗版（行级）
         data/processed/male_clean_event.csv     男胎「抽血事件级」（每 (孕妇,抽血) 一行）
         data/processed/female_clean_event.csv   女胎「抽血事件级」
         docs/data_report.md                     数据画像报告

v3 相对 v1/v2 的关键修订（对应队友质检 4 个问题）：
  1) 语义修正：visit_idx/n_visits = 该孕妇「第几次抽血 / 共抽血几次」，抽血按检测日期时序编号
     （同一抽血的多行共享同一 visit_idx，行序不再是“检测次数”）。技术重复由 rep_* 列标识。
  2) 达标/删失标记上移到「事件级」，默认口径=抽血组内均值 ≥ 阈值（避免“任一行达标=取最大、
     最乐观、系统性低估首次达标孕周”），同时保留 any 口径作敏感性对照（见 *_any 后缀）。
  3) 源数据孕周时序异常打 flag_chrono（例：A151 抽血日期更晚但孕周回退），只标不修。
  4) GC 阈值改为稳健离群(箱线)打 flag_gc；题面 40%~60% 只作参考统计单列，避免 41% 假红。
  —— 事件内多次测序 = 技术重复（测量误差），事件之间 = 随访（纵向），孕妇之间 = 个体差异；
     供三层嵌套模型使用（孕妇/抽血/测序）。

数据纪律（分工文档 + 评阅要点）：
  * 只打 flag / 做标记，绝不删行、不插补、不做类别平衡。
  * 女胎标签只能用 AB 列(aneuploidy_raw)。AE 列女胎全「是」，不可作标签。
  * 男胎 1082 行 / 1021 抽血事件 / 267 孕妇，任何把行或事件当独立样本的做法都是错的。

运行：  python src/clean.py   （在仓库根目录执行）
输出 CSV 均 UTF-8(带 BOM)。.gitignore 忽略 data/processed/*，由本脚本复现。
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
RAW_INPUT = ROOT / "data" / "raw" / "附件.xlsx"
if not RAW_INPUT.exists():
    RAW_INPUT = ROOT / "附件.xlsx"
OUT_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "docs" / "data_report.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 配置
Y_THRESHOLD = 0.04          # 达标阈值：Y 染色体浓度 >= 4%
WEEK_REGRESS_EPS = 0.5      # 孕周“回退”判定的容差（周）
QC_CUT = {                  # 测序比例类 flag（可调，仅标记不删行）
    "map_ratio": (0.60, None),
    "dup_ratio": (None, 0.30),
    "filt_ratio": (None, 0.50),
}

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
REP_COLS = ["rep_in_draw", "n_reps_in_draw", "is_tech_repeat"]

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
    s = clean_str(v)
    if s is None:
        return None
    s = (s.replace("＋", "+").replace("（", "(").replace("）", ")")
          .replace("周", "w").replace("W", "w").replace(" ", ""))
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


# ---------------------------------------------------------------- 读取/命名
def read_sheet(sheet):
    it = sheet.iter_rows(values_only=True)
    header = next(it)  # 列定位一律用字母（女胎 U/V 表头为空）
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


# ---------------------------------------------------------------- 抽血事件 & 时序
def draw_key_date(draw):
    """抽血的时序键：取组内最早 draw_date（缺失取极大值），并列时用 draw_idx。"""
    ds = [r["draw_date"] for r in draw if r["draw_date"] is not None]
    return (min(ds) if ds else float("inf"))


def add_draw_visit(recs):
    """核心语义：把每次“抽血”(draw_idx) 当作一个事件/一次随访。

    - visit_idx  = 该抽血在其孕妇所有抽血里的时序编号（按最早检测日期排，并列按 draw_idx）
    - n_visits   = 该孕妇的抽血次数（事件数）
    - rep_in_draw / n_reps_in_draw / is_tech_repeat = 一次抽血内多次测序（技术重复）
    - flag_chrono= 该抽血的代表孕周 比 上一抽血（按时序）回退了超过 WEEK_REGRESS_EPS 周
    不改变行在表中的顺序。
    """
    groups = OrderedDict()
    for r in recs:
        groups.setdefault(r["mother_id"], []).append(r)
    for rows in groups.values():
        draws = OrderedDict()
        for r in rows:
            draws.setdefault(r["draw_idx"], []).append(r)
        ordered = sorted(draws.values(), key=lambda d: (draw_key_date(d),
                                                        (d[0]["draw_idx"] or 0)))
        n = len(ordered)
        prev_w = None
        for vi, draw in enumerate(ordered, start=1):
            weeks = [r["week"] for r in draw if r["week"] is not None]
            w_rep = (sum(weeks) / len(weeks)) if weeks else None
            chrono = 0
            if (prev_w is not None and w_rep is not None
                    and w_rep < prev_w - WEEK_REGRESS_EPS):
                chrono = 1
            prev_w = w_rep if w_rep is not None else prev_w
            k = len(draw)
            for ri, r in enumerate(draw, start=1):
                r["visit_idx"] = vi
                r["n_visits"] = n
                r["rep_in_draw"] = ri
                r["n_reps_in_draw"] = k
                r["is_tech_repeat"] = 1 if k > 1 else 0
                r["flag_chrono"] = chrono


def add_qc_flags(recs):
    """测序质量体检（只打 flag，不删行）。GC 用稳健离群(箱线)，不直接套 40%~60%。"""
    gc_vals = [r["gc"] for r in recs if r["gc"] is not None]
    if len(gc_vals) >= 4:
        q1, q3 = quantile(gc_vals, 0.25), quantile(gc_vals, 0.75)
        gc_lo, gc_hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    else:
        gc_lo, gc_hi = -1e9, 1e9
    for r in recs:
        gc = r["gc"]
        r["flag_gc"] = 1 if (gc is not None and not (gc_lo <= gc <= gc_hi)) else 0
        for col, (lo, hi) in QC_CUT.items():
            v = r[col]
            bad = v is not None and ((lo is not None and v < lo)
                                     or (hi is not None and v > hi))
            r["flag_" + col] = 1 if bad else 0
        r["flag_any"] = 1 if (r["flag_gc"] or r["flag_map_ratio"]
                              or r["flag_dup_ratio"] or r["flag_filt_ratio"]
                              or r["flag_chrono"]) else 0


def add_female_labels(recs):
    for r in recs:
        ab = r["aneuploidy_raw"] or ""
        r["label"] = 1 if ab else 0
        r["label_T13"] = 1 if "T13" in ab else 0
        r["label_T18"] = 1 if "T18" in ab else 0
        r["label_T21"] = 1 if "T21" in ab else 0


# ---------------------------------------------------------------- 事件级聚合
def build_event_records(recs, is_male):
    """每个 (孕妇, draw_idx) 一行；行内属性先由 add_draw_visit 写好 visit/rep/chrono。"""
    groups = OrderedDict()
    for r in recs:
        groups.setdefault(r["mother_id"], []).append(r)
    evs = []
    for mid, rows in groups.items():
        by_draw = OrderedDict()
        for r in rows:
            by_draw.setdefault(r["draw_idx"], []).append(r)
        for dr in sorted(by_draw, key=lambda x: (x is None, x)):
            rep = by_draw[dr]
            weeks = [r["week"] for r in rep if r["week"] is not None]
            ev = {
                "mother_id": mid,
                "draw_idx": dr,
                "visit_idx": rep[0]["visit_idx"],
                "n_reps": len(rep),
                "is_tech_repeat": rep[0]["is_tech_repeat"],
                "flag_chrono": max(r["flag_chrono"] for r in rep),
                "age": next((r["age"] for r in rep if r["age"] is not None), None),
                "height": next((r["height"] for r in rep if r["height"] is not None), None),
                "weight": next((r["weight"] for r in rep if r["weight"] is not None), None),
                "bmi": next((r["bmi"] for r in rep if r["bmi"] is not None), None),
                "draw_date": min((r["draw_date"] for r in rep if r["draw_date"] is not None),
                                 default=None),
                "week_min": min(weeks) if weeks else None,
                "week_max": max(weeks) if weeks else None,
                "week_mean": (sum(weeks) / len(weeks)) if weeks else None,
            }
            if is_male:
                y = [r["y_conc"] for r in rep if r["y_conc"] is not None]
                m = (sum(y) / len(y)) if y else None
                ev.update({
                    "y_conc_n": len(y),
                    "y_conc_min": min(y) if y else None,
                    "y_conc_mean": m,
                    "y_conc_max": max(y) if y else None,
                    "y_conc_sd": ((sum((v - m) ** 2 for v in y) / (len(y) - 1)) ** 0.5
                                  if len(y) >= 2 else None),
                    "qual_any": 1 if any(v >= Y_THRESHOLD for v in y) else 0,
                    "qual_mean": 1 if (m is not None and m >= Y_THRESHOLD) else 0,
                    "flag_any": 1 if any(r["flag_any"] for r in rep) else 0,
                })
            else:
                ab = sorted({r["aneuploidy_raw"] for r in rep if r["aneuploidy_raw"]})
                ev.update({
                    "label": 1 if any(r["label"] for r in rep) else 0,
                    "label_T13": 1 if any(r["label_T13"] for r in rep) else 0,
                    "label_T18": 1 if any(r["label_T18"] for r in rep) else 0,
                    "label_T21": 1 if any(r["label_T21"] for r in rep) else 0,
                    "aneuploidy_raw": "/".join(ab) if ab else None,
                    "flag_any": 1 if any(r["flag_any"] for r in rep) else 0,
                })
            evs.append(ev)
    return evs


def add_event_survival(ev_male):
    """事件级“首次达标”区间标记（问题二输入），时间轴=孕周。

    默认口径 = 事件内抽血均值 ≥ 阈值（qual_mean）；同时给出 any 口径（qual_any）做对照。
    每个孕妇：事件按 (week_mean, visit_idx) 升序。
    - reached / first_pass_week / first_pass_visit / t_left / t_right / censored
      （无后缀 = qual_mean 口径；带 _any 后缀 = qual_any 口径）
    censored：none=落在(t_left,t_right]之间首达；left=首抽即达(真实≤t_right,下界0)；
              right=全程未达(右删失，真实>最后一次抽血孕周)。
    """
    groups = OrderedDict()
    for e in ev_male:
        groups.setdefault(e["mother_id"], []).append(e)

    def run(qualcol, suf):
        for e in ev_male:  # 复位
            e["reached" + suf] = False
            e["first_pass_week" + suf] = None
            e["first_pass_visit" + suf] = None
            e["t_left" + suf] = None
            e["t_right" + suf] = None
            e["censored" + suf] = None
        for rows in groups.values():
            rows.sort(key=lambda e: (e["week_mean"] if e["week_mean"] is not None else 1e9,
                                     e["visit_idx"]))
            hits = [e for e in rows if e[qualcol]]
            hit = hits[0] if hits else None
            fp_w = hit["week_mean"] if hit else None
            fp_v = hit["visit_idx"] if hit else None
            last_w = max((e["week_mean"] for e in rows if e["week_mean"] is not None),
                         default=None)
            if hit is not None:
                pre = [e for e in rows if not e[qualcol]
                       and e["week_mean"] is not None and e["week_mean"] < fp_w]
                if pre:
                    ce, tl, tr = "none", max(e["week_mean"] for e in pre), fp_w
                else:
                    ce, tl, tr = "left", 0.0, fp_w
            else:
                ce, tl, tr = "right", last_w, None
            for e in rows:
                e["reached" + suf] = hit is not None
                e["first_pass_week" + suf] = fp_w
                e["first_pass_visit" + suf] = fp_v
                e["t_left" + suf] = tl
                e["t_right" + suf] = tr
                e["censored" + suf] = ce

    run("qual_mean", "")
    run("qual_any", "_any")


def attach_survival_to_rows(male, ev_male):
    """把事件级（默认 mean 口径）首达标记回填到行，同一孕妇同抽血各行列相同。"""
    key = {}
    for e in ev_male:
        key.setdefault((e["mother_id"], e["draw_idx"]), e)
    for r in male:
        e = key.get((r["mother_id"], r["draw_idx"]))
        if e is None:
            continue
        for c in ("reached", "first_pass_week", "first_pass_visit",
                  "t_left", "t_right", "censored"):
            r[c] = e[c]


# ---------------------------------------------------------------- 输出
def write_csv(path, recs, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)


# ---------------------------------------------------------------- 报告
def make_report(male, female, ev_male, ev_female):
    L = []
    def line(s=""):
        L.append(s)

    nm, nf = len(male), len(female)
    line("# 数据报告 data_report（由 src/clean.py v3 生成）")
    line("")
    line(f"- 数据来源：data/raw/附件.xlsx（男胎行 {nm} / 女胎行 {nf}）")
    line(f"- 事件级（一次抽血一行）：男胎 {len(ev_male)} / 女胎 {len(ev_female)}")
    line("- 复现：`python src/clean.py`（产物 data/processed/，不入库）")
    line(f"- 达标阈值 Y ≥ {Y_THRESHOLD}；女胎标签仅用 AB 列")
    line("")
    line("## 〇、数据可用性声明（对照评阅要点）")
    line("- 附件数据除少量空缺/录入不一致外无需常规预处理：本管道**不删行、不插补、"
         "不做类别平衡**，只加派生列与标记。")
    line("- 数据含不可忽略的测量误差：以 (孕妇, 抽血) 区分**技术重复(一次采血多次测序)**，"
         "其组内离散度量化测量误差；随访/个体差异在更高层。推荐三层嵌套模型"
         "（孕妇/抽血/测序）—— 对应问题二、三对“检测误差影响”的要求。")
    line("- 识别到的空缺/时序异常仅记录、不处置（见第九节）。")

    # 一、基本画像（事件级为主）
    line("")
    line("## 一、基本画像")
    for tag, recs, evs in (("男胎", male, ev_male), ("女胎", female, ev_female)):
        mothers = {r["mother_id"] for r in recs}
        nmap = Counter()
        for e in evs:
            nmap[e["mother_id"]] += 1          # 事件数 = 该孕妇抽血次数
        draws_dist = Counter(nmap.values())
        edist = Counter(e["n_reps"] for e in evs)
        weeks = [r["week"] for r in recs if r["week"] is not None]
        bmis = [r["bmi"] for r in recs if r["bmi"] is not None]
        line("")
        line(f"### {tag}")
        line(f"- 行级 **{len(recs)}** / 抽血事件 **{len(evs)}** / 孕妇 **{len(mothers)}** 人。")
        line(f"- 孕妇抽血次数分布：{dict(sorted(draws_dist.items()))}（抽血=一次随访）")
        line(f"- 每抽血含测序次数分布(1=该次只测1次)：{dict(sorted(edist.items()))}")
        line(f"- 孕周 {fmt(min(weeks),1)}–{fmt(max(weeks),1)}（非空 {len(weeks)}/{len(recs)}）")
        line(f"- BMI {fmt(min(bmis))}–{fmt(max(bmis))}（非空 {len(bmis)}/{len(recs)}）")
        if tag == "男胎":
            yc = [r["y_conc"] for r in recs if r["y_conc"] is not None]
            line(f"- Y 浓度 min={fmt(min(yc))} 中位={fmt(quantile(yc,0.5))} "
                 f"max={fmt(max(yc))}（非空 {len(yc)}/{len(recs)}）")

    # 二、缺失
    line("")
    line("## 二、缺失值（仅列非全空项；只记录不处置）")
    for tag, recs in (("男胎", male), ("女胎", female)):
        mt = [(c, sum(1 for r in recs if r.get(c) is None)) for c in MAP.values()]
        mt = [(c, n) for c, n in mt if n]
        line(f"**{tag}**：" + ("无缺失。" if not mt else
                               "；".join(f"`{c}` 缺 {n}/{len(recs)}（{pct(n, len(recs))}）"
                                         for c, n in mt)))

    # 三、BMI 核对
    line("")
    line("## 三、BMI 核对（身高体重重算 vs 原 K 列）")
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
        line(f"- {tag}：可核对 {n_both}；不一致 {n_mis} 条，最大偏差 {max_dev:.4f}。"
             + ("一致。" if n_mis == 0 else "偏差极小，建模用原 K 列。"))

    # 四、QC 体检
    line("")
    line("## 四、测序质量体检（只打 flag 不删行；GC 用稳健离群）")
    for tag, recs in (("男胎", male), ("女胎", female)):
        gc_vals = [r["gc"] for r in recs if r["gc"] is not None]
        nominal = sum(1 for g in gc_vals if g is not None and
                      not (0.40 <= g <= 0.60))
        line(f"### {tag}")
        for col in ("gc", "map_ratio", "dup_ratio", "uniq_reads", "filt_ratio", "reads_raw"):
            nn = [r[col] for r in recs if r[col] is not None]
            if not nn:
                continue
            q1, q3 = quantile(nn, 0.25), quantile(nn, 0.75)
            iqr = q3 - q1
            n_out = sum(1 for x in nn if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr)
            flag = sum(1 for r in recs if r.get("flag_" + col, 0))
            line(f"- `{col}`：min={fmt(min(nn))} 中位={fmt(quantile(nn,0.5))} "
                 f"max={fmt(max(nn))}；稳健离群≈{n_out}；flag {flag} 条。")
        line(f"- GC 名义区间 40%~60% 之外（参考，非删除 flag）：{nominal}/{len(gc_vals)} 条"
             f"（本数据 GC 集中在 ~0.40 附近，固定区间不适配，故 flag_gc 用稳健离群）。")
        chr_n = sum(1 for r in recs if r["flag_chrono"])
        any_a = sum(1 for r in recs if r["flag_any"])
        line(f"- 时序回退 flag_chrono：{chr_n} 行；任一 flag：**{any_a}** 行"
             f"（占 {pct(any_a, len(recs))}）。")

    # 五、男胎达标（事件级）
    line("")
    line("## 五、男胎：Y 达标 & 首次达标（**事件级**，问题二输入）")
    first_m = {}
    for e in ev_male:
        first_m.setdefault(e["mother_id"], []).append(e)
    for label, suf in (("均值口径(默认)", ""), ("任一重复口径(对照)", "_any")):
        n_reach = sum(1 for rows in first_m.values()
                      if any(e["reached" + suf] for e in rows))
        cens = Counter()
        for rows in first_m.values():
            rows_sorted = sorted(rows, key=lambda e: e["visit_idx"])
            cens[rows_sorted[0]["censored" + suf]] += 1
        fp = []
        for rows in first_m.values():
            v = [e["first_pass_week" + suf] for e in rows if e["reached" + suf]
                 and e["first_pass_week" + suf] is not None]
            if v:
                fp.append(v[0])
        fp = sorted(fp)
        line(f"- 按**{label}**：曾达标孕妇 {n_reach}/{len(first_m)}；删失 {dict(cens)}；")
        if fp:
            line(f"  首次达标孕周 {fmt(min(fp),1)}–{fmt(max(fp),1)}，"
                 f"中位 {fmt(quantile(fp,0.5),1)}（n={len(fp)} 孕妇）。")
    # 两口径不一致的孕妇
    diffs = []
    for mid, rows in first_m.items():
        def fpw(suf):
            hs = sorted([e for e in rows if e["reached" + suf]],
                        key=lambda e: e["first_pass_week" + suf] or 1e9)
            return hs[0]["first_pass_week" + suf] if hs else None
        a, b = fpw(""), fpw("_any")
        if a is not None and b is not None and abs(a - b) > 1e-9:
            diffs.append((mid, b, a))
    if diffs:
        line(f"- 两口径首次达标不同的孕妇：{len(diffs)} 位 → " +
             "; ".join(f"{m}(any {x:.2f}→mean {y:.2f})" for m, x, y in sorted(diffs)))
        line("  （结论：any 口径=对同一次抽血取最大，系统性更早；默认请用均值口径。）")

    # 六、女胎标签
    line("")
    line("## 六、女胎标签（仅用 AB 列）")
    ab_n = sum(1 for e in ev_female if e["label"])
    line(f"- AE 列女胎全「是」不可作标签；label 异常：{ab_n}/{len(ev_female)} 事件、"
         f"{sum(1 for r in female if r['label'])}/{len(female)} 行。标签天然不平衡，"
         "**不做平衡/过采样**，建模走统计判定。")

    # 七、孕周解析
    bad = [r for r in male + female if r["week"] is None]
    line("")
    line(f"## 七、孕周解析：无法解析 {len(bad)} 条"
         + (f"，样例 {[r['week_raw_s'] for r in bad[:5]]}" if bad else "。"))

    # 八、技术重复 / 测量误差
    line("")
    line("## 八、技术重复（一次采血多次测序，量化测量误差）")
    rep_m = [e for e in ev_male if e["n_reps"] >= 2]
    rep_f = [e for e in ev_female if e["n_reps"] >= 2]
    m_rep_mothers = {e["mother_id"] for e in rep_m}
    rows_aff = sum(1 for r in male if r["mother_id"] in m_rep_mothers)
    line(f"- 男胎：{len(rep_m)} 个抽血事件含重复（占 {pct(len(rep_m), len(ev_male))}），"
         f"涉及 {len(m_rep_mothers)} 位孕妇 / 这 {len(m_rep_mothers)} 位共 {rows_aff} 行；"
         f"技术重复行 {sum(e['n_reps'] for e in rep_m)} 条。")
    line(f"- 女胎：{len(rep_f)} 事件含重复（占 {pct(len(rep_f), len(ev_female))}）。")
    sds = [e["y_conc_sd"] for e in rep_m if e["y_conc_sd"] is not None]
    if sds:
        line(f"- 男胎事件内 Y 浓度 SD（测量误差量级，n≥2 共 {len(sds)} 事件）："
             f"min={fmt(min(sds))} 中位={fmt(quantile(sds,0.5))} max={fmt(max(sds))}。")

    # 九、识别到的空缺 / 时序异常
    line("")
    line("## 九、识别到的空缺 / 时序/录入异常（仅记录，不处置）")
    line("- 空缺：见第二节（男胎 `lmp`12、`gravidity`300；女胎 `lmp`8、`bmi`1、`gravidity`165）。")
    for tag, recs in (("男胎", male), ("女胎", female)):
        ch = sorted({(r["mother_id"], r["draw_idx"]) for r in recs if r["flag_chrono"]})
        if ch:
            line(f"- {tag} 时序回退（孕周未随抽血日期推进，flag_chrono=1）："
                 f"{len(ch)} 个抽血事件 → " +
                 ", ".join(f"{m}#{d}" for m, d in ch[:8]) + ("…" if len(ch) > 8 else ""))
    for tag, recs in (("男胎", male), ("女胎", female)):
        # 同孕妇固定属性不一致
        per = {}
        for r in recs:
            for col in ("height", "age"):
                if r.get(col) is None:
                    continue
                per.setdefault((col, r["mother_id"]), set()).add(r[col])
        for col in ("height", "age"):
            n_inc = sum(1 for (c, _m), vs in per.items() if c == col and len(vs) > 1)
            if n_inc:
                line(f"- {tag}：{n_inc} 位孕妇 `{col}` 跨行不一致（多为录入差异/周岁变化）。")

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

    # 行级：抽血事件时序 + 技术重复 + 时序回退 flag
    add_draw_visit(male)
    add_draw_visit(female)
    # 男胎行级达标事实
    for r in male:
        r["is_qualified"] = 1 if (r["y_conc"] is not None
                                  and r["y_conc"] >= Y_THRESHOLD) else 0
    add_qc_flags(male)
    add_qc_flags(female)
    add_female_labels(female)

    # 事件级
    ev_male = build_event_records(male, True)
    ev_female = build_event_records(female, False)
    add_event_survival(ev_male)
    attach_survival_to_rows(male, ev_male)

    # 列集合
    base = [MAP[c] for c in LETTERS]
    surv_cols = ["reached", "first_pass_week", "first_pass_visit",
                 "t_left", "t_right", "censored"]
    male_clean_cols = (base + ["week", "week_raw_s", "visit_idx", "n_visits",
                               "rep_in_draw", "n_reps_in_draw", "is_tech_repeat",
                               "is_qualified"] + surv_cols
                       + ["flag_chrono", "flag_gc", "flag_map_ratio",
                          "flag_dup_ratio", "flag_filt_ratio", "flag_any"])
    female_clean_cols = (base + ["week", "week_raw_s", "visit_idx", "n_visits",
                                 "rep_in_draw", "n_reps_in_draw", "is_tech_repeat",
                                 "label", "label_T13", "label_T18", "label_T21",
                                 "flag_chrono", "flag_gc", "flag_map_ratio",
                                 "flag_dup_ratio", "flag_filt_ratio", "flag_any"])
    male_min_cols = ["mother_id", "age", "height", "weight", "bmi", "week",
                     "y_conc", "draw_idx", "visit_idx", "n_visits"] + REP_COLS
    male_event_cols = (["mother_id", "draw_idx", "visit_idx", "n_reps", "is_tech_repeat",
                        "flag_chrono", "age", "height", "weight", "bmi", "draw_date",
                        "week_min", "week_max", "week_mean",
                        "y_conc_n", "y_conc_min", "y_conc_mean", "y_conc_max", "y_conc_sd",
                        "qual_any", "qual_mean"]
                       + surv_cols + [c + "_any" for c in surv_cols]
                       + ["flag_any"])
    female_event_cols = ["mother_id", "draw_idx", "visit_idx", "n_reps", "is_tech_repeat",
                         "flag_chrono", "age", "height", "weight", "bmi", "draw_date",
                         "week_min", "week_max", "week_mean",
                         "label", "label_T13", "label_T18", "label_T21",
                         "aneuploidy_raw", "flag_any"]

    write_csv(OUT_DIR / "male_min.csv", male, male_min_cols)
    write_csv(OUT_DIR / "male_clean.csv", male, male_clean_cols)
    write_csv(OUT_DIR / "female_clean.csv", female, female_clean_cols)
    write_csv(OUT_DIR / "male_clean_event.csv", ev_male, male_event_cols)
    write_csv(OUT_DIR / "female_clean_event.csv", ev_female, female_event_cols)
    REPORT_PATH.write_text(make_report(male, female, ev_male, ev_female), encoding="utf-8")

    print("OK -> data/processed/{male_min,male_clean,female_clean,male_clean_event,"
          "female_clean_event}.csv + docs/data_report.md  [v3]")
    print(f"male rows={len(male)} events={len(ev_male)} | "
          f"female rows={len(female)} events={len(ev_female)}")


if __name__ == "__main__":
    main()
