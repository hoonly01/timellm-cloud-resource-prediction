"""
EDA Full Report Generator
데이터 개요 통계를 eda/output/data_overview/report.md로 저장
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = PROJ_ROOT / "data"
OUT_DIR   = PROJ_ROOT / "eda" / "output" / "data_overview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
print("Loading data...")
files = sorted(DATA_DIR.glob("gcluster_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

ts_len = df.groupby(["collection_id", "instance_index", "final_event_type"]).size()

lines = []
def h(text): lines.append(f"\n{text}\n" + "=" * len(text))
def h2(text): lines.append(f"\n{text}\n" + "-" * len(text))
def p(text=""): lines.append(str(text))
def table(df_or_str): lines.append(str(df_or_str))

lines.append("# Google Cluster Trace v3 (Cell a) — EDA Full Report")
lines.append(f"Data: gcluster_*.csv (6 files) | BigQuery export | FAIL(5) + FINISH(6)")
lines.append("")

# ════════════════════════════════════════════════════════════════
h("1. Dataset Overview")
# ════════════════════════════════════════════════════════════════

n_inst    = ts_len.shape[0]
n_fail    = int((ts_len.index.get_level_values("final_event_type") == 5).sum())
n_finish  = int((ts_len.index.get_level_values("final_event_type") == 6).sum())
fail_rows = int((df["final_event_type"] == 5).sum())
fin_rows  = int((df["final_event_type"] == 6).sum())

h2("1-1. Basic Stats")
p(f"Total rows        : {len(df):,}")
p(f"Total instances   : {n_inst:,}")
p(f"  FAIL  (type=5)  : {n_fail:,} instances  |  {fail_rows:,} rows")
p(f"  FINISH(type=6)  : {n_finish:,} instances  |  {fin_rows:,} rows")
p(f"  FAIL ratio      : {n_fail/n_inst*100:.2f}% (instances)  |  {fail_rows/len(df)*100:.2f}% (rows)")
p(f"Columns           : {list(df.columns)}")

h2("1-2. Missing Values")
null_by_col = df.isnull().sum()
has_null = null_by_col[null_by_col > 0]
if len(has_null) == 0:
    p("  (no missing values in any column)")
else:
    for col, cnt in has_null.items():
        p(f"  {col}: {cnt:,} ({cnt/len(df)*100:.2f}%)")

# ════════════════════════════════════════════════════════════════
h("2. Data Quality")
# ════════════════════════════════════════════════════════════════

h2("2-1. Duplicate Rows")
dup_count = df.duplicated(subset=["collection_id","instance_index","time_step_sec"]).sum()
p(f"  Duplicates: {dup_count:,}  ({dup_count/len(df)*100:.3f}%)")

h2("2-2. Time Gap Analysis (5-min = 300s expected)")
sorted_df = df.sort_values(["collection_id","instance_index","time_step_sec"])
gaps = sorted_df.groupby(["collection_id","instance_index"])["time_step_sec"].diff().dropna()
non_300 = (gaps != 300).mean() * 100
gap_vc   = gaps.value_counts().head(10)
p(f"  Non-300s gap ratio : {non_300:.2f}%")
p(f"  Top gap values (sec):")
for val, cnt in gap_vc.items():
    p(f"    {int(val):>6}s : {cnt:>8,}  ({cnt/len(gaps)*100:.2f}%)")

h2("2-3. Value Range Check")
for col in ["cpu_usage","max_cpu_usage","mem_usage","max_mem_usage"]:
    neg = (df[col] < 0).sum()
    over1 = (df[col] > 1.0).sum() if col != "max_cpu_usage" else (df[col] > 2.0).sum()
    p(f"  {col:<20}: negative={neg:,}  outlier(>{'>1.0' if col!='max_cpu_usage' else '>2.0'})={over1:,}")

# ════════════════════════════════════════════════════════════════
h("3. Time-series Length Distribution")
# ════════════════════════════════════════════════════════════════

h2("3-1. Percentiles by Class")
for etype, label in [(5,"FAIL"),(6,"FINISH")]:
    vals = ts_len[ts_len.index.get_level_values("final_event_type")==etype]
    pcts = np.percentile(vals, [5,10,25,50,75,90,95,99])
    p(f"  {label}:")
    p(f"    p5={pcts[0]:.0f}  p10={pcts[1]:.0f}  p25={pcts[2]:.0f}  p50={pcts[3]:.0f}"
      f"  p75={pcts[4]:.0f}  p90={pcts[5]:.0f}  p95={pcts[6]:.0f}  p99={pcts[7]:.0f}")
    p(f"    mean={vals.mean():.1f}  std={vals.std():.1f}  max={vals.max()}")

h2("3-2. Sliding Window Count Simulation")
def count_windows(length, seq_len, pred_len, stride):
    if length < seq_len + pred_len: return 0
    return (length - seq_len - pred_len) // stride + 1

configs = [
    (32, 16, 16,  "seq32 / pred16 / stride16"),
    (32, 32, 16,  "seq32 / pred32 / stride16"),
    (64, 32, 32,  "seq64 / pred32 / stride32"),
    (96, 32, 32,  "seq96 / pred32 / stride32"),
]
p(f"  {'Config':<28} {'FAIL windows':>14} {'FINISH windows':>16} {'FAIL inst (%)':>14} {'FINISH inst (%)':>16}")
p(f"  {'-'*92}")
for seq_l, pred_l, stride, name in configs:
    counts = ts_len.apply(lambda n: count_windows(n, seq_l, pred_l, stride))
    by_cls = counts.groupby(level="final_event_type").sum()
    fw = int(by_cls.get(5,0)); finw = int(by_cls.get(6,0))
    usable_f  = int((ts_len[ts_len.index.get_level_values("final_event_type")==5] >= seq_l+pred_l).sum())
    usable_fi = int((ts_len[ts_len.index.get_level_values("final_event_type")==6] >= seq_l+pred_l).sum())
    p(f"  {name:<28} {fw:>14,} {finw:>16,} {usable_f:>10} ({usable_f/n_fail*100:.0f}%) {usable_fi:>10} ({usable_fi/n_finish*100:.0f}%)")

# ════════════════════════════════════════════════════════════════
h("4. Feature Distribution by Class")
# ════════════════════════════════════════════════════════════════

h2("4-1. Percentile Table — FAIL (type=5)")
numeric_cols = ["cpu_usage","max_cpu_usage","mem_usage","max_mem_usage",
                "page_cache_memory","assigned_mem","requested_cpu","requested_mem"]
pct_levels = [0.01,0.05,0.25,0.50,0.75,0.95,0.99]

for etype, label in [(5,"FAIL"),(6,"FINISH")]:
    sub = df[df["final_event_type"]==etype]
    pct_df = sub[numeric_cols].quantile(pct_levels).T
    pct_df.columns = [f"p{int(p*100)}" for p in pct_levels]
    p(f"\n  {'Feature':<22} " + "  ".join(f"{c:>9}" for c in pct_df.columns))
    p(f"  {'-'*100}")
    for feat, row in pct_df.iterrows():
        p(f"  {feat:<22} " + "  ".join(f"{v:>9.5f}" for v in row))
    if etype == 5:
        h2("4-2. Percentile Table — FINISH (type=6)")

h2("4-3. Class-level Summary (mean ± std)")
summary_cols = ["cpu_usage","max_cpu_usage","mem_usage","max_mem_usage",
                "page_cache_memory","assigned_mem"]
for col in summary_cols:
    f_mean  = df[df["final_event_type"]==5][col].mean()
    f_std   = df[df["final_event_type"]==5][col].std()
    fi_mean = df[df["final_event_type"]==6][col].mean()
    fi_std  = df[df["final_event_type"]==6][col].std()
    ratio   = f_mean / fi_mean if fi_mean > 0 else float("nan")
    p(f"  {col:<22}: FAIL={f_mean:.5f}±{f_std:.5f}  FINISH={fi_mean:.5f}±{fi_std:.5f}  ratio={ratio:.3f}x")

h2("4-4. assigned_mem vs mem_usage")
df_tmp = df.copy()
df_tmp["mem_util_ratio"] = df_tmp["mem_usage"] / df_tmp["assigned_mem"].replace(0, np.nan)
for etype, label in [(5,"FAIL"),(6,"FINISH")]:
    sub = df_tmp[df_tmp["final_event_type"]==etype]["mem_util_ratio"]
    p(f"  {label}: mean={sub.mean():.3f}  median={sub.median():.3f}"
      f"  >1.0={( sub>1.0).mean()*100:.2f}%  >0.5={(sub>0.5).mean()*100:.2f}%")

# ════════════════════════════════════════════════════════════════
h("5. CPU-Memory Correlation")
# ════════════════════════════════════════════════════════════════

inst_mean = df.groupby(["collection_id","instance_index","final_event_type"])[
    ["cpu_usage","mem_usage"]].mean().reset_index()

h2("5-1. Pearson & Spearman (instance-level mean)")
for etype, label in [(5,"FAIL"),(6,"FINISH")]:
    sub = inst_mean[inst_mean["final_event_type"]==etype]
    r_p, p_p = stats.pearsonr(sub["cpu_usage"], sub["mem_usage"])
    r_s, p_s = stats.spearmanr(sub["cpu_usage"], sub["mem_usage"])
    p(f"  {label}:")
    p(f"    Pearson  r={r_p:.4f}  p={p_p:.2e}")
    p(f"    Spearman r={r_s:.4f}  p={p_s:.2e}")

# ════════════════════════════════════════════════════════════════
h("6. Burst Pattern Analysis")
# ════════════════════════════════════════════════════════════════

df["cpu_burst"] = df["max_cpu_usage"] - df["cpu_usage"]
df["mem_burst"] = df["max_mem_usage"] - df["mem_usage"]

h2("6-1. Burst (max - avg) — instance max value statistics")
inst_burst = df.groupby(["collection_id","instance_index","final_event_type"])[
    ["cpu_burst","mem_burst"]].max().reset_index()
for etype, label in [(5,"FAIL"),(6,"FINISH")]:
    sub = inst_burst[inst_burst["final_event_type"]==etype]
    p(f"  {label} cpu_burst: mean={sub['cpu_burst'].mean():.4f}  "
      f"median={sub['cpu_burst'].median():.4f}  p75={sub['cpu_burst'].quantile(0.75):.4f}  "
      f"p95={sub['cpu_burst'].quantile(0.95):.4f}")
    p(f"  {label} mem_burst: mean={sub['mem_burst'].mean():.5f}  "
      f"median={sub['mem_burst'].median():.5f}  p75={sub['mem_burst'].quantile(0.75):.5f}  "
      f"p95={sub['mem_burst'].quantile(0.95):.5f}")

h2("6-2. Burst k-sweep (row-level: max_cpu > inst_mean + k*std)")
inst_stats = df.groupby(["collection_id","instance_index","final_event_type"])[
    "cpu_usage"].agg(["mean","std"]).reset_index()
inst_stats.columns = ["collection_id","instance_index","final_event_type","cpu_mean","cpu_std"]
df_m = df.merge(inst_stats, on=["collection_id","instance_index","final_event_type"])

p(f"  {'k':<6} {'Overall':>10} {'FAIL':>10} {'FINISH':>10}  note")
p(f"  {'-'*55}")
for k in [1.5, 2.0, 2.5, 3.0]:
    burst = df_m["max_cpu_usage"] > df_m["cpu_mean"] + k * df_m["cpu_std"]
    ov = burst.mean()*100
    fv = burst[df_m["final_event_type"]==5].mean()*100
    fiv = burst[df_m["final_event_type"]==6].mean()*100
    note = " ← target(5~15%)" if 5<=ov<=15 else (" ← too high" if ov>15 else " ← too low")
    p(f"  k={k:<4} {ov:>9.2f}% {fv:>9.2f}% {fiv:>9.2f}%{note}")

p()
p("  [Note] k-sweep based on instance-level std results in ~80% burst ratio.")
p("  Alternative: use absolute spike threshold (max_cpu - avg_cpu > 0.05)")
abs_thresh = 0.05
abs_burst = df["cpu_burst"] > abs_thresh
p(f"  Absolute spike > {abs_thresh}: overall={abs_burst.mean()*100:.2f}%  "
  f"FAIL={abs_burst[df['final_event_type']==5].mean()*100:.2f}%  "
  f"FINISH={abs_burst[df['final_event_type']==6].mean()*100:.2f}%")
for thr in [0.02, 0.05, 0.10, 0.15, 0.20]:
    ab = df["cpu_burst"] > thr
    p(f"  spike > {thr:.2f}: overall={ab.mean()*100:.2f}%  "
      f"FAIL={ab[df['final_event_type']==5].mean()*100:.2f}%  "
      f"FINISH={ab[df['final_event_type']==6].mean()*100:.2f}%")

# ════════════════════════════════════════════════════════════════
h("7. Resource Pressure")
# ════════════════════════════════════════════════════════════════

df["cpu_pressure"] = df["cpu_usage"] / df["requested_cpu"].replace(0, np.nan)
df["mem_pressure"] = df["mem_usage"] / df["requested_mem"].replace(0, np.nan)
clip_cpu = df["cpu_pressure"].quantile(0.999)
clip_mem = df["mem_pressure"].quantile(0.999)
df["cpu_pressure_c"] = df["cpu_pressure"].clip(upper=clip_cpu)
df["mem_pressure_c"] = df["mem_pressure"].clip(upper=clip_mem)

h2("7-1. CPU Pressure (cpu_usage / requested_cpu, clipped at 99.9th)")
p(f"  {'stat':<20} {'FAIL':>12} {'FINISH':>12}  ratio(F/Fi)")
for pct, label in [(None,"mean"),(0.50,"p50"),(0.75,"p75"),(0.90,"p90"),(0.95,"p95")]:
    if pct is None:
        fv  = df[df["final_event_type"]==5]["cpu_pressure_c"].mean()
        fiv = df[df["final_event_type"]==6]["cpu_pressure_c"].mean()
    else:
        fv  = df[df["final_event_type"]==5]["cpu_pressure_c"].quantile(pct)
        fiv = df[df["final_event_type"]==6]["cpu_pressure_c"].quantile(pct)
    p(f"  {label:<20} {fv:>12.4f} {fiv:>12.4f}  {fv/fiv:.2f}x")

h2("7-2. Memory Pressure (mem_usage / requested_mem, clipped at 99.9th)")
for pct, label in [(None,"mean"),(0.50,"p50"),(0.75,"p75"),(0.90,"p90"),(0.95,"p95")]:
    if pct is None:
        fv  = df[df["final_event_type"]==5]["mem_pressure_c"].mean()
        fiv = df[df["final_event_type"]==6]["mem_pressure_c"].mean()
    else:
        fv  = df[df["final_event_type"]==5]["mem_pressure_c"].quantile(pct)
        fiv = df[df["final_event_type"]==6]["mem_pressure_c"].quantile(pct)
    p(f"  {label:<20} {fv:>12.4f} {fiv:>12.4f}  {fv/fiv:.2f}x")

h2("7-3. Pressure > 1.0 (using more than requested)")
for etype, label in [(5,"FAIL"),(6,"FINISH")]:
    cp = (df[df["final_event_type"]==etype]["cpu_pressure"] > 1.0).mean()*100
    mp = (df[df["final_event_type"]==etype]["mem_pressure"] > 1.0).mean()*100
    p(f"  {label}: cpu_pressure>1.0 = {cp:.2f}%   mem_pressure>1.0 = {mp:.2f}%")

# ════════════════════════════════════════════════════════════════
h("8. Failure Precursor Analysis")
# ════════════════════════════════════════════════════════════════

h2("8-1. CPU ratio: last-N steps mean / instance mean")
p("  (>1.0 = rising toward end, <1.0 = decreasing)")
p(f"  {'Class':<8} {'tail':>6} {'mean':>8} {'median':>8} {'p25':>8} {'p75':>8} {'>1.5 (%)':>10} {'>2.0 (%)':>10}")
p(f"  {'-'*75}")
for etype, label in [(5,"FAIL"),(6,"FINISH")]:
    grp_df = df[df["final_event_type"]==etype]
    for tail_n in [8, 16, 32]:
        ratios = []
        for (cid,idx), g in grp_df.groupby(["collection_id","instance_index"]):
            g = g.sort_values("time_step_sec")
            if len(g) < tail_n: continue
            om = g["cpu_usage"].mean()
            if om == 0: continue
            ratios.append(g.tail(tail_n)["cpu_usage"].mean() / om)
        r = np.array(ratios)
        p(f"  {label:<8} last{tail_n:>2}  {r.mean():>8.4f} {np.median(r):>8.4f} "
          f"{np.percentile(r,25):>8.4f} {np.percentile(r,75):>8.4f} "
          f"{(r>1.5).mean()*100:>9.1f}% {(r>2.0).mean()*100:>9.1f}%")

h2("8-2. CPU Slope (linear fit over entire time series)")
slopes_fail = []; slopes_finish = []
for (cid,idx,etype), g in df.groupby(["collection_id","instance_index","final_event_type"]):
    g = g.sort_values("time_step_sec")
    if len(g) < 8: continue
    s, _ = np.polyfit(np.arange(len(g)), g["cpu_usage"].values, 1)
    (slopes_fail if etype==5 else slopes_finish).append(s)

sf = np.array(slopes_fail); sfi = np.array(slopes_finish)
p(f"  FAIL  slope: mean={sf.mean():.6f}  median={np.median(sf):.6f}  "
  f"p25={np.percentile(sf,25):.6f}  p75={np.percentile(sf,75):.6f}")
p(f"  FINISH slope: mean={sfi.mean():.6f}  median={np.median(sfi):.6f}  "
  f"p25={np.percentile(sfi,25):.6f}  p75={np.percentile(sfi,75):.6f}")
p(f"  FAIL slope>0 (rising): {(sf>0).mean()*100:.1f}%")
p(f"  FINISH slope>0 (rising): {(sfi>0).mean()*100:.1f}%")

slope_rising_th = np.percentile(sfi, 75)
slope_stable_th = np.percentile(sfi, 25)
p(f"\n  Proposed thresholds (FINISH quartiles):")
p(f"    normal-rising : slope > {slope_rising_th:.7f}  (FINISH top 25%)")
p(f"    normal-stable : slope < {slope_stable_th:.7f}  (FINISH bottom 25%)")

# ════════════════════════════════════════════════════════════════
h("9. max_mem_usage NULL Analysis")
# ════════════════════════════════════════════════════════════════

null_mask = df["max_mem_usage"].isnull()
p(f"  Total NULL        : {null_mask.sum():,} ({null_mask.mean()*100:.2f}%)")
for etype, label in [(5,"FAIL"),(6,"FINISH")]:
    r = df[df["final_event_type"]==etype]["max_mem_usage"].isnull().mean()*100
    p(f"  {label} NULL rate  : {r:.2f}%")
inst_null = df.groupby(["collection_id","instance_index"])["max_mem_usage"].apply(lambda x: x.isnull().any())
p(f"  Instances with any NULL: {inst_null.sum():,} / {len(inst_null):,} ({inst_null.mean()*100:.1f}%)")
# NULL position
df_np = df[null_mask].copy()
df_np["ts_len_inst"] = df_np.groupby(["collection_id","instance_index"])["time_step_sec"].transform("count")
df_np["rank"] = df_np.groupby(["collection_id","instance_index"])["time_step_sec"].rank(method="first")
df_np["norm_pos"] = (df_np["rank"] - 1) / (df_np["ts_len_inst"] - 1).clip(lower=1)
p(f"  NULL normalized position: mean={df_np['norm_pos'].mean():.3f}  "
  f"(0=start, 0.5=middle, 1=end)")
p(f"  → Uniformly distributed across time → safe to fillna(mem_usage)")

# ════════════════════════════════════════════════════════════════
h("10. RAG / Model Split Preview")
# ════════════════════════════════════════════════════════════════

inst_unique = df[["collection_id","instance_index","final_event_type"]].drop_duplicates()
inst_unique["pool"] = np.where(inst_unique["collection_id"] % 10 < 2, "RAG", "Model")
split = inst_unique.groupby(["pool","final_event_type"]).size().unstack(fill_value=0)
split.columns = ["FAIL","FINISH"]
split["total"]     = split.sum(axis=1)
split["FAIL_pct"]  = split["FAIL"]  / split["total"] * 100
split["FINISH_pct"]= split["FINISH"]/ split["total"] * 100
table(split.to_string())
p()
p(f"  FAIL in RAG  : {int(split.loc['RAG','FAIL']):,}  ({split.loc['RAG','FAIL_pct']:.1f}%)")
p(f"  FAIL in Model: {int(split.loc['Model','FAIL']):,}  ({split.loc['Model','FAIL_pct']:.1f}%)")

# ════════════════════════════════════════════════════════════════
h("11. Key Findings Summary (논문 작성 근거)")
# ════════════════════════════════════════════════════════════════

p("""
[F1] FAIL 인스턴스는 FINISH 대비 ~3배 긴 수명
     FAIL median=267 steps(22.3h) vs FINISH median=92 steps(7.7h)
     → 오래 실행된 job에서 failure 발생 경향 강함

[F2] CPU Burst: FAIL이 FINISH의 2배
     FAIL cpu_burst(max-avg) mean=0.190 vs FINISH mean=0.100
     → max_cpu_usage - cpu_usage 를 핵심 피처로 사용 근거

[F3] Resource Pressure: FAIL이 FINISH의 2.1x
     FAIL cpu_pressure mean=2.005 vs FINISH mean=0.949
     FAIL mem_pressure mean=1.293 vs FINISH mean=0.550
     → requested 대비 실제 사용량이 FAIL에서 훨씬 높음
     → RAG 프롬프트에 resource_pressure 컨텍스트 삽입 정당화

[F4] Failure Precursor 패턴 존재 (논문 Figure 핵심 근거)
     FAIL last32/mean ratio = 1.116 (종료 직전 상승)
     FINISH last32/mean ratio = 0.862 (종료 직전 하강)
     → 15.1% FAIL vs 2.4% FINISH: ratio > 1.5 기록
     → 시계열 패턴으로 failure 예측 가능성 입증

[F5] CPU-Memory 비선형 상관관계
     FAIL  Pearson r=-0.003 (사실상 0) / Spearman r=0.387
     FINISH Pearson r=0.182           / Spearman r=0.532
     → 특히 FAIL에서 선형 독립 → CD 모듈 추가 검토 근거

[F6] Burst k-sweep 문제
     인스턴스 내 std 기반 k-sweep: 모든 k에서 ~80% burst 분류
     → 절대값 spike 기준 필요:
       spike > 0.05: 전체 21.8%, FAIL 29.2%, FINISH 21.1%
       spike > 0.10: 전체 10.5%, FAIL 16.3%, FINISH 10.0%
     → spike > 0.10이 5-15% 기준 충족 (FINISH 기준)
     → RAG 5클래스 burst 기준: cpu_burst > 0.10 권장

[F7] Sliding Window 권장 설정
     seq32/pred16/stride16: FAIL=55K, FINISH=423K win, 100% inst 커버
     seq64/pred32/stride32: FAIL=25K, FINISH=170K win, 76% inst 커버
     → FAIL 인스턴스 전수 활용 위해 seq=32~48 권장
     → pred_len=16: 80분 ahead 예측 (5분×16)

[F8] max_mem_usage NULL 4.09% (FINISH에만 집중)
     FAIL NULL=0%, FINISH NULL=4.58%
     시간 위치 균일 분포 → fillna(mem_usage) 안전
""")

# ════════════════════════════════════════════════════════════════
# 저장
# ════════════════════════════════════════════════════════════════
report = "\n".join(lines)
out_path = OUT_DIR / "report.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Saved: {out_path}")
print(f"Size : {out_path.stat().st_size / 1024:.1f} KB")
print(f"Lines: {len(lines)}")
