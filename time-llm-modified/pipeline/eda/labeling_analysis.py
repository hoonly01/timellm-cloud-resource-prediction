"""
EDA for Google Cluster Trace v3 (Cell a): 레이블링·윈도우 설계 분석
- Section 1: 데이터 품질 검증 (중복, time gap, percentile table)
- Section 2: 피처 분포 심화 (log-scale, assigned_mem vs mem, page_cache)
- Section 3: Failure 직전 패턴 (precursor, mem_pressure)
- Section 4: Burst k-sweep + normal-stable/rising slope
- Section 5: Sliding window 설계 근거 (ACF, window simulation)
- Section 6: RAG DB 구성 가능성 (5클래스 window 수)
- Section 7: max_mem_usage NULL 처리 방안
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = PROJ_ROOT / "data"
OUT_DIR   = PROJ_ROOT / "eda" / "output" / "labeling_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
print("Loading data...")
files = sorted(DATA_DIR.glob("gcluster_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"  Loaded {len(df):,} rows, {df['collection_id'].nunique()} unique collection_ids\n")

# ── 기초 집계 (공용) ─────────────────────────────────────────────────────────
ts_len = df.groupby(["collection_id", "instance_index", "final_event_type"]).size()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: 데이터 품질 검증
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Data Quality Validation")
print("=" * 60)

# 1-A. 중복 행
dup_count = df.duplicated(subset=["collection_id", "instance_index", "time_step_sec"]).sum()
print(f"\n[Duplicates]")
print(f"  Duplicate rows: {dup_count:,} ({dup_count/len(df)*100:.3f}%)")

# 1-B. time_step_sec gap 분석
print(f"\n[Time Gap Analysis]")
sorted_df = df.sort_values(["collection_id", "instance_index", "time_step_sec"])
gaps = sorted_df.groupby(["collection_id", "instance_index"])["time_step_sec"].diff().dropna()
non_300 = (gaps != 300).mean() * 100
gap_vc = gaps.value_counts().head(10)
print(f"  Non-300sec gap ratio: {non_300:.2f}%")
print(f"  Top-10 gap values (sec):")
for val, cnt in gap_vc.items():
    print(f"    gap={val:.0f}s : {cnt:,} ({cnt/len(gaps)*100:.2f}%)")

# 1-C. Percentile table (클래스별)
print(f"\n[Percentile Table by Class]")
numeric_cols = ["cpu_usage", "max_cpu_usage", "mem_usage", "max_mem_usage",
                "page_cache_memory", "assigned_mem", "requested_cpu", "requested_mem"]
pct_levels = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

for etype, label in [(5, "FAIL"), (6, "FINISH")]:
    sub = df[df["final_event_type"] == etype]
    pct_df = sub[numeric_cols].quantile(pct_levels).T
    pct_df.columns = [f"p{int(p*100)}" for p in pct_levels]
    print(f"\n  {label}:")
    print(pct_df.to_string(float_format=lambda x: f"{x:.5f}"))

# 1-D. 시각화: gap 분포 + 음수/이상치 체크
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# gap histogram (300 제외, top 이상치만)
gap_filtered = gaps[gaps != 300]
if len(gap_filtered) > 0:
    axes[0].hist(gap_filtered.clip(0, 3000), bins=50, color="#e67e22", edgecolor="k", alpha=0.8)
    axes[0].set_title(f"Non-300s Gap Distribution\n(total: {len(gap_filtered):,}, {non_300:.2f}%)")
    axes[0].set_xlabel("gap (sec)")
    axes[0].set_ylabel("count")
else:
    axes[0].text(0.5, 0.5, "All gaps = 300s\n(Perfect continuity)", ha="center", va="center",
                 transform=axes[0].transAxes, fontsize=13)
    axes[0].set_title("Time Gap Distribution")

# cpu_usage 값 범위 확인 (log scale)
axes[1].hist(df["cpu_usage"].clip(upper=df["cpu_usage"].quantile(0.999)),
             bins=100, color="#3498db", alpha=0.75, edgecolor="none", log=True)
axes[1].set_title("cpu_usage Distribution (log-y)")
axes[1].set_xlabel("cpu_usage (normalized)")
axes[1].set_ylabel("count (log scale)")

plt.suptitle("Section 1: Data Quality", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "08_data_quality.png", dpi=150)
plt.close()
print("\n  → 08_data_quality.png saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: 피처 분포 심화
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Feature Distribution Deep Dive")
print("=" * 60)

# 2-A. log1p KDE 분포 비교 (cpu, mem, page_cache)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
log_features = ["cpu_usage", "mem_usage", "page_cache_memory",
                "max_cpu_usage", "max_mem_usage", "assigned_mem"]

for ax, feat in zip(axes.flat, log_features):
    for etype, label, color in [(5, "FAIL", "#e74c3c"), (6, "FINISH", "#2ecc71")]:
        vals = np.log1p(df[df["final_event_type"] == etype][feat].dropna())
        vals = vals[np.isfinite(vals)]
        if len(vals) > 100:
            vals.plot.kde(ax=ax, label=label, color=color, bw_method=0.3)
    ax.set_title(f"log1p({feat})")
    ax.set_xlabel("log1p(value)")
    ax.legend(fontsize=8)

plt.suptitle("Section 2: Log-Scale Feature KDE by Class", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "09_feature_log_dist.png", dpi=150)
plt.close()
print("  → 09_feature_log_dist.png saved")

# 2-B. assigned_mem vs mem_usage (인스턴스 평균)
inst_means = df.groupby(["collection_id", "instance_index", "final_event_type"])[
    ["mem_usage", "assigned_mem", "page_cache_memory"]].mean().reset_index()

df_temp = df.copy()
df_temp["mem_util_ratio"] = df_temp["mem_usage"] / df_temp["assigned_mem"].replace(0, np.nan)
ratio_over1_fail   = (df_temp[df_temp["final_event_type"]==5]["mem_util_ratio"] > 1.0).mean() * 100
ratio_over1_finish = (df_temp[df_temp["final_event_type"]==6]["mem_util_ratio"] > 1.0).mean() * 100
print(f"\n[assigned_mem vs mem_usage]")
print(f"  mem_util_ratio > 1.0 (FAIL):   {ratio_over1_fail:.2f}%")
print(f"  mem_util_ratio > 1.0 (FINISH): {ratio_over1_finish:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (etype, label, color) in zip(axes, [(5, "FAIL", "#e74c3c"), (6, "FINISH", "#2ecc71")]):
    sub = inst_means[inst_means["final_event_type"] == etype]
    hb = ax.hexbin(sub["assigned_mem"], sub["mem_usage"],
                   gridsize=40, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax, label="count")
    # y=x 기준선
    lim = max(sub["assigned_mem"].quantile(0.99), sub["mem_usage"].quantile(0.99))
    ax.plot([0, lim], [0, lim], "b--", alpha=0.6, label="y=x (full util)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(f"assigned_mem vs mem_usage\n({label})")
    ax.set_xlabel("mean assigned_mem")
    ax.set_ylabel("mean mem_usage")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "10_assigned_vs_mem.png", dpi=150)
plt.close()
print("  → 10_assigned_vs_mem.png saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Failure 직전 패턴 (Precursor Analysis)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: Failure Precursor Analysis")
print("=" * 60)

# 3-A. FAIL 인스턴스 마지막 N steps vs 전체 평균 비율
print("\n[CPU Precursor: last N steps vs instance mean]")
precursor_results = []

for (cid, idx, etype), grp in df.groupby(["collection_id", "instance_index", "final_event_type"]):
    grp = grp.sort_values("time_step_sec")
    n = len(grp)
    if n < 16:
        continue
    overall_mean = grp["cpu_usage"].mean()
    if overall_mean == 0:
        continue
    for tail_n in [8, 16, 32]:
        if n >= tail_n:
            tail_mean = grp.tail(tail_n)["cpu_usage"].mean()
            precursor_results.append({
                "final_event_type": etype,
                "tail_n": tail_n,
                "ratio": tail_mean / overall_mean
            })

prec_df = pd.DataFrame(precursor_results)
print(prec_df.groupby(["final_event_type", "tail_n"])["ratio"].describe().to_string())

# 3-B. mem_pressure 분포
df["mem_pressure"] = df["mem_usage"] / df["requested_mem"].replace(0, np.nan)
clip_val = df["mem_pressure"].quantile(0.999)
df["mem_pressure_clipped"] = df["mem_pressure"].clip(upper=clip_val)

mem_press_stats = df.groupby("final_event_type")["mem_pressure_clipped"].describe()
print(f"\n[mem_pressure = mem_usage/requested_mem (clipped at 99.9th)]")
print(mem_press_stats.to_string())

# 시각화
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# (0,0)~(0,1): tail ratio violin
ax_v = fig.add_subplot(gs[0, :2])
tail_data_fail   = [prec_df[(prec_df["final_event_type"]==5) & (prec_df["tail_n"]==n)]["ratio"].clip(0,4).values
                    for n in [8, 16, 32]]
tail_data_finish = [prec_df[(prec_df["final_event_type"]==6) & (prec_df["tail_n"]==n)]["ratio"].clip(0,4).values
                    for n in [8, 16, 32]]

positions_f = [1, 2, 3]
positions_fi = [5, 6, 7]
vp1 = ax_v.violinplot(tail_data_fail,   positions=positions_f,  showmedians=True, widths=0.7)
vp2 = ax_v.violinplot(tail_data_finish, positions=positions_fi, showmedians=True, widths=0.7)
for body in vp1["bodies"]: body.set_facecolor("#e74c3c"); body.set_alpha(0.7)
for body in vp2["bodies"]: body.set_facecolor("#2ecc71"); body.set_alpha(0.7)
ax_v.axhline(1.0, color="navy", linestyle="--", alpha=0.5, label="ratio=1 (no change)")
ax_v.set_xticks([1,2,3,5,6,7])
ax_v.set_xticklabels(["FAIL\nlast8","FAIL\nlast16","FAIL\nlast32",
                       "FINISH\nlast8","FINISH\nlast16","FINISH\nlast32"])
ax_v.set_title("CPU: last-N mean / instance mean (ratio>1 = rising before end)")
ax_v.set_ylabel("ratio")
ax_v.legend()

# (0,2): FAIL tail-32 ratio histogram
ax_h = fig.add_subplot(gs[0, 2])
fail_ratios = prec_df[(prec_df["final_event_type"]==5) & (prec_df["tail_n"]==32)]["ratio"].clip(0,4)
fin_ratios  = prec_df[(prec_df["final_event_type"]==6) & (prec_df["tail_n"]==32)]["ratio"].clip(0,4)
ax_h.hist(fail_ratios,  bins=50, alpha=0.6, color="#e74c3c", label="FAIL",   density=True)
ax_h.hist(fin_ratios,   bins=50, alpha=0.6, color="#2ecc71", label="FINISH", density=True)
ax_h.axvline(1.0, color="navy", linestyle="--")
ax_h.set_title("CPU ratio (last32 / mean)\ndensity")
ax_h.set_xlabel("ratio")
ax_h.legend()

# (1,0)~(1,1): mem_pressure 분포
ax_mp = fig.add_subplot(gs[1, :2])
for etype, label, color in [(5,"FAIL","#e74c3c"), (6,"FINISH","#2ecc71")]:
    vals = df[df["final_event_type"]==etype]["mem_pressure_clipped"]
    ax_mp.hist(vals, bins=80, alpha=0.5, color=color, label=label, density=True)
ax_mp.set_title(f"Memory Pressure (mem_usage/requested_mem, clipped={clip_val:.2f})")
ax_mp.set_xlabel("pressure ratio")
ax_mp.set_ylabel("density")
ax_mp.legend()

# (1,2): FAIL ratio > 1.5 비율
ax_bar = fig.add_subplot(gs[1, 2])
over15 = prec_df[prec_df["ratio"] > 1.5].groupby(["final_event_type","tail_n"]).size()
total  = prec_df.groupby(["final_event_type","tail_n"]).size()
ratio_over15 = (over15 / total * 100).reset_index(name="pct")
ratio_over15["label"] = ratio_over15["final_event_type"].map({5:"FAIL",6:"FINISH"})
for tail_n, color in [(8,"#f39c12"),(16,"#e67e22"),(32,"#d35400")]:
    sub = ratio_over15[ratio_over15["tail_n"]==tail_n]
    ax_bar.bar(sub["label"], sub["pct"], label=f"last{tail_n}", color=color, alpha=0.8, width=0.25,
               align="center")
ax_bar.set_title("% instances with ratio > 1.5")
ax_bar.set_ylabel("%")
ax_bar.legend(fontsize=8)

plt.suptitle("Section 3: Failure Precursor Patterns", fontsize=13)
plt.savefig(OUT_DIR / "11_failure_precursor.png", dpi=150)
plt.close()
print("  → 11_failure_precursor.png saved")

# 3-C. FAIL ratio > 1.5 요약
for tail_n in [8, 16, 32]:
    for etype, label in [(5,"FAIL"),(6,"FINISH")]:
        sub = prec_df[(prec_df["final_event_type"]==etype) & (prec_df["tail_n"]==tail_n)]
        pct = (sub["ratio"] > 1.5).mean() * 100
        print(f"  {label} last{tail_n}: {pct:.1f}% have ratio > 1.5")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Burst k-sweep + Slope 분석
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: Burst Threshold k-sweep + Slope Analysis")
print("=" * 60)

# 4-A. 인스턴스별 cpu_usage mean/std
inst_stats = df.groupby(["collection_id", "instance_index", "final_event_type"])[
    "cpu_usage"].agg(["mean", "std"]).reset_index()
inst_stats.columns = ["collection_id", "instance_index", "final_event_type", "cpu_mean", "cpu_std"]

df_m = df.merge(inst_stats, on=["collection_id", "instance_index", "final_event_type"])

print("\n[Burst k-sweep: % of time steps where max_cpu_usage > mean + k*std]")
k_results = {}
for k in [1.5, 2.0, 2.5, 3.0]:
    burst_mask = df_m["max_cpu_usage"] > df_m["cpu_mean"] + k * df_m["cpu_std"]
    overall = burst_mask.mean() * 100
    fail_r   = burst_mask[df_m["final_event_type"]==5].mean() * 100
    finish_r = burst_mask[df_m["final_event_type"]==6].mean() * 100
    k_results[k] = {"overall": overall, "FAIL": fail_r, "FINISH": finish_r}
    print(f"  k={k}: overall={overall:.2f}%  FAIL={fail_r:.2f}%  FINISH={finish_r:.2f}%")

# 5~15% 범위에 속하는 k 추천
best_k = None
for k in [1.5, 2.0, 2.5, 3.0]:
    if 5.0 <= k_results[k]["overall"] <= 15.0:
        best_k = k
        break
if best_k is None:
    best_k = 2.0  # fallback
print(f"\n  Recommended k: {best_k} (burst ratio {k_results[best_k]['overall']:.2f}%)")

# 4-B. CPU slope 분석 (FINISH 인스턴스)
print("\n[CPU Slope Analysis (FINISH instances)]")
slopes = []
for (cid, idx, etype), grp in df.groupby(["collection_id", "instance_index", "final_event_type"]):
    grp = grp.sort_values("time_step_sec")
    if len(grp) < 8:
        continue
    x = np.arange(len(grp))
    slope, _ = np.polyfit(x, grp["cpu_usage"].values, 1)
    slopes.append({"collection_id": cid, "instance_index": idx,
                   "final_event_type": etype, "slope": slope})

slope_df = pd.DataFrame(slopes)
slope_stats = slope_df.groupby("final_event_type")["slope"].describe()
print(slope_stats.to_string())

# slope 기준: FINISH의 상위 25%를 "rising"
finish_slopes = slope_df[slope_df["final_event_type"]==6]["slope"]
slope_rising_threshold = finish_slopes.quantile(0.75)
slope_stable_threshold = finish_slopes.quantile(0.25)
print(f"\n  Slope thresholds (FINISH):")
print(f"    normal-rising : slope > {slope_rising_threshold:.6f} (top 25%)")
print(f"    normal-stable : slope < {slope_stable_threshold:.6f} (bottom 25%)")
print(f"    (remainder = 'normal-moderate')")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# k-sweep bar
ks = list(k_results.keys())
x_pos = np.arange(len(ks))
w = 0.25
axes[0].bar(x_pos - w, [k_results[k]["overall"] for k in ks], w, label="Overall", color="#3498db")
axes[0].bar(x_pos,     [k_results[k]["FAIL"]    for k in ks], w, label="FAIL",    color="#e74c3c")
axes[0].bar(x_pos + w, [k_results[k]["FINISH"]  for k in ks], w, label="FINISH",  color="#2ecc71")
axes[0].axhline(5,  color="gray", linestyle="--", alpha=0.7, linewidth=1)
axes[0].axhline(15, color="gray", linestyle="--", alpha=0.7, linewidth=1)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([f"k={k}" for k in ks])
axes[0].set_ylabel("burst time step %")
axes[0].set_title(f"Burst k-sweep\n(target 5~15%, best k={best_k})")
axes[0].legend(fontsize=8)

# slope 분포 KDE
for etype, label, color in [(5,"FAIL","#e74c3c"), (6,"FINISH","#2ecc71")]:
    sub = slope_df[slope_df["final_event_type"]==etype]["slope"].clip(
        slope_df["slope"].quantile(0.005), slope_df["slope"].quantile(0.995))
    sub.plot.kde(ax=axes[1], label=label, color=color, bw_method=0.3)
axes[1].axvline(0, color="navy", linestyle="--", alpha=0.6)
axes[1].axvline(slope_rising_threshold, color="orange", linestyle="--", alpha=0.8,
                label=f"rising th={slope_rising_threshold:.5f}")
axes[1].axvline(slope_stable_threshold, color="purple", linestyle="--", alpha=0.8,
                label=f"stable th={slope_stable_threshold:.5f}")
axes[1].set_title("CPU Slope Distribution by Class")
axes[1].set_xlabel("slope (cpu_usage / step)")
axes[1].legend(fontsize=7)

# FINISH slope 히스토그램 + 구분선
finish_s = slope_df[slope_df["final_event_type"]==6]["slope"].clip(
    slope_df["slope"].quantile(0.01), slope_df["slope"].quantile(0.99))
axes[2].hist(finish_s, bins=60, color="#2ecc71", alpha=0.75, edgecolor="none")
axes[2].axvline(slope_stable_threshold, color="purple", linestyle="--",
                label=f"stable|moderate ({slope_stable_threshold:.5f})")
axes[2].axvline(slope_rising_threshold, color="orange", linestyle="--",
                label=f"moderate|rising ({slope_rising_threshold:.5f})")
# 영역 채우기
axes[2].axvspan(finish_s.min(), slope_stable_threshold, alpha=0.15, color="purple", label="stable zone")
axes[2].axvspan(slope_rising_threshold, finish_s.max(), alpha=0.15, color="orange", label="rising zone")
axes[2].set_title("FINISH Slope → stable/rising boundary")
axes[2].set_xlabel("slope")
axes[2].legend(fontsize=7)

plt.suptitle("Section 4: Burst Threshold & Slope Analysis", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "12_burst_k_sweep.png", dpi=150)
plt.close()
print("  → 12_burst_k_sweep.png saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Sliding Window 설계 근거
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: Sliding Window Design")
print("=" * 60)

# 5-A. ts_len 분위수
print("\n[TS Length Percentiles]")
for etype, label in [(5, "FAIL"), (6, "FINISH")]:
    vals = ts_len[ts_len.index.get_level_values("final_event_type") == etype]
    pcts = np.percentile(vals, [10, 25, 50, 75, 90, 95])
    print(f"  {label}: p10={pcts[0]:.0f}  p25={pcts[1]:.0f}  p50={pcts[2]:.0f}  "
          f"p75={pcts[3]:.0f}  p90={pcts[4]:.0f}  p95={pcts[5]:.0f}")

# 5-B. Window count simulation
def count_windows(length, seq_len, pred_len, stride):
    if length < seq_len + pred_len:
        return 0
    return (length - seq_len - pred_len) // stride + 1

print("\n[Window Count Simulation]")
configs = [
    (32, 16, 16,  "seq32/pred16/stride16"),
    (32, 32, 16,  "seq32/pred32/stride16"),
    (64, 32, 32,  "seq64/pred32/stride32"),
    (96, 32, 32,  "seq96/pred32/stride32"),
]
window_summary = {}
for seq_l, pred_l, stride, name in configs:
    counts = ts_len.apply(lambda n: count_windows(n, seq_l, pred_l, stride))
    by_cls = counts.groupby(level="final_event_type").sum()
    fail_w   = int(by_cls.get(5, 0))
    finish_w = int(by_cls.get(6, 0))
    usable = ts_len.apply(lambda n: n >= seq_l + pred_l)
    usable_fail   = int(usable[usable.index.get_level_values("final_event_type")==5].sum())
    usable_finish = int(usable[usable.index.get_level_values("final_event_type")==6].sum())
    window_summary[name] = {
        "fail_windows": fail_w, "finish_windows": finish_w,
        "usable_fail_inst": usable_fail, "usable_finish_inst": usable_finish
    }
    print(f"  {name}: FAIL={fail_w:,}win({usable_fail}inst)  FINISH={finish_w:,}win({usable_finish}inst)")

# 5-C. Autocorrelation (FINISH 50개 샘플)
print("\n[Autocorrelation Analysis (FINISH 샘플 50개)]")
try:
    from statsmodels.tsa.stattools import acf
    acf_available = True
except ImportError:
    acf_available = False
    print("  statsmodels not available, skipping ACF")

acf_matrix = []
sample_count = 0
for (cid, idx, etype), grp in df.groupby(["collection_id", "instance_index", "final_event_type"]):
    if etype != 6 or sample_count >= 50:
        continue
    grp = grp.sort_values("time_step_sec")
    if len(grp) >= 97:  # seq=96 + 1
        if acf_available:
            a = acf(grp["cpu_usage"].values, nlags=96, fft=True)
            acf_matrix.append(a)
        sample_count += 1

effective_lag = None
if acf_available and len(acf_matrix) > 0:
    mean_acf = np.mean(acf_matrix, axis=0)
    # 0.1 아래로 떨어지는 첫 lag
    below_01 = np.where(mean_acf[1:] < 0.1)[0]
    effective_lag = int(below_01[0] + 1) if len(below_01) > 0 else 96
    print(f"  Samples used: {len(acf_matrix)}")
    print(f"  Mean ACF drops below 0.1 at lag: {effective_lag}")
    print(f"  → Effective lookback: ~{effective_lag} steps (~{effective_lag*5/60:.1f}h)")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ts_len 분포
for ax, (etype, label, color) in zip([axes[0,0], axes[0,1]],
                                      [(5,"FAIL","#e74c3c"),(6,"FINISH","#2ecc71")]):
    vals = ts_len[ts_len.index.get_level_values("final_event_type")==etype]
    ax.hist(vals, bins=60, color=color, alpha=0.8, edgecolor="none")
    for p, pv in zip([10,50,90], np.percentile(vals,[10,50,90])):
        ax.axvline(pv, linestyle="--", label=f"p{p}={pv:.0f}", linewidth=1.5)
    ax.set_title(f"TS Length Distribution ({label})")
    ax.set_xlabel("# time steps")
    ax.set_ylabel("# instances")
    ax.legend(fontsize=8)

# window count bar
names = list(window_summary.keys())
fail_ws   = [window_summary[n]["fail_windows"]   / 1000 for n in names]
finish_ws = [window_summary[n]["finish_windows"] / 1000 for n in names]
x_pos = np.arange(len(names))
axes[1,0].bar(x_pos - 0.2, fail_ws,   0.4, label="FAIL",   color="#e74c3c")
axes[1,0].bar(x_pos + 0.2, finish_ws, 0.4, label="FINISH", color="#2ecc71")
axes[1,0].set_xticks(x_pos)
axes[1,0].set_xticklabels(names, rotation=15, fontsize=8)
axes[1,0].set_ylabel("windows (K)")
axes[1,0].set_title("Window Count by Config")
axes[1,0].legend()

# ACF
if acf_available and len(acf_matrix) > 0:
    axes[1,1].plot(range(len(mean_acf)), mean_acf, color="#3498db", linewidth=2)
    axes[1,1].axhline(0.1, color="red", linestyle="--", label="0.1 threshold")
    if effective_lag:
        axes[1,1].axvline(effective_lag, color="orange", linestyle="--",
                          label=f"lag={effective_lag}")
    axes[1,1].axhline(0, color="gray", alpha=0.5)
    axes[1,1].set_title(f"Mean ACF (FINISH, n={len(acf_matrix)})")
    axes[1,1].set_xlabel("lag (5-min steps)")
    axes[1,1].set_ylabel("autocorrelation")
    axes[1,1].legend(fontsize=8)
else:
    axes[1,1].text(0.5, 0.5, "ACF not computed\n(statsmodels unavailable or\ninsufficient samples)",
                  ha="center", va="center", transform=axes[1,1].transAxes)

plt.suptitle("Section 5: Sliding Window Design", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "13_window_simulation.png", dpi=150)
plt.close()
print("  → 13_window_simulation.png saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: RAG DB 구성 가능성
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 6: RAG DB Feasibility (5-class simulation)")
print("=" * 60)

# slope_df와 inst_stats를 합쳐서 5클래스 분류
inst_class = inst_stats.merge(
    slope_df[["collection_id", "instance_index", "final_event_type", "slope"]],
    on=["collection_id", "instance_index", "final_event_type"],
    how="left"
)

# burst 여부: 인스턴스 내 burst step 비율 (k=best_k)
df_m2 = df.merge(inst_stats, on=["collection_id", "instance_index", "final_event_type"])
df_m2["is_burst_step"] = df_m2["max_cpu_usage"] > df_m2["cpu_mean"] + best_k * df_m2["cpu_std"]
burst_ratio_per_inst = df_m2.groupby(
    ["collection_id", "instance_index", "final_event_type"])["is_burst_step"].mean().reset_index()
burst_ratio_per_inst.columns = ["collection_id","instance_index","final_event_type","burst_ratio"]

inst_class = inst_class.merge(burst_ratio_per_inst,
                               on=["collection_id","instance_index","final_event_type"], how="left")

# 5클래스 분류 규칙
BURST_RATIO_THRESH = 0.05  # burst step이 5% 이상

def classify_5class(row):
    if row["final_event_type"] == 5:
        # FAIL
        if row["burst_ratio"] >= BURST_RATIO_THRESH:
            return "failure-sudden"
        else:
            return "failure-gradual"
    else:
        # FINISH
        if row["burst_ratio"] >= BURST_RATIO_THRESH:
            return "burst"
        elif row["slope"] >= slope_rising_threshold:
            return "normal-rising"
        else:
            return "normal-stable"

inst_class["rag_class"] = inst_class.apply(classify_5class, axis=1)

print("\n[5-Class Distribution (all instances)]")
class_dist = inst_class["rag_class"].value_counts()
print(class_dist.to_string())

# RAG vs Model 분리
inst_class["pool"] = np.where(inst_class["collection_id"] % 10 < 2, "RAG", "Model")

print("\n[5-Class × Pool Distribution]")
pool_dist = inst_class.groupby(["pool","rag_class"]).size().unstack(fill_value=0)
print(pool_dist.to_string())

# RAG pool window count (seq_len=32, stride=32)
SEQ_LEN_RAG = 32
PRED_LEN_RAG = 16
STRIDE_RAG = 32

rag_inst_ids = inst_class[inst_class["pool"]=="RAG"][
    ["collection_id","instance_index","final_event_type","rag_class"]]

rag_ts = ts_len.reset_index()
rag_ts.columns = ["collection_id","instance_index","final_event_type","ts_len"]
rag_ts = rag_ts.merge(rag_inst_ids, on=["collection_id","instance_index","final_event_type"])
rag_ts["n_windows"] = rag_ts["ts_len"].apply(
    lambda n: count_windows(n, SEQ_LEN_RAG, PRED_LEN_RAG, STRIDE_RAG))

print(f"\n[RAG DB Window Count (seq={SEQ_LEN_RAG}, pred={PRED_LEN_RAG}, stride={STRIDE_RAG})]")
rag_win_by_class = rag_ts.groupby("rag_class")["n_windows"].sum()
print(rag_win_by_class.to_string())
print(f"  Total RAG windows: {rag_ts['n_windows'].sum():,}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 전체 5클래스 분포
colors_5 = {"normal-stable": "#27ae60", "normal-rising": "#f39c12",
            "burst": "#e67e22", "failure-gradual": "#c0392b", "failure-sudden": "#8e44ad"}
class_dist.plot(kind="bar", ax=axes[0], color=[colors_5.get(c,"gray") for c in class_dist.index],
                edgecolor="k")
axes[0].set_title("5-Class Distribution\n(all instances)")
axes[0].set_xlabel("")
axes[0].set_ylabel("# instances")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                     ha="center", va="bottom", fontsize=8)
axes[0].tick_params(axis="x", rotation=30)

# RAG vs Model 분포 비교
pool_dist_pct = pool_dist.div(pool_dist.sum(axis=1), axis=0) * 100
pool_dist_pct.T.plot(kind="bar", ax=axes[1], color=["#3498db","#e74c3c"], edgecolor="k")
axes[1].set_title("5-Class % by Pool\n(RAG vs Model)")
axes[1].set_xlabel("")
axes[1].set_ylabel("%")
axes[1].legend(title="Pool")
axes[1].tick_params(axis="x", rotation=30)

# RAG window count bar
rag_win_by_class.plot(kind="bar", ax=axes[2],
                       color=[colors_5.get(c,"gray") for c in rag_win_by_class.index],
                       edgecolor="k")
axes[2].set_title(f"RAG DB Windows by Class\n(seq={SEQ_LEN_RAG},pred={PRED_LEN_RAG},stride={STRIDE_RAG})")
axes[2].set_xlabel("")
axes[2].set_ylabel("# windows")
for p in axes[2].patches:
    axes[2].annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                     ha="center", va="bottom", fontsize=8)
axes[2].tick_params(axis="x", rotation=30)

plt.suptitle("Section 6: RAG DB Feasibility (5-class)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "14_rag_db_feasibility.png", dpi=150)
plt.close()
print("  → 14_rag_db_feasibility.png saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: max_mem_usage NULL 처리
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 7: max_mem_usage NULL Analysis")
print("=" * 60)

null_mask = df["max_mem_usage"].isnull()
print(f"\n  Total NULL: {null_mask.sum():,} ({null_mask.mean()*100:.2f}%)")

# 클래스별 NULL 비율
null_by_class = df.groupby("final_event_type")["max_mem_usage"].apply(lambda x: x.isnull().mean()*100)
null_by_class.index = null_by_class.index.map({5:"FAIL",6:"FINISH"})
print(f"  NULL by class:\n{null_by_class.to_string()}")

# NULL이 있는 인스턴스 비율
inst_has_null = df.groupby(["collection_id","instance_index"])["max_mem_usage"]\
                  .apply(lambda x: x.isnull().any())
print(f"\n  Instances with any NULL: {inst_has_null.sum():,} / {len(inst_has_null):,} ({inst_has_null.mean()*100:.1f}%)")

# NULL의 정규화 위치 분포 (시작/중간/끝)
df_null_pos = df[null_mask].copy()
df_null_pos["ts_len_inst"] = df_null_pos.groupby(["collection_id","instance_index"])\
                                         ["time_step_sec"].transform("count")
df_null_pos["rank"] = df_null_pos.groupby(["collection_id","instance_index"])\
                                  ["time_step_sec"].rank(method="first")
df_null_pos["norm_pos"] = (df_null_pos["rank"] - 1) / (df_null_pos["ts_len_inst"] - 1).clip(lower=1)
print(f"\n  NULL position (normalized 0=start, 1=end):")
print(f"  {df_null_pos['norm_pos'].describe().to_string()}")

# 대체 방안: max_mem_usage가 NULL인 경우 mem_usage로 대체
df["max_mem_usage_filled"] = df["max_mem_usage"].fillna(df["mem_usage"])
remaining_null = df["max_mem_usage_filled"].isnull().sum()
print(f"\n  After fillna(mem_usage): {remaining_null:,} remaining NULLs")
print(f"  → Recommended: max_mem_usage.fillna(mem_usage)")

# ═══════════════════════════════════════════════════════════════════════════════
# 최종 요약 저장
# ═══════════════════════════════════════════════════════════════════════════════
summary = {
    "data_quality": {
        "duplicate_rows": int(dup_count),
        "non_300s_gap_pct": round(non_300, 3),
    },
    "burst_threshold": {
        "recommended_k": float(best_k),
        "overall_burst_pct": round(k_results[best_k]["overall"], 2),
        "fail_burst_pct": round(k_results[best_k]["FAIL"], 2),
        "finish_burst_pct": round(k_results[best_k]["FINISH"], 2),
    },
    "slope_thresholds": {
        "normal_rising_gt": round(float(slope_rising_threshold), 7),
        "normal_stable_lt": round(float(slope_stable_threshold), 7),
    },
    "window_configs": window_summary,
    "rag_db": {
        "5class_distribution": class_dist.to_dict(),
        "rag_windows_by_class": rag_win_by_class.to_dict(),
        "total_rag_windows": int(rag_ts["n_windows"].sum()),
    },
    "effective_lookback_lag": effective_lag,
    "max_mem_usage_null_pct": round(null_mask.mean() * 100, 2),
    "null_treatment": "fillna(mem_usage)",
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("=== DONE ===")
print("=" * 60)
print(f"\nOutputs saved to: {OUT_DIR}")
print(json.dumps(summary, indent=2))
