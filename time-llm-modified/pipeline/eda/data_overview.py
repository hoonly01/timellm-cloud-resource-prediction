"""
EDA for Google Cluster Trace v3 (Cell a): 데이터 개요
- 클래스 분포 (FAIL/FINISH)
- 인스턴스별 시계열 길이 분포
- CPU/Memory 사용량 분포 및 상관관계
- Burst 패턴 분석 (max - avg 기반)
- NULL 현황 (requested_cpu, requested_mem)
- resource_pressure 분포
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── 설정 ────────────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = PROJ_ROOT / "data"
OUT_DIR   = PROJ_ROOT / "eda" / "output" / "data_overview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
print("Loading data...")
files = sorted(DATA_DIR.glob("gcluster_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"  Total rows : {len(df):,}")
print(f"  Columns    : {list(df.columns)}")

# ── 1. 기본 통계 ─────────────────────────────────────────────────────────────
print("\n=== Basic Info ===")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDescribe (numeric):")
print(df.describe().T.to_string())

# ── 2. 클래스 분포 ───────────────────────────────────────────────────────────
# 행 기준
row_dist = df["final_event_type"].value_counts().rename({5: "FAIL", 6: "FINISH"})
# 인스턴스 기준
inst_dist = (df.groupby("final_event_type")["collection_id"]
               .nunique()
               .rename({5: "FAIL", 6: "FINISH"}))
n_instances = df.groupby(["collection_id", "instance_index"]).ngroups

print(f"\n=== Class Distribution ===")
print(f"  Total instances : {n_instances:,}")
print(f"  Row dist  :\n{row_dist}")
print(f"  Inst dist :\n{inst_dist}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
row_dist.plot(kind="bar", ax=axes[0], color=["#e74c3c", "#2ecc71"], edgecolor="k")
axes[0].set_title("Row Count by Class")
axes[0].set_xlabel("")
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                     ha="center", va="bottom", fontsize=9)

inst_dist.plot(kind="bar", ax=axes[1], color=["#e74c3c", "#2ecc71"], edgecolor="k")
axes[1].set_title("Instance Count by Class")
axes[1].set_xlabel("")
axes[1].set_ylabel("Count")
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                     ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_class_distribution.png", dpi=150)
plt.close()
print("  → 01_class_distribution.png saved")

# ── 3. 시계열 길이 분포 ──────────────────────────────────────────────────────
ts_len = df.groupby(["collection_id", "instance_index", "final_event_type"]).size().reset_index(name="ts_len")
print(f"\n=== Time-series Length ===")
print(ts_len.groupby("final_event_type")["ts_len"].describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (etype, label, color) in zip(axes, [(5, "FAIL", "#e74c3c"), (6, "FINISH", "#2ecc71")]):
    vals = ts_len[ts_len["final_event_type"] == etype]["ts_len"]
    ax.hist(vals, bins=60, color=color, edgecolor="k", alpha=0.8)
    ax.axvline(vals.median(), color="navy", linestyle="--", label=f"median={vals.median():.0f}")
    ax.set_title(f"TS Length - {label}")
    ax.set_xlabel("# time steps (5-min buckets)")
    ax.set_ylabel("# instances")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "02_ts_length_distribution.png", dpi=150)
plt.close()
print("  → 02_ts_length_distribution.png saved")

# ── 4. CPU / Memory 분포 ────────────────────────────────────────────────────
metrics = ["cpu_usage", "max_cpu_usage", "mem_usage", "max_mem_usage"]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for col_idx, metric in enumerate(metrics):
    for row_idx, (etype, label, color) in enumerate([(5, "FAIL", "#e74c3c"), (6, "FINISH", "#2ecc71")]):
        ax = axes[row_idx][col_idx]
        data = df[df["final_event_type"] == etype][metric].dropna()
        ax.hist(data, bins=80, color=color, alpha=0.75, edgecolor="none")
        ax.set_title(f"{metric}\n({label})", fontsize=9)
        ax.set_xlabel("normalized value")
        if col_idx == 0:
            ax.set_ylabel(label)
plt.suptitle("CPU / Memory Usage Distribution by Class", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_cpu_mem_distribution.png", dpi=150)
plt.close()
print("  → 03_cpu_mem_distribution.png saved")

# ── 5. CPU-Memory 상관관계 ───────────────────────────────────────────────────
# 인스턴스 평균값 기반 scatter
inst_mean = df.groupby(["collection_id", "instance_index", "final_event_type"])[
    ["cpu_usage", "mem_usage"]].mean().reset_index()

# Pearson / Spearman 상관계수
from scipy import stats
for etype, label in [(5, "FAIL"), (6, "FINISH")]:
    sub = inst_mean[inst_mean["final_event_type"] == etype]
    r_p, p_p = stats.pearsonr(sub["cpu_usage"], sub["mem_usage"])
    r_s, p_s = stats.spearmanr(sub["cpu_usage"], sub["mem_usage"])
    print(f"\n{label} CPU-Mem correlation (instance mean):")
    print(f"  Pearson  r={r_p:.4f}, p={p_p:.2e}")
    print(f"  Spearman r={r_s:.4f}, p={p_s:.2e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (etype, label, color) in zip(axes, [(5, "FAIL", "#e74c3c"), (6, "FINISH", "#2ecc71")]):
    sub = inst_mean[inst_mean["final_event_type"] == etype]
    # 밀도가 높아 hexbin 사용
    hb = ax.hexbin(sub["cpu_usage"], sub["mem_usage"], gridsize=50,
                   cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax, label="count")
    r, _ = stats.pearsonr(sub["cpu_usage"], sub["mem_usage"])
    ax.set_title(f"CPU vs Memory ({label})\nPearson r={r:.3f}")
    ax.set_xlabel("mean cpu_usage")
    ax.set_ylabel("mean mem_usage")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_cpu_mem_correlation.png", dpi=150)
plt.close()
print("  → 04_cpu_mem_correlation.png saved")

# ── 6. Burst 패턴 분석 ──────────────────────────────────────────────────────
df["cpu_burst"] = df["max_cpu_usage"] - df["cpu_usage"]
df["mem_burst"] = df["max_mem_usage"] - df["mem_usage"]

# 인스턴스별 최대 burst
inst_burst = df.groupby(["collection_id", "instance_index", "final_event_type"])[
    ["cpu_burst", "mem_burst"]].max().reset_index()

print("\n=== Burst Analysis ===")
print(inst_burst.groupby("final_event_type")[["cpu_burst", "mem_burst"]].describe())

# burst 분포 비교 (violin)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, title in zip(axes,
                              ["cpu_burst", "mem_burst"],
                              ["CPU Burst (max - avg)", "Memory Burst (max - avg)"]):
    plot_data = [
        inst_burst[inst_burst["final_event_type"] == 5][metric].clip(upper=inst_burst[metric].quantile(0.99)),
        inst_burst[inst_burst["final_event_type"] == 6][metric].clip(upper=inst_burst[metric].quantile(0.99)),
    ]
    vp = ax.violinplot(plot_data, positions=[0, 1], showmedians=True)
    vp["bodies"][0].set_facecolor("#e74c3c")
    vp["bodies"][1].set_facecolor("#2ecc71")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FAIL", "FINISH"])
    ax.set_title(title)
    ax.set_ylabel("burst magnitude (normalized)")
plt.suptitle("Burst Pattern by Class (99th percentile clipped)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "05_burst_pattern.png", dpi=150)
plt.close()
print("  → 05_burst_pattern.png saved")

# burst 임계값 제안 (95th percentile of FAIL)
cpu_thresh = inst_burst[inst_burst["final_event_type"] == 5]["cpu_burst"].quantile(0.5)
mem_thresh = inst_burst[inst_burst["final_event_type"] == 5]["mem_burst"].quantile(0.5)
print(f"\n  Burst threshold suggestion (FAIL median):")
print(f"    cpu_burst >= {cpu_thresh:.4f}")
print(f"    mem_burst >= {mem_thresh:.4f}")

# ── 7. Resource Pressure ────────────────────────────────────────────────────
null_cpu = df["requested_cpu"].isnull().mean() * 100
null_mem = df["requested_mem"].isnull().mean() * 100
print(f"\n=== Resource Request NULL rate ===")
print(f"  requested_cpu NULL: {null_cpu:.1f}%")
print(f"  requested_mem NULL: {null_mem:.1f}%")

df_rp = df.dropna(subset=["requested_cpu", "requested_mem"]).copy()
df_rp["cpu_pressure"]  = df_rp["cpu_usage"]  / df_rp["requested_cpu"].replace(0, np.nan)
df_rp["mem_pressure"]  = df_rp["mem_usage"]  / df_rp["requested_mem"].replace(0, np.nan)

# 이상치 클리핑 (99.9th)
clip_cpu = df_rp["cpu_pressure"].quantile(0.999)
clip_mem = df_rp["mem_pressure"].quantile(0.999)
df_rp["cpu_pressure"] = df_rp["cpu_pressure"].clip(upper=clip_cpu)
df_rp["mem_pressure"] = df_rp["mem_pressure"].clip(upper=clip_mem)

print("\n=== Resource Pressure (row-level, clipped at 99.9th) ===")
print(df_rp.groupby("final_event_type")[["cpu_pressure", "mem_pressure"]].describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, title in zip(axes,
                              ["cpu_pressure", "mem_pressure"],
                              ["CPU Pressure (usage/requested)", "Mem Pressure (usage/requested)"]):
    for etype, label, color in [(5, "FAIL", "#e74c3c"), (6, "FINISH", "#2ecc71")]:
        vals = df_rp[df_rp["final_event_type"] == etype][metric]
        ax.hist(vals, bins=80, alpha=0.5, label=label, color=color, density=True)
    ax.set_title(title)
    ax.set_xlabel("pressure ratio")
    ax.set_ylabel("density")
    ax.legend()
plt.suptitle("Resource Pressure by Class", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "06_resource_pressure.png", dpi=150)
plt.close()
print("  → 06_resource_pressure.png saved")

# ── 8. 시간대별 평균 CPU/Mem (FAIL vs FINISH) ───────────────────────────────
# 인스턴스 시계열을 정규화 위치(0~1)로 변환 후 평균
print("\nComputing normalized time position profiles...")
df_norm = df.copy()
df_norm["ts_len_instance"] = df_norm.groupby(
    ["collection_id", "instance_index"])["time_step_sec"].transform("count")
df_norm["rank"] = df_norm.groupby(
    ["collection_id", "instance_index"])["time_step_sec"].rank(method="first")
df_norm["pos"] = ((df_norm["rank"] - 1) / (df_norm["ts_len_instance"] - 1).clip(lower=1) * 19).astype(int)

profile = df_norm.groupby(["final_event_type", "pos"])[
    ["cpu_usage", "mem_usage"]].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title in zip(axes,
                              ["cpu_usage", "mem_usage"],
                              ["Mean CPU Usage over Normalized Time",
                               "Mean Memory Usage over Normalized Time"]):
    for etype, label, color in [(5, "FAIL", "#e74c3c"), (6, "FINISH", "#2ecc71")]:
        sub = profile[profile["final_event_type"] == etype]
        ax.plot(sub["pos"] / 19, sub[metric], label=label, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Normalized time (0=start, 1=end)")
    ax.set_ylabel("mean value (normalized)")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "07_temporal_profile.png", dpi=150)
plt.close()
print("  → 07_temporal_profile.png saved")

# ── 9. RAG / Model 분리 미리보기 ─────────────────────────────────────────────
inst_unique = df[["collection_id", "instance_index", "final_event_type"]].drop_duplicates()
inst_unique["split"] = np.where(inst_unique["collection_id"] % 10 < 2, "RAG", "Model")
split_dist = inst_unique.groupby(["split", "final_event_type"]).size().unstack(fill_value=0)
split_dist.columns = ["FAIL", "FINISH"]
print(f"\n=== RAG / Model Split Preview ===")
print(split_dist)

# ── 요약 저장 ────────────────────────────────────────────────────────────────
summary = {
    "total_rows": len(df),
    "total_instances": n_instances,
    "fail_instances": int(ts_len[ts_len["final_event_type"] == 5].shape[0]),
    "finish_instances": int(ts_len[ts_len["final_event_type"] == 6].shape[0]),
    "ts_len_fail_median": float(ts_len[ts_len["final_event_type"] == 5]["ts_len"].median()),
    "ts_len_finish_median": float(ts_len[ts_len["final_event_type"] == 6]["ts_len"].median()),
    "requested_cpu_null_pct": round(null_cpu, 2),
    "requested_mem_null_pct": round(null_mem, 2),
    "cpu_burst_fail_median": round(cpu_thresh, 5),
    "mem_burst_fail_median": round(mem_thresh, 5),
}
import json
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== Done ===")
print(f"All outputs saved to: {OUT_DIR}")
print(json.dumps(summary, indent=2))
