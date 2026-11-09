"""
Regression-oriented EDA for Time-LLM + RAG ablation.

Produces 4 analyses in eda/output/forecast_diagnosis/:
  B1. Per-feature ACF (lag 0..31)
  B2. Input window vs target window correlation
  B3. Retrieval consistency (RAG efficacy diagnostic)
  B4. Per-instance difficulty via naive baselines

Run (from pipeline/): python eda/forecast_diagnosis.py
"""

import argparse
import json
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FEATURE_NAMES = [
    "cpu_usage", "max_cpu_usage", "mem_usage", "max_mem_usage",
    "page_cache_memory", "assigned_mem",
    "cpu_pressure", "mem_pressure", "cpu_burst", "mem_burst",
]


def extract_stats(x):
    """Matches Time-LLM/utils/rag.py extract_stats."""
    N, T, C = x.shape
    t_c = np.arange(T, dtype=np.float32) - (T - 1) / 2.0
    t_var = (t_c ** 2).mean()
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    xmin = x.min(axis=1)
    xmax = x.max(axis=1)
    trend = ((x - mean[:, np.newaxis, :]) * t_c[np.newaxis, :, np.newaxis]).mean(axis=1) / t_var
    return np.concatenate([mean, std, xmin, xmax, trend], axis=1).astype(np.float32)


def b1_acf(x_windows, out_dir):
    N, T, C = x_windows.shape
    acf = np.zeros((C, T), dtype=np.float64)
    for c in range(C):
        xc = x_windows[..., c].astype(np.float64)
        m = xc.mean(axis=1, keepdims=True)
        v = xc.var(axis=1) + 1e-12
        xc_centered = xc - m
        acf[c, 0] = 1.0
        for k in range(1, T):
            cov = (xc_centered[:, :-k] * xc_centered[:, k:]).mean(axis=1) / v
            acf[c, k] = cov.mean()
    payload = {FEATURE_NAMES[c]: acf[c].tolist() for c in range(C)}
    with open(out_dir / "01_acf.json", "w") as f:
        json.dump({"n_windows_sampled": int(N), "acf": payload}, f, indent=2)
    fig, ax = plt.subplots(figsize=(11, 6))
    for c in range(C):
        ax.plot(range(T), acf[c], label=FEATURE_NAMES[c], lw=1.4)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF (mean over windows)")
    ax.set_title(f"Per-feature autocorrelation  (N={N:,} sampled windows, T={T})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "01_acf_per_feature.png", dpi=110)
    plt.close(fig)
    return acf


def b2_input_target(x, y, out_dir):
    N, T, C = x.shape
    x_last = x[:, -1, :]
    y_first = y[:, 0, :]
    x_mean = x.mean(axis=1)
    y_mean = y.mean(axis=1)
    t_c = np.arange(T, dtype=np.float32) - (T - 1) / 2.0
    t_var = (t_c ** 2).mean() + 1e-12
    x_trend = ((x - x_mean[:, None, :]) * t_c[None, :, None]).mean(axis=1) / t_var
    y_trend = ((y - y_mean[:, None, :]) * t_c[None, :, None]).mean(axis=1) / t_var

    pairs = [
        ("input_last_vs_target_first", x_last, y_first),
        ("input_mean_vs_target_mean", x_mean, y_mean),
        ("input_trend_vs_target_trend", x_trend, y_trend),
    ]
    corr = np.zeros((len(pairs), C))
    for i, (_, a, b) in enumerate(pairs):
        for c in range(C):
            ac, bc = a[:, c], b[:, c]
            if ac.std() < 1e-12 or bc.std() < 1e-12:
                corr[i, c] = 0.0
            else:
                corr[i, c] = np.corrcoef(ac, bc)[0, 1]
    out = {pairs[i][0]: {FEATURE_NAMES[c]: float(corr[i, c]) for c in range(C)} for i in range(len(pairs))}
    with open(out_dir / "02_input_target.json", "w") as f:
        json.dump({"n_windows": int(N), "correlations": out}, f, indent=2)

    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(C)
    width = 0.27
    colors = ["steelblue", "darkorange", "seagreen"]
    for i, (name, _, _) in enumerate(pairs):
        ax.bar(x_pos + (i - 1) * width, corr[i], width, label=name, color=colors[i])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Input -> Target window correlation (N={N:,})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "02_input_target_corr.png", dpi=110)
    plt.close(fig)
    return corr


def b3_retrieval(val_x, val_y, rag_db_dir, n_query, top_k, out_dir):
    index = faiss.read_index(str(rag_db_dir / "faiss.index"))
    meta = np.load(rag_db_dir / "metadata.npz")
    rag_y = meta["y"]
    M, T, C = rag_y.shape

    baseline_std = rag_y.std(axis=(0, 1))

    rng = np.random.default_rng(2026)
    n_query = min(n_query, len(val_x))
    q_idx = rng.choice(len(val_x), n_query, replace=False)
    q_x = val_x[q_idx]
    q_y = val_y[q_idx]

    q_emb = extract_stats(q_x)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
    q_emb = q_emb.astype(np.float32)
    _, indices = index.search(q_emb, top_k)

    inter_std = np.zeros((n_query, C))
    ret_mae = np.zeros((n_query, C))
    for i in range(n_query):
        r_y = rag_y[indices[i]]
        r_mean_over_t = r_y.mean(axis=1)
        inter_std[i] = r_mean_over_t.std(axis=0)
        pred = r_y.mean(axis=0)
        ret_mae[i] = np.abs(pred - q_y[i]).mean(axis=0)

    avg_inter = inter_std.mean(axis=0)
    avg_mae = ret_mae.mean(axis=0)
    ratio = avg_inter / (baseline_std + 1e-12)

    results = {
        "n_query": int(n_query),
        "top_k": int(top_k),
        "rag_db_size": int(M),
        "per_feature": {
            FEATURE_NAMES[c]: {
                "inter_retrieved_std": float(avg_inter[c]),
                "baseline_std": float(baseline_std[c]),
                "ratio": float(ratio[c]),
                "retrieved_mean_mae": float(avg_mae[c]),
            }
            for c in range(C)
        },
    }
    with open(out_dir / "03_retrieval.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(C)
    ax1.bar(x_pos, ratio, color="steelblue")
    ax1.axhline(1.0, color="red", ls="--", lw=1, label="random retrieval baseline")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(FEATURE_NAMES, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("inter-retrieved std / baseline std")
    ax1.set_title(f"Retrieval consistency  (n_query={n_query}, top_k={top_k}, DB={M:,})")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x_pos, avg_mae, color="darkorange")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(FEATURE_NAMES, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("MAE")
    ax2.set_title("Retrieved-mean (top-k) prediction MAE")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "03_retrieval_consistency.png", dpi=110)
    plt.close(fig)
    return results


def b4_difficulty(val_x, val_y, out_dir):
    N, T, C = val_x.shape
    x_last = val_x[:, -1, :]
    x_mean = val_x.mean(axis=1)
    pred_last = np.repeat(x_last[:, None, :], T, axis=1)
    pred_mean = np.repeat(x_mean[:, None, :], T, axis=1)
    mae_last = np.abs(pred_last - val_y).mean(axis=1)
    mae_mean = np.abs(pred_mean - val_y).mean(axis=1)

    stats = {}
    for c in range(C):
        stats[FEATURE_NAMES[c]] = {
            "naive_last": {
                "mean": float(mae_last[:, c].mean()),
                "median": float(np.median(mae_last[:, c])),
                "p90": float(np.percentile(mae_last[:, c], 90)),
                "p99": float(np.percentile(mae_last[:, c], 99)),
            },
            "naive_mean": {
                "mean": float(mae_mean[:, c].mean()),
                "median": float(np.median(mae_mean[:, c])),
                "p90": float(np.percentile(mae_mean[:, c], 90)),
                "p99": float(np.percentile(mae_mean[:, c], 99)),
            },
        }
    with open(out_dir / "04_difficulty.json", "w") as f:
        json.dump({"n_windows": int(N), "per_feature": stats}, f, indent=2)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for c in range(C):
        ax = axes[c // 5][c % 5]
        a = mae_last[:, c]
        b = mae_mean[:, c]
        upper = max(np.percentile(a, 99), np.percentile(b, 99))
        rng_lim = (0.0, max(upper, 1e-9))
        ax.hist(a, bins=60, range=rng_lim, alpha=0.5, label="naive_last", color="steelblue")
        ax.hist(b, bins=60, range=rng_lim, alpha=0.5, label="naive_mean", color="darkorange")
        ax.set_title(FEATURE_NAMES[c], fontsize=9)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Per-window MAE distribution  (val.npz, N={N:,}, naive baselines)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "04_difficulty_distribution.png", dpi=110)
    plt.close(fig)
    return stats


def build_report(out_dir, acf, corr, ret, diff, rag_info):
    L = []
    L.append("# Regression EDA Report\n\n")
    L.append("Time-LLM + RAG ablation 진단 (32-step forecast 관점)\n\n")

    L.append("## 1. Context\n\n")
    L.append("기존 `eda/output/data_overview/report.md`는 FAIL vs FINISH classification 관점이라 forecast regression에서 RAG가 효과 낼 영역인지 직접 답하지 못한다. 이 보고서는 ACF, input-target correlation, retrieval consistency, naive-baseline difficulty 네 축으로 사전 진단한다.\n\n")

    L.append("## 2. RAG DB 상태\n\n")
    L.append(f"- `n_windows`: {rag_info.get('n_windows', '?')}\n")
    L.append(f"- `fail_windows`: {rag_info.get('fail_windows', '?')}\n")
    L.append(f"- `rag_class_dist`: {rag_info.get('rag_class_dist', '?')}\n\n")
    L.append(
        "현재 저장소 snapshot에서 B3 retrieval consistency는 `rag_db/`에 포함된 현재 DB 상태 기준으로 계산되었다. "
        "반면 B2 input-target correlation과 B4 difficulty는 `processed/train.npz`와 `processed/val.npz` 기준이며, "
        "full retrieval pool 규모 및 class ratio는 `processed/split_summary.json`에서 별도로 확인해야 한다.\n\n"
    )

    L.append("## 3. Per-feature ACF (B1)\n\n")
    L.append("![](01_acf_per_feature.png)\n\n")
    L.append("| feature | lag=1 | lag=8 | lag=16 | lag=31 |\n")
    L.append("|---|---|---|---|---|\n")
    for c, fn in enumerate(FEATURE_NAMES):
        L.append(f"| {fn} | {acf[c, 1]:.3f} | {acf[c, 8]:.3f} | {acf[c, 16]:.3f} | {acf[c, 31]:.3f} |\n")
    L.append("\n해석: lag 31에서 ACF가 빠르게 0에 수렴 -> patch만으로 32-step 예측 어려움 -> RAG/외부 신호 헤드룸 큼. 반대로 천천히 떨어지면 patch 자체가 충분.\n\n")

    L.append("## 4. Input -> Target correlation (B2)\n\n")
    L.append("![](02_input_target_corr.png)\n\n")
    L.append("| feature | last/first | mean/mean | trend/trend |\n")
    L.append("|---|---|---|---|\n")
    for c, fn in enumerate(FEATURE_NAMES):
        L.append(f"| {fn} | {corr[0, c]:.3f} | {corr[1, c]:.3f} | {corr[2, c]:.3f} |\n")
    L.append("\n해석: 모든 r이 약하면(<0.3) input window가 target window를 결정하지 못함 -> 외부(RAG) 정보가 의미 가짐. r이 강하면(>0.7) input만으로 미래 결정됨.\n\n")

    L.append("## 5. Retrieval consistency (B3 — RAG 효과 보조 진단)\n\n")
    L.append("![](03_retrieval_consistency.png)\n\n")
    L.append("| feature | inter_std / baseline_std | retrieved-mean MAE |\n")
    L.append("|---|---|---|\n")
    for fn in FEATURE_NAMES:
        d = ret["per_feature"][fn]
        L.append(f"| {fn} | {d['ratio']:.3f} | {d['retrieved_mean_mae']:.5f} |\n")
    L.append("\n해석:\n")
    L.append("- **ratio < 0.5**: retrieved future dispersion이 global baseline dispersion보다 작아, retrieval이 future uncertainty를 좁힐 가능성을 시사\n")
    L.append("- **ratio가 1.0에 가까움**: retrieved set dispersion이 global baseline dispersion과 비슷해, 현재 진단만으로는 retrieval이 uncertainty를 거의 줄이지 못했을 가능성을 시사\n")
    L.append("- **retrieved-mean MAE를 A0 (`results/A0_Baseline.csv` epoch 9) val_mae**와 비교: retrieval-only 평균이 모델 mae보다 작으면 보조 prior로서의 잠재력이 있고, 크면 현재 구성에서는 모델이 이미 retrieval 평균보다 강함을 뜻함.\n\n")

    L.append("## 6. Per-instance difficulty (B4)\n\n")
    L.append("![](04_difficulty_distribution.png)\n\n")
    L.append("| feature | naive_last mean | naive_last p99 | naive_mean mean | naive_mean p99 |\n")
    L.append("|---|---|---|---|---|\n")
    for fn in FEATURE_NAMES:
        d = diff[fn]
        L.append(f"| {fn} | {d['naive_last']['mean']:.5f} | {d['naive_last']['p99']:.5f} | {d['naive_mean']['mean']:.5f} | {d['naive_mean']['p99']:.5f} |\n")
    L.append("\n해석: heavy-tail 분포면 일부 hard window가 metric을 지배. A0 model MAE를 trivial baseline(naive_last / naive_mean)과 직접 비교해 모델이 학습으로 얼마나 개선했는지 확인.\n\n")

    L.append("## 7. 결론 분류 가이드\n\n")
    L.append("각 feature를 위 4개 지표로 다음 3그룹으로 분류:\n\n")
    L.append("- **(A) patch로 충분**: ACF가 lag 31까지 양수 유지 + I-T corr 강함 + retrieved-mean MAE >= A0 model MAE\n")
    L.append("- **(B) RAG 헤드룸 있음**: I-T corr 약함 + retrieval ratio < 0.6 + retrieved-mean MAE < naive baseline\n")
    L.append("- **(C) 본질적으로 어려움**: ACF 빠르게 0, retrieval ratio ~ 1, naive baseline error 자체가 큼\n\n")
    L.append("table의 숫자 기준으로 수동 분류 후 다음 섹션의 권고를 적용.\n\n")

    L.append("## 8. 다음 실험 방향 권고\n\n")
    L.append("- **(B) 그룹 존재**: 현재 RAG 텍스트 prompt 방식 유지 + full DB 기반으로 ablation 재실행. A1/A2/A3 차이가 명확해질 가능성.\n")
    L.append("- **(B) 그룹 없거나 retrieved-mean MAE > A0**: RAG 텍스트 prompt가 무용한 영역. 텍스트 우회하고 retrieved time-series 벡터를 patch_embedding과 직접 concat하거나 cross-attention으로 inject하는 구조 변경 필요.\n")
    L.append("- **(A) 우세**: forecast가 RAG 없이도 잘 됨. RAG 실험 종료 + 모델 capacity(LLM layers 늘리기, LoRA 부분 unfreeze)로 방향 전환.\n")

    with open(out_dir / "report.md", "w") as f:
        f.writelines(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="processed")
    ap.add_argument("--rag_db_dir", default="rag_db")
    ap.add_argument("--out_dir", default="eda/output/forecast_diagnosis")
    ap.add_argument("--n_query", type=int, default=1000)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--acf_sample", type=int, default=5000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proc = Path(args.processed_dir)
    rag_dir = Path(args.rag_db_dir)

    print(f"[load] {proc}/train.npz, val.npz; {rag_dir}/...")
    train = np.load(proc / "train.npz")
    val = np.load(proc / "val.npz")
    train_x, train_y = train["x"], train["y"]
    val_x, val_y = val["x"], val["y"]
    print(f"  train x={train_x.shape}, y={train_y.shape}")
    print(f"  val   x={val_x.shape}, y={val_y.shape}")

    with open(rag_dir / "db_info.json") as f:
        rag_info = json.load(f)
    print(f"  RAG DB n_windows={rag_info.get('n_windows')}, fail_windows={rag_info.get('fail_windows')}")

    rng = np.random.default_rng(2026)
    sample_idx = rng.choice(len(train_x), min(args.acf_sample, len(train_x)), replace=False)

    print("\n[B1] Per-feature ACF...")
    acf = b1_acf(train_x[sample_idx], out)

    print("[B2] Input-Target correlation...")
    corr = b2_input_target(train_x, train_y, out)

    print("[B3] Retrieval consistency...")
    ret = b3_retrieval(val_x, val_y, rag_dir, args.n_query, args.top_k, out)

    print("[B4] Naive-baseline difficulty...")
    diff = b4_difficulty(val_x, val_y, out)

    print("[report] Generating report.md...")
    build_report(out, acf, corr, ret, diff, rag_info)

    print("\n=== Done ===")
    print(f"Outputs in {out}/")
    for p in sorted(out.glob("*")):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
