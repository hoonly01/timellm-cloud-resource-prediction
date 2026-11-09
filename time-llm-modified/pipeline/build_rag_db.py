"""
RAG DB 구축: 통계 벡터 임베딩 (mean/std/min/max/trend per channel) + FAISS IndexFlatL2

흐름:
  rag.npz (x: N, 32, 10)
    → per-window 통계 벡터 (50-dim)
    → L2 정규화
    → FAISS IndexFlatL2 구축
    → rag_db/ 저장
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # time-llm-modified/
from utils.rag import extract_stats  # same embedding as the retriever

EMBD_DIM = 50  # 5 stats × 10 features
TOP_K = 5

FEATURE_NAMES = [
    "cpu_usage", "max_cpu_usage",
    "mem_usage", "max_mem_usage",
    "page_cache_memory", "assigned_mem",
    "cpu_pressure", "mem_pressure",
    "cpu_burst", "mem_burst",
]
CPU_PRESSURE_IDX = 6
MEM_PRESSURE_IDX = 7
CPU_BURST_IDX = 8


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="processed")
    parser.add_argument("--out_dir", type=str, default="rag_db")
    args = parser.parse_args()

    RAG_NPZ = Path(args.processed_dir) / "rag.npz"
    OUT_DIR = Path(args.out_dir)

    # Step 1: Load
    print("Step 1: Loading rag.npz...")
    d = np.load(RAG_NPZ)
    x_raw     = d["x"]
    y_raw     = d["y"]
    rag_class = d["rag_class"]
    is_fail   = d["is_fail"]
    inst_id   = d["inst_id"]
    N = len(x_raw)
    print(f"  Loaded {N:,} windows, shape={x_raw.shape}")

    # Step 2: Stats embeddings
    print("\nStep 2: Extracting statistical features...")
    embeddings = extract_stats(x_raw)                         # (N, 50)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = (embeddings / norms).astype(np.float32)
    print(f"  Embeddings shape: {embeddings.shape}  "
          f"range [{embeddings.min():.3f}, {embeddings.max():.3f}]")

    # Step 3: FAISS
    print("\nStep 3: Building FAISS IndexFlatL2...")
    index = faiss.IndexFlatL2(EMBD_DIM)
    index.add(embeddings)
    print(f"  ntotal: {index.ntotal:,}  dim: {EMBD_DIM}")

    # Step 4: Save
    print("\nStep 4: Saving RAG DB...")
    OUT_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    np.save(OUT_DIR / "embeddings.npy", embeddings)
    np.savez_compressed(
        OUT_DIR / "metadata.npz",
        y=y_raw, rag_class=rag_class, is_fail=is_fail, inst_id=inst_id,
    )

    db_info = {
        "n_windows":        int(N),
        "embd_dim":         EMBD_DIM,
        "top_k_default":    TOP_K,
        "features":         FEATURE_NAMES,
        "cpu_pressure_idx": CPU_PRESSURE_IDX,
        "mem_pressure_idx": MEM_PRESSURE_IDX,
        "cpu_burst_idx":    CPU_BURST_IDX,
        "rag_class_dist":   {str(c): int((rag_class == c).sum()) for c in range(5)},
        "fail_windows":     int(is_fail.sum()),
    }
    with open(OUT_DIR / "db_info.json", "w") as f:
        json.dump(db_info, f, indent=2)

    print(f"\n=== Done ===")
    print(json.dumps(db_info, indent=2))
