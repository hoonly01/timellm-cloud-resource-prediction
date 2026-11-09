"""
전처리 파이프라인 — Google Cluster Trace v3 (Cell a)

출력:
  processed/train.npz  — {'x':(N,32,10), 'y':(N,32,10), 'label':(N,), 'inst_id':(N,2)}
  processed/val.npz
  processed/test.npz
  processed/rag.npz    — {'x':..., 'y':..., 'rag_class':(N,), 'is_fail':(N,), 'inst_id':(N,2)}
  processed/split_summary.json

정규화: 없음 (raw values 저장).
  - Time-LLM 학습: RevIN (모델 내부) 사용
  - 이미 0~1 범위 normalized된 Google 데이터라 PaP 텍스트도 LLM-interpretable

label 컬럼 용도:
  - 학습 loss: 미사용 (MAE forecasting only)
  - train split의 FAIL:FINISH = 1:5 다운샘플링에만 사용

피처 순서 (10개):
  0  cpu_usage
  1  max_cpu_usage
  2  mem_usage
  3  max_mem_usage
  4  page_cache_memory
  5  assigned_mem
  6  cpu_pressure   = cpu_usage / requested_cpu
  7  mem_pressure   = mem_usage / requested_mem
  8  cpu_burst      = max_cpu_usage - cpu_usage
  9  mem_burst      = max_mem_usage - mem_usage
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 상수 ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data")
OUT_DIR   = Path("processed")
SEQ_LEN   = 32
PRED_LEN  = 32
STRIDE    = 32
BURST_THR = 0.10          # cpu_burst > 0.10 → burst class
SLOPE_RISING_THR = 4e-7   # FINISH slope > this → normal-rising

FEATURES = [
    "cpu_usage", "max_cpu_usage",
    "mem_usage", "max_mem_usage",
    "page_cache_memory", "assigned_mem",
    "cpu_pressure", "mem_pressure",
    "cpu_burst", "mem_burst",
]

RAG_CLASS_MAP = {
    0: "normal-stable",
    1: "normal-rising",
    2: "burst",
    3: "failure-gradual",
    4: "failure-sudden",
}

# ── 유틸 ────────────────────────────────────────────────────────────────────
def instance_hash_mod10(cid: int, idx: int) -> int:
    key = f"{cid}_{idx}".encode()
    return int(hashlib.md5(key).hexdigest(), 16) % 10


def count_windows(n: int) -> int:
    if n < SEQ_LEN + PRED_LEN:
        return 0
    return (n - SEQ_LEN - PRED_LEN) // STRIDE + 1


# ── Step 1: 로드 & 정제 ──────────────────────────────────────────────────────
def load_and_clean() -> pd.DataFrame:
    print("Step 1: Loading data...")
    files = sorted(DATA_DIR.glob("gcluster_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"  Loaded {len(df):,} rows")

    # max_mem_usage NULL → fillna(mem_usage)
    df["max_mem_usage"] = df["max_mem_usage"].fillna(df["mem_usage"])

    # 음수 clip
    for col in FEATURES[:6]:  # 파생 피처 생성 전 원본만
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    # max_cpu_usage 이상치 clip
    df["max_cpu_usage"] = df["max_cpu_usage"].clip(upper=2.0)

    print(f"  After clean: {df.isnull().sum().sum()} NULLs remaining")
    return df


# ── Step 2: 파생 피처 ────────────────────────────────────────────────────────
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Step 2: Adding derived features...")
    df["cpu_pressure"] = (
        df["cpu_usage"] / df["requested_cpu"].replace(0, np.nan)
    ).fillna(0.0).clip(upper=50.0)

    df["mem_pressure"] = (
        df["mem_usage"] / df["requested_mem"].replace(0, np.nan)
    ).fillna(0.0).clip(upper=50.0)

    df["cpu_burst"] = (df["max_cpu_usage"] - df["cpu_usage"]).clip(lower=0.0)
    df["mem_burst"] = (df["max_mem_usage"] - df["mem_usage"]).clip(lower=0.0)

    assert all(f in df.columns for f in FEATURES), "Missing feature columns"
    print(f"  Features: {FEATURES}")
    return df


# ── Step 3: Instance-level stratified split ──────────────────────────────────
def make_splits(df: pd.DataFrame) -> dict:
    print("Step 3: Stratified instance split...")

    inst = (
        df[["collection_id", "instance_index", "final_event_type"]]
        .drop_duplicates()
        .copy()
    )
    inst["hash_mod10"] = inst.apply(
        lambda r: instance_hash_mod10(r["collection_id"], r["instance_index"]), axis=1
    )

    # RAG: hash < 2 (클래스별 독립)
    inst["pool"] = "model"
    inst.loc[inst["hash_mod10"] < 2, "pool"] = "rag"

    # Model pool → Train/Val/Test (70/15/15, 클래스별 독립)
    model = inst[inst["pool"] == "model"].copy()
    model["split"] = "train"

    for etype in [5, 6]:
        sub_idx = model[model["final_event_type"] == etype].index
        n = len(sub_idx)
        n_val  = max(1, int(n * 0.15))
        n_test = max(1, int(n * 0.15))
        rng = np.random.default_rng(seed=42 + etype)
        shuffled = rng.permutation(sub_idx)
        model.loc[shuffled[:n_test],            "split"] = "test"
        model.loc[shuffled[n_test:n_test+n_val],"split"] = "val"

    inst = inst.merge(
        model[["collection_id", "instance_index", "split"]],
        on=["collection_id", "instance_index"], how="left"
    )
    inst["split"] = inst["split"].fillna("rag")

    # 요약 출력
    summary = inst.groupby(["split", "final_event_type"]).size().unstack(fill_value=0)
    summary.columns = ["FAIL", "FINISH"]
    print(summary.to_string())

    return inst.set_index(["collection_id", "instance_index"]).to_dict("index")


# ── Step 4: 인스턴스별 RAG 5클래스 레이블 ────────────────────────────────────
def compute_rag_class(grp: pd.Series, final_event_type: int) -> int:
    """인스턴스 전체 시계열 → RAG 5클래스 정수 반환"""
    cpu_burst_max = grp["cpu_burst"].max()

    if final_event_type == 5:  # FAIL
        if cpu_burst_max > BURST_THR:
            return 4  # failure-sudden
        return 3  # failure-gradual
    else:  # FINISH
        if cpu_burst_max > BURST_THR:
            return 2  # burst
        n = len(grp)
        if n >= 4:
            slope = np.polyfit(np.arange(n), grp["cpu_usage"].values, 1)[0]
            if slope > SLOPE_RISING_THR:
                return 1  # normal-rising
        return 0  # normal-stable


# ── Step 5 & 6: 슬라이딩 윈도우 생성 ────────────────────────────────────────
def generate_windows(df: pd.DataFrame, split_map: dict):
    print("Step 4-6: Generating sliding windows...")

    buckets = {"train": [], "val": [], "test": [], "rag": []}

    grouped = df.sort_values("time_step_sec").groupby(
        ["collection_id", "instance_index", "final_event_type"]
    )

    for (cid, idx, etype), grp in grouped:
        grp = grp.sort_values("time_step_sec")
        info = split_map.get((cid, idx))
        if info is None:
            continue
        split = info["split"]
        feat_arr = grp[FEATURES].values.astype(np.float32)  # (T, 10)
        n = len(feat_arr)
        nw = count_windows(n)
        if nw == 0:
            continue

        rag_cls = compute_rag_class(grp, etype)
        binary_label = 1 if etype == 5 else 0

        for w in range(nw):
            start = w * STRIDE
            end   = start + SEQ_LEN + PRED_LEN
            window = feat_arr[start:end]         # (64, 10)
            x = window[:SEQ_LEN]                 # (32, 10)
            y = window[SEQ_LEN:]                 # (32, 10)

            buckets[split].append({
                "x": x, "y": y,
                "label": binary_label,
                "rag_class": rag_cls,
                "inst_id": np.array([cid, idx], dtype=np.int64),
            })

    return buckets


# ── Train FINISH 다운샘플링 (1:5) ────────────────────────────────────────────
def downsample_train(train_items: list) -> list:
    fail_items   = [it for it in train_items if it["label"] == 1]
    finish_items = [it for it in train_items if it["label"] == 0]
    n_fail   = len(fail_items)
    n_target = n_fail * 5
    rng = np.random.default_rng(seed=0)
    if len(finish_items) > n_target:
        sel = rng.choice(len(finish_items), size=n_target, replace=False)
        finish_items = [finish_items[i] for i in sel]
    result = fail_items + finish_items
    rng.shuffle(result)
    print(f"  Train after downsample: FAIL={n_fail:,}  FINISH={len(finish_items):,}  total={len(result):,}")
    return result


# ── 저장 ────────────────────────────────────────────────────────────────────
def save_npz(items: list, path: Path, is_rag: bool = False):
    if not items:
        print(f"  [WARN] No items for {path.name}, skipping")
        return
    xs   = np.stack([it["x"]       for it in items])
    ys   = np.stack([it["y"]       for it in items])
    ids  = np.stack([it["inst_id"] for it in items])
    lbl = np.array([it["label"] for it in items], dtype=np.int8)
    if is_rag:
        cls     = np.array([it["rag_class"] for it in items], dtype=np.int8)
        is_fail = (cls >= 3).astype(np.int8)  # failure-gradual(3) or failure-sudden(4)
        np.savez_compressed(path, x=xs, y=ys, rag_class=cls, is_fail=is_fail,
                            label=lbl, inst_id=ids)
        print(f"  Saved {path.name}: {xs.shape}  is_fail={is_fail.sum():,}")
    else:
        np.savez_compressed(path, x=xs, y=ys, label=lbl, inst_id=ids)
        print(f"  Saved {path.name}: {xs.shape}  FAIL={lbl.sum():,}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    argparse.ArgumentParser().parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # 1. 로드 & 정제
    df = load_and_clean()

    # 2. 파생 피처
    df = add_derived_features(df)

    # 3. Split map 생성
    split_map = make_splits(df)

    # 4~6. 윈도우 생성
    buckets = generate_windows(df, split_map)

    train_items  = buckets["train"]
    val_items    = buckets["val"]
    test_items   = buckets["test"]
    rag_items    = buckets["rag"]

    print(f"\nWindow counts (before downsample):")
    for name, items in [("train", train_items), ("val", val_items),
                        ("test", test_items), ("rag", rag_items)]:
        n_fail   = sum(it["label"] == 1 for it in items)
        n_finish = sum(it["label"] == 0 for it in items)
        print(f"  {name:<6}: total={len(items):>7,}  FAIL={n_fail:>6,}  FINISH={n_finish:>7,}")

    if not train_items:
        raise ValueError("No train windows generated. Check the input CSVs.")

    # Train 다운샘플링
    print("\nStep 5: Train downsampling (FAIL:FINISH = 1:5)...")
    train_items = downsample_train(train_items)

    # 저장
    print("\nStep 7: Saving npz files...")
    save_npz(train_items, OUT_DIR / "train.npz", is_rag=False)
    save_npz(val_items,   OUT_DIR / "val.npz",   is_rag=False)
    save_npz(test_items,  OUT_DIR / "test.npz",  is_rag=False)
    save_npz(rag_items,   OUT_DIR / "rag.npz",   is_rag=True)

    # Split 요약
    rag_class_dist = {}
    for cls_int, cls_name in RAG_CLASS_MAP.items():
        cnt = sum(1 for it in rag_items if it["rag_class"] == cls_int)
        rag_class_dist[cls_name] = cnt

    summary = {
        "seq_len": SEQ_LEN,
        "pred_len": PRED_LEN,
        "stride": STRIDE,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "burst_threshold": BURST_THR,
        "slope_rising_threshold": SLOPE_RISING_THR,
        "windows": {
            "train": len(train_items),
            "val":   len(val_items),
            "test":  len(test_items),
            "rag":   len(rag_items),
        },
        "rag_class_distribution": rag_class_dist,
    }
    with open(OUT_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Done ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
