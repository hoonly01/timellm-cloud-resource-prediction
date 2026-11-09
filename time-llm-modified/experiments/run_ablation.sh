#!/usr/bin/env bash
# Run one ablation experiment (A0-A6) with the settings used in the final report.
#
#   bash experiments/run_ablation.sh A1
#   sbatch --gres=gpu:1 experiments/run_ablation.sh A1   # on a Slurm cluster
#
# Environment variables:
#   DATA_DIR    train/val/test.npz written by pipeline/preprocess.py  (default: pipeline/processed)
#   RAG_DB_DIR  index written by pipeline/build_rag_db.py              (default: pipeline/rag_db)
#   JOB_ID      suffix of results/<setting>/metrics_<JOB_ID>.csv        (default: $SLURM_JOB_ID or a timestamp)
set -euo pipefail
cd "$(dirname "$0")/.."  # time-llm-modified/

DATA_DIR=${DATA_DIR:-pipeline/processed}
RAG_DB_DIR=${RAG_DB_DIR:-pipeline/rag_db}
JOB_ID=${JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}}

exp=${1:-}
case "$exp" in
  A0) des=A0_Baseline;     rag=(--use_rag 0) ;;
  A1) des=A1_RAG_PaP;      rag=(--use_rag 1 --rag_db_dir "$RAG_DB_DIR" --rag_top_k 5   --rag_mode normal) ;;
  A2) des=A2_Random_RAG;   rag=(--use_rag 1 --rag_db_dir "$RAG_DB_DIR" --rag_top_k 5   --rag_mode random) ;;
  A3) des=A3_Fixed_Prompt; rag=(--use_rag 1 --rag_db_dir "$RAG_DB_DIR" --rag_top_k 5   --rag_mode fixed) ;;
  A4) des=A4_RAG_top10;    rag=(--use_rag 1 --rag_db_dir "$RAG_DB_DIR" --rag_top_k 10  --rag_mode normal) ;;
  A5) des=A5_RAG_top20;    rag=(--use_rag 1 --rag_db_dir "$RAG_DB_DIR" --rag_top_k 20  --rag_mode normal) ;;
  A6) des=A6_RAG_top100;   rag=(--use_rag 1 --rag_db_dir "$RAG_DB_DIR" --rag_top_k 100 --rag_mode normal) ;;
  *) echo "usage: $0 {A0|A1|A2|A3|A4|A5|A6}" >&2; exit 1 ;;
esac

accelerate launch \
  --mixed_precision bf16 \
  --num_processes 1 \
  run_main.py \
  --task_name        long_term_forecast \
  --is_training      1 \
  --root_path        "$DATA_DIR" \
  --data_path        dummy \
  --data             GCluster \
  --model_id         "GCluster_TimeLLM_$exp" \
  --model            TimeLLM \
  --model_comment    "TimeLLM-GCluster-$exp" \
  --features         M \
  --seq_len          32 \
  --label_len        0 \
  --pred_len         32 \
  --enc_in           10 \
  --dec_in           10 \
  --c_out            10 \
  --d_model          32 \
  --d_ff             128 \
  --llm_model        GPT2 \
  --llm_dim          768 \
  --llm_layers       3 \
  --prompt_domain    1 \
  --batch_size       8 \
  --eval_batch_size  8 \
  --learning_rate    0.0001 \
  --num_workers      4 \
  --itr              1 \
  --train_epochs     30 \
  "${rag[@]}" \
  --des              "$des" \
  --job_id           "$JOB_ID"
