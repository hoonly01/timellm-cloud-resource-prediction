# 원본 대비 변경 사항

이 디렉토리는 [KimMeen/Time-LLM](https://github.com/KimMeen/Time-LLM) `b13e881`(= [`../time-llm-original/`](../time-llm-original/))을 수정한 버전입니다.
Apache License 2.0 §4(b)에 따라 수정한 파일을 아래에 밝힙니다. 전체 비교는 레포 루트에서 다음 명령으로 볼 수 있습니다.

```bash
diff -ru time-llm-original time-llm-modified
```

## 수정한 upstream 파일

| 파일 | 변경 내용 |
|---|---|
| `data_provider/data_loader.py` | `Dataset_GClusterInstance` 추가. `pipeline/preprocess.py`가 만든 `train/val/test.npz` 슬라이딩 윈도우를 읽음 |
| `data_provider/data_factory.py` | `GCluster` 데이터셋 등록 |
| `models/TimeLLM.py` | `forward()`/`forecast()`에 `rag_contexts` 인자 추가. 검색 결과를 `Historical context: ...`로 프롬프트 끝에 붙임. 원본 프롬프트의 입력 추세(upward/downward) 문구는 제거 |
| `utils/tools.py` | `vali()`가 `rag_retriever`를 받아 평가 때도 RAG 문맥을 넣음. feature별 MAE/MSE 반환. `np.Inf` → `np.inf` (NumPy 2.0 호환) |
| `run_main.py` | `--use_rag`, `--rag_db_dir`, `--rag_top_k`, `--rag_mode {normal,random,fixed}`, `--job_id` 추가. epoch마다 `results/<setting>/metrics_<job_id>.csv`에 val/test 지표 기록. DeepSpeed 플러그인 제거. 학습 후 best-epoch checkpoint를 지우지 않고 남김 |

## 새로 추가한 파일

| 경로 | 내용 |
|---|---|
| `utils/rag.py` | `RAGRetriever`: 윈도우 → 50차원 통계 임베딩 → FAISS 검색 → 프롬프트 문맥 문자열 |
| `dataset/prompt_bank/GCluster.txt` | Google Cluster Trace 데이터셋 설명 프롬프트 |
| `pipeline/` | BigQuery 추출 SQL, 전처리, RAG DB 구축, 시각화, EDA, 테스트 |
| `experiments/run_ablation.sh` | A0~A6 실험 실행 |
| `results/` | 최종 보고서에 사용한 A0~A6 epoch별 지표 |
| `pyproject.toml`, `uv.lock`, `.python-version` | 로컬 개발 환경(uv) |
