# Cloud Resource Time Series Forecasting via Time-LLM with Retrieval-Conditioned Prompt-as-Prefix

[[Slides (PDF, Korean)]](docs/slides.pdf)

<p align="center">
  <img src="docs/architecture.webp" width="85%" alt="Time-LLM with retrieval-conditioned Prompt-as-Prefix">
</p>

## Abstract

[Time-LLM(Jin et al., ICLR 2024)](https://github.com/KimMeen/Time-LLM)은 사전학습 언어 모델의 가중치를 고정한 채, 데이터셋 설명과 입력 통계로 구성된 Prompt-as-Prefix(PaP)를 시계열 입력 앞에 붙여 예측을 수행한다.
본 프로젝트는 클라우드 자원 사용량 예측을 위해, 입력 구간과 유사한 과거 패턴을 검색하여 PaP에 반영하도록 Time-LLM을 확장했다.
이를 위해 [Google Cluster Trace 2019](https://github.com/google/cluster-data)의 시계열 구간으로 FAISS 색인을 구축하고, 입력마다 검색한 유사 구간으로 PaP를 동적으로 구성한다.
GPT-2 backbone으로 수행한 소거 실험(A0-A6)에서 검색 기반 프롬프트는 test MAE를 0.063% 낮췄으며, 같은 형식의 고정 프롬프트는 오차를 높임을 확인했다.
BigQuery 데이터 추출부터 누수 없는 데이터 분할, FAISS 색인 구축, 학습까지 약 912만 행 규모의 트레이스 데이터를 다루는 전체 파이프라인을 구현했다.

[Time-LLM (Jin et al., ICLR 2024)](https://github.com/KimMeen/Time-LLM) forecasts time series with a frozen pretrained language model, prefixing the input with a Prompt-as-Prefix (PaP) that consists of a dataset description and input statistics.
This project extends Time-LLM to cloud resource forecasting by retrieving past patterns similar to the input window and adding them to the PaP.
To this end, a FAISS index is built over windows of [Google Cluster Trace 2019](https://github.com/google/cluster-data), and the PaP is constructed dynamically from the similar windows retrieved for each input.
In ablations A0-A6 with a GPT-2 backbone, the retrieval-conditioned prompt lowers test MAE by 0.063%, while a fixed prompt of the same format increases error.
The project implements the full pipeline on 9.1M rows of trace data, from BigQuery extraction and leakage-free splitting to FAISS index construction and training.

## Method

Starting from the official Time-LLM implementation (`time-llm-original/`), the project adds three components.

1. **Retrieval-conditioned PaP**\
   The retrieved summary is appended to the original prompt as `Historical context` (`models/TimeLLM.py`, `utils/rag.py`). Lines marked `+` are added by this project; line breaks are inserted for readability.
   ```diff
     <|start_prompt|>
     Dataset description: ...
     Task description: forecast the next 32 steps given the previous 32 steps information;
     Input statistics: min value ..., max value ..., median value ..., top 5 lags are : [...]
   + Historical context:
   +   dominant pattern is burst, failure rate 60% (3 of 5 similar patterns),
   +   Future outlook —
   +     cpu_usage:    mean 0.031  max 0.089,
   +     mem_usage:    mean 0.041  max 0.112,
   +     cpu_pressure: mean 0.412  max 0.891,
   +     mem_pressure: mean 0.203  max 0.445,
   +     cpu_burst:    max 1.834.
     <|<end_prompt>|>
   ```
   The retriever runs in three modes, `normal` (similarity search), `random`, and `fixed`, which separates the effect of retrieval from the effect of a longer prompt.
2. **Leakage-free, class-balanced retrieval pool**\
   Instances are assigned to the pool by hash, so pool instances never appear in the training, validation, or test splits. Each window is embedded by the mean, standard deviation, minimum, maximum, and linear trend of every metric (50 dimensions), L2-normalized, and indexed with `IndexFlatL2`. Windows are labeled with five pattern classes: `normal-stable`, `normal-rising`, `burst`, `failure-gradual`, and `failure-sudden`. All FAIL instances (about 2.1K) are kept, and FINISH instances are sampled at 1/4 during extraction (`pipeline/extract_data.sql`).
3. **Cloud trace integration**\
   A loader for pre-windowed cluster traces (`Dataset_GClusterInstance`), a domain prompt (`dataset/prompt_bank/GCluster.txt`), and four derived metrics are added: `cpu_pressure` and `mem_pressure` (usage divided by the requested amount) and `cpu_burst` and `mem_burst` (maximum minus mean usage). Validation and test MSE and MAE are logged for every metric at every epoch.

## Experimental Setup

| | |
|---|---|
| Data | Google Cluster Trace 2019, Cell a. `instance_usage` aggregated into 5-minute buckets, 9,117,645 rows. Instances whose last event is FAIL or FINISH, without restarts, with at least 64 steps |
| Metrics (10) | `cpu_usage`, `max_cpu_usage`, `mem_usage`, `max_mem_usage`, `page_cache_memory`, `assigned_mem`, `cpu_pressure`, `mem_pressure`, `cpu_burst`, `mem_burst` |
| Windows | input 32, horizon 32, stride 32. Train 86,712 / val 28,917 / test 28,181 / retrieval pool 47,525 (`pipeline/processed/split_summary.json`) |
| Backbone | `openai-community/gpt2`, first 3 layers, frozen |
| Training | batch size 8, learning rate 1e-4, up to 30 epochs, early stopping with patience 10, bf16, one RTX 3090 |
| Evaluation | MSE and MAE on the test split |

| ID | Setting | Prefix appended to the prompt |
|---|---|---|
| A0 | Baseline | none (original Time-LLM prompt) |
| A1 | RAG_PaP | summary of the top-5 nearest pool windows |
| A2 | Random_RAG | summary of 5 random pool windows |
| A3 | Fixed_Prompt | one fixed summary of the same format for every input |
| A4-A6 | RAG_top{10,20,100} | A1 with k = 10, 20, 100 |

## Main Results

| ID | Setting | Test MSE | Δ vs. A0 | Test MAE | Δ vs. A0 |
|---|---|---:|---:|---:|---:|
| A0 | Baseline | 0.18429 | | 0.05378 | |
| A1 | RAG_PaP (k = 5) | 0.18418 | -0.059% | 0.05375 | -0.063% |
| A2 | Random_RAG (k = 5) | 0.18436 | +0.036% | 0.05371 | -0.142% |
| A3 | Fixed_Prompt | 0.18492 | +0.343% | 0.05383 | +0.084% |
| A4 | RAG_top10 | 0.18430 | +0.006% | 0.05374 | -0.069% |
| A5 | RAG_top20 | 0.18484 | +0.297% | 0.05366 | -0.235% |
| A6 | RAG_top100 | 0.18390 | -0.212% | 0.05393 | +0.276% |

Change in test MAE relative to A0 for selected metrics:

| Metric | A1 RAG_PaP | A2 Random_RAG | A3 Fixed_Prompt |
|---|---:|---:|---:|
| `cpu_usage` | -0.39% | -0.24% | +0.00% |
| `mem_usage` | -0.67% | -0.13% | +0.38% |
| `max_mem_usage` | -0.61% | +0.02% | +0.34% |
| `mem_burst` | -1.42% | -1.07% | +0.95% |

Each test value is taken at the epoch with the lowest validation value of the same quantity: MSE at the lowest validation MSE, MAE at the lowest validation MAE, and per-metric MAE at the lowest validation MAE of that metric. All runs use a single seed (2021). Per-epoch logs of these runs are in `time-llm-modified/results/` (`A0_Baseline.csv` to `A6_RAG_top100.csv`).

- A fixed prefix of the same format (A3) does not lower error on any of the four metrics above, so the gain of A1 does not come from a longer prompt alone.
- Random retrieval (A2) also lowers overall MAE, by more than A1 does. The benefit of similarity search over random sampling is therefore not established.
- Larger k helps some metrics and hurts others. At k = 100 (A6), the MAE of `page_cache_memory` drops by 24.65% and that of `mem_burst` by 4.32%, while the MAE of `max_mem_usage` rises steadily with k (-0.61% at k = 5 to +0.37% at k = 100).
- All overall differences are below 0.35% and come from one seed. Their statistical significance is not assessed.

**Limitations**\
The data consists mostly of stable regimes, so the effect on abrupt load changes needs to be measured separately. Limited GPU time prevented scaling to the full trace and tuning hyperparameters. Generalization to other cells and to other traces such as Azure or Alibaba is untested.

## Getting Started

**Installation**

```bash
cd time-llm-modified
uv sync                                  # Python 3.12, versions pinned in uv.lock
```

**Data**\
Run `pipeline/extract_data.sql` on BigQuery after replacing `YOUR_BUCKET`, then download the exported `gcluster_*.csv` files into `pipeline/data/`.

```bash
cd pipeline
uv run python preprocess.py              # processed/{train,val,test,rag}.npz
uv run python build_rag_db.py            # rag_db/
uv run pytest tests
cd ..
```

**Training and evaluation**

```bash
uv run bash experiments/run_ablation.sh A1   # one of A0, A1, ..., A6
```

Metrics are written per epoch to `results/<setting>/metrics_<JOB_ID>.csv`; the reported runs are stored under shorter names (`results/A0_Baseline.csv`, ...). The environment variables `DATA_DIR`, `RAG_DB_DIR`, and `JOB_ID` override the default paths and run ID. GPT-2 weights are downloaded from Hugging Face on first use. The reported runs used a conda environment on a Slurm cluster. `uv.lock` describes an environment in which the code imports and the tests pass.

## Repository Structure

```text
.
├── LICENSE                         # Apache License 2.0
├── docs/                           # slides (PDF and images), architecture figure
├── time-llm-original/              # KimMeen/Time-LLM @ b13e881, unmodified
└── time-llm-modified/
    ├── CHANGES.md                  # file-level changes from time-llm-original
    ├── pyproject.toml, uv.lock     # uv environment (Python 3.12)
    ├── models/TimeLLM.py           # injects rag_contexts into the prompt
    ├── utils/rag.py                # RAGRetriever: statistics embedding + FAISS search
    ├── data_provider/              # Dataset_GClusterInstance
    ├── run_main.py                 # --use_rag, --rag_mode, --rag_top_k
    ├── experiments/run_ablation.sh # A0-A6
    ├── results/                    # per-epoch metrics of the reported runs (A0-A6)
    └── pipeline/
        ├── extract_data.sql        # BigQuery extraction
        ├── preprocess.py           # sliding windows, splits, retrieval pool
        ├── build_rag_db.py         # FAISS index over the retrieval pool
        ├── visualize_rag_patterns.py  # class samples and PCA of the retrieval pool
        ├── eda/                    # data overview, labeling analysis, forecast diagnosis
        └── tests/
```

## Slides

The slides below are rendered from `docs/slides.pdf` (Korean).

<details>
<summary>Show all slides</summary>

![Slide 1](docs/slides/slide-01.webp)
![Slide 2](docs/slides/slide-02.webp)
![Slide 3](docs/slides/slide-03.webp)
![Slide 4](docs/slides/slide-04.webp)
![Slide 5](docs/slides/slide-05.webp)
![Slide 6](docs/slides/slide-06.webp)
![Slide 7](docs/slides/slide-07.webp)
![Slide 8](docs/slides/slide-08.webp)
![Slide 9](docs/slides/slide-09.webp)
![Slide 10](docs/slides/slide-10.webp)
![Slide 11](docs/slides/slide-11.webp)
![Slide 12](docs/slides/slide-12.webp)
![Slide 13](docs/slides/slide-13.webp)
![Slide 14](docs/slides/slide-14.webp)
![Slide 15](docs/slides/slide-15.webp)
![Slide 16](docs/slides/slide-16.webp)
![Slide 17](docs/slides/slide-17.webp)
![Slide 18](docs/slides/slide-18.webp)
![Slide 19](docs/slides/slide-19.webp)
![Slide 20](docs/slides/slide-20.webp)
![Slide 21](docs/slides/slide-21.webp)
![Slide 22](docs/slides/slide-22.webp)
![Slide 23](docs/slides/slide-23.webp)
![Slide 24](docs/slides/slide-24.webp)
![Slide 25](docs/slides/slide-25.webp)
![Slide 26](docs/slides/slide-26.webp)
![Slide 27](docs/slides/slide-27.webp)
![Slide 28](docs/slides/slide-28.webp)
![Slide 29](docs/slides/slide-29.webp)
![Slide 30](docs/slides/slide-30.webp)
![Slide 31](docs/slides/slide-31.webp)
![Slide 32](docs/slides/slide-32.webp)
![Slide 33](docs/slides/slide-33.webp)
![Slide 34](docs/slides/slide-34.webp)

</details>

## Acknowledgements

This code builds on [Time-LLM](https://github.com/KimMeen/Time-LLM). The data is ClusterData2019 from [google/cluster-data](https://github.com/google/cluster-data). Raw data, preprocessed windows (`.npz`), and the retrieval index are not redistributed.

```bibtex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## License

This project is released under the Apache License 2.0, the same license as Time-LLM.
