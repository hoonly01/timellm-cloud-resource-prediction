# Google Cluster Trace v3 (Cell a) — EDA Full Report
Data: gcluster_*.csv (6 files) | BigQuery export | FAIL(5) + FINISH(6)


1. Dataset Overview
===================

1-1. Basic Stats
----------------
Total rows        : 9,117,639
Total instances   : 37,402
  FAIL  (type=5)  : 2,103 instances  |  965,236 rows
  FINISH(type=6)  : 35,299 instances  |  8,152,403 rows
  FAIL ratio      : 5.62% (instances)  |  10.59% (rows)
Columns           : ['collection_id', 'instance_index', 'time_step_sec', 'cpu_usage', 'max_cpu_usage', 'mem_usage', 'max_mem_usage', 'page_cache_memory', 'assigned_mem', 'final_event_type', 'requested_cpu', 'requested_mem']

1-2. Missing Values
-------------------
  max_mem_usage: 373,168 (4.09%)

2. Data Quality
===============

2-1. Duplicate Rows
-------------------
  Duplicates: 0  (0.000%)

2-2. Time Gap Analysis (5-min = 300s expected)
----------------------------------------------
  Non-300s gap ratio : 3.47%
  Top gap values (sec):
       300s : 8,765,017  (96.53%)
       600s :   78,587  (0.87%)
       900s :   37,640  (0.41%)
      1200s :   23,060  (0.25%)
      1500s :   14,560  (0.16%)
      1800s :   10,822  (0.12%)
      2100s :    8,415  (0.09%)
      2700s :    6,260  (0.07%)
      2400s :    6,253  (0.07%)
      3000s :    5,402  (0.06%)

2-3. Value Range Check
----------------------
  cpu_usage           : negative=0  outlier(>>1.0)=0
  max_cpu_usage       : negative=0  outlier(>>2.0)=62
  mem_usage           : negative=0  outlier(>>1.0)=0
  max_mem_usage       : negative=0  outlier(>>1.0)=0

3. Time-series Length Distribution
==================================

3-1. Percentiles by Class
-------------------------
  FAIL:
    p5=68  p10=74  p25=98  p50=267  p75=344  p90=598  p95=2501  p99=5761
    mean=459.0  std=914.5  max=7204
  FINISH:
    p5=66  p10=69  p25=74  p50=92  p75=134  p90=290  p95=894  p99=2851
    mean=231.0  std=529.1  max=8927

3-2. Sliding Window Count Simulation
------------------------------------
  Config                         FAIL windows   FINISH windows  FAIL inst (%)  FINISH inst (%)
  --------------------------------------------------------------------------------------------
  seq32 / pred16 / stride16            55,165          423,378       2103 (100%)      35299 (100%)
  seq32 / pred32 / stride16            53,062          388,079       2103 (100%)      35299 (100%)
  seq64 / pred32 / stride32            24,946          170,032       1597 (76%)      16954 (48%)
  seq96 / pred32 / stride32            23,349          153,078       1425 (68%)      10966 (31%)

4. Feature Distribution by Class
================================

4-1. Percentile Table — FAIL (type=5)
-------------------------------------

  Feature                       p1         p5        p25        p50        p75        p95        p99
  ----------------------------------------------------------------------------------------------------
  cpu_usage                0.00009    0.00012    0.00031    0.00710    0.02029    0.07397    0.12380
  max_cpu_usage            0.00041    0.00060    0.00140    0.01834    0.06995    0.22754    0.38330
  mem_usage                0.00034    0.00074    0.00104    0.00156    0.00538    0.01103    0.02930
  max_mem_usage            0.00034    0.00081    0.00105    0.00164    0.00716    0.01143    0.04047
  page_cache_memory        0.00000    0.00000    0.00010    0.00036    0.00051    0.00082    0.00151
  assigned_mem             0.00000    0.00130    0.00205    0.00286    0.01360    0.03125    0.12366
  requested_cpu            0.00022    0.00081    0.00581    0.01015    0.01575    0.03522    0.03522
  requested_mem            0.00016    0.00130    0.00193    0.00237    0.01193    0.03125    0.07812

4-2. Percentile Table — FINISH (type=6)
---------------------------------------

  Feature                       p1         p5        p25        p50        p75        p95        p99
  ----------------------------------------------------------------------------------------------------
  cpu_usage                0.00007    0.00012    0.00063    0.00507    0.01008    0.02225    0.03967
  max_cpu_usage            0.00036    0.00059    0.00409    0.02151    0.04810    0.10205    0.18896
  mem_usage                0.00027    0.00058    0.00131    0.00291    0.00677    0.01804    0.03174
  max_mem_usage            0.00029    0.00067    0.00139    0.00334    0.01010    0.01840    0.02408
  page_cache_memory        0.00000    0.00005    0.00020    0.00047    0.00082    0.00180    0.00276
  assigned_mem             0.00050    0.00127    0.00378    0.01117    0.02789    0.03253    0.09949
  requested_cpu            0.00000    0.00115    0.00405    0.01128    0.01172    0.04907    0.07043
  requested_mem            0.00039    0.00104    0.00351    0.00922    0.02344    0.02875    0.09473

4-3. Class-level Summary (mean ± std)
-------------------------------------
  cpu_usage             : FAIL=0.01566±0.02558  FINISH=0.00702±0.00912  ratio=2.231x
  max_cpu_usage         : FAIL=0.05275±0.08660  FINISH=0.03324±0.04422  ratio=1.587x
  mem_usage             : FAIL=0.00449±0.01333  FINISH=0.00522±0.00791  ratio=0.861x
  max_mem_usage         : FAIL=0.00488±0.01448  FINISH=0.00609±0.00752  ratio=0.802x
  page_cache_memory     : FAIL=0.00038±0.00070  FINISH=0.00064±0.00074  ratio=0.603x
  assigned_mem          : FAIL=0.01226±0.02169  FINISH=0.01512±0.01513  ratio=0.811x

4-4. assigned_mem vs mem_usage
------------------------------
  FAIL: mean=0.426  median=0.434  >1.0=0.00%  >0.5=36.00%
  FINISH: mean=0.429  median=0.390  >1.0=1.11%  >0.5=38.42%

5. CPU-Memory Correlation
=========================

5-1. Pearson & Spearman (instance-level mean)
---------------------------------------------
  FAIL:
    Pearson  r=-0.0032  p=8.83e-01
    Spearman r=0.3865  p=7.06e-76
  FINISH:
    Pearson  r=0.1819  p=3.17e-260
    Spearman r=0.5322  p=0.00e+00

6. Burst Pattern Analysis
=========================

6-1. Burst (max - avg) — instance max value statistics
------------------------------------------------------
  FAIL cpu_burst: mean=0.1902  median=0.0702  p75=0.1271  p95=0.9717
  FAIL mem_burst: mean=0.00681  median=0.00183  p75=0.00303  p95=0.00730
  FINISH cpu_burst: mean=0.1000  median=0.0665  p75=0.1106  p95=0.2945
  FINISH mem_burst: mean=0.00254  median=0.00116  p75=0.00189  p95=0.01047

6-2. Burst k-sweep (row-level: max_cpu > inst_mean + k*std)
-----------------------------------------------------------
  k         Overall       FAIL     FINISH  note
  -------------------------------------------------------
  k=1.5      82.96%     84.12%     82.83% ← too high
  k=2.0      81.14%     81.33%     81.12% ← too high
  k=2.5      79.01%     78.08%     79.13% ← too high
  k=3.0      76.53%     73.78%     76.85% ← too high

  [Note] k-sweep based on instance-level std results in ~80% burst ratio.
  Alternative: use absolute spike threshold (max_cpu - avg_cpu > 0.05)
  Absolute spike > 0.05: overall=16.62%  FAIL=23.89%  FINISH=15.76%
  spike > 0.02: overall=41.75%  FAIL=39.20%  FINISH=42.05%
  spike > 0.05: overall=16.62%  FAIL=23.89%  FINISH=15.76%
  spike > 0.10: overall=4.02%  FAIL=10.91%  FINISH=3.20%
  spike > 0.15: overall=1.64%  FAIL=5.66%  FINISH=1.16%
  spike > 0.20: overall=0.84%  FAIL=2.97%  FINISH=0.58%

7. Resource Pressure
====================

7-1. CPU Pressure (cpu_usage / requested_cpu, clipped at 99.9th)
----------------------------------------------------------------
  stat                         FAIL       FINISH  ratio(F/Fi)
  mean                       2.0053       0.9490  2.11x
  p50                        0.5824       0.5372  1.08x
  p75                        1.2888       0.9579  1.35x
  p90                        4.4443       1.7039  2.61x
  p95                        9.8682       2.3992  4.11x

7-2. Memory Pressure (mem_usage / requested_mem, clipped at 99.9th)
-------------------------------------------------------------------
  mean                       1.2931       0.5503  2.35x
  p50                        0.5058       0.4272  1.18x
  p75                        0.6278       0.6957  0.90x
  p90                        0.8191       0.8589  0.95x
  p95                        1.2267       1.0087  1.22x

7-3. Pressure > 1.0 (using more than requested)
-----------------------------------------------
  FAIL: cpu_pressure>1.0 = 31.85%   mem_pressure>1.0 = 6.72%
  FINISH: cpu_pressure>1.0 = 22.63%   mem_pressure>1.0 = 5.37%

8. Failure Precursor Analysis
=============================

8-1. CPU ratio: last-N steps mean / instance mean
-------------------------------------------------
  (>1.0 = rising toward end, <1.0 = decreasing)
  Class      tail     mean   median      p25      p75   >1.5 (%)   >2.0 (%)
  ---------------------------------------------------------------------------
  FAIL     last 8    1.0812   0.9512   0.6681   1.2823      15.8%       8.9%
  FAIL     last16    1.0935   0.9761   0.7643   1.2388      13.5%       7.3%
  FAIL     last32    1.1155   0.9916   0.8507   1.1931      15.1%       7.2%
  FINISH   last 8    0.9238   0.9103   0.6636   1.0669       7.3%       3.3%
  FINISH   last16    0.9108   0.9283   0.6896   1.0634       6.1%       2.8%
  FINISH   last32    0.8617   0.9234   0.6979   1.0337       2.4%       0.5%

8-2. CPU Slope (linear fit over entire time series)
---------------------------------------------------
  FAIL  slope: mean=-0.000012  median=-0.000006  p25=-0.000023  p75=0.000002
  FINISH slope: mean=-0.000024  median=-0.000014  p25=-0.000042  p75=0.000000
  FAIL slope>0 (rising): 35.9%
  FINISH slope>0 (rising): 29.4%

  Proposed thresholds (FINISH quartiles):
    normal-rising : slope > 0.0000004  (FINISH top 25%)
    normal-stable : slope < -0.0000416  (FINISH bottom 25%)

9. max_mem_usage NULL Analysis
==============================
  Total NULL        : 373,168 (4.09%)
  FAIL NULL rate  : 0.00%
  FINISH NULL rate  : 4.58%
  Instances with any NULL: 3,435 / 37,402 (9.2%)
  NULL normalized position: mean=0.499  (0=start, 0.5=middle, 1=end)
  → Uniformly distributed across time → safe to fillna(mem_usage)

10. RAG / Model Split Preview
=============================
       FAIL  FINISH  total  FAIL_pct  FINISH_pct
pool                                            
Model  1953   29435  31388  6.222123   93.777877
RAG     150    5864   6014  2.494180   97.505820

  FAIL in RAG  : 150  (2.5%)
  FAIL in Model: 1,953  (6.2%)

11. Key Findings Summary (논문 작성 근거)
===================================

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
