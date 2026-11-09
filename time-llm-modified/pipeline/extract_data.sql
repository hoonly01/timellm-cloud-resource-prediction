-- BigQuery: export the raw CSVs consumed by preprocess.py (pipeline/data/gcluster_*.csv).
-- Replace YOUR_BUCKET with your own GCS bucket, then download the CSVs into pipeline/data/.

EXPORT DATA
  OPTIONS (
    uri = 'gs://YOUR_BUCKET/gcluster_*.csv',
    format = 'CSV',
    overwrite = true,
    header = true
  )
AS (

-- ============================================================
-- Google Cluster Trace v3 (Cell a) 추출 쿼리
-- 
-- 목적: RAG 벡터 DB + 학습 데이터셋 공용 원본 CSV 생성
--
-- 포함: FINISH(6), FAIL(5)
-- 제외: EVICT(4) - 외부 스케줄러 결정, 리소스 패턴과 무관
-- 제외: KILL(7), 진행중 상태, LOST(8), 재시작 인스턴스
--
-- 필터: row >= 64 (seq_len + pred_len 최대 실험 커버)
-- 샘플링: FAIL 전수(2.1K) + FINISH 1/4(~35K)
-- ============================================================

WITH

-- ① 재시작 인스턴스 식별
--    FAIL/EVICT 이후 re-SCHEDULE 발생 → 레이블 모호성으로 제외
restart_instances AS (
  SELECT DISTINCT collection_id, instance_index
  FROM (
    SELECT
      collection_id,
      instance_index,
      type,
      LAG(type) OVER (
        PARTITION BY collection_id, instance_index
        ORDER BY time
      ) AS prev_type
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
  )
  WHERE type = 3            -- SCHEDULE
    AND prev_type IN (4, 5) -- 직전이 EVICT or FAIL
),

-- ② 인스턴스별 최종 이벤트 타입 추출
--    전체 이벤트 중 마지막이 FINISH(6) 또는 FAIL(5)인 것만
--    EVICT(4) 제외: 외부 스케줄러 결정, 리소스 패턴으로 예측 불가
target_events AS (
  SELECT
    collection_id,
    instance_index,
    final_event_type
  FROM (
    SELECT
      collection_id,
      instance_index,
      ARRAY_AGG(type ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS final_event_type
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
    GROUP BY 1, 2
  )
  WHERE final_event_type IN (5, 6)  -- FAIL + FINISH만
),

-- ③ row 수 기준 필터링
--    5분 버킷 기준 64개 이상 (≈ 5시간 20분+)
--    seq_len(32) + pred_len(32) 최대 실험 범위 커버
long_running_instances AS (
  SELECT
    collection_id,
    instance_index
  FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
  GROUP BY 1, 2
  HAVING COUNT(DISTINCT FLOOR(start_time / 300000000)) >= 64
),

-- ④ 최종 대상 인스턴스 확정
--    조건: row 충족 + 유효 이벤트(FAIL/FINISH) + 재시작 없음
filtered_instances AS (
  SELECT
    t.collection_id,
    t.instance_index,
    t.final_event_type
  FROM target_events t
  JOIN long_running_instances l USING (collection_id, instance_index)
  WHERE NOT EXISTS (
    SELECT 1
    FROM restart_instances r
    WHERE r.collection_id = t.collection_id
      AND r.instance_index = t.instance_index
  )
),

-- ⑤ 클래스별 샘플링
--    FAIL: 전수 포함 (~2.1K)
--    FINISH: 1/4 샘플링 (~35K) - Val/Test 원분포 확보용
sampled_instances AS (
  -- FAIL 전수
  SELECT collection_id, instance_index, final_event_type
  FROM filtered_instances
  WHERE final_event_type = 5

  UNION ALL

  -- FINISH 1/4 샘플링
  SELECT collection_id, instance_index, final_event_type
  FROM filtered_instances
  WHERE final_event_type = 6
    AND ABS(MOD(FARM_FINGERPRINT(
      CONCAT(CAST(collection_id AS STRING), '_',
             CAST(instance_index AS STRING))
    ), 4)) = 0
),

-- ⑥ 인스턴스별 최초 리소스 요청량 추출
--    첫 SCHEDULE(3) 이벤트 기준
--    Python에서 resource_pressure = cpu_usage / requested_cpu 계산용
resource_requests AS (
  SELECT
    collection_id,
    instance_index,
    resource_request.cpus   AS requested_cpu,
    resource_request.memory AS requested_mem
  FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
  WHERE type = 3  -- SCHEDULE
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY collection_id, instance_index
    ORDER BY time ASC
  ) = 1
),

-- ⑦ 5분 버킷 단위 리소스 사용량 집계
--    시간 가중 평균으로 합산
resampled_usage AS (
  SELECT
    u.collection_id,
    u.instance_index,
    FLOOR(u.start_time / 300000000) * 300 AS time_step_sec,

    -- CPU 평균 (시간 가중)
    SAFE_DIVIDE(
      SUM(u.average_usage.cpus * (u.end_time - u.start_time)),
      SUM(u.end_time - u.start_time)
    ) AS cpu_usage,

    -- CPU 최대 (burst 감지용)
    MAX(u.maximum_usage.cpus) AS max_cpu_usage,

    -- Memory 평균 (시간 가중)
    SAFE_DIVIDE(
      SUM(u.average_usage.memory * (u.end_time - u.start_time)),
      SUM(u.end_time - u.start_time)
    ) AS mem_usage,

    -- Memory 최대 (압박 감지용)
    MAX(u.maximum_usage.memory) AS max_mem_usage,

    -- Page Cache Memory (시간 가중)
    SAFE_DIVIDE(
      SUM(u.page_cache_memory * (u.end_time - u.start_time)),
      SUM(u.end_time - u.start_time)
    ) AS page_cache_memory,

    -- Assigned Memory (시간 가중)
    SAFE_DIVIDE(
      SUM(u.assigned_memory * (u.end_time - u.start_time)),
      SUM(u.end_time - u.start_time)
    ) AS assigned_mem

  FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage` u
  JOIN sampled_instances s USING (collection_id, instance_index)
  GROUP BY 1, 2, 3
)

-- ⑧ 최종 출력 (12개 컬럼)
SELECT
  r.collection_id,
  r.instance_index,
  r.time_step_sec,
  r.cpu_usage,
  r.max_cpu_usage,
  r.mem_usage,
  r.max_mem_usage,
  r.page_cache_memory,
  r.assigned_mem,
  s.final_event_type,    -- 5=FAIL, 6=FINISH
  rq.requested_cpu,      -- NULL 가능 (Python에서 처리)
  rq.requested_mem       -- NULL 가능 (Python에서 처리)
FROM resampled_usage r
JOIN sampled_instances s USING (collection_id, instance_index)
LEFT JOIN resource_requests rq USING (collection_id, instance_index)
ORDER BY r.collection_id, r.instance_index, r.time_step_sec

);
