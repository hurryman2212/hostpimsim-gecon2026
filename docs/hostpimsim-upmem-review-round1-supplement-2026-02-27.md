# hostpimsim-upmem Review Round 1 Supplement (2026-02-27)

## 1) TRNS@64 / TS@1 회귀 원인 (코드 경로 기준)

이전 최적화(`339d9fb`)에서 preload 초기화 hot-path를 줄이기 위해 아래 eager zero-fill이 제거되었습니다.

- `contrib/hostpimsim-upmem/upmem.cc`
  - `ensure_dpu_mram_locked()`의 `memset(buf, 0, rank.mram_size)` 제거
  - `map_real_range()`의 `memset(base + offset, 0, length)` 제거

영향:

- zero 비용이 초기화 시점에서 런타임 first-touch(page fault) 시점으로 이동
- cold-start 분포가 커널/입력크기/DPU 수에 따라 달라져 케이스별 wall-time 편차 발생
- 특히 다음 패턴에서 편차가 커짐
  - `TRNS@64`: 초소형 다건 write fanout (`64B` write가 대량 반복)
  - `TS@1`: 단일 DPU에서 초기 memory touch 비중이 상대적으로 큼

참고 로그(패턴 확인):

- `/root/preload_compare_logs/xfer_trace_subset/ours_preload/prim/TRNS/dpu_64.log`
- `/root/preload_compare_logs/xfer_trace_subset/ours_preload/prim/TS/dpu_1.log`

## 2) 수정/우회안 반영

즉시 롤백 가능한 런타임 우회 토글을 추가했습니다.

- `HOSTPIMSIM_UPMEM_EAGER_ZERO_ALLOC=1` (기본값: off)
  - MRAM/DAX 할당 시 eager zero-fill을 복원
  - cold-start 회귀가 관측되면 즉시 활성화 가능

코드 반영 위치:

- `contrib/hostpimsim-upmem/upmem.cc`
  - `ensure_dpu_mram_locked()`에서 토글 on 시 eager zero-fill 수행
  - `map_real_range()`에서 토글 on 시 eager zero-fill 수행
- `contrib/hostpimsim-upmem/README.md`
  - 새 env knob 문서화

## 3) 재현/재측정 절차 (지정 스크립트 기준)

### 기준(pre-optimization) 라이브러리 재빌드

```bash
git worktree add /tmp/hostpimsim-feb3f6d feb3f6d
cmake -S /tmp/hostpimsim-feb3f6d -B /tmp/hostpimsim-feb3f6d/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/hostpimsim-feb3f6d/build --target libhostpimsim-upmem-preload.so -j"$(nproc)"
```

### 회귀 케이스 재측정 (TS/TRNS, 1/8/64 DPU)

```bash
/root/run_preload_vs_original_benchmark.py \
  --preload-lib /tmp/hostpimsim-feb3f6d/build/lib/libhostpimsim-upmem-preload.so \
  --skip-checksum --bench-regex '^(TS|TRNS)$' \
  --log-root /root/preload_compare_logs/baseline_refreshed_subset_feb3f6d

/root/run_preload_vs_original_benchmark.py \
  --preload-lib /root/Projects/hostpimsim/.climpire-worktrees/e3413ae9/build/lib/libhostpimsim-upmem-preload.so \
  --skip-checksum --bench-regex '^(TS|TRNS)$' \
  --log-root /root/preload_compare_logs/current_refreshed_subset_final
```

### 전체 51케이스 재측정

```bash
/root/run_preload_vs_original_benchmark.py \
  --preload-lib /root/Projects/hostpimsim/.climpire-worktrees/e3413ae9/build/lib/libhostpimsim-upmem-preload.so \
  --log-root /root/preload_compare_logs/current_refreshed_full_final
```

## 4) 결과

### 4.1 회귀 2건 비회귀 확인 (동일 세션 기준 baseline 대비)

기준: `baseline_refreshed_subset_feb3f6d` vs `current_refreshed_subset_final`

- `TRNS@64`: `9.543333s -> 8.696235s` (`+8.876%`)
- `TS@1`: `61.293456s -> 60.937797s` (`+0.580%`)

요구사항(`>= 0%`) 충족.

### 4.2 전체 51케이스 핵심 지표

출처: `/root/preload_compare_logs/current_refreshed_full_final`

- `mode_stats`
  - `original_simulator`: `ok=51 fail=0 timeout=0`
  - `ours_preload`: `ok=51 fail=0 timeout=0`
- `avg_wall_ratio (ours/sim)`: `0.973556`
- `ours <= sim` 비율: `40/51`
- `avg_dpu_stat_similarity`: `41.28%`

### 4.3 케이스별 speedup 요약 (preload wall-time, 기존 baseline reference 대비)

기준: `/root/preload_compare_logs/baseline_e3413ae9_full_default` vs `/root/preload_compare_logs/current_refreshed_full_final`

- 평균 speedup: `+10.008%`
- 회귀 케이스: `2/51`
  - `HST-L@8`: `-2.615%`
  - `MLP@1`: `-0.024%`

## 5) 영향 범위

- 영향 가능 영역
  - cold-start first-touch 비중이 큰 커널
  - extreme DPU scale (`1` 또는 `64`)에서 초기 메모리 접촉 분포가 치우친 입력
- 검증 완료 범위
  - checksum + PrIM 전체(`51/51`)에서 실패/타임아웃 `0`
  - 회귀 지적 2건(`TRNS@64`, `TS@1`) 비회귀 확인

## 6) 롤백 조건 / 즉시 우회 조건

### 조건

- 회귀 감지:
  - `TRNS@64` 또는 `TS@1`이 baseline 대비 `< 0%` (2회 연속), 또는
  - 전체 `avg_wall_ratio > 1.0`

### 즉시 조치

- 1차(런타임):

```bash
export HOSTPIMSIM_UPMEM_EAGER_ZERO_ALLOC=1
```

- 2차(코드): 회귀가 지속되면 `339d9fb` hot-path 변경분을 부분 롤백

## 7) QA/릴리즈 검증 매트릭스 및 차단 기준

검증 매트릭스:

- 커널: `checksum + PrIM(16)`
- DPU 수: `1, 8, 64`
- 모드: `original_simulator`, `ours_preload`
- 필수 체크:
  - `rc=0`, timeout 없음
  - `TRNS@64 >= 0%`, `TS@1 >= 0%` (baseline 대비)
  - `avg_wall_ratio <= 1.0`
  - `ours<=sim` 비율 유지/개선

차단 기준:

- 위 필수 체크 중 하나라도 실패 시 릴리즈 차단

적용 일정(이번 턴):

1. 코드 반영 및 문서 업데이트 완료
2. 회귀 2건 우선 재측정 완료
3. 전체 51케이스 재측정 완료
4. 차단 기준 재검토 후 승인/보류 결정
