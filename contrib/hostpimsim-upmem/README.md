# hostpimsim-upmem (LD_PRELOAD)

`hostpimsim-upmem` is an **LD_PRELOAD-side UPMEM hardware simulation shim** built on top of `libhostpimsim`.

## Purpose

This contrib module targets the workflow:

```bash
LD_PRELOAD="/usr/local/lib/liboverlaysys.so:/path/to/libhostpimsim-upmem.so" \
UPMEM_PROFILE=backend=hw,regionMode=safe \
make test
```

It is designed for `/root/dpu_demo` and PrIM-style host applications where we intercept userspace syscalls and emulate required UPMEM rank/device behavior in-process.

## What it currently provides

- OverlaySys syscall-hook integration (syscall-only interception model).
- `libhostpimsim` vdev-backed virtual device exposure:
  - `/dev/dpu_rank0`
  - `/dev/dpu_dax0`
- Virtual sysfs mediation for libudev/libdpuhw discovery paths.
- CI/rank command decode path plus launch execution path (48-bit decode/execute).
- 48-bit decode is native C++ (no runtime Python dependency).
- Functional support for representative UPMEM benchmark workloads (checksum + PrIM suite used in this repo workflow).

## Scope / limitations

- Functional simulator path, **not cycle-accurate timing**.
- Throughput is highly sensitive to build type (`Release` strongly recommended for benchmark/runtime validation).
- Execution-step budget is capped (default internal cap), and can be overridden via env for heavy workloads.

## Build

From project root, recommended (Release):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target hostpimsim-upmem -j"$(nproc)"
```

Debug build for trace-oriented debugging:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --target hostpimsim-upmem -j"$(nproc)"
```

Output:

- `build/lib/libhostpimsim-upmem.so`

## Useful runtime env knobs

- `HOSTPIMSIM_UPMEM_TRACE_EXEC=1`
  - Enables execution trace logging to stderr.
- `HOSTPIMSIM_UPMEM_MAX_EXEC_STEPS=<N>`
  - Overrides launch execution-step cap for long-running kernels.
- `HOSTPIMSIM_UPMEM_PROFILE_SIG=1`
  - Prints instruction-signature frequency profile for launch48.
- `HOSTPIMSIM_UPMEM_PROFILE_SIG_LIMIT=<N>`
  - Limits printed Top-N entries for signature profile (`default: 24`).
- `HOSTPIMSIM_UPMEM_EAGER_ZERO_ALLOC=1`
  - Restores eager zero-fill on MRAM/DAX allocation paths.
  - Use as an immediate rollback/workaround toggle when cold-start kernels
    show first-touch regressions (for example, `TRNS@64`, `TS@1`).
- `HOSTPIMSIM_UPMEM_LAUNCH_TIMING=1`
  - Prints launch48 decode/execute timing breakdown (`decode_us`, `exec_us`, `total_us`).
- `HOSTPIMSIM_UPMEM_REPLAY_MODEL=1|2`
  - Enables an experimental replay-hazard model scaffold (48-bit path).
  - `1` (guarded, default when set): keeps the conservative low-active-thread gate.
  - `2` (strict): disables that gate and applies replay checks in a more fsim-like way.
  - Intended for fsim-behavior investigation, not default benchmarking.
- `HOSTPIMSIM_UPMEM_REPLAY_STRICT_SCOPE=all|rd`
  - Scope control for strict mode (`HOSTPIMSIM_UPMEM_REPLAY_MODEL=2`).
  - `all` (default): strict gate applies to all signatures.
  - `rd`: strict gate applies only to signatures that map to fsim `_must_replay_rd` call-sites; others fall back to guarded gating.
- `HOSTPIMSIM_UPMEM_REPLAY_STRICT_RD_SIGS=<csv>`
  - Fine-grained strict subset when `REPLAY_STRICT_SCOPE=rd`.
  - Default (unset): `div_step,mul_step,sd` (all rd-mapped signatures).
  - Examples:
    - `HOSTPIMSIM_UPMEM_REPLAY_STRICT_RD_SIGS=div_step,sd`
    - `HOSTPIMSIM_UPMEM_REPLAY_STRICT_RD_SIGS=mul_step`
  - Accepted tokens: `all`, `div|div_step|div_step:rrri`, `mul|mul_step|mul_step:rrrici`, `sd|sd:rir`.

### Decode cache (48-bit path)

- Decoded 48-bit IRAM programs are cached in-process and also persisted to:
  - `/tmp/hostpimsim-upmem-decode-cache/<hash>.txt`
- This significantly reduces repeated decode overhead on subsequent runs of the same DPU binary.

## Review Supplement

- Round 1 보완 문서:
  - `docs/hostpimsim-upmem-review-round1-supplement-2026-02-27.md`
