#pragma once

#include <cstddef>
#include <cstdint>

#ifndef DPU_NR_TASKLETS
#define DPU_NR_TASKLETS 24
#endif
static constexpr size_t kNumTasklets = DPU_NR_TASKLETS;

static constexpr size_t kIRAMSizeBytes = 24 * 1024;
static constexpr size_t kWRAMSizeBytes = 64 * 1024;

static constexpr size_t kMramSizeBytes = 64 * 1024 * 1024;

static constexpr size_t kChipNumDpus = 8;
static constexpr size_t kRankNumBanks = 8;

static constexpr size_t kCiNumGroupSlots = 8;

static constexpr size_t kNumGPRegPerTasklet = 24;

/*
 * Convert internal "x" command layout to the physical CI frame layout.
 *
 * UPMEM CI words shuffle byte positions before they are emitted to the wire/
 * MMIO-facing 0x33-prefixed command format. The mapping below mirrors the
 * reference implementation's byte order transform and preserves command tag.
 */
static inline constexpr int64_t encode_frame_from_x(uint64_t x, uint8_t tag) {
  uint64_t cmd = 0x3300000000000000ULL;
  cmd |= ((x >> (2 * 8)) & 0xFFULL) << 0;
  cmd |= ((x >> (5 * 8)) & 0xFFULL) << 8;
  cmd |= ((x >> (4 * 8)) & 0xFFULL) << 16;
  cmd |= ((x >> (0 * 8)) & 0xFFULL) << 24;
  cmd |= ((x >> (1 * 8)) & 0xFFULL) << 32;
  cmd |= ((x >> (3 * 8)) & 0xFFULL) << 40;
  cmd |= static_cast<uint64_t>(tag) << 48;
  return cmd;
}

/* Logical command kind for decoded 0x98 thread-control frames. */
enum class ThreadCmdKind : uint8_t {
  Boot = 0,
  Resume = 1,
  ClearRun = 2,
  ReadRun = 3,
};
bool decode_thread_command(uint64_t cmd, ThreadCmdKind &kind,
                           uint8_t &thread_id);
