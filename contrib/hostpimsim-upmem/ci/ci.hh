#pragma once

#include "hostpimsim.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

struct upmem_runtime;

class CI {
public:
  CI();
  CI(void *mapping_base, size_t mapping_size);
  explicit CI(size_t mapping_offset, size_t mapping_size = sizeof(uint64_t));

  void rebind(void *mapping_base, size_t mapping_size);
  void rebind(size_t mapping_offset, size_t mapping_size = sizeof(uint64_t));

  void *mapping_base() const;
  size_t mapping_size() const;

  static void reset_state(upmem_runtime *rt, size_t ci_index);
  static uint32_t payload_for_command(upmem_runtime *rt, size_t ci,
                                      uint64_t cmd_word, bool *needs_mask_fuzz);

  std::mutex exec_lock;

  uint64_t committed = 0;
  std::atomic<uint64_t> updated{0};

  uint8_t pc_mode = 0x04u;
  uint8_t dma_mux_status = 0x00u;
  uint8_t stack_up_mask = 0x00u;
  uint8_t selected_mask = 0x00u;
  std::array<uint8_t, 8> group_mask{};
  bool payload_needs_mask_fuzz = false;
  uint8_t payload_fuzz_idx = 0;

  pim_region_t *mmio_region = nullptr;
  pim_region_t *rw_region = nullptr;

  uint8_t dpu_commit_pending = 0;
  uint8_t dpu_commit_done = 0;
  uint64_t dpu_cmd = 0;
  uint32_t dpu_payload_accum = 0;
  bool dpu_needs_mask_fuzz = false;
  bool dpu_fanout_active = false;
  bool dpu_exec_per_dpu = false;
  bool dpu_serial_exec_done = false;

private:
  void *mapping_base_ = nullptr;
  size_t mapping_size_ = 0;
};
