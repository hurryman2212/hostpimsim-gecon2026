#pragma once

#include "hostpimsim.h"
#include "../dpu/dpu.hh"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

class upmem_pim_rank;

class CI {
public:
  CI();
  CI(void *mapping_base, size_t mapping_size);
  explicit CI(size_t mapping_offset, size_t mapping_size = sizeof(uint64_t));

  void rebind(void *mapping_base, size_t mapping_size);
  void rebind(size_t mapping_offset, size_t mapping_size = sizeof(uint64_t));

  void *mapping_base() const;
  size_t mapping_size() const;

  void reset_state();
  static uint32_t payload_for_command(upmem_pim_rank *rank, size_t ci,
                                      uint64_t cmd_word, bool *needs_mask_fuzz);

  std::mutex exec_lock;

  uint64_t committed = 0;
  std::atomic<uint64_t> updated{0};

  uint8_t pc_mode = 0x04u;
  uint8_t dma_mux_status = 0x00u;
  uint8_t stack_up_mask = 0x00u;
  uint8_t selected_mask = 0x00u;
  std::array<uint8_t, kNumGroups> group_mask{};
  std::array<uint8_t, kNumDpusPerCi> dma_mux_status_per_dpu{};
  std::array<std::array<uint8_t, 256>, kNumDpusPerCi> dma_ctrl_regs{};
  uint8_t dma_ctrl_read_register = 0x00u;
  uint64_t structure = 0;
  bool iram_write_structure_valid = false;
  uint16_t iram_write_addr_hi = 0;
  bool wram_write_structure_valid = false;
  uint16_t wram_write_addr = 0;
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
