#include "ci.hh"

CI::CI() = default;

CI::CI(void *mapping_base, size_t mapping_size)
    : mapping_base_(mapping_base), mapping_size_(mapping_size) {}

CI::CI(size_t mapping_offset, size_t mapping_size)
    : mapping_base_(reinterpret_cast<void *>(mapping_offset)),
      mapping_size_(mapping_size) {}

void CI::rebind(void *mapping_base, size_t mapping_size) {
  mapping_base_ = mapping_base;
  mapping_size_ = mapping_size;
}

void CI::rebind(size_t mapping_offset, size_t mapping_size) {
  mapping_base_ = reinterpret_cast<void *>(mapping_offset);
  mapping_size_ = mapping_size;
}

void *CI::mapping_base() const { return mapping_base_; }

size_t CI::mapping_size() const { return mapping_size_; }

void CI::reset_state() {
  selected_mask = 0xFFu;
  group_mask.fill(0u);
  group_mask[0] = 0x01u;

  pc_mode = 0x04u;
  dma_mux_status = 0x00u;
  dma_mux_status_per_dpu.fill(0x00u);
  for (auto &regs : dma_ctrl_regs) {
    regs.fill(0x00u);
  }
  dma_ctrl_read_register = 0x00u;
  stack_up_mask = 0x00u;

  structure = 0;
  iram_write_structure_valid = false;
  iram_write_addr_hi = 0;
  wram_write_structure_valid = false;
  wram_write_addr = 0;
}

uint32_t CI::payload_for_command(upmem_pim_rank *rank, size_t ci,
                                 uint64_t cmd_word, bool *needs_mask_fuzz) {
  return Pipeline::payload_for_command(rank, ci, cmd_word, needs_mask_fuzz);
}
