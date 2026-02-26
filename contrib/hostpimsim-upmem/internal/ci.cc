#include "internal.hh"
#include "runtime.hh"

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

void CI::reset_state(struct upmem_runtime *rt, size_t ci_index) {
  if (!rt || ci_index >= kNumCis) {
    return;
  }

  auto &ci = rt->cis[ci_index];
  ci.selected_mask = 0xFFu;
  ci.group_masks.fill(0u);
  ci.group_masks[0] = 0x01u;

  ci.pc_mode = 0x04u;
  ci.dma_mux_status.fill(0x00u);
  for (auto &regs : ci.dma_ctrl_regs) {
    regs.fill(0x00u);
  }
  ci.dma_ctrl_read_register = 0x00u;
  ci.stack_up_mask = 0x00u;

  ci.structure = 0;
  ci.iram_write_structure_valid = false;
  ci.iram_write_addr_hi = 0;
  ci.wram_write_structure_valid = false;
  ci.wram_write_addr = 0;
}

uint32_t CI::payload_for_command(struct upmem_runtime *rt, size_t ci,
                                 uint64_t cmd_word, bool *needs_mask_fuzz) {
  return upmem_pipeline_payload_for_command(rt, ci, cmd_word, needs_mask_fuzz);
}

uint32_t upmem_runtime_payload_for_command(struct upmem_runtime *rt, size_t ci,
                                           uint64_t cmd_word,
                                           bool *needs_mask_fuzz) {
  return upmem_pipeline_payload_for_command(rt, ci, cmd_word, needs_mask_fuzz);
}

uint32_t upmem_runtime_run_state_for_dpu(struct upmem_runtime *rt, size_t ci,
                                         uint8_t dpu_local) {
  return upmem_pipeline_run_state_for_dpu(rt, ci, dpu_local);
}
