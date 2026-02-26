#include "internal.hh"
#include "runtime.hh"

MRAM::MRAM(void *mapping_base, size_t mapping_size)
    : mapping_base_(mapping_base), mapping_size_(mapping_size) {}

void MRAM::rebind(void *mapping_base, size_t mapping_size) {
  mapping_base_ = mapping_base;
  mapping_size_ = mapping_size;
}

void *MRAM::mapping_base() const { return mapping_base_; }

size_t MRAM::mapping_size() const { return mapping_size_; }

void MRAM::bind(struct upmem_runtime *rt, size_t dpu_global_index,
                void *mram_base, size_t mram_size) {
  if (!rt || dpu_global_index >= kNumDpus) {
    return;
  }

  auto *base = static_cast<uint8_t *>(mram_base);
  auto &state = rt->dpus[dpu_global_index];
  state.mram_base = base;
  state.mram_size = mram_size;

  if (base && mram_size > 0) {
    rt->fallback_mram_base = base;
    rt->fallback_mram_size = mram_size;
  }
}

bool MRAM::is_bound(const DpuState &dpu) {
  return dpu.mram_base != nullptr && dpu.mram_size != 0;
}
