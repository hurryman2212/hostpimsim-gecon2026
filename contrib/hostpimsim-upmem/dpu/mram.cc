#include "../upmem.hh"

MRAM::MRAM(void *mapping_base, size_t mapping_size)
    : mapping_base_(mapping_base), mapping_size_(mapping_size) {}

void MRAM::rebind(void *mapping_base, size_t mapping_size) {
  mapping_base_ = mapping_base;
  mapping_size_ = mapping_size;
}

void *MRAM::mapping_base() const { return mapping_base_; }

size_t MRAM::mapping_size() const { return mapping_size_; }

bool MRAM::is_bound(const upmem_dpu &dpu) {
  return dpu.mram_base() != nullptr && dpu.mram_size() != 0;
}
