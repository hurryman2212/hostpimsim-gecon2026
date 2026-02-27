#include "../upmem.hh"

namespace {
constexpr size_t kIramWriteWordBytes = 6;
constexpr size_t kIramSlotBytes = 8;
constexpr size_t kIramWriteSlots = IRAM_SIZE / kIramSlotBytes;
}

IRAM::IRAM(size_t iram_size_bytes) : iram_size_bytes_(iram_size_bytes) {}

size_t IRAM::size_bytes() const { return iram_size_bytes_; }

bool IRAM::slot_in_bounds(uint16_t iram_slot) {
  if (iram_slot >= kIramWriteSlots) {
    return false;
  }

  const size_t byte_off =
      IRAM_OFFSET + static_cast<size_t>(iram_slot) * kIramSlotBytes;
  return (byte_off + kIramWriteWordBytes) <= PRIVATE_MEM_SIZE;
}

void IRAM::write_word(upmem_dpu &dpu, uint16_t iram_slot, uint64_t value48) {
  if (!slot_in_bounds(iram_slot)) {
    return;
  }

  const size_t byte_off =
      IRAM_OFFSET + static_cast<size_t>(iram_slot) * kIramSlotBytes;
  for (size_t i = 0; i < kIramWriteWordBytes; ++i) {
    dpu.private_mem[byte_off + i] =
        static_cast<uint8_t>((value48 >> (8u * i)) & 0xFFu);
  }
  for (size_t i = kIramWriteWordBytes; i < kIramSlotBytes; ++i) {
    dpu.private_mem[byte_off + i] = 0;
  }

  invalidate_decoded_program_cache_48(&dpu);
}
