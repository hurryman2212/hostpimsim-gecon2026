#include "dpu.hh"

#include <cstring>

static constexpr size_t kIramSlotBytes = 8;
static constexpr size_t kIramWriteWordBytes = 6;
static constexpr uint64_t kIramInst48Mask = 0x0000FFFFFFFFFFFFULL;

bool IRAM::slot_in_bounds(uint16_t iram_slot) const {
  constexpr size_t kIramWriteSlots = kIRAMSizeBytes / kIramSlotBytes;
  if (iram_slot >= kIramWriteSlots) {
    return false;
  }

  const size_t off = static_cast<size_t>(iram_slot) * kIramSlotBytes;
  return (off + kIramWriteWordBytes) <= sizeof(inst_);
}

uint64_t IRAM::read_word(uint16_t iram_slot) const {
  if (!slot_in_bounds(iram_slot)) {
    return 0u;
  }

  const size_t off = static_cast<size_t>(iram_slot) * kIramSlotBytes;
  uint64_t value = 0u;
  std::memcpy(&value, inst_ + off, sizeof(value));
  return value & kIramInst48Mask;
}

void IRAM::write_word(uint16_t iram_slot, uint64_t inst48) {
  if (!slot_in_bounds(iram_slot)) {
    return;
  }

  const size_t off = static_cast<size_t>(iram_slot) * kIramSlotBytes;
  const uint64_t value = inst48 & kIramInst48Mask;
  std::memcpy(inst_ + off, &value, sizeof(value));
}
