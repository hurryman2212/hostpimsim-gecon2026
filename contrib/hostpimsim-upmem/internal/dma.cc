#include "internal.hh"
#include "runtime.hh"

#include <algorithm>

namespace {
constexpr uint8_t kOpLdma = 0x50;
constexpr uint8_t kOpLdmai = 0x51;
constexpr uint8_t kOpSdma = 0x52;
}

bool DMAEngine::is_opcode(uint8_t opcode) {
  return opcode == kOpLdma || opcode == kOpSdma || opcode == kOpLdmai;
}

bool DMAEngine::is_signature(std::string_view signature) {
  if (signature == "ldma:rri" || signature == "sdma:rri") {
    return true;
  }

  return signature.starts_with("ldmai:");
}

bool DMAEngine::transfer(DpuState &dpu, uint32_t wram_addr, uint32_t mram_addr,
                         size_t bytes, bool load_to_wram) {
  if (wram_addr >= WRAM_SIZE || mram_addr >= dpu.mram_size ||
      !dpu.mram_base || bytes == 0) {
    return false;
  }

  size_t max_copy = bytes;
  max_copy = std::min(max_copy, static_cast<size_t>(WRAM_SIZE - wram_addr));
  max_copy = std::min(max_copy, static_cast<size_t>(dpu.mram_size - mram_addr));
  if (max_copy == 0) {
    return false;
  }

  if (load_to_wram) {
    return WRAM::store(dpu, wram_addr, dpu.mram_base + mram_addr, max_copy);
  }

  return WRAM::load(dpu, wram_addr, dpu.mram_base + mram_addr, max_copy);
}
