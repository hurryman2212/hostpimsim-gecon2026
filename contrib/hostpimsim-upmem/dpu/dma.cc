#include "dpu.hh"

#include <algorithm>

bool DMAEngine::dma_to_iram_from_mram(IRAM &iram, const MRAM &mram,
                                      size_t bytes, uint32_t mram_offset,
                                      uint16_t iram_slot) const {
  const size_t mram_size = mram.get_size();
  if (!mram.get_addr() || mram_size == 0 || bytes == 0) {
    return false;
  }

  const size_t iram_size = iram.get_size();
  const size_t iram_off = static_cast<size_t>(iram_slot) * 6;
  if (mram_offset >= mram_size || iram_off >= iram_size) {
    return false;
  }

  const size_t copy_bytes =
      std::min({bytes, mram_size - static_cast<size_t>(mram_offset),
                iram_size - iram_off});
  if (copy_bytes < 6) {
    return false;
  }

  const size_t slots = copy_bytes / 6;
  size_t copied_slots = 0;
  for (size_t slot = 0; slot < slots; ++slot) {
    const uint16_t dst_slot =
        static_cast<uint16_t>(static_cast<size_t>(iram_slot) + slot);
    if (!iram.slot_in_bounds(dst_slot)) {
      break;
    }

    uint64_t word = 0;
    const uint32_t src_off =
        static_cast<uint32_t>(static_cast<size_t>(mram_offset) + slot * 6);
    word |= static_cast<uint64_t>(mram.load_u8(src_off + 0)) << 0;
    word |= static_cast<uint64_t>(mram.load_u8(src_off + 1)) << 8;
    word |= static_cast<uint64_t>(mram.load_u8(src_off + 2)) << 16;
    word |= static_cast<uint64_t>(mram.load_u8(src_off + 3)) << 24;
    word |= static_cast<uint64_t>(mram.load_u8(src_off + 4)) << 32;
    word |= static_cast<uint64_t>(mram.load_u8(src_off + 5)) << 40;
    iram.write_word(dst_slot, word);
    ++copied_slots;
  }

  return copied_slots > 0;
}

bool DMAEngine::dma_to_wram_from_mram(WRAM &wram, const MRAM &mram,
                                      size_t bytes, uint32_t wram_offset,
                                      uint32_t mram_offset) const {
  const size_t mram_size = mram.get_size();
  if (!mram.get_addr() || mram_size == 0 || bytes == 0) {
    return false;
  }

  const size_t wram_size = wram.get_size();
  if (mram_offset >= mram_size || wram_offset >= wram_size) {
    return false;
  }

  const size_t copy_bytes =
      std::min({bytes, mram_size - static_cast<size_t>(mram_offset),
                wram_size - static_cast<size_t>(wram_offset)});
  if (copy_bytes == 0) {
    return false;
  }

  mram.host_copy_from_mram(wram.get_addr() + wram_offset, mram_offset,
                           copy_bytes);

  return true;
}
bool DMAEngine::dma_to_mram_from_wram(MRAM &mram, const WRAM &wram,
                                      size_t bytes, uint32_t mram_offset,
                                      uint32_t wram_offset) const {
  const size_t mram_size = mram.get_size();
  if (!mram.get_addr() || mram_size == 0 || bytes == 0) {
    return false;
  }

  const size_t wram_size = wram.get_size();
  if (mram_offset >= mram_size || wram_offset >= wram_size) {
    return false;
  }

  const size_t copy_bytes =
      std::min({bytes, mram_size - static_cast<size_t>(mram_offset),
                wram_size - static_cast<size_t>(wram_offset)});
  if (copy_bytes == 0) {
    return false;
  }

  mram.host_copy_to_mram(mram_offset, wram.get_addr() + wram_offset,
                         copy_bytes);

  return true;
}

bool DMAEngine::is_opcode(uint8_t opcode) const {
  return opcode == 0x50 || opcode == 0x51 || opcode == 0x52;
}

void DMAEngine::set_dma_mux_status(uint8_t v) { dma_mux_status_ = v; }

uint8_t DMAEngine::get_dma_mux_status() const { return dma_mux_status_; }

void DMAEngine::set_dma_ctrl_reg(uint8_t v) { dma_ctrl_reg_ = v; }

uint8_t DMAEngine::get_dma_ctrl_reg() const { return dma_ctrl_reg_; }
