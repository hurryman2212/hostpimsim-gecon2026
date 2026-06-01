#include "dpu.hh"

#include <cstring>

static inline uint16_t bswap16(uint16_t value) {
  return static_cast<uint16_t>((value >> 8) | (value << 8));
}

static inline uint32_t bswap32(uint32_t value) {
  return ((value & 0x000000FFu) << 24) | ((value & 0x0000FF00u) << 8) |
         ((value & 0x00FF0000u) >> 8) | ((value & 0xFF000000u) >> 24);
}

static inline uint64_t bswap64(uint64_t value) {
  value = ((value & 0x00000000FFFFFFFFULL) << 32) |
          ((value & 0xFFFFFFFF00000000ULL) >> 32);
  value = ((value & 0x0000FFFF0000FFFFULL) << 16) |
          ((value & 0xFFFF0000FFFF0000ULL) >> 16);
  value = ((value & 0x00FF00FF00FF00FFULL) << 8) |
          ((value & 0xFF00FF00FF00FF00ULL) >> 8);
  return value;
}

bool WRAM::rel_in_bounds(uint32_t offset, size_t size) const {
  const size_t off = static_cast<size_t>(offset);
  return off <= sizeof(data_) && size <= (sizeof(data_) - off);
}

bool WRAM::word_in_bounds(uint32_t word_offset) const {
  const size_t off = static_cast<size_t>(word_offset) * 4;
  return off + 4u <= sizeof(data_);
}

uint8_t WRAM::load_u8(uint32_t offset) const {
  if (!rel_in_bounds(offset, 1u)) {
    return 0u;
  }
  return data_[offset];
}

uint32_t WRAM::load_u16(uint32_t offset, bool big_endian) const {
  uint16_t value = 0u;
  if (!rel_in_bounds(offset, sizeof(value))) {
    return 0;
  }
  std::memcpy(&value, data_ + offset, sizeof(value));
  if (big_endian) {
    value = bswap16(value);
  }
  return value;
}

uint32_t WRAM::load_u32(uint32_t offset, bool big_endian) const {
  uint32_t value = 0u;
  if (!rel_in_bounds(offset, sizeof(value))) {
    return 0;
  }
  std::memcpy(&value, data_ + offset, sizeof(value));
  if (big_endian) {
    value = bswap32(value);
  }
  return value;
}

uint64_t WRAM::load_u64(uint32_t offset, bool big_endian) const {
  uint64_t value = 0u;
  if (!rel_in_bounds(offset, sizeof(value))) {
    return 0;
  }
  std::memcpy(&value, data_ + offset, sizeof(value));
  if (big_endian) {
    value = bswap64(value);
  }
  return value;
}

void WRAM::store_u8(uint32_t offset, uint8_t value) {
  if (!rel_in_bounds(offset, 1)) {
    return;
  }
  data_[offset] = value;
}

void WRAM::store_u16(uint32_t offset, uint16_t value, bool big_endian) {
  if (!rel_in_bounds(offset, sizeof(value))) {
    return;
  }
  if (big_endian) {
    value = bswap16(value);
  }
  std::memcpy(data_ + offset, &value, sizeof(value));
}

void WRAM::store_u32(uint32_t offset, uint32_t value, bool big_endian) {
  if (!rel_in_bounds(offset, sizeof(value))) {
    return;
  }
  if (big_endian) {
    value = bswap32(value);
  }
  std::memcpy(data_ + offset, &value, sizeof(value));
}

void WRAM::store_u64(uint32_t offset, uint64_t value, bool big_endian) {
  if (!rel_in_bounds(offset, sizeof(value))) {
    return;
  }
  if (big_endian) {
    value = bswap64(value);
  }
  std::memcpy(data_ + offset, &value, sizeof(value));
}

uint32_t WRAM::read_word(uint32_t word_offset) const {
  if (!word_in_bounds(word_offset)) {
    return 0;
  }
  const uint32_t byte_addr = word_offset * 4;
  return load_u32(byte_addr, false);
}

void WRAM::write_word(uint32_t word_offset, uint32_t value) {
  if (!word_in_bounds(word_offset)) {
    return;
  }
  const uint32_t byte_addr = word_offset * 4;
  store_u32(byte_addr, value, false);
}
