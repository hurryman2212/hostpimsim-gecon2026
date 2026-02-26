#include "internal.hh"
#include "runtime.hh"

#include <cstring>

namespace {
constexpr size_t kWramWords = WRAM_SIZE / sizeof(uint32_t);
}

WRAM::WRAM(size_t wram_size_bytes) : wram_size_bytes_(wram_size_bytes) {}

size_t WRAM::size_bytes() const { return wram_size_bytes_; }

bool WRAM::rel_in_bounds(uint32_t addr, size_t size) {
  const uint64_t end = static_cast<uint64_t>(addr) + static_cast<uint64_t>(size);
  return end <= static_cast<uint64_t>(WRAM_SIZE);
}

bool WRAM::word_in_bounds(uint32_t word_addr) {
  if (word_addr >= kWramWords) {
    return false;
  }

  const size_t byte_off = WRAM_OFFSET + static_cast<size_t>(word_addr) * 4u;
  return (byte_off + sizeof(uint32_t)) <= PRIVATE_MEM_SIZE;
}

bool WRAM::load(DpuState &dpu, uint32_t addr, void *dst, size_t size) {
  if (!dst || !rel_in_bounds(addr, size)) {
    return false;
  }

  const size_t abs = WRAM_OFFSET + static_cast<size_t>(addr);
  std::memcpy(dst, dpu.private_mem.data() + abs, size);
  return true;
}

bool WRAM::store(DpuState &dpu, uint32_t addr, const void *src, size_t size) {
  if (!src || !rel_in_bounds(addr, size)) {
    return false;
  }

  const size_t abs = WRAM_OFFSET + static_cast<size_t>(addr);
  std::memcpy(dpu.private_mem.data() + abs, src, size);
  return true;
}

uint8_t WRAM::load_u8(DpuState &dpu, uint32_t addr) {
  if (!rel_in_bounds(addr, 1)) {
    return 0;
  }
  const size_t abs = WRAM_OFFSET + static_cast<size_t>(addr);
  return dpu.private_mem[abs];
}

uint32_t WRAM::load_u32(DpuState &dpu, uint32_t addr, bool big_endian) {
  if (!rel_in_bounds(addr, 4)) {
    return 0;
  }
  const size_t abs = WRAM_OFFSET + static_cast<size_t>(addr);

  uint32_t v = 0;
  std::memcpy(&v, dpu.private_mem.data() + abs, sizeof(v));
  return big_endian ? static_cast<uint32_t>(__builtin_bswap32(v)) : v;
}

uint64_t WRAM::load_u64(DpuState &dpu, uint32_t addr, bool big_endian) {
  if (!rel_in_bounds(addr, 8)) {
    return 0;
  }
  const size_t abs = WRAM_OFFSET + static_cast<size_t>(addr);

  uint64_t v = 0;
  std::memcpy(&v, dpu.private_mem.data() + abs, sizeof(v));
  return big_endian ? static_cast<uint64_t>(__builtin_bswap64(v)) : v;
}

void WRAM::store_u8(DpuState &dpu, uint32_t addr, uint8_t value) {
  if (!rel_in_bounds(addr, 1)) {
    return;
  }
  const size_t abs = WRAM_OFFSET + static_cast<size_t>(addr);
  dpu.private_mem[abs] = value;
}

void WRAM::store_u16(DpuState &dpu, uint32_t addr, uint16_t value,
                     bool big_endian) {
  if (!rel_in_bounds(addr, 2)) {
    return;
  }
  const size_t abs = WRAM_OFFSET + static_cast<size_t>(addr);

  const uint16_t v =
      big_endian ? static_cast<uint16_t>(__builtin_bswap16(value)) : value;
  std::memcpy(dpu.private_mem.data() + abs, &v, sizeof(v));
}

void WRAM::store_u32(DpuState &dpu, uint32_t addr, uint32_t value,
                     bool big_endian) {
  if (!rel_in_bounds(addr, 4)) {
    return;
  }
  const size_t abs = WRAM_OFFSET + static_cast<size_t>(addr);

  const uint32_t v =
      big_endian ? static_cast<uint32_t>(__builtin_bswap32(value)) : value;
  std::memcpy(dpu.private_mem.data() + abs, &v, sizeof(v));
}

void WRAM::store_u64(DpuState &dpu, uint32_t addr, uint64_t value,
                     bool big_endian) {
  if (!rel_in_bounds(addr, 8)) {
    return;
  }
  const size_t abs = WRAM_OFFSET + static_cast<size_t>(addr);

  const uint64_t v =
      big_endian ? static_cast<uint64_t>(__builtin_bswap64(value)) : value;
  std::memcpy(dpu.private_mem.data() + abs, &v, sizeof(v));
}

void WRAM::write_word(DpuState &dpu, uint32_t word_addr, uint32_t value) {
  if (!word_in_bounds(word_addr)) {
    return;
  }

  const size_t byte_off = WRAM_OFFSET + static_cast<size_t>(word_addr) * 4u;
  std::memcpy(dpu.private_mem.data() + byte_off, &value, sizeof(value));
}

uint32_t WRAM::read_word(const DpuState &dpu, uint32_t word_addr) {
  if (!word_in_bounds(word_addr)) {
    return 0u;
  }

  const size_t byte_off = WRAM_OFFSET + static_cast<size_t>(word_addr) * 4u;
  uint32_t value = 0;
  std::memcpy(&value, dpu.private_mem.data() + byte_off, sizeof(value));
  return value;
}

bool wram_word_in_bounds(uint32_t word_addr) {
  return WRAM::word_in_bounds(word_addr);
}

void write_wram_word(DpuState &dpu, uint32_t word_addr, uint32_t value) {
  WRAM::write_word(dpu, word_addr, value);
}

uint32_t read_wram_word(const DpuState &dpu, uint32_t word_addr) {
  return WRAM::read_word(dpu, word_addr);
}

bool wram_rel_in_bounds(uint32_t addr, size_t size) {
  return WRAM::rel_in_bounds(addr, size);
}

bool wram_load(DpuState &dpu, uint32_t addr, void *dst, size_t size) {
  return WRAM::load(dpu, addr, dst, size);
}

bool wram_store(DpuState &dpu, uint32_t addr, const void *src, size_t size) {
  return WRAM::store(dpu, addr, src, size);
}

uint8_t wram_load_u8(DpuState &dpu, uint32_t addr) {
  return WRAM::load_u8(dpu, addr);
}

uint32_t wram_load_u32(DpuState &dpu, uint32_t addr, bool big_endian) {
  return WRAM::load_u32(dpu, addr, big_endian);
}

uint64_t wram_load_u64(DpuState &dpu, uint32_t addr, bool big_endian) {
  return WRAM::load_u64(dpu, addr, big_endian);
}

void wram_store_u8(DpuState &dpu, uint32_t addr, uint8_t value) {
  WRAM::store_u8(dpu, addr, value);
}

void wram_store_u16(DpuState &dpu, uint32_t addr, uint16_t value,
                    bool big_endian) {
  WRAM::store_u16(dpu, addr, value, big_endian);
}

void wram_store_u32(DpuState &dpu, uint32_t addr, uint32_t value,
                    bool big_endian) {
  WRAM::store_u32(dpu, addr, value, big_endian);
}

void wram_store_u64(DpuState &dpu, uint32_t addr, uint64_t value,
                    bool big_endian) {
  WRAM::store_u64(dpu, addr, value, big_endian);
}
