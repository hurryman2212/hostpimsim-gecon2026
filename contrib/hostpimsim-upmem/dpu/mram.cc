#include "dpu.hh"

#include <cstring>

#include <sys/mman.h>

void MRAM::release_owned_mapping() {
  if (owns_mapping_ && addr_ && size_ != 0u) {
    munmap(addr_, size_);
  }
  addr_ = nullptr;
  size_ = 0u;
  owns_mapping_ = false;
}

MRAM::~MRAM() { release_owned_mapping(); }

bool MRAM::allocate() {
  if (addr_ && size_ != 0u) {
    return true;
  }

  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif

  void *mapped =
      mmap(nullptr, kMramSizeBytes, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (mapped == MAP_FAILED) {
    return false;
  }

  addr_ = static_cast<uint8_t *>(mapped);
  size_ = kMramSizeBytes;
  owns_mapping_ = true;
  return true;
}

bool MRAM::translate_aperture_offset(uint32_t logical_byte_offset,
                                     size_t &aperture_offset) const {
  // Translation is only valid for bound MRAM apertures and in-range accesses.
  if (!this->byte_in_bounds(logical_byte_offset, 1u) || !this->get_addr()) {
    return false;
  }

  // HostPIMSim does not expose rank/DAX MRAM mmap yet; all UPMEM benchmark
  // transfers enter through ioctl and use logical MRAM offsets. Keep the
  // backing store logical-linear so host transfers and DMA stay contiguous.
  aperture_offset = static_cast<size_t>(logical_byte_offset);
  if (aperture_offset >= size_) {
    return false;
  }
  return true;
}

static uint8_t *translate_byte_ptr(const MRAM &mram,
                                   uint32_t logical_byte_offset) {
  size_t aperture_offset = 0;
  if (!mram.translate_aperture_offset(logical_byte_offset, aperture_offset)) {
    return nullptr;
  }

  // Safe because caller requested mutable access API (store/copy-to path).
  return const_cast<uint8_t *>(mram.get_addr()) + aperture_offset;
}

void MRAM::bind(uint8_t *addr_in, size_t size_in) {
  release_owned_mapping();
  addr_ = addr_in;
  size_ = size_in;
}

bool MRAM::byte_in_bounds(uint32_t byte_offset, size_t size) const {
  if (size == 0) {
    return false;
  }

  const size_t logical_size = kMramSizeBytes;
  const size_t off = static_cast<size_t>(byte_offset);
  if (off > logical_size) {
    return false;
  }

  return size <= (logical_size - off);
}

uint8_t MRAM::load_u8(uint32_t byte_offset) const {
  // OOB/unbound reads are treated as zero to keep call sites simple and safe.
  const uint8_t *ptr = translate_byte_ptr(*this, byte_offset);
  return ptr ? *ptr : 0u;
}

void MRAM::store_u8(uint32_t byte_offset, uint8_t value) {
  uint8_t *ptr = translate_byte_ptr(*this, byte_offset);
  if (ptr) {
    *ptr = value;
  }
}

bool MRAM::host_copy_to_mram(uint32_t offset_in_mram, const uint8_t *src,
                             size_t size) {
  if (!src || !get_addr()) {
    return false;
  }
  if (size == 0) {
    return true;
  }
  if (!byte_in_bounds(offset_in_mram, size)) {
    return false;
  }

  std::memcpy(addr_ + offset_in_mram, src, size);
  return true;
}

bool MRAM::host_copy_from_mram(uint8_t *dst, uint32_t offset_in_mram,
                               size_t size) const {
  if (!dst || !get_addr()) {
    return false;
  }
  if (size == 0) {
    return true;
  }
  if (!byte_in_bounds(offset_in_mram, size)) {
    return false;
  }

  std::memcpy(dst, addr_ + offset_in_mram, size);
  return true;
}
