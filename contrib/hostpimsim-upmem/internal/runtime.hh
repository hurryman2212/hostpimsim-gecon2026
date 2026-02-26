#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

constexpr size_t kNumCis = 8;
constexpr size_t kNumDpusPerCi = 8;
constexpr size_t kNumDpus = kNumCis * kNumDpusPerCi;
constexpr size_t kNumGroups = 8;
constexpr size_t kNumTasklets = 24;

constexpr size_t ATOMIC_OFFSET = 0;
constexpr size_t ATOMIC_SIZE = 256;
constexpr size_t WRAM_OFFSET = 512;
constexpr size_t WRAM_SIZE = (128u * 1024u);
constexpr size_t IRAM_OFFSET = (384u * 1024u);
constexpr size_t IRAM_SIZE = (48u * 1024u);
constexpr size_t MRAM_OFFSET = (512u * 1024u);
constexpr size_t MRAM_SIZE = (64u * 1024u * 1024u);
constexpr size_t PRIVATE_MEM_SIZE = MRAM_OFFSET;

struct DpuState {
  std::array<uint8_t, PRIVATE_MEM_SIZE> private_mem{};

  uint8_t *mram_base = nullptr;
  size_t mram_size = 0;

  uint32_t running_tasklets = 0;
  bool launch_pending = false;
};

struct CiState {
  uint8_t selected_mask = 0xFF;
  std::array<uint8_t, kNumGroups> group_masks{};

  uint8_t pc_mode = 0x04;
  std::array<uint8_t, kNumDpusPerCi> dma_mux_status{};
  std::array<std::array<uint8_t, 256>, kNumDpusPerCi> dma_ctrl_regs{};
  uint8_t dma_ctrl_read_register = 0x00;
  uint8_t stack_up_mask = 0x00;

  uint64_t structure = 0;

  bool iram_write_structure_valid = false;
  uint16_t iram_write_addr_hi = 0;

  bool wram_write_structure_valid = false;
  uint16_t wram_write_addr = 0;
};

struct upmem_runtime {
  std::array<CiState, kNumCis> cis{};
  std::array<DpuState, kNumDpus> dpus{};

  uint8_t *fallback_mram_base = nullptr;
  size_t fallback_mram_size = 0;

  std::array<std::array<uint64_t, kNumTasklets>, 4> thread_cmd_cache{};
  bool thread_cmd_cache_ready = false;
};

inline size_t dpu_index(size_t ci, uint8_t dpu_local_id) {
  return static_cast<size_t>(dpu_local_id) * kNumCis + ci;
}

upmem_runtime *upmem_runtime_create(void);
void upmem_runtime_destroy(upmem_runtime *rt);
void upmem_runtime_reset(upmem_runtime *rt);

uint32_t upmem_runtime_payload_for_command(upmem_runtime *rt, size_t ci,
                                           uint64_t cmd_word,
                                           bool *needs_mask_fuzz);
uint32_t upmem_runtime_run_state_for_dpu(upmem_runtime *rt, size_t ci,
                                         uint8_t dpu_local);

void upmem_runtime_bind_mram(upmem_runtime *rt, size_t dpu_global_index,
                             void *mram_base, size_t mram_size);

uint32_t upmem_pipeline_payload_for_command(upmem_runtime *rt, size_t ci,
                                            uint64_t cmd_word,
                                            bool *needs_mask_fuzz);
uint32_t upmem_pipeline_run_state_for_dpu(upmem_runtime *rt, size_t ci,
                                          uint8_t dpu_local);

void invalidate_decoded_program_cache_48(const DpuState *dpu);

bool iram_slot_in_bounds(uint16_t iram_slot);
void write_iram_word(DpuState &dpu, uint16_t iram_slot, uint64_t value48);

bool wram_word_in_bounds(uint32_t word_addr);
void write_wram_word(DpuState &dpu, uint32_t word_addr, uint32_t value);
uint32_t read_wram_word(const DpuState &dpu, uint32_t word_addr);

bool wram_rel_in_bounds(uint32_t addr, size_t size);
bool wram_load(DpuState &dpu, uint32_t addr, void *dst, size_t size);
bool wram_store(DpuState &dpu, uint32_t addr, const void *src, size_t size);
uint8_t wram_load_u8(DpuState &dpu, uint32_t addr);
uint32_t wram_load_u32(DpuState &dpu, uint32_t addr, bool big_endian);
uint64_t wram_load_u64(DpuState &dpu, uint32_t addr, bool big_endian);
void wram_store_u8(DpuState &dpu, uint32_t addr, uint8_t value);
void wram_store_u16(DpuState &dpu, uint32_t addr, uint16_t value,
                    bool big_endian);
void wram_store_u32(DpuState &dpu, uint32_t addr, uint32_t value,
                    bool big_endian);
void wram_store_u64(DpuState &dpu, uint32_t addr, uint64_t value,
                    bool big_endian);
