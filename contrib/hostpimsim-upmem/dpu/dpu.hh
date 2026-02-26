#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

struct DpuState;
struct CiState;
struct dpu_state;
struct upmem_runtime;

class WRAM {
public:
  explicit WRAM(size_t wram_size_bytes);

  size_t size_bytes() const;

  static bool rel_in_bounds(uint32_t addr, size_t size);
  static bool word_in_bounds(uint32_t word_addr);

  static bool load(DpuState &dpu, uint32_t addr, void *dst, size_t size);
  static bool store(DpuState &dpu, uint32_t addr, const void *src, size_t size);

  static uint8_t load_u8(DpuState &dpu, uint32_t addr);
  static uint32_t load_u32(DpuState &dpu, uint32_t addr, bool big_endian);
  static uint64_t load_u64(DpuState &dpu, uint32_t addr, bool big_endian);

  static void store_u8(DpuState &dpu, uint32_t addr, uint8_t value);
  static void store_u16(DpuState &dpu, uint32_t addr, uint16_t value,
                        bool big_endian);
  static void store_u32(DpuState &dpu, uint32_t addr, uint32_t value,
                        bool big_endian);
  static void store_u64(DpuState &dpu, uint32_t addr, uint64_t value,
                        bool big_endian);

  static void write_word(DpuState &dpu, uint32_t word_addr, uint32_t value);
  static uint32_t read_word(const DpuState &dpu, uint32_t word_addr);

private:
  size_t wram_size_bytes_ = 0;
};

class IRAM {
public:
  explicit IRAM(size_t iram_size_bytes);

  size_t size_bytes() const;

  static bool slot_in_bounds(uint16_t iram_slot);
  static void write_word(DpuState &dpu, uint16_t iram_slot, uint64_t value48);

private:
  size_t iram_size_bytes_ = 0;
};

class DMAEngine {
public:
  DMAEngine() = default;

  static bool is_opcode(uint8_t opcode);
  static bool is_signature(std::string_view signature);
  static bool transfer(DpuState &dpu, uint32_t wram_addr, uint32_t mram_addr,
                       size_t bytes, bool load_to_wram);
};

class MRAM {
public:
  MRAM(void *mapping_base, size_t mapping_size);

  void rebind(void *mapping_base, size_t mapping_size);

  void *mapping_base() const;
  size_t mapping_size() const;

  static void bind(upmem_runtime *rt, size_t dpu_global_index,
                   void *mram_base, size_t mram_size);
  static bool is_bound(const DpuState &dpu);

private:
  void *mapping_base_ = nullptr;
  size_t mapping_size_ = 0;
};

class Pipeline {
public:
  explicit Pipeline(size_t tasklet_count);

  size_t tasklet_count() const;

  static void reset_ci(CiState &ci);
  static uint32_t payload_for_command(upmem_runtime *rt, size_t ci,
                                      uint64_t cmd_word, bool *needs_mask_fuzz);

  static uint32_t perf_counter_read_32(const dpu_state &state);
  static uint32_t perf_counter_config(dpu_state &state, uint32_t config_bits);
  static void perf_counter_retire_step(dpu_state &state, bool is_replay_step);

  static uint32_t low_u32(uint64_t value);
  static uint8_t byte(uint64_t value, unsigned shift);
  static uint8_t dreg_slot(int reg_num);

private:
  struct RegisterSet {
    uint8_t perf_mode_same = 0;
    uint8_t perf_mode_cycles = 1;
    uint8_t perf_mode_instructions = 2;
    uint8_t perf_mode_nothing = 3;
    uint8_t dreg_divisor = 2;
  };

  RegisterSet register_set_{};
  size_t tasklet_count_ = 0;

  static const Pipeline &instance();

  uint32_t perf_counter_read_32_impl(const dpu_state &state) const;
  uint32_t perf_counter_config_impl(dpu_state &state,
                                    uint32_t config_bits) const;
  void perf_counter_retire_step_impl(dpu_state &state,
                                     bool is_replay_step) const;

  uint32_t low_u32_impl(uint64_t value) const;
  uint8_t byte_impl(uint64_t value, unsigned shift) const;
  uint8_t dreg_slot_impl(int reg_num) const;
};
