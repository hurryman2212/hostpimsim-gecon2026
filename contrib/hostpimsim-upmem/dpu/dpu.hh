#pragma once

#include "../internal.hh"

#include <array>
#include <atomic>

class MRAM {
private:
  uint8_t *addr_{nullptr};
  size_t size_{0};
  bool owns_mapping_{false};

  void release_owned_mapping();

public:
  MRAM() = default;
  ~MRAM();
  MRAM(const MRAM &) = delete;
  MRAM &operator=(const MRAM &) = delete;

  bool allocate();
  bool translate_aperture_offset(uint32_t logical_byte_offset,
                                 size_t &aperture_offset) const;

  void bind(uint8_t *addr, size_t size);

  bool byte_in_bounds(uint32_t byte_offset, size_t size) const;
  uint8_t load_u8(uint32_t byte_offset) const;
  void store_u8(uint32_t byte_offset, uint8_t value);

  bool host_copy_to_mram(uint32_t offset_in_mram, const uint8_t *src,
                         size_t size);
  bool host_copy_from_mram(uint8_t *dst, uint32_t offset_in_mram,
                           size_t size) const;

  uint8_t *get_addr() { return addr_; }
  const uint8_t *get_addr() const { return addr_; }
  size_t get_size() const { return size_; }
};

class IRAM {
private:
  uint8_t inst_[kIRAMSizeBytes]{};

public:
  bool slot_in_bounds(uint16_t iram_slot) const;

  uint64_t read_word(uint16_t iram_slot) const;
  void write_word(uint16_t iram_slot, uint64_t inst48);

  uint8_t *get_addr() { return inst_; }
  const uint8_t *get_addr() const { return inst_; }
  size_t get_size() const { return sizeof(inst_); }
};

class WRAM {
private:
  uint8_t data_[kWRAMSizeBytes]{};

public:
  bool rel_in_bounds(uint32_t offset, size_t size) const;
  bool word_in_bounds(uint32_t word_offset) const;

  uint8_t load_u8(uint32_t offset) const;
  uint32_t load_u16(uint32_t offset, bool big_endian) const;
  uint32_t load_u32(uint32_t offset, bool big_endian) const;
  uint64_t load_u64(uint32_t offset, bool big_endian) const;

  void store_u8(uint32_t offset, uint8_t value);
  void store_u16(uint32_t offset, uint16_t value, bool big_endian);
  void store_u32(uint32_t offset, uint32_t value, bool big_endian);
  void store_u64(uint32_t offset, uint64_t value, bool big_endian);

  uint32_t read_word(uint32_t word_offset) const;
  void write_word(uint32_t word_offset, uint32_t value);

  uint8_t *get_addr() { return data_; }
  const uint8_t *get_addr() const { return data_; }
  size_t get_size() const { return sizeof(data_); }
};

class DMAEngine {
private:
  uint8_t dma_mux_status_{0x00u};
  uint8_t dma_ctrl_reg_{0x00u};

public:
  bool dma_to_iram_from_mram(IRAM &iram, const MRAM &mram, size_t bytes,
                             uint32_t mram_offset, uint16_t iram_slot) const;

  bool dma_to_wram_from_mram(WRAM &wram, const MRAM &mram, size_t bytes,
                             uint32_t wram_offset, uint32_t mram_offset) const;
  bool dma_to_mram_from_wram(MRAM &mram, const WRAM &wram, size_t bytes,
                             uint32_t mram_offset, uint32_t wram_offset) const;

  bool is_opcode(uint8_t opcode) const;

  void set_dma_mux_status(uint8_t v);
  uint8_t get_dma_mux_status() const;

  void set_dma_ctrl_reg(uint8_t v);
  uint8_t get_dma_ctrl_reg() const;
};

class Pipeline {
private:
  struct RegisterFile {
    std::array<std::array<uint32_t, kNumGPRegPerTasklet>, kNumTasklets>
        gp_regs_{};
    std::array<uint64_t, kNumTasklets> pc_regs_{};
    std::array<uint8_t, kNumTasklets> zf_regs_{};
    std::array<uint8_t, kNumTasklets> cf_regs_{};

    // Performance Counters (DPU-level canonical state)
    uint32_t perf_counter_raw_{0};
    uint8_t perf_counter_mode_{0};
  } register_file_{};

  uint64_t run_bits_{0};
  uint64_t sleep_bits_{0};
  uint64_t replay_bits_{0};
  std::array<uint8_t, 256> atomic_bits_{};

  const size_t nr_tasklet_{kNumTasklets};

public:
  uint32_t perf_counter_read32() const;
  uint32_t perf_counter_config(uint32_t config_bits);
  void perf_counter_retire_step(bool is_replay_step);

  uint32_t low_u32(uint64_t value);
  uint8_t byte(uint64_t value, unsigned shift);
  uint8_t dreg_slot(int reg_num);

  /*
   * Execute DPU program with the current state and the given
   * IRAM/WRAM/MRAM/DMAEngine context.
   *
   * Return value represents whether to proceed the next command or not.
   */
  bool execute_once(IRAM &iram, WRAM &wram, DMAEngine &dma_engine, MRAM &mram,
                    int tasklet_idx);

  RegisterFile &register_file() { return register_file_; }
  const RegisterFile &register_file() const { return register_file_; }

  uint64_t get_run_bits() const { return run_bits_; }
  void set_run_bits(uint64_t bits) { run_bits_ = bits; }

  uint64_t get_sleep_bits() const { return sleep_bits_; }
  void set_sleep_bits(uint64_t bits) { sleep_bits_ = bits; }

  uint64_t get_replay_bits() const { return replay_bits_; }
  void set_replay_bits(uint64_t bits) { replay_bits_ = bits; }

  std::array<uint8_t, 256> &atomic_bits() { return atomic_bits_; }
  const std::array<uint8_t, 256> &atomic_bits() const { return atomic_bits_; }

  size_t get_nr_tasklet() const { return nr_tasklet_; }
};

/* This instance is used `user_data` for the lower-level DPU handler. */
class UPMEMDPU {
private:
  DMAEngine dma_engine_{};
  IRAM iram_{};
  MRAM mram_{};
  Pipeline pipeline_{};
  WRAM wram_{};

  std::atomic<uint32_t> last_control_result_{0u};
  std::atomic<uint64_t> last_control_generation_{0u};

public:
  /*
   * Control function to execute a single CI command.
   *
   * In case of an execution command, each tasklet is fine-grained
   * multithreaded, which is time-sliced in a round-robin manner.
   *
   * Updates this DPU's cached 32-bit control payload result internally
   * (last_control_result_ / generation).
   */
  void control(uint64_t ci_cmd);

  void clear_last_control_result() {
    last_control_result_.store(0xFFFFFFFFu, std::memory_order_release);
  }
  void set_last_control_result(uint32_t value) {
    last_control_result_.store(value, std::memory_order_release);
    (void)last_control_generation_.fetch_add(1u, std::memory_order_release);
  }
  uint32_t get_last_control_result() const {
    return last_control_result_.load(std::memory_order_acquire);
  }
  uint64_t get_last_control_generation() const {
    return last_control_generation_.load(std::memory_order_acquire);
  }

  DMAEngine &dma_engine() { return dma_engine_; }
  const DMAEngine &dma_engine() const { return dma_engine_; }
  IRAM &iram() { return iram_; }
  const IRAM &iram() const { return iram_; }
  MRAM &mram() { return mram_; }
  const MRAM &mram() const { return mram_; }
  Pipeline &pipeline() { return pipeline_; }
  const Pipeline &pipeline() const { return pipeline_; }
  WRAM &wram() { return wram_; }
  const WRAM &wram() const { return wram_; }
};
