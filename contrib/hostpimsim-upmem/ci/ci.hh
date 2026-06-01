#pragma once

#include "../internal.hh"

#include <array>
#include <atomic>

#include <hostpimsim.h>

class CIState {
private:
  mutable std::mutex lock_{};

  uint64_t committed_{0};
  std::atomic<uint64_t> updated_{0};

  uint8_t pc_mode_{0x04u};
  uint8_t stack_up_mask_{0x00u};

  uint8_t selected_mask_{0x00u};
  uint8_t enabled_mask_{0xFFu};
  std::array<uint8_t, kCiNumGroupSlots> group_mask_{};

  uint8_t dma_ctrl_read_register_{0x00u};
  uint64_t structure_{0};
  bool iram_write_structure_valid_{false};
  uint16_t iram_write_addr_hi_{0};
  bool wram_write_structure_valid_{false};
  uint16_t wram_write_addr_{0};

  uint32_t dispatched_payload_{0};
  bool dispatched_needs_mask_fuzz_{false};
  std::atomic<uint64_t> dispatched_generation_{0};

  struct DispatchPending {
    bool expected_set{false};
    uint64_t dispatch_generation_before{0};
    bool active{false};
  } dispatch_pending_{};

  struct ControlPending {
    bool expected_set{false};
    uint64_t dispatch_generation_before{0};
    uint8_t selected_mask{0};
    uint8_t execution_mask{0};
    uint8_t payload{0};
    std::array<uint64_t, kChipNumDpus> control_generations{};
    bool active{false};
  } control_pending_{};

  struct RunStatePending {
    bool expected_set{false};
    uint64_t dispatch_generation_before{0};
    uint8_t selected_mask{0};
    uint8_t execution_mask{0};
    uint8_t payload{0};
    std::array<uint64_t, kChipNumDpus> control_generations{};
    bool active{false};
  } run_state_pending_{};

public:
  static inline constexpr int8_t first_selected_dpu(uint8_t mask) {
    for (uint8_t dpu = 0; dpu < kChipNumDpus; ++dpu) {
      if ((mask & static_cast<uint8_t>(1u << dpu)) != 0u) {
        return dpu;
      }
    }
    return static_cast<uint8_t>(kChipNumDpus);
  }

  void reset_protocol_state();

  uint64_t get_committed() const;
  void set_committed(uint64_t word);

  uint64_t
  load_updated(std::memory_order order = std::memory_order_acquire) const;
  void store_updated(uint64_t word,
                     std::memory_order order = std::memory_order_release);

  uint8_t get_selected_mask() const;
  void set_selected_mask(uint8_t mask);
  uint8_t get_enabled_mask() const;
  void set_enabled_mask(uint8_t mask);

  uint8_t get_group_mask(size_t group_slot_nr) const;
  void set_group_mask(size_t group_slot_nr, uint8_t mask);

  uint8_t get_pc_mode() const;
  void set_pc_mode(uint8_t mode);
  uint8_t exchange_stack_up_mask(uint8_t value);

  uint8_t get_dma_ctrl_read_register() const;
  void set_dma_ctrl_read_register(uint8_t value);

  uint64_t get_structure() const;
  void set_structure(uint64_t value);

  bool is_iram_write_structure_valid() const;
  void set_iram_write_structure_valid(bool valid);
  uint16_t get_iram_write_addr_hi() const;
  void set_iram_write_addr_hi(uint16_t addr_hi);

  bool is_wram_write_structure_valid() const;
  void set_wram_write_structure_valid(bool valid);
  uint16_t get_wram_write_addr() const;
  void set_wram_write_addr(uint16_t addr);

  void set_dispatched_result(uint32_t payload, bool needs_mask_fuzz);
  uint32_t get_dispatched_payload() const;
  bool get_dispatched_needs_mask_fuzz() const;
  uint64_t load_dispatched_generation(
      std::memory_order order = std::memory_order_acquire) const;

  void set_dispatch_pending(bool expected_set,
                            uint64_t dispatch_generation_before);
  bool get_dispatch_pending(bool &expected_set,
                            uint64_t &dispatch_generation_before) const;
  bool has_dispatch_pending() const;
  void clear_dispatch_pending();

  void set_control_pending(
      bool expected_set, uint64_t dispatch_generation_before,
      uint8_t selected_mask, uint8_t execution_mask,
      const std::array<uint64_t, kChipNumDpus> &control_generations,
      uint8_t payload = 0u);
  bool get_control_pending(bool &expected_set,
                           uint64_t &dispatch_generation_before,
                           uint8_t &selected_mask,
                           uint8_t &execution_mask) const;
  uint8_t get_control_pending_payload() const;
  uint64_t get_control_pending_generation(uint8_t dpu_idx) const;
  bool has_control_pending() const;
  void clear_control_pending();

  void set_run_state_pending(
      bool expected_set, uint64_t dispatch_generation_before,
      uint8_t selected_mask, uint8_t execution_mask,
      const std::array<uint64_t, kChipNumDpus> &control_generations,
      uint8_t payload = 0u);
  bool get_run_state_pending(bool &expected_set,
                             uint64_t &dispatch_generation_before,
                             uint8_t &selected_mask,
                             uint8_t &execution_mask) const;
  uint8_t get_run_state_pending_payload() const;
  uint64_t get_run_state_pending_control_generation(uint8_t dpu_idx) const;
  bool has_run_state_pending() const;
  void clear_run_state_pending();
};

class CI {
private:
  CIState state_{};

  pim_region_t *region_write_{nullptr};
  pim_region_t *region_read_{nullptr};

public:
  CIState &state();
  const CIState &state() const;

  pim_region_t *get_write_region();
  const pim_region_t *get_write_region() const;
  void set_write_region(pim_region_t *region);

  pim_region_t *get_read_region();
  const pim_region_t *get_read_region() const;
  void set_read_region(pim_region_t *region);

  static inline constexpr uint8_t b(uint64_t x, unsigned shift) {
    return static_cast<uint8_t>((x >> shift) & 0xFFu);
  }

  static inline constexpr uint64_t
  wram_read_word_frame_for_addr(uint32_t address) {
    uint64_t x = 0x700344000000ULL;
    const uint64_t a2 = static_cast<uint64_t>(address) << 2;

    x |= (static_cast<uint64_t>(24u & 31u) << 34);
    x |= ((a2 >> 0) & 0xFULL) << 20;
    x |= ((a2 >> 4) & 0xFULL) << 16;
    x |= ((a2 >> 8) & 0x1ULL) << 15;
    x |= ((a2 >> 9) & 0x1ULL) << 14;
    x |= ((a2 >> 10) & 0x1ULL) << 13;
    x |= ((a2 >> 11) & 0x1ULL) << 12;
    x |= ((a2 >> 12) & 0xFFFULL) << 0;

    return encode_frame_from_x(x, 0x99u);
  }

  static inline constexpr uint64_t wram_write_addr_x(uint32_t address) {
    uint64_t x = 0x7c0044000000ULL;
    const uint64_t a2 = static_cast<uint64_t>(address) << 2;

    x |= (static_cast<uint64_t>(24u & 31u) << 34);
    x |= ((a2 >> 0) & 0xFULL) << 20;
    x |= ((a2 >> 4) & 0x7ULL) << 39;
    x |= ((a2 >> 7) & 0x1ULL) << 24;
    x |= ((a2 >> 8) & 0x1ULL) << 15;
    x |= ((a2 >> 9) & 0x1ULL) << 14;
    x |= ((a2 >> 10) & 0x1ULL) << 13;
    x |= ((a2 >> 11) & 0x1ULL) << 12;
    x |= ((a2 >> 12) & 0xFFFULL) << 0;

    return x;
  }

  static inline constexpr uint64_t
  wram_write_word_structure_for_addr(uint32_t address) {
    const uint64_t x = wram_write_addr_x(address);

    uint64_t cmd = 0x1100000000000000ULL;
    cmd |= (0xFFULL << 0);
    cmd |= (0x03ULL << 8);
    cmd |= ((x >> (0 * 8)) & 0xFFULL) << 16;
    cmd |= ((x >> (1 * 8)) & 0xFFULL) << 24;
    cmd |= ((x >> (3 * 8)) & 0xFFULL) << 32;
    cmd |= (0x99ULL << 40);
    return cmd;
  }

  static inline constexpr uint64_t
  wram_write_word_frame_upper_for_addr(uint32_t address) {
    const uint64_t x = wram_write_addr_x(address);
    uint64_t cmd = 0x3300000000000000ULL;
    cmd |= ((x >> (2 * 8)) & 0xFFULL) << 32;
    cmd |= ((x >> (5 * 8)) & 0xFFULL) << 40;
    cmd |= ((x >> (4 * 8)) & 0xFFULL) << 48;
    return cmd;
  }

  static inline constexpr bool
  decode_wram_write_word_structure(uint64_t cmd, uint16_t &address) {
    constexpr size_t kWramWords = kWRAMSizeBytes / 4u;
    for (uint32_t candidate = 0; candidate < kWramWords; ++candidate) {
      if (wram_write_word_structure_for_addr(candidate) == cmd) {
        address = static_cast<uint16_t>(candidate);
        return true;
      }
    }
    return false;
  }

  static inline constexpr bool
  decode_wram_write_word_frame_low5(uint64_t cmd, uint8_t &low5) {
    const uint64_t key = cmd & 0xFFFFFFFF00000000ULL;
    for (uint32_t candidate = 0; candidate < 32u; ++candidate) {
      if (wram_write_word_frame_upper_for_addr(candidate) == key) {
        low5 = static_cast<uint8_t>(candidate);
        return true;
      }
    }
    return false;
  }

  static inline constexpr bool decode_wram_read_word_frame(uint64_t cmd,
                                                           uint32_t &address) {
    constexpr size_t kWramWords = kWRAMSizeBytes / 4u;
    for (uint32_t candidate = 0; candidate < kWramWords; ++candidate) {
      if (wram_read_word_frame_for_addr(candidate) == cmd) {
        address = candidate;
        return true;
      }
    }
    return false;
  }

  static inline constexpr bool decode_iram_write_structure(uint64_t cmd,
                                                           uint16_t &addr_hi) {
    const uint8_t op = static_cast<uint8_t>((cmd >> 56) & 0xFFu);
    if (op != 0x11u) {
      return false;
    }

    const uint8_t b0 = static_cast<uint8_t>((cmd >> 0) & 0xFFu);
    const uint8_t b1 = static_cast<uint8_t>((cmd >> 8) & 0xFFu);
    const uint8_t b2 = static_cast<uint8_t>((cmd >> 16) & 0xFFu);
    const uint8_t b3 = static_cast<uint8_t>((cmd >> 24) & 0xFFu);
    const uint8_t b4 = static_cast<uint8_t>((cmd >> 32) & 0xFFu);
    const uint8_t b5 = static_cast<uint8_t>((cmd >> 40) & 0xFFu);

    if (b0 != 0xFFu || b1 != 0x00u || b3 != 0x47u || b4 != 0x02u ||
        b5 != 0x00u) {
      return false;
    }

    addr_hi = b2;
    return true;
  }

  static inline constexpr bool
  decode_dma_ctrl_write_frame(uint64_t word, uint8_t &address, uint8_t &data) {
    const uint8_t b0 = b(word, 0);
    const uint8_t b1 = b(word, 8);
    const uint8_t b2 = b(word, 16);
    const uint8_t b3 = b(word, 24);

    if (((b0 & 0xF0u) != 0x60u) || ((b1 & 0xF0u) != 0x60u) ||
        ((b2 & 0xF0u) != 0x60u) || ((b3 & 0xF0u) != 0x60u)) {
      return false;
    }

    address = static_cast<uint8_t>(((b0 & 0x0Fu) << 4) | (b1 & 0x0Fu));
    data = static_cast<uint8_t>(((b2 & 0x0Fu) << 4) | (b3 & 0x0Fu));
    return true;
  }

  static inline constexpr uint64_t ready_result(bool expected_color_set,
                                                uint32_t payload) {
    uint64_t result = 0x000000FF00000000ULL;
    const uint8_t color = expected_color_set ? 0xFFu : 0x00u;
    result |= static_cast<uint64_t>(color) << 48;
    result |= static_cast<uint64_t>(payload);
    return result;
  }

  static inline constexpr uint64_t nop_style_result(bool expected_color_set,
                                                    uint32_t payload) {
    uint64_t result = 0xFF0000FF00000000ULL;
    const uint8_t color = expected_color_set ? 0xFFu : 0x00u;
    result |= static_cast<uint64_t>(color) << 48;
    result |= static_cast<uint64_t>(payload & 0xFFu);
    return result;
  }

  static inline constexpr bool is_software_reset_cmd(uint64_t word) {
    return (word & 0xFFFF00000000FF00ULL) == 0x01FF00000000FF00ULL;
  }
};
