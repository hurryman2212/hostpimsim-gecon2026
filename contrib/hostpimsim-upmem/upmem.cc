#include "upmem.hh"

#include <cstdlib>
#include <cstring>
#include <thread>

#include <sys/mman.h>

static uint8_t get_bound_mask(const UPMEMPIMChip &chip, uint8_t selected_mask) {
  uint8_t bound_mask = 0u;
  for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
    const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
    if ((selected_mask & bit) == 0u) {
      continue;
    }

    const UPMEMDPU &selected_dpu = chip.dpu(dpu_idx);
    if (selected_dpu.mram().get_addr() &&
        selected_dpu.mram().get_size() != 0u) {
      bound_mask |= bit;
    }
  }

  return bound_mask;
}

static bool is_thread_command_word(uint64_t ci_word, uint8_t selected_mask) {
  if (selected_mask == 0u) {
    return false;
  }

  const uint8_t selected = CIState::first_selected_dpu(selected_mask);
  if (selected >= kChipNumDpus) {
    return false;
  }

  // Only classification matters here; decoded outputs are ignored.
  ThreadCmdKind kind{};
  uint8_t thread_id = 0;
  return decode_thread_command(ci_word, kind, thread_id);
}

static constexpr size_t kSparseTransferRequestMinBytes = 64;
static constexpr uint8_t kSparseTransferRequestMinBanks = 2;
static constexpr uint32_t kDefaultPoolThreads = 16u;
static constexpr const char *kPoolThreadsEnv = "HOSTPIMSIM_UPMEM_POOL_THREADS";
static bool make_sparse_bank_transfers(
    const dpu_transfer_mram_abi *tm, size_t size,
    std::array<UPMEMPIMRank::MRAMBankTransferRequest, kChipNumDpus> &requests);

/* DPU handlers */

static void mmio_handler(pim_device_t *dev, pim_region_t *region, size_t offset,
                         uint32_t depth, uint64_t value, void *user_data) {
  (void)dev;
  (void)region;
  (void)offset;
  (void)depth;

  UPMEMPIMChip *chip = static_cast<UPMEMPIMChip *>(user_data);
  if (!chip) {
    return;
  }

  chip->propagate_ci(value);
}
static void dpu_handler(pim_device_t *dev, pim_region_t *region, size_t offset,
                        uint32_t depth, uint64_t value, void *user_data) {
  (void)dev;
  (void)region;
  (void)offset;
  (void)value;

  UPMEMPIMChip *chip = static_cast<UPMEMPIMChip *>(user_data);
  if (!chip || depth == 0u || depth > kChipNumDpus) {
    return;
  }

  const size_t local_dpu_idx = static_cast<size_t>(depth - 1u);
  chip->dpu(local_dpu_idx).control(value);
}

/* Lazy resource allocation implementations */

static inline void initialize_ci_state_defaults(CIState *state) {
  if (!state) {
    return;
  }
  state->reset_protocol_state();
  state->set_selected_mask(0xFFu);
  state->set_group_mask(0, 0xFFu);
  for (size_t group_slot_nr = 1; group_slot_nr < kCiNumGroupSlots;
       ++group_slot_nr) {
    state->set_group_mask(group_slot_nr, 0);
  }
  state->set_committed(0);
  state->store_updated(0, std::memory_order_relaxed);
}

static inline bool protect_region_rw(pim_region_t *region) {
  return region &&
         (pim_region_mprotect(region, PIM_ACCESS_MODE_RW) == PIM_SUCCESS);
}

static constexpr size_t kCtrlCiWordSize = sizeof(uint64_t);
static constexpr size_t kCtrlMmioBase = 0x20000u;
static constexpr size_t kCtrlRwBase = 0x28000u;

static inline pim_region_t *get_or_create_region(pim_device_t *device,
                                                 size_t offset, size_t size,
                                                 pim_region_type_t type) {
  if (!device) {
    return nullptr;
  }
  pim_region_t *region = pim_device_get_region(device, offset);
  if (region) {
    return region;
  }
  return pim_region_create(device, offset, size, type);
}

static bool try_lazy_populate_ci_bank(UPMEMPIMRank *rank, uint8_t ci_idx) {
  if (!rank || ci_idx >= kRankNumBanks) {
    return false;
  }

  pim_device_t *device = rank->get_device();
  if (!device) {
    return false;
  }

  UPMEMPIMChip &chip = rank->chip(ci_idx);
  CI &ci = chip.ci();
  if (ci.get_write_region() && ci.get_read_region()) {
    return true;
  }

  const size_t mmio_offset =
      kCtrlMmioBase + static_cast<size_t>(ci_idx) * kCtrlCiWordSize;
  pim_region_t *mmio_region = get_or_create_region(
      device, mmio_offset, kCtrlCiWordSize, PIM_REGION_CTRL_MMIO);
  if (!protect_region_rw(mmio_region)) {
    return false;
  }

  const size_t rw_offset =
      kCtrlRwBase + static_cast<size_t>(ci_idx) * kCtrlCiWordSize;
  pim_region_t *rw_region = get_or_create_region(
      device, rw_offset, kCtrlCiWordSize, PIM_REGION_CTRL_RW);
  if (!protect_region_rw(rw_region)) {
    return false;
  }
  ci.set_read_region(rw_region);

  const pim_error_t reg_err = pim_register_dpu_handler(
      device, mmio_region, 0, 0, kCtrlCiWordSize, mmio_handler, &chip);
  if (reg_err != PIM_SUCCESS && reg_err != PIM_ERR_ALREADY_EXISTS) {
    return false;
  }
  for (size_t dpu_local = 0; dpu_local < kChipNumDpus; ++dpu_local) {
    const pim_error_t worker_err = pim_register_dpu_handler(
        device, mmio_region, 0, static_cast<uint32_t>(dpu_local + 1),
        kCtrlCiWordSize, dpu_handler, &chip);
    if (worker_err == PIM_ERR_ALREADY_EXISTS) {
      continue;
    }
    if (worker_err != PIM_SUCCESS) {
      return false;
    }
  }
  ci.set_write_region(mmio_region);

  initialize_ci_state_defaults(&ci.state());

  return true;
}

static constexpr size_t kRankApertureSizeBytes =
    (kMramSizeBytes * kChipNumDpus * kRankNumBanks * 2);
static constexpr size_t kCiColorStateBase =
    kRankApertureSizeBytes - kRankNumBanks;

static uint8_t *rank_ci_color_state(UPMEMPIMRank *rank) {
  if (!rank) {
    return nullptr;
  }

  pim_device_t *device = rank->get_device();
  if (!device) {
    return nullptr;
  }

  pim_region_t *region = pim_device_get_region(device, kCiColorStateBase);
  if (!region) {
    return nullptr;
  }
  if (pim_region_get_size(region) < kRankNumBanks) {
    return nullptr;
  }
  return static_cast<uint8_t *>(pim_region_get_ptr(region));
}

static bool try_lazy_populate_ci_color(UPMEMPIMRank *rank) {
  if (!rank) {
    return false;
  }
  pim_device_t *device = rank->get_device();
  if (!device) {
    return false;
  }

  bool created = false;
  pim_region_t *color_region = pim_device_get_region(device, kCiColorStateBase);
  if (!color_region) {
    color_region = pim_region_create(device, kCiColorStateBase, kRankNumBanks,
                                     PIM_REGION_RAM);
    created = true;
  }
  if (!protect_region_rw(color_region)) {
    return false;
  }
  if (created) {
    std::memset(pim_region_get_ptr(color_region), 0, kRankNumBanks);
  }
  return true;
}

static uint32_t configured_pool_threads() {
  const char *value = std::getenv(kPoolThreadsEnv);
  if (value && value[0] != '\0') {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end != value && *end == '\0' && parsed <= kRankNumDpus) {
      return static_cast<uint32_t>(parsed);
    }
  }

  uint32_t target_threads = pim_get_hardware_concurrency();
  if (target_threads == 0u || target_threads > kDefaultPoolThreads) {
    target_threads = kDefaultPoolThreads;
  }
  return target_threads;
}

static bool lazy_populate_thread(pim_device_t *device) {
  if (!device) {
    return false;
  }

  uint32_t current_threads = pim_get_max_active_dpus(device);
  if (current_threads == 0) {
    const uint32_t target_threads = configured_pool_threads();
    if (target_threads == 0u) {
      return true;
    }
    if (pim_pool_start(device, target_threads) != PIM_SUCCESS) {
      return false;
    }
    current_threads = pim_get_max_active_dpus(device);
    if (current_threads == 0) {
      return false;
    }
  }

  return true;
}

static bool decode_dpu_global_idx(size_t dpu_global_idx, uint8_t &ci_idx,
                                  uint8_t &dpu_local_idx) {
  if (dpu_global_idx >= kRankNumDpus) {
    return false;
  }

  // kernel/libdpu ordering: dpu_global = dpu_local * nr_ci + ci.
  ci_idx = static_cast<uint8_t>(dpu_global_idx % kRankNumBanks);
  dpu_local_idx = static_cast<uint8_t>(dpu_global_idx / kRankNumBanks);
  return dpu_local_idx < kChipNumDpus;
}

static bool try_lazy_populate_dpu(UPMEMPIMRank *rank, size_t dpu_global_idx) {
  if (!rank || dpu_global_idx >= kRankNumDpus) {
    return false;
  }

  pim_device_t *device = rank->get_device();
  if (!device) {
    return false;
  }

  const uint8_t ci_idx = static_cast<uint8_t>(dpu_global_idx % kRankNumBanks);
  if (!try_lazy_populate_ci_bank(rank, ci_idx)) {
    return false;
  }
  if (!try_lazy_populate_ci_color(rank)) {
    return false;
  }

  UPMEMDPU &dpu = rank->dpu_by_global(dpu_global_idx);
  if (dpu.mram().get_addr() && dpu.mram().get_size() > 0) {
    return true;
  }

  if (!dpu.mram().allocate()) {
    return false;
  }

  return lazy_populate_thread(device);
}

/* Class UPMEMPIMChip */

/*
 * Propagate the current CI write state and execute one command dispatch step.
 *
 * This is the authoritative command execution path used by MMIO writes:
 *   pim_write() -> mmio_handler(value) -> propagate_ci(value).
 *
 * Per selected DPU, this function calls the lower-level DPU handler (up to
 * kChipNumDpus times) for execution-class commands so side effects stay scoped
 * to the current selected mask.
 */
void UPMEMPIMChip::propagate_ci(uint64_t ci_word) {
  ci_.state().set_committed(ci_word);
  pim_region_t *region = ci_.get_write_region();
  if (!region) {
    return;
  }
  CIState &state = ci_.state();
  bool needs_mask_fuzz = false;

  auto publish = [&](uint32_t payload) {
    state.set_dispatched_result(payload, needs_mask_fuzz);
  };

  auto invoke_selected_handlers = [&](uint8_t selected_mask, bool skip_unbound,
                                      bool direct_control) -> uint8_t {
    uint8_t invoked_mask = 0u;

    for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
      const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
      if ((selected_mask & bit) == 0u) {
        continue;
      }

      UPMEMDPU &chip_dpu = dpu(dpu_idx);
      if (skip_unbound &&
          (!chip_dpu.mram().get_addr() || chip_dpu.mram().get_size() == 0u)) {
        continue;
      }

      chip_dpu.clear_last_control_result();
      if (direct_control) {
        chip_dpu.control(ci_word);
      } else {
        const pim_error_t invoke_err = pim_invoke_dpu_handler(
            region, 0, static_cast<uint32_t>(dpu_idx + 1u), ci_word);
        if (invoke_err != PIM_SUCCESS) {
          continue;
        }
      }
      invoked_mask |= bit;
    }

    return invoked_mask;
  };

  constexpr uint64_t kCiIdentity = 0x01FF000000000000ULL;
  if (ci_word == kCiIdentity) {
    publish(0x00000001u);
    return;
  }

  if (CI::is_software_reset_cmd(ci_word)) {
    state.reset_protocol_state();
    publish(0x00000000u);
    return;
  }

  constexpr uint64_t kCiByteOrder = 0x7777777777777777ULL;
  if (ci_word == kCiByteOrder) {
    publish(0x0F8FCFEFu);
    return;
  }

  const uint8_t opcode = CI::b(ci_word, 56);
  const uint8_t tag = CI::b(ci_word, 48);

  if (opcode == 0x11u) {
    state.set_structure(ci_word);

    uint16_t iram_addr_hi = 0;
    if (CI::decode_iram_write_structure(ci_word, iram_addr_hi)) {
      state.set_iram_write_structure_valid(true);
      state.set_iram_write_addr_hi(iram_addr_hi);
    } else {
      state.set_iram_write_structure_valid(false);
      state.set_iram_write_addr_hi(0u);
    }

    uint16_t wram_addr = 0;
    if (CI::decode_wram_write_word_structure(ci_word, wram_addr)) {
      state.set_wram_write_structure_valid(true);
      state.set_wram_write_addr(wram_addr);
    } else {
      state.set_wram_write_structure_valid(false);
      state.set_wram_write_addr(0u);
    }

    publish(0x000000FFu);
    return;
  }

  if (opcode != 0x33u) {
    publish(0u);
    return;
  }

  const uint8_t b0 = CI::b(ci_word, 0);
  const uint8_t b1 = CI::b(ci_word, 8);
  const uint8_t b2 = CI::b(ci_word, 16);
  const uint8_t b3 = CI::b(ci_word, 24);
  const uint8_t b4 = CI::b(ci_word, 32);
  const uint8_t b5 = CI::b(ci_word, 40);

  const uint8_t selected_mask = state.get_selected_mask();
  uint8_t control_execution_mask = get_bound_mask(*this, selected_mask);
  if (control_execution_mask == 0u) {
    control_execution_mask = selected_mask;
  }

  if (state.is_iram_write_structure_valid()) {
    needs_mask_fuzz = true;

    const uint16_t addr = static_cast<uint16_t>(
        (static_cast<uint16_t>(state.get_iram_write_addr_hi()) << 8) | tag);

    uint64_t inst48 = 0;
    inst48 |= static_cast<uint64_t>(b0) << 0;
    inst48 |= static_cast<uint64_t>(b1) << 8;
    inst48 |= static_cast<uint64_t>(b2) << 16;
    inst48 |= static_cast<uint64_t>(b3) << 24;
    inst48 |= static_cast<uint64_t>(b4) << 32;
    inst48 |= static_cast<uint64_t>(b5) << 40;

    for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
      const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
      if ((selected_mask & bit) == 0u) {
        continue;
      }
      dpu(dpu_idx).iram().write_word(addr, inst48);
    }

    publish(selected_mask);
    return;
  }

  if (state.is_wram_write_structure_valid()) {
    needs_mask_fuzz = true;

    uint16_t addr = state.get_wram_write_addr();
    const uint32_t data = static_cast<uint32_t>(ci_word & 0xFFFFFFFFu);

    uint8_t low5 = 0;
    if (CI::decode_wram_write_word_frame_low5(ci_word, low5)) {
      addr = static_cast<uint16_t>((addr & static_cast<uint16_t>(~0x1Fu)) |
                                   static_cast<uint16_t>(low5));
    }

    for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
      const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
      if ((selected_mask & bit) == 0u) {
        continue;
      }
      dpu(dpu_idx).wram().write_word(addr, data);
    }

    publish(selected_mask);
    return;
  }

  if (tag == 0x00u) {
    if (b0 == 0x08u && b1 == 0xFFu) {
      state.set_selected_mask(0xFFu);
      publish(state.get_selected_mask());
      return;
    }
    if (b0 == 0x0Au) {
      const uint8_t dpu_idx = static_cast<uint8_t>(b1 & 0x7u);
      state.set_selected_mask(static_cast<uint8_t>(1u << dpu_idx));
      publish(state.get_selected_mask());
      return;
    }
    if (b0 == 0x09u) {
      const uint8_t group_id = static_cast<uint8_t>(b1 & 0x7u);
      state.set_selected_mask(state.get_group_mask(group_id));
      publish(state.get_selected_mask());
      return;
    }
    if (b0 == 0x0Cu) {
      const uint8_t group_id = static_cast<uint8_t>(b1 & 0x7u);
      state.set_group_mask(group_id, state.get_selected_mask());
      publish(state.get_selected_mask());
      return;
    }
    if (b0 == 0x16u && b1 == 0x02u) {
      publish(state.get_pc_mode());
      return;
    }
    if (b0 == 0x10u && b1 == 0x02u) {
      const uint8_t selected = CIState::first_selected_dpu(selected_mask);
      if (selected_mask == 0u || selected >= kChipNumDpus) {
        publish(0u);
        return;
      }

      UPMEMDPU &selected_dpu = dpu(selected);
      const uint8_t read_register = state.get_dma_ctrl_read_register();
      if (read_register == 0x02u) {
        publish(selected_dpu.dma_engine().get_dma_mux_status());
      } else {
        publish(selected_dpu.dma_engine().get_dma_ctrl_reg());
      }
      return;
    }

    if (b0 == 0x84u && b1 == 0x02u) {
      publish(invoke_selected_handlers(control_execution_mask, false, false));
      return;
    }

    if (b0 == 0xF2u && b1 == 0x02u) {
      publish(state.exchange_stack_up_mask(0xFFu));
      return;
    }
    if (b0 == 0xF0u && b1 == 0x02u) {
      publish(state.exchange_stack_up_mask(0x00u));
      return;
    }

    if (b1 == 0x02u) {
      switch (b0) {
      case 0x80u:
        publish(0u);
        return;
      case 0x82u:
      case 0xB0u:
      case 0xB4u:
      case 0xF4u:
        publish(0u);
        return;
      default:
        break;
      }

      publish(state.get_selected_mask());
      return;
    }
  }

  if (tag == 0xA7u) {
    uint8_t address = 0;
    uint8_t data = 0;
    if (CI::decode_dma_ctrl_write_frame(ci_word, address, data)) {
      if (address == 0xFFu) {
        state.set_dma_ctrl_read_register(data);
      }

      for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
        UPMEMDPU &chip_dpu = dpu(dpu_idx);
        chip_dpu.dma_engine().set_dma_ctrl_reg(data);
        if (address == 0x80u || address == 0x82u || address == 0x84u) {
          chip_dpu.dma_engine().set_dma_mux_status((data == 0u) ? 0x00u
                                                                : 0x03u);
        }
      }

      publish(0x000000FFu);
      return;
    }

    needs_mask_fuzz = true;
    publish(selected_mask);
    return;
  }

  if (tag == 0xA4u) {
    if (b1 == 0x00u && b2 == 0x00u && b3 == 0x00u && b4 == 0x00u &&
        b5 == 0x00u) {
      state.set_pc_mode(b0);
      publish(0x000000FFu);
      return;
    }

    needs_mask_fuzz = true;
    publish(selected_mask);
    return;
  }

  if (tag == 0xA6u) {
    if (b0 == 0 && b1 == 0 && b2 == 0 && b3 == 0 && b4 == 0 && b5 == 0) {
      publish(0x000000FFu);
      return;
    }

    needs_mask_fuzz = true;
    publish(selected_mask);
    return;
  }

  const bool is_thread_command = is_thread_command_word(ci_word, selected_mask);

  if (is_thread_command) {
    publish(invoke_selected_handlers(control_execution_mask, true, true));
    return;
  }

  if (tag == 0x99u) {
    uint32_t word_addr = 0;
    if (CI::decode_wram_read_word_frame(ci_word, word_addr)) {
      const uint8_t selected = CIState::first_selected_dpu(selected_mask);
      if (selected >= kChipNumDpus || selected_mask == 0u) {
        publish(0u);
        return;
      }

      publish(dpu(selected).wram().read_word(word_addr));
      return;
    }
  }

  needs_mask_fuzz = true;
  publish(selected_mask);
}

CI &UPMEMPIMChip::ci() { return ci_; }

const CI &UPMEMPIMChip::ci() const { return ci_; }

UPMEMDPU &UPMEMPIMChip::dpu(size_t dpu_idx) { return dpus_[dpu_idx]; }

const UPMEMDPU &UPMEMPIMChip::dpu(size_t dpu_idx) const {
  return dpus_[dpu_idx];
}

size_t UPMEMPIMChip::get_nr_dpu() const { return kChipNumDpus; }

/* Class UPMEMPIMRank */

static uint8_t build_active_bank_indices(uint8_t active_bank_mask,
                                         uint8_t active_bank_indices[]) {
  uint8_t active_bank_count = 0u;
  for (uint8_t bank_idx = 0; bank_idx < kRankNumBanks; ++bank_idx) {
    const uint8_t bank_bit = static_cast<uint8_t>(1u << bank_idx);
    if ((active_bank_mask & bank_bit) == 0u) {
      continue;
    }

    active_bank_indices[active_bank_count] = bank_idx;
    ++active_bank_count;
  }

  return active_bank_count;
}

static MRAM *
resolve_bank_group_mrams(UPMEMPIMRank &rank, uint8_t dpu_local_idx,
                         const uint8_t *active_bank_indices,
                         uint8_t active_bank_count, uint32_t offset_in_mram,
                         size_t size,
                         std::array<MRAM *, kRankNumBanks> &mrams) {
  if (!active_bank_indices || active_bank_count == 0u ||
      dpu_local_idx >= kChipNumDpus) {
    errno = EFAULT;
    return nullptr;
  }

  MRAM *anchor = nullptr;
  for (uint8_t active_idx = 0; active_idx < active_bank_count; ++active_idx) {
    const uint8_t bank_idx = active_bank_indices[active_idx];
    if (bank_idx >= kRankNumBanks) {
      errno = EFAULT;
      return nullptr;
    }

    const size_t dpu_global_idx =
        static_cast<size_t>(dpu_local_idx) * kRankNumBanks + bank_idx;
    if (!try_lazy_populate_dpu(&rank, dpu_global_idx)) {
      errno = ENOMEM;
      return nullptr;
    }

    MRAM &mram = rank.dpu_by_global(dpu_global_idx).mram();
    if (!mram.get_addr() || !mram.byte_in_bounds(offset_in_mram, size)) {
      errno = EFAULT;
      return nullptr;
    }

    mrams[bank_idx] = &mram;
    if (!anchor) {
      anchor = &mram;
    }
  }

  return anchor;
}

bool UPMEMPIMRank::host_copy_to_mram_sparse(
    const MRAMBankTransferRequest &request,
    const dpu_transfer_mram_abi &transfer) {
  std::array<const uint8_t *, kRankNumBanks> src_by_bank{};
  for (uint8_t active_idx = 0; active_idx < request.active_bank_count;
       ++active_idx) {
    const uint8_t bank_idx = request.active_bank_indices[active_idx];
    const size_t dpu_global_idx =
        static_cast<size_t>(request.dpu_local_idx) * kRankNumBanks + bank_idx;
    src_by_bank[bank_idx] =
        static_cast<const uint8_t *>(transfer.ptr[dpu_global_idx]);
    if (!src_by_bank[bank_idx]) {
      errno = EFAULT;
      return false;
    }
  }

  return host_copy_to_mram_bank_group(
      request.dpu_local_idx, request.active_bank_mask, transfer.offset_in_mram,
      src_by_bank, static_cast<size_t>(transfer.size));
}

bool UPMEMPIMRank::host_copy_from_mram_sparse(
    const MRAMBankTransferRequest &request,
    const dpu_transfer_mram_abi &transfer) {
  std::array<uint8_t *, kRankNumBanks> dst_by_bank{};
  for (uint8_t active_idx = 0; active_idx < request.active_bank_count;
       ++active_idx) {
    const uint8_t bank_idx = request.active_bank_indices[active_idx];
    const size_t dpu_global_idx =
        static_cast<size_t>(request.dpu_local_idx) * kRankNumBanks + bank_idx;
    dst_by_bank[bank_idx] =
        static_cast<uint8_t *>(transfer.ptr[dpu_global_idx]);
    if (!dst_by_bank[bank_idx]) {
      errno = EFAULT;
      return false;
    }
  }

  return host_copy_from_mram_bank_group(
      request.dpu_local_idx, request.active_bank_mask, transfer.offset_in_mram,
      dst_by_bank, static_cast<size_t>(transfer.size));
}

bool UPMEMPIMRank::host_copy_to_mram_bank_group(
    uint8_t dpu_local_idx, uint8_t active_bank_mask, uint32_t offset_in_mram,
    const std::array<const uint8_t *, kRankNumBanks> &src_by_bank,
    size_t size) {
  if (active_bank_mask == 0u) {
    errno = EFAULT;
    return false;
  }
  if (size == 0u) {
    return true;
  }

  uint8_t active_bank_indices[kRankNumBanks]{};
  const uint8_t active_bank_count =
      build_active_bank_indices(active_bank_mask, active_bank_indices);

  std::array<MRAM *, kRankNumBanks> mrams{};
  if (!resolve_bank_group_mrams(*this, dpu_local_idx, active_bank_indices,
                                active_bank_count, offset_in_mram, size,
                                mrams)) {
    return false;
  }

  for (uint8_t active_idx = 0; active_idx < active_bank_count; ++active_idx) {
    const uint8_t bank_idx = active_bank_indices[active_idx];
    const uint8_t *src = src_by_bank[bank_idx];
    if (!src) {
      errno = EFAULT;
      return false;
    }
    std::memcpy(mrams[bank_idx]->get_addr() + offset_in_mram, src, size);
  }

  return true;
}

bool UPMEMPIMRank::host_copy_from_mram_bank_group(
    uint8_t dpu_local_idx, uint8_t active_bank_mask, uint32_t offset_in_mram,
    std::array<uint8_t *, kRankNumBanks> &dst_by_bank, size_t size) {
  if (active_bank_mask == 0u) {
    errno = EFAULT;
    return false;
  }
  if (size == 0u) {
    return true;
  }

  uint8_t active_bank_indices[kRankNumBanks]{};
  const uint8_t active_bank_count =
      build_active_bank_indices(active_bank_mask, active_bank_indices);

  std::array<MRAM *, kRankNumBanks> mrams{};
  if (!resolve_bank_group_mrams(*this, dpu_local_idx, active_bank_indices,
                                active_bank_count, offset_in_mram, size,
                                mrams)) {
    return false;
  }

  for (uint8_t active_idx = 0; active_idx < active_bank_count; ++active_idx) {
    const uint8_t bank_idx = active_bank_indices[active_idx];
    uint8_t *dst = dst_by_bank[bank_idx];
    if (!dst) {
      errno = EFAULT;
      return false;
    }
    std::memcpy(dst, mrams[bank_idx]->get_addr() + offset_in_mram, size);
  }

  return true;
}

static bool transfer_in_bounds(size_t mram_size,
                               const dpu_transfer_mram_abi *tm) {
  if (!tm) {
    return false;
  }

  const size_t off = static_cast<size_t>(tm->offset_in_mram);
  const size_t size = static_cast<size_t>(tm->size);
  if (off > mram_size) {
    return false;
  }
  if (size > (mram_size - off)) {
    return false;
  }
  return true;
}

long UPMEMPIMRank::host_copy_to_rank(const dpu_transfer_mram_abi *tm) {
  if (!transfer_in_bounds(kMramSizeBytes, tm)) {
    errno = EINVAL;
    return -1;
  }

  const size_t off = static_cast<size_t>(tm->offset_in_mram);
  const size_t size = static_cast<size_t>(tm->size);
  if (size == 0) {
    return 0;
  }

  std::array<MRAMBankTransferRequest, kChipNumDpus> requests{};
  std::array<bool, kRankNumDpus> copied_by_sparse_request{};
  if (make_sparse_bank_transfers(tm, size, requests)) {
    for (const MRAMBankTransferRequest &request : requests) {
      if (request.active_bank_count < kSparseTransferRequestMinBanks) {
        continue;
      }

      for (uint8_t active_idx = 0; active_idx < request.active_bank_count;
           ++active_idx) {
        const uint8_t bank_idx = request.active_bank_indices[active_idx];
        const size_t dpu_global_idx =
            static_cast<size_t>(request.dpu_local_idx) * kRankNumBanks +
            bank_idx;
        copied_by_sparse_request[dpu_global_idx] = true;
      }

      if (!host_copy_to_mram_sparse(request, *tm)) {
        return -1;
      }
    }
  }

  for (size_t dpu_global_idx = 0; dpu_global_idx < kRankNumDpus;
       ++dpu_global_idx) {
    if (!tm->ptr[dpu_global_idx] || copied_by_sparse_request[dpu_global_idx]) {
      continue;
    }

    if (!try_lazy_populate_dpu(this, dpu_global_idx)) {
      errno = ENOMEM;
      return -1;
    }

    UPMEMDPU &dpu = dpu_by_global(dpu_global_idx);
    MRAM &mram = dpu.mram();
    if (!mram.host_copy_to_mram(
            static_cast<uint32_t>(off),
            static_cast<const uint8_t *>(tm->ptr[dpu_global_idx]), size)) {
      errno = EFAULT;
      return -1;
    }
  }

  return 0;
}

long UPMEMPIMRank::host_copy_from_rank(const dpu_transfer_mram_abi *tm) {
  if (!transfer_in_bounds(kMramSizeBytes, tm)) {
    errno = EINVAL;
    return -1;
  }

  const size_t off = static_cast<size_t>(tm->offset_in_mram);
  const size_t size = static_cast<size_t>(tm->size);
  if (size == 0) {
    return 0;
  }

  std::array<MRAMBankTransferRequest, kChipNumDpus> requests{};
  std::array<bool, kRankNumDpus> copied_by_sparse_request{};
  if (make_sparse_bank_transfers(tm, size, requests)) {
    for (const MRAMBankTransferRequest &request : requests) {
      if (request.active_bank_count < kSparseTransferRequestMinBanks) {
        continue;
      }

      for (uint8_t active_idx = 0; active_idx < request.active_bank_count;
           ++active_idx) {
        const uint8_t bank_idx = request.active_bank_indices[active_idx];
        const size_t dpu_global_idx =
            static_cast<size_t>(request.dpu_local_idx) * kRankNumBanks +
            bank_idx;
        copied_by_sparse_request[dpu_global_idx] = true;
      }

      if (!host_copy_from_mram_sparse(request, *tm)) {
        return -1;
      }
    }
  }

  for (size_t dpu_global_idx = 0; dpu_global_idx < kRankNumDpus;
       ++dpu_global_idx) {
    if (!tm->ptr[dpu_global_idx] || copied_by_sparse_request[dpu_global_idx]) {
      continue;
    }

    if (!try_lazy_populate_dpu(this, dpu_global_idx)) {
      errno = ENOMEM;
      return -1;
    }

    UPMEMDPU &dpu = dpu_by_global(dpu_global_idx);
    MRAM &mram = dpu.mram();
    if (!mram.host_copy_from_mram(
            static_cast<uint8_t *>(tm->ptr[dpu_global_idx]),
            static_cast<uint32_t>(off), size)) {
      errno = EFAULT;
      return -1;
    }
  }

  return 0;
}

UPMEMPIMChip &UPMEMPIMRank::chip(size_t bank_idx) { return chips_[bank_idx]; }

const UPMEMPIMChip &UPMEMPIMRank::chip(size_t bank_idx) const {
  return chips_[bank_idx];
}

UPMEMDPU &UPMEMPIMRank::dpu_by_global(size_t dpu_global_idx) {
  const size_t bank_idx = dpu_global_idx % kRankNumBanks;
  const size_t dpu_idx = dpu_global_idx / kRankNumBanks;
  return chips_[bank_idx].dpu(dpu_idx);
}

const UPMEMDPU &UPMEMPIMRank::dpu_by_global(size_t dpu_global_idx) const {
  const size_t bank_idx = dpu_global_idx % kRankNumBanks;
  const size_t dpu_idx = dpu_global_idx / kRankNumBanks;
  return chips_[bank_idx].dpu(dpu_idx);
}

pim_device_t *UPMEMPIMRank::get_device() { return device_; }

const pim_device_t *UPMEMPIMRank::get_device() const { return device_; }

void UPMEMPIMRank::set_device(pim_device_t *device) { device_ = device; }

/* Rank management implementations */

UPMEMPIMRank *upmem_create_rank(pim_vdev_t *vdev) {
  (void)vdev;

  auto *rank = new (std::nothrow) UPMEMPIMRank();
  if (!rank) {
    errno = ENOMEM;
    return nullptr;
  }

  void *device_backing = mmap(nullptr, kRankApertureSizeBytes, PROT_NONE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (device_backing == MAP_FAILED) {
    delete rank;
    errno = ENOMEM;
    return nullptr;
  }

  pim_device_t *device =
      pim_device_init(device_backing, kRankApertureSizeBytes);
  if (!device) {
    munmap(device_backing, kRankApertureSizeBytes);
    delete rank;
    errno = ENOMEM;
    return nullptr;
  }
  rank->set_device(device);

  return rank;
}

void upmem_destroy_rank(UPMEMPIMRank *rank) {
  if (!rank) {
    return;
  }

  pim_device_t *device = rank->get_device();
  void *device_backing = nullptr;
  size_t device_backing_size = 0;
  if (device) {
    (void)pim_device_barrier(device);
    device_backing = pim_device_get_ptr(device);
    device_backing_size = pim_device_get_size(device);
    rank->set_device(nullptr);
  }

  if (device) {
    pim_device_deinit(device);
  }

  delete rank;

  if (device_backing && device_backing_size > 0) {
    munmap(device_backing, device_backing_size);
  }
}

/* ioctl() implementations */

static bool make_sparse_bank_transfers(
    const dpu_transfer_mram_abi *tm, size_t size,
    std::array<UPMEMPIMRank::MRAMBankTransferRequest, kChipNumDpus> &requests) {
  if (!tm || size < kSparseTransferRequestMinBytes) {
    return false;
  }

  bool have_multi_bank_request = false;
  for (size_t dpu_global_idx = 0; dpu_global_idx < kRankNumDpus;
       ++dpu_global_idx) {
    if (!tm->ptr[dpu_global_idx]) {
      continue;
    }

    uint8_t bank_idx = 0u;
    uint8_t dpu_local_idx = 0u;
    if (!decode_dpu_global_idx(dpu_global_idx, bank_idx, dpu_local_idx)) {
      return false;
    }

    UPMEMPIMRank::MRAMBankTransferRequest &request = requests[dpu_local_idx];
    const uint8_t bank_bit = static_cast<uint8_t>(1u << bank_idx);
    if ((request.active_bank_mask & bank_bit) != 0u ||
        request.active_bank_count >= kRankNumBanks) {
      return false;
    }

    request.dpu_local_idx = dpu_local_idx;
    request.active_bank_indices[request.active_bank_count] = bank_idx;
    ++request.active_bank_count;
    request.active_bank_mask |= bank_bit;
    if (request.active_bank_count >= kSparseTransferRequestMinBanks) {
      have_multi_bank_request = true;
    }
  }

  return have_multi_bank_request;
}

long upmem_ioctl_write_to_rank(UPMEMPIMRank *rank, unsigned long arg) {
  if (!rank || arg == 0) {
    errno = EINVAL;
    return -1;
  }

  return rank->host_copy_to_rank(
      reinterpret_cast<const dpu_transfer_mram_abi *>(arg));
}

long upmem_ioctl_read_from_rank(UPMEMPIMRank *rank, unsigned long arg) {
  if (!rank || arg == 0) {
    errno = EINVAL;
    return -1;
  }

  return rank->host_copy_from_rank(
      reinterpret_cast<const dpu_transfer_mram_abi *>(arg));
}

static inline bool is_run_state_read_command(uint64_t ci_word) {
  return CI::b(ci_word, 56) == 0x33u && CI::b(ci_word, 48) == 0x00u &&
         CI::b(ci_word, 0) == 0x84u && CI::b(ci_word, 8) == 0x02u;
}

static inline bool has_single_selected_dpu(uint8_t selected_mask) {
  return selected_mask != 0u &&
         (selected_mask & static_cast<uint8_t>(selected_mask - 1u)) == 0u;
}

static inline bool has_active_run_selected(UPMEMPIMRank *rank, size_t bank_idx,
                                           uint8_t selected_mask) {
  if (!rank || selected_mask == 0u) {
    return false;
  }

  for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
    const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
    if ((selected_mask & bit) == 0u) {
      continue;
    }

    if (rank->chip(bank_idx).dpu(dpu_idx).pipeline().get_run_bits() != 0u) {
      return true;
    }
  }

  return false;
}

static inline uint64_t run_state_pending_nop(bool expected_set) {
  return CI::nop_style_result(!expected_set, 0u);
}

static bool write_ci_rw(CI &ci, uint64_t word) {
  pim_region_t *rw_region = ci.get_read_region();
  if (!rw_region) {
    errno = ENODEV;
    return false;
  }

  if (pim_region_get_size(rw_region) < sizeof(word) ||
      !pim_region_get_ptr(rw_region)) {
    errno = EFAULT;
    return false;
  }

  std::memcpy(pim_region_get_ptr(rw_region), &word, sizeof(word));
  ci.state().store_updated(word, std::memory_order_release);
  return true;
}

static constexpr size_t kCiUpdateControlWaitSpinBudget = 32768u;
static constexpr uint64_t CI_EMPTY_CMD = 0x0000000000000000ULL;

long upmem_ioctl_commit_commands(UPMEMPIMRank *rank, unsigned long arg) {
  if (!rank || arg == 0) {
    errno = EINVAL;
    return -1;
  }

  uint64_t input[kRankNumBanks]{};
  std::memcpy(input, reinterpret_cast<const void *>(arg), sizeof(input));

  struct CommitBankContext {
    bool active{false};
    size_t bank_idx{0};
    CI *ci{nullptr};
    CIState *state{nullptr};

    uint64_t ci_word{CI_EMPTY_CMD};
    uint64_t dispatch_generation_before{0};
    uint8_t selected_mask{0u};
    uint8_t control_execution_mask{0u};
    uint8_t opcode{0u};
    uint8_t tag{0u};

    bool is_run_state_command{false};
    bool wrote_run_state_sync{false};
    bool is_thread_command{false};
    bool wait_for_control_results{false};
    bool defer_run_state{false};
    bool retry_pending_run_state{false};
    bool pending_expected_set{false};
    uint64_t pending_dispatch_generation_before{0};
    uint8_t run_state_pending_mask{0u};
    std::array<uint64_t, kChipNumDpus> run_state_pending_generations{};

    UPMEMDPU *pending_dpus[kChipNumDpus]{};
    uint8_t pending_indices[kChipNumDpus]{};
    uint8_t pending_bits[kChipNumDpus]{};
    uint64_t pending_generations[kChipNumDpus]{};
    size_t pending_count{0};
  };

  std::array<CommitBankContext, kRankNumBanks> bank_ctx{};

  size_t run_state_submit_count = 0;
  for (size_t bank_idx = 0; bank_idx < kRankNumBanks; ++bank_idx) {
    if (is_run_state_read_command(input[bank_idx])) {
      ++run_state_submit_count;
    }
  }

  uint8_t *ci_color_state = nullptr;
  for (size_t bank_idx = 0; bank_idx < kRankNumBanks; ++bank_idx) {
    const size_t marker_global_idx = bank_idx;
    if (!try_lazy_populate_dpu(rank, marker_global_idx)) {
      errno = ENOMEM;
      return -1;
    }

    if (!ci_color_state) {
      ci_color_state = rank_ci_color_state(rank);
      if (!ci_color_state) {
        errno = ENODEV;
        return -1;
      }
    }

    CI &ci = rank->chip(bank_idx).ci();
    CIState &state = ci.state();
    CommitBankContext &ctx = bank_ctx[bank_idx];
    ctx.active = true;
    ctx.bank_idx = bank_idx;
    ctx.ci = &ci;
    ctx.state = &state;

    ctx.ci_word = input[bank_idx];
    state.set_committed(ctx.ci_word);
    ctx.dispatch_generation_before =
        state.load_dispatched_generation(std::memory_order_acquire);

    ctx.selected_mask = state.get_selected_mask();
    ctx.control_execution_mask =
        get_bound_mask(rank->chip(bank_idx), ctx.selected_mask);
    if (ctx.control_execution_mask == 0u) {
      ctx.control_execution_mask = ctx.selected_mask;
    }
    ctx.opcode = CI::b(ctx.ci_word, 56);
    ctx.tag = CI::b(ctx.ci_word, 48);

    ctx.is_run_state_command = is_run_state_read_command(ctx.ci_word);

    bool control_pending_expected_set = false;
    uint64_t control_pending_before = 0;
    uint8_t control_pending_mask = 0u;
    uint8_t control_execution_mask = 0u;
    if (!ctx.is_run_state_command &&
        state.get_control_pending(control_pending_expected_set,
                                  control_pending_before, control_pending_mask,
                                  control_execution_mask)) {
      (void)control_pending_before;
      (void)control_pending_mask;
      (void)control_execution_mask;
      const uint64_t pending_word =
          run_state_pending_nop(control_pending_expected_set);
      if (!write_ci_rw(ci, pending_word)) {
        return -1;
      }
      ctx.active = false;
      continue;
    }

    bool dispatch_pending_expected_set = false;
    uint64_t dispatch_pending_before = 0;
    if (!ctx.is_run_state_command &&
        state.get_dispatch_pending(dispatch_pending_expected_set,
                                   dispatch_pending_before)) {
      const uint64_t dispatch_now =
          state.load_dispatched_generation(std::memory_order_acquire);
      if (dispatch_now == dispatch_pending_before) {
        const uint64_t pending_word =
            run_state_pending_nop(dispatch_pending_expected_set);
        if (!write_ci_rw(ci, pending_word)) {
          return -1;
        }
        ctx.active = false;
        continue;
      }

      state.clear_dispatch_pending();
    }

    if (ctx.is_run_state_command) {
      uint8_t pending_selected_mask = 0u;
      uint8_t pending_execution_mask = 0u;
      if (state.get_run_state_pending(
              ctx.pending_expected_set, ctx.pending_dispatch_generation_before,
              pending_selected_mask, pending_execution_mask)) {
        (void)ctx.pending_dispatch_generation_before;
        ctx.retry_pending_run_state = true;
        ctx.defer_run_state = true;
        if (pending_execution_mask != 0u) {
          ctx.run_state_pending_mask = pending_execution_mask;
        } else if (pending_selected_mask != 0u) {
          ctx.run_state_pending_mask =
              get_bound_mask(rank->chip(bank_idx), pending_selected_mask);
          if (ctx.run_state_pending_mask == 0u) {
            ctx.run_state_pending_mask = pending_selected_mask;
          }
        } else {
          ctx.run_state_pending_mask = ctx.control_execution_mask;
        }
      } else if (has_single_selected_dpu(ctx.control_execution_mask) &&
                 has_active_run_selected(rank, bank_idx,
                                         ctx.control_execution_mask)) {
        ctx.defer_run_state = true;
        ctx.run_state_pending_mask = ctx.control_execution_mask;
      } else {
        ctx.wait_for_control_results = true;
      }

      if (ctx.defer_run_state) {
        for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
          const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
          if ((ctx.run_state_pending_mask & bit) == 0u) {
            continue;
          }

          ctx.run_state_pending_generations[dpu_idx] =
              rank->chip(bank_idx).dpu(dpu_idx).get_last_control_generation();
        }
      }
    } else {
      state.clear_run_state_pending();
    }

    if (!ctx.is_run_state_command && ctx.opcode == 0x33u && ctx.tag == 0x98u) {
      ctx.is_thread_command =
          is_thread_command_word(ctx.ci_word, ctx.selected_mask);
    }

    if (ctx.is_thread_command) {
      ctx.wait_for_control_results = true;
    }

    if (ctx.wait_for_control_results) {
      const bool skip_unbound = ctx.is_thread_command;
      const uint8_t monitored_selected_mask =
          (ctx.is_thread_command || ctx.is_run_state_command)
              ? ctx.control_execution_mask
              : ctx.selected_mask;
      for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
        const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
        if ((monitored_selected_mask & bit) == 0u) {
          continue;
        }

        UPMEMDPU &chip_dpu = rank->chip(bank_idx).dpu(dpu_idx);
        if (skip_unbound &&
            (!chip_dpu.mram().get_addr() || chip_dpu.mram().get_size() == 0u)) {
          continue;
        }

        ctx.pending_dpus[ctx.pending_count] = &chip_dpu;
        ctx.pending_indices[ctx.pending_count] = dpu_idx;
        ctx.pending_bits[ctx.pending_count] = bit;
        ctx.pending_generations[ctx.pending_count] =
            chip_dpu.get_last_control_generation();
        ++ctx.pending_count;
      }
    }

    pim_region_t *mmio_region = ci.get_write_region();
    if (!mmio_region) {
      errno = ENODEV;
      return -1;
    }

    ctx.wrote_run_state_sync =
        ctx.is_run_state_command && run_state_submit_count <= 1u;

    rank->chip(bank_idx).propagate_ci(ctx.ci_word);
    if (ctx.wrote_run_state_sync) {
      const pim_error_t barrier_err = pim_region_barrier(mmio_region);
      if (barrier_err != PIM_SUCCESS) {
        errno = EFAULT;
        return -1;
      }
    }
  }

  for (size_t bank_idx = 0; bank_idx < kRankNumBanks; ++bank_idx) {
    CommitBankContext &ctx = bank_ctx[bank_idx];
    if (!ctx.active || !ctx.ci || !ctx.state) {
      continue;
    }

    CI &ci = *ctx.ci;
    CIState &state = *ctx.state;

    if (ctx.is_run_state_command && !ctx.wrote_run_state_sync) {
      pim_region_t *mmio_region = ci.get_write_region();
      if (!mmio_region) {
        errno = ENODEV;
        return -1;
      }

      const pim_error_t barrier_err = pim_region_barrier(mmio_region);
      if (barrier_err != PIM_SUCCESS) {
        errno = EFAULT;
        return -1;
      }
    }

    uint64_t updated_word = CI_EMPTY_CMD;
    bool advance_color = false;
    if (ctx.ci_word != CI_EMPTY_CMD) {
      const bool expected_set = (ci_color_state[bank_idx] & 0x1u) != 0u;

      constexpr uint64_t CI_BYTE_ORDER_CMD = 0x7777777777777777ULL;
      constexpr uint64_t CI_BYTE_ORDER_UPDATED = 0x000103FF0F8FCFEFULL;

      if (ctx.ci_word == CI_BYTE_ORDER_CMD) {
        updated_word = CI_BYTE_ORDER_UPDATED;
        advance_color = true;
        state.clear_run_state_pending();
        state.clear_dispatch_pending();
        state.clear_control_pending();
      } else if (ctx.defer_run_state) {
        const bool pending_expected = ctx.retry_pending_run_state
                                          ? ctx.pending_expected_set
                                          : expected_set;

        state.set_run_state_pending(
            pending_expected, ctx.dispatch_generation_before, ctx.selected_mask,
            ctx.run_state_pending_mask, ctx.run_state_pending_generations);
        state.clear_dispatch_pending();
        state.clear_control_pending();

        updated_word = run_state_pending_nop(pending_expected);
        if (!ctx.retry_pending_run_state) {
          advance_color = true;
        }
      } else if (ctx.wait_for_control_results) {
        constexpr size_t kCommitControlWaitSpinBudget = 32768u;
        uint8_t monitored_mask = 0u;
        uint8_t unfinished_mask = 0u;
        uint8_t payload = 0u;
        bool all_ready = true;

        for (size_t idx = 0; idx < ctx.pending_count; ++idx) {
          UPMEMDPU *chip_dpu = ctx.pending_dpus[idx];
          if (!chip_dpu) {
            continue;
          }

          monitored_mask |= ctx.pending_bits[idx];
          size_t spins = 0;
          while (chip_dpu->get_last_control_generation() ==
                     ctx.pending_generations[idx] &&
                 spins < kCommitControlWaitSpinBudget) {
            std::this_thread::yield();
            ++spins;
          }

          if (chip_dpu->get_last_control_generation() ==
              ctx.pending_generations[idx]) {
            all_ready = false;
            unfinished_mask |= ctx.pending_bits[idx];
            continue;
          }

          if (chip_dpu->get_last_control_result() != 0u) {
            payload |= ctx.pending_bits[idx];
          }
        }

        if (all_ready || monitored_mask == 0u) {
          updated_word = CI::ready_result(expected_set, payload);
          state.clear_run_state_pending();
          state.clear_dispatch_pending();
          state.clear_control_pending();
        } else {
          std::array<uint64_t, kChipNumDpus> pending_control_generations{};
          for (size_t idx = 0; idx < ctx.pending_count; ++idx) {
            const uint8_t dpu_idx = ctx.pending_indices[idx];
            if (dpu_idx >= kChipNumDpus) {
              continue;
            }
            if ((unfinished_mask & ctx.pending_bits[idx]) == 0u) {
              continue;
            }
            pending_control_generations[dpu_idx] = ctx.pending_generations[idx];
          }

          if (ctx.is_thread_command) {
            state.set_control_pending(
                expected_set, ctx.dispatch_generation_before, ctx.selected_mask,
                unfinished_mask, pending_control_generations, payload);
            state.clear_run_state_pending();
          } else {
            state.set_run_state_pending(
                expected_set, ctx.dispatch_generation_before, ctx.selected_mask,
                unfinished_mask, pending_control_generations, payload);
            state.clear_control_pending();
          }
          state.clear_dispatch_pending();
          updated_word = run_state_pending_nop(expected_set);
        }

        advance_color = true;
      } else {
        constexpr size_t kCommitDispatchWaitSpinBudget = 128u;
        size_t spins = 0;
        while (state.load_dispatched_generation(std::memory_order_acquire) ==
                   ctx.dispatch_generation_before &&
               spins < kCommitDispatchWaitSpinBudget) {
          std::this_thread::yield();
          ++spins;
        }

        if (state.load_dispatched_generation(std::memory_order_acquire) ==
            ctx.dispatch_generation_before) {
          state.set_dispatch_pending(expected_set,
                                     ctx.dispatch_generation_before);
          updated_word = run_state_pending_nop(expected_set);
        } else {
          const bool needs_mask_fuzz = state.get_dispatched_needs_mask_fuzz();
          const uint32_t payload = state.get_dispatched_payload();

          updated_word = needs_mask_fuzz
                             ? CI::nop_style_result(expected_set, payload)
                             : CI::ready_result(expected_set, payload);
          state.clear_dispatch_pending();
          state.clear_control_pending();
        }

        advance_color = true;
        state.clear_run_state_pending();
      }
    } else {
      state.clear_run_state_pending();
      state.clear_dispatch_pending();
      state.clear_control_pending();
    }

    if (advance_color) {
      uint8_t next_color =
          static_cast<uint8_t>((ci_color_state[bank_idx] ^ 1u) & 1u);
      if (CI::is_software_reset_cmd(ctx.ci_word)) {
        next_color = 0u;
      }
      ci_color_state[bank_idx] = next_color;
    }

    if (!write_ci_rw(ci, updated_word)) {
      return -1;
    }
  }

  return 0;
}

long upmem_ioctl_update_commands(UPMEMPIMRank *rank, unsigned long arg) {
  if (!rank || arg == 0) {
    errno = EINVAL;
    return -1;
  }

  uint64_t output[kRankNumBanks]{};
  for (size_t bank_idx = 0; bank_idx < kRankNumBanks; ++bank_idx) {
    UPMEMPIMChip &chip = rank->chip(bank_idx);
    CI &ci = chip.ci();
    CIState &state = ci.state();

    bool expected_set = false;
    uint64_t dispatch_generation_before = 0;
    uint8_t selected_mask = 0u;
    uint8_t execution_mask = 0u;
    if (state.get_run_state_pending(expected_set, dispatch_generation_before,
                                    selected_mask, execution_mask)) {
      const uint64_t pending_nop = run_state_pending_nop(expected_set);
      const uint64_t dispatch_now =
          state.load_dispatched_generation(std::memory_order_acquire);

      if (dispatch_now == dispatch_generation_before) {
        output[bank_idx] = pending_nop;
        continue;
      }

      const uint8_t observed_mask = execution_mask;

      uint8_t payload = state.get_run_state_pending_payload();
      bool all_ready = true;
      uint8_t unfinished_mask = 0u;
      for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
        const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
        if ((observed_mask & bit) == 0u) {
          continue;
        }

        const uint64_t before_control_generation =
            state.get_run_state_pending_control_generation(dpu_idx);
        uint64_t now_control_generation =
            chip.dpu(dpu_idx).get_last_control_generation();

        size_t spins = 0;
        while (now_control_generation == before_control_generation &&
               spins < kCiUpdateControlWaitSpinBudget) {
          std::this_thread::yield();
          now_control_generation =
              chip.dpu(dpu_idx).get_last_control_generation();
          ++spins;
        }

        if (now_control_generation == before_control_generation) {
          all_ready = false;
          unfinished_mask |= bit;
          continue;
        }

        if (chip.dpu(dpu_idx).get_last_control_result() != 0u) {
          payload |= bit;
        }
      }

      if (!all_ready) {
        output[bank_idx] = pending_nop;
        std::array<uint64_t, kChipNumDpus> refreshed_generations{};
        uint8_t refreshed_unfinished_mask = unfinished_mask;
        for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
          const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
          if ((unfinished_mask & bit) == 0u) {
            continue;
          }

          const uint64_t before_control_generation =
              state.get_run_state_pending_control_generation(dpu_idx);
          const uint64_t now_control_generation =
              chip.dpu(dpu_idx).get_last_control_generation();
          if (now_control_generation != before_control_generation) {
            refreshed_unfinished_mask &= static_cast<uint8_t>(~bit);
            if (chip.dpu(dpu_idx).get_last_control_result() != 0u) {
              payload |= bit;
            }
            continue;
          }

          refreshed_generations[dpu_idx] = now_control_generation;
        }

        if (refreshed_unfinished_mask == 0u) {
          const uint64_t ready_word = CI::ready_result(expected_set, payload);
          state.clear_run_state_pending();
          state.clear_dispatch_pending();
          state.clear_control_pending();

          if (!write_ci_rw(ci, ready_word)) {
            return -1;
          }

          output[bank_idx] = ready_word;
          continue;
        }

        const uint64_t next_dispatch_before =
            (dispatch_now == 0u) ? 0u : (dispatch_now - 1u);
        state.set_run_state_pending(expected_set, next_dispatch_before,
                                    selected_mask, refreshed_unfinished_mask,
                                    refreshed_generations, payload);
        continue;
      }

      const uint64_t ready_word = CI::ready_result(expected_set, payload);
      state.clear_run_state_pending();
      state.clear_dispatch_pending();
      state.clear_control_pending();

      if (!write_ci_rw(ci, ready_word)) {
        return -1;
      }

      output[bank_idx] = ready_word;
      continue;
    }

    bool control_expected_set = false;
    uint64_t control_dispatch_before = 0;
    uint8_t control_selected_mask = 0u;
    uint8_t control_execution_mask = 0u;
    if (state.get_control_pending(control_expected_set, control_dispatch_before,
                                  control_selected_mask,
                                  control_execution_mask)) {
      const uint64_t pending_nop = run_state_pending_nop(control_expected_set);
      const uint64_t dispatch_now =
          state.load_dispatched_generation(std::memory_order_acquire);
      if (dispatch_now == control_dispatch_before) {
        output[bank_idx] = pending_nop;
        continue;
      }

      const uint8_t observed_mask = control_execution_mask;

      uint8_t payload = state.get_control_pending_payload();
      bool all_ready = true;
      uint8_t unfinished_mask = 0u;
      for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
        const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
        if ((observed_mask & bit) == 0u) {
          continue;
        }

        const uint64_t before_generation =
            state.get_control_pending_generation(dpu_idx);
        const uint64_t now_generation =
            chip.dpu(dpu_idx).get_last_control_generation();

        if (now_generation == before_generation) {
          all_ready = false;
          unfinished_mask |= bit;
          continue;
        }

        if (chip.dpu(dpu_idx).get_last_control_result() != 0u) {
          payload |= bit;
        }
      }

      if (!all_ready) {
        std::array<uint64_t, kChipNumDpus> refreshed_generations{};
        uint8_t refreshed_unfinished_mask = unfinished_mask;
        for (uint8_t dpu_idx = 0; dpu_idx < kChipNumDpus; ++dpu_idx) {
          const uint8_t bit = static_cast<uint8_t>(1u << dpu_idx);
          if ((unfinished_mask & bit) == 0u) {
            continue;
          }

          const uint64_t before_generation =
              state.get_control_pending_generation(dpu_idx);
          const uint64_t now_generation =
              chip.dpu(dpu_idx).get_last_control_generation();
          if (now_generation != before_generation) {
            refreshed_unfinished_mask &= static_cast<uint8_t>(~bit);
            if (chip.dpu(dpu_idx).get_last_control_result() != 0u) {
              payload |= bit;
            }
            continue;
          }

          refreshed_generations[dpu_idx] = now_generation;
        }

        if (refreshed_unfinished_mask == 0u) {
          const uint64_t ready_word =
              CI::ready_result(control_expected_set, payload);
          state.clear_control_pending();
          state.clear_dispatch_pending();

          if (!write_ci_rw(ci, ready_word)) {
            return -1;
          }

          output[bank_idx] = ready_word;
          continue;
        }

        const uint64_t next_dispatch_before =
            (dispatch_now == 0u) ? 0u : (dispatch_now - 1u);
        state.set_control_pending(
            control_expected_set, next_dispatch_before, control_selected_mask,
            refreshed_unfinished_mask, refreshed_generations, payload);
        output[bank_idx] = pending_nop;
        continue;
      }

      const uint64_t ready_word =
          CI::ready_result(control_expected_set, payload);
      state.clear_control_pending();
      state.clear_dispatch_pending();

      if (!write_ci_rw(ci, ready_word)) {
        return -1;
      }

      output[bank_idx] = ready_word;
      continue;
    }

    bool dispatch_expected_set = false;
    uint64_t dispatch_pending_before = 0;
    if (state.get_dispatch_pending(dispatch_expected_set,
                                   dispatch_pending_before)) {
      const uint64_t pending_nop = run_state_pending_nop(dispatch_expected_set);
      const uint64_t dispatch_now =
          state.load_dispatched_generation(std::memory_order_acquire);
      if (dispatch_now == dispatch_pending_before) {
        output[bank_idx] = pending_nop;
        continue;
      }

      const bool needs_mask_fuzz = state.get_dispatched_needs_mask_fuzz();
      const uint32_t payload = state.get_dispatched_payload();
      const uint64_t ready_word =
          needs_mask_fuzz ? CI::nop_style_result(dispatch_expected_set, payload)
                          : CI::ready_result(dispatch_expected_set, payload);

      state.clear_dispatch_pending();

      if (!write_ci_rw(ci, ready_word)) {
        return -1;
      }

      output[bank_idx] = ready_word;
      continue;
    }

    output[bank_idx] = state.load_updated(std::memory_order_acquire);
  }

  std::memcpy(reinterpret_cast<void *>(arg), output, sizeof(output));
  return 0;
}

long upmem_ioctl_debug_mode(UPMEMPIMRank *rank, unsigned long arg) {
  (void)rank;
  (void)arg;
  return 0;
}

struct dpu_slice_target_abi {
  uint32_t type;
  uint32_t value;
};
struct dpu_configuration_slice_info_abi {
  uint64_t byte_order;
  uint64_t structure_value;
  dpu_slice_target_abi slice_target;
  uint8_t host_mux_mram_state;
  uint8_t reserved0[3];
  uint32_t dpus_per_group[kCiNumGroupSlots];
  uint32_t enabled_dpus;
  uint8_t all_dpus_are_enabled;
};
long upmem_ioctl_slice_info(UPMEMPIMRank *rank, unsigned long arg) {
  if (!rank || arg == 0) {
    errno = EINVAL;
    return -1;
  }

  dpu_configuration_slice_info_abi slice_info[kRankNumBanks]{};
  std::memcpy(slice_info, reinterpret_cast<const void *>(arg),
              sizeof(slice_info));

  constexpr uint8_t kAllDpusMask =
      static_cast<uint8_t>((1u << kChipNumDpus) - 1u);
  for (size_t bank_idx = 0; bank_idx < kRankNumBanks; ++bank_idx) {
    uint8_t enabled_mask =
        static_cast<uint8_t>(slice_info[bank_idx].enabled_dpus & kAllDpusMask);
    if (slice_info[bank_idx].all_dpus_are_enabled) {
      enabled_mask = kAllDpusMask;
    }

    CIState &state = rank->chip(bank_idx).ci().state();
    state.set_enabled_mask(enabled_mask);
    state.set_selected_mask(enabled_mask);
    state.set_group_mask(0, enabled_mask);

    for (size_t group_slot_nr = 1; group_slot_nr < kCiNumGroupSlots;
         ++group_slot_nr) {
      const uint8_t group_mask = state.get_group_mask(group_slot_nr);
      state.set_group_mask(group_slot_nr,
                           static_cast<uint8_t>(group_mask & enabled_mask));
    }
  }

  return 0;
}
