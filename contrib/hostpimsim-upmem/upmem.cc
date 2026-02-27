#include "upmem.hh"
#include "hostpimsim.h"

#include <cstdarg>
#include <cstring>
#include <new>

#include <sys/mman.h>

static constexpr uintptr_t kCiHandlerTag = 0x1u;
static constexpr uintptr_t kCiDpuHandlerTag = 0x2u;

static inline void *encode_ci_handler_user_data(uint8_t ci) {
  const uintptr_t raw = (static_cast<uintptr_t>(ci) << 8u) | kCiHandlerTag;
  return reinterpret_cast<void *>(raw);
}

static inline bool decode_ci_handler_user_data(const void *user_data,
                                               uint8_t *ci_out) {
  if (!user_data || !ci_out) {
    return false;
  }

  const uintptr_t raw = reinterpret_cast<uintptr_t>(user_data);
  if ((raw & 0xFFu) != kCiHandlerTag) {
    return false;
  }

  *ci_out = static_cast<uint8_t>((raw >> 8u) & 0xFFu);
  return true;
}

static inline void *encode_ci_dpu_handler_user_data(uint8_t ci,
                                                    uint8_t dpu_local) {
  const uintptr_t raw = (static_cast<uintptr_t>(dpu_local) << 16u) |
                        (static_cast<uintptr_t>(ci) << 8u) | kCiDpuHandlerTag;
  return reinterpret_cast<void *>(raw);
}

static inline bool decode_ci_dpu_handler_user_data(const void *user_data,
                                                   uint8_t *ci_out,
                                                   uint8_t *dpu_local_out) {
  if (!user_data || !ci_out || !dpu_local_out) {
    return false;
  }

  const uintptr_t raw = reinterpret_cast<uintptr_t>(user_data);
  if ((raw & 0xFFu) != kCiDpuHandlerTag) {
    return false;
  }

  *ci_out = static_cast<uint8_t>((raw >> 8u) & 0xFFu);
  *dpu_local_out = static_cast<uint8_t>((raw >> 16u) & 0xFFu);
  return true;
}

static constexpr size_t UPMEM_MAX_DPUS_PER_RANK = upmem_pim_rank::kNumDpus;
static constexpr size_t UPMEM_NB_CI = upmem_pim_rank::kNumChips;
static constexpr size_t UPMEM_NB_DPU_PER_CI = upmem_pim_rank::kNumDpusPerChip;
[[maybe_unused]] static constexpr size_t UPMEM_NB_GROUPS =
    upmem_pim_rank::kNumGroups;
static constexpr size_t UPMEM_MRAM_SIZE = upmem_pim_rank::kMramSize;
static constexpr size_t UPMEM_DAX_REGION_SIZE = upmem_pim_rank::kDaxRegionSize;

/* ------------------------------------------------------------------------- */
/* UPMEM-compatible constants used by libdpuhw/libdpu                        */
/* ------------------------------------------------------------------------- */

struct dpu_transfer_mram_abi {
  void *ptr[UPMEM_MAX_DPUS_PER_RANK];
  uint32_t offset_in_mram;
  uint32_t size;
};

/* slice_info ioctl currently handled as a no-op. */

static constexpr uint64_t CI_EMPTY_CMD = 0x0000000000000000ULL;
static constexpr uint64_t CI_BYTE_ORDER_CMD = 0x7777777777777777ULL;

/* ------------------------------------------------------------------------- */
/* Global simulator rank                                                     */
/* ------------------------------------------------------------------------- */

static thread_local upmem_pim_rank *g_upmem_tls_state = nullptr;

class upmem_state_scope {
public:
  explicit upmem_state_scope(upmem_pim_rank *rank) noexcept
      : prev_(g_upmem_tls_state) {
    g_upmem_tls_state = rank;
  }

  ~upmem_state_scope() { g_upmem_tls_state = prev_; }

private:
  upmem_pim_rank *prev_;
};

static upmem_pim_rank &upmem_state(void) {
  upmem_pim_rank *rank = g_upmem_tls_state;
  if (!rank) {
    abort();
  }
  return *rank;
}

static bool ensure_pim_device_layer_locked(upmem_pim_rank *rank);
static void reset_pim_device_layer_locked(void);
static bool register_ci_handlers_locked(void);
static void ci_mmio_dpu_handler(pim_device_t *dev, pim_region_t *region,
                                size_t offset, uint64_t value, void *user_data);
static void ci_dpu_worker_handler(pim_device_t *dev, pim_region_t *region,
                                  size_t offset, uint64_t value,
                                  void *user_data);
static bool ci_is_thread_command_locked(uint64_t cmd_word);
static bool ci_command_executes_per_dpu_locked(size_t ci, uint64_t cmd_word,
                                               bool *needs_mask_fuzz);
static void ci_compute_single_dpu_payload(size_t ci, uint8_t dpu_local,
                                          uint64_t cmd_word,
                                          uint32_t *payload_out,
                                          bool *needs_mask_fuzz_out,
                                          uint8_t *selected_mask_out);
static void ci_signal_commit_progress_locked(void);
static void ci_finalize_commit_for_ci_locked(size_t ci);

upmem_dpu::upmem_dpu()
    : dma_engine_(), iram_(kDefaultIramSize),
      mram_(nullptr, kDefaultMramSize), pipeline_(DPU_NR_TASKLETS),
      wram_(kDefaultWramSize) {}

void upmem_dpu::bind_mram_region(uint8_t *mram_base, size_t mram_size) {
  mram_.rebind(static_cast<void *>(mram_base), mram_size);
}

uint8_t *upmem_dpu::mram_base() const {
  return static_cast<uint8_t *>(mram_.mapping_base());
}

size_t upmem_dpu::mram_size() const { return mram_.mapping_size(); }

DMAEngine &upmem_dpu::dma_engine() { return dma_engine_; }
const DMAEngine &upmem_dpu::dma_engine() const { return dma_engine_; }

IRAM &upmem_dpu::iram() { return iram_; }
const IRAM &upmem_dpu::iram() const { return iram_; }

MRAM &upmem_dpu::mram() { return mram_; }
const MRAM &upmem_dpu::mram() const { return mram_; }

Pipeline &upmem_dpu::pipeline() { return pipeline_; }
const Pipeline &upmem_dpu::pipeline() const { return pipeline_; }

WRAM &upmem_dpu::wram() { return wram_; }
const WRAM &upmem_dpu::wram() const { return wram_; }

upmem_pim_chip::upmem_pim_chip() : dpus_(), ci_write_(), ci_read_() {}

void upmem_pim_chip::bind_ci_windows(size_t chip_index) {
  const size_t mmio_offset = kCtrlMmioBase + chip_index * kCtrlCiWordSize;
  const size_t rw_offset = kCtrlRwBase + chip_index * kCtrlCiWordSize;
  ci_write_.rebind(mmio_offset, kCtrlCiWordSize);
  ci_read_.rebind(rw_offset, kCtrlCiWordSize);
}

upmem_dpu &upmem_pim_chip::dpu(size_t dpu_local) { return dpus_[dpu_local]; }
const upmem_dpu &upmem_pim_chip::dpu(size_t dpu_local) const {
  return dpus_[dpu_local];
}

CI &upmem_pim_chip::ci_write() { return ci_write_; }
const CI &upmem_pim_chip::ci_write() const { return ci_write_; }

CI &upmem_pim_chip::ci_read() { return ci_read_; }
const CI &upmem_pim_chip::ci_read() const { return ci_read_; }

upmem_pim_rank::upmem_pim_rank() {
  for (size_t chip_index = 0; chip_index < kNumChips; ++chip_index) {
    chips_[chip_index].bind_ci_windows(chip_index);
  }
}

size_t upmem_pim_rank::chip_index_from_dpu_global(size_t dpu_global_index) {
  return dpu_global_index % kNumChips;
}

size_t upmem_pim_rank::dpu_local_from_dpu_global(size_t dpu_global_index) {
  return dpu_global_index / kNumChips;
}

upmem_dpu &upmem_pim_rank::dpu_by_global(size_t dpu_global_index) {
  return chips_[chip_index_from_dpu_global(dpu_global_index)].dpu(
      dpu_local_from_dpu_global(dpu_global_index));
}

const upmem_dpu &upmem_pim_rank::dpu_by_global(size_t dpu_global_index) const {
  return chips_[chip_index_from_dpu_global(dpu_global_index)].dpu(
      dpu_local_from_dpu_global(dpu_global_index));
}

upmem_pim_chip &upmem_pim_rank::chip(size_t chip_index) {
  return chips_[chip_index];
}

const upmem_pim_chip &upmem_pim_rank::chip(size_t chip_index) const {
  return chips_[chip_index];
}

void upmem_pim_rank::bind_dpu_mram(size_t dpu_global_index, uint8_t *mram_base,
                                   size_t mram_size_bytes) {
  dpu_by_global(dpu_global_index).bind_mram_region(mram_base, mram_size_bytes);
  if (mram_base && mram_size_bytes > 0) {
    fallback_mram_base = mram_base;
    fallback_mram_size = mram_size_bytes;
  }
}

CI &upmem_pim_rank::ci_write_lane(size_t ci_index) {
  return chips_[ci_index].ci_write();
}

const CI &upmem_pim_rank::ci_write_lane(size_t ci_index) const {
  return chips_[ci_index].ci_write();
}

CI &upmem_pim_rank::ci_read_lane(size_t ci_index) {
  return chips_[ci_index].ci_read();
}

const CI &upmem_pim_rank::ci_read_lane(size_t ci_index) const {
  return chips_[ci_index].ci_read();
}

upmem_pim_rank *upmem_create_rank(pim_vdev_t *vdev) {
  (void)vdev;

  upmem_pim_rank *rank = new (std::nothrow) upmem_pim_rank();
  if (!rank) {
    errno = ENOMEM;
    return nullptr;
  }

  rank->mram_size = UPMEM_MRAM_SIZE;
  rank->dax_size = UPMEM_DAX_REGION_SIZE;

  if (!ensure_pim_device_layer_locked(rank)) {
    if (errno == 0) {
      errno = ENOMEM;
    }
    upmem_destroy_rank(rank);
    return nullptr;
  }

  if (!rank->dev) {
    errno = ENODEV;
    upmem_destroy_rank(rank);
    return nullptr;
  }

  if (!upmem_init_rank(rank, rank->dev)) {
    if (errno == 0) {
      errno = ENOMEM;
    }
    upmem_destroy_rank(rank);
    return nullptr;
  }

  return rank;
}

static inline upmem_device_user_data *
upmem_get_device_user_data(pim_device_t *dev) {
  if (!dev) {
    return nullptr;
  }
  return static_cast<upmem_device_user_data *>(pim_device_get_user_data(dev));
}

static inline void upmem_clear_device_user_data(pim_device_t *dev) {
  if (!dev) {
    return;
  }
  upmem_device_user_data *ud = upmem_get_device_user_data(dev);
  pim_device_set_user_data(dev, nullptr);
  delete ud;
}

static inline upmem_pim_rank *upmem_state_from_device(pim_device_t *dev) {
  upmem_device_user_data *ud = upmem_get_device_user_data(dev);
  return ud ? ud->rank : nullptr;
}

bool upmem_init_rank(upmem_pim_rank *rank, pim_device_t *device) {
  if (!rank || !device) {
    errno = EINVAL;
    return false;
  }

  upmem_device_user_data *device_ud = upmem_get_device_user_data(device);
  if (!device_ud) {
    device_ud = new (std::nothrow) upmem_device_user_data();
    if (!device_ud) {
      errno = ENOMEM;
      return false;
    }
    pim_device_set_user_data(device, device_ud);
  } else if (device_ud->rank && device_ud->rank != rank) {
    errno = EBUSY;
    return false;
  }

  if (rank->dev && rank->dev != device) {
    errno = EBUSY;
    return false;
  }

  device_ud->rank = rank;
  rank->dev = device;
  return true;
}

static void release_upmem_state_on_last_close(upmem_pim_rank *rank) {
  if (!rank) {
    return;
  }

  StateLockGuard lock_guard(rank->lock);

  if (rank->dev) {
    upmem_clear_device_user_data(rank->dev);
    (void)pim_device_deinit(rank->dev);
    rank->dev = nullptr;
  }

  rank->dax_region = nullptr;
  for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
    CI &write_lane = rank->ci_write_lane(ci);
    CI &read_lane = rank->ci_read_lane(ci);
    write_lane.mmio_region = nullptr;
    write_lane.rw_region = nullptr;
    read_lane.mmio_region = nullptr;
    read_lane.rw_region = nullptr;
    write_lane.committed = CI_EMPTY_CMD;
    write_lane.updated.store(CI_EMPTY_CMD, std::memory_order_relaxed);
    write_lane.payload_needs_mask_fuzz = false;
    write_lane.payload_fuzz_idx = 0;
    write_lane.dpu_commit_pending = 0;
    write_lane.dpu_commit_done = 0;
    write_lane.dpu_cmd = CI_EMPTY_CMD;
    write_lane.dpu_payload_accum = 0;
    write_lane.dpu_needs_mask_fuzz = false;
    write_lane.dpu_fanout_active = false;
    write_lane.dpu_exec_per_dpu = false;
    write_lane.dpu_serial_exec_done = false;
  }

  rank->ci_commit_active = false;
  rank->ci_commit_cv.notify_all();
  rank->ci_commit_pending = 0;
  rank->ci_commit_done = 0;
  rank->ci_commit_expected_color = 0;
  rank->ci_commit_reset_mask = 0;

  if (rank->dax_backing && rank->dax_backing != MAP_FAILED) {
    (void)munmap(rank->dax_backing, rank->dax_size);
    rank->dax_backing = nullptr;
  }

  for (size_t dpu = 0; dpu < UPMEM_MAX_DPUS_PER_RANK; ++dpu) {
    if (rank->dpu_mram[dpu] && rank->dpu_mram[dpu] != MAP_FAILED) {
      (void)munmap(rank->dpu_mram[dpu], rank->mram_size);
      rank->dpu_mram[dpu] = nullptr;
      rank->bind_dpu_mram(dpu, nullptr, 0);
    }
  }

  rank->fallback_mram_base = nullptr;
  rank->fallback_mram_size = 0;
}

void upmem_destroy_rank(upmem_pim_rank *rank) {
  if (!rank) {
    return;
  }

  release_upmem_state_on_last_close(rank);
  delete rank;
}

/* ------------------------------------------------------------------------- */
/* Minimal rank simulation (ioctl ABI)                                        */
/* ------------------------------------------------------------------------- */

static int ensure_dpu_mram_locked(size_t dpu_id) {
  upmem_pim_rank &rank = upmem_state();
  if (dpu_id >= UPMEM_MAX_DPUS_PER_RANK) {
    errno = EINVAL;
    return -1;
  }

  if (rank.dpu_mram[dpu_id]) {
    rank.bind_dpu_mram(dpu_id, rank.dpu_mram[dpu_id], rank.mram_size);
    return 0;
  }

  void *buf = mmap(nullptr, rank.mram_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (buf == MAP_FAILED) {
    return -1;
  }
  rank.dpu_mram[dpu_id] = static_cast<uint8_t *>(buf);
  rank.bind_dpu_mram(dpu_id, rank.dpu_mram[dpu_id], rank.mram_size);

  return 0;
}

static bool transfer_in_bounds(const struct dpu_transfer_mram_abi *tm) {
  upmem_pim_rank &rank = upmem_state();
  if (!tm) {
    return false;
  }

  const size_t off = static_cast<size_t>(tm->offset_in_mram);
  const size_t size = static_cast<size_t>(tm->size);
  if (off > rank.mram_size) {
    return false;
  }
  if (size > (rank.mram_size - off)) {
    return false;
  }

  return true;
}

static uint8_t ci_mask_from_commands(const uint64_t *cmds) {
  if (!cmds) {
    return 0;
  }

  uint8_t mask = 0;
  for (size_t i = 0; i < UPMEM_NB_CI; ++i) {
    if (cmds[i] != CI_EMPTY_CMD) {
      mask |= (uint8_t)(1u << i);
    }
  }
  return mask;
}

static uint8_t ci_all_dpus_mask(void) { return 0xFFu; }

static inline uint64_t ci_updated_load(const upmem_pim_rank &rank, size_t ci) {
  return rank.ci_write_lane(ci).updated.load(std::memory_order_relaxed);
}

static inline void ci_updated_store(upmem_pim_rank &rank, size_t ci,
                                    uint64_t value) {
  rank.ci_write_lane(ci).updated.store(value, std::memory_order_relaxed);
}

[[maybe_unused]] static uint8_t ci_popcount_u8(uint8_t v) {
  uint8_t n = 0;
  for (uint8_t i = 0; i < UPMEM_NB_DPU_PER_CI; ++i) {
    if ((v & (uint8_t)(1u << i)) != 0) {
      ++n;
    }
  }
  return n;
}

static void ci_reset_selection_state(void) {
  upmem_pim_rank &rank = upmem_state();
  const uint8_t all = ci_all_dpus_mask();
  for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
    CI &lane = rank.ci_write_lane(ci);
    lane.selected_mask = all;
    lane.group_mask.fill(0u);
    lane.group_mask[0] = all;
  }
}

static bool is_ci_software_reset_cmd(uint64_t cmd_word) {
  return (cmd_word & 0xFFFF00000000FF00ULL) == 0x01FF00000000FF00ULL;
}

static bool ci_is_run_state_read_cmd(uint64_t cmd_word) {
  const uint8_t opcode = static_cast<uint8_t>((cmd_word >> 56) & 0xFFu);
  const uint8_t tag = static_cast<uint8_t>((cmd_word >> 48) & 0xFFu);
  const uint8_t b0 = static_cast<uint8_t>((cmd_word >> 0) & 0xFFu);
  const uint8_t b1 = static_cast<uint8_t>((cmd_word >> 8) & 0xFFu);

  return opcode == 0x33u && tag == 0x00u && b0 == 0x84u && b1 == 0x02u;
}

static uint32_t ci_payload_from_command(size_t ci, uint64_t cmd_word,
                                        bool *needs_mask_fuzz) {
  upmem_pim_rank &rank = upmem_state();
  return CI::payload_for_command(&rank, ci, cmd_word, needs_mask_fuzz);
}

static uint64_t ci_ready_result(bool expected_color_set, uint32_t payload) {
  uint64_t result = 0x000000FF00000000ULL;
  const uint8_t color = expected_color_set ? 0xFFu : 0x00u;
  result |= ((uint64_t)color << 48);
  result |= (uint64_t)payload;
  return result;
}

static uint64_t ci_nop_style_result(bool expected_color_set, uint32_t payload) {
  uint64_t result = 0xFF0000FF00000000ULL; /* CI_NOP marker + valid bits */
  const uint8_t color = expected_color_set ? 0xFFu : 0x00u;
  result |= ((uint64_t)color << 48);
  result |= (uint64_t)(payload & 0xFFu);
  return result;
}

static bool ci_is_thread_command_locked(uint64_t cmd_word) {
  return Pipeline::is_thread_command(cmd_word);
}

static bool ci_command_executes_per_dpu_locked(size_t ci, uint64_t cmd_word,
                                               bool *needs_mask_fuzz) {
  upmem_pim_rank &rank = upmem_state();
  if (needs_mask_fuzz) {
    *needs_mask_fuzz = false;
  }

  if (ci >= UPMEM_NB_CI) {
    return false;
  }
  const CI &ci_state = rank.ci_write_lane(ci);

  const uint8_t opcode = static_cast<uint8_t>((cmd_word >> 56) & 0xFFu);
  if (opcode != 0x33u) {
    return false;
  }

  if (ci_is_run_state_read_cmd(cmd_word)) {
    return true;
  }

  if (ci_state.iram_write_structure_valid ||
      ci_state.wram_write_structure_valid) {
    if (needs_mask_fuzz) {
      *needs_mask_fuzz = true;
    }
    return true;
  }

  const uint8_t tag = static_cast<uint8_t>((cmd_word >> 48) & 0xFFu);
  const uint8_t b0 = static_cast<uint8_t>((cmd_word >> 0) & 0xFFu);
  const uint8_t b1 = static_cast<uint8_t>((cmd_word >> 8) & 0xFFu);
  const uint8_t b2 = static_cast<uint8_t>((cmd_word >> 16) & 0xFFu);
  const uint8_t b3 = static_cast<uint8_t>((cmd_word >> 24) & 0xFFu);

  if (tag == 0xA7u) {
    const bool dma_ctrl_write =
        ((b0 & 0xF0u) == 0x60u) && ((b1 & 0xF0u) == 0x60u) &&
        ((b2 & 0xF0u) == 0x60u) && ((b3 & 0xF0u) == 0x60u);
    if (dma_ctrl_write) {
      return true;
    }
  }

  return ci_is_thread_command_locked(cmd_word);
}

static void ci_compute_single_dpu_payload(size_t ci, uint8_t dpu_local,
                                          uint64_t cmd_word,
                                          uint32_t *payload_out,
                                          bool *needs_mask_fuzz_out,
                                          uint8_t *selected_mask_out) {
  upmem_pim_rank &rank = upmem_state();
  if (payload_out) {
    *payload_out = 0u;
  }
  if (needs_mask_fuzz_out) {
    *needs_mask_fuzz_out = false;
  }
  if (selected_mask_out) {
    *selected_mask_out = 0u;
  }

  if (ci >= UPMEM_NB_CI || dpu_local >= UPMEM_NB_DPU_PER_CI) {
    return;
  }

  CI &ci_state = rank.ci_write_lane(ci);
  const uint8_t original_mask = ci_state.selected_mask;
  const uint8_t worker_mask =
      static_cast<uint8_t>(original_mask & static_cast<uint8_t>(1u << dpu_local));

  if (worker_mask == 0u) {
    if (selected_mask_out) {
      *selected_mask_out = original_mask;
    }
    return;
  }

  ci_state.selected_mask = worker_mask;

  bool worker_needs_mask_fuzz = false;
  const uint32_t payload_part =
      CI::payload_for_command(&rank, ci, cmd_word, &worker_needs_mask_fuzz);

  ci_state.selected_mask = original_mask;

  if (payload_out) {
    *payload_out = payload_part;
  }
  if (needs_mask_fuzz_out) {
    *needs_mask_fuzz_out = worker_needs_mask_fuzz;
  }
  if (selected_mask_out) {
    *selected_mask_out = ci_state.selected_mask;
  }
}

static void ci_signal_commit_progress_locked(void) {
  upmem_pim_rank &rank = upmem_state();
  if (!rank.ci_commit_active) {
    return;
  }

  if (rank.ci_commit_done >= rank.ci_commit_pending) {
    rank.ci_commit_cv.notify_all();
  }
}

static void ci_finalize_commit_for_ci_locked(size_t ci) {
  upmem_pim_rank &rank = upmem_state();
  if (ci >= UPMEM_NB_CI) {
    return;
  }

  CI &write_lane = rank.ci_write_lane(ci);
  CI &read_lane = rank.ci_read_lane(ci);
  const bool expected_set =
      (rank.ci_commit_expected_color & (uint8_t)(1u << ci)) != 0;
  const uint32_t payload = write_lane.dpu_payload_accum;
  const bool needs_mask_fuzz = write_lane.dpu_needs_mask_fuzz;

  ci_updated_store(rank, ci,
                   needs_mask_fuzz ? ci_nop_style_result(expected_set, payload)
                                   : ci_ready_result(expected_set, payload));
  write_lane.payload_needs_mask_fuzz = needs_mask_fuzz;
  write_lane.payload_fuzz_idx = 0;

  if (read_lane.rw_region) {
    (void)pim_write_raw(read_lane.rw_region, 0, 8, ci_updated_load(rank, ci));
  }

  write_lane.dpu_fanout_active = false;
  write_lane.dpu_exec_per_dpu = false;
  write_lane.dpu_serial_exec_done = false;
  write_lane.dpu_commit_pending = 0;
  write_lane.dpu_commit_done = 0;

  if (rank.ci_commit_active) {
    rank.ci_commit_done++;
    ci_signal_commit_progress_locked();
  }
}

static bool env_flag_enabled_once(const char *name) {
  const char *v = getenv(name);
  return v && strcmp(v, "0") != 0;
}

static bool ci_trace_enabled(void) {
  static const bool enabled =
      env_flag_enabled_once("HOSTPIMSIM_UPMEM_TRACE_CI");
  return enabled;
}

static bool xfer_trace_enabled(void) {
  static const bool enabled =
      env_flag_enabled_once("HOSTPIMSIM_UPMEM_TRACE_XFER");
  return enabled;
}

static void xfer_tracef(const char *fmt, ...) {
  if (!xfer_trace_enabled() || !fmt) {
    return;
  }

  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[hostpimsim-upmem-xfer] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

static void ci_tracef(const char *fmt, ...) {
  if (!ci_trace_enabled() || !fmt) {
    return;
  }

  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[hostpimsim-upmem-ci] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

static void ci_mmio_dpu_handler(pim_device_t *dev, pim_region_t *region,
                                size_t offset, uint64_t value,
                                void *user_data) {
  uint8_t ci_id = 0;
  if (!decode_ci_handler_user_data(user_data, &ci_id) || ci_id >= UPMEM_NB_CI) {
    return;
  }

  upmem_pim_rank *rank_ptr = upmem_state_from_device(dev);
  if (!rank_ptr) {
    return;
  }
  upmem_state_scope scope(rank_ptr);

  upmem_pim_rank &rank = *rank_ptr;
  const size_t ci = static_cast<size_t>(ci_id);
  const uint64_t cmd = value;
  const uint8_t fanout_mask = ci_all_dpus_mask();
  bool dispatch_workers = false;
  bool per_dpu_exec = false;

  {
    StateLockGuard lock_guard(rank.lock);
    CI &lane = rank.ci_write_lane(ci);

    [[maybe_unused]] const bool expected_set =
        (rank.ci_commit_expected_color & (uint8_t)(1u << ci)) != 0;

    lane.dpu_fanout_active = false;
    lane.dpu_exec_per_dpu = false;
    lane.dpu_serial_exec_done = false;
    lane.dpu_commit_pending = 0;
    lane.dpu_commit_done = 0;
    lane.dpu_payload_accum = 0;
    lane.dpu_needs_mask_fuzz = false;
    lane.dpu_cmd = cmd;

    if (cmd == CI_EMPTY_CMD) {
      ci_updated_store(rank, ci, CI_EMPTY_CMD);
      lane.payload_needs_mask_fuzz = false;
      lane.payload_fuzz_idx = 0;
    } else if (cmd == CI_BYTE_ORDER_CMD) {
      ci_updated_store(rank, ci, 0x000103FF0F8FCFEFULL);
      lane.payload_needs_mask_fuzz = false;
      lane.payload_fuzz_idx = 0;
    } else {
      if (is_ci_software_reset_cmd(cmd)) {
        rank.ci_commit_reset_mask |= (uint8_t)(1u << ci);
      }

      bool needs_mask_fuzz = false;
      per_dpu_exec =
          ci_command_executes_per_dpu_locked(ci, cmd, &needs_mask_fuzz);
      dispatch_workers = true;

      lane.dpu_fanout_active = true;
      lane.dpu_exec_per_dpu = per_dpu_exec;
      lane.dpu_serial_exec_done = false;
      lane.dpu_commit_pending = UPMEM_NB_DPU_PER_CI;
      lane.dpu_needs_mask_fuzz = per_dpu_exec && needs_mask_fuzz;
      lane.payload_needs_mask_fuzz = lane.dpu_needs_mask_fuzz;
      lane.payload_fuzz_idx = 0;
    }

    if (!dispatch_workers) {
      CI &read_lane = rank.ci_read_lane(ci);
      if (read_lane.rw_region) {
        (void)pim_write_raw(read_lane.rw_region, 0, 8,
                            ci_updated_load(rank, ci));
      }
      if (rank.ci_commit_active) {
        rank.ci_commit_done++;
        ci_signal_commit_progress_locked();
      }
    }

    if (ci_trace_enabled()) {
      ci_tracef("  ci%zu cmd=0x%016llx -> upd=0x%016llx fuzz=%d fanout=%d "
                "per_dpu=%d mask=0x%02x",
                ci, (unsigned long long)cmd,
                (unsigned long long)ci_updated_load(rank, ci),
                lane.payload_needs_mask_fuzz ? 1 : 0, dispatch_workers ? 1 : 0,
                per_dpu_exec ? 1 : 0,
                dispatch_workers ? (unsigned)fanout_mask : 0u);
    }
  }

  if (!dispatch_workers) {
    return;
  }

  for (uint8_t dpu_local = 0; dpu_local < UPMEM_NB_DPU_PER_CI; ++dpu_local) {
    (void)pim_invoke_dpu_handler(dev, region, offset, cmd,
                                 static_cast<uint32_t>(dpu_local + 1));
  }
}

static void ci_dpu_worker_handler(pim_device_t *dev, pim_region_t *region,
                                  size_t offset, uint64_t value,
                                  void *user_data) {
  (void)dev;
  (void)region;
  (void)offset;
  (void)value;

  uint8_t ci_id = 0;
  uint8_t dpu_local_id = 0;
  if (!decode_ci_dpu_handler_user_data(user_data, &ci_id, &dpu_local_id) ||
      ci_id >= UPMEM_NB_CI || dpu_local_id >= UPMEM_NB_DPU_PER_CI) {
    return;
  }

  upmem_pim_rank *rank_ptr = upmem_state_from_device(dev);
  if (!rank_ptr) {
    return;
  }
  upmem_state_scope scope(rank_ptr);

  upmem_pim_rank &rank = *rank_ptr;
  const size_t ci = static_cast<size_t>(ci_id);

  uint64_t cmd_word = CI_EMPTY_CMD;
  {
    StateLockGuard lock_guard(rank.lock);
    CI &lane = rank.ci_write_lane(ci);
    if (!lane.dpu_fanout_active) {
      return;
    }

    const uint8_t pending = lane.dpu_commit_pending;
    uint8_t done = lane.dpu_commit_done;
    if (pending == 0 || done >= pending) {
      return;
    }

    cmd_word = lane.dpu_cmd;
    if (!lane.dpu_exec_per_dpu) {
      if (!lane.dpu_serial_exec_done) {
        bool needs_mask_fuzz = false;
        const uint32_t payload =
            ci_payload_from_command(ci, cmd_word, &needs_mask_fuzz);
        lane.dpu_payload_accum = payload;
        lane.dpu_needs_mask_fuzz = needs_mask_fuzz;
        lane.dpu_serial_exec_done = true;
      }

      ++done;
      lane.dpu_commit_done = done;
      if (rank.ci_commit_active) {
        rank.ci_commit_done++;
      }
      if (done == pending) {
        ci_finalize_commit_for_ci_locked(ci);
      }
      return;
    }
  }

  uint32_t payload_part = 0;
  bool needs_mask_fuzz = false;
  uint8_t selected_mask_after = 0;
  bool update_selected_mask = false;
  if (ci_is_run_state_read_cmd(cmd_word)) {
    payload_part = Pipeline::run_state_for_dpu(&rank, ci, dpu_local_id);
  } else {
    std::lock_guard<std::mutex> exec_guard(rank.ci_write_lane(ci).exec_lock);
    ci_compute_single_dpu_payload(ci, dpu_local_id, cmd_word, &payload_part,
                                  &needs_mask_fuzz, &selected_mask_after);
    update_selected_mask = true;
  }

  {
    StateLockGuard lock_guard(rank.lock);
    CI &lane = rank.ci_write_lane(ci);
    if (!lane.dpu_fanout_active) {
      return;
    }

    const uint8_t pending = lane.dpu_commit_pending;
    uint8_t done = lane.dpu_commit_done;
    if (pending == 0 || done >= pending) {
      return;
    }

    lane.dpu_payload_accum |= payload_part;
    lane.dpu_needs_mask_fuzz = lane.dpu_needs_mask_fuzz || needs_mask_fuzz;
    if (update_selected_mask) {
      lane.selected_mask = selected_mask_after;
    }

    ++done;
    lane.dpu_commit_done = done;
    if (rank.ci_commit_active) {
      rank.ci_commit_done++;
    }
    if (done == pending) {
      ci_finalize_commit_for_ci_locked(ci);
    }
  }
}

long upmem_ioctl_write_to_rank(upmem_pim_rank *rank, unsigned long arg) {
  upmem_state_scope scope(rank);
  if (arg == 0) {
    return -EINVAL;
  }

  struct dpu_transfer_mram_abi tm;
  memcpy(&tm, reinterpret_cast<const void *>(arg), sizeof(tm));

  if (!transfer_in_bounds(&tm)) {
    return -EINVAL;
  }

  const size_t off = static_cast<size_t>(tm.offset_in_mram);
  const size_t size = static_cast<size_t>(tm.size);
  if (size == 0) {
    return 0;
  }

  if (xfer_trace_enabled()) {
    size_t ptr_count = 0;
    for (size_t i = 0; i < UPMEM_MAX_DPUS_PER_RANK; ++i) {
      if (tm.ptr[i]) {
        ++ptr_count;
      }
    }
    xfer_tracef("write: off=0x%zx size=%zu ptrs=%zu", off, size, ptr_count);
  }

  // Serialized by rank_ioctl_cb()->pim_vdev_lock(vdev).
  for (size_t i = 0; i < UPMEM_MAX_DPUS_PER_RANK; ++i) {
    if (!tm.ptr[i]) {
      continue;
    }

    if (ensure_dpu_mram_locked(i) != 0) {
      return -ENOMEM;
    }
    memcpy(rank->dpu_mram[i] + off, tm.ptr[i], size);
  }

  return 0;
}

long upmem_ioctl_read_from_rank(upmem_pim_rank *rank, unsigned long arg) {
  upmem_state_scope scope(rank);
  if (arg == 0) {
    return -EINVAL;
  }

  struct dpu_transfer_mram_abi tm;
  memcpy(&tm, reinterpret_cast<const void *>(arg), sizeof(tm));

  if (!transfer_in_bounds(&tm)) {
    return -EINVAL;
  }

  const size_t off = static_cast<size_t>(tm.offset_in_mram);
  const size_t size = static_cast<size_t>(tm.size);
  if (size == 0) {
    return 0;
  }

  if (xfer_trace_enabled()) {
    size_t ptr_count = 0;
    for (size_t i = 0; i < UPMEM_MAX_DPUS_PER_RANK; ++i) {
      if (tm.ptr[i]) {
        ++ptr_count;
      }
    }
    xfer_tracef("read: off=0x%zx size=%zu ptrs=%zu", off, size, ptr_count);
  }

  // Serialized by rank_ioctl_cb()->pim_vdev_lock(vdev).
  for (size_t i = 0; i < UPMEM_MAX_DPUS_PER_RANK; ++i) {
    if (!tm.ptr[i]) {
      continue;
    }

    if (!rank->dpu_mram[i]) {
      memset(tm.ptr[i], 0, size);
      continue;
    }

    memcpy(tm.ptr[i], rank->dpu_mram[i] + off, size);
  }

  return 0;
}

long upmem_ioctl_commit_commands(upmem_pim_rank *rank, unsigned long arg) {
  upmem_state_scope scope(rank);
  if (arg == 0) {
    return -EINVAL;
  }

  uint64_t input[UPMEM_NB_CI];
  memcpy(input, reinterpret_cast<const void *>(arg), sizeof(input));

  (void)ci_trace_enabled();

  uint8_t full_mask = 0;
  uint8_t expected_color = 0;
  uint32_t pending_units = 0;
  uint8_t commit_units_by_ci[UPMEM_NB_CI] = {};
  pim_region_t *mmio_regions[UPMEM_NB_CI] = {};

  {
    std::unique_lock<std::mutex> lock(rank->lock);

    if (!rank->dev) {
      return -ENODEV;
    }
    while (rank->ci_commit_active) {
      rank->ci_commit_cv.wait(lock);
      if (!rank->dev) {
        return -ENODEV;
      }
    }

    for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
      rank->ci_write_lane(ci).committed = input[ci];
    }

    full_mask = ci_mask_from_commands(input);
    expected_color = static_cast<uint8_t>(rank->ci_sim_color & full_mask);

    rank->ci_last_mask = full_mask;
    rank->ci_last_expected_color = expected_color;

    rank->ci_commit_active = true;
    rank->ci_commit_pending = 0;
    rank->ci_commit_done = 0;
    for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
      CI &lane = rank->ci_write_lane(ci);
      lane.dpu_commit_pending = 0;
      lane.dpu_commit_done = 0;
      lane.dpu_cmd = CI_EMPTY_CMD;
      lane.dpu_payload_accum = 0;
      lane.dpu_needs_mask_fuzz = false;
      lane.dpu_fanout_active = false;
      lane.dpu_exec_per_dpu = false;
      lane.dpu_serial_exec_done = false;
    }
    rank->ci_commit_expected_color = expected_color;
    rank->ci_commit_reset_mask = 0;

    ci_tracef(
        "commit: mask=0x%02x sim_color_before=0x%02x expected_color=0x%02x",
        full_mask, rank->ci_sim_color, expected_color);

    for (size_t i = 0; i < UPMEM_NB_CI; ++i) {
      CI &lane = rank->ci_write_lane(i);
      if (input[i] == CI_EMPTY_CMD) {
        ci_updated_store(*rank, i, CI_EMPTY_CMD);
        lane.payload_needs_mask_fuzz = false;
        lane.payload_fuzz_idx = 0;
        continue;
      }

      mmio_regions[i] = lane.mmio_region;

      uint8_t commit_units = 1;
      if (input[i] != CI_BYTE_ORDER_CMD) {
        commit_units = static_cast<uint8_t>(UPMEM_NB_DPU_PER_CI + 1u);
      }

      commit_units_by_ci[i] = commit_units;
      pending_units += commit_units;
    }

    rank->ci_commit_pending = pending_units;
  }

  uint8_t dispatched_mask = 0;
  bool dispatch_failed = false;

  for (size_t i = 0; i < UPMEM_NB_CI; ++i) {
    if (input[i] == CI_EMPTY_CMD) {
      continue;
    }

    pim_region_t *region = mmio_regions[i];
    if (!region || pim_write(region, 0, 8, input[i]) != PIM_SUCCESS) {
      dispatch_failed = true;
      break;
    }

    dispatched_mask |= static_cast<uint8_t>(1u << i);
  }

  {
    std::unique_lock<std::mutex> lock(rank->lock);

    uint32_t dispatched_units = 0;
    for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
      if ((dispatched_mask & static_cast<uint8_t>(1u << ci)) != 0) {
        dispatched_units += commit_units_by_ci[ci];
      }
    }
    rank->ci_commit_pending = dispatched_units;

    while (rank->ci_commit_active &&
           rank->ci_commit_done < rank->ci_commit_pending) {
      rank->ci_commit_cv.wait(lock);
      if (!rank->dev) {
        dispatch_failed = true;
        break;
      }
    }

    /*
     * Userspace UFI toggles color before each CI command commit.
     * Mirror only dispatched CI lanes.
     */
    rank->ci_sim_color ^= dispatched_mask;

    /*
     * ci_exec_reset_cmd() clears context->color for reset CIs after commit.
     */
    if (rank->ci_commit_reset_mask != 0) {
      rank->ci_sim_color &= static_cast<uint8_t>(~rank->ci_commit_reset_mask);
    }

    rank->ci_update_generation++;

    ci_tracef("commit: sim_color_after=0x%02x generation=%u reset_mask=0x%02x",
              rank->ci_sim_color, rank->ci_update_generation,
              rank->ci_commit_reset_mask);

    rank->ci_last_mask = dispatched_mask;
    rank->ci_commit_active = false;
    rank->ci_commit_cv.notify_all();
  }

  if (dispatch_failed) {
    return -EIO;
  }

  return 0;
}

long upmem_ioctl_update_commands(upmem_pim_rank *rank, unsigned long arg) {
  upmem_state_scope scope(rank);
  if (arg == 0) {
    return -EINVAL;
  }

  uint64_t output[UPMEM_NB_CI];
  for (size_t i = 0; i < UPMEM_NB_CI; ++i) {
    output[i] = ci_updated_load(*rank, i);
  }

  if (ci_trace_enabled()) {
    StateLockGuard lock_guard(rank->lock);
    rank->ci_update_generation++;

    ci_tracef("update: generation=%u mask=0x%02x expected=0x%02x",
              rank->ci_update_generation, rank->ci_last_mask,
              rank->ci_last_expected_color);
    for (size_t i = 0; i < UPMEM_NB_CI; ++i) {
      if (rank->ci_write_lane(i).committed != CI_EMPTY_CMD) {
        ci_tracef("  ci%zu out=0x%016llx", i, (unsigned long long)output[i]);
      }
    }
  }

  memcpy(reinterpret_cast<void *>(arg), output, sizeof(output));
  return 0;
}

long upmem_ioctl_debug_mode(upmem_pim_rank *rank, unsigned long arg) {
  /*
   * This is currently a no-op; If it returns `-1` with `errno` set to `ENOTTY`,
   * the caller will likely fail.
   */
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
  uint32_t dpus_per_group[upmem_pim_rank::kNumGroups];
  uint32_t enabled_dpus;
  uint8_t all_dpus_are_enabled;
};
long upmem_ioctl_slice_info(upmem_pim_rank *rank, unsigned long arg) {
  if (arg == 0) {
    return -EINVAL;
  }

  dpu_configuration_slice_info_abi slice_info[kUpmemNumCi];
  memcpy(slice_info, reinterpret_cast<const void *>(arg), sizeof(slice_info));

  constexpr uint8_t kAllDpusMask =
      static_cast<uint8_t>((1u << kUpmemNumDpusPerCi) - 1u);

  // Serialized by rank_ioctl_cb()->pim_vdev_lock(vdev).
  for (size_t ci = 0; ci < kUpmemNumCi; ++ci) {
    uint8_t enabled_mask =
        static_cast<uint8_t>(slice_info[ci].enabled_dpus & kAllDpusMask);
    if (slice_info[ci].all_dpus_are_enabled) {
      enabled_mask = kAllDpusMask;
    }

    CI &lane = rank->ci_write_lane(ci);
    lane.selected_mask = enabled_mask;
    lane.group_mask[0] = enabled_mask;
    for (size_t group = 1; group < upmem_pim_rank::kNumGroups; ++group) {
      lane.group_mask[group] &= enabled_mask;
    }
  }

  return 0;
}

static void setup_mem(void) {
  upmem_pim_rank &rank = upmem_state();
  Pipeline::reset_rank(&rank);
  for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
    CI &lane = rank.ci_write_lane(ci);
    lane.committed = CI_EMPTY_CMD;
    ci_updated_store(rank, ci, CI_EMPTY_CMD);
    lane.payload_needs_mask_fuzz = false;
    lane.payload_fuzz_idx = 0;
    lane.pc_mode = 0x04u;        /* DPU_PC_MODE_16 */
    lane.dma_mux_status = 0x00u; /* Host-side mux by default. */
    lane.stack_up_mask = 0x00u;
    lane.dpu_commit_pending = 0;
    lane.dpu_commit_done = 0;
    lane.dpu_cmd = CI_EMPTY_CMD;
    lane.dpu_payload_accum = 0;
    lane.dpu_needs_mask_fuzz = false;
    lane.dpu_fanout_active = false;
    lane.dpu_exec_per_dpu = false;
    lane.dpu_serial_exec_done = false;
  }
  rank.ci_sim_color = 0;
  rank.ci_last_expected_color = 0;
  rank.ci_last_mask = 0;
  rank.ci_update_generation = 0;
  ci_reset_selection_state();
}

static size_t upmem_page_size(void) {
  static const size_t page_size = [] {
    const long ps = sysconf(_SC_PAGESIZE);
    return (ps > 0) ? static_cast<size_t>(ps) : 4096u;
  }();
  return page_size;
}

static bool map_real_range(void *base, size_t offset, size_t length) {
  upmem_pim_rank &rank = upmem_state();
  if (!base || base == MAP_FAILED || length == 0) {
    return false;
  }
  if (offset > rank.dax_size || length > (rank.dax_size - offset)) {
    return false;
  }

  const size_t page_size = upmem_page_size();
  const size_t aligned_off = offset & ~(page_size - 1u);
  const size_t bias = offset - aligned_off;
  size_t map_len = bias + length;
  map_len = (map_len + page_size - 1u) & ~(page_size - 1u);

  void *addr = static_cast<uint8_t *>(base) + aligned_off;
  void *mapped = mmap(addr, map_len, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  if (mapped == MAP_FAILED) {
    return false;
  }

  return true;
}

static bool map_real_rank_aperture(void *base) {
  upmem_pim_rank &rank = upmem_state();
  /* CI windows (same offsets used by xeon_sp backend helpers). */
  if (!map_real_range(base, 0x20000u, 0x1000u)) {
    return false;
  }
  if (!map_real_range(base, 0x28000u, 0x1000u)) {
    return false;
  }

  /*
   * MRAM aperture layout compatible with xeon_sp rank mapping:
   * - 64MB logical MRAM per DPU
   * - spread into 1MB-spaced chunks with 128KB real ranges per bank line
   * - reserve 8GB VA, map only real ranges as RW
   */
  constexpr size_t kChunkStride = 0x100000u;         /* 1MB */
  constexpr size_t kBankStride = 0x40000u;           /* 256KB */
  constexpr size_t kBankMappedSpan = 0x20040u;       /* 128KB + lane skew */
  constexpr size_t kLogicalBytesPerChunk = 0x10000u; /* 64KB */

  const size_t chunk_count =
      (rank.mram_size + kLogicalBytesPerChunk - 1u) / kLogicalBytesPerChunk;

  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    const size_t chunk_base = chunk * kChunkStride;
    for (size_t bank = 0; bank < 4; ++bank) {
      const size_t off = chunk_base + bank * kBankStride;
      if (!map_real_range(base, off, kBankMappedSpan)) {
        return false;
      }
    }
  }

  return true;
}

static bool create_ram_regions_for_range(size_t off, size_t len) {
  upmem_pim_rank &rank = upmem_state();
  if (!rank.dev || len == 0) {
    return false;
  }

  constexpr size_t kCiWriteBase = 0x20000u;
  constexpr size_t kCiReadBase = 0x28000u;
  constexpr size_t kCiLaneBytes = UPMEM_NB_CI * sizeof(uint64_t);

  const size_t range_end = off + len;
  size_t cursor = off;

  struct Hole {
    size_t begin;
    size_t end;
  };
  const Hole holes[] = {
      {kCiWriteBase, kCiWriteBase + kCiLaneBytes},
      {kCiReadBase, kCiReadBase + kCiLaneBytes},
  };

  for (const auto &hole : holes) {
    if (cursor >= range_end) {
      break;
    }
    if (range_end <= hole.begin || cursor >= hole.end) {
      continue;
    }

    if (cursor < hole.begin) {
      const size_t seg_len = hole.begin - cursor;
      if (!pim_region_create(rank.dev, cursor, seg_len, PIM_REGION_RAM)) {
        return false;
      }
    }

    if (cursor < hole.end) {
      cursor = hole.end;
    }
  }

  if (cursor < range_end) {
    if (!pim_region_create(rank.dev, cursor, range_end - cursor,
                           PIM_REGION_RAM)) {
      return false;
    }
  }

  return true;
}

static bool create_pim_regions_locked(void) {
  upmem_pim_rank &rank = upmem_state();
  if (!rank.dev) {
    return false;
  }

  constexpr size_t kCiWriteBase = 0x20000u;
  constexpr size_t kCiReadBase = 0x28000u;
  constexpr size_t kCiWordSize = sizeof(uint64_t);

  rank.dax_region = nullptr;
  for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
    CI &write_lane = rank.ci_write_lane(ci);
    CI &read_lane = rank.ci_read_lane(ci);
    write_lane.mmio_region = nullptr;
    write_lane.rw_region = nullptr;
    read_lane.mmio_region = nullptr;
    read_lane.rw_region = nullptr;
  }

  /* One CTRL_MMIO region per CI write lane. */
  for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
    const size_t off = kCiWriteBase + ci * kCiWordSize;
    pim_region_t *r =
        pim_region_create(rank.dev, off, kCiWordSize, PIM_REGION_CTRL_MMIO);
    if (!r) {
      return false;
    }
    rank.ci_write_lane(ci).mmio_region = r;
    if (!rank.dax_region) {
      rank.dax_region = r;
    }
  }

  /* One CTRL_RW region per CI read lane. */
  for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
    const size_t off = kCiReadBase + ci * kCiWordSize;
    pim_region_t *r =
        pim_region_create(rank.dev, off, kCiWordSize, PIM_REGION_CTRL_RW);
    if (!r) {
      return false;
    }
    rank.ci_read_lane(ci).rw_region = r;
  }

  /* RAM regions for each contiguous MRAM-mapped range in the aperture. */
  constexpr size_t kChunkStride = 0x100000u;         /* 1MB */
  constexpr size_t kBankStride = 0x40000u;           /* 256KB */
  constexpr size_t kBankMappedSpan = 0x20040u;       /* 128KB + lane skew */
  constexpr size_t kLogicalBytesPerChunk = 0x10000u; /* 64KB */

  const size_t chunk_count =
      (rank.mram_size + kLogicalBytesPerChunk - 1u) / kLogicalBytesPerChunk;

  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    const size_t chunk_base = chunk * kChunkStride;
    for (size_t bank = 0; bank < 4; ++bank) {
      const size_t off = chunk_base + bank * kBankStride;
      if (!create_ram_regions_for_range(off, kBankMappedSpan)) {
        return false;
      }
    }
  }

  return true;
}

static bool register_ci_handlers_locked(void) {
  upmem_pim_rank &rank = upmem_state();
  if (!rank.dev) {
    return false;
  }

  constexpr size_t kCiRegOffset = 0;
  constexpr size_t kCiRegSize = sizeof(uint64_t);

  for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
    pim_region_t *region = rank.ci_write_lane(ci).mmio_region;
    if (!region) {
      return false;
    }

    pim_error_t err = pim_register_dpu_handler(
        rank.dev, region, kCiRegOffset, kCiRegSize, 0, ci_mmio_dpu_handler,
        encode_ci_handler_user_data(static_cast<uint8_t>(ci)));
    if (err != PIM_SUCCESS) {
      return false;
    }

    for (size_t dpu_local = 0; dpu_local < UPMEM_NB_DPU_PER_CI; ++dpu_local) {
      err = pim_register_dpu_handler(
          rank.dev, region, kCiRegOffset, kCiRegSize,
          static_cast<uint32_t>(dpu_local + 1), ci_dpu_worker_handler,
          encode_ci_dpu_handler_user_data(static_cast<uint8_t>(ci),
                                          static_cast<uint8_t>(dpu_local)));
      if (err != PIM_SUCCESS) {
        return false;
      }
    }
  }

  return true;
}

/* ------------------------------------------------------------------------- */
/* vfile initialization */
/* ------------------------------------------------------------------------- */

static void reset_pim_device_layer_locked(void) {
  upmem_pim_rank &rank = upmem_state();
  if (rank.dev) {
    upmem_clear_device_user_data(rank.dev);
    (void)pim_device_deinit(rank.dev);
    rank.dev = nullptr;
  }

  rank.dax_region = nullptr;
  for (size_t ci = 0; ci < UPMEM_NB_CI; ++ci) {
    CI &write_lane = rank.ci_write_lane(ci);
    CI &read_lane = rank.ci_read_lane(ci);
    write_lane.mmio_region = nullptr;
    write_lane.rw_region = nullptr;
    read_lane.mmio_region = nullptr;
    read_lane.rw_region = nullptr;
    write_lane.dpu_commit_pending = 0;
    write_lane.dpu_commit_done = 0;
    write_lane.dpu_cmd = CI_EMPTY_CMD;
    write_lane.dpu_payload_accum = 0;
    write_lane.dpu_needs_mask_fuzz = false;
    write_lane.dpu_fanout_active = false;
    write_lane.dpu_exec_per_dpu = false;
    write_lane.dpu_serial_exec_done = false;
  }

  rank.ci_commit_active = false;
  rank.ci_commit_cv.notify_all();
  rank.ci_commit_pending = 0;
  rank.ci_commit_done = 0;
  rank.ci_commit_expected_color = 0;
  rank.ci_commit_reset_mask = 0;

  if (rank.dax_backing && rank.dax_backing != MAP_FAILED) {
    (void)munmap(rank.dax_backing, rank.dax_size);
    rank.dax_backing = nullptr;
  }
}

static bool ensure_pim_device_layer_locked(upmem_pim_rank *rank) {
  upmem_state_scope scope(rank);
  if (rank->dev && rank->dax_region && rank->dax_backing &&
      rank->dax_backing != MAP_FAILED) {
    return true;
  }

  reset_pim_device_layer_locked();

  rank->dax_backing = mmap(nullptr, rank->dax_size, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (rank->dax_backing == MAP_FAILED) {
    rank->dax_backing = nullptr;
    return false;
  }

  if (!map_real_rank_aperture(rank->dax_backing)) {
    reset_pim_device_layer_locked();
    return false;
  }

  rank->dev = pim_device_init(rank->dax_backing, rank->dax_size);
  if (!rank->dev) {
    reset_pim_device_layer_locked();
    return false;
  }

  if (!create_pim_regions_locked()) {
    reset_pim_device_layer_locked();
    return false;
  }

  if (!register_ci_handlers_locked()) {
    reset_pim_device_layer_locked();
    return false;
  }

  if (pim_pool_start(rank->dev, UPMEM_NB_CI * UPMEM_NB_DPU_PER_CI) !=
      PIM_SUCCESS) {
    reset_pim_device_layer_locked();
    return false;
  }

  setup_mem();
  return true;
}
