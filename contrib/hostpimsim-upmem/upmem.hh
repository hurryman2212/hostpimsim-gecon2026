#pragma once

#include "ci/ci.hh"
#include "dpu/dpu.hh"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

#ifndef DPU_NR_TASKLETS
#define DPU_NR_TASKLETS 24
#endif

struct upmem_runtime;

inline constexpr size_t kUpmemNumCi = 8;
inline constexpr size_t kUpmemNumDpusPerCi = 8;
inline constexpr size_t kUpmemNumDpus = kUpmemNumCi * kUpmemNumDpusPerCi;

class upmem_dpu {
public:
  static constexpr size_t kDefaultMramSize = (64u * 1024u * 1024u);
  static constexpr size_t kDefaultIramSize = (48u * 1024u);
  static constexpr size_t kDefaultWramSize = (128u * 1024u);

  upmem_dpu();

  void bind_mram_region(uint8_t *mram_base, size_t mram_size);
  uint8_t *mram_base() const;
  size_t mram_size() const;

  DMAEngine &dma_engine();
  const DMAEngine &dma_engine() const;
  IRAM &iram();
  const IRAM &iram() const;
  MRAM &mram();
  const MRAM &mram() const;
  Pipeline &pipeline();
  const Pipeline &pipeline() const;
  WRAM &wram();
  const WRAM &wram() const;

private:
  DMAEngine dma_engine_;
  IRAM iram_;
  MRAM mram_;
  Pipeline pipeline_;
  WRAM wram_;
};

class upmem_pim_chip {
public:
  static constexpr size_t kNumDpus = kUpmemNumDpusPerCi;
  static constexpr size_t kCtrlMmioBase = 0x20000u;
  static constexpr size_t kCtrlRwBase = 0x28000u;
  static constexpr size_t kCtrlCiWordSize = sizeof(uint64_t);

  upmem_pim_chip();

  void bind_ci_windows(size_t chip_index);

  upmem_dpu &dpu(size_t dpu_local);
  const upmem_dpu &dpu(size_t dpu_local) const;

  CI &ci_write();
  const CI &ci_write() const;

  CI &ci_read();
  const CI &ci_read() const;

private:
  std::array<upmem_dpu, kNumDpus> dpus_{};
  CI ci_write_{};
  CI ci_read_{};
};

class upmem_pim_rank {
public:
  static constexpr size_t kNumChips = kUpmemNumCi;
  static constexpr size_t kNumDpusPerChip = upmem_pim_chip::kNumDpus;
  static constexpr size_t kNumDpus = kUpmemNumDpus;
  static constexpr size_t kNumGroups = 8;
  static constexpr size_t kMramSize = (64u * 1024u * 1024u);
  static constexpr size_t kDaxRegionSize = (8ull * 1024ull * 1024ull * 1024ull);

  upmem_pim_rank();

  upmem_dpu &dpu_by_global(size_t dpu_global_index);
  const upmem_dpu &dpu_by_global(size_t dpu_global_index) const;

  upmem_pim_chip &chip(size_t chip_index);
  const upmem_pim_chip &chip(size_t chip_index) const;

  void bind_dpu_mram(size_t dpu_global_index, uint8_t *mram_base,
                     size_t mram_size);

  CI &ci_write_lane(size_t ci_index);
  const CI &ci_write_lane(size_t ci_index) const;

  CI &ci_read_lane(size_t ci_index);
  const CI &ci_read_lane(size_t ci_index) const;

  std::mutex lock;

  size_t mram_size = kMramSize;
  uint8_t *dpu_mram[kUpmemNumDpus]{};

  uint8_t ci_sim_color = 0;
  uint8_t ci_last_expected_color = 0;
  uint8_t ci_last_mask = 0;
  uint32_t ci_update_generation = 0;

  void *dax_backing = nullptr;
  size_t dax_size = kDaxRegionSize;

  pim_device_t *dev = nullptr;
  pim_region_t *dax_region = nullptr;

  bool ci_commit_active = false;
  std::condition_variable ci_commit_cv;
  uint32_t ci_commit_pending = 0;
  uint32_t ci_commit_done = 0;
  uint8_t ci_commit_expected_color = 0;
  uint8_t ci_commit_reset_mask = 0;

  struct upmem_runtime *runtime = nullptr;

private:
  static size_t chip_index_from_dpu_global(size_t dpu_global_index);
  static size_t dpu_local_from_dpu_global(size_t dpu_global_index);

  std::array<upmem_pim_chip, kUpmemNumCi> chips_{};
};

using StateLockGuard = std::lock_guard<std::mutex>;

/* Rank management implementations */

struct upmem_device_user_data {
  upmem_pim_rank *rank{nullptr};
};

upmem_pim_rank *upmem_create_rank(pim_vdev_t *vdev);
bool upmem_init_rank(upmem_pim_rank *rank, pim_device_t *device);
void upmem_destroy_rank(upmem_pim_rank *rank);

/* ioctl() implementations */

long upmem_ioctl_write_to_rank(upmem_pim_rank *rank, unsigned long arg);
long upmem_ioctl_read_from_rank(upmem_pim_rank *rank, unsigned long arg);
long upmem_ioctl_commit_commands(upmem_pim_rank *rank, unsigned long arg);
long upmem_ioctl_update_commands(upmem_pim_rank *rank, unsigned long arg);
long upmem_ioctl_debug_mode(upmem_pim_rank *rank, unsigned long arg);
long upmem_ioctl_slice_info(upmem_pim_rank *rank, unsigned long arg);
