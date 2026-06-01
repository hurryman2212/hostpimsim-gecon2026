/**
 * @file monitor.cc
 * @brief Polling-based Monitor Implementation (Optimized Sharding)
 *
 * This file contains:
 * - Sharded polling-based monitor threads for CTRL_MMIO regions
 * - Adaptive polling with exponential backoff for CPU efficiency
 *
 * Key optimizations:
 * - Region snapshot taken ONCE at thread startup (not inside hot loop)
 * - Thread-local pending vector reused across scans to avoid hot-path
 *   allocations while still dispatching outside scan
 * - Monitor thread count capped to number of CTRL_MMIO regions
 * - Lock-free register sampling in hot path (no region mutex contention)
 */

#include "internal.hh"

/*============================================================================
 * Busy Wait Optimization
 *============================================================================*/
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#include <immintrin.h>
#define mm_pause() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define mm_pause() asm volatile("yield" ::: "memory")
#endif

static constexpr uint32_t kDefaultNumMonitors = 1;

/*============================================================================
 * External Write Monitoring Implementation (Optimized Sharding)
 *============================================================================*/

/**
 * @brief Monitor thread context for dividing work among multiple monitors
 */
struct monitor_ctx {
  pim_device_t *dev;
  uint32_t thread_id;
  uint32_t total_threads;
};

static inline void
monitor_store_u32_if_dev(pim_device_t *dev,
                         std::atomic<uint32_t> pim_device_t::*member,
                         uint32_t value) noexcept {
  if (dev) {
    (dev->*member).store(value, std::memory_order_relaxed);
  }
}

static inline uint32_t monitor_load_u32_or_zero(
    pim_device_t *dev,
    const std::atomic<uint32_t> pim_device_t::*member) noexcept {
  return dev ? (dev->*member).load(std::memory_order_relaxed) : 0;
}

static inline bool
monitor_try_sample_entry_value_volatile(pim_region_t *region, size_t offset,
                                        const dpu_handler_entry *entry,
                                        uint64_t *sampled_value) noexcept {
  if (!region || !entry || !sampled_value) {
    return false;
  }
  const auto *addr =
      reinterpret_cast<const volatile uint8_t *>(region->ptr) + offset;
  return pim_load_register_u64_volatile(addr, entry->register_size,
                                        sampled_value);
}

static inline bool
monitor_refresh_entry_snapshot_volatile(pim_region_t *region, size_t offset,
                                        dpu_handler_entry *entry,
                                        uint64_t *sampled_value) noexcept {
  if (!entry || !sampled_value) {
    return false;
  }
  const uint32_t seq_before = entry->write_seq.load(std::memory_order_acquire);
  if ((seq_before & 1U) != 0U) {
    *sampled_value = 0;
    return false;
  }
  uint64_t loaded = 0;
  if (!monitor_try_sample_entry_value_volatile(region, offset, entry,
                                               &loaded)) {
    *sampled_value = 0;
    return false;
  }
  const uint32_t seq_after = entry->write_seq.load(std::memory_order_acquire);
  if ((seq_before != seq_after) || ((seq_after & 1U) != 0U)) {
    *sampled_value = 0;
    return false;
  }
  const uint64_t prev =
      entry->last_value.exchange(loaded, std::memory_order_relaxed);
  const bool changed = (prev != loaded);
  *sampled_value = loaded;
  return changed;
}

/**
 * @brief Scan function: iterate registered DPU handlers on CTRL_MMIO regions.
 *
 * Monitor workers use adaptive polling for low latency and shard CTRL_MMIO
 * regions by modulo assignment (region N -> monitor N % total_monitors).
 * Region snapshots are taken once at thread startup to keep the hot path free
 * from device-level lock contention.
 */
static bool monitor_scan(pim_device_t *dev,
                         const std::vector<pim_region_t *> &my_regions) {
  bool work_found = false;

  // Pending handlers to invoke after scan
  struct pending_handler {
    pim_region_t *region;
    size_t offset;
    uint64_t value;
  };
  static thread_local std::vector<pending_handler> pending;
  pending.clear();

  // SCAN PHASE: Check each assigned region
  for (auto *region : my_regions) {
    if (!dev->monitor_running.load(std::memory_order_relaxed)) {
      break;
    }

    while (region->handler_table_mutating.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    region->active_dispatch_ops.fetch_add(1, std::memory_order_acq_rel);
    if (region->handler_table_mutating.load(std::memory_order_acquire)) {
      region->active_dispatch_ops.fetch_sub(1, std::memory_order_acq_rel);
      continue;
    }

    for (auto &[offset, entry] : region->handlers) {
      uint64_t current_value = 0;

      // NOTE: If multiple writes occur between polling cycles, only the final
      // value triggers a handler invocation.
      if (monitor_refresh_entry_snapshot_volatile(region, offset, &entry,
                                                  &current_value)) {
        work_found = true;
        pending.push_back({region, offset, current_value});
      }
    }
    region->active_dispatch_ops.fetch_sub(1, std::memory_order_acq_rel);
  }

  // DISPATCH PHASE: cached bytes were already updated during scan.
  for (auto &p : pending) {
    (void)pim_invoke_dpu_handler_internal(dev, p.region, p.offset, p.value,
                                          false, 0);
  }

  return work_found;
}

static void monitor_thread_func(monitor_ctx ctx) {
  pim_device_t *dev = ctx.dev;
  const uint32_t thread_id = ctx.thread_id;
  const uint32_t total_threads = ctx.total_threads;

  // Configuration for Adaptive Polling
  uint32_t spin_limit =
      dev->active_poll_cnt_limit.load(std::memory_order_relaxed);
  uint32_t current_spin_limit = spin_limit; // Progressively reduced on idle
  int spin_counter = static_cast<int>(spin_limit);

  // 1. SNAPSHOT ONCE (Outside the Loop)
  // Take region snapshot under dev->mutex to avoid topology races.
  std::vector<pim_region_t *> my_regions;
  {
    std::lock_guard<std::mutex> lock(dev->mutex);
    size_t region_idx = 0;
    for (const auto &entry : dev->regions) {
      auto *region = entry.second;
      if (region->type == PIM_REGION_CTRL_MMIO) {
        // Sharding: round-robin distribution among monitor threads
        if ((region_idx % total_threads) == thread_id) {
          my_regions.push_back(region);
        }
        ++region_idx;
      }
    }
  }

  // 2. MAIN POLLING LOOP (Lock-Free on dev->mutex)
  while (dev->monitor_running.load(std::memory_order_relaxed)) {
    bool work_found = false;

    // SCAN PHASE
    try {
      work_found = monitor_scan(dev, my_regions);
    } catch (...) {
      // CRUCIAL: Never let exceptions escape monitor worker threads
      // (std::terminate). Fail-stop monitor mode for this device.
      dev->monitor_running.store(false, std::memory_order_release);
      return;
    }

    // 3. ADAPTIVE WAIT PHASE
    if (work_found) {
      // Refresh runtime-configured spin limit.
      spin_limit = dev->active_poll_cnt_limit.load(std::memory_order_relaxed);
      // Hot streak! Reset spin budget AND current_spin_limit to full.
      current_spin_limit = spin_limit;
      spin_counter = static_cast<int>(spin_limit);
    } else {
      // No work found. Spin or sleep?
      if (spin_counter > 0) {
        // BUSY WAIT: one pause per scan iteration.
        mm_pause();
        --spin_counter;
      } else {
        // COLD: Sleep to reduce CPU usage.
        auto sleep_us =
            dev->device_sleep_duration_us.load(std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));

        // Runtime poll-limit updates apply after each sleep cycle.
        spin_limit = dev->active_poll_cnt_limit.load(std::memory_order_relaxed);
        if (current_spin_limit > spin_limit) {
          current_spin_limit = spin_limit;
        }

        // After waking, progressively reduce the spin limit.
        // Each consecutive idle wakeup divides current_spin_limit, so:
        //   1st wakeup: spin_limit / divisor
        //   2nd wakeup: spin_limit / divisor^2
        //   ...until it converges to near-0 CPU when truly idle.
        uint32_t divisor =
            dev->wakeup_spin_divisor.load(std::memory_order_relaxed);
        current_spin_limit = current_spin_limit / divisor;
        spin_counter = static_cast<int>(current_spin_limit);
      }
    }
  }
}

pim_error_t pim_start_monitor(pim_device_t *dev, uint32_t num_monitors,
                              uint32_t device_sleep_duration_us,
                              uint32_t active_poll_cnt_limit,
                              uint32_t wakeup_spin_divisor) noexcept {
  if (!dev) {
    return PIM_ERR_INVALID_ARG;
  }

  // CRUCIAL: Serialize monitor lifecycle transitions and vector mutation.
  std::lock_guard<std::mutex> monitor_ctl_lock(dev->monitor_ctl_mutex);

  size_t total_mmio_regions = 0;
  uint32_t actual_monitors = kDefaultNumMonitors;

  // CRUCIAL: Hold g_sigsegv_mutex while checking/setting monitor_running and
  // snapshotting topology. Region create/free takes the same global mutex to
  // avoid check-then-lock races during monitor startup.
  {
    std::lock_guard<std::mutex> sigsegv_lock(g_sigsegv_mutex);
    std::lock_guard<std::mutex> lock(dev->mutex);

    if (dev->teardown_in_progress.load(std::memory_order_acquire) ||
        dev->sigsegv_transition_in_progress.load(std::memory_order_acquire)) {
      return PIM_ERR_BUSY;
    }

    if (dev->monitor_running.load(std::memory_order_relaxed)) {
      return PIM_ERR_ALREADY_EXISTS;
    }

    for (const auto &entry : dev->regions) {
      auto *region = entry.second;
      if (region->type == PIM_REGION_CTRL_MMIO) {
        ++total_mmio_regions;
      }
    }

    // CRUCIAL: In same-process mixed mode (polling + SIGSEGV), SIGSEGV MMIO
    // dispatch intentionally does not commit MMIO memory. Re-baseline
    // cached register bytes to current memory before monitor starts to avoid
    // stale synthetic diffs on the first scan.
    if (dev->sigsegv_enabled.load(std::memory_order_acquire)) {
      for (const auto &entry : dev->regions) {
        auto *region = entry.second;
        if (region->type != PIM_REGION_CTRL_MMIO) {
          continue;
        }
        while (region->handler_table_mutating.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        region->active_dispatch_ops.fetch_add(1, std::memory_order_acq_rel);
        if (region->handler_table_mutating.load(std::memory_order_acquire)) {
          region->active_dispatch_ops.fetch_sub(1, std::memory_order_acq_rel);
          continue;
        }
        for (auto &[offset, entry] : region->handlers) {
          uint64_t ignored_value = 0;
          (void)monitor_refresh_entry_snapshot_volatile(region, offset, &entry,
                                                        &ignored_value);
        }
        region->active_dispatch_ops.fetch_sub(1, std::memory_order_acq_rel);
      }
    }

    pim_set_sleep_duration_us(dev, device_sleep_duration_us);
    pim_set_poll_limit(dev, active_poll_cnt_limit);
    pim_set_wakeup_spin_divisor(dev, wakeup_spin_divisor);

    // Configure monitor thread count from the protected topology snapshot.
    actual_monitors = (num_monitors > 0) ? num_monitors : kDefaultNumMonitors;
    if (total_mmio_regions > 0 && actual_monitors > total_mmio_regions) {
      actual_monitors = static_cast<uint32_t>(total_mmio_regions);
    }
    dev->num_monitors.store(actual_monitors, std::memory_order_relaxed);
    dev->monitor_running.store(true, std::memory_order_release);
  }

  // Spawn monitor threads, each watching a portion of CTRL_MMIO regions
  try {
    dev->monitor_threads.reserve(actual_monitors);
    for (uint32_t i = 0; i < actual_monitors; ++i) {
      monitor_ctx ctx{dev, i, actual_monitors};
      dev->monitor_threads.emplace_back(monitor_thread_func, ctx);
    }
  } catch (...) {
    // CRUCIAL: C API must not leak C++ exceptions across the ABI boundary.
    dev->monitor_running.store(false, std::memory_order_release);
    pim_join_and_clear_threads(dev->monitor_threads);
    dev->num_monitors.store(0, std::memory_order_relaxed);
    return PIM_ERR_NO_MEMORY;
  }

  return PIM_SUCCESS;
}

pim_error_t pim_stop_monitor(pim_device_t *dev) noexcept {
  if (!dev) {
    return PIM_ERR_INVALID_ARG;
  }

  // CRUCIAL: Serialize stop/start and monitor_threads ownership.
  std::lock_guard<std::mutex> monitor_ctl_lock(dev->monitor_ctl_mutex);

  // CRUCIAL: A monitor thread cannot stop+join itself.
  // Reject from monitor-worker context; control thread must stop monitors.
  if (pim_current_thread_in_thread_vector_locked(dev->monitor_threads)) {
    return PIM_ERR_BUSY;
  }

  // Signal stop
  (void)dev->monitor_running.exchange(false, std::memory_order_acq_rel);

  // CRUCIAL: Always join/clear monitor_threads, even when monitor_running was
  // already false (e.g., worker fail-stop path after catching an exception).
  pim_join_and_clear_threads(dev->monitor_threads);
  dev->num_monitors.store(0, std::memory_order_relaxed);

  return PIM_SUCCESS;
}

bool pim_is_monitor_running(pim_device_t *dev) noexcept {
  if (!dev) {
    return false;
  }
  return dev->monitor_running.load();
}

uint32_t pim_get_num_monitors(pim_device_t *dev) noexcept {
  if (!dev) {
    return 0;
  }
  return dev->num_monitors.load(std::memory_order_relaxed);
}

void pim_set_sleep_duration_us(pim_device_t *dev, uint32_t sleep_us) noexcept {
  monitor_store_u32_if_dev(dev, &pim_device_t::device_sleep_duration_us,
                           (sleep_us > 0) ? sleep_us : kDefaultSleepDurationUs);
}

uint32_t pim_get_sleep_duration_us(pim_device_t *dev) noexcept {
  return monitor_load_u32_or_zero(dev, &pim_device_t::device_sleep_duration_us);
}

void pim_set_poll_limit(pim_device_t *dev, uint32_t poll_limit) noexcept {
  monitor_store_u32_if_dev(dev, &pim_device_t::active_poll_cnt_limit,
                           (poll_limit > 0) ? poll_limit : kDefaultPollLimit);
}

uint32_t pim_get_poll_limit(pim_device_t *dev) noexcept {
  return monitor_load_u32_or_zero(dev, &pim_device_t::active_poll_cnt_limit);
}

void pim_set_wakeup_spin_divisor(pim_device_t *dev, uint32_t divisor) noexcept {
  monitor_store_u32_if_dev(dev, &pim_device_t::wakeup_spin_divisor,
                           (divisor > 0) ? divisor : kDefaultWakeupSpinDivisor);
}

uint32_t pim_get_wakeup_spin_divisor(pim_device_t *dev) noexcept {
  return monitor_load_u32_or_zero(dev, &pim_device_t::wakeup_spin_divisor);
}
