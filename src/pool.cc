/**
 * @file pool.cc
 * @brief Thread Pool Implementation for DPU Handler Execution
 *
 * This file contains the unified thread pool that executes DPU handler
 * callbacks. The pool is shared by both:
 * - Polling-based monitor (pim_start_monitor)
 * - SIGSEGV-based monitor (pim_device_enable_sigsegv)
 *
 * Key features:
 * - Dynamic resizing via pim_set_max_active_dpus()
 * - Condition variable for efficient wakeup
 * - Graceful shutdown with task queue drain
 */

#include "internal.hh"

#include <linux/futex.h>

/*============================================================================
 * Thread Pool Implementation
 *============================================================================*/

static inline pim_error_t pool_validate_device(pim_device_t *dev) noexcept {
  return dev ? PIM_SUCCESS : PIM_ERR_INVALID_ARG;
}

static inline pim_error_t pool_start_failure_error(pim_device_t *dev) noexcept {
  return dev->teardown_in_progress.load(std::memory_order_acquire)
             ? PIM_ERR_BUSY
             : PIM_ERR_NO_MEMORY;
}

static inline pim_error_t
pool_reject_worker_context_locked(pim_device_t *dev) noexcept {
  return pim_current_thread_in_thread_vector_locked(dev->handler_pool)
             ? PIM_ERR_BUSY
             : PIM_SUCCESS;
}

static inline void pool_wake_device_barrier_all(pim_device_t *dev) noexcept {
  if (!dev) {
    return;
  }
  (void)syscall(SYS_futex, reinterpret_cast<int *>(&dev->futex_cnt_finished),
                FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}

/**
 * @brief Worker thread function for the handler thread pool
 *
 * Workers exit when pool is shutting down (pool_running == false)
 * and the queue is empty.
 */
static void pool_worker_func(pim_device_t *dev) noexcept {
  while (true) {
    std::function<void()> task;

    {
      std::unique_lock<std::mutex> lock(dev->pool_mutex);
      dev->pool_cv.wait(lock, [dev] {
        return !dev->pool_running.load(std::memory_order_relaxed) ||
               !dev->task_queue.empty();
      });

      // Check if we should exit (shutdown)
      if (!dev->pool_running.load(std::memory_order_relaxed) &&
          dev->task_queue.empty()) {
        return;
      }

      // Get task from queue
      if (!dev->task_queue.empty()) {
        task = std::move(dev->task_queue.front());
        dev->task_queue.pop();
      }
    }

    // Execute task outside the lock
    if (task) {
      try {
        task();
      } catch (...) {
        // CRUCIAL: Keep worker threads alive even if user callback throws.
      }

      const uint32_t finished =
          dev->futex_cnt_finished.fetch_add(1u, std::memory_order_acq_rel) + 1u;
      const uint32_t commited =
          dev->futex_cnt_commited.load(std::memory_order_acquire);
      if (finished == commited) {
        pool_wake_device_barrier_all(dev);
      }
    }
  }
}

/**
 * @brief Start pool internals.
 *
 * CRUCIAL: Caller must hold dev->pool_ctl_mutex.
 */
static bool pool_start_locked(pim_device_t *dev, uint32_t limit_concurrency) {
  if (dev->teardown_in_progress.load(std::memory_order_acquire)) {
    return false;
  }

  if (dev->pool_running.load(std::memory_order_relaxed)) {
    return true; // Already running
  }

  uint32_t concurrency = limit_concurrency;
  if (concurrency == 0) {
    // Count CTRL_MMIO regions from stable topology snapshot.
    std::lock_guard<std::mutex> lock(dev->mutex);
    for (const auto &entry : dev->regions) {
      auto *region = entry.second;
      if (region && region->type == PIM_REGION_CTRL_MMIO) {
        ++concurrency;
      }
    }
    if (concurrency == 0) {
      dev->max_active_dpus.store(0, std::memory_order_relaxed);
      return true; // No CTRL_MMIO regions
    }
  }

  dev->max_active_dpus.store(concurrency, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(dev->pool_mutex);
    dev->pool_running.store(true, std::memory_order_relaxed);
  }
  try {
    dev->handler_pool.reserve(concurrency);
    for (uint32_t i = 0; i < concurrency; ++i) {
      dev->handler_pool.emplace_back(pool_worker_func, dev);
    }
  } catch (...) {
    // CRUCIAL: C API must not leak C++ exceptions or terminate on partial
    // pool startup. Roll back to a clean stopped state.
    {
      std::lock_guard<std::mutex> lock(dev->pool_mutex);
      dev->pool_running.store(false, std::memory_order_relaxed);
    }
    dev->pool_cv.notify_all();
    pim_join_and_clear_threads(dev->handler_pool);
    dev->max_active_dpus.store(0, std::memory_order_relaxed);
    return false;
  }

  return true;
}

/**
 * @brief Stop pool internals.
 *
 * CRUCIAL: Caller must hold dev->pool_ctl_mutex.
 */
static void pool_stop_locked(pim_device_t *dev) {
  bool was_running = false;
  {
    std::lock_guard<std::mutex> lock(dev->pool_mutex);
    was_running = dev->pool_running.exchange(false, std::memory_order_relaxed);
  }

  if (!was_running && dev->handler_pool.empty()) {
    dev->max_active_dpus.store(0, std::memory_order_relaxed);
    return;
  }

  dev->pool_cv.notify_all();
  pool_wake_device_barrier_all(dev);

  pim_join_and_clear_threads(dev->handler_pool);
  dev->max_active_dpus.store(0, std::memory_order_relaxed);
  pool_wake_device_barrier_all(dev);
}

pim_error_t pim_pool_start(pim_device_t *dev,
                           uint32_t limit_concurrency) noexcept {
  pim_error_t arg_err = pool_validate_device(dev);
  if (arg_err != PIM_SUCCESS) {
    return arg_err;
  }
  // CRUCIAL: Serialize pool lifecycle and handler_pool ownership.
  std::lock_guard<std::mutex> pool_ctl_lock(dev->pool_ctl_mutex);
  if (!pool_start_locked(dev, limit_concurrency)) {
    return pool_start_failure_error(dev);
  }
  return PIM_SUCCESS;
}

pim_error_t pim_pool_stop(pim_device_t *dev) noexcept {
  pim_error_t arg_err = pool_validate_device(dev);
  if (arg_err != PIM_SUCCESS) {
    return arg_err;
  }
  // CRUCIAL: Serialize pool lifecycle and handler_pool ownership.
  std::lock_guard<std::mutex> pool_ctl_lock(dev->pool_ctl_mutex);

  // CRUCIAL: A pool worker cannot stop+join itself.
  // Reject in worker context to avoid self-join termination.
  pim_error_t worker_err = pool_reject_worker_context_locked(dev);
  if (worker_err != PIM_SUCCESS) {
    return worker_err;
  }
  pool_stop_locked(dev);
  return PIM_SUCCESS;
}

pim_error_t pim_set_max_active_dpus(pim_device_t *dev,
                                    uint32_t concurrency) noexcept {
  pim_error_t arg_err = pool_validate_device(dev);
  if (arg_err != PIM_SUCCESS) {
    return arg_err;
  }

  // CRUCIAL: Resize must be mutually exclusive with start/stop.
  std::lock_guard<std::mutex> pool_ctl_lock(dev->pool_ctl_mutex);

  // CRUCIAL: Resizing can stop+join workers internally.
  // Disallow from worker context to avoid self-join hazards.
  pim_error_t worker_err = pool_reject_worker_context_locked(dev);
  if (worker_err != PIM_SUCCESS) {
    return worker_err;
  }

  if (dev->teardown_in_progress.load(std::memory_order_acquire)) {
    return PIM_ERR_BUSY;
  }

  dev->max_active_dpus.store(concurrency, std::memory_order_relaxed);

  // If pool is not running, nothing more to do
  if (!dev->pool_running.load(std::memory_order_relaxed)) {
    return PIM_SUCCESS;
  }

  size_t current_size = dev->handler_pool.size();
  if (concurrency == current_size) {
    return PIM_SUCCESS;
  }

  /*
   * Avoid in-place shrinking/expansion races and join-order deadlocks.
   * Recreate the pool with the requested size after draining queued tasks.
   */
  pool_stop_locked(dev);
  if (concurrency > 0) {
    if (!pool_start_locked(dev, concurrency)) {
      return pool_start_failure_error(dev);
    }
  }
  return PIM_SUCCESS;
}

uint32_t pim_get_max_active_dpus(pim_device_t *dev) noexcept {
  if (!dev) {
    return 0;
  }
  return dev->max_active_dpus.load(std::memory_order_relaxed);
}
