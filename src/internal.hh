/**
 * @file internal.hh
 * @brief Internal shared structures and utilities for hostpimsim
 *
 * This header is NOT part of the public API. It is used internally.
 */

#pragma once

#include "hostpimsim.h"

#include <condition_variable>
#include <functional>
#include <map>
#include <queue>

static constexpr uint32_t kDefaultSleepDurationUs = 100;
static constexpr uint32_t kDefaultPollLimit = 20000;
static constexpr uint32_t kDefaultWakeupSpinDivisor = 4;

static inline long futex(void *addr1, int op, int val1,
                         const struct timespec *timeout, void *addr2,
                         int val3) {
  return syscall(SYS_futex, addr1, op, val1, timeout, addr2, val3);
}

static inline pim_error_t pim_return_with_errno(pim_error_t err_code,
                                                int fallback_errno) noexcept {
  if (errno <= 0) {
    errno = (fallback_errno > 0) ? fallback_errno : EIO;
  }
  return err_code;
}

/*============================================================================
 * Internal Structures
 *============================================================================*/

/**
 * @brief DPU handler entry with last known value for change detection
 */
struct dpu_handler_entry {
  pim_dpu_handler_t handler;
  void *user_data{nullptr};
  std::atomic<uint64_t> last_value{0}; /**< Last known register value */
  std::atomic<uint32_t> write_seq{
      0};                  /**< Even=stable, odd=writer in progress */
  size_t register_size{0}; /**< Register width in bytes (must be 1/2/4/8) */
  uint32_t depth{0};       /**< Handler depth level (0 = root/MMIO default) */

  dpu_handler_entry() = default;

  dpu_handler_entry(const dpu_handler_entry &other) noexcept
      : handler(other.handler), user_data(other.user_data),
        last_value(other.last_value.load(std::memory_order_relaxed)),
        write_seq(other.write_seq.load(std::memory_order_relaxed)),
        register_size(other.register_size), depth(other.depth) {}

  dpu_handler_entry &operator=(const dpu_handler_entry &other) noexcept {
    if (this == &other) {
      return *this;
    }
    handler = other.handler;
    user_data = other.user_data;
    last_value.store(other.last_value.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    write_seq.store(other.write_seq.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
    register_size = other.register_size;
    depth = other.depth;
    return *this;
  }
};

struct dpu_handler_key {
  size_t offset{0};
  uint32_t depth{0};

  bool operator==(const dpu_handler_key &other) const noexcept {
    return (offset == other.offset) && (depth == other.depth);
  }
};

struct dpu_handler_key_hash {
  size_t operator()(const dpu_handler_key &key) const noexcept {
    const size_t h0 = std::hash<size_t>{}(key.offset);
    const size_t h1 = std::hash<uint32_t>{}(key.depth);
    return h0 ^ (h1 + 0x9e3779b97f4a7c15ULL + (h0 << 6) + (h0 >> 2));
  }
};

static constexpr uint32_t kSigsegvFutexWait = 0;
static constexpr uint32_t kSigsegvFutexWork = 1;
static constexpr uint32_t kSigsegvFutexExit = UINT32_MAX;
/**
 * @brief SIGSEGV callback context for async-signal-safe handler invocation
 *
 * Each CTRL_MMIO region gets one callback thread that waits on a futex.
 * The signal handler writes data here and wakes the thread via futex.
 */
struct sigsegv_callback_ctx {
  pim_region_t *region; /**< Associated CTRL_MMIO region */
  std::thread thread;   /**< Callback thread */
  std::atomic<uint32_t> futex_word{kSigsegvFutexWait}; /**< futex state */
  std::atomic<size_t> offset{0};  /**< Handler offset within region */
  std::atomic<uint64_t> value{0}; /**< Written value */
};

/**
 * @brief Memory region structure (a view into device memory)
 */
struct pim_region {
  pim_device_t *device; /**< Parent device */
  size_t offset{0};     /**< Offset within device's shm */
  size_t size{0};       /**< Region size in bytes */
  size_t page_size{0};  /**< System page size captured at region creation */
  void *ptr{nullptr}; /**< Pointer to start of region (device->ptr + offset) */
  pim_region_type_t type; /**< Region type */
  std::unordered_map<size_t, dpu_handler_entry>
      handlers; /**< Root DPU handlers (depth=0) */
  std::unordered_map<dpu_handler_key, dpu_handler_entry, dpu_handler_key_hash>
      nested_handlers; /**< Nested DPU handlers (depth>0) */
  std::atomic<bool> handler_table_mutating{
      false}; /**< True while handler map is being structurally mutated */
  std::atomic<uint32_t> active_dispatch_ops{
      0};           /**< Concurrent lock-free dispatch/write ops */
  std::mutex mutex; /**< Thread safety */
};

/**
 * @brief PIM device structure (holds single shm file)
 */
struct pim_device {
  std::string path;         /**< Shared memory or device file path */
  int fd{-1};               /**< File descriptor */
  void *ptr{nullptr};       /**< Mapped memory pointer */
  size_t size{0};           /**< Device memory size */
  void *user_data{nullptr}; /**< Opaque caller-owned context pointer */
  std::atomic<bool> teardown_in_progress{
      false}; /**< True while deinit/destroy is tearing this device down */
  bool backing_ownership{
      false}; /**< True only for backing created by pim_device_create() */
  std::map<size_t, pim_region_t *>
      regions;      /**< Regions by start offset (views) */
  std::mutex mutex; /**< Thread safety */

  // Thread-based Monitor thread state
  std::mutex monitor_ctl_mutex;             /**< Serializes start/stop */
  std::vector<std::thread> monitor_threads; /**< Background monitor threads */
  std::atomic<bool> monitor_running{false}; /**< Monitor running flag */
  std::atomic<uint32_t> device_sleep_duration_us{
      kDefaultSleepDurationUs}; /**< Sleep duration in us */
  std::atomic<uint32_t> active_poll_cnt_limit{
      kDefaultPollLimit}; /**< Maximum spin iterations before sleep */
  std::atomic<uint32_t> num_monitors{0}; /**< Number of monitor threads */
  std::atomic<uint32_t> wakeup_spin_divisor{
      kDefaultWakeupSpinDivisor}; /**< Divisor for spin budget after waking from
                                     sleep */

  // Thread pool for DPU handler execution (replaces per-call std::async)
  std::mutex pool_ctl_mutex;                /**< Serializes pool control */
  std::vector<std::thread> handler_pool;    /**< Worker threads */
  std::atomic<uint32_t> max_active_dpus{0}; /**< Max concurrent DPU handlers */
  std::queue<std::function<void()>> task_queue; /**< Pending handler tasks */
  std::mutex pool_mutex;                        /**< Protects task_queue */
  std::condition_variable pool_cv;              /**< Notifies workers */
  std::atomic<bool> pool_running{false};        /**< Pool active flag */
  std::atomic<uint32_t> futex_cnt_commited{
      0}; /**< Total submitted pool tasks */
  std::atomic<uint32_t> futex_cnt_finished{0}; /**< Total finished pool tasks */

  // SIGSEGV-based monitor state
  std::atomic<bool> sigsegv_enabled{false}; /**< SIGSEGV monitoring enabled */
  std::atomic<bool> sigsegv_transition_in_progress{
      false}; /**< True while enable/disable transition is in progress */
  std::vector<std::unique_ptr<sigsegv_callback_ctx>>
      sigsegv_callbacks; /**< Callback threads */
};

/*============================================================================
 * Internal Helper Functions
 *============================================================================*/

/**
 * @brief Return true when handler register width is supported.
 */
static inline bool
pim_is_supported_register_size(size_t register_size) noexcept {
  return register_size == 1 || register_size == 2 || register_size == 4 ||
         register_size == 8;
}

/**
 * @brief Return bit mask for a supported register width.
 */
static inline uint64_t pim_register_value_mask(size_t register_size) noexcept {
  if (!pim_is_supported_register_size(register_size)) {
    return 0;
  }
  if (register_size == sizeof(uint64_t)) {
    return UINT64_MAX;
  }
  return (UINT64_C(1) << (register_size * 8U)) - UINT64_C(1);
}

/**
 * @brief Load a supported-width register as little-endian uint64.
 *
 * Returns false when arguments are invalid or the register size is unsupported.
 */
static inline bool pim_load_register_u64(const void *addr, size_t register_size,
                                         uint64_t *value) noexcept {
  if (!addr || !value || !pim_is_supported_register_size(register_size)) {
    if (value) {
      *value = 0;
    }
    return false;
  }

  const auto *bytes = reinterpret_cast<const uint8_t *>(addr);
  uint64_t loaded = 0;
  for (size_t i = 0; i < register_size; ++i) {
    loaded |= static_cast<uint64_t>(bytes[i]) << (i * 8U);
  }
  *value = loaded;
  return true;
}

/**
 * @brief Volatile variant of pim_load_register_u64().
 */
static inline bool pim_load_register_u64_volatile(const volatile void *addr,
                                                  size_t register_size,
                                                  uint64_t *value) noexcept {
  if (!addr || !value || !pim_is_supported_register_size(register_size)) {
    if (value) {
      *value = 0;
    }
    return false;
  }

  const auto *bytes = reinterpret_cast<const volatile uint8_t *>(addr);
  uint64_t loaded = 0;
  for (size_t i = 0; i < register_size; ++i) {
    loaded |= static_cast<uint64_t>(bytes[i]) << (i * 8U);
  }
  *value = loaded;
  return true;
}

/**
 * @brief Submit a task to the device's thread pool
 *
 * Thread-safe: uses pool_mutex internally.
 */
static inline bool pool_submit(pim_device_t *dev, std::function<void()> task) {
  std::lock_guard<std::mutex> lock(dev->pool_mutex);
  if (!dev->pool_running.load(std::memory_order_relaxed)) {
    return false;
  }
  try {
    dev->task_queue.push(std::move(task));
  } catch (...) {
    return false;
  }

  dev->pool_cv.notify_one();
  (void)dev->futex_cnt_commited.fetch_add(1u, std::memory_order_acq_rel);
  return true;
}

/**
 * @brief Return true if `id` matches a thread currently tracked in `threads`.
 *
 * CRUCIAL: Caller must serialize access to `threads` (e.g., via control mutex).
 */
static inline bool
pim_thread_vector_contains_id_locked(const std::vector<std::thread> &threads,
                                     const std::thread::id &id) noexcept {
  for (const auto &thread : threads) {
    if (thread.get_id() == id) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Return true if current thread is in `threads`.
 *
 * CRUCIAL: Caller must serialize access to `threads` (e.g., via control mutex).
 */
static inline bool pim_current_thread_in_thread_vector_locked(
    const std::vector<std::thread> &threads) noexcept {
  return pim_thread_vector_contains_id_locked(threads,
                                              std::this_thread::get_id());
}

/**
 * @brief Return true if current thread is a SIGSEGV callback worker.
 *
 * CRUCIAL: Caller must serialize access to `callbacks` (e.g., g_sigsegv_mutex).
 */
static inline bool pim_current_thread_in_sigsegv_callbacks_locked(
    const std::vector<std::unique_ptr<sigsegv_callback_ctx>>
        &callbacks) noexcept {
  const std::thread::id self = std::this_thread::get_id();
  for (const auto &ctx : callbacks) {
    if (ctx && ctx->thread.get_id() == self) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Join all joinable threads and clear the vector.
 */
static inline void
pim_join_and_clear_threads(std::vector<std::thread> &threads) noexcept {
  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  threads.clear();
}

// pim_pool_start / pim_pool_stop are declared in hostpimsim.h (public API)

/**
 * @brief Global SIGSEGV configuration mutex shared across compilation units.
 *
 * CRUCIAL: Region/handler topology mutations in hostpimsim.cc must take this
 * mutex to serialize with pim_device_enable_sigsegv()/disable and avoid
 * lock-free signal-handler races.
 */
extern std::mutex g_sigsegv_mutex;

/**
 * @brief Internal variant of pim_invoke_dpu_handler with optional state update.
 *
 * CRUCIAL: Some callers (default polling monitor) already updated cached
 * register bytes under region lock during scan and must dispatch without
 * writing them again.
 */
pim_error_t pim_invoke_dpu_handler_internal(pim_device_t *dev,
                                            pim_region_t *region, size_t offset,
                                            uint64_t value,
                                            bool update_last_value,
                                            uint32_t depth) noexcept;
