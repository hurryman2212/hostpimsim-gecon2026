/**
 * @file hostpimsim.cc
 * @brief Host-DRAM-like PIM Simulator Library Implementation
 *
 * This library provides a data-level simulation of PIM (Processing-In-Memory)
 * hardware, using shared memory (such as POSIX shared memory file or /dev/uio*
 * device file which supports mmap) as the backing store. Regions are views into
 * this shared memory at different offsets. Writes to control registers can
 * trigger DPU (Data Processing Unit) handlers.
 */

#include "internal.hh"

#include <cstring>

#include <sys/mman.h>
#include <sys/stat.h>

#include <linux/futex.h>

/*============================================================================
 * Internal Helper Functions
 *============================================================================*/

/**
 * @brief Check if an offset + size is within region bounds
 */
static inline bool check_bounds(pim_region_t *region, size_t offset,
                                size_t access_size) {
  return region && (offset <= region->size) &&
         (access_size <= region->size - offset);
}

/**
 * @brief Check alignment requirement for typed register access.
 */
static inline bool is_aligned(size_t offset, size_t alignment) {
  return (alignment == 0) || (offset % alignment == 0);
}

static inline bool pim_region_access_mode_to_prot(pim_access_mode_t access,
                                                  int *prot) noexcept {
  if (!prot) {
    return false;
  }
  switch (access) {
  case PIM_ACCESS_MODE_NONE:
    *prot = PROT_NONE;
    return true;
  case PIM_ACCESS_MODE_RO:
    *prot = PROT_READ;
    return true;
  case PIM_ACCESS_MODE_WO:
    *prot = PROT_WRITE;
    return true;
  case PIM_ACCESS_MODE_RW:
    *prot = PROT_READ | PROT_WRITE;
    return true;
  }
  return false;
}

static inline bool pim_region_page_aligned_range(pim_region_t *region,
                                                 uintptr_t *start,
                                                 size_t *size) noexcept {
  if (!region || !region->ptr || !start || !size || region->size == 0 ||
      region->page_size == 0) {
    return false;
  }
  const size_t page_size = region->page_size;
  if ((page_size & (page_size - 1U)) != 0U) {
    return false;
  }

  const uintptr_t addr = reinterpret_cast<uintptr_t>(region->ptr);
  if (region->size >
      static_cast<size_t>(std::numeric_limits<uintptr_t>::max() - addr)) {
    return false;
  }
  const uintptr_t end = addr + region->size;
  const uintptr_t mask = ~(static_cast<uintptr_t>(page_size) - 1U);
  const uintptr_t aligned_start = addr & mask;

  if (end > static_cast<uintptr_t>(std::numeric_limits<uintptr_t>::max() -
                                   (page_size - 1U))) {
    return false;
  }
  const uintptr_t aligned_end = (end + page_size - 1U) & mask;
  if (aligned_end < aligned_start) {
    return false;
  }

  const uintptr_t range_u = aligned_end - aligned_start;
  if (range_u == 0 ||
      range_u > static_cast<uintptr_t>(std::numeric_limits<size_t>::max())) {
    return false;
  }
  *start = aligned_start;
  *size = static_cast<size_t>(range_u);
  return true;
}

/**
 * @brief Return true when region topology must not be mutated.
 */
static inline bool is_topology_mutation_blocked(pim_device_t *dev) noexcept {
  return dev->teardown_in_progress.load(std::memory_order_acquire) ||
         dev->monitor_running.load(std::memory_order_acquire) ||
         dev->sigsegv_enabled.load(std::memory_order_acquire) ||
         dev->sigsegv_transition_in_progress.load(std::memory_order_acquire);
}

struct topology_mutation_scope {
  std::unique_lock<std::mutex> monitor_ctl_lock;
  std::unique_lock<std::mutex> sigsegv_lock;
};

/**
 * @brief Acquire topology-mutation locks and report if mutation is blocked.
 */
static inline bool
acquire_topology_mutation_scope(pim_device_t *dev,
                                topology_mutation_scope *scope) noexcept {
  if (!dev || !scope) {
    return false;
  }
  scope->monitor_ctl_lock =
      std::unique_lock<std::mutex>(dev->monitor_ctl_mutex);
  scope->sigsegv_lock = std::unique_lock<std::mutex>(g_sigsegv_mutex);
  return !is_topology_mutation_blocked(dev);
}

template <typename Obj, typename Ret>
static inline Ret get_member_or_default(Obj *obj, Ret Obj::*member,
                                        Ret fallback) noexcept {
  return obj ? obj->*member : fallback;
}

/**
 * @brief Validate a MMIO handler API target (dev/region ownership + type).
 */
static inline pim_error_t
validate_mmio_handler_target(pim_device_t *dev, pim_region_t *region) noexcept {
  if (!dev || !region) {
    return PIM_ERR_INVALID_ARG;
  }
  if (region->device != dev || region->type != PIM_REGION_CTRL_MMIO) {
    return PIM_ERR_INVALID_ARG;
  }
  return PIM_SUCCESS;
}

/**
 * @brief Validate if handler-table mutation is currently allowed.
 *
 * CRUCIAL: Caller must hold `g_sigsegv_mutex`.
 */
static inline pim_error_t
validate_handler_table_mutation_state_locked(pim_device_t *dev) noexcept {
  if (dev->teardown_in_progress.load(std::memory_order_acquire) ||
      dev->monitor_running.load(std::memory_order_acquire) ||
      dev->sigsegv_transition_in_progress.load(std::memory_order_acquire) ||
      dev->sigsegv_enabled.load(std::memory_order_acquire)) {
    return PIM_ERR_BUSY;
  }
  return PIM_SUCCESS;
}

/**
 * @brief Validate typed register access and return typed pointer.
 */
template <typename T>
static inline pim_error_t get_typed_region_ptr(pim_region_t *region,
                                               size_t offset,
                                               T **typed_ptr) noexcept {
  if (!region || !region->ptr || !typed_ptr) {
    return PIM_ERR_INVALID_ARG;
  }
  if (!is_aligned(offset, alignof(T))) {
    return PIM_ERR_INVALID_ARG;
  }
  if (!check_bounds(region, offset, sizeof(T))) {
    return PIM_ERR_INVALID_ARG;
  }

  *typed_ptr =
      reinterpret_cast<T *>(static_cast<uint8_t *>(region->ptr) + offset);
  return PIM_SUCCESS;
}

static inline pim_error_t get_region_byte_ptr(pim_region_t *region,
                                              size_t offset, size_t size,
                                              uint8_t **byte_ptr) noexcept {
  if (!region || !region->ptr || !byte_ptr) {
    return PIM_ERR_INVALID_ARG;
  }
  if (!check_bounds(region, offset, size)) {
    return PIM_ERR_INVALID_ARG;
  }
  *byte_ptr = static_cast<uint8_t *>(region->ptr) + offset;
  return PIM_SUCCESS;
}

/**
 * @brief Guard lock-free handler-table reads against structural mutations.
 */
class region_dispatch_guard {
public:
  explicit region_dispatch_guard(pim_region_t *region) noexcept
      : region_(region) {
    if (!region_) {
      return;
    }
    for (;;) {
      while (region_->handler_table_mutating.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      region_->active_dispatch_ops.fetch_add(1, std::memory_order_acq_rel);
      if (!region_->handler_table_mutating.load(std::memory_order_acquire)) {
        armed_ = true;
        return;
      }
      region_->active_dispatch_ops.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  ~region_dispatch_guard() noexcept {
    if (armed_ && region_) {
      region_->active_dispatch_ops.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  bool active() const noexcept { return armed_; }

private:
  pim_region_t *region_{nullptr};
  bool armed_{false};
};

/**
 * @brief Resolve MMIO handler metadata from lock-free reader paths.
 */
static inline bool resolve_mmio_handler(pim_region_t *region, size_t offset,
                                        uint64_t value, bool update_last_value,
                                        pim_dpu_handler_t *handler,
                                        void **user_data) {
  if (!handler || !user_data) {
    return false;
  }
  *handler = nullptr;
  *user_data = nullptr;

  if (!region || region->type != PIM_REGION_CTRL_MMIO) {
    return false;
  }

  auto it = region->handlers.find(offset);
  if (it == region->handlers.end()) {
    return false;
  }

  *handler = it->second.handler;
  *user_data = it->second.user_data;
  if (update_last_value) {
    dpu_handler_entry &entry = it->second;
    entry.last_value.store(value & pim_register_value_mask(entry.register_size),
                           std::memory_order_relaxed);
  }
  return true;
}

static inline bool
resolve_mmio_nested_handler(pim_region_t *region, size_t offset, uint32_t depth,
                            uint64_t value, bool update_last_value,
                            pim_dpu_handler_t *handler, void **user_data) {
  if (!handler || !user_data) {
    return false;
  }
  *handler = nullptr;
  *user_data = nullptr;

  if (!region || region->type != PIM_REGION_CTRL_MMIO || depth == 0u) {
    return false;
  }

  std::lock_guard<std::mutex> lock(region->mutex);
  const dpu_handler_key key{offset, depth};
  auto it = region->nested_handlers.find(key);
  if (it == region->nested_handlers.end()) {
    return false;
  }

  dpu_handler_entry &entry = it->second;
  *handler = entry.handler;
  *user_data = entry.user_data;
  if (update_last_value) {
    entry.last_value.store(value & pim_register_value_mask(entry.register_size),
                           std::memory_order_relaxed);
  }
  return true;
}

static inline void
mmio_handler_mark_write_begin(dpu_handler_entry *entry) noexcept {
  if (!entry) {
    return;
  }
  (void)entry->write_seq.fetch_add(1, std::memory_order_acq_rel); // odd
}

static inline void
mmio_handler_mark_write_end(dpu_handler_entry *entry) noexcept {
  if (!entry) {
    return;
  }
  (void)entry->write_seq.fetch_add(1, std::memory_order_release); // even
}

/**
 * @brief Invoke a DPU handler while containing all C++ exceptions.
 *
 * CRUCIAL: Public APIs are noexcept. User-provided callbacks must never be
 * allowed to unwind into library code paths.
 */
static inline void invoke_dpu_handler_noexcept(
    pim_device_t *dev, pim_region_t *region, size_t offset, uint32_t depth,
    uint64_t value, pim_dpu_handler_t handler, void *user_data) noexcept {
  if (!handler) {
    return;
  }
  try {
    handler(dev, region, offset, depth, value, user_data);
  } catch (...) {
    // Intentionally swallowed to preserve noexcept C API behavior.
  }
}

/**
 * @brief Dispatch a resolved DPU handler with pool/direct fallback policy.
 *
 * CRUCIAL: This is the common hot-path dispatch logic shared by callers that
 * already resolved handler metadata.
 */
static inline void
dispatch_resolved_dpu_handler(pim_device_t *dev, pim_region_t *region,
                              size_t offset, uint32_t depth, uint64_t value,
                              pim_dpu_handler_t handler, void *user_data) {
  // Dispatch policy:
  // 1) Prefer the shared pool for bounded concurrency.
  // 2) Fallback to direct call if pool cannot accept the task right now.
  if (dev->pool_running.load(std::memory_order_relaxed)) {
    try {
      if (!pool_submit(dev, region, offset, depth, value, handler, user_data)) {
        invoke_dpu_handler_noexcept(dev, region, offset, depth, value, handler,
                                    user_data);
      }
    } catch (...) {
      // CRUCIAL: Never leak C++ exceptions across C API paths.
      invoke_dpu_handler_noexcept(dev, region, offset, depth, value, handler,
                                  user_data);
    }
  } else {
    invoke_dpu_handler_noexcept(dev, region, offset, depth, value, handler,
                                user_data);
  }
}

/**
 * @brief Write to control register and invoke DPU handler.
 *
 * Lock-free hot path: write and handler-map lookup are done without
 * region->mutex to reduce write-path serialization.
 */
template <typename T>
static void write_and_invoke_handler(pim_region_t *region, size_t offset,
                                     T value, T *target) {
  pim_device_t *dev = region ? region->device : nullptr;
  bool sigsegv_active = false;
  pim_dpu_handler_t handler = nullptr;
  void *user_data = nullptr;
  region_dispatch_guard guard(region);
  if (!guard.active()) {
    return;
  }

  dpu_handler_entry *entry = nullptr;
  if (region && region->type == PIM_REGION_CTRL_MMIO) {
    auto it = region->handlers.find(offset);
    if (it != region->handlers.end()) {
      entry = &it->second;
    }
  }

  sigsegv_active = dev && dev->sigsegv_enabled.load(std::memory_order_relaxed);
  if (entry && !sigsegv_active) {
    mmio_handler_mark_write_begin(entry);
  }

  *target = value;

  if (entry) {
    handler = entry->handler;
    user_data = entry->user_data;
    if (!sigsegv_active) {
      entry->last_value.store(static_cast<uint64_t>(value) &
                                  pim_register_value_mask(entry->register_size),
                              std::memory_order_relaxed);
      mmio_handler_mark_write_end(entry);
    }
  }

  // MMIO command decode is ordering-sensitive. Run the top-level handler on the
  // caller thread so back-to-back writes cannot sit behind long DPU work in the
  // shared pool; nested per-DPU handlers still use the normal async path.
  if (!sigsegv_active && handler && dev) {
    invoke_dpu_handler_noexcept(dev, region, offset, 0u,
                                static_cast<uint64_t>(value), handler,
                                user_data);
  }
}

/**
 * @brief Write to register and update cached register bytes (no handler).
 */
template <typename T>
static void write_raw_atomic(pim_region_t *region, size_t offset, T value,
                             T *target) {
  region_dispatch_guard guard(region);
  if (!guard.active()) {
    return;
  }

  dpu_handler_entry *entry = nullptr;
  bool sigsegv_active = false;
  if (region && region->type == PIM_REGION_CTRL_MMIO) {
    auto it = region->handlers.find(offset);
    if (it != region->handlers.end()) {
      entry = &it->second;
    }
    pim_device_t *dev = region->device;
    sigsegv_active =
        dev && dev->sigsegv_enabled.load(std::memory_order_relaxed);
    if (entry && !sigsegv_active) {
      mmio_handler_mark_write_begin(entry);
    }
  }

  *target = value;

  // Update cached value so monitor won't re-trigger for this typed write.
  if (entry && !sigsegv_active) {
    entry->last_value.store(static_cast<uint64_t>(value) &
                                pim_register_value_mask(entry->register_size),
                            std::memory_order_relaxed);
    mmio_handler_mark_write_end(entry);
  }
}

template <typename T, bool RawWrite>
static inline pim_error_t write_typed_value(pim_region_t *region, size_t offset,
                                            T value) noexcept {
  T *target = nullptr;
  pim_error_t err = get_typed_region_ptr(region, offset, &target);
  if (err != PIM_SUCCESS) {
    return err;
  }

  if constexpr (RawWrite) {
    write_raw_atomic(region, offset, value, target);
  } else {
    write_and_invoke_handler(region, offset, value, target);
  }
  return PIM_SUCCESS;
}

template <typename T>
static inline pim_error_t read_typed_value(pim_region_t *region, size_t offset,
                                           T *value) noexcept {
  if (!value) {
    return PIM_ERR_INVALID_ARG;
  }
  T *source = nullptr;
  pim_error_t err = get_typed_region_ptr(region, offset, &source);
  if (err != PIM_SUCCESS) {
    return err;
  }
  *value = *source;
  return PIM_SUCCESS;
}

/*============================================================================
 * Device Management Implementation
 *============================================================================*/

/**
 * @brief Common setup path shared by create/attach/init.
 *
 * Constructs the device object with consistent core fields.
 * Optional zero-initialization is controlled by the caller.
 * Caller owns map/fd cleanup on nullptr return.
 */
static pim_device_t *pim_device_setup_common(void *ptr, size_t size, int fd,
                                             const char *path, bool init,
                                             bool backing_ownership) noexcept {
  if (init) {
    // Zero-initialize only when requested by caller.
    std::memset(ptr, 0, size);
  }

  auto *dev = new (std::nothrow) pim_device_t;
  if (!dev) {
    return nullptr;
  }

  dev->ptr = ptr;
  dev->size = size;
  dev->fd = fd;
  if (path) {
    dev->path = path;
  }
  dev->backing_ownership = backing_ownership;

  return dev;
}

static inline pim_device_t *
pim_device_fail_setup(void *ptr, size_t size, int fd,
                      const char *runtime_shm_to_unlink) noexcept {
  if (ptr && ptr != MAP_FAILED) {
    munmap(ptr, size);
  }
  if (fd >= 0) {
    close(fd);
  }
  if (runtime_shm_to_unlink) {
    shm_unlink(runtime_shm_to_unlink);
  }
  return nullptr;
}

pim_device_t *pim_device_create(const char *shm_path, size_t size) noexcept {
  if (!shm_path || size == 0) {
    return nullptr;
  }

  // CRUCIAL: Runtime-owned backing must be created fresh by this process.
  // O_EXCL prevents taking ownership of a pre-existing shm object.
  int fd = shm_open(shm_path, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd < 0) {
    return nullptr;
  }

  // Set the size
  if (ftruncate(fd, static_cast<off_t>(size)) < 0) {
    // CRUCIAL: create-path failure must clean up the just-created shm object.
    return pim_device_fail_setup(nullptr, 0, fd, shm_path);
  }

  // Map the entire memory
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    // CRUCIAL: avoid leaking orphaned shm objects on setup failure.
    return pim_device_fail_setup(nullptr, 0, fd, shm_path);
  }

  // Setup device with mapped memory
  pim_device_t *dev =
      pim_device_setup_common(ptr, size, fd, shm_path, true, true);
  if (!dev) {
    return pim_device_fail_setup(ptr, size, fd, shm_path);
  }

  return dev;
}

pim_device_t *pim_device_attach(const char *path, size_t size,
                                size_t offset) noexcept {
  if (!path) {
    return nullptr;
  }
  if (offset > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
    return nullptr;
  }
  const off_t map_offset = static_cast<off_t>(offset);

  // Open existing shm file (do NOT create)
  int fd = open(path, O_RDWR);
  if (fd < 0) {
    return nullptr;
  }

  struct stat st;
  if (fstat(fd, &st) < 0) {
    return pim_device_fail_setup(nullptr, 0, fd, nullptr);
  }

  const bool is_regular_file = S_ISREG(st.st_mode);
  size_t max_mappable_size = 0;
  if (is_regular_file) {
    if (st.st_size < 0) {
      return pim_device_fail_setup(nullptr, 0, fd, nullptr);
    }

    const uint64_t file_size = static_cast<uint64_t>(st.st_size);
    const uint64_t off = static_cast<uint64_t>(offset);
    if (off > file_size) {
      return pim_device_fail_setup(nullptr, 0, fd, nullptr);
    }

    const uint64_t remaining64 = file_size - off;
    if (remaining64 >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return pim_device_fail_setup(nullptr, 0, fd, nullptr);
    }
    max_mappable_size = static_cast<size_t>(remaining64);
  }

  // Query the file size if size not specified
  if (size == 0) {
    if (!is_regular_file) {
      // For non-regular files/devices, caller must provide explicit map size.
      return pim_device_fail_setup(nullptr, 0, fd, nullptr);
    }
    size = max_mappable_size;
  } else if (is_regular_file && size > max_mappable_size) {
    return pim_device_fail_setup(nullptr, 0, fd, nullptr);
  }

  if (size == 0) {
    return pim_device_fail_setup(nullptr, 0, fd, nullptr);
  }

  // Map the memory at specified offset
  void *ptr =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, map_offset);
  if (ptr == MAP_FAILED) {
    return pim_device_fail_setup(nullptr, 0, fd, nullptr);
  }

  // Setup device with mapped memory
  pim_device_t *dev = pim_device_setup_common(ptr, size, fd, path, true, false);
  if (!dev) {
    return pim_device_fail_setup(ptr, size, fd, nullptr);
  }

  return dev;
}

pim_device_t *pim_device_init(void *ptr, size_t size) noexcept {
  if (!ptr || size == 0) {
    return nullptr;
  }

  return pim_device_setup_common(ptr, size, -1, nullptr, false, false);
}

/**
 * @brief Common teardown path shared by deinit/destroy.
 *
 * Stops monitor/SIGSEGV/pool and frees region objects.
 * Does NOT unmap memory, close fd, or delete the device.
 *
 * @return PIM_SUCCESS on full cleanup, or the first teardown error code.
 */
static bool pim_is_internal_runtime_thread(pim_device_t *dev) noexcept {
  if (!dev) {
    return false;
  }

  // Monitor workers
  {
    std::lock_guard<std::mutex> monitor_ctl_lock(dev->monitor_ctl_mutex);
    if (pim_current_thread_in_thread_vector_locked(dev->monitor_threads)) {
      return true;
    }
  }

  // Pool workers
  {
    std::lock_guard<std::mutex> pool_ctl_lock(dev->pool_ctl_mutex);
    if (pim_current_thread_in_thread_vector_locked(dev->handler_pool)) {
      return true;
    }
  }

  // SIGSEGV callback workers
  {
    std::lock_guard<std::mutex> sigsegv_lock(g_sigsegv_mutex);
    if (pim_current_thread_in_sigsegv_callbacks_locked(
            dev->sigsegv_callbacks)) {
      return true;
    }
  }

  return false;
}

static pim_error_t pim_device_cleanup_common(pim_device_t *dev) noexcept {
  // CRUCIAL: Teardown from an internal worker thread can self-join and
  // terminate the process. Require teardown from a non-runtime thread.
  if (pim_is_internal_runtime_thread(dev)) {
    return PIM_ERR_BUSY;
  }

  // Always call stop: monitor_running can already be false after fail-stop
  // worker exit, while monitor_threads still hold joinable threads.
  (void)pim_stop_monitor(dev);

  // Pool teardown is independent and idempotent.
  pim_pool_stop(dev);

  pim_error_t sigsegv_err = pim_device_disable_sigsegv(dev);
  if (sigsegv_err != PIM_SUCCESS) {
    // CRUCIAL: If SIGSEGV teardown fails, do not free regions/device memory.
    // Callback/signal paths may still reference those objects.
    return sigsegv_err;
  }

  // Free all regions
  std::lock_guard<std::mutex> lock(dev->mutex);
  for (const auto &entry : dev->regions) {
    delete entry.second;
  }
  dev->regions.clear();
  return PIM_SUCCESS;
}

static inline pim_error_t
pim_device_begin_teardown(pim_device_t *dev) noexcept {
  if (!dev) {
    return PIM_ERR_INVALID_ARG;
  }

  bool expected = false;
  if (!dev->teardown_in_progress.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return PIM_ERR_BUSY;
  }
  return PIM_SUCCESS;
}

static pim_error_t pim_device_finalize(pim_device_t *dev,
                                       bool destroy_backing) noexcept {
  pim_error_t begin_err = pim_device_begin_teardown(dev);
  if (begin_err != PIM_SUCCESS) {
    return begin_err;
  }

  // Reuse common teardown; backing cleanup depends on finalization mode.
  pim_error_t err = pim_device_cleanup_common(dev);
  if (err != PIM_SUCCESS) {
    dev->teardown_in_progress.store(false, std::memory_order_release);
    return err;
  }

  if (destroy_backing) {
    // Unmap and close backing for destroy mode.
    if (dev->ptr && dev->ptr != MAP_FAILED) {
      munmap(dev->ptr, dev->size);
    }
    if (dev->fd >= 0) {
      close(dev->fd);
    }
  }

  delete dev;
  return PIM_SUCCESS;
}

pim_error_t pim_device_deinit(pim_device_t *dev) noexcept {
  return pim_device_finalize(dev, false);
}

pim_error_t pim_device_destroy(pim_device_t *dev) noexcept {
  return pim_device_finalize(dev, true);
}

pim_error_t pim_device_unlink(pim_device_t *dev) noexcept {
  if (!dev) {
    return PIM_ERR_INVALID_ARG;
  }

  // Strict policy: only unlink backings this runtime created itself.
  // Attached/init devices may point to user-managed files or device nodes.
  if (!dev->backing_ownership) {
    return PIM_ERR_NOT_SUPPORTED;
  }

  if (shm_unlink(dev->path.c_str()) < 0) {
    return PIM_ERR_SHM_OPEN;
  }

  return PIM_SUCCESS;
}

const char *pim_device_get_path(pim_device_t *dev) noexcept {
  return dev ? dev->path.c_str() : nullptr;
}

size_t pim_device_get_size(pim_device_t *dev) noexcept {
  return get_member_or_default(dev, &pim_device_t::size, size_t{0});
}

size_t pim_device_get_ram_size(pim_device_t *dev) noexcept {
  if (!dev) {
    return 0;
  }

  std::lock_guard<std::mutex> lock(dev->mutex);
  size_t total = 0;
  for (const auto &entry : dev->regions) {
    const pim_region_t *region = entry.second;
    if (region && region->type == PIM_REGION_RAM) {
      total += region->size;
    }
  }
  return total;
}

void *pim_device_get_ptr(pim_device_t *dev) noexcept {
  return get_member_or_default(dev, &pim_device_t::ptr,
                               static_cast<void *>(nullptr));
}

pim_region_t *pim_device_get_region(pim_device_t *dev, size_t offset) noexcept {
  if (!dev) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(dev->mutex);
  auto it = dev->regions.find(offset);
  return (it != dev->regions.end()) ? it->second : nullptr;
}

pim_region_t *pim_device_get_next_region(pim_device_t *dev,
                                         pim_region_t *region) noexcept {
  if (!dev) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(dev->mutex);
  if (dev->regions.empty()) {
    return nullptr;
  }
  if (!region) {
    return dev->regions.begin()->second;
  }

  auto it = dev->regions.find(region->offset);
  if (it == dev->regions.end() || it->second != region) {
    return nullptr;
  }
  ++it;
  return (it != dev->regions.end()) ? it->second : nullptr;
}

pim_error_t pim_device_user_lock(pim_device_t *dev) noexcept {
  if (!dev) {
    return PIM_ERR_INVALID_ARG;
  }
  dev->user_mutex.lock();
  return PIM_SUCCESS;
}

pim_error_t pim_device_user_unlock(pim_device_t *dev) noexcept {
  if (!dev) {
    return PIM_ERR_INVALID_ARG;
  }
  dev->user_mutex.unlock();
  return PIM_SUCCESS;
}

/*============================================================================
 * Region Management Implementation
 *============================================================================*/

pim_region_t *pim_region_create(pim_device_t *dev, size_t offset, size_t size,
                                pim_region_type_t type) noexcept {
  if (!dev || size == 0) {
    return nullptr;
  }

  topology_mutation_scope scope;
  if (!acquire_topology_mutation_scope(dev, &scope)) {
    return nullptr;
  }

  // Check bounds
  if (offset > dev->size || size > dev->size - offset) {
    return nullptr;
  }

  const size_t page_size = sysconf(_SC_PAGESIZE);
  if (page_size == -1U) {
    return nullptr;
  }

  // Check for overlaps against adjacent ranges in offset order.
  {
    std::lock_guard<std::mutex> lock(dev->mutex);
    const size_t new_end = offset + size;
    auto next_it = dev->regions.lower_bound(offset);
    if (next_it != dev->regions.end()) {
      pim_region_t *next = next_it->second;
      if (new_end > next->offset) {
        return nullptr; // Overlaps next
      }
    }
    if (next_it != dev->regions.begin()) {
      auto prev_it = std::prev(next_it);
      pim_region_t *prev = prev_it->second;
      if (prev->offset + prev->size > offset) {
        return nullptr; // Overlaps previous
      }
    }
  }

  // Create the region
  auto *region = new (std::nothrow) pim_region_t;
  if (!region) {
    return nullptr;
  }

  region->device = dev;
  region->offset = offset;
  region->size = size;
  region->type = type;
  region->page_size = page_size;
  region->ptr = static_cast<uint8_t *>(dev->ptr) + offset;

  // Add to device's region map
  {
    std::lock_guard<std::mutex> lock(dev->mutex);
    try {
      auto inserted = dev->regions.emplace(offset, region);
      if (!inserted.second) {
        delete region;
        return nullptr;
      }
    } catch (...) {
      delete region;
      return nullptr;
    }
  }

  return region;
}

pim_error_t pim_region_free(pim_region_t *region) noexcept {
  if (!region || !region->device) {
    return PIM_ERR_INVALID_ARG;
  }

  pim_device_t *dev = region->device;

  topology_mutation_scope scope;
  if (!acquire_topology_mutation_scope(dev, &scope)) {
    return PIM_ERR_BUSY;
  }

  std::lock_guard<std::mutex> lock(dev->mutex);

  // Find and remove from device's region map
  auto it = dev->regions.find(region->offset);
  if (it == dev->regions.end() || it->second != region) {
    return PIM_ERR_NOT_FOUND;
  }
  dev->regions.erase(it);

  delete region;
  return PIM_SUCCESS;
}

void *pim_region_get_ptr(pim_region_t *region) noexcept {
  return get_member_or_default(region, &pim_region_t::ptr,
                               static_cast<void *>(nullptr));
}

size_t pim_region_get_size(pim_region_t *region) noexcept {
  return get_member_or_default(region, &pim_region_t::size, size_t{0});
}

size_t pim_region_get_offset(pim_region_t *region) noexcept {
  return get_member_or_default(region, &pim_region_t::offset, size_t{0});
}

pim_device_t *pim_region_get_device(pim_region_t *region) noexcept {
  return get_member_or_default(region, &pim_region_t::device,
                               static_cast<pim_device_t *>(nullptr));
}

pim_error_t pim_region_user_lock(pim_region_t *region) noexcept {
  if (!region) {
    return PIM_ERR_INVALID_ARG;
  }
  region->user_mutex.lock();
  return PIM_SUCCESS;
}

pim_error_t pim_region_user_unlock(pim_region_t *region) noexcept {
  if (!region) {
    return PIM_ERR_INVALID_ARG;
  }
  region->user_mutex.unlock();
  return PIM_SUCCESS;
}

/*============================================================================
 * DPU Handler Registration Implementation
 *============================================================================*/

pim_error_t pim_register_dpu_handler(pim_device_t *dev, pim_region_t *region,
                                     size_t offset, uint32_t depth,
                                     size_t register_size,
                                     pim_dpu_handler_t handler,
                                     void *user_data) noexcept {
  pim_error_t target_err = validate_mmio_handler_target(dev, region);
  if (target_err != PIM_SUCCESS || !handler) {
    return PIM_ERR_INVALID_ARG;
  }

  // CRUCIAL: Serialize handler table mutation against SIGSEGV enable/disable.
  std::lock_guard<std::mutex> sigsegv_lock(g_sigsegv_mutex);

  pim_error_t state_err = validate_handler_table_mutation_state_locked(dev);
  if (state_err != PIM_SUCCESS) {
    return state_err;
  }

  if (!pim_is_supported_register_size(register_size)) {
    return PIM_ERR_INVALID_ARG;
  }
  if (!is_aligned(offset, register_size)) {
    return PIM_ERR_INVALID_ARG;
  }

  // Overflow-safe bound check.
  if (offset > region->size || register_size > region->size - offset) {
    return PIM_ERR_INVALID_ARG;
  }

  std::lock_guard<std::mutex> lock(region->mutex);
  region->handler_table_mutating.store(true, std::memory_order_release);
  if (region->active_dispatch_ops.load(std::memory_order_acquire) != 0U) {
    region->handler_table_mutating.store(false, std::memory_order_release);
    return PIM_ERR_BUSY;
  }

  // Reject overlap at the same depth only.
  const size_t new_start = offset;
  const size_t new_end = offset + register_size;
  if (depth == 0u) {
    for (const auto &[existing_offset, existing] : region->handlers) {
      const size_t existing_start = existing_offset;
      const size_t existing_end = existing_offset + existing.register_size;
      if (new_start < existing_end && existing_start < new_end) {
        region->handler_table_mutating.store(false, std::memory_order_release);
        return PIM_ERR_ALREADY_EXISTS;
      }
    }
  } else {
    for (const auto &[existing_key, existing] : region->nested_handlers) {
      if (existing_key.depth != depth) {
        continue;
      }
      const size_t existing_start = existing_key.offset;
      const size_t existing_end = existing_key.offset + existing.register_size;
      if (new_start < existing_end && existing_start < new_end) {
        region->handler_table_mutating.store(false, std::memory_order_release);
        return PIM_ERR_ALREADY_EXISTS;
      }
    }
  }

  // Snapshot current register bytes as initial monitor baseline.
  auto *src = static_cast<uint8_t *>(region->ptr) + offset;
  dpu_handler_entry new_entry;
  new_entry.handler = handler;
  new_entry.user_data = user_data;
  new_entry.register_size = register_size;
  new_entry.depth = depth;
  uint64_t initial_value = 0;
  if (!pim_load_register_u64(src, register_size, &initial_value)) {
    region->handler_table_mutating.store(false, std::memory_order_release);
    return PIM_ERR_INVALID_ARG;
  }
  new_entry.last_value.store(initial_value, std::memory_order_relaxed);

  try {
    if (depth == 0u) {
      region->handlers[offset] = std::move(new_entry);
    } else {
      region->nested_handlers[dpu_handler_key{offset, depth}] =
          std::move(new_entry);
    }
  } catch (...) {
    region->handler_table_mutating.store(false, std::memory_order_release);
    return PIM_ERR_NO_MEMORY;
  }
  region->handler_table_mutating.store(false, std::memory_order_release);
  return PIM_SUCCESS;
}

pim_error_t pim_unregister_dpu_handler(pim_device_t *dev, pim_region_t *region,
                                       size_t offset, uint32_t depth) noexcept {
  pim_error_t target_err = validate_mmio_handler_target(dev, region);
  if (target_err != PIM_SUCCESS) {
    return target_err;
  }

  // CRUCIAL: Serialize handler table mutation against SIGSEGV enable/disable.
  std::lock_guard<std::mutex> sigsegv_lock(g_sigsegv_mutex);

  pim_error_t state_err = validate_handler_table_mutation_state_locked(dev);
  if (state_err != PIM_SUCCESS) {
    return state_err;
  }

  std::lock_guard<std::mutex> lock(region->mutex);
  region->handler_table_mutating.store(true, std::memory_order_release);
  if (region->active_dispatch_ops.load(std::memory_order_acquire) != 0U) {
    region->handler_table_mutating.store(false, std::memory_order_release);
    return PIM_ERR_BUSY;
  }

  bool removed = false;

  if (depth == 0u) {
    auto it = region->handlers.find(offset);
    if (it != region->handlers.end()) {
      region->handlers.erase(it);
      removed = true;
    }
  } else {
    auto nested_it =
        region->nested_handlers.find(dpu_handler_key{offset, depth});
    if (nested_it != region->nested_handlers.end()) {
      region->nested_handlers.erase(nested_it);
      removed = true;
    }
  }

  region->handler_table_mutating.store(false, std::memory_order_release);
  return removed ? PIM_SUCCESS : PIM_ERR_NOT_FOUND;
}

pim_error_t pim_invoke_dpu_handler_internal(pim_device_t *dev,
                                            pim_region_t *region, size_t offset,
                                            uint64_t value,
                                            bool update_last_value,
                                            uint32_t depth) noexcept {
  pim_error_t target_err = validate_mmio_handler_target(dev, region);
  if (target_err != PIM_SUCCESS) {
    return target_err;
  }

  pim_dpu_handler_t handler = nullptr;
  void *user_data = nullptr;
  region_dispatch_guard guard(region);
  if (!guard.active()) {
    return PIM_ERR_BUSY;
  }

  bool resolved = false;
  if (depth == 0u) {
    resolved = resolve_mmio_handler(region, offset, value, update_last_value,
                                    &handler, &user_data);
  } else {
    resolved = resolve_mmio_nested_handler(
        region, offset, depth, value, update_last_value, &handler, &user_data);
  }

  if (!resolved) {
    return PIM_ERR_NOT_FOUND;
  }

  dispatch_resolved_dpu_handler(dev, region, offset, depth, value, handler,
                                user_data);

  return PIM_SUCCESS;
}

static inline pim_error_t
pim_wait_pool_barrier(std::atomic<uint32_t> *futex_cnt_commited,
                      std::atomic<uint32_t> *futex_cnt_finished) noexcept {
  if (!futex_cnt_commited || !futex_cnt_finished) {
    return PIM_ERR_INVALID_ARG;
  }
  for (;;) {
    const uint32_t finished =
        futex_cnt_finished->load(std::memory_order_acquire);
    const uint32_t commited =
        futex_cnt_commited->load(std::memory_order_acquire);

    if (finished == commited) {
      return PIM_SUCCESS;
    }

    (void)futex(reinterpret_cast<int *>(futex_cnt_finished), FUTEX_WAIT_PRIVATE,
                static_cast<int>(finished), nullptr, nullptr, 0);
  }
}

pim_error_t pim_invoke_dpu_handler(pim_region_t *region, size_t offset,
                                   uint32_t depth, uint64_t value) noexcept {
  pim_device_t *dev = region ? region->device : nullptr;
  return pim_invoke_dpu_handler_internal(dev, region, offset, value, true,
                                         depth);
}

/*============================================================================
 * Memory Access Implementation
 *============================================================================*/

pim_error_t pim_device_barrier(pim_device_t *dev) noexcept {
  if (!dev) {
    return PIM_ERR_INVALID_ARG;
  }

  {
    std::lock_guard<std::mutex> pool_ctl_lock(dev->pool_ctl_mutex);
    if (pim_current_thread_in_thread_vector_locked(dev->handler_pool)) {
      return PIM_ERR_WOULD_DEADLOCK;
    }
  }
  return pim_wait_pool_barrier(&dev->futex_cnt_commited,
                               &dev->futex_cnt_finished);
}

pim_error_t pim_region_barrier(pim_region_t *region) noexcept {
  if (!region || !region->device) {
    return PIM_ERR_INVALID_ARG;
  }

  pim_device_t *dev = region->device;
  {
    std::lock_guard<std::mutex> pool_ctl_lock(dev->pool_ctl_mutex);
    if (pim_current_thread_in_thread_vector_locked(dev->handler_pool)) {
      return PIM_ERR_WOULD_DEADLOCK;
    }
  }
  return pim_wait_pool_barrier(&region->futex_cnt_commited,
                               &region->futex_cnt_finished);
}

static inline bool pim_is_supported_access_size(size_t size) noexcept {
  return size == 1u || size == 2u || size == 4u || size == 8u;
}

template <bool RawWrite>
static inline pim_error_t write_sized_value(pim_region_t *region, size_t offset,
                                            size_t size,
                                            uint64_t value) noexcept {
  if (!pim_is_supported_access_size(size)) {
    return PIM_ERR_INVALID_ARG;
  }

  switch (size) {
  case 1u:
    return write_typed_value<uint8_t, RawWrite>(region, offset,
                                                static_cast<uint8_t>(value));
  case 2u:
    return write_typed_value<uint16_t, RawWrite>(region, offset,
                                                 static_cast<uint16_t>(value));
  case 4u:
    return write_typed_value<uint32_t, RawWrite>(region, offset,
                                                 static_cast<uint32_t>(value));
  case 8u:
    return write_typed_value<uint64_t, RawWrite>(region, offset,
                                                 static_cast<uint64_t>(value));
  default:
    return PIM_ERR_INVALID_ARG;
  }
}

static inline pim_error_t read_sized_value(pim_region_t *region, size_t offset,
                                           size_t size,
                                           uint64_t *value) noexcept {
  if (!value || !pim_is_supported_access_size(size)) {
    return PIM_ERR_INVALID_ARG;
  }

  switch (size) {
  case 1u: {
    uint8_t v = 0;
    pim_error_t err = read_typed_value<uint8_t>(region, offset, &v);
    if (err != PIM_SUCCESS) {
      return err;
    }
    *value = static_cast<uint64_t>(v);
    return PIM_SUCCESS;
  }
  case 2u: {
    uint16_t v = 0;
    pim_error_t err = read_typed_value<uint16_t>(region, offset, &v);
    if (err != PIM_SUCCESS) {
      return err;
    }
    *value = static_cast<uint64_t>(v);
    return PIM_SUCCESS;
  }
  case 4u: {
    uint32_t v = 0;
    pim_error_t err = read_typed_value<uint32_t>(region, offset, &v);
    if (err != PIM_SUCCESS) {
      return err;
    }
    *value = static_cast<uint64_t>(v);
    return PIM_SUCCESS;
  }
  case 8u: {
    uint64_t v = 0;
    pim_error_t err = read_typed_value<uint64_t>(region, offset, &v);
    if (err != PIM_SUCCESS) {
      return err;
    }
    *value = v;
    return PIM_SUCCESS;
  }
  default:
    return PIM_ERR_INVALID_ARG;
  }
}

pim_error_t pim_write(pim_region_t *region, size_t offset, size_t size,
                      uint64_t value) noexcept {
  return write_sized_value<false>(region, offset, size, value);
}

pim_error_t pim_write_sync(pim_region_t *region, size_t offset, size_t size,
                           uint64_t value) noexcept {
  pim_error_t err = write_sized_value<false>(region, offset, size, value);
  if (err != PIM_SUCCESS) {
    return err;
  }
  return pim_region_barrier(region);
}

pim_error_t pim_write_raw(pim_region_t *region, size_t offset, size_t size,
                          uint64_t value) noexcept {
  return write_sized_value<true>(region, offset, size, value);
}

pim_error_t pim_read(pim_region_t *region, size_t offset, size_t size,
                     uint64_t *value) noexcept {
  return read_sized_value(region, offset, size, value);
}

pim_error_t pim_memcpy_to(pim_region_t *region, size_t offset, const void *data,
                          size_t size) noexcept {
  if (!data) {
    return PIM_ERR_INVALID_ARG;
  }
  uint8_t *target = nullptr;
  pim_error_t err = get_region_byte_ptr(region, offset, size, &target);
  if (err != PIM_SUCCESS) {
    return err;
  }
  std::memcpy(target, data, size);

  return PIM_SUCCESS;
}

pim_error_t pim_memcpy_from(pim_region_t *region, size_t offset, void *data,
                            size_t size) noexcept {
  if (!data) {
    return PIM_ERR_INVALID_ARG;
  }
  uint8_t *source = nullptr;
  pim_error_t err = get_region_byte_ptr(region, offset, size, &source);
  if (err != PIM_SUCCESS) {
    return err;
  }
  std::memcpy(data, source, size);

  return PIM_SUCCESS;
}

pim_error_t pim_region_msync(pim_region_t *region) noexcept {
  if (!region || !region->ptr) {
    return PIM_ERR_INVALID_ARG;
  }
  if (msync(region->ptr, region->size, MS_SYNC) < 0) {
    return pim_return_with_errno(PIM_ERR_MSYNC, EIO);
  }
  return PIM_SUCCESS;
}

pim_error_t pim_region_mlock(pim_region_t *region) noexcept {
  if (!region || !region->ptr) {
    return PIM_ERR_INVALID_ARG;
  }
  if (mlock(region->ptr, region->size) < 0) {
    return pim_return_with_errno(PIM_ERR_MLOCK, EIO);
  }
  return PIM_SUCCESS;
}

pim_error_t pim_region_munlock(pim_region_t *region) noexcept {
  if (!region || !region->ptr) {
    return PIM_ERR_INVALID_ARG;
  }
  if (munlock(region->ptr, region->size) < 0) {
    return pim_return_with_errno(PIM_ERR_MUNLOCK, EIO);
  }
  return PIM_SUCCESS;
}

pim_error_t pim_region_mprotect(pim_region_t *region,
                                pim_access_mode_t access) noexcept {
  if (!region || !region->ptr) {
    return PIM_ERR_INVALID_ARG;
  }

  int prot = 0;
  if (!pim_region_access_mode_to_prot(access, &prot)) {
    return PIM_ERR_INVALID_ARG;
  }

  uintptr_t aligned_start = 0;
  size_t aligned_size = 0;
  if (!pim_region_page_aligned_range(region, &aligned_start, &aligned_size)) {
    return PIM_ERR_INVALID_ARG;
  }

  if (mprotect(reinterpret_cast<void *>(aligned_start), aligned_size, prot) <
      0) {
    return pim_return_with_errno(PIM_ERR_MPROTECT, EIO);
  }
  return PIM_SUCCESS;
}

pim_error_t pim_region_memset(pim_region_t *region, int value,
                              size_t size) noexcept {
  uint8_t *target = nullptr;
  pim_error_t err = get_region_byte_ptr(region, 0, size, &target);
  if (err != PIM_SUCCESS) {
    return err;
  }
  std::memset(target, value, size);
  return PIM_SUCCESS;
}
