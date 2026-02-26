#pragma once

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#include <fcntl.h>

#include <sys/eventfd.h>
#include <sys/wait.h>

#ifdef __cplusplus
#define PIM_NOEXCEPT noexcept
#else
#define PIM_NOEXCEPT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Type Definitions
 *============================================================================*/

/**
 * @brief Region types
 */
typedef enum pim_region_type {
  PIM_REGION_RAM = 0, /**< Regular RAM region */
  PIM_REGION_CTRL_MMIO =
      1, /**< MMIO-style control region (host writes trigger DPU handlers) */
  PIM_REGION_CTRL_RW =
      2, /**< Read-write control region (DPU writes, host reads) */
} pim_region_type_t;

/**
 * @brief Error codes
 */
typedef enum pim_error {
  PIM_SUCCESS = 0,             /**< Operation completed successfully */
  PIM_ERR_INVALID_ARG = -1,    /**< One or more arguments are invalid */
  PIM_ERR_NO_MEMORY = -2,      /**< Allocation failed (heap/container growth) */
  PIM_ERR_SHM_OPEN = -3,       /**< POSIX shm open/unlink operation failed */
  PIM_ERR_NOT_FOUND = -4,      /**< Requested resource/entry was not found */
  PIM_ERR_ALREADY_EXISTS = -5, /**< Resource/entry already exists or overlaps */
  PIM_ERR_OUT_OF_BOUNDS = -6,  /**< Requested range exceeds region bounds */
  PIM_ERR_SIGNAL = -7, /**< Signal handler registration/query/restore failed */
  PIM_ERR_BUSY = -8,   /**< Operation rejected due to active runtime state */
  PIM_ERR_NOT_SUPPORTED = -9, /**< Operation is intentionally unsupported */
  PIM_ERR_WOULD_DEADLOCK =
      -10,                /**< Operation would deadlock in caller context */
  PIM_ERR_MMAP = -11,     /**< mmap() failed (see errno) */
  PIM_ERR_MSYNC = -12,    /**< msync() failed (see errno) */
  PIM_ERR_MPROTECT = -13, /**< mprotect() failed (see errno) */
  PIM_ERR_MLOCK = -14,    /**< mlock() failed (see errno) */
  PIM_ERR_MUNLOCK = -15,  /**< munlock() failed (see errno) */
} pim_error_t;

/**
 * @brief Access modes
 */
typedef enum pim_access_mode {
  PIM_ACCESS_MODE_NONE = 0, /**< No access */
  PIM_ACCESS_MODE_RO = 1,   /**< Read-only mode */
  PIM_ACCESS_MODE_WO = 2,   /**< Write-only mode */
  PIM_ACCESS_MODE_RW = 3,   /**< Read-write mode */
} pim_access_mode_t;

/**
 * @brief Opaque handle for a PIM device
 */
typedef struct pim_device pim_device_t;

/**
 * @brief Opaque handle for a memory region (view into device memory)
 */
typedef struct pim_region pim_region_t;

/**
 * @brief DPU command handler callback type
 *
 * @param dev       The PIM device
 * @param region    The region where the trigger write occurred
 * @param offset    Offset within the region where write occurred
 * @param value     The written value payload
 * @param user_data User-provided context data
 */
typedef void (*pim_dpu_handler_t)(pim_device_t *dev, pim_region_t *region,
                                  size_t offset, uint64_t value,
                                  void *user_data);

/*============================================================================
 * Device Allocation (each function is mutually exclusive!)
 *============================================================================*/

/**
 * @brief Create a new PIM device backed by a SINGLE shared memory file
 *
 * This creates and maps a single shm file that contains all memory regions.
 * Regions are then created as views into this shared memory space.
 * The object must not already exist; this API uses exclusive creation.
 * This wipes the memory to zero.
 *
 * @param shm_path   Path for shm file (e.g., "/pim")
 *                   Will be created under /dev/shm/
 * @param size       Size of the shared memory in bytes
 * @return           Device handle on success, NULL on failure
 */
pim_device_t *pim_device_create(const char *shm_path, size_t size) PIM_NOEXCEPT;

/**
 * @brief Attach to an existing PIM device shared memory file
 *
 * Opens an existing shared memory or mmap-capable file (does NOT create if
 * missing) and maps a portion.
 * This wipes the memory to zero.
 *
 * @param path       Path for shared memory object or mmap-capable file
 * @param size       Size of the region to map
 *                   - size > 0: map exactly this byte count
 *                   - size == 0: use (st_size - offset) only for regular files
 *                                with a meaningful stat size
 *                                (non-regular files/devices must pass size > 0)
 * @param offset     Offset within the file to start mapping
 * @return           Device handle on success, NULL if file doesn't exist or
 * error
 */
pim_device_t *pim_device_attach(const char *path, size_t size,
                                size_t offset) PIM_NOEXCEPT;

/**
 * @brief Setup a PIM device with already mapped memory
 *
 * Creates a device handle for externally mmap-ed memory. The caller is
 * responsible for managing the memory lifetime. Call pim_device_deinit()
 * instead of pim_device_destroy() to release the handle without unmapping.
 *
 * @param ptr        Pointer to already mapped memory
 * @param size       Size of the mapped region in bytes
 * @return           Device handle on success, NULL on failure
 */
pim_device_t *pim_device_init(void *ptr, size_t size) PIM_NOEXCEPT;

/*============================================================================
 * Device Deallocation (phase-specific; not interchangeable)
 *============================================================================*/

/**
 * @brief Deinitialize a PIM device without unmapping memory
 *
 * Use this for devices created with pim_device_init(). Does NOT unmap
 * or close any file descriptors. Do not use this for create/attach devices
 * unless you explicitly want to keep their mapping/fd alive.
 *
 * @param dev        Device handle
 * @return           PIM_SUCCESS on success, error code on failure
 *                   (PIM_ERR_BUSY if called from internal runtime worker
 *                   context or during concurrent teardown)
 */
pim_error_t pim_device_deinit(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Destroy a PIM device and free all resources
 *
 * This will unmap memory and close the backing fd, but does NOT unlink the
 * shm object/path.
 * Call pim_device_unlink() before destroy if you want to remove the shm file.
 *
 * @param dev    Device handle
 * @return       PIM_SUCCESS on success, error code on failure
 *               (PIM_ERR_BUSY if called from internal runtime worker
 *               context or during concurrent teardown)
 */
pim_error_t pim_device_destroy(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Unlink the device backing object
 *
 * Strict policy: this succeeds only for a backing object created by
 * pim_device_create() in this runtime. It calls shm_unlink() on that POSIX
 * shared-memory object.
 *
 * For pim_device_attach() / pim_device_init() handles, this returns
 * PIM_ERR_NOT_SUPPORTED to avoid deleting user-managed files or device nodes.
 *
 * @param dev    Device handle
 * @return       PIM_SUCCESS or error code
 */
pim_error_t pim_device_unlink(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Get device shared memory or device path
 *
 * @param dev    Device handle
 * @return       Backed shared memory or device path string
 */
const char *pim_device_get_path(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Get mapped device memory size
 *
 * @param dev    Device handle
 * @return       Mapped device memory size in bytes
 */
size_t pim_device_get_size(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Get total size of RAM regions in a device
 *
 * Sums sizes of regions whose type is `PIM_REGION_RAM`.
 *
 * @param dev    Device handle
 * @return       Total RAM-region size in bytes
 */
size_t pim_device_get_ram_size(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Get the base memory-mapped pointer for the entire device
 *
 * @param dev    Device handle
 * @return       Pointer to mapped memory, NULL on error
 */
void *pim_device_get_ptr(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Attach opaque user data to a device
 *
 * @param dev         Device handle
 * @param user_data   Caller-owned opaque pointer
 */
void pim_device_set_user_data(pim_device_t *dev, void *user_data) PIM_NOEXCEPT;

/**
 * @brief Read opaque user data from a device
 *
 * @param dev    Device handle
 * @return       Caller-owned opaque pointer, or NULL
 */
void *pim_device_get_user_data(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Get the matching region by its start offset within a device
 *
 * @param dev       Device handle
 * @param offset    Region start offset in bytes
 * @return          Region handle if found, otherwise NULL
 */
pim_region_t *pim_device_get_region(pim_device_t *dev,
                                    size_t offset) PIM_NOEXCEPT;

/**
 * @brief Iterate regions in ascending offset order
 *
 * @param dev       Device handle
 * @param region    Current region, or NULL to get the first region
 * @return          Next region, or NULL when iteration is exhausted
 */
pim_region_t *pim_device_get_next_region(pim_device_t *dev,
                                         pim_region_t *region) PIM_NOEXCEPT;

/*============================================================================
 * Region Management
 *============================================================================*/

/**
 * @brief Create a memory region as a view into the device's shared memory
 *
 * Regions are views/slices of the device's single shared memory file.
 * The control region and data region share the same backing shm file
 * but at different offsets.
 *
 * Region topology changes are rejected while monitor threads are running,
 * while SIGSEGV mode is enabled, or while SIGSEGV enable/disable transition
 * is in progress.
 *
 * @param dev       Device handle
 * @param offset    Byte offset within device's shm where region starts
 * @param size      Size of the region in bytes
 * @param type      Region type (DATA or CONTROL)
 * @return          Region handle on success, NULL on failure
 */
pim_region_t *pim_region_create(pim_device_t *dev, size_t offset, size_t size,
                                pim_region_type_t type) PIM_NOEXCEPT;

/**
 * @brief Free a memory region
 *
 * Region topology changes are rejected while monitor threads are running,
 * while SIGSEGV mode is enabled, or while SIGSEGV enable/disable transition
 * is in progress.
 *
 * @param region    Region handle
 * @return          PIM_SUCCESS, PIM_ERR_INVALID_ARG, PIM_ERR_NOT_FOUND,
 *                  or PIM_ERR_BUSY
 */
pim_error_t pim_region_free(pim_region_t *region) PIM_NOEXCEPT;

/**
 * @brief Get the memory-mapped pointer for the region
 *
 * @param region    Region handle
 * @return          Pointer to mapped memory, NULL on error
 */
void *pim_region_get_ptr(pim_region_t *region) PIM_NOEXCEPT;

/**
 * @brief Get the size of a region
 *
 * @param region    Region handle
 * @return          Size in bytes
 */
size_t pim_region_get_size(pim_region_t *region) PIM_NOEXCEPT;

/**
 * @brief Get the offset of a region within the device's shm
 *
 * @param region    Region handle
 * @return          Offset in bytes
 */
size_t pim_region_get_offset(pim_region_t *region) PIM_NOEXCEPT;

/**
 * @brief Get the parent device of a region
 *
 * @param region    Region handle
 * @return          Device handle
 */
pim_device_t *pim_region_get_device(pim_region_t *region) PIM_NOEXCEPT;

/**
 * @brief Sync a region's memory range to backing storage
 *
 * @param region    Region handle
 * @return          PIM_SUCCESS, PIM_ERR_INVALID_ARG, or PIM_ERR_MSYNC
 */
pim_error_t pim_region_msync(pim_region_t *region) PIM_NOEXCEPT;

/**
 * @brief Lock a region's pages in memory
 *
 * @param region    Region handle
 * @return          PIM_SUCCESS, PIM_ERR_INVALID_ARG, or PIM_ERR_MLOCK
 */
pim_error_t pim_region_mlock(pim_region_t *region) PIM_NOEXCEPT;

/**
 * @brief Unlock a region's pages in memory
 *
 * @param region    Region handle
 * @return          PIM_SUCCESS, PIM_ERR_INVALID_ARG, or PIM_ERR_MUNLOCK
 */
pim_error_t pim_region_munlock(pim_region_t *region) PIM_NOEXCEPT;

/**
 * @brief Change region page protections using access modes
 *
 * Mapping:
 * - PIM_ACCESS_MODE_NONE -> PROT_NONE
 * - PIM_ACCESS_MODE_RO   -> PROT_READ
 * - PIM_ACCESS_MODE_WO   -> PROT_WRITE
 * - PIM_ACCESS_MODE_RW   -> PROT_READ | PROT_WRITE
 *
 * Note: page protection is page-granular; pages covering the region are
 * changed as a whole.
 *
 * @param region    Region handle
 * @param access    Access mode
 * @return          PIM_SUCCESS, PIM_ERR_INVALID_ARG, or PIM_ERR_MPROTECT
 */
pim_error_t pim_region_mprotect(pim_region_t *region,
                                pim_access_mode_t access) PIM_NOEXCEPT;

/**
 * @brief Fill the beginning of a region with a byte value
 *
 * Equivalent to `memset(region_ptr, value, size)` for the given region.
 *
 * @param region    Region handle
 * @param value     Byte value (0..255) used for fill
 * @param size      Number of bytes to fill from region offset 0
 * @return          PIM_SUCCESS or PIM_ERR_INVALID_ARG
 */
pim_error_t pim_region_memset(pim_region_t *region, int value,
                              size_t size) PIM_NOEXCEPT;

/*============================================================================
 * DPU Handler Registration
 *============================================================================*/

/**
 * @brief Register a DPU handler for a specific offset in a CTRL_MMIO region
 *
 * When a write occurs to the specified offset in the CTRL_MMIO region,
 * the handler will be invoked with the written value.
 *
 * @param dev           Device handle
 * @param region        Control region handle (must be CTRL_MMIO)
 * @param offset        Offset within the region to monitor
 * @param register_size Register width in bytes (must be 1, 2, 4, or 8)
 * @param depth         Handler depth level (0 = root/default MMIO dispatch)
 * @param handler       Callback function
 * @param user_data     User context passed to handler
 * @return              PIM_SUCCESS or error code
 *
 * Registration is rejected with PIM_ERR_BUSY while monitor threads are running,
 * while SIGSEGV mode is enabled, while SIGSEGV enable/disable transition is in
 * progress, or while device teardown is in progress. If handler-table
 * allocation fails, returns PIM_ERR_NO_MEMORY.
 * The offset must be naturally aligned to register_size.
 * Register widths other than 1/2/4/8 are rejected with PIM_ERR_INVALID_ARG.
 *
 * Depth semantics:
 * - depth == 0: root handler path used by typed writes / monitor / SIGSEGV.
 * - depth > 0: nested handler path, invocable explicitly via
 *              pim_invoke_dpu_handler(..., depth).
 */
pim_error_t pim_register_dpu_handler(pim_device_t *dev, pim_region_t *region,
                                     size_t offset, size_t register_size,
                                     uint32_t depth, pim_dpu_handler_t handler,
                                     void *user_data) PIM_NOEXCEPT;

/**
 * @brief Unregister a DPU handler
 *
 * @param dev        Device handle
 * @param region     Control region handle
 * @param offset     Offset that was registered
 * @param depth      Handler depth level to unregister (0 = root/default)
 * @return           PIM_SUCCESS or error code
 *
 * Unregistration is rejected with PIM_ERR_BUSY while monitor threads are
 * running, while SIGSEGV mode is enabled, while SIGSEGV enable/disable
 * transition is in progress, or while device teardown is in progress.
 */
pim_error_t pim_unregister_dpu_handler(pim_device_t *dev, pim_region_t *region,
                                       size_t offset,
                                       uint32_t depth) PIM_NOEXCEPT;

/**
 * @brief Directly invoke the DPU handler registered at a given offset
 *
 * Looks up the handler at the specified offset and calls it with the
 * provided value, without performing any memory write. This is the
 * recommended dispatch method when pim_write(..., size=4/8, ...) cannot be used
 * (e.g., when the hardware uses interleaved or encoded memory writes
 * that are decoded externally, such as in a custom monitor scan or
 * vdev ioctl callback).
 *
 * If a thread pool is running, the handler is submitted to the pool.
 * Otherwise it is called synchronously in the caller's thread.
 *
 * @param dev        Device handle
 * @param region     Control region handle (must be CTRL_MMIO)
 * @param offset     Offset where the handler is registered
 * @param value      Value to pass to the handler
 * @param depth      Handler depth level to invoke (0 = root/default)
 * @return           PIM_SUCCESS, PIM_ERR_INVALID_ARG, or PIM_ERR_NOT_FOUND
 */
pim_error_t pim_invoke_dpu_handler(pim_device_t *dev, pim_region_t *region,
                                   size_t offset, uint64_t value,
                                   uint32_t depth) PIM_NOEXCEPT;

/**
 * @brief Wait until all submitted pool tasks for this device are finished
 *
 * Uses per-device counters (`futex_cnt_commited` / `futex_cnt_finished`) and
 * futex wait/wake on `futex_cnt_finished`.
 *
 * @param dev       Device handle
 * @return          PIM_SUCCESS, PIM_ERR_INVALID_ARG, or PIM_ERR_WOULD_DEADLOCK
 */
pim_error_t pim_device_barrier(pim_device_t *dev) PIM_NOEXCEPT;

/*============================================================================
 * Memory Access Operations (with DPU trigger support)
 *============================================================================*/

/**
 * @brief Write a typed register value to a region
 *
 * If the region is a control region and a handler is registered at the
 * offset, the handler will be invoked after the write.
 * This API returns immediately (no device barrier wait).
 * Supported `size` values are 1, 2, 4, or 8 bytes.
 *
 * @param region    Region handle
 * @param offset    Byte offset within the region
 * @param size      Register width in bytes (1/2/4/8)
 * @param value     Value to write (truncated to `size`)
 * @return          PIM_SUCCESS or error code
 */
pim_error_t pim_write(pim_region_t *region, size_t offset, size_t size,
                      uint64_t value) PIM_NOEXCEPT;

/**
 * @brief Write a typed register value and wait for device pool quiescence
 *
 * Equivalent to `pim_write(...)` followed by
 * `pim_device_barrier(region->device)`.
 * Supported `size` values are 1, 2, 4, or 8 bytes. Alignment is validated as
 * `offset % size == 0`.
 */
pim_error_t pim_write_sync(pim_region_t *region, size_t offset, size_t size,
                           uint64_t value) PIM_NOEXCEPT;

/**
 * @brief Write a typed register value without invoking DPU handler
 *
 * Supported `size` values are 1, 2, 4, or 8 bytes. Alignment is validated as
 * `offset % size == 0`.
 */
pim_error_t pim_write_raw(pim_region_t *region, size_t offset, size_t size,
                          uint64_t value) PIM_NOEXCEPT;

/**
 * @brief Read a typed register value
 *
 * Supported `size` values are 1, 2, 4, or 8 bytes. Alignment is validated as
 * `offset % size == 0`. On success, the loaded value is zero-extended into
 * `*value`.
 */
pim_error_t pim_read(pim_region_t *region, size_t offset, size_t size,
                     uint64_t *value) PIM_NOEXCEPT;

/**
 * @brief Bulk write to a region
 *
 * Note: This does NOT trigger DPU handlers. Use pim_write(..., size=4/8, ...)
 * for control register writes that should trigger DPU.
 *
 * @param region    Region handle
 * @param offset    Byte offset within the region
 * @param data      Source data buffer
 * @param size      Number of bytes to write
 * @return          PIM_SUCCESS or error code
 */
pim_error_t pim_memcpy_to(pim_region_t *region, size_t offset, const void *data,
                          size_t size) PIM_NOEXCEPT;

/**
 * @brief Bulk read from a region
 *
 * @param region    Region handle
 * @param offset    Byte offset within the region
 * @param data      Destination data buffer
 * @param size      Number of bytes to read
 * @return          PIM_SUCCESS or error code
 */
pim_error_t pim_memcpy_from(pim_region_t *region, size_t offset, void *data,
                            size_t size) PIM_NOEXCEPT;

/*============================================================================
 * Thread Pool
 *============================================================================*/

/**
 * @brief Start the thread pool with the specified concurrency
 *
 * The thread pool executes DPU handler callbacks.
 *
 * This is NOT called automatically by pim_start_monitor(). Use this for:
 * - SIGSEGV-based monitoring
 * - Concurrent handler execution with polling monitor
 *
 * @param dev         Device handle
 * @param concurrency Number of worker threads to spawn.
 *                    If 0, uses number of CTRL_MMIO regions.
 *                    If no CTRL_MMIO regions exist, pool remains stopped.
 * @return            PIM_SUCCESS, PIM_ERR_INVALID_ARG, PIM_ERR_BUSY,
 *                    or PIM_ERR_NO_MEMORY
 */
pim_error_t pim_pool_start(pim_device_t *dev,
                           uint32_t concurrency) PIM_NOEXCEPT;

/**
 * @brief Stop the thread pool and wait for all workers to finish
 *
 * Drains the task queue before returning.
 * Must be called from a non-pool-worker control thread.
 * Calls from a pool worker return PIM_ERR_BUSY to avoid self-join termination.
 *
 * @param dev          Device handle
 * @return             PIM_SUCCESS, PIM_ERR_INVALID_ARG, or PIM_ERR_BUSY
 */
pim_error_t pim_pool_stop(pim_device_t *dev) PIM_NOEXCEPT;

/*============================================================================
 * External Write Monitoring
 *============================================================================*/

/**
 * @brief Start monitoring for external writes to CTRL_MMIO regions
 *
 * Spawns one or more background threads that poll registered control offsets
 * for value changes. When a value changes (from any source, including direct
 * writes from external processes to the shm file), the registered DPU handler
 * is invoked.
 *
 * When multiple monitors are specified, CTRL_MMIO regions are distributed
 * among them using modulo assignment (region N is handled by monitor N %
 * num_monitors). This can improve polling throughput for devices with many
 * regions.
 *
 * Uses adaptive polling: busy-spins for active_poll_cnt_limit iterations
 * for low latency, then sleeps for device_sleep_duration_us when idle.
 * Passing 0 for any tuning argument resets it to library defaults:
 * sleep=100us, poll_limit=20000, wakeup_spin_divisor=4.
 *
 * @param dev                       Device handle
 * @param num_monitors              Number of monitor threads (1 = single
 *                                  monitor, >1 = divide regions among threads,
 *                                  capped at number of CTRL_MMIO regions)
 *                                  (0 -> default 1)
 * @param device_sleep_duration_us  Sleep duration in microseconds when idle
 *                                  (0 -> default 100)
 * @param active_poll_cnt_limit     Maximum spin iterations before sleeping
 *                                  (0 -> default 20000)
 * @param wakeup_spin_divisor       Divisor for spin budget after waking from
 *                                  sleep (e.g., 4 means limit/4 spins,
 *                                  0 -> default 4)
 * @return                          PIM_SUCCESS or error code
 *                                  (PIM_ERR_BUSY during teardown or while a
 *                                  SIGSEGV enable/disable transition is active)
 */
pim_error_t pim_start_monitor(pim_device_t *dev, uint32_t num_monitors,
                              uint32_t device_sleep_duration_us,
                              uint32_t active_poll_cnt_limit,
                              uint32_t wakeup_spin_divisor) PIM_NOEXCEPT;

/**
 * @brief Stop the monitoring threads
 *
 * @param dev               Device handle
 * @return                  PIM_SUCCESS, PIM_ERR_INVALID_ARG, or PIM_ERR_BUSY
 *                          (if called from a monitor worker thread)
 */
pim_error_t pim_stop_monitor(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Check if monitor is running
 *
 * @param dev               Device handle
 * @return                  true if running, false if not
 */
bool pim_is_monitor_running(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Get number of active monitor threads
 *
 * @param dev               Device handle
 * @return                  Number of monitor threads currently running
 */
uint32_t pim_get_num_monitors(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Set maximum concurrent DPU handlers (thread pool size)
 *
 * If the pool is already running, this dynamically resizes it:
 * - Increasing spawns additional worker threads
 * - Decreasing signals excess workers to exit
 *
 * If set to 0 while running, the pool is stopped and handlers execute
 * directly in monitor/SIGSEGV callback threads.
 * Calls from a pool worker return PIM_ERR_BUSY to avoid self-join during
 * internal stop/restart.
 *
 * @param dev               Device handle
 * @param concurrency       Number of worker threads (0 = no pool)
 * @return                  PIM_SUCCESS, PIM_ERR_INVALID_ARG, PIM_ERR_BUSY,
 *                          or PIM_ERR_NO_MEMORY
 */
pim_error_t pim_set_max_active_dpus(pim_device_t *dev,
                                    uint32_t concurrency) PIM_NOEXCEPT;

/**
 * @brief Get maximum concurrent DPU handlers
 *
 * @param dev               Device handle
 * @return                  Current max_active_dpus value (0 means no limit)
 */
uint32_t pim_get_max_active_dpus(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Set monitor sleep duration
 *
 * Controls how long the monitor thread sleeps when no activity is detected
 * and the spin budget is exhausted. Can be called while monitor is running.
 * Passing 0 resets this value to default 100us.
 *
 * @param dev               Device handle
 * @param sleep_us          Sleep duration in microseconds
 */
void pim_set_sleep_duration_us(pim_device_t *dev,
                               uint32_t sleep_us) PIM_NOEXCEPT;

/**
 * @brief Get monitor sleep duration
 *
 * @param dev               Device handle
 * @return                  Current sleep duration in microseconds
 */
uint32_t pim_get_sleep_duration_us(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Set active poll count limit (spin budget)
 *
 * Controls how many spin iterations the monitor performs before sleeping.
 * Higher values reduce latency but increase CPU usage. Can be called while
 * monitor is running (takes effect after next sleep cycle).
 * Passing 0 resets this value to default 20000.
 *
 * @param dev               Device handle
 * @param poll_limit        Number of spin iterations before sleeping
 */
void pim_set_poll_limit(pim_device_t *dev, uint32_t poll_limit) PIM_NOEXCEPT;

/**
 * @brief Get active poll count limit (spin budget)
 *
 * @param dev               Device handle
 * @return                  Current poll limit value
 */
uint32_t pim_get_poll_limit(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Set wakeup spin divisor
 *
 * Controls the spin budget after waking from sleep. After sleeping, the
 * monitor sets spin_counter = poll_limit / divisor. This provides a brief
 * "re-warmup" period to catch follow-up writes without fully re-entering
 * the hot spin phase.
 *
 * - divisor=1: Full spin budget after waking (most responsive, higher CPU)
 * - divisor=4: Default; 25% of spin budget after waking
 * - divisor=0: Reset to default divisor=4
 *
 * Can be called while monitor is running.
 *
 * @param dev               Device handle
 * @param divisor           Divisor for post-sleep spin budget
 */
void pim_set_wakeup_spin_divisor(pim_device_t *dev,
                                 uint32_t divisor) PIM_NOEXCEPT;

/**
 * @brief Get wakeup spin divisor
 *
 * @param dev               Device handle
 * @return                  Current divisor value (default: 4)
 */
uint32_t pim_get_wakeup_spin_divisor(pim_device_t *dev) PIM_NOEXCEPT;

/*============================================================================
 * SIGSEGV-based Write Protection Monitoring
 *============================================================================*/

/**
 * @brief Enable global SIGSEGV-based write monitoring
 *
 * Installs a SIGSEGV handler that can intercept writes to CTRL_MMIO regions.
 * After calling this, use pim_device_enable_sigsegv() to enable monitoring
 * for specific devices.
 *
 * The handler uses mprotect() to mark CTRL_MMIO region pages as read-only.
 * When a write occurs, the handler:
 *   1. Decodes the write instruction to extract the value
 *   2. Spawns the DPU handler asynchronously
 *   3. Skips the instruction (page stays protected)
 *
 * @param enable_chain      Whether to chain to old handler on SIGSEGV miss
 * @return                  PIM_SUCCESS or PIM_ERR_SIGNAL
 */
pim_error_t pim_g_enable_sigsegv_handler(bool enable_chain) PIM_NOEXCEPT;

/**
 * @brief Disable global SIGSEGV-based write monitoring
 *
 * Restores original signal handler. All devices must be disabled first
 * via pim_device_disable_sigsegv().
 *
 * @return                  PIM_SUCCESS or error code
 */
pim_error_t pim_g_disable_sigsegv_handler(void) PIM_NOEXCEPT;

/**
 * @brief Enable SIGSEGV monitoring for a specific device
 *
 * Spawns callback threads and write-protects CTRL_MMIO regions for this device.
 * Requires pim_g_enable_sigsegv_handler() to be called first.
 * Call this in a quiescent control phase (no concurrent host writes to the
 * device mapping while transition is in progress).
 *
 * Requirements:
 *   - Only PIM_REGION_CTRL_MMIO regions are protected
 *   - CTRL_MMIO regions should be page-aligned for best results
 *   - Use PIM_REGION_CTRL_RW for regions that need polling-based monitoring
 *
 * @param dev               Device handle
 * @return                  PIM_SUCCESS, PIM_ERR_INVALID_ARG,
 *                          PIM_ERR_ALREADY_EXISTS, PIM_ERR_NOT_FOUND
 *                          (no CTRL_MMIO regions), PIM_ERR_MPROTECT
 *                          (mprotect failure), PIM_ERR_NO_MEMORY, or
 *                          PIM_ERR_BUSY (teardown or another SIGSEGV
 *                          transition in progress)
 */
pim_error_t pim_device_enable_sigsegv(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Disable SIGSEGV monitoring for a specific device
 *
 * Makes CTRL_MMIO regions writable again and stops callback threads.
 * Call this in a quiescent control phase (no concurrent host writes to the
 * device mapping while transition is in progress).
 *
 * @param dev               Device handle
 * @return                  PIM_SUCCESS, PIM_ERR_INVALID_ARG,
 *                          PIM_ERR_MPROTECT, or
 *                          PIM_ERR_BUSY (if called from SIGSEGV callback
 *                          worker context or another SIGSEGV transition is
 *                          in progress)
 */
pim_error_t pim_device_disable_sigsegv(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Check if SIGSEGV monitor is enabled for a device
 *
 * @param dev               Device handle
 * @return                  1 if enabled, 0 if not
 */
bool pim_is_sigsegv_enabled(pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Check if the global SIGSEGV handler is currently installed
 *
 * @return                  1 if installed, 0 otherwise
 */
bool pim_g_is_sigsegv_handler_enabled(void) PIM_NOEXCEPT;

/**
 * @brief Set whether to chain to the old signal handler on SIGSEGV miss
 *
 * By default, if a SIGSEGV occurs that doesn't match a CTRL_MMIO region,
 * the library chains to the previous handler. When using with an
 * LD_PRELOAD library that handles signal chaining itself, disable this
 * to avoid redundant overhead.
 *
 * @param chain             1 to chain (default), 0 to skip chaining
 */
void pim_g_set_sigsegv_chain_handler(bool chain) PIM_NOEXCEPT;

/**
 * @brief Get the current SIGSEGV handler chaining setting
 *
 * @return                  1 if chaining enabled, 0 if disabled
 */
bool pim_g_is_sigsegv_chain_handler_enabled(void) PIM_NOEXCEPT;

/**
 * @brief SIGSEGV handler function for use by LD_PRELOAD libraries
 *
 * This function processes a SIGSEGV and determines if it was caused by
 * a write to a CTRL_MMIO region. Intended for use by LD_PRELOAD libraries
 * that intercept signals.
 *
 * @param info      Signal info from sigaction
 * @param ucontext  User context from sigaction
 * @return          false if processed (do not forward), true if should forward
 */
bool _pim_sigsegv_handler_func(siginfo_t *info, void *ucontext) PIM_NOEXCEPT;

/*============================================================================
 * Virtual Device (syscall-intercepted device emulation)
 *
 * Provides a generalized framework for emulating character devices at the
 * syscall level. Applications that open /dev/<device>, perform ioctl(),
 * mmap()/munmap(), and close() are transparently redirected without kernel
 * involvement.
 *
 * This layer handles:
 *   - Real fd allocation/tracking for vdev/vsysfs sessions
 *   - Path glob matching on SYS_open and SYS_openat
 *   - Dispatch of vdev callbacks: open/ioctl/mmap/close
 *   - Built-in virtual sysfs file I/O (read/write/lseek/getdents/stat family)
 *
 * The syscall interception hook must be provided by liboverlaysys and call
 * _pim_vsyscall_dispatch() from its external hook entrypoint.
 *
 * `_pim_vsyscall_dispatch()` REQUIRES liboverlaysys to be active in process.
 * `pim_vfile_owns_fd()` is a utility query API.
 *============================================================================*/

/**
 * @brief Opaque handle for a virtual device root.
 */
typedef struct pim_vdev_root pim_vdev_root_t;

/**
 * @brief Opaque handle for a virtual device instance.
 */
typedef struct pim_vdev pim_vdev_t;

/**
 * @brief Opaque handle for a virtual sysfs root.
 */
typedef struct pim_vsysfs_root pim_vsysfs_root_t;

/**
 * @brief Opaque handle for a virtual sysfs instance.
 */
typedef struct pim_vsysfs pim_vsysfs_t;

/**
 * @brief Opaque handle for a virtual module instance.
 */
typedef struct pim_vmodule pim_vmodule_t;

/*----------------------------------------------------------------------------
 * Virtual Device Callback Types
 *
 * All callbacks receive the vdev handle and the owning real fd for the
 * open-file description handled by this virtual-device session.
 *----------------------------------------------------------------------------*/

/**
 * @brief Called when an application opens this vdev instance path
 *
 * @param vdev      Virtual device instance handle
 * @param path      The actual path the application opened
 * @param flags     Open flags (O_RDONLY, O_RDWR, etc.)
 * @return          Real fd to reserve/use for this virtual session, or
 *                  negative errno on failure
 */
typedef int (*pim_vdev_open_fn)(pim_vdev_t *vdev, const char *path, int flags);

/**
 * @brief Called when the application performs ioctl() on a virtual fd
 *
 * @param vdev      Virtual device instance handle
 * @param fd        Owning real fd returned by open callback
 * @param cmd       ioctl command number
 * @param arg       ioctl argument (typically a user pointer)
 * @return          0 on success, negative errno on failure
 */
typedef long (*pim_vdev_ioctl_fn)(pim_vdev_t *vdev, int fd, unsigned long cmd,
                                  unsigned long arg);

/**
 * @brief Called when the application calls mmap() on a virtual fd
 *
 * The callback should return a pointer to the mapped region, or MAP_FAILED.
 * The caller (syscall hook) is responsible for returning this pointer to the
 * application by intercepting the SYS_mmap return value.
 *
 * @param vdev      Virtual device instance handle
 * @param fd        Owning real fd returned by open callback
 * @param length    Requested map length
 * @param prot      Protection flags (PROT_READ, PROT_WRITE, etc.)
 * @param flags     Map flags (MAP_SHARED, MAP_PRIVATE, etc.)
 * @param offset    File offset for mapping
 * @return          Pointer to mapped region, or MAP_FAILED on error
 */
typedef void *(*pim_vdev_mmap_fn)(pim_vdev_t *vdev, int fd, size_t length,
                                  int prot, int flags, off_t offset);

/**
 * @brief Called when the application closes a virtual fd
 *
 * @param vdev      Virtual device instance handle
 * @param fd        Owning real fd returned by open callback
 * @return          0 on success, negative errno on failure
 */
typedef int (*pim_vdev_close_fn)(pim_vdev_t *vdev, int fd);

/*----------------------------------------------------------------------------
 * Virtual Device Root / Instance Lifecycle
 *----------------------------------------------------------------------------*/

/**
 * @brief Create a virtual device root under /dev/
 *
 * @param name_glob  Device-name glob (e.g., "dpu_rank*" => "/dev/dpu_rank*")
 * @return           Root handle, or NULL on failure
 */
pim_vdev_root_t *pim_vdev_root_create(const char *name_glob) PIM_NOEXCEPT;

/**
 * @brief Destroy a virtual device root
 */
void pim_vdev_root_destroy(pim_vdev_root_t *root) PIM_NOEXCEPT;

/**
 * @brief Get root name glob (without /dev/ prefix)
 */
const char *pim_vdev_root_get_name_glob(pim_vdev_root_t *root) PIM_NOEXCEPT;

/**
 * @brief Create a concrete virtual device instance under a root
 *
 * @param root      Root handle from pim_vdev_root_create()
 * @param name      Concrete device name (e.g., "dpu_rank0")
 * @return          Virtual device instance, or NULL on failure
 */
pim_vdev_t *pim_vdev_create(pim_vdev_root_t *root,
                            const char *name) PIM_NOEXCEPT;

/**
 * @brief Destroy a virtual device instance
 */
void pim_vdev_destroy(pim_vdev_t *vdev) PIM_NOEXCEPT;

/**
 * @brief Get virtual device instance name (without /dev/ prefix)
 */
const char *pim_vdev_get_name(pim_vdev_t *vdev) PIM_NOEXCEPT;

/**
 * @brief Get absolute /dev path of a virtual device instance
 */
const char *pim_vdev_get_path(pim_vdev_t *vdev) PIM_NOEXCEPT;

/**
 * @brief Get owning root of a virtual device instance
 */
pim_vdev_root_t *pim_vdev_get_root(pim_vdev_t *vdev) PIM_NOEXCEPT;

/**
 * @brief Set opaque user data on a virtual device instance
 *
 * @param vdev        Virtual device instance
 * @param user_data   Caller-owned opaque pointer
 */
void pim_vdev_set_user_data(pim_vdev_t *vdev, void *user_data) PIM_NOEXCEPT;

/**
 * @brief Get opaque user data from a virtual device instance
 *
 * @param vdev        Virtual device instance
 * @return            Caller-owned opaque pointer, or NULL
 */
void *pim_vdev_get_user_data(pim_vdev_t *vdev) PIM_NOEXCEPT;

/**
 * @brief Lock a virtual device instance lifetime
 *
 * While held, the instance object remains alive even if removed from
 * registration lists by pim_vdev_destroy().
 *
 * @param vdev        Virtual device instance
 * @return            PIM_SUCCESS, PIM_ERR_INVALID_ARG, PIM_ERR_NOT_FOUND,
 *                    PIM_ERR_NO_MEMORY, or PIM_ERR_BUSY on refcount overflow
 */
pim_error_t pim_vdev_lock(pim_vdev_t *vdev) PIM_NOEXCEPT;

/**
 * @brief Unlock a virtual device instance lifetime hold
 *
 * Drops one hold previously acquired by pim_vdev_lock().
 *
 * @param vdev        Virtual device instance
 * @return            PIM_SUCCESS, PIM_ERR_INVALID_ARG, or PIM_ERR_NOT_FOUND
 */
pim_error_t pim_vdev_unlock(pim_vdev_t *vdev) PIM_NOEXCEPT;

/**
 * @brief Bind a real fd to a virtual device instance and associated PIM device
 *
 * @param vdev      Virtual device instance
 * @param fd        Real file descriptor key (must be >= 0)
 * @param dev       Associated PIM device handle
 * @return          `PIM_SUCCESS`, `PIM_ERR_INVALID_ARG`, or `PIM_ERR_NO_MEMORY`
 */
pim_error_t pim_vdev_register_device(pim_vdev_t *vdev, int fd,
                                     pim_device_t *dev) PIM_NOEXCEPT;

/**
 * @brief Unbind a real fd from a virtual device instance
 *
 * @param vdev      Virtual device instance
 * @param fd        Real file descriptor key (must be >= 0)
 * @return          `PIM_SUCCESS`, `PIM_ERR_INVALID_ARG`, or `PIM_ERR_NOT_FOUND`
 */
pim_error_t pim_vdev_unregister_device(pim_vdev_t *vdev, int fd) PIM_NOEXCEPT;

/**
 * @brief Look up a PIM device bound to (vdev, fd)
 *
 * @param vdev      Virtual device instance
 * @param fd        Real file descriptor key (must be >= 0)
 * @return          Bound `pim_device_t*` on success, otherwise `NULL`
 */
pim_device_t *pim_vdev_find_device(pim_vdev_t *vdev, int fd) PIM_NOEXCEPT;

/*----------------------------------------------------------------------------
 * Virtual Device Callback Registration
 *----------------------------------------------------------------------------*/

/**
 * @brief Set the open callback for a virtual device root
 *
 * @param root      Virtual device root handle
 * @param fn        Open callback function
 */
void pim_vdev_root_set_open_cb(pim_vdev_root_t *root,
                               pim_vdev_open_fn fn) PIM_NOEXCEPT;

/**
 * @brief Set the ioctl callback for a virtual device root
 */
void pim_vdev_root_set_ioctl_cb(pim_vdev_root_t *root,
                                pim_vdev_ioctl_fn fn) PIM_NOEXCEPT;

/**
 * @brief Set the mmap callback for a virtual device root
 */
void pim_vdev_root_set_mmap_cb(pim_vdev_root_t *root,
                               pim_vdev_mmap_fn fn) PIM_NOEXCEPT;

/**
 * @brief Set the close callback for a virtual device root
 */
void pim_vdev_root_set_close_cb(pim_vdev_root_t *root,
                                pim_vdev_close_fn fn) PIM_NOEXCEPT;

/*----------------------------------------------------------------------------
 * Virtual Sysfs Root / Instance Lifecycle
 *----------------------------------------------------------------------------*/

/**
 * @brief Create a virtual sysfs root bound to a virtual-device root
 *
 * @param vdev_root           Owning vdev root for class-instance naming
 * @param class_name          Class directory name under /sys/class
 *                            (e.g., "dpu_rank")
 * @param bus_name            Bus directory name under /sys/devices
 *                            (e.g., "platform")
 * @param instance_name_glob  Bus-instance glob under /sys/devices/<bus_name>
 *                            (e.g., "dpu_region_mem.*")
 * @return                    Root handle, or NULL on failure
 */
pim_vsysfs_root_t *
pim_vsysfs_root_create(pim_vdev_root_t *vdev_root, const char *class_name,
                       const char *bus_name,
                       const char *instance_name_glob) PIM_NOEXCEPT;

/**
 * @brief Destroy a virtual sysfs root
 */
void pim_vsysfs_root_destroy(pim_vsysfs_root_t *root) PIM_NOEXCEPT;

/**
 * @brief Get class-side root glob (relative to /sys/class)
 */
const char *
pim_vsysfs_root_get_class_glob(pim_vsysfs_root_t *root) PIM_NOEXCEPT;

/**
 * @brief Get devices-side root glob (relative to /sys/devices)
 */
const char *
pim_vsysfs_root_get_devices_glob(pim_vsysfs_root_t *root) PIM_NOEXCEPT;

/**
 * @brief Set/update a default attribute for future or existing instances
 */
pim_error_t
pim_vsysfs_root_create_default_attr(pim_vsysfs_root_t *root, const char *name,
                                    const char *value,
                                    pim_access_mode_t access) PIM_NOEXCEPT;

/**
 * @brief Get a default attribute from a virtual sysfs root
 *
 * Returns a thread-local snapshot pointer. The returned pointer remains valid
 * until the next call to this API on the same thread.
 */
const char *pim_vsysfs_root_find_default_attr(pim_vsysfs_root_t *root,
                                              const char *name) PIM_NOEXCEPT;

/**
 * @brief Remove a default attribute from a virtual sysfs root and all existing
 *        instances under that root
 */
pim_error_t pim_vsysfs_root_remove_default_attr(pim_vsysfs_root_t *root,
                                                const char *name) PIM_NOEXCEPT;

/**
 * @brief Create a virtual sysfs instance under a root
 *
 * Class-side instance name is derived from `vdev` name.
 *
 * @param root           Root handle
 * @param vdev           Concrete virtual-device instance
 * @param instance_name  Concrete bus-instance name matched by root glob
 *                       (e.g., "dpu_region_mem.0")
 * @return               Virtual sysfs instance, or NULL on failure
 */
pim_vsysfs_t *pim_vsysfs_create(pim_vsysfs_root_t *root, pim_vdev_t *vdev,
                                const char *instance_name) PIM_NOEXCEPT;

/**
 * @brief Destroy a virtual sysfs instance
 */
void pim_vsysfs_destroy(pim_vsysfs_t *vsysfs) PIM_NOEXCEPT;

/**
 * @brief Get class-side instance name
 */
const char *pim_vsysfs_get_class_name(pim_vsysfs_t *vsysfs) PIM_NOEXCEPT;

/**
 * @brief Get devices-side instance name
 */
const char *pim_vsysfs_get_device_name(pim_vsysfs_t *vsysfs) PIM_NOEXCEPT;

/**
 * @brief Get absolute class-side directory path
 */
const char *pim_vsysfs_get_class_dir(pim_vsysfs_t *vsysfs) PIM_NOEXCEPT;

/**
 * @brief Get absolute devices-side directory path
 */
const char *pim_vsysfs_get_devices_dir(pim_vsysfs_t *vsysfs) PIM_NOEXCEPT;

/**
 * @brief Set (or update) an attribute in a virtual sysfs directory
 *
 * The value string is returned verbatim when read() is called on the
 * corresponding sysfs file. Include a trailing newline if needed.
 *
 * @param vsysfs    Virtual sysfs handle
 * @param name      Attribute name (e.g., "capabilities")
 * @param value     Attribute value string (e.g., "1\n")
 * @param access    Access mode
 * @return          PIM_SUCCESS, PIM_ERR_INVALID_ARG, or PIM_ERR_NO_MEMORY
 */
pim_error_t pim_vsysfs_create_attr(pim_vsysfs_t *vsysfs, const char *name,
                                   const char *value,
                                   pim_access_mode_t access) PIM_NOEXCEPT;

/**
 * @brief Remove an attribute from a virtual sysfs directory
 *
 * @param vsysfs    Virtual sysfs handle
 * @param name      Attribute name
 * @return          PIM_SUCCESS or PIM_ERR_NOT_FOUND
 */
pim_error_t pim_vsysfs_remove_attr(pim_vsysfs_t *vsysfs,
                                   const char *name) PIM_NOEXCEPT;

/**
 * @brief Get an attribute value from a virtual sysfs instance
 *
 * Returns a thread-local snapshot pointer. The returned pointer remains valid
 * until the next call to this API on the same thread.
 */
const char *pim_vsysfs_find_attr(pim_vsysfs_t *vsysfs,
                                 const char *name) PIM_NOEXCEPT;

/**
 * @brief Get the virtual sysfs associated with a virtual device instance
 */
pim_vsysfs_t *pim_vdev_get_vsysfs(pim_vdev_t *vdev) PIM_NOEXCEPT;

/*----------------------------------------------------------------------------
 * Virtual Module Instance Lifecycle
 *
 * Emulates /sys/module/<module_name>/...
 *----------------------------------------------------------------------------*/

/**
 * @brief Create a virtual module instance
 *
 * @param module_name  Concrete module name (e.g., "dpu")
 * @return             Virtual module instance, or NULL on failure
 */
pim_vmodule_t *pim_vmodule_create(const char *module_name) PIM_NOEXCEPT;

/**
 * @brief Destroy a virtual module instance
 */
void pim_vmodule_destroy(pim_vmodule_t *vmodule) PIM_NOEXCEPT;

/**
 * @brief Get virtual module name (without /sys/module/ prefix)
 */
const char *pim_vmodule_get_name(pim_vmodule_t *vmodule) PIM_NOEXCEPT;

/**
 * @brief Get absolute module directory path (/sys/module/<name>)
 */
const char *pim_vmodule_get_dir(pim_vmodule_t *vmodule) PIM_NOEXCEPT;

/**
 * @brief Set (or update) an attribute in a virtual module directory
 */
pim_error_t pim_vmodule_create_attr(pim_vmodule_t *vmodule, const char *name,
                                    const char *value,
                                    pim_access_mode_t access) PIM_NOEXCEPT;

/**
 * @brief Remove an attribute from a virtual module directory
 */
pim_error_t pim_vmodule_remove_attr(pim_vmodule_t *vmodule,
                                    const char *name) PIM_NOEXCEPT;

/**
 * @brief Get an attribute value from a virtual module instance
 *
 * Returns a thread-local snapshot pointer. The returned pointer remains valid
 * until the next call to this API on the same thread.
 */
const char *pim_vmodule_find_attr(pim_vmodule_t *vmodule,
                                  const char *name) PIM_NOEXCEPT;

/*----------------------------------------------------------------------------
 * Gloabl Virtual File Control
 *----------------------------------------------------------------------------*/

/**
 * @brief Create a lightweight kernel-backed fd used by the vfile layer
 *
 * This helper reserves a real descriptor for virtual-file sessions
 * (vdev/vsysfs/vmodule/vdir). Internally it uses `eventfd()` as a minimal
 * placeholder fd.
 *
 * @param flags     Open-style flags; `O_CLOEXEC` and `O_NONBLOCK` are honored.
 * @return          Non-negative fd on success, or -1 on failure with `errno`
 *                  set by `eventfd()`.
 */
static inline int pim_vfile_open(int flags) PIM_NOEXCEPT {
  return eventfd(0, flags & ((O_CLOEXEC ? EFD_CLOEXEC : 0) |
                             (O_NONBLOCK ? EFD_NONBLOCK : 0)));
}

/**
 * @brief Check if a file descriptor belongs to the virtual device layer
 *
 * @param fd        File descriptor to check
 * @return          true if fd is managed by the vfile layer
 *                  (vdev/vsysfs/virtual-dir owned fd)
 */
bool pim_vfile_owns_fd(int fd) PIM_NOEXCEPT;

/**
 * @brief Clear all virtual-file state managed by hostpimsim
 *
 * This closes all currently owned virtual fds and clears all registered
 * virtual nodes (vdev, vsysfs, vmodule) and their internal metadata.
 */
void pim_vfile_clear(void) PIM_NOEXCEPT;

/*----------------------------------------------------------------------------
 * Syscall Dispatch (called from external LD_PRELOAD hook)
 *
 * These functions REQUIRE liboverlaysys as the active LD_PRELOAD syscall
 * interceptor. The _ prefix denotes this mandatory dependency.
 *----------------------------------------------------------------------------*/

/**
 * @brief Dispatch a raw syscall through the virtual device layer
 *
 * This is the single entry point called by the external syscall hook
 * (e.g., overlaysys_syscall_hook). It examines the syscall number and
 * arguments, and if the fd or path matches a registered virtual device
 * or virtual sysfs, handles it without forwarding to the kernel.
 *
 * @param number    Syscall number
 * @param a-f       Raw syscall arguments
 * @param forward   Output: set to 0 to suppress kernel forwarding
 * @return          Wrapper-style syscall return value:
 *                  non-negative success value (fd/bytes/etc) or -1 on failure
 *                  with errno set
 */
long _pim_vsyscall_dispatch(long number, long a, long b, long c, long d, long e,
                            long f, int *forward) PIM_NOEXCEPT;

#ifdef __cplusplus
}
#endif
