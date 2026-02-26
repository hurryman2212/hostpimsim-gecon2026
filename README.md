# libhostpimsim

A Linux library for **building host-memory-type Processing-In-Memory (PIM) simulators**. It backs simulated PIM memory with host shared memory (`/dev/shm/*` or device files), providing write detection, DPU handler dispatch, and a thread pool — so that simulator authors can focus on the PIM execution model rather than the plumbing. It is designed to support the fundamental requirement that high-performance userspace PIM applications depend on: allocating and directly accessing mapped memory via standard Linux APIs.

## Overview

libhostpimsim provides the infrastructure for building PIM simulators where:
- **Userspace-first direct memory mapping** — applications can `open` + `mmap` simulator-backed memory and access it directly from userspace using standard Linux APIs (`open`, `mmap`, `mlock`, `munmap`, `close`)
- **Single shared memory (device) file** backs all control and data regions
- **Memory-layout-agnostic design** — no fixed chunk size or region layout is imposed; users define arbitrary regions at arbitrary offsets to match any PIM architecture
- **Per-chunk DPU handlers** enable parallel processing across chunks; within-chunk parallelism (e.g., multi-threaded DPU (Data Processing Unit) execution) is also possible with a custom DPU handler
- **External processes** can directly access the shared memory via mmap
- **Monitor thread** detects writes from external processes
- **SIGSEGV-based write interposition** — can be used with [liboverlaysys](https://github.com/hurryman2212/overlaysys) to install SIGSEGV-based write interposition, reducing the latency and CPU overhead of polling-based detection
- **Virtual device (vdev) layer** — can be used with [liboverlaysys](https://github.com/hurryman2212/overlaysys) to intercepts syscalls (`open`, `ioctl`, `read`, `write`, `mmap`, `munmap`, `close`) at the userspace level to emulate character devices and sysfs attributes; this enables transparent redirection of real device driver ioctls to the simulator without any kernel module

Simulators built on libhostpimsim can be cycle-accurate or functional, and interact with real applications through shared memory.

## Use Cases

- **PIM Hardware Emulation**: Run PIM applications on systems without actual PIM hardware installed
- **PIM Algorithm Development**: Develop and test PIM-optimized algorithms before hardware availability
- **Software Stack Prototyping**: Build PIM runtime libraries, compilers, and drivers using realistic memory semantics
- **Performance Modeling**: Integrate with cycle-accurate simulators for latency/throughput analysis
- **Application Porting**: Port existing applications to PIM with immediate functional verification
- **Hardware/Software Co-design**: Iterate on PIM ISA and memory layout before tape-out
- **Next-Generation ISA-Compatible Hardware Development**: Use the virtual device layer to develop next ISA-compatible PIM H/W systems with different bus technologies or memory layouts, while transparently supporting existing PIM software binaries
- **Education & Research**: Teach PIM concepts with hands-on shared memory interaction

## Architecture

```
┌─────────────────┐                                      ┌─────────────────────────────┐
│   Application   │                                      │       PIM Simulator         │
└────────┬────────┘                                      │                             │
         │                                               │  ┌───────┐ ┌─────────────┐  │
         │                                               │  │Ctrl. 0│ │ RAM BANK 0  │  │
         │           /dev/shm/pim_mem                    │  ├───────┤ ├─────────────┤  │
         │          ┌────────────────┐                   │  │Ctrl. 1│ │ RAM BANK 1  │  │
         │          │                │                   │  ├───────┤ ├─────────────┤  │
         │          │   Chunk 0      │                   │  │Ctrl. 2│ │ RAM BANK 2  │  │
         │          │   Chunk 1      │                   │  ├───────┤ ├─────────────┤  │
         │          │   Chunk 2      │                   │  │  ...  │ │    ...      │  │
         │          │   ...          │                   │  └───────┘ └─────────────┘  │
         │          │   Chunk N      │                   │                             │
         │          │                │                   │  Monitor Thread (polling)   │
         │          └────────────────┘                   │        │                    │
         │                                               │        ▼                    │
         │                                               │  ┌─────────────────────┐    │
         │                                               │  │ DPU Handler Threads │    │
         │                                               │  │ (parallel per chunk)│    │
         │                                               │  └─────────────────────┘    │
         │                                               └─────────────────────────────┘
         │
═════════╪═══════════════════════════════════════════════════════════════════════════════
         │
         │  ┌─────────────────────────────────────────────────────────────────────────┐
         │  │ STAGE 1: open() + mmap() + mlock()                                      │
         │  └─────────────────────────────────────────────────────────────────────────┘
         │
         │     int fd = open("/dev/shm/pim_mem", O_RDWR);
         │     uint8_t *mem = mmap(NULL, total_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
         │     mlock(mem, total_size);  // REQUIRED: Pin pages in RAM
         │
         ├─────────────────────────────────────────┐
         │                                         │
         │                                         ▼
         │                                ┌────────────────┐
         │                                │ Shared Memory  │
         │                                │  (mapped)      │
         │                                └────────────────┘
         │
         │  ┌─────────────────────────────────────────────────────────────────────────┐
         │  │ STAGE 2: Stream data to all chunks (non-temporal stores)                │
         │  └─────────────────────────────────────────────────────────────────────────┘
         │
         │     for each chunk:
         │         stream_floats_to_dram(chunk_data, src_A, len);
         │         stream_floats_to_dram(chunk_data + offset, src_B, len);
         │     sfence();
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
    │ Data 0  │       │ Data 1  │       │ Data 2  │       │ Data N  │
    │ A,B     │       │ A,B     │       │ A,B     │  ...  │ A,B     │
    └─────────┘       └─────────┘       └─────────┘       └─────────┘
         │
         │  ┌─────────────────────────────────────────────────────────────────────────┐
         │  │ STAGE 3: Issue CMD_EXEC to each chunk's control register                │
         │  └─────────────────────────────────────────────────────────────────────────┘
         │
         │     for each chunk:
         │         stream_store_64(&ctrl[REG_OPCODE], OP_VADD);
         │         stream_store_64(&ctrl[REG_CMD], CMD_EXEC);
         │     sfence();
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
    │ Ctrl 0  │       │ Ctrl 1  │       │ Ctrl 2  │       │ Ctrl N  │
    │ CMD=1   │       │ CMD=1   │       │ CMD=1   │  ...  │ CMD=1   │
    └────┬────┘       └────┬────┘       └────┬────┘       └────┬────┘
         │                 │                 │                 │
         │     Monitor detects CMD changes (polling)           │
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
    │ DPU 0   │       │ DPU 1   │       │ DPU 2   │       │ DPU N   │
    │ Thread  │       │ Thread  │       │ Thread  │  ...  │ Thread  │
    └─────────┘       └─────────┘       └─────────┘       └─────────┘
         │                 │                 │                 │
         │        (parallel execution)       │                 │
         ▼                 ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
    │ Data 0  │       │ Data 1  │       │ Data 2  │       │ Data N  │
    │ C=A+B   │       │ C=A+B   │       │ C=A+B   │  ...  │ C=A+B   │
    └─────────┘       └─────────┘       └─────────┘       └─────────┘
```

> **Real PIM applications flushing data to memory:**
> - **Non-temporal stores** (`movntps`, `movnti`, etc.) bypass the CPU cache and write directly to Write-Combining (WC) buffers. `sfence` drains WC buffers to memory — no `clflush` needed.
> - **Regular stores** (`mov`, `memcpy`) land in the L1/L2/L3 cache. To guarantee visibility in shared memory, each dirty cache line must be flushed with `clflush` / `clflushopt`, followed by `sfence`.

### Memory Region Types

| Type                   | Description                              | DPU Trigger |
| ---------------------- | ---------------------------------------- | ----------- |
| `PIM_REGION_RAM`       | Regular memory bank                      | No          |
| `PIM_REGION_CTRL_MMIO` | MMIO-style control (host writes trigger) | Yes         |
| `PIM_REGION_CTRL_RW`   | Control region (DPU writes, host reads)  | No          |

### Current Limitations

- Consecutive writes to the same `CTRL_MMIO` register may be coalesced, although hostpimsim guarantees that a single write to a `CTRL_MMIO`-region register triggers the DPU handler no more than once.
- There is no Write-Only register support. Reading `CTRL_MMIO` register (which is supposed to be Write-Only) may return non-fixed incorrect/outdated value.
- Each register that has a DPU handler in a `CTRL_MMIO` region must use a register size of exactly `1`, `2`, `4`, or `8` bytes.
- There is currently no public API to wait for a specific DPU thread to be completed.
- This library currently does not simulate "PIM chip wiring" (e.g. bank interleaving for each PIM chip).

### Write Detection Methods

The library provides two methods for detecting writes to `CTRL_MMIO` regions from external processes:

#### 1. Polling-Based Monitor Thread

The default approach uses a background thread that polls control registers for changes. When multiple monitors are configured, they shard the watchlist by assigning `region_idx % num_monitors` to each monitor thread.

**Adaptive Polling Strategy:**
The monitor uses a three-phase adaptive polling strategy to balance latency and CPU usage:

1. **Spin Phase (Hot)**: When activity is expected, the monitor busy-spins while using `PAUSE`/`YIELD` instruction for up to `active_poll_cnt_limit` iterations. This provides sub-microsecond response latency at the cost of CPU cycles.

2. **Sleep Phase (Cold)**: After exhausting the spin budget without detecting changes, the monitor sleeps for `device_sleep_duration_us` microseconds. Upon waking, it resumes with a **progressively reduced** spin budget—each consecutive idle wakeup divides the current spin limit by `wakeup_spin_divisor` (e.g., limit/4, limit/16, limit/64...). This exponential backoff converges CPU usage to near-0% when truly idle.

3. **Hot Streak Reset**: When a change is detected, both the spin counter and the current spin limit reset to the full `active_poll_cnt_limit`, anticipating more writes in quick succession (common in batch operations).

This adaptive approach minimizes latency during active periods while reducing CPU overhead during idle periods.

**Handler Execution:**
When a change is detected, the monitor collects pending handlers during scan and dispatches after scan:
- **With thread pool**: Handlers are submitted to the pool (`pool_submit()`) for concurrent execution
- **Without thread pool** (default): Handlers are executed directly by the monitor thread, sequentially
- **Hot-path allocation control**: each monitor worker reuses an internal `thread_local` pending queue between scans, avoiding repeated vector allocations during active polling

```
┌──────────────────┐     ┌──────────────────────────────────────────┐
│  External App    │     │           Monitor Threads                │
│                  │     │  (sharded by region % num_monitors)      │
└────────┬─────────┘     └──────────────────┬───────────────────────┘
         │                                  │
         │               ┌──────────────────┴───────────────────┐
         │               │                                      │
         │        ┌──────┴──────┐                        ┌──────┴──────┐
         │        │  Monitor 0  │                        │  Monitor 1  │
         │        │  regions:   │                        │  regions:   │
         │        │  0, 2, 4... │                        │  1, 3, 5... │
         │        └──────┬──────┘                        └──────┬──────┘
         │               │                                      │
         │               │ ┌────────────────────────────────────┴─┐
         │               │ │ Adaptive Polling (per monitor):      │
         │               │ │  current_spin_limit = spin_limit     │
         │               │ │  loop:                               │
         │               │ │   SCAN: collect pending handlers     │
         │               │ │         (lock-free sampling)         │
         │               │ │   DISPATCH: invoke handlers          │
         │               │ │         (no locks held)              │
         │               │ │   if change_detected:                │
         │               │ │     reset spin_counter & limit       │
         │               │ │   else if spin_counter > 0:          │
         │               │ │     mm_pause(); spin_counter--       │
         │               │ │   else:                              │
         │               │ │     sleep(device_sleep_duration_us)  │
         │               │ │     current_spin_limit /= divisor    │
         │               │ └──────────────────────────────────────┘
         │               │
         │  write to     │
         │  region 0     │
         │──────────────>│ (Monitor 0 owns region 0)
         │     (shm)     │
         │               │ detect value change
         │               │
         │               │ (lock released, then...)
         │               │ execute handler directly
         │               │──────────────────────────>
         │               │
         │               │ reset current_spin_limit & spin_counter
```

**Pros:**
- Works with any application (no special requirements)
- No signal handling complexity
- Robust and portable

**Cons:**
- Latency depends on polling frequency
- High idle-time CPU usage from continuous polling even with the adaptive polling

**Usage Pattern:**
```c
// Simple usage (no pool, handlers run directly in monitor thread):
pim_start_monitor(dev, 1, 100, 20000, 4);  // 1 monitor, 100us sleep, 20000 spins, divisor=4

// With thread pool for concurrent handler execution:
pim_pool_start(dev, 8);  // 8 worker threads
pim_start_monitor(dev, 4, 100, 20000, 4);  // 4 monitor threads
// ...
pim_stop_monitor(dev);
pim_pool_stop(dev);
```

Passing `0` for `sleep_us`, `poll_limit`, or `spin_divisor` resets that field to defaults: `100`, `20000`, and `4` respectively. The same defaulting behavior applies to the runtime setters (`pim_set_sleep_duration_us`, `pim_set_poll_limit`, `pim_set_wakeup_spin_divisor`).

#### 2. SIGSEGV-Based Write Protection (with liboverlaysys)

For lower latency, the library can use memory protection and SIGSEGV handling to detect writes instantly.

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  External App    │     │  SIGSEGV Handler │     │  Callback Thread │
│  (LD_PRELOAD)    │     │  (in-process)    │     │  (futex wait)    │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         │  write CMD=EXEC        │                        │
         │  (to mprotect'd page)  │                        │
         │───────────────────────>│                        │
         │        SIGSEGV!        │                        │
         │                        │ decode instruction     │
         │                        │ emulate write          │
         │                        │ (CTRL_MMIO ignored)    │
         │                        │ futex_wake             │
         │                        │───────────────────────>│
         │                        │                        │ execute handler
         │                        │                        │──────────────>
         │  (instruction skipped) │                        │
         │<───────────────────────│                        │
```

**Write Handling by Region Type:**
The SIGSEGV handler only commits writes to `CTRL_RW` regions (for status reads). Writes to `CTRL_MMIO` regions are **not committed to memory**—the handler extracts the written value for handler dispatch but skips the actual memory write. This prevents the polling monitor (if running in another process) from detecting a "change" and triggering a duplicate handler invocation.

**Callback Thread Count:**
One callback thread is spawned per write-protected page (CTRL_MMIO region). Each thread waits on a futex until the signal handler wakes it. This means the number of callback threads equals the number of CTRL_MMIO regions in the device.

**Pros:**
- Near-zero latency (synchronous with write instruction)
- No polling overhead   

**Cons:**
- Requires LD_PRELOAD (must be loaded before the application starts)
- System call & signal handling overhead
- Instruction to trigger SIGSEGV interception must be supported by our library

**Usage Pattern:**
```c
// 1. Enable global handler (once per process)
pim_g_enable_sigsegv_handler(true);

// 2. Enable for specific devices (write-protects their CTRL_MMIO regions)
pim_device_enable_sigsegv(dev);

// ... writes to CTRL_MMIO pages now trigger handlers instantly ...

// 3. Cleanup
pim_device_disable_sigsegv(dev);
pim_g_disable_sigsegv_handler();
```

```bash
OVERLAYSYS_DEFSIGHAND=1 LD_PRELOAD=/path/to/your_pim_hook.so ./your_application
```

#### 3. Virtual Device Layer (with liboverlaysys)

Some hardware interleaves or encodes data in host-mapped memory regions, making the values written by the SDK non-trivially decodable at the userspace level. While polling and SIGSEGV can still detect that a write occurred, they cannot reliably extract the original data without knowledge of the hardware-specific encoding. The **vdev** layer sidesteps this entirely by intercepting at the syscall boundary (`ioctl`/`read`/`write`/`mmap`), where data is in its original, unencoded form.

```
┌──────────────────┐     ┌─────────────────────────┐     ┌──────────────────────┐
│  Application     │     │  LD_PRELOAD hook        │     │  vdev / vsysfs       │
│  (SDK code)      │     │  (liboverlaysys)        │     │  (libhostpimsim)     │
└────────┬─────────┘     └────────┬────────────────┘     └────────┬─────────────┘
         │                        │                               │
         │  open("/dev/dpu0")     │                               │
         │───────────────────────>│  _pim_vsyscall_dispatch()     │
         │                        │──────────────────────────────>│
         │                        │  glob matches vdev            │
         │                        │  → open callback              │
         │                        │  → return real fd             │
         │  fd = 42               │<──────────────────────────────│
         │<───────────────────────│                               │
         │                        │                               │
         │  ioctl(fd, CMD, arg)   │                               │
         │───────────────────────>│  lookup owned fd → vdev       │
         │                        │──────────────────────────────>│
         │                        │  → ioctl callback             │
         │                        │  (decode command, run DPU)    │
         │  return value          │<──────────────────────────────│
         │<───────────────────────│                               │
         │                        │                               │
         │  read("/sys/.../cap")  │                               │
         │───────────────────────>│  glob matches vsysfs          │
         │                        │──────────────────────────────>│
         │                        │  → auto-return attr value     │
         │  "1\n"                 │<──────────────────────────────│
         │<───────────────────────│                               │
```

**Key Design Points:**
- **Real file descriptors** are used for virtual sessions (reserved internally or returned by `open` callback) for extended binary compatibility
- **Root + instance model (vdev)** — create a root glob with `pim_vdev_root_create()` (for matching), then create concrete devices with `pim_vdev_create()`
- **Open callback contract** — `pim_vdev_open_fn` returns the real fd to use for this session (negative `errno` on failure)
- **Root + instance model (vsysfs)** — create a root bound to a `pim_vdev_root_t` with `pim_vsysfs_root_create(vdev_root, class_name, bus_name, instance_name_glob)`, then create concrete entries with `pim_vsysfs_create(root, vdev, instance_name)`
- **Virtual sysfs attributes** — `pim_vsysfs_t` objects serve key-value attributes from concrete instance directories
- **Callback fd contract** — `ioctl`/`mmap`/`close` callbacks receive the owning real fd (no per-fd session pointer API)
- **Built-in read/write handling** — vdev only exposes `open/ioctl/mmap/close` callbacks; vsysfs file I/O is auto-served (`read`, `write`, `lseek`)
- **Device association is automatic** — vsysfs instances track device association from `pim_vdev_register_device(vdev, fd, dev)` mappings; no separate attach API is required
- **Virtual-fs snapshot cache** — directory/file topology is cached and rebuilt only when generation changes (create/destroy/attr shape updates)
- **Per-fd data caches** — opened vsysfs fds cache attribute payload/revision, and opened virtual directories cache dirent rows by generation
- **No kernel dependency** — the vdev layer is entirely userspace; the syscall interceptor (liboverlaysys) provides the hook

**Virtual sysfs auto-read behavior:**
When an application opens a path matching `<class_dir>/<attr_name>` (or `<devices_dir>/<attr_name>`), the library auto-serves file-style I/O:
- `read()` returns the current attribute value with cursor tracking (successive reads advance until EOF).
- `write()` updates the attribute value at the current file offset and advances the offset.
- `lseek(SEEK_SET, 0)` resets the cursor.
- `open(..., O_TRUNC)` truncates the attribute value (when writable).
- `open(..., O_APPEND)` makes writes append to the end of the current value.
- Access mode is enforced: writing through an `O_RDONLY` fd fails with `EBADF`.
- Attribute permission is configured per key via `pim_vsysfs_create_attr(..., access)` using `PIM_ACCESS_MODE_NONE`, `PIM_ACCESS_MODE_RO`, `PIM_ACCESS_MODE_RW`, or `PIM_ACCESS_MODE_WO`.

**Pros:**
- Works with unmodified SDK binaries depending on specific kernel module
- Intercepts at syscall boundary — no data format issues (no byte interleaving, no instruction decoding)
- Supports the full device lifecycle: open → ioctl/read/write/mmap → close
- Virtual sysfs avoids the need for a real kernel module to expose capabilities

**Cons:**
- Requires LD_PRELOAD (must be loaded before the application starts)
- System call handling overhead
- Must implement callbacks for every ioctl command the SDK uses

**Usage Pattern:**
```c
// 1. Create vmodule
pim_vmodule_t *vmodule = pim_vmodule_create("dpu");
pim_vmodule_create_attr(vmodule, "version", "7.1\n", PIM_ACCESS_MODE_RO);

// 2. Create vdev roots and set callbacks
pim_vdev_root_t *vdev_rank_root = pim_vdev_root_create("dpu_rank*"); // /dev/dpu_rank*
pim_vdev_root_set_open_cb(vdev_rank_root, my_rank_open_cb);
pim_vdev_root_set_ioctl_cb(vdev_rank_root, my_rank_ioctl_cb);
pim_vdev_root_set_mmap_cb(vdev_rank_root, my_rank_mmap_cb);
pim_vdev_root_set_close_cb(vdev_rank_root, my_rank_close_cb);
// pim_vdev_root_t *vdev_dax_root = pim_vdev_root_create("dax*.*"); // /dev/dax*.*
// pim_vdev_root_set_open_cb(vdev_dax_root, my_dax_open_cb);
// pim_vdev_root_set_ioctl_cb(vdev_dax_root, my_dax_ioctl_cb);
// pim_vdev_root_set_mmap_cb(vdev_dax_root, my_dax_mmap_cb);
// pim_vdev_root_set_close_cb(vdev_dax_root, my_dax_close_cb);

// 3. Create vsysfs roots bound to each vdev root
// /sys/class/dpu_rank/dpu_rank*, /sys/devices/platform/dpu_region_mem.*/dpu_rank*
pim_vsysfs_root_t *vsysfs_rank_root = pim_vsysfs_root_create(
    vdev_rank_root, "dpu_rank", "platform", "dpu_region_mem.*");
// /sys/class/dpu_dax/dax*.*, /sys/devices/platform/dpu_region_mem.*/dax*.*
// pim_vsysfs_root_t *vsysfs_dax_root = pim_vsysfs_root_create(
//     vdev_dax_root, "dpu_dax", "platform", "dpu_region_mem.*");

// Optional per-root default attrs for future instances:
// pim_vsysfs_root_create_default_attr(vsysfs_dax_root, "size", "8589934592\n", PIM_ACCESS_MODE_RO);
// pim_vsysfs_root_create_default_attr(vsysfs_dax_root, "numa", "0\n", PIM_ACCESS_MODE_RO);

// 4. Create concrete vdev and vsysfs instances
// /dev/dpu_rank0
pim_vdev_t *vdev_rank0 = pim_vdev_create(vdev_rank_root,
    "dpu_rank0"); // Return error if the name is already being used.
pim_vsysfs_t *vsysfs_rank0 = pim_vsysfs_create(vsysfs_rank_root,
    vdev_rank0, "dpu_region_mem.0");
// /dev/dax0.0
// pim_vdev_t *vdev_dax0_0 = pim_vdev_create(vdev_dax_root, "dax0.0");
// pim_vsysfs_t *vsysfs_dax0_0 = pim_vsysfs_create(vsysfs_dax_root,
//     vdev_dax0_0, "dpu_region_mem.0");

// In the LD_PRELOAD hook:
//  overlaysys_syscall_hook = _pim_vsyscall_dispatch;
//  pim_vdev_register_device(vdev_rank0, fd, dev_rank0); // You can `pim_vdev_unregister_device(vdev_rank0, fd)`.
//  pim_vdev_register_device(vdev_dax0_0, fd2, dev_rank0);

// 5. Set vsysfs attributes
// pim_vsysfs_create_attr(vsysfs_dax0_0, "size", "8589934592\n", PIM_ACCESS_MODE_RO);
// pim_vsysfs_create_attr(vsysfs_dax0_0, "numa", "0\n", PIM_ACCESS_MODE_RO);

// 6. Get associated vsysfs with vdev
// pim_vsysfs_t *vsysfs = pim_vdev_get_vsysfs(vdev_rank0);
// Get vsysfs attribute (returns `const char *`)
// uint64_t size = strtoull(pim_vsysfs_find_attr(vsysfs, "size"), NULL, 0);
// Delete vsysfs attribute
// pim_vsysfs_remove_attr(vsysfs, "numa");

// 7. Cleanup
// pim_vsysfs_destroy(vsysfs_dax0_0);
pim_vsysfs_destroy(vsysfs_rank0);
// pim_vdev_destroy(vdev_dax0_0);
pim_vdev_destroy(vdev_rank0);
// pim_vdev_root_destroy(vdev_dax_root);
pim_vdev_root_destroy(vdev_rank_root);
// pim_vsysfs_root_destroy(vsysfs_dax_root);
pim_vsysfs_root_destroy(vsysfs_rank_root);
pim_vmodule_destroy(vmodule);
```

#### DPU Thread Pool

The thread pool provides concurrent handler execution, decoupling detection from execution. It is **optional** — without a pool, handlers execute directly in the detecting thread.

The pool also **limits the number of concurrently running DPU handlers** to the worker count. When all workers are busy, new tasks queue until a worker becomes available. This prevents resource exhaustion when many regions trigger simultaneously.

**With Polling Monitor:**

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Monitor Thread  │     │   Thread Pool    │     │  DPU Handlers    │
│  (polling loop)  │     │   (workers)      │     │                  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         │  detect change         │                        │
         │  (scan phase)          │                        │
         │                        │                        │
         │  if pool_running:      │                        │
         │    pool_submit(task)   │                        │
         │───────────────────────>│                        │
         │  (non-blocking)        │  worker dequeues       │
         │                        │───────────────────────>│
         │  continue polling      │                        │ execute handler
         │                        │                        │
         │  else (no pool):       │                        │
         │    handler() directly  │                        │
         │────────────────────────────────────────────────>│
         │  (blocking)            │                        │
```

- **With pool**: Monitor submits task and continues polling immediately. Handlers run concurrently in worker threads.
- **Without pool**: Monitor executes handler directly after scan. Sequential execution per monitor thread.

**With SIGSEGV Handler:**

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Main Thread    │     │ Callback Thread  │     │   Thread Pool    │     │  DPU Handlers    │
│                  │     │ (per CTRL_MMIO)  │     │   (workers)      │     │                  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │                        │
         │  write to CTRL_MMIO    │                        │                        │
         │  ───► SIGSEGV          │                        │                        │
         │                        │                        │                        │
         │  signal handler:       │                        │                        │
         │    decode instruction  │                        │                        │
         │    emulate write       │                        │                        │
         │    futex_wake ────────>│                        │                        │
         │                        │                        │                        │
         │  instruction skipped   │  if pool_running:      │                        │
         │  (resume execution)    │    pool_submit(task)   │                        │
         │<───────────────────────│───────────────────────>│                        │
         │                        │  (non-blocking)        │  worker dequeues       │
         │                        │                        │───────────────────────>│
         │                        │  wait for next futex   │                        │ execute handler
         │                        │                        │                        │
         │                        │  else (no pool):       │                        │
         │                        │    handler() directly  │                        │
         │                        │────────────────────────────────────────────────>│
         │                        │                        │                        │
```

- **With pool**: Callback thread submits and immediately returns to futex wait. Adds queue overhead but can limit the number of active DPUs.
- **Without pool**: Callback thread executes handler directly (lower latency) but there is no way to limit the number of active DPUs.

**When to Use a Thread Pool:**

| Scenario                                             | Recommendation                                              |
| ---------------------------------------------------- | ----------------------------------------------------------- |
| Maximum throughput with polling-based monitor        | Use pool with workers = `std::thread::hardware_concurrency` |
| Handler does I/O or blocking ops                     | Use pool or SIGSEGV handler                                 |
| Lowest per-call latency                              | No pool (direct handler call)                               |
| Single region / Max performance with SIGSEGV handler | No pool needed (lower latency)                              |
| Limit concurrent DPU execution with SIGSEGV handler  | Use pool (set concurrency via `pim_pool_start(dev, n)`)     |

**Usage Pattern:**
```c
// Start pool before monitor (or before enabling SIGSEGV)
pim_pool_start(dev, 8);  // 8 worker threads

// Option A: With polling monitor
pim_start_monitor(dev, 4, 100, 20000, 4);
// ...
pim_stop_monitor(dev);

// Option B: With SIGSEGV handler
pim_g_enable_sigsegv_handler(true);
pim_device_enable_sigsegv(dev);
// ...
pim_device_disable_sigsegv(dev);
pim_g_disable_sigsegv_handler();

// Stop pool after stopping detection
pim_pool_stop(dev);
```

#### Comparison

| Feature                                                    | Polling Monitor          | SIGSEGV Handler                     | Virtual Device (vdev)                    |
| ---------------------------------------------------------- | ------------------------ | ----------------------------------- | ---------------------------------------- |
| Latency                                                    | ~microseconds (adaptive) | ~microseconds (signal + futex wake) | ~nanoseconds (direct function call)      |
| Idle-time CPU overhead                                     | Adaptive polling         | None (event-driven)                 | None (event-driven)                      |
| Require LD_PRELOAD for transparency                        | No                       | Yes                                 | Yes                                      |
| Async-Signal-Safe                                          | Yes                      | Yes (runs in callback thread)       | Yes                                      |
| Detect writes from other processes                         | Yes                      | No (only detects in-process writes) | No (only in-process via intercepted API) |
| Require custom userspace ioctl/driver interception routine | No                       | No                                  | Yes                                      |

## Building

### Prerequisites

- CMake 3.16.3+
- C++20 compiler (clang++ or g++)
- Ninja (recommended) or Make
- Linux
- x86-64 or AArch64 target architecture

### Build Steps

For simulator throughput and benchmark runs, **Release build is strongly recommended**.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

For instruction-level debugging / trace-heavy development, use Debug build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc)"
```

> If you reuse the same build directory, re-run `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=...` before rebuilding.

You can also use presets (see `CMakePresets.json`):

```bash
cmake --preset release
cmake --build --preset release
```

Or:

```bash
cmake --preset debug
cmake --build --preset debug
```

### Output

- `lib/libhostpimsim.a` - Static library
- `lib/libhostpimsim.so` - Shared library

### Installation

```bash
sudo cmake --install build
sudo ldconfig
```

Default install prefix: `/usr/local`

## Usage

### Interacting from Another Process

External processes can access the PIM shared memory directly via `mmap()`. Here's a minimal example:

```c
#include <stdint.h>

#include <fcntl.h>

#include <sys/mman.h>

#define NUM_CHUNKS 4
#define RAM_SIZE (64UL * 1024 * 1024)             // 64MB per chunk
#define CTRL_MMIO_SIZE 4096                       // 4KB MMIO page
#define CTRL_RW_SIZE 4096                         // 4KB RW page
#define CTRL_SIZE (CTRL_MMIO_SIZE + CTRL_RW_SIZE) // 8KB total per chunk

int main() {
  // Open shm file (already created by PIM simulator)
  int fd = open("/dev/shm/pim", O_RDWR);
  // Or: dev = open("/dev/uio0", O_RDWR);

  // Memory layout: [RAM_0 | RAM_1 | ... | RAM_N] [CTRL_0 | CTRL_1 | ... |
  // CTRL_N]
  size_t ram_region_size = NUM_CHUNKS * RAM_SIZE;
  size_t ctrl_region_size = NUM_CHUNKS * CTRL_SIZE;
  size_t total_size = ram_region_size + ctrl_region_size;

  // Map the shared memory
  uint8_t *mem = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                      0); // `offset` might not be honored.
  // Physical memory address may not be contiguous.

  // Lock pages in RAM (recommended for real PIM - prevents page faults)
  mlock(mem, total_size);

  // Access control and data regions
  // (offsets depend on your simulator's memory layout)
  float *ram = (float *)(mem); // RAM region
  volatile uint64_t *ctrl_mmio =
      (volatile uint64_t *)(mem + RAM_SIZE); // Control region
  volatile uint64_t *ctrl_rw =
      (volatile uint64_t *)(mem + RAM_SIZE + CTRL_MMIO_SIZE); // Control region

  // Write data, configure registers, trigger operation...

  // Cleanup
  munlock(mem, total_size);
  munmap(mem, total_size);
  close(fd);
  return 0;
}
```

### Using the Library in Your Simulator

```c
#include <stdio.h>

#include <hostpimsim.h>

void my_dpu_handler(pim_device_t *dev, pim_region_t *region, size_t offset,
                    uint64_t value, void *user_data) {
  size_t chunk_id = (size_t)user_data;
  printf("DPU %zu triggered with CMD: %lu\n", chunk_id, value);

  // ... run your simulation ...
}

#define NUM_CHUNKS 4
#define RAM_SIZE (64UL * 1024 * 1024)             // 64MB per chunk
#define CTRL_MMIO_SIZE 4096                       // 4KB MMIO page
#define CTRL_RW_SIZE 4096                         // 4KB RW page
#define CTRL_SIZE (CTRL_MMIO_SIZE + CTRL_RW_SIZE) // 8KB total per chunk

int main() {
  // Memory layout: [RAM_0 | RAM_1 | ... | RAM_N] [CTRL_0 | CTRL_1 | ... |
  // CTRL_N]
  size_t ram_region_size = NUM_CHUNKS * RAM_SIZE;
  size_t ctrl_region_size = NUM_CHUNKS * CTRL_SIZE;
  size_t total_size = ram_region_size + ctrl_region_size;

  // Create device (or attach to existing)
  pim_device_t *dev = pim_device_create("/pim", total_size);
  // Or: dev = pim_device_attach("/dev/uio0", total_size, 0);

  // Create regions: RAM first, then control registers
  for (size_t i = 0; i < NUM_CHUNKS; i++) {
    size_t data_offset = i * RAM_SIZE;
    size_t ctrl_base = ram_region_size + i * CTRL_SIZE;

    // RAM region
    pim_region_t *ram =
        pim_region_create(dev, data_offset, RAM_SIZE, PIM_REGION_RAM);
    // CTRL_MMIO page (Control register, write-protected in SIGSEGV mode)
    pim_region_t *ctrl_mmio =
        pim_region_create(dev, ctrl_base, CTRL_MMIO_SIZE, PIM_REGION_CTRL_MMIO);
    // CTRL_RW page (STATUS registers, etc.)
    pim_region_t *ctrl_rw = pim_region_create(dev, ctrl_base + CTRL_MMIO_SIZE,
                                              CTRL_RW_SIZE, PIM_REGION_CTRL_RW);

    // Register DPU handler on CMD register (offset 0 within MMIO page)
    pim_register_dpu_handler(dev, ctrl_mmio, 0, sizeof(uint64_t), 0,
                             my_dpu_handler, (void *)i);
  }

  // Start monitoring for external writes (1 monitor thread)
  pim_start_monitor(dev, 1, 100, 20000,
                    4); // 1 monitor, 100us sleep, 20000 spins, divisor=4

  // ... run your simulation ...

  // Cleanup
  pim_stop_monitor(dev);
  pim_device_unlink(dev); // Remove runtime-created shm object (optional)
  pim_device_destroy(dev);
  return 0;
}
```

## API Reference

### Device Management
| Function                                   | Description                                                                                      |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `pim_device_create(shm_path, size)`        | Create PIM device with shared memory file (and wipe it)                                          |
| `pim_device_attach(path, size, offset)`    | Attach to existing shared memory or device file (and wipe it)                                    |
| `pim_device_init(ptr, size)`               | Initialize handle for already-mapped memory as PIM device                                        |
| `pim_device_deinit(dev)`                   | Release handle without unmapping memory                                                          |
| `pim_device_destroy(dev)`                  | Destroy device and free resources                                                                |
| `pim_device_unlink(dev)`                   | Delete runtime-created shm object only                                                           |
| `pim_device_get_path(dev)`                 | Get backing path                                                                                 |
| `pim_device_get_size(dev)`                 | Get mapped device memory size                                                                    |
| `pim_device_get_ram_size(dev)`             | Get total size of all `PIM_REGION_RAM` regions                                                   |
| `pim_device_get_ptr(dev)`                  | Get device base pointer                                                                          |
| `pim_device_set_user_data(dev, user_data)` | Attach opaque user data to a device                                                              |
| `pim_device_get_user_data(dev)`            | Read opaque user data from a device                                                              |
| `pim_device_get_region(dev, offset)`       | Get region whose start offset exactly matches `offset`                                           |
| `pim_device_get_next_region(dev, cur)`     | Iterate regions by ascending offset (`cur=NULL` returns first); Returns `NULL` if iteration ends |
| `pim_device_barrier(dev)`                  | Wait for device pool barrier (`PIM_ERR_WOULD_DEADLOCK` in pool-worker context)                   |

> **Device Allocation Functions:**
> - `pim_device_create()` - Creates new shm file and maps it
> - `pim_device_attach()` - Opens existing backing file
>   - pass `size=0` only for regular files (uses `st_size - offset`)
>   - for non-regular files/devices, pass explicit `size > 0`
> - `pim_device_init()` - Wraps externally mmap-ed memory; use `pim_device_deinit()` to release
> - `pim_device_deinit()` / `pim_device_destroy()` return `pim_error_t` status
>   - `pim_device_deinit()` is for `pim_device_init()` handles (no unmap/close)
>   - `pim_device_destroy()` unmaps/closes create/attach handles
> - `pim_device_unlink()` - Valid only for `pim_device_create()` handles in this runtime
>
> `pim_device_barrier(dev)` is a device-level synchronization API. It blocks until all submitted pool tasks are finished, and returns `PIM_ERR_WOULD_DEADLOCK` when called from a pool-worker context.

### Region Management
| Function                                     | Description                                  |
| -------------------------------------------- | -------------------------------------------- |
| `pim_region_create(dev, offset, size, type)` | Create region view at offset                 |
| `pim_region_free(region)`                    | Free region handle                           |
| `pim_region_get_ptr(region)`                 | Get region pointer                           |
| `pim_region_get_size(region)`                | Get region size                              |
| `pim_region_get_offset(region)`              | Get region offset in shm                     |
| `pim_region_get_device(region)`              | Get parent device                            |
| `pim_region_msync(region)`                   | Sync region range to backing file            |
| `pim_region_mlock(region)`                   | Lock region pages in memory                  |
| `pim_region_munlock(region)`                 | Unlock region pages in memory                |
| `pim_region_mprotect(region, access)`        | Change region page protection by access mode |
| `pim_region_memset(region, value, size)`     | Fill region bytes from offset 0              |

Region topology APIs (`pim_region_create` / `pim_region_free`) are control-path operations. They are rejected while monitor threads are active, while SIGSEGV mode is active, and during SIGSEGV enable/disable transition windows.

### DPU Handlers
| Function                                                                         | Description                                       |
| -------------------------------------------------------------------------------- | ------------------------------------------------- |
| `pim_register_dpu_handler(dev, region, offset, size, depth, handler, user_data)` | Register handler                                  |
| `pim_unregister_dpu_handler(dev, region, offset, depth)`                         | Remove handler at offset/depth                    |
| `pim_invoke_dpu_handler(dev, region, offset, value, depth)`                      | Directly invoke handler without writing to memory |

The `size` parameter in `pim_register_dpu_handler()` specifies the register width and must be `1`, `2`, `4`, or `8` bytes.
`depth=0` is the root/default handler path (typed writes/monitor/SIGSEGV). `depth>0` is for nested explicit dispatch.

Handler table mutation APIs (`pim_register_dpu_handler` /
`pim_unregister_dpu_handler`) are rejected with `PIM_ERR_BUSY` while monitor
threads are active, while SIGSEGV mode is active, during SIGSEGV transition
windows, or while device teardown is in progress.

`pim_invoke_dpu_handler()` looks up the handler registered at `(offset, depth)` and calls it with the provided `value`, without performing any memory write. This is the recommended dispatch method when `pim_write(...)` cannot be used — for example, when the hardware uses interleaved or encoded memory writes that are decoded externally (e.g., in simulator-side decode loops or vdev ioctl handlers). If a thread pool is running, the handler is submitted to the pool; otherwise it runs synchronously.

### Memory Access

These functions are for **same-process** access (e.g., simulator code writing to its own regions).

| Function                                      | Description                                                                              |
| --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `pim_write(region, offset, size, value)`      | Trigger + return immediately (if thread pool is available), typed write (`size`=1/2/4/8) |
| `pim_write_sync(region, offset, size, value)` | Trigger + wait until all submitted pool tasks are finished                               |
| `pim_write_raw(region, offset, size, value)`  | Atomic typed write without handler invocation                                            |
| `pim_read(region, offset, size, &value)`      | Typed read (`size`=1/2/4/8), zero-extended into `uint64_t`                               |
| `pim_memcpy_to(region, offset, data, size)`   | Bulk write (no DPU trigger, not atomic)                                                  |
| `pim_memcpy_from(region, offset, data, size)` | Bulk read                                                                                |

> **SIGSEGV caveat for raw writes:** `pim_write_raw(..., size, ...)` cannot bypass SIGSEGV interception on write-protected `CTRL_MMIO` pages. If SIGSEGV mode is enabled for that page, the write is handled by the SIGSEGV path.

### Thread Pool

The thread pool executes DPU handler callbacks. It is used by both polling-based and SIGSEGV-based monitoring.

| Function                          | Description                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------ |
| `pim_pool_start(dev, n)`          | Start thread pool (n=0 uses CTRL_MMIO region count)                            |
| `pim_pool_stop(dev)`              | Stop thread pool from a control thread (`PIM_ERR_BUSY` in pool-worker context) |
| `pim_set_max_active_dpus(dev, n)` | Resize pool dynamically (`PIM_ERR_BUSY` in pool-worker context)                |
| `pim_get_max_active_dpus(dev)`    | Get current thread pool size                                                   |

Call `pim_pool_start(dev, n)` before starting detection. Pass `n=0` to spawn one worker per CTRL_MMIO region. Use `pim_set_max_active_dpus()` to resize dynamically while running.

When the pool is not running (max_active_dpus = 0), handlers are executed directly by the monitor thread or SIGSEGV callback thread.

### Thread-based Monitoring

The monitor threads detect writes from **any processes** that directly access the shared memory via `mmap()`. They poll control regions and trigger DPU handlers when values change.

| Function                                                                   | Description                                                     |
| -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `pim_start_monitor(dev, num_monitors, sleep_us, poll_limit, spin_divisor)` | Start monitor threads with adaptive polling                     |
| `pim_stop_monitor(dev)`                                                    | Stop monitor threads (`PIM_ERR_BUSY` in monitor-worker context) |
| `pim_is_monitor_running(dev)`                                              | Check monitor status                                            |
| `pim_get_num_monitors(dev)`                                                | Get number of active monitor threads                            |
| `pim_set_sleep_duration_us(dev, sleep_us)`                                 | Set monitor sleep duration (microseconds)                       |
| `pim_get_sleep_duration_us(dev)`                                           | Get monitor sleep duration                                      |
| `pim_set_poll_limit(dev, poll_limit)`                                      | Set spin budget before sleeping                                 |
| `pim_get_poll_limit(dev)`                                                  | Get spin budget setting                                         |
| `pim_set_wakeup_spin_divisor(dev, divisor)`                                | Set post-sleep spin divisor                                     |
| `pim_get_wakeup_spin_divisor(dev)`                                         | Get post-sleep spin divisor                                     |

When `num_monitors > 1`, CTRL_MMIO regions are distributed among monitor threads using modulo assignment (region N is handled by monitor N % num_monitors). This can improve polling throughput for devices with many regions.

The monitor uses **adaptive polling**: it busy-spins for `poll_limit` iterations for low latency, then sleeps for `sleep_us` microseconds when idle. The sleep duration and poll limit can be adjusted at runtime using the setter functions.

Passing `0` to `pim_start_monitor(..., sleep_us, poll_limit, spin_divisor)` or to the individual monitor setters resets that parameter to default (`sleep_us=100`, `poll_limit=20000`, `spin_divisor=4`).

### SIGSEGV-based Monitoring

For near-zero latency write detection without polling overhead, the library can use SIGSEGV-based memory protection:

| Function                                     | Description                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------- |
| `pim_g_enable_sigsegv_handler(enable_chain)` | Install global SIGSEGV handler                                                  |
| `pim_g_disable_sigsegv_handler()`            | Restore original signal handler                                                 |
| `pim_g_is_sigsegv_handler_enabled()`         | Check whether the global SIGSEGV handler is currently installed                 |
| `pim_device_enable_sigsegv(dev)`             | Enable SIGSEGV monitoring for device                                            |
| `pim_device_disable_sigsegv(dev)`            | Disable SIGSEGV monitoring (`PIM_ERR_BUSY` in SIGSEGV callback-worker context)  |
| `pim_is_sigsegv_enabled(dev)`                | Check if device has SIGSEGV monitoring enabled                                  |
| `pim_g_set_sigsegv_chain_handler(boolean)`   | Set whether to chain to old handler on miss                                     |
| `pim_g_is_sigsegv_chain_handler_enabled()`   | Get current chaining setting                                                    |
| `_pim_sigsegv_handler_func(info, ucontext)`  | LD_PRELOAD-facing SIGSEGV helper (returns whether caller should forward signal) |

**Usage Pattern:**
```c
// 1. Enable global handler (once per process)
pim_g_enable_sigsegv_handler(true);

// 2. Enable for specific devices (write-protects their CTRL_MMIO regions)
pim_device_enable_sigsegv(dev);

// ... writes to CTRL_MMIO pages now trigger handlers instantly ...

// 3. Cleanup
pim_device_disable_sigsegv(dev);
pim_g_disable_sigsegv_handler();
```

> **Note:** For LD_PRELOAD usage, `_pim_sigsegv_handler_func(info, ucontext)` is exposed for libraries that intercept signals directly.

> **Instruction Decode/Skip Note:** In SIGSEGV mode, the handler decodes the faulting store instruction to extract value/size and then advances PC/RIP to skip that instruction. Current decoding support is narrow (common x86_64 `MOV`/`MOVNTI` stores and AArch64 `STR`/`STUR` stores). If an instruction cannot be decoded, control is chained to the previous/default SIGSEGV handler path.

### Virtual Device

| Function                                  | Description                                                     |
| ----------------------------------------- | --------------------------------------------------------------- |
| `pim_vdev_root_create(name_glob)`         | Create `/dev` root matcher (e.g., `"dpu_rank*"`).               |
| `pim_vdev_root_destroy(root)`             | Destroy vdev root.                                              |
| `pim_vdev_root_get_name_glob(root)`       | Get root name glob.                                             |
| `pim_vdev_root_set_open_cb(root, fn)`     | Register root open callback.                                    |
| `pim_vdev_root_set_ioctl_cb(root, fn)`    | Register root ioctl callback.                                   |
| `pim_vdev_root_set_mmap_cb(root, fn)`     | Register root mmap callback.                                    |
| `pim_vdev_root_set_close_cb(root, fn)`    | Register root close callback.                                   |
| `pim_vdev_create(root, name)`             | Create concrete vdev instance (e.g., `"dpu_rank0"`).            |
| `pim_vdev_destroy(vdev)`                  | Destroy vdev instance.                                          |
| `pim_vdev_get_name(vdev)`                 | Get concrete instance name.                                     |
| `pim_vdev_get_path(vdev)`                 | Get concrete absolute path (e.g., `/dev/dpu_rank0`).            |
| `pim_vdev_get_root(vdev)`                 | Get owning vdev root.                                           |
| `pim_vdev_set_user_data(vdev, ptr)`       | Attach caller-owned opaque user data pointer to vdev.           |
| `pim_vdev_get_user_data(vdev)`            | Read caller-owned opaque user data pointer from vdev.           |
| `pim_vdev_lock(vdev)`                     | Acquire vdev lifetime (prevents object destruction while held). |
| `pim_vdev_unlock(vdev)`                   | Release one lifetime hold acquired by `pim_vdev_lock()`.        |
| `pim_vdev_register_device(vdev, fd, dev)` | Bind `(vdev, fd)` to a `pim_device_t`.                          |
| `pim_vdev_unregister_device(vdev, fd)`    | Remove `(vdev, fd)` binding.                                    |
| `pim_vdev_find_device(vdev, fd)`          | Look up `pim_device_t` by `(vdev, fd)`.                         |
| `pim_vdev_get_vsysfs(vdev)`               | Get associated vsysfs.                                          |

> **Note:** `pim_vdev_lock()` / `pim_vdev_unlock()` must be used as a pair.

### Virtual Sysfs

| Function                                                                      | Description                                                                                                                                                                                |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `pim_vsysfs_root_create(vdev_root, class_name, bus_name, instance_name_glob)` | Create a vsysfs root bound to a vdev root. Class side is `/sys/class/<class_name>/<vdev_name>`; devices side is `/sys/devices/<bus_name>/<instance_name>/<vdev_name>` (when names differ). |
| `pim_vsysfs_root_destroy(root)`                                               | Destroy vsysfs root.                                                                                                                                                                       |
| `pim_vsysfs_root_get_class_glob(root)`                                        | Get root class glob.                                                                                                                                                                       |
| `pim_vsysfs_root_get_devices_glob(root)`                                      | Get root devices glob.                                                                                                                                                                     |
| `pim_vsysfs_root_create_default_attr(root, name, value, access)`              | Set default attribute and access mode for instances under this root.                                                                                                                       |
| `pim_vsysfs_root_find_default_attr(root, name)`                               | Get root default attribute.                                                                                                                                                                |
| `pim_vsysfs_root_remove_default_attr(root, name)`                             | Remove root default attribute and remove the key from existing instances under that root.                                                                                                  |
| `pim_vsysfs_create(root, vdev, instance_name)`                                | Create concrete vsysfs instance bound to a concrete vdev instance.                                                                                                                         |
| `pim_vsysfs_destroy(vsysfs)`                                                  | Destroy vsysfs instance.                                                                                                                                                                   |
| `pim_vsysfs_get_class_name(vsysfs)`                                           | Get class-side name.                                                                                                                                                                       |
| `pim_vsysfs_get_device_name(vsysfs)`                                          | Get devices-side name.                                                                                                                                                                     |
| `pim_vsysfs_get_class_dir(vsysfs)`                                            | Get class-side absolute directory.                                                                                                                                                         |
| `pim_vsysfs_get_devices_dir(vsysfs)`                                          | Get devices-side absolute directory.                                                                                                                                                       |
| `pim_vsysfs_create_attr(vsysfs, name, value, access)`                         | Set/update instance attribute and access mode.                                                                                                                                             |
| `pim_vsysfs_find_attr(vsysfs, name)`                                          | Get instance attribute value.                                                                                                                                                              |
| `pim_vsysfs_remove_attr(vsysfs, name)`                                        | Remove instance attribute.                                                                                                                                                                 |

### Virtual Module

| Function                                                | Description                                                |
| ------------------------------------------------------- | ---------------------------------------------------------- |
| `pim_vmodule_create(module_name)`                       | Create a virtual module under `/sys/module/<module_name>`. |
| `pim_vmodule_destroy(vmodule)`                          | Destroy virtual module instance.                           |
| `pim_vmodule_get_name(vmodule)`                         | Get module name (without `/sys/module/` prefix).           |
| `pim_vmodule_get_dir(vmodule)`                          | Get absolute module directory path (`/sys/module/<name>`). |
| `pim_vmodule_create_attr(vmodule, name, value, access)` | Set/update module attribute with access mode.              |
| `pim_vmodule_find_attr(vmodule, name)`                  | Get module attribute value.                                |
| `pim_vmodule_remove_attr(vmodule, name)`                | Remove module attribute.                                   |

### Virtual Syscall

| Function                                                     | Description                               |
| ------------------------------------------------------------ | ----------------------------------------- |
| `_pim_vsyscall_dispatch(number, a, b, c, d, e, f, &forward)` | Dispatch raw syscall through vfile layer. |

> **Note:** `_pim_vsyscall_dispatch()` requires `liboverlaysys` as the active LD_PRELOAD syscall interceptor. It is the single entry point called from the external hook — it handles `open()`/`openat()`, `access()`/`faccessat()`/`faccessat2()`, `readlink()`/`readlinkat()`, `getdents()`/`getdents64()`, `ioctl()`, `read()`/`write()`, `mmap()`/`munmap()`, `close()`/`close_range()`, `dup()`/`dup2()`/`dup3()`, `lseek()`, and stat-family syscalls for owned vfile fds, and returns wrapper-style errors (`-1` with `errno` set; `0` if successful).

### Global Virtual File Control

| Function                | Description                                                                    |
| ----------------------- | ------------------------------------------------------------------------------ |
| `pim_vfile_open(flags)` | Reserve fd number for vfile layer; Useful when implementing `open()` callback. |
| `pim_vfile_owns_fd(fd)` | Check if fd belongs to vfile layer.                                            |
| `pim_vfile_clear()`     | Destruct all vfile-related data globally.                                      |

> **Important:** When using `liboverlaysys` (LD_PRELOAD syscall interception), explicitly call `pim_vfile_clear()` from your `__attribute__((destructor))` cleanup function to release all vfile-owned fds and virtual-node state before process teardown.

### Combining Both Monitoring Modes

Thread-based polling and SIGSEGV-based monitoring are **not mutually exclusive** and can be used together in a multi-process scenario:

- **Simulator process**: Runs the polling monitor (`pim_start_monitor`) to detect writes from external processes
- **Application process**: Uses LD_PRELOAD with SIGSEGV handler for low-latency command dispatch

This works because the SIGSEGV handler **emulates** the write instruction without actually writing to the CTRL_MMIO page (the page remains read-only). Since no actual memory write occurs, the polling monitor in the simulator process sees no value change and does not trigger a duplicate handler invocation.

```
┌─────────────────────┐                     ┌─────────────────────┐
│  Application        │                     │  Simulator          │
│  (LD_PRELOAD)       │                     │  (polling monitor)  │
└──────────┬──────────┘                     └──────────┬──────────┘
           │                                           │
           │  write CMD=EXEC                           │
           │  ───────► SIGSEGV!                        │
           │           │                               │
           │  handler decodes & dispatches             │
           │  (page stays PROT_READ)                   │
           │           │                               │
           │  instruction skipped                      │  polling sees no change
           │  ◄───────                                 │  (value unchanged)
           │                                           │
```

This separation allows the simulator to handle writes from **both**:
1. **LD_PRELOAD applications**: Instant dispatch via SIGSEGV (in-process)
2. **Legacy applications**: Detected by polling monitor (cross-process)

## Error Codes

| Code | Name                     | Description                                      |
| ---- | ------------------------ | ------------------------------------------------ |
| 0    | `PIM_SUCCESS`            | Operation completed successfully                 |
| -1   | `PIM_ERR_INVALID_ARG`    | Invalid argument or malformed request            |
| -2   | `PIM_ERR_NO_MEMORY`      | Allocation failed (heap/container growth)        |
| -3   | `PIM_ERR_SHM_OPEN`       | POSIX shm open/unlink operation failed           |
| -4   | `PIM_ERR_NOT_FOUND`      | Requested resource/entry not found               |
| -5   | `PIM_ERR_ALREADY_EXISTS` | Resource/entry already exists or overlaps        |
| -6   | `PIM_ERR_OUT_OF_BOUNDS`  | Requested range exceeds region bounds            |
| -7   | `PIM_ERR_SIGNAL`         | Signal-handler registration/query/restore failed |
| -8   | `PIM_ERR_BUSY`           | Operation rejected due to active runtime state   |
| -9   | `PIM_ERR_NOT_SUPPORTED`  | Operation intentionally unsupported              |
| -10  | `PIM_ERR_WOULD_DEADLOCK` | Operation would deadlock in caller context       |
| -11  | `PIM_ERR_MMAP`           | `mmap()` failed (see `errno`)                    |
| -12  | `PIM_ERR_MSYNC`          | `msync()` failed (see `errno`)                   |
| -13  | `PIM_ERR_MPROTECT`       | `mprotect()` failed (see `errno`)                |
| -14  | `PIM_ERR_MLOCK`          | `mlock()` failed (see `errno`)                   |
| -15  | `PIM_ERR_MUNLOCK`        | `munlock()` failed (see `errno`)                 |

## Access Modes

`pim_access_mode_t` is currently used for region protection (`pim_region_mprotect`) and virtual attribute permissions (`pim_vsysfs_*`, `pim_vmodule_*`):

| Code | Name                   | Description |
| ---- | ---------------------- | ----------- |
| 0    | `PIM_ACCESS_MODE_NONE` | No access   |
| 1    | `PIM_ACCESS_MODE_RO`   | Read-only   |
| 2    | `PIM_ACCESS_MODE_WO`   | Write-only  |
| 3    | `PIM_ACCESS_MODE_RW`   | Read/Write  |

## License

See LICENSE file for details.
