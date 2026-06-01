/**
 * @file vsyscall.cc
 * @brief Syscall dispatch, owned-fd lifecycle, and virtual-fs path handling
 *
 * The external syscall hook (e.g., liboverlaysys) must call
 * `_pim_vsyscall_dispatch()`. This module owns:
 *   - owned virtual-fd tables and alias tracking
 *   - virtual-fs snapshot/cache for /dev and /sys nodes
 *   - open/read/write/ioctl/mmap/stat/getdents/... syscall dispatch
 */

#include "vfile.hh"

#include <algorithm>
#include <cstring>
#include <unordered_set>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/vfs.h>

#include <linux/magic.h>

/*============================================================================
 * Global State
 *============================================================================*/

static constexpr mode_t kVsyscallVsysfsAttrModeRO = 0444;
static constexpr mode_t kVsyscallVsysfsAttrModeWO = 0222;
static constexpr mode_t kVsyscallVsysfsAttrModeRW = 0644;

static constexpr size_t kVsyscallStatBlockUnitBytes = 512U;
static constexpr blksize_t kVsyscallStatPreferredBlksize = 4096;
static constexpr size_t kVsyscallStatfsNameMax = 255U;

static constexpr mode_t kVsyscallVirtualChrNodeMode = 0600;
static constexpr mode_t kVsyscallVirtualDirNodeMode = 0555;
static constexpr unsigned int kVsyscallVirtualChrRdevMajor =
    1U; // default major (/dev/null-like)
static constexpr unsigned int kVsyscallVirtualChrRdevMinor =
    3U; // default minor (/dev/null-like)
static constexpr int kVsyscallProcFdinfoVirtualMntId = 0;

struct vsyscall_global_state {
  std::mutex fd_table_mutex;
  std::unordered_map<int, std::shared_ptr<vfile_fd_entry>> fd_table;
  std::mutex mmap_table_mutex;
  std::map<uintptr_t, size_t> mmap_table;
};
/*
 * Keep syscall-dispatch global tables alive until process death.
 *
 * Rationale:
 * - _pim_vsyscall_dispatch can still be reached in late exit/flush paths
 *   (e.g., stdio write during _IO_cleanup) depending on preload hook lifetime.
 * - File-scope C++ containers/mutexes with static storage may be destroyed
 * before those late syscalls, which can turn fd lookup into use-after-dtor
 * crashes.
 *
 * We intentionally allocate once and never destroy to keep lookup path safe
 * even when teardown hooks are omitted.
 */
static vsyscall_global_state *const g_vsyscall_global_state =
    new vsyscall_global_state();

static std::mutex &g_fd_table_mutex = g_vsyscall_global_state->fd_table_mutex;
static std::unordered_map<int, std::shared_ptr<vfile_fd_entry>> &g_fd_table =
    g_vsyscall_global_state->fd_table;

static std::mutex &g_mmap_table_mutex =
    g_vsyscall_global_state->mmap_table_mutex;
static std::map<uintptr_t, size_t> &g_mmap_table =
    g_vsyscall_global_state->mmap_table;

static const int kMaxGroups = []() noexcept -> int {
#ifdef _SC_NGROUPS_MAX
  const long max_groups_long = sysconf(_SC_NGROUPS_MAX);
  if (max_groups_long > 0 &&
      max_groups_long <= static_cast<long>(std::numeric_limits<int>::max())) {
    return static_cast<int>(max_groups_long);
  }
#endif
  return std::numeric_limits<int>::max();
}();

/*============================================================================
 * FD Table
 *============================================================================*/

static inline void vsyscall_consume_syscall(int *forward) noexcept {
  if (forward) {
    *forward = 0;
  }
}

static inline long vsyscall_consume_and_fail(int *forward, int err) noexcept {
  vsyscall_consume_syscall(forward);
  return vfile_fail_with_errno(err);
}

static inline long vsyscall_wrapper_from_negative(long ret,
                                                  int fallback_err) noexcept {
  if (ret >= 0) {
    return ret;
  }

  // Callback contract is negative errno. Also accept -1 + errno as fallback.
  if (ret == -1) {
    return vfile_fail_with_errno((errno > 0) ? errno : fallback_err);
  }

  const long mapped = -ret;
  if (mapped > 0 &&
      mapped <= static_cast<long>(std::numeric_limits<int>::max())) {
    return vfile_fail_with_errno(static_cast<int>(mapped));
  }
  return vfile_fail_with_errno(fallback_err);
}

static inline bool vsyscall_fd_opened_with_path_only(int open_flags) noexcept {
  return (open_flags & O_PATH) != 0;
}

static inline int vsyscall_fd_access_mode(int open_flags) noexcept {
#ifdef O_ACCMODE
  return open_flags & O_ACCMODE;
#else
  (void)open_flags;
  return O_RDONLY;
#endif
}

static inline bool vsyscall_fd_can_write(int open_flags) noexcept {
  const int mode = vsyscall_fd_access_mode(open_flags);
  return mode == O_WRONLY || mode == O_RDWR;
}

static inline int vsyscall_fcntl_setfl_supported_mask() noexcept {
  int mask = 0;
#ifdef O_APPEND
  mask |= O_APPEND;
#endif
#ifdef O_ASYNC
  mask |= O_ASYNC;
#endif
#ifdef O_DIRECT
  mask |= O_DIRECT;
#endif
#ifdef O_NOATIME
  mask |= O_NOATIME;
#endif
#ifdef O_NONBLOCK
  mask |= O_NONBLOCK;
#endif
  return mask;
}

static inline bool
vsyscall_vsysfs_attr_access_allows_read(pim_access_mode_t access) noexcept {
  return (access & PIM_ACCESS_MODE_RO) != 0;
}

static inline bool
vsyscall_vsysfs_attr_access_allows_write(pim_access_mode_t access) noexcept {
  return (access & PIM_ACCESS_MODE_WO) != 0;
}

static inline mode_t
vsyscall_vsysfs_attr_perm_bits(pim_access_mode_t access) noexcept {
  switch (access) {
  case PIM_ACCESS_MODE_NONE:
    return 0;
  case PIM_ACCESS_MODE_RO:
    return kVsyscallVsysfsAttrModeRO;
  case PIM_ACCESS_MODE_WO:
    return kVsyscallVsysfsAttrModeWO;
  case PIM_ACCESS_MODE_RW:
    return kVsyscallVsysfsAttrModeRW;
  }
  return kVsyscallVsysfsAttrModeRW;
}

static inline pim_access_mode_t
vsyscall_vsysfs_get_attr_access_locked(const pim_vsysfs_t *vsysfs,
                                       const std::string &attr_name) noexcept {
  if (!vsysfs) {
    return PIM_ACCESS_MODE_RW;
  }
  auto it = vsysfs->attr_access.find(attr_name);
  if (it == vsysfs->attr_access.end()) {
    return PIM_ACCESS_MODE_RW;
  }
  return it->second;
}

static inline pim_access_mode_t
vsyscall_vmodule_get_attr_access_locked(const pim_vmodule_t *vmodule,
                                        const std::string &attr_name) noexcept {
  if (!vmodule) {
    return PIM_ACCESS_MODE_RW;
  }
  auto it = vmodule->attr_access.find(attr_name);
  if (it == vmodule->attr_access.end()) {
    return PIM_ACCESS_MODE_RW;
  }
  return it->second;
}

static inline int vsyscall_kernel_close_fd(int fd) noexcept {
  return (::close(fd) == 0) ? 0 : -1;
}

static inline long
vsyscall_fail_enomem_with_optional_close(int fd, bool close_fd) noexcept {
  if (close_fd) {
    (void)vsyscall_kernel_close_fd(fd);
  }
  return vfile_fail_with_errno(ENOMEM);
}

template <typename Fn>
static inline void vsyscall_run_best_effort(Fn &&fn) noexcept {
  try {
    fn();
  } catch (...) {
    // Best effort only.
  }
}

static inline bool vsyscall_range_end(uintptr_t start, size_t length,
                                      uintptr_t *end) noexcept {
  if (!end) {
    return false;
  }
  if (length >
      static_cast<size_t>(std::numeric_limits<uintptr_t>::max() - start)) {
    return false;
  }
  *end = start + length;
  return true;
}

static inline void vsyscall_track_mmap_region(void *addr,
                                              size_t length) noexcept {
  if (addr == MAP_FAILED || length == 0) {
    return;
  }
  const uintptr_t start = reinterpret_cast<uintptr_t>(addr);
  uintptr_t end = 0;
  if (!vsyscall_range_end(start, length, &end)) {
    return;
  }
  (void)end;

  vsyscall_run_best_effort([&]() {
    std::lock_guard<std::mutex> lock(g_mmap_table_mutex);
    g_mmap_table[start] = length;
  });
}

static inline bool
vsyscall_fd_bind_existing(int fd, const std::shared_ptr<vfile_fd_entry> &entry);

static inline long
vsyscall_alloc_tracked_fd(std::shared_ptr<vfile_fd_entry> entry,
                          int *forward) noexcept {
  int fd = vsyscall_fd_alloc(entry);
  if (fd < 0) {
    return vsyscall_consume_and_fail(forward, (errno > 0) ? errno : ENOMEM);
  }
  if (entry && entry->type == vfile_fd_entry::FD_VDEV && entry->owner_fd < 0) {
    entry->owner_fd = fd;
  }
  vsyscall_consume_syscall(forward);
  return fd;
}

static inline bool
vsyscall_bind_existing_fd_or_close(int fd,
                                   const std::shared_ptr<vfile_fd_entry> &entry,
                                   int fallback_errno) noexcept {
  if (vsyscall_fd_bind_existing(fd, entry)) {
    return true;
  }

  const int bind_errno = (errno > 0) ? errno : fallback_errno;
  (void)vsyscall_kernel_close_fd(fd);
  errno = bind_errno;
  return false;
}

static inline bool
vsyscall_fd_bind_existing(int fd,
                          const std::shared_ptr<vfile_fd_entry> &entry) {
  if (fd < 0 || !entry) {
    errno = EINVAL;
    return false;
  }

  bool inserted = false;
  const bool ok = vfile_try_or_false_with_errno(
      [&]() {
        std::lock_guard<std::mutex> lock(g_fd_table_mutex);
        auto [_, emplaced] = g_fd_table.emplace(fd, entry);
        inserted = emplaced;
        if (!inserted) {
          errno = EEXIST;
        }
      },
      ENOMEM);
  if (!ok) {
    return false;
  }
  return inserted;
}

int vsyscall_fd_alloc(std::shared_ptr<vfile_fd_entry> entry) {
  int fd = pim_vfile_open(entry->flags);
  if (fd < 0) {
    if (errno <= 0) {
      errno = EMFILE;
    }
    return -1;
  }
  if (!vsyscall_fd_bind_existing(fd, entry)) {
    const int bind_errno = (errno > 0) ? errno : ENOMEM;
    (void)vsyscall_kernel_close_fd(fd);
    errno = bind_errno;
    return -1;
  }
  return fd;
}

std::shared_ptr<vfile_fd_entry> vsyscall_fd_lookup(int fd) {
  std::lock_guard<std::mutex> lock(g_fd_table_mutex);
  auto it = g_fd_table.find(fd);
  if (it == g_fd_table.end())
    return {};
  return it->second;
}

void vsyscall_fd_free(int fd) {
  std::lock_guard<std::mutex> lock(g_fd_table_mutex);
  g_fd_table.erase(fd);
}

static inline bool vsyscall_fd_is_owned(int fd) noexcept {
  return vsyscall_fd_lookup(fd) != nullptr;
}

static inline bool vsyscall_fd_has_alias_locked(
    const std::shared_ptr<vfile_fd_entry> &entry) noexcept {
  if (!entry) {
    return false;
  }
  for (const auto &[fd, candidate] : g_fd_table) {
    (void)fd;
    if (candidate.get() == entry.get()) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Acquire an fd entry for non-close operations.
 *
 * Returns a strong reference and increments active operation count.
 * If fd is closing or missing, returns empty.
 */
static std::shared_ptr<vfile_fd_entry> vsyscall_fd_acquire_for_op(int fd) {
  std::shared_ptr<vfile_fd_entry> entry = vsyscall_fd_lookup(fd);
  if (!entry) {
    return {};
  }

  std::lock_guard<std::mutex> lock(entry->op_mutex);
  if (entry->closing) {
    return {};
  }
  ++entry->active_ops;
  return entry;
}

static inline std::shared_ptr<vfile_fd_entry>
vsyscall_acquire_owned_fd_for_dispatch(int fd, int *forward) noexcept {
  auto entry = vsyscall_fd_acquire_for_op(fd);
  if (!entry) {
    return {};
  }
  vsyscall_consume_syscall(forward);
  return entry;
}

/**
 * @brief Release an fd entry after a non-close operation.
 */
static void
vsyscall_fd_release_after_op(const std::shared_ptr<vfile_fd_entry> &entry) {
  if (!entry) {
    return;
  }

  std::lock_guard<std::mutex> lock(entry->op_mutex);
  if (entry->active_ops > 0) {
    --entry->active_ops;
  }
  if (entry->closing && entry->active_ops == 0) {
    entry->op_cv.notify_all();
  }
}

/**
 * @brief RAII guard that guarantees active-op release on all return paths.
 */
class vsyscall_active_op_guard {
public:
  explicit vsyscall_active_op_guard(
      std::shared_ptr<vfile_fd_entry> entry) noexcept
      : entry_(std::move(entry)) {}

  ~vsyscall_active_op_guard() noexcept { vsyscall_fd_release_after_op(entry_); }

private:
  std::shared_ptr<vfile_fd_entry> entry_;
};

enum class vsyscall_path_resolve_status { NOT_OURS, RESOLVED, ERROR };

struct vsyscall_dispatch_resolved_path {
  vsyscall_path_resolve_status status{vsyscall_path_resolve_status::NOT_OURS};
  std::string abs_path;
  bool from_owned_dfd{false};
  bool trailing_slash{false};
  int resolve_errno{0};
};

enum class vsyscall_dispatch_path_status { READY, RETURN };

struct vsyscall_dispatch_path_result {
  vsyscall_dispatch_path_status status{vsyscall_dispatch_path_status::RETURN};
  vsyscall_dispatch_resolved_path path;
  long ret{0};
};

enum class vsyscall_dispatch_node_status { RESOLVED, PROC_FDINFO, STOP };

struct vsyscall_dispatch_node_result {
  vsyscall_dispatch_node_status status{vsyscall_dispatch_node_status::STOP};
  vfile_virtual_path_info node;
  int proc_fdinfo_target_fd{-1};
  bool from_owned_dfd{false};
  long ret{0};
};

enum class vsyscall_stat_lookup_status { READY, RETURN };

struct vsyscall_stat_lookup_result {
  vsyscall_stat_lookup_status status{vsyscall_stat_lookup_status::RETURN};
  struct stat st{};
  long ret{0};
};

static inline bool vsyscall_path_has_trailing_slash(const char *path) noexcept {
  if (!path) {
    return false;
  }
  const size_t len = strlen(path);
  return (len > 0) && (path[len - 1] == '/');
}

static inline bool vsyscall_is_relative_path(const char *path) noexcept {
  return path && path[0] != '\0' && path[0] != '/';
}

static inline bool vsyscall_parse_proc_self_numeric_path(
    const std::string &abs_path, const char *prefix, int *parsed_fd) noexcept {
  if (!parsed_fd) {
    return false;
  }
  if (!prefix) {
    return false;
  }
  const size_t prefix_len = strlen(prefix);
  if (abs_path.size() <= prefix_len ||
      abs_path.compare(0, prefix_len, prefix) != 0) {
    return false;
  }

  const char *digits = abs_path.c_str() + prefix_len;
  int fd = 0;
  for (const char *p = digits; *p != '\0'; ++p) {
    const char ch = *p;
    if (ch < '0' || ch > '9') {
      return false;
    }
    const int digit = ch - '0';
    if (fd > ((std::numeric_limits<int>::max() - digit) / 10)) {
      return false;
    }
    fd = (fd * 10) + digit;
  }

  *parsed_fd = fd;
  return true;
}

static inline bool
vsyscall_rewrite_proc_self_fd_path_for_open(const std::string &abs_path,
                                            std::string *rewritten_abs_path) {
  if (!rewritten_abs_path) {
    return false;
  }

  int procfd = -1;
  if (!vsyscall_parse_proc_self_numeric_path(abs_path, "/proc/self/fd/",
                                             &procfd)) {
    return false;
  }

  const std::shared_ptr<vfile_fd_entry> procfd_entry =
      vsyscall_fd_lookup(procfd);
  if (!procfd_entry || procfd_entry->virtual_path.empty()) {
    return false;
  }

  std::string normalized;
  if (!vfile_normalize_absolute_path(procfd_entry->virtual_path, &normalized)) {
    return false;
  }
  *rewritten_abs_path = std::move(normalized);
  return true;
}

static vsyscall_path_resolve_status
vsyscall_resolve_path_at(int dfd, const char *pathname,
                         std::string *resolved_path, bool *from_owned_dfd,
                         bool *trailing_slash, int *resolve_errno) {
  if (!resolved_path || !from_owned_dfd || !trailing_slash || !resolve_errno ||
      !pathname) {
    return vsyscall_path_resolve_status::NOT_OURS;
  }
  *from_owned_dfd = false;
  *trailing_slash = false;
  *resolve_errno = 0;

  if (pathname[0] == '\0') {
    return vsyscall_path_resolve_status::NOT_OURS;
  }
  *trailing_slash = vsyscall_path_has_trailing_slash(pathname);

  if (pathname[0] == '/') {
    if (!vfile_normalize_absolute_path(pathname, resolved_path)) {
      return vsyscall_path_resolve_status::NOT_OURS;
    }
    return vsyscall_path_resolve_status::RESOLVED;
  }

  if (dfd == AT_FDCWD) {
    return vsyscall_path_resolve_status::NOT_OURS;
  }

  auto dfd_entry = vsyscall_fd_lookup(dfd);
  if (!dfd_entry) {
    return vsyscall_path_resolve_status::NOT_OURS;
  }

  *from_owned_dfd = true;
  if (dfd_entry->type != vfile_fd_entry::FD_VDIR ||
      dfd_entry->virtual_path.empty()) {
    *resolve_errno = ENOTDIR;
    return vsyscall_path_resolve_status::ERROR;
  }

  std::string joined = dfd_entry->virtual_path;
  if (joined != "/") {
    joined.push_back('/');
  }
  joined.append(pathname);

  if (!vfile_normalize_absolute_path(joined, resolved_path)) {
    *resolve_errno = EINVAL;
    return vsyscall_path_resolve_status::ERROR;
  }
  return vsyscall_path_resolve_status::RESOLVED;
}

static inline vsyscall_dispatch_resolved_path
vsyscall_dispatch_resolve_path(int dfd, const char *pathname) {
  vsyscall_dispatch_resolved_path result;
  std::string resolved_path;
  const vsyscall_path_resolve_status status = vsyscall_resolve_path_at(
      dfd, pathname, &resolved_path, &result.from_owned_dfd,
      &result.trailing_slash, &result.resolve_errno);
  result.status = status;
  if (status == vsyscall_path_resolve_status::RESOLVED) {
    result.abs_path = std::move(resolved_path);
  }
  return result;
}

static inline vsyscall_dispatch_path_result
vsyscall_dispatch_resolve_path_for_dispatch(int dfd, const char *pathname,
                                            int *forward) {
  vsyscall_dispatch_path_result result;
  result.path = vsyscall_dispatch_resolve_path(dfd, pathname);
  if (result.path.status == vsyscall_path_resolve_status::ERROR) {
    result.ret = vsyscall_consume_and_fail(
        forward,
        (result.path.resolve_errno > 0) ? result.path.resolve_errno : EINVAL);
    return result;
  }
  if (result.path.status != vsyscall_path_resolve_status::RESOLVED) {
    result.ret = 0;
    return result;
  }
  result.status = vsyscall_dispatch_path_status::READY;
  return result;
}

template <typename Fn>
static inline long vsyscall_dispatch_owned_fd(int fd, int *forward, Fn &&fn) {
  auto entry = vsyscall_acquire_owned_fd_for_dispatch(fd, forward);
  if (!entry) {
    return 0; /* not ours */
  }
  vsyscall_active_op_guard release_guard(entry);
  return fn(entry);
}

template <typename FnT>
static inline FnT vsyscall_get_callback(const std::shared_ptr<pim_vdev_t> &vdev,
                                        FnT pim_vdev_t::*member) noexcept {
  std::lock_guard<std::mutex> lock(vdev->mutex);
  return vdev.get()->*member;
}

static inline void
vsyscall_get_open_config(const std::shared_ptr<pim_vdev_t> &vdev,
                         pim_vdev_open_fn *open_fn) noexcept {
  if (!open_fn) {
    return;
  }
  std::lock_guard<std::mutex> lock(vdev->mutex);
  *open_fn = vdev->open_fn;
}

template <typename FnT, typename Invoker>
inline long vsyscall_dispatch_vdev_entry_call(
    const std::shared_ptr<vfile_fd_entry> &entry, FnT pim_vdev_t::*member,
    int missing_callback_errno, int non_vdev_errno, Invoker &&invoke) {
  if (entry->type != vfile_fd_entry::FD_VDEV) {
    return vfile_fail_with_errno(non_vdev_errno);
  }
  if (vsyscall_fd_opened_with_path_only(entry->flags)) {
    return vfile_fail_with_errno(EBADF);
  }
  auto vdev = entry->vdev;
  if (!vdev) {
    return vfile_fail_with_errno(EIO);
  }
  if (entry->owner_fd < 0) {
    return vfile_fail_with_errno(EBADF);
  }
  const FnT fn = vsyscall_get_callback(vdev, member);
  if (!fn) {
    return vfile_fail_with_errno(missing_callback_errno);
  }
  return vsyscall_wrapper_from_negative(invoke(fn, vdev.get(), entry->owner_fd),
                                        EIO);
}

template <typename FnT, typename Invoker>
static inline long vsyscall_dispatch_owned_vdev_callback(
    int fd, int *forward, FnT pim_vdev_t::*member, int missing_callback_errno,
    int non_vdev_errno, Invoker &&invoke) {
  return vsyscall_dispatch_owned_fd(
      fd, forward,
      [&invoke, member, missing_callback_errno,
       non_vdev_errno](const std::shared_ptr<vfile_fd_entry> &entry) -> long {
        return vsyscall_dispatch_vdev_entry_call(
            entry, member, missing_callback_errno, non_vdev_errno, invoke);
      });
}

static inline std::shared_ptr<vfile_fd_entry>
vsyscall_make_device_entry(const std::shared_ptr<pim_vdev_t> &vdev,
                           int owner_fd = -1, std::string path = std::string(),
                           int open_flags = 0) {
  auto entry = std::make_shared<vfile_fd_entry>();
  entry->type = vfile_fd_entry::FD_VDEV;
  entry->vdev = vdev;
  entry->owner_fd = owner_fd;
  entry->flags = open_flags;
  entry->virtual_path = std::move(path);
  return entry;
}

static inline std::shared_ptr<vfile_fd_entry> vsyscall_make_vsysfs_entry(
    const std::shared_ptr<pim_vsysfs_t> &vsysfs, std::string attr_name,
    std::string path = std::string(), int open_flags = 0) {
  auto entry = std::make_shared<vfile_fd_entry>();
  entry->type = vfile_fd_entry::FD_VSYSFS;
  entry->vsysfs = vsysfs;
  entry->flags = open_flags;
  entry->sysfs_attr_name = std::move(attr_name);
  entry->sysfs_read_offset = 0;
  entry->sysfs_cached_value.clear();
  entry->sysfs_cached_attr_revision = 0;
  entry->virtual_path = std::move(path);
  return entry;
}

static inline std::shared_ptr<vfile_fd_entry> vsyscall_make_vmodule_entry(
    const std::shared_ptr<pim_vmodule_t> &vmodule, std::string attr_name,
    std::string path = std::string(), int open_flags = 0) {
  auto entry = std::make_shared<vfile_fd_entry>();
  entry->type = vfile_fd_entry::FD_VMODULE;
  entry->vmodule = vmodule;
  entry->flags = open_flags;
  entry->sysfs_attr_name = std::move(attr_name);
  entry->sysfs_read_offset = 0;
  entry->sysfs_cached_value.clear();
  entry->sysfs_cached_attr_revision = 0;
  entry->virtual_path = std::move(path);
  return entry;
}

static inline size_t vsyscall_proc_fdinfo_pos_for_entry(
    const std::shared_ptr<vfile_fd_entry> &entry) noexcept {
  if (!entry) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(entry->op_mutex);
  switch (entry->type) {
  case vfile_fd_entry::FD_VSYSFS:
  case vfile_fd_entry::FD_VMODULE:
  case vfile_fd_entry::FD_PROC_FDINFO:
    return entry->sysfs_read_offset;
  case vfile_fd_entry::FD_VDIR:
    return entry->dir_read_offset;
  case vfile_fd_entry::FD_VDEV:
    return 0;
  }
  return 0;
}

static inline uint64_t vsyscall_proc_fdinfo_ino_for_entry(
    const std::shared_ptr<vfile_fd_entry> &entry) noexcept {
  if (!entry) {
    return 0;
  }
  if (!entry->virtual_path.empty()) {
    return static_cast<uint64_t>(std::hash<std::string>{}(entry->virtual_path));
  }
  return reinterpret_cast<uint64_t>(entry.get());
}

static inline bool vsyscall_build_proc_fdinfo_payload(
    int target_fd, const std::shared_ptr<vfile_fd_entry> &target_entry,
    std::string *payload) {
  if (!payload || target_fd < 0 || !target_entry) {
    return false;
  }

  const size_t pos = vsyscall_proc_fdinfo_pos_for_entry(target_entry);
  const unsigned int target_flags =
      static_cast<unsigned int>(target_entry->flags);
  const unsigned long long ino = static_cast<unsigned long long>(
      vsyscall_proc_fdinfo_ino_for_entry(target_entry));

  char buf[256];
  const int printed = snprintf(
      buf, sizeof(buf), "pos:\t%zu\nflags:\t0%o\nmnt_id:\t%d\nino:\t%llu\n",
      pos, target_flags, kVsyscallProcFdinfoVirtualMntId, ino);
  if (printed < 0 || static_cast<size_t>(printed) >= sizeof(buf)) {
    return false;
  }
  try {
    *payload = std::string(buf, static_cast<size_t>(printed));
  } catch (...) {
    return false;
  }
  return true;
}

static inline std::shared_ptr<vfile_fd_entry> vsyscall_make_proc_fdinfo_entry(
    int target_fd, const std::shared_ptr<vfile_fd_entry> &target_entry,
    std::string path, int open_flags) {
  auto entry = std::make_shared<vfile_fd_entry>();
  entry->type = vfile_fd_entry::FD_PROC_FDINFO;
  entry->flags = open_flags;
  entry->sysfs_attr_name.clear();
  entry->sysfs_read_offset = 0;
  entry->sysfs_cached_attr_revision = 1;
  entry->sysfs_cached_attr_access = PIM_ACCESS_MODE_RO;
  entry->virtual_path = std::move(path);
  if (!vsyscall_build_proc_fdinfo_payload(target_fd, target_entry,
                                          &entry->sysfs_cached_value)) {
    return {};
  }
  return entry;
}

enum class vsyscall_stat_kind { CHR_DEVICE, DIR, REG_ATTR };

static inline bool
vsyscall_is_vsysfs_class_symlink_node(const vfile_virtual_path_info &node,
                                      std::string *target_path) noexcept {
  if (node.kind != vfile_virtual_path_kind::VDIR || !node.vsysfs) {
    return false;
  }

  bool root_uses_distinct_globs = true;
  if (auto root = node.vsysfs->root.lock()) {
    std::lock_guard<std::mutex> root_lock(root->mutex);
    root_uses_distinct_globs = (root->class_glob != root->devices_glob);
  }
  if (!root_uses_distinct_globs) {
    return false;
  }

  const std::string &class_dir = node.vsysfs->class_dir;
  const std::string &devices_dir = node.vsysfs->devices_dir;
  if (class_dir.empty() || devices_dir.empty() || class_dir == devices_dir) {
    return false;
  }
  if (node.path != class_dir) {
    return false;
  }

  if (target_path) {
    *target_path = devices_dir;
  }
  return true;
}

static inline bool
vsyscall_is_virtual_attr_symlink_name(const std::string &attr_name) noexcept {
  return attr_name == "subsystem" || attr_name == "device";
}

static inline bool
vsyscall_trim_virtual_symlink_target(std::string *target_path) noexcept {
  if (!target_path) {
    return false;
  }
  while (!target_path->empty() &&
         (target_path->back() == '\n' || target_path->back() == '\r')) {
    target_path->pop_back();
  }
  return !target_path->empty();
}

static inline bool vsyscall_try_get_virtual_attr_symlink_target(
    const vfile_virtual_path_info &node, std::string *target_path) noexcept {
  if (!target_path) {
    return false;
  }
  target_path->clear();

  if (node.kind == vfile_virtual_path_kind::VSYSFS_ATTR && node.vsysfs &&
      vsyscall_is_virtual_attr_symlink_name(node.attr_name)) {
    std::lock_guard<std::mutex> lock(node.vsysfs->mutex);
    auto it = node.vsysfs->attrs.find(node.attr_name);
    if (it == node.vsysfs->attrs.end()) {
      return false;
    }
    *target_path = it->second;
    return vsyscall_trim_virtual_symlink_target(target_path);
  }

  if (node.kind == vfile_virtual_path_kind::VMODULE_ATTR && node.vmodule &&
      node.attr_name == "subsystem") {
    std::lock_guard<std::mutex> lock(node.vmodule->mutex);
    auto it = node.vmodule->attrs.find(node.attr_name);
    if (it == node.vmodule->attrs.end()) {
      return false;
    }
    *target_path = it->second;
    return vsyscall_trim_virtual_symlink_target(target_path);
  }

  return false;
}

static inline bool vsyscall_try_get_entry_attr_symlink_target(
    const std::shared_ptr<vfile_fd_entry> &entry,
    std::string *target_path) noexcept {
  if (!entry || !target_path) {
    return false;
  }
  target_path->clear();

  if (entry->type == vfile_fd_entry::FD_VSYSFS && entry->vsysfs &&
      vsyscall_is_virtual_attr_symlink_name(entry->sysfs_attr_name)) {
    std::lock_guard<std::mutex> lock(entry->vsysfs->mutex);
    auto it = entry->vsysfs->attrs.find(entry->sysfs_attr_name);
    if (it == entry->vsysfs->attrs.end()) {
      return false;
    }
    *target_path = it->second;
    return vsyscall_trim_virtual_symlink_target(target_path);
  }

  if (entry->type == vfile_fd_entry::FD_VMODULE && entry->vmodule &&
      entry->sysfs_attr_name == "subsystem") {
    std::lock_guard<std::mutex> lock(entry->vmodule->mutex);
    auto it = entry->vmodule->attrs.find(entry->sysfs_attr_name);
    if (it == entry->vmodule->attrs.end()) {
      return false;
    }
    *target_path = it->second;
    return vsyscall_trim_virtual_symlink_target(target_path);
  }

  return false;
}

static inline bool vsyscall_build_symlink_stat(const std::string &target,
                                               struct stat *stbuf,
                                               int *stat_errno) noexcept {
  if (!stbuf || !stat_errno) {
    return false;
  }

  memset(stbuf, 0, sizeof(*stbuf));
  stbuf->st_uid = geteuid();
  stbuf->st_gid = getegid();
  stbuf->st_blksize = kVsyscallStatPreferredBlksize;
  stbuf->st_nlink = 1;
  stbuf->st_ino = static_cast<ino_t>(std::hash<std::string>{}(target));
  stbuf->st_mode = S_IFLNK | 0777;
  stbuf->st_size = static_cast<off_t>(target.size());
  stbuf->st_blocks = static_cast<blkcnt_t>(
      (target.size() + (kVsyscallStatBlockUnitBytes - 1U)) /
      kVsyscallStatBlockUnitBytes);
  *stat_errno = 0;
  return true;
}

static inline bool
vsyscall_stat_kind_from_fd_entry_type(vfile_fd_entry::fd_type type,
                                      vsyscall_stat_kind *kind,
                                      int *stat_errno) noexcept {
  if (!kind || !stat_errno) {
    if (stat_errno) {
      *stat_errno = EINVAL;
    }
    return false;
  }

  switch (type) {
  case vfile_fd_entry::FD_VDEV:
    *kind = vsyscall_stat_kind::CHR_DEVICE;
    *stat_errno = 0;
    return true;
  case vfile_fd_entry::FD_VDIR:
    *kind = vsyscall_stat_kind::DIR;
    *stat_errno = 0;
    return true;
  case vfile_fd_entry::FD_VSYSFS:
  case vfile_fd_entry::FD_VMODULE:
  case vfile_fd_entry::FD_PROC_FDINFO:
    *kind = vsyscall_stat_kind::REG_ATTR;
    *stat_errno = 0;
    return true;
  }

  *stat_errno = EBADF;
  return false;
}

static inline bool
vsyscall_stat_kind_from_virtual_node_kind(vfile_virtual_path_kind type,
                                          vsyscall_stat_kind *kind,
                                          int *stat_errno) noexcept {
  if (!kind || !stat_errno) {
    if (stat_errno) {
      *stat_errno = EINVAL;
    }
    return false;
  }

  switch (type) {
  case vfile_virtual_path_kind::NONE:
    *stat_errno = ENOENT;
    return false;
  case vfile_virtual_path_kind::VDEV_NODE:
    *kind = vsyscall_stat_kind::CHR_DEVICE;
    *stat_errno = 0;
    return true;
  case vfile_virtual_path_kind::VDIR:
    *kind = vsyscall_stat_kind::DIR;
    *stat_errno = 0;
    return true;
  case vfile_virtual_path_kind::VSYSFS_ATTR:
  case vfile_virtual_path_kind::VMODULE_ATTR:
    *kind = vsyscall_stat_kind::REG_ATTR;
    *stat_errno = 0;
    return true;
  }

  *stat_errno = ENOENT;
  return false;
}

static inline void vsyscall_init_stat_common(struct stat *stbuf,
                                             ino_t ino) noexcept {
  if (!stbuf) {
    return;
  }
  memset(stbuf, 0, sizeof(*stbuf));
  stbuf->st_uid = geteuid();
  stbuf->st_gid = getegid();
  stbuf->st_blksize = kVsyscallStatPreferredBlksize;
  stbuf->st_nlink = 1;
  stbuf->st_ino = ino;
}

static inline bool
vsyscall_apply_stat_kind(vsyscall_stat_kind kind, size_t payload_size,
                         pim_access_mode_t attr_access, unsigned int rdev_major,
                         unsigned int rdev_minor, struct stat *stbuf,
                         int *stat_errno) noexcept {
  if (!stbuf || !stat_errno) {
    return false;
  }

  switch (kind) {
  case vsyscall_stat_kind::CHR_DEVICE:
    stbuf->st_mode = S_IFCHR | kVsyscallVirtualChrNodeMode;
    stbuf->st_rdev = makedev(rdev_major, rdev_minor);
    *stat_errno = 0;
    return true;
  case vsyscall_stat_kind::DIR:
    stbuf->st_mode = S_IFDIR | kVsyscallVirtualDirNodeMode;
    stbuf->st_nlink = 2;
    *stat_errno = 0;
    return true;
  case vsyscall_stat_kind::REG_ATTR:
    stbuf->st_mode = S_IFREG | vsyscall_vsysfs_attr_perm_bits(attr_access);
    stbuf->st_size = static_cast<off_t>(payload_size);
    stbuf->st_blocks = static_cast<blkcnt_t>(
        (payload_size + (kVsyscallStatBlockUnitBytes - 1U)) /
        kVsyscallStatBlockUnitBytes);
    *stat_errno = 0;
    return true;
  }

  *stat_errno = EINVAL;
  return false;
}

static inline bool
vsyscall_build_vdev_stat(const std::shared_ptr<vfile_fd_entry> &entry,
                         struct stat *stbuf, int *stat_errno) noexcept {
  if (!entry || !stbuf || !stat_errno) {
    return false;
  }

#if defined(O_NOFOLLOW) && defined(O_PATH)
  if (entry->type == vfile_fd_entry::FD_VDIR &&
      (entry->flags & O_NOFOLLOW) != 0 && (entry->flags & O_PATH) != 0 &&
      !entry->virtual_path.empty()) {
    vfile_virtual_path_info node;
    int resolve_errno = 0;
    if (vfile_resolve_virtual_node(entry->virtual_path, &node,
                                   &resolve_errno)) {
      std::string target;
      if (vsyscall_is_vsysfs_class_symlink_node(node, &target)) {
        return vsyscall_build_symlink_stat(target, stbuf, stat_errno);
      }
    }
  }
  if ((entry->type == vfile_fd_entry::FD_VSYSFS ||
       entry->type == vfile_fd_entry::FD_VMODULE) &&
      (entry->flags & O_NOFOLLOW) != 0 && (entry->flags & O_PATH) != 0) {
    std::string target;
    if (vsyscall_try_get_entry_attr_symlink_target(entry, &target)) {
      return vsyscall_build_symlink_stat(target, stbuf, stat_errno);
    }
  }
#endif

  vsyscall_stat_kind kind = vsyscall_stat_kind::CHR_DEVICE;
  if (!vsyscall_stat_kind_from_fd_entry_type(entry->type, &kind, stat_errno)) {
    return false;
  }

  size_t payload_size = 0;
  pim_access_mode_t attr_access = PIM_ACCESS_MODE_RW;
  unsigned int rdev_major = kVsyscallVirtualChrRdevMajor;
  unsigned int rdev_minor = kVsyscallVirtualChrRdevMinor;
  if (entry->type == vfile_fd_entry::FD_VDEV) {
    const auto &vdev = entry->vdev;
    if (!vdev) {
      *stat_errno = EBADF;
      return false;
    }
    std::lock_guard<std::mutex> vdev_lock(vdev->mutex);
    rdev_major = vdev->major;
    rdev_minor = vdev->minor;
  }
  if (entry->type == vfile_fd_entry::FD_VSYSFS) {
    const auto &vsysfs = entry->vsysfs;
    if (!vsysfs) {
      *stat_errno = EBADF;
      return false;
    }
    {
      std::lock_guard<std::mutex> attrs_lock(vsysfs->mutex);
      auto it = vsysfs->attrs.find(entry->sysfs_attr_name);
      if (it == vsysfs->attrs.end()) {
        *stat_errno = ENOENT;
        return false;
      }
      payload_size = it->second.size();
      attr_access = vsyscall_vsysfs_get_attr_access_locked(
          vsysfs.get(), entry->sysfs_attr_name);
    }
  } else if (entry->type == vfile_fd_entry::FD_VMODULE) {
    const auto &vmodule = entry->vmodule;
    if (!vmodule) {
      *stat_errno = EBADF;
      return false;
    }
    {
      std::lock_guard<std::mutex> attrs_lock(vmodule->mutex);
      auto it = vmodule->attrs.find(entry->sysfs_attr_name);
      if (it == vmodule->attrs.end()) {
        *stat_errno = ENOENT;
        return false;
      }
      payload_size = it->second.size();
      attr_access = vsyscall_vmodule_get_attr_access_locked(
          vmodule.get(), entry->sysfs_attr_name);
    }
  } else if (entry->type == vfile_fd_entry::FD_PROC_FDINFO) {
    payload_size = entry->sysfs_cached_value.size();
    attr_access = entry->sysfs_cached_attr_access;
  }

  const ino_t inode =
      (entry->type == vfile_fd_entry::FD_VDIR)
          ? static_cast<ino_t>(std::hash<std::string>{}(entry->virtual_path))
          : reinterpret_cast<ino_t>(entry.get());
  vsyscall_init_stat_common(stbuf, inode);
  return vsyscall_apply_stat_kind(kind, payload_size, attr_access, rdev_major,
                                  rdev_minor, stbuf, stat_errno);
}

static inline bool
vsyscall_build_stat_from_virtual_path(const vfile_virtual_path_info &node,
                                      struct stat *stbuf,
                                      int *stat_errno) noexcept {
  if (!stbuf || !stat_errno) {
    return false;
  }

  vsyscall_init_stat_common(
      stbuf, static_cast<ino_t>(std::hash<std::string>{}(node.path)));

  vsyscall_stat_kind kind = vsyscall_stat_kind::CHR_DEVICE;
  if (!vsyscall_stat_kind_from_virtual_node_kind(node.kind, &kind,
                                                 stat_errno)) {
    return false;
  }
  const size_t payload_size =
      ((node.kind == vfile_virtual_path_kind::VSYSFS_ATTR) ||
       (node.kind == vfile_virtual_path_kind::VMODULE_ATTR))
          ? node.attr_size
          : 0;
  const pim_access_mode_t attr_access =
      ((node.kind == vfile_virtual_path_kind::VSYSFS_ATTR) ||
       (node.kind == vfile_virtual_path_kind::VMODULE_ATTR))
          ? node.attr_access
          : PIM_ACCESS_MODE_RW;
  unsigned int rdev_major = kVsyscallVirtualChrRdevMajor;
  unsigned int rdev_minor = kVsyscallVirtualChrRdevMinor;
  if (node.kind == vfile_virtual_path_kind::VDEV_NODE && node.vdev) {
    std::lock_guard<std::mutex> vdev_lock(node.vdev->mutex);
    rdev_major = node.vdev->major;
    rdev_minor = node.vdev->minor;
  }
  return vsyscall_apply_stat_kind(kind, payload_size, attr_access, rdev_major,
                                  rdev_minor, stbuf, stat_errno);
}

static inline long vsyscall_return_not_ours_or_fail(bool from_owned_dfd,
                                                    int *forward,
                                                    int owned_dfd_errno);
static inline vsyscall_dispatch_node_result
vsyscall_dispatch_resolve_node(int dfd, const char *pathname, int *forward,
                               int not_found_owned_errno);

static inline bool
vsyscall_path_is_under_sys(const std::string &path) noexcept {
  return (path == "/sys") || (path.rfind("/sys/", 0) == 0);
}

static inline bool vsyscall_is_sysfs_virtual_node_for_statfs(
    const vfile_virtual_path_info &node) noexcept {
  if (node.kind == vfile_virtual_path_kind::VSYSFS_ATTR ||
      node.kind == vfile_virtual_path_kind::VMODULE_ATTR) {
    return true;
  }
  if (node.kind == vfile_virtual_path_kind::VDIR &&
      vsyscall_path_is_under_sys(node.path)) {
    return true;
  }
  return false;
}

static inline bool vsyscall_is_sysfs_owned_entry_for_statfs(
    const std::shared_ptr<vfile_fd_entry> &entry) noexcept {
  if (!entry) {
    return false;
  }

  if (entry->type == vfile_fd_entry::FD_VSYSFS) {
    return true;
  }
  if (entry->type == vfile_fd_entry::FD_VMODULE) {
    return true;
  }
  if (entry->type == vfile_fd_entry::FD_VDIR &&
      vsyscall_path_is_under_sys(entry->virtual_path)) {
    return true;
  }
  return false;
}

template <typename StatfsT>
static inline void vsyscall_fill_statfs_sysfs(StatfsT *stfs) noexcept {
  if (!stfs) {
    return;
  }

  memset(stfs, 0, sizeof(*stfs));
  stfs->f_type = static_cast<decltype(stfs->f_type)>(SYSFS_MAGIC);
  stfs->f_bsize =
      static_cast<decltype(stfs->f_bsize)>(kVsyscallStatPreferredBlksize);
  stfs->f_namelen =
      static_cast<decltype(stfs->f_namelen)>(kVsyscallStatfsNameMax);
  stfs->f_frsize =
      static_cast<decltype(stfs->f_frsize)>(kVsyscallStatPreferredBlksize);
}

template <typename StatfsT>
static inline long vsyscall_dispatch_statfs_for_path(const char *pathname,
                                                     StatfsT *statfsbuf,
                                                     int *forward) {
  if (!pathname) {
    return 0; /* let kernel report invalid pointer */
  }

  const vsyscall_dispatch_node_result resolved =
      vsyscall_dispatch_resolve_node(AT_FDCWD, pathname, forward, ENOENT);
  if (resolved.status != vsyscall_dispatch_node_status::RESOLVED) {
    return resolved.ret;
  }

  if (!vsyscall_is_sysfs_virtual_node_for_statfs(resolved.node)) {
    return vsyscall_return_not_ours_or_fail(resolved.from_owned_dfd, forward,
                                            ENOENT);
  }

  vsyscall_consume_syscall(forward);
  if (!statfsbuf) {
    return vfile_fail_with_errno(EFAULT);
  }
  vsyscall_fill_statfs_sysfs(statfsbuf);
  return 0;
}

template <typename StatfsT>
inline long dispatch_fstatfs_typed(int fd, StatfsT *statfsbuf, int *forward) {
  auto entry = vsyscall_fd_acquire_for_op(fd);
  if (!entry) {
    return 0; /* not ours */
  }
  vsyscall_active_op_guard release_guard(entry);

  if (!vsyscall_is_sysfs_owned_entry_for_statfs(entry)) {
    return 0; /* forward to kernel for non-sysfs owned fd */
  }

  vsyscall_consume_syscall(forward);
  if (!statfsbuf) {
    return vfile_fail_with_errno(EFAULT);
  }
  vsyscall_fill_statfs_sysfs(statfsbuf);
  return 0;
}

#ifdef SYS_statfs64
static inline long dispatch_statfs64(const char *pathname, size_t buf_size,
                                     struct statfs64 *buf, int *forward) {
  if (buf_size != sizeof(struct statfs64)) {
    // Validate size only when path belongs to our virtual sysfs.
    const vsyscall_dispatch_node_result resolved =
        vsyscall_dispatch_resolve_node(AT_FDCWD, pathname, forward, ENOENT);
    if (resolved.status != vsyscall_dispatch_node_status::RESOLVED) {
      return resolved.ret;
    }
    if (!vsyscall_is_sysfs_virtual_node_for_statfs(resolved.node)) {
      return vsyscall_return_not_ours_or_fail(resolved.from_owned_dfd, forward,
                                              ENOENT);
    }
    return vsyscall_consume_and_fail(forward, EINVAL);
  }
  return vsyscall_dispatch_statfs_for_path(pathname, buf, forward);
}
#endif

static inline long vsyscall_return_not_ours_or_fail(bool from_owned_dfd,
                                                    int *forward,
                                                    int owned_dfd_errno) {
  if (from_owned_dfd) {
    return vsyscall_consume_and_fail(forward, owned_dfd_errno);
  }
  return 0; /* not ours */
}

enum class vsyscall_node_lookup_status { RESOLVED, NOT_FOUND, ERROR };

static inline vsyscall_node_lookup_status
vsyscall_lookup_virtual_node(const std::string &abs_path,
                             vfile_virtual_path_info *out_node,
                             int *resolve_errno) {
  if (!out_node || !resolve_errno) {
    if (resolve_errno) {
      *resolve_errno = EINVAL;
    }
    return vsyscall_node_lookup_status::ERROR;
  }
  *resolve_errno = 0;

  if (!vfile_resolve_virtual_node(abs_path, out_node, resolve_errno)) {
    if (*resolve_errno != 0) {
      return vsyscall_node_lookup_status::ERROR;
    }
    return vsyscall_node_lookup_status::NOT_FOUND;
  }
  return vsyscall_node_lookup_status::RESOLVED;
}

static inline vsyscall_dispatch_node_result
vsyscall_dispatch_lookup_node_from_resolved_path(
    const vsyscall_dispatch_resolved_path &path, int *forward,
    int not_found_owned_errno) {
  vsyscall_dispatch_node_result result;
  result.from_owned_dfd = path.from_owned_dfd;

  int resolve_errno = 0;
  vfile_virtual_path_info node;
  const vsyscall_node_lookup_status status =
      vsyscall_lookup_virtual_node(path.abs_path, &node, &resolve_errno);
  if (status == vsyscall_node_lookup_status::RESOLVED) {
    if (path.trailing_slash && node.kind != vfile_virtual_path_kind::VDIR) {
      result.ret = vsyscall_consume_and_fail(forward, ENOTDIR);
      return result;
    }
    result.status = vsyscall_dispatch_node_status::RESOLVED;
    result.node = std::move(node);
    return result;
  }
  if (status == vsyscall_node_lookup_status::ERROR) {
    result.ret = vsyscall_consume_and_fail(
        forward, (resolve_errno > 0) ? resolve_errno : EIO);
    return result;
  }

  result.ret = vsyscall_return_not_ours_or_fail(
      path.from_owned_dfd, forward,
      (not_found_owned_errno > 0) ? not_found_owned_errno : ENOENT);
  return result;
}

static inline vsyscall_dispatch_node_result
vsyscall_dispatch_resolve_node(int dfd, const char *pathname, int *forward,
                               int not_found_owned_errno) {
  vsyscall_dispatch_node_result result;
  const vsyscall_dispatch_path_result path_result =
      vsyscall_dispatch_resolve_path_for_dispatch(dfd, pathname, forward);
  result.from_owned_dfd = path_result.path.from_owned_dfd;
  if (path_result.status != vsyscall_dispatch_path_status::READY) {
    result.ret = path_result.ret;
    return result;
  }
  return vsyscall_dispatch_lookup_node_from_resolved_path(
      path_result.path, forward, not_found_owned_errno);
}

static inline long vsyscall_build_stat_for_owned_entry(
    const std::shared_ptr<vfile_fd_entry> &entry, struct stat *stbuf) {
  if (!stbuf) {
    return vfile_fail_with_errno(EFAULT);
  }

  int stat_errno = EIO;
  if (!vsyscall_build_vdev_stat(entry, stbuf, &stat_errno)) {
    return vfile_fail_with_errno(stat_errno);
  }
  return 0;
}

static inline vsyscall_stat_lookup_result
vsyscall_dispatch_lookup_stat(int dfd, const char *pathname, int flags,
                              int *forward) {
  vsyscall_stat_lookup_result result;
#ifdef AT_EMPTY_PATH
  const bool empty_path_request =
      ((flags & AT_EMPTY_PATH) != 0) &&
      ((pathname == nullptr) || (pathname[0] == '\0'));
  if (empty_path_request) {
    auto entry = vsyscall_acquire_owned_fd_for_dispatch(dfd, forward);
    if (!entry) {
      result.ret = 0; /* not ours */
      return result;
    }

    vsyscall_active_op_guard release_guard(entry);
    const long stat_ret =
        vsyscall_build_stat_for_owned_entry(entry, &result.st);
    if (stat_ret < 0) {
      result.ret = stat_ret;
      return result;
    }
    result.status = vsyscall_stat_lookup_status::READY;
    result.ret = 0;
    return result;
  }
#endif

  if (!pathname) {
    result.ret = 0;
    return result;
  }

  const vsyscall_dispatch_node_result resolved =
      vsyscall_dispatch_resolve_node(dfd, pathname, forward, ENOENT);
  if (resolved.status != vsyscall_dispatch_node_status::RESOLVED) {
    result.ret = resolved.ret;
    return result;
  }

  int stat_errno = EIO;

  std::string class_symlink_target;
  const bool class_symlink = vsyscall_is_vsysfs_class_symlink_node(
      resolved.node, &class_symlink_target);
  std::string attr_symlink_target;
  const bool attr_symlink = vsyscall_try_get_virtual_attr_symlink_target(
      resolved.node, &attr_symlink_target);
#ifdef AT_SYMLINK_NOFOLLOW
  const bool nofollow = ((flags & AT_SYMLINK_NOFOLLOW) != 0);
#else
  const bool nofollow = false;
#endif

  if (class_symlink && nofollow) {
    if (!vsyscall_build_symlink_stat(class_symlink_target, &result.st,
                                     &stat_errno)) {
      result.ret = vsyscall_consume_and_fail(
          forward, (stat_errno > 0) ? stat_errno : EIO);
      return result;
    }
  } else if (attr_symlink && nofollow) {
    if (!vsyscall_build_symlink_stat(attr_symlink_target, &result.st,
                                     &stat_errno)) {
      result.ret = vsyscall_consume_and_fail(
          forward, (stat_errno > 0) ? stat_errno : EIO);
      return result;
    }
  } else if (class_symlink) {
    vfile_virtual_path_info target_node;
    int resolve_errno = 0;
    if (!vfile_resolve_virtual_node(class_symlink_target, &target_node,
                                    &resolve_errno)) {
      result.ret = vsyscall_consume_and_fail(
          forward, (resolve_errno > 0) ? resolve_errno : ENOENT);
      return result;
    }
    if (!vsyscall_build_stat_from_virtual_path(target_node, &result.st,
                                               &stat_errno)) {
      result.ret = vsyscall_consume_and_fail(
          forward, (stat_errno > 0) ? stat_errno : EIO);
      return result;
    }
  } else if (attr_symlink) {
    vfile_virtual_path_info target_node;
    int resolve_errno = 0;
    if (!vfile_resolve_virtual_node(attr_symlink_target, &target_node,
                                    &resolve_errno)) {
      result.ret = vsyscall_consume_and_fail(
          forward, (resolve_errno > 0) ? resolve_errno : ENOENT);
      return result;
    }
    if (!vsyscall_build_stat_from_virtual_path(target_node, &result.st,
                                               &stat_errno)) {
      result.ret = vsyscall_consume_and_fail(
          forward, (stat_errno > 0) ? stat_errno : EIO);
      return result;
    }
  } else if (!vsyscall_build_stat_from_virtual_path(resolved.node, &result.st,
                                                    &stat_errno)) {
    result.ret =
        vsyscall_consume_and_fail(forward, (stat_errno > 0) ? stat_errno : EIO);
    return result;
  }

  vsyscall_consume_syscall(forward);
  result.status = vsyscall_stat_lookup_status::READY;
  result.ret = 0;
  return result;
}

template <typename SinkFn>
inline long vsyscall_dispatch_lookup_stat_and_store(int dfd,
                                                    const char *pathname,
                                                    int flags, int *forward,
                                                    SinkFn &&sink) {
  vsyscall_stat_lookup_result lookup =
      vsyscall_dispatch_lookup_stat(dfd, pathname, flags, forward);
  if (lookup.status != vsyscall_stat_lookup_status::READY) {
    return lookup.ret;
  }
  sink(lookup.st);
  return 0;
}

static inline int vsyscall_supported_at_path_flags() noexcept {
  int supported = 0;
#ifdef AT_SYMLINK_NOFOLLOW
  supported |= AT_SYMLINK_NOFOLLOW;
#endif
#ifdef AT_EMPTY_PATH
  supported |= AT_EMPTY_PATH;
#endif
  return supported;
}

static inline int vsyscall_supported_at_lookup_flags() noexcept {
  int supported = vsyscall_supported_at_path_flags();
#ifdef AT_NO_AUTOMOUNT
  supported |= AT_NO_AUTOMOUNT;
#endif
  return supported;
}

static inline int vsyscall_validate_known_flags(int flags,
                                                int supported) noexcept {
  if ((flags & ~supported) != 0) {
    return EINVAL;
  }
  return 0;
}

static inline int vsyscall_supported_open_flags() noexcept {
  int supported = O_ACCMODE;
#ifdef O_APPEND
  supported |= O_APPEND;
#endif
#ifdef O_ASYNC
  supported |= O_ASYNC;
#endif
#ifdef O_CLOEXEC
  supported |= O_CLOEXEC;
#endif
#ifdef O_CREAT
  supported |= O_CREAT;
#endif
#ifdef O_DIRECT
  supported |= O_DIRECT;
#endif
#ifdef O_DIRECTORY
  supported |= O_DIRECTORY;
#endif
#ifdef O_DSYNC
  supported |= O_DSYNC;
#endif
#ifdef O_EXCL
  supported |= O_EXCL;
#endif
#ifdef O_LARGEFILE
  supported |= O_LARGEFILE;
#endif
#ifdef O_NOATIME
  supported |= O_NOATIME;
#endif
#ifdef O_NOCTTY
  supported |= O_NOCTTY;
#endif
#ifdef O_NOFOLLOW
  supported |= O_NOFOLLOW;
#endif
#ifdef O_NONBLOCK
  supported |= O_NONBLOCK;
#endif
  supported |= O_PATH;
#ifdef O_RSYNC
  supported |= O_RSYNC;
#endif
#ifdef O_SYNC
  supported |= O_SYNC;
#endif
#ifdef O_TMPFILE
  supported |= O_TMPFILE;
#endif
#ifdef O_TRUNC
  supported |= O_TRUNC;
#endif
  return supported;
}

static inline int vsyscall_validate_open_flags(long flags_long) noexcept {
  const int flags = static_cast<int>(flags_long);
  if (static_cast<long>(flags) != flags_long) {
    return EINVAL;
  }

  const int flag_err =
      vsyscall_validate_known_flags(flags, vsyscall_supported_open_flags());
  if (flag_err != 0) {
    return flag_err;
  }

  if ((flags & O_PATH) != 0) {
    // Keep Linux-compatible O_PATH behavior: other open bits are accepted and
    // ignored by the kernel for path-only fds.
    return 0;
  }
  return 0;
}

static inline vsyscall_stat_lookup_result
vsyscall_dispatch_lookup_stat_validated(int dfd, const char *pathname,
                                        int flags, int validation_err,
                                        int *forward) {
  vsyscall_stat_lookup_result result;

  if (validation_err == 0) {
    return vsyscall_dispatch_lookup_stat(dfd, pathname, flags, forward);
  }

  // For owned relative-path lookups, keep Linux-style flag validation
  // precedence: invalid flags should return EINVAL before ENOTDIR/ENOENT.
  if (dfd != AT_FDCWD && vsyscall_fd_is_owned(dfd) &&
      (vsyscall_is_relative_path(pathname) ||
       (pathname && pathname[0] == '\0'))) {
    result.ret = vsyscall_consume_and_fail(forward, validation_err);
    return result;
  }

  const vsyscall_dispatch_path_result path_result =
      vsyscall_dispatch_resolve_path_for_dispatch(dfd, pathname, forward);
  if (path_result.status != vsyscall_dispatch_path_status::READY) {
    result.ret = path_result.ret;
    return result;
  }
  if (path_result.path.from_owned_dfd) {
    result.ret = vsyscall_consume_and_fail(forward, validation_err);
    return result;
  }

  const vsyscall_dispatch_node_result resolved =
      vsyscall_dispatch_lookup_node_from_resolved_path(path_result.path,
                                                       forward, ENOENT);
  if (resolved.status != vsyscall_dispatch_node_status::RESOLVED) {
    result.ret = resolved.ret;
    return result;
  }

  vsyscall_consume_syscall(forward);
  result.ret = vfile_fail_with_errno(validation_err);
  return result;
}

static inline long dispatch_newfstatat(int dfd, const char *pathname,
                                       struct stat *statbuf, int flags,
                                       int *forward) {
  const int flag_err = vsyscall_validate_known_flags(
      flags, vsyscall_supported_at_lookup_flags());
  const vsyscall_stat_lookup_result lookup =
      vsyscall_dispatch_lookup_stat_validated(dfd, pathname, flags, flag_err,
                                              forward);
  if (lookup.status != vsyscall_stat_lookup_status::READY) {
    return lookup.ret;
  }
  if (!statbuf) {
    return vfile_fail_with_errno(EFAULT);
  }
  *statbuf = lookup.st;
  return 0;
}

#ifdef SYS_statx
static inline unsigned int vsyscall_supported_statx_mask_bits() noexcept {
  unsigned int supported = 0U;
#ifdef STATX_TYPE
  supported |= STATX_TYPE;
#endif
#ifdef STATX_MODE
  supported |= STATX_MODE;
#endif
#ifdef STATX_NLINK
  supported |= STATX_NLINK;
#endif
#ifdef STATX_UID
  supported |= STATX_UID;
#endif
#ifdef STATX_GID
  supported |= STATX_GID;
#endif
#ifdef STATX_ATIME
  supported |= STATX_ATIME;
#endif
#ifdef STATX_MTIME
  supported |= STATX_MTIME;
#endif
#ifdef STATX_CTIME
  supported |= STATX_CTIME;
#endif
#ifdef STATX_INO
  supported |= STATX_INO;
#endif
#ifdef STATX_SIZE
  supported |= STATX_SIZE;
#endif
#ifdef STATX_BLOCKS
  supported |= STATX_BLOCKS;
#endif
#ifdef STATX_BASIC_STATS
  supported |= STATX_BASIC_STATS;
#endif
#ifdef STATX_BTIME
  supported |= STATX_BTIME;
#endif
#ifdef STATX_MNT_ID
  supported |= STATX_MNT_ID;
#endif
#ifdef STATX_DIOALIGN
  supported |= STATX_DIOALIGN;
#endif
#ifdef STATX_SUBVOL
  supported |= STATX_SUBVOL;
#endif
#ifdef STATX_WRITE_ATOMIC
  supported |= STATX_WRITE_ATOMIC;
#endif
  return supported;
}

static inline int vsyscall_validate_statx_mask(unsigned int mask) noexcept {
  if ((mask & ~vsyscall_supported_statx_mask_bits()) != 0U) {
    return EINVAL;
  }
  return 0;
}

static inline int vsyscall_validate_statx_flags(int flags) noexcept {
  int supported_flags = vsyscall_supported_at_lookup_flags();
#ifdef AT_STATX_SYNC_TYPE
  supported_flags |= AT_STATX_SYNC_TYPE;
#endif
  const int flags_err = vsyscall_validate_known_flags(flags, supported_flags);
  if (flags_err != 0) {
    return flags_err;
  }

#ifdef AT_STATX_SYNC_TYPE
  const int sync_flags = flags & AT_STATX_SYNC_TYPE;
  bool sync_ok = false;
#ifdef AT_STATX_SYNC_AS_STAT
  sync_ok = (sync_flags == AT_STATX_SYNC_AS_STAT);
#else
  sync_ok = (sync_flags == 0);
#endif
#ifdef AT_STATX_FORCE_SYNC
  sync_ok = sync_ok || (sync_flags == AT_STATX_FORCE_SYNC);
#endif
#ifdef AT_STATX_DONT_SYNC
  sync_ok = sync_ok || (sync_flags == AT_STATX_DONT_SYNC);
#endif
  if (!sync_ok) {
    return EINVAL;
  }
#endif
  return 0;
}

static inline void vsyscall_fill_statx_from_stat(const struct stat &st,
                                                 unsigned int requested_mask,
                                                 struct statx *stx) noexcept {
  if (!stx) {
    return;
  }

  memset(stx, 0, sizeof(*stx));
  stx->stx_mask = STATX_TYPE | STATX_MODE | STATX_NLINK | STATX_UID |
                  STATX_GID | STATX_INO | STATX_SIZE | STATX_BLOCKS;
#ifdef STATX_BLKSIZE
  stx->stx_mask |= STATX_BLKSIZE;
#endif
#ifdef STATX_RDEV_MAJOR
  stx->stx_mask |= STATX_RDEV_MAJOR;
#endif
#ifdef STATX_RDEV_MINOR
  stx->stx_mask |= STATX_RDEV_MINOR;
#endif
  if (requested_mask != 0U) {
    stx->stx_mask &= requested_mask;
  }
  stx->stx_blksize = static_cast<uint32_t>(st.st_blksize);
  stx->stx_nlink = st.st_nlink;
  stx->stx_uid = st.st_uid;
  stx->stx_gid = st.st_gid;
  stx->stx_mode = st.st_mode;
  stx->stx_ino = st.st_ino;
  stx->stx_size = st.st_size;
  stx->stx_blocks = st.st_blocks;
  stx->stx_rdev_major = major(st.st_rdev);
  stx->stx_rdev_minor = minor(st.st_rdev);
}

static inline long dispatch_statx(int dfd, const char *pathname, int flags,
                                  unsigned int mask, struct statx *statxbuf,
                                  int *forward) {
  const int flag_err = vsyscall_validate_statx_flags(flags);
  const int mask_err = vsyscall_validate_statx_mask(mask);
  const int validation_err = (flag_err != 0) ? flag_err : mask_err;
  const vsyscall_stat_lookup_result lookup =
      vsyscall_dispatch_lookup_stat_validated(dfd, pathname, flags,
                                              validation_err, forward);
  if (lookup.status != vsyscall_stat_lookup_status::READY) {
    return lookup.ret;
  }
  if (!statxbuf) {
    return vfile_fail_with_errno(EFAULT);
  }
  vsyscall_fill_statx_from_stat(lookup.st, mask, statxbuf);
  return 0;
}
#endif

static inline long vsyscall_finalize_entry_if_last_alias(
    const std::shared_ptr<vfile_fd_entry> &entry) {
  bool needs_finalize = false;
  {
    std::lock_guard<std::mutex> table_lock(g_fd_table_mutex);
    needs_finalize = !vsyscall_fd_has_alias_locked(entry);
  }

  if (!needs_finalize) {
    return 0;
  }

  {
    std::unique_lock<std::mutex> lock(entry->op_mutex);
    entry->closing = true;
    entry->op_cv.wait(lock, [&entry] { return entry->active_ops == 0; });
  }

  if (entry->type != vfile_fd_entry::FD_VDEV) {
    return 0;
  }

  auto vdev = entry->vdev;
  if (!vdev) {
    return vfile_fail_with_errno(EIO);
  }
  pim_vdev_close_fn close_fn = nullptr;
  {
    std::lock_guard<std::mutex> lock(vdev->mutex);
    close_fn = vdev->close_fn;
  }
  if (!close_fn) {
    return 0;
  }

  /*
   * close_fn parameters:
   *   - vsyscall_raw: owning virtual-device descriptor
   *   - owner_fd: canonical real fd for this virtual-device session
   */
  return vsyscall_wrapper_from_negative(close_fn(vdev.get(), entry->owner_fd),
                                        EIO);
}

static inline std::shared_ptr<vfile_fd_entry>
vsyscall_detach_owned_fd_entry(int fd) {
  {
    std::lock_guard<std::mutex> table_lock(g_fd_table_mutex);
    auto it = g_fd_table.find(fd);
    if (it == g_fd_table.end()) {
      return {};
    }
    std::shared_ptr<vfile_fd_entry> entry = it->second;
    g_fd_table.erase(it);
    return entry;
  }
}

static inline int vsyscall_close_kernel_fd_if_needed(int fd,
                                                     bool close_kernel_fd) {
  if (!close_kernel_fd) {
    return 0;
  }
  if (vsyscall_kernel_close_fd(fd) < 0) {
    return (errno > 0) ? errno : EIO;
  }
  return 0;
}

static inline long vsyscall_finalize_detached_owned_fd(
    const std::shared_ptr<vfile_fd_entry> &entry, int kernel_close_errno) {
  const long finalize_ret = vsyscall_finalize_entry_if_last_alias(entry);
  if (kernel_close_errno != 0) {
    return vfile_fail_with_errno(kernel_close_errno);
  }
  return finalize_ret;
}

static long close_owned_fd(int fd, bool close_kernel_fd) {
  const std::shared_ptr<vfile_fd_entry> entry =
      vsyscall_detach_owned_fd_entry(fd);
  if (!entry) {
    return 0; /* not ours */
  }

  const int close_errno =
      vsyscall_close_kernel_fd_if_needed(fd, close_kernel_fd);
  return vsyscall_finalize_detached_owned_fd(entry, close_errno);
}

static inline long vsyscall_close_owned_fd_list(const std::vector<int> &fds,
                                                bool close_kernel_fd) {
  for (int fd : fds) {
    long close_ret = close_owned_fd(fd, close_kernel_fd);
    if (close_ret < 0) {
      return close_ret;
    }
  }
  return 0;
}

static inline void vsyscall_wait_and_finalize_entry_best_effort(
    const std::shared_ptr<vfile_fd_entry> &entry) noexcept {
  if (!entry) {
    return;
  }

  {
    std::unique_lock<std::mutex> lock(entry->op_mutex);
    entry->closing = true;
    entry->op_cv.wait(lock, [&entry] { return entry->active_ops == 0; });
  }

  if (entry->type != vfile_fd_entry::FD_VDEV) {
    return;
  }

  auto vdev = entry->vdev;
  if (!vdev) {
    return;
  }

  pim_vdev_close_fn close_fn = nullptr;
  {
    std::lock_guard<std::mutex> lock(vdev->mutex);
    close_fn = vdev->close_fn;
  }
  if (!close_fn) {
    return;
  }

  (void)close_fn(vdev.get(), entry->owner_fd);
}

static inline void vsyscall_drain_owned_fd_batch_best_effort(
    std::unordered_map<int, std::shared_ptr<vfile_fd_entry>>
        detached) noexcept {
  if (detached.empty()) {
    return;
  }

  std::unordered_set<vfile_fd_entry *> finalized_entries;
  finalized_entries.reserve(detached.size());

  for (const auto &[fd, entry] : detached) {
    if (fd >= 0) {
      (void)vsyscall_kernel_close_fd(fd);
    }
    if (!entry) {
      continue;
    }
    if (finalized_entries.insert(entry.get()).second) {
      vsyscall_wait_and_finalize_entry_best_effort(entry);
    }
  }
}

static inline void vsyscall_close_all_owned_fds_best_effort() noexcept {
  while (true) {
    std::unordered_map<int, std::shared_ptr<vfile_fd_entry>> detached;
    {
      std::lock_guard<std::mutex> lock(g_fd_table_mutex);
      if (g_fd_table.empty()) {
        break;
      }
      detached.swap(g_fd_table);
    }
    vsyscall_drain_owned_fd_batch_best_effort(std::move(detached));
  }
}

static inline void vsyscall_clear_mmap_tracking() noexcept {
  std::lock_guard<std::mutex> lock(g_mmap_table_mutex);
  g_mmap_table.clear();
}

template <typename KernelStep, typename FinalizeStep>
static inline long vsyscall_dispatch_fd_mutation(bool should_intercept,
                                                 int *forward,
                                                 KernelStep &&kernel_step,
                                                 FinalizeStep &&finalize_step) {
  if (!should_intercept) {
    return 0; /* not ours */
  }
  vsyscall_consume_syscall(forward);
  const long kernel_ret = kernel_step();
  if (kernel_ret < 0) {
    return kernel_ret;
  }
  return finalize_step(kernel_ret);
}

static inline std::vector<int>
vsyscall_collect_owned_fds_in_range(unsigned int first, unsigned int last) {
  std::vector<int> owned_fds;
  std::lock_guard<std::mutex> table_lock(g_fd_table_mutex);
  owned_fds.reserve(g_fd_table.size());
  for (const auto &[fd, entry] : g_fd_table) {
    (void)entry;
    if (fd >= 0 && static_cast<unsigned int>(fd) >= first &&
        static_cast<unsigned int>(fd) <= last) {
      owned_fds.push_back(fd);
    }
  }
  return owned_fds;
}

struct vsyscall_dup2_prepare_result {
  std::shared_ptr<vfile_fd_entry> old_entry;
  bool should_intercept{false};
};

static inline vsyscall_dup2_prepare_result
vsyscall_prepare_dup2_context(int oldfd, int newfd) {
  vsyscall_dup2_prepare_result result;
  std::lock_guard<std::mutex> table_lock(g_fd_table_mutex);
  auto old_it = g_fd_table.find(oldfd);
  if (old_it != g_fd_table.end()) {
    result.old_entry = old_it->second;
  }
  const bool new_owned = (g_fd_table.find(newfd) != g_fd_table.end());
  result.should_intercept = static_cast<bool>(result.old_entry) || new_owned;
  return result;
}

static inline long vsyscall_rebind_fd_alias_after_dup(
    int target_fd, const std::shared_ptr<vfile_fd_entry> &new_entry,
    bool close_target_fd_on_failure) {
  std::shared_ptr<vfile_fd_entry> replaced_entry;
  const bool rebound = vfile_try_or_false_with_errno(
      [&]() {
        std::lock_guard<std::mutex> table_lock(g_fd_table_mutex);
        auto it = g_fd_table.find(target_fd);
        if (it != g_fd_table.end()) {
          replaced_entry = it->second;
        }
        if (new_entry) {
          g_fd_table[target_fd] = new_entry;
        } else if (it != g_fd_table.end()) {
          g_fd_table.erase(it);
        }
      },
      ENOMEM);
  if (!rebound) {
    return vsyscall_fail_enomem_with_optional_close(target_fd,
                                                    close_target_fd_on_failure);
  }

  if (replaced_entry && replaced_entry.get() != new_entry.get()) {
    long close_ret = vsyscall_finalize_entry_if_last_alias(replaced_entry);
    if (close_ret < 0) {
      if (close_target_fd_on_failure) {
        (void)vsyscall_kernel_close_fd(target_fd);
      }
      return close_ret;
    }
  }

  return static_cast<long>(target_fd);
}

static inline long
vsyscall_vsysfs_read_from_value_locked(vfile_fd_entry *entry,
                                       const std::string &value, void *buf,
                                       size_t count) noexcept {
  if (entry->sysfs_read_offset > value.size()) {
    entry->sysfs_read_offset = value.size();
  }

  const size_t remaining = value.size() - entry->sysfs_read_offset;
  if (remaining == 0) {
    return 0; /* EOF */
  }
  const size_t to_copy = (count < remaining) ? count : remaining;
  memcpy(buf, value.c_str() + entry->sysfs_read_offset, to_copy);
  entry->sysfs_read_offset += to_copy;
  return static_cast<long>(to_copy);
}

static inline int
vsyscall_attr_refresh_cache_locked(vfile_fd_entry *entry) noexcept {
  if (!entry) {
    return EBADF;
  }

  if (entry->type == vfile_fd_entry::FD_VSYSFS) {
    if (!entry->vsysfs) {
      return EBADF;
    }

    std::lock_guard<std::mutex> attrs_lock(entry->vsysfs->mutex);
    const uint64_t locked_revision =
        entry->vsysfs->attrs_revision.load(std::memory_order_relaxed);
    if (entry->sysfs_cached_attr_revision == locked_revision) {
      return 0;
    }

    auto it = entry->vsysfs->attrs.find(entry->sysfs_attr_name);
    if (it == entry->vsysfs->attrs.end()) {
      return ENOENT;
    }
    const pim_access_mode_t access = vsyscall_vsysfs_get_attr_access_locked(
        entry->vsysfs.get(), entry->sysfs_attr_name);
    try {
      entry->sysfs_cached_value = it->second;
    } catch (...) {
      return ENOMEM;
    }
    entry->sysfs_cached_attr_access = access;
    entry->sysfs_cached_attr_revision = locked_revision;
    if (entry->sysfs_read_offset > entry->sysfs_cached_value.size()) {
      entry->sysfs_read_offset = entry->sysfs_cached_value.size();
    }
    return 0;
  }

  if (entry->type == vfile_fd_entry::FD_VMODULE) {
    if (!entry->vmodule) {
      return EBADF;
    }

    std::lock_guard<std::mutex> attrs_lock(entry->vmodule->mutex);
    const uint64_t locked_revision =
        entry->vmodule->attrs_revision.load(std::memory_order_relaxed);
    if (entry->sysfs_cached_attr_revision == locked_revision) {
      return 0;
    }

    auto it = entry->vmodule->attrs.find(entry->sysfs_attr_name);
    if (it == entry->vmodule->attrs.end()) {
      return ENOENT;
    }
    const pim_access_mode_t access = vsyscall_vmodule_get_attr_access_locked(
        entry->vmodule.get(), entry->sysfs_attr_name);
    try {
      entry->sysfs_cached_value = it->second;
    } catch (...) {
      return ENOMEM;
    }
    entry->sysfs_cached_attr_access = access;
    entry->sysfs_cached_attr_revision = locked_revision;
    if (entry->sysfs_read_offset > entry->sysfs_cached_value.size()) {
      entry->sysfs_read_offset = entry->sysfs_cached_value.size();
    }
    return 0;
  }

  if (entry->type == vfile_fd_entry::FD_PROC_FDINFO) {
    return 0;
  }

  return EBADF;
}

static inline int vsyscall_vsysfs_truncate_attr_for_open(
    const vfile_virtual_path_info &node) noexcept {
  if (!node.vsysfs) {
    return EBADF;
  }
  bool size_changed = false;
  {
    std::lock_guard<std::mutex> attrs_lock(node.vsysfs->mutex);
    auto it = node.vsysfs->attrs.find(node.attr_name);
    if (it == node.vsysfs->attrs.end()) {
      return ENOENT;
    }
    const pim_access_mode_t access = vsyscall_vsysfs_get_attr_access_locked(
        node.vsysfs.get(), node.attr_name);
    if (!vsyscall_vsysfs_attr_access_allows_write(access)) {
      return EACCES;
    }
    size_changed = !it->second.empty();
    it->second.clear();
    (void)vfile_bump_nonzero_revision(node.vsysfs->attrs_revision);
  }
  if (size_changed) {
    vfile_notify_virtual_fs_changed();
  }
  return 0;
}

static inline int vsyscall_vmodule_truncate_attr_for_open(
    const vfile_virtual_path_info &node) noexcept {
  if (!node.vmodule) {
    return EBADF;
  }
  bool size_changed = false;
  {
    std::lock_guard<std::mutex> attrs_lock(node.vmodule->mutex);
    auto it = node.vmodule->attrs.find(node.attr_name);
    if (it == node.vmodule->attrs.end()) {
      return ENOENT;
    }
    const pim_access_mode_t access = vsyscall_vmodule_get_attr_access_locked(
        node.vmodule.get(), node.attr_name);
    if (!vsyscall_vsysfs_attr_access_allows_write(access)) {
      return EACCES;
    }
    size_changed = !it->second.empty();
    it->second.clear();
    (void)vfile_bump_nonzero_revision(node.vmodule->attrs_revision);
  }
  if (size_changed) {
    vfile_notify_virtual_fs_changed();
  }
  return 0;
}

static long dispatch_attr_read(const std::shared_ptr<vfile_fd_entry> &entry,
                               void *buf, size_t count) {
  if (!entry || (entry->type != vfile_fd_entry::FD_VSYSFS &&
                 entry->type != vfile_fd_entry::FD_VMODULE &&
                 entry->type != vfile_fd_entry::FD_PROC_FDINFO)) {
    return vfile_fail_with_errno(EBADF);
  }
  if (vsyscall_fd_opened_with_path_only(entry->flags)) {
    return vfile_fail_with_errno(EBADF);
  }
  if (!buf && count != 0U) {
    return vfile_fail_with_errno(EFAULT);
  }
  if (vsyscall_fd_access_mode(entry->flags) == O_WRONLY) {
    return vfile_fail_with_errno(EBADF);
  }

  long ret = 0;
  {
    std::lock_guard<std::mutex> fd_lock(entry->op_mutex);
    const int refresh_errno = vsyscall_attr_refresh_cache_locked(entry.get());
    if (refresh_errno != 0) {
      return vfile_fail_with_errno(refresh_errno);
    }
    if (!vsyscall_vsysfs_attr_access_allows_read(
            entry->sysfs_cached_attr_access)) {
      return vfile_fail_with_errno(EACCES);
    }
    ret = vsyscall_vsysfs_read_from_value_locked(
        entry.get(), entry->sysfs_cached_value, buf, count);
  }
  return ret;
}

static long dispatch_attr_write(const std::shared_ptr<vfile_fd_entry> &entry,
                                const void *buf, size_t count) {
  if (!entry || (entry->type != vfile_fd_entry::FD_VSYSFS &&
                 entry->type != vfile_fd_entry::FD_VMODULE &&
                 entry->type != vfile_fd_entry::FD_PROC_FDINFO)) {
    return vfile_fail_with_errno(EBADF);
  }
  if (vsyscall_fd_opened_with_path_only(entry->flags)) {
    return vfile_fail_with_errno(EBADF);
  }
  if (!buf && count != 0U) {
    return vfile_fail_with_errno(EFAULT);
  }
  if (!vsyscall_fd_can_write(entry->flags)) {
    return vfile_fail_with_errno(EBADF);
  }
  if (count == 0U) {
    return 0;
  }

  std::lock_guard<std::mutex> fd_lock(entry->op_mutex);

  const int refresh_errno = vsyscall_attr_refresh_cache_locked(entry.get());
  if (refresh_errno != 0) {
    return vfile_fail_with_errno(refresh_errno);
  }
  if (!vsyscall_vsysfs_attr_access_allows_write(
          entry->sysfs_cached_attr_access)) {
    return vfile_fail_with_errno(EACCES);
  }

  std::string updated;
  try {
    updated = entry->sysfs_cached_value;
  } catch (...) {
    return vfile_fail_with_errno(ENOMEM);
  }

  const bool append_mode = ((entry->flags & O_APPEND) != 0);
  const size_t offset = append_mode ? updated.size() : entry->sysfs_read_offset;
  if (offset > updated.size()) {
    try {
      updated.resize(offset, '\0');
    } catch (...) {
      return vfile_fail_with_errno(ENOMEM);
    }
  }

  const size_t end = offset + count;
  if (end < offset) {
    return vfile_fail_with_errno(EINVAL);
  }
  if (end > updated.size()) {
    try {
      updated.resize(end);
    } catch (...) {
      return vfile_fail_with_errno(ENOMEM);
    }
  }
  memcpy(updated.data() + offset, buf, count);

  bool size_changed = false;
  if (entry->type == vfile_fd_entry::FD_VSYSFS) {
    std::lock_guard<std::mutex> attrs_lock(entry->vsysfs->mutex);
    auto it = entry->vsysfs->attrs.find(entry->sysfs_attr_name);
    if (it == entry->vsysfs->attrs.end()) {
      return vfile_fail_with_errno(ENOENT);
    }
    const pim_access_mode_t access = vsyscall_vsysfs_get_attr_access_locked(
        entry->vsysfs.get(), entry->sysfs_attr_name);
    if (!vsyscall_vsysfs_attr_access_allows_write(access)) {
      return vfile_fail_with_errno(EACCES);
    }
    const size_t old_size = it->second.size();
    try {
      it->second = updated;
    } catch (...) {
      return vfile_fail_with_errno(ENOMEM);
    }
    size_changed = (old_size != it->second.size());

    const uint64_t next =
        vfile_bump_nonzero_revision(entry->vsysfs->attrs_revision);
    entry->sysfs_cached_attr_access = access;
    entry->sysfs_cached_attr_revision = next;
  } else if (entry->type == vfile_fd_entry::FD_VMODULE) {
    std::lock_guard<std::mutex> attrs_lock(entry->vmodule->mutex);
    auto it = entry->vmodule->attrs.find(entry->sysfs_attr_name);
    if (it == entry->vmodule->attrs.end()) {
      return vfile_fail_with_errno(ENOENT);
    }
    const pim_access_mode_t access = vsyscall_vmodule_get_attr_access_locked(
        entry->vmodule.get(), entry->sysfs_attr_name);
    if (!vsyscall_vsysfs_attr_access_allows_write(access)) {
      return vfile_fail_with_errno(EACCES);
    }
    const size_t old_size = it->second.size();
    try {
      it->second = updated;
    } catch (...) {
      return vfile_fail_with_errno(ENOMEM);
    }
    size_changed = (old_size != it->second.size());

    const uint64_t next =
        vfile_bump_nonzero_revision(entry->vmodule->attrs_revision);
    entry->sysfs_cached_attr_access = access;
    entry->sysfs_cached_attr_revision = next;
  } else {
    return vfile_fail_with_errno(EBADF);
  }

  entry->sysfs_cached_value = std::move(updated);
  entry->sysfs_read_offset = end;
  if (size_changed) {
    vfile_notify_virtual_fs_changed();
  }
  return static_cast<long>(count);
}

static inline long
vsyscall_seek_owned_entry_cursor(const std::shared_ptr<vfile_fd_entry> &entry,
                                 size_t vfile_fd_entry::*cursor,
                                 int badfd_errno, off_t offset, int whence) {
  if (!entry || !cursor) {
    return vfile_fail_with_errno(badfd_errno);
  }

  long ret = -1;
  int seek_errno = EINVAL;
  {
    std::lock_guard<std::mutex> fd_lock(entry->op_mutex);
    seek_errno =
        vfile_seek_cursor_locked(&(entry.get()->*cursor), offset, whence, &ret);
  }

  if (seek_errno != 0) {
    return vfile_fail_with_errno(seek_errno);
  }
  return ret;
}

static long dispatch_lseek(int fd, off_t offset, int whence, int *forward) {
  return vsyscall_dispatch_owned_fd(
      fd, forward,
      [offset, whence](const std::shared_ptr<vfile_fd_entry> &entry) {
        if (vsyscall_fd_opened_with_path_only(entry->flags)) {
          return vfile_fail_with_errno(EBADF);
        }
        if (entry->type == vfile_fd_entry::FD_VSYSFS ||
            entry->type == vfile_fd_entry::FD_VMODULE ||
            entry->type == vfile_fd_entry::FD_PROC_FDINFO) {
          return vsyscall_seek_owned_entry_cursor(
              entry, &vfile_fd_entry::sysfs_read_offset, EBADF, offset, whence);
        }
        if (entry->type == vfile_fd_entry::FD_VDIR) {
          return vdir_seek_entry_cursor(entry, offset, whence);
        }
        return vfile_fail_with_errno(ESPIPE); /* vdev fds are not seekable */
      });
}

static inline long vsyscall_fail_virtual_readlink_unsupported(int *forward) {
  // Virtual nodes currently model dirs, attrs, and character devices.
  // Only explicit symlink-like virtual nodes are handled in readlink(2).
  vsyscall_consume_syscall(forward);
  return vfile_fail_with_errno(EINVAL);
}

static inline void
vsyscall_trim_readlink_target_in_place(std::string *target) noexcept {
  if (!target) {
    return;
  }
  while (!target->empty()) {
    const char c = target->back();
    if (c == '\n' || c == '\r' || c == '\0') {
      target->pop_back();
    } else {
      break;
    }
  }
}

static inline std::vector<std::string>
vsyscall_split_absolute_path_components(const std::string &abs_path) {
  std::vector<std::string> parts;
  size_t i = 0;
  const size_t n = abs_path.size();
  while (i < n) {
    while (i < n && abs_path[i] == '/') {
      ++i;
    }
    if (i >= n) {
      break;
    }
    size_t j = i;
    while (j < n && abs_path[j] != '/') {
      ++j;
    }
    parts.push_back(abs_path.substr(i, j - i));
    i = j;
  }
  return parts;
}

static inline bool
vsyscall_make_readlink_relative_target(const std::string &link_path,
                                       const std::string &target_abs,
                                       std::string *relative_target) {
  if (!relative_target) {
    return false;
  }

  std::string link_norm;
  std::string target_norm;
  if (!vfile_normalize_absolute_path(link_path, &link_norm) ||
      !vfile_normalize_absolute_path(target_abs, &target_norm)) {
    return false;
  }

  std::string link_parent = "/";
  const size_t slash = link_norm.rfind('/');
  if (slash != std::string::npos) {
    link_parent = (slash == 0) ? "/" : link_norm.substr(0, slash);
  }

  const std::vector<std::string> from_parts =
      vsyscall_split_absolute_path_components(link_parent);
  const std::vector<std::string> to_parts =
      vsyscall_split_absolute_path_components(target_norm);

  size_t common = 0;
  while (common < from_parts.size() && common < to_parts.size() &&
         from_parts[common] == to_parts[common]) {
    ++common;
  }

  std::string rel;
  for (size_t i = common; i < from_parts.size(); ++i) {
    if (!rel.empty()) {
      rel.push_back('/');
    }
    rel.append("..");
  }
  for (size_t i = common; i < to_parts.size(); ++i) {
    if (!rel.empty()) {
      rel.push_back('/');
    }
    rel.append(to_parts[i]);
  }
  if (rel.empty()) {
    rel = ".";
  }

  *relative_target = std::move(rel);
  return true;
}

static inline long vsyscall_return_readlink_target(const std::string &target,
                                                   char *buf, size_t bufsiz,
                                                   int *forward) {
  vsyscall_consume_syscall(forward);

  if (!buf) {
    return vfile_fail_with_errno(EFAULT);
  }
  if (bufsiz == 0) {
    return vfile_fail_with_errno(EINVAL);
  }

  const size_t n = std::min(target.size(), bufsiz);
  if (n > 0) {
    memcpy(buf, target.data(), n);
  }
  return static_cast<long>(n);
}

static inline bool
vsyscall_try_get_virtual_readlink_target(const vfile_virtual_path_info &node,
                                         std::string *target) {
  if (!target) {
    return false;
  }
  target->clear();

  if (node.kind == vfile_virtual_path_kind::VSYSFS_ATTR && node.vsysfs &&
      (node.attr_name == "subsystem" || node.attr_name == "device")) {
    std::lock_guard<std::mutex> lock(node.vsysfs->mutex);
    auto it = node.vsysfs->attrs.find(node.attr_name);
    if (it == node.vsysfs->attrs.end()) {
      return false;
    }
    *target = it->second;
    vsyscall_trim_readlink_target_in_place(target);
    return !target->empty();
  }

  if (node.kind == vfile_virtual_path_kind::VMODULE_ATTR && node.vmodule &&
      node.attr_name == "subsystem") {
    std::lock_guard<std::mutex> lock(node.vmodule->mutex);
    auto it = node.vmodule->attrs.find(node.attr_name);
    if (it == node.vmodule->attrs.end()) {
      return false;
    }
    *target = it->second;
    vsyscall_trim_readlink_target_in_place(target);
    return !target->empty();
  }

  if (node.kind == vfile_virtual_path_kind::VDIR &&
      node.path == "/sys/class/subsystem") {
    *target = "/sys/class";
    return true;
  }

  return false;
}

static inline bool
vsyscall_extract_virtual_class_subsystem_name(const std::string &abs_path,
                                              std::string *class_name) {
  if (!class_name) {
    return false;
  }

  constexpr const char *kClassPrefix = "/sys/class/";
  constexpr const char *kSubsystemSuffix = "/subsystem";

  const std::string prefix(kClassPrefix);
  const std::string suffix(kSubsystemSuffix);
  if (abs_path == "/sys/class/subsystem") {
    return false;
  }
  if (abs_path.size() <= prefix.size() + suffix.size()) {
    return false;
  }
  if (abs_path.compare(0, prefix.size(), prefix) != 0) {
    return false;
  }
  if (abs_path.compare(abs_path.size() - suffix.size(), suffix.size(),
                       suffix) != 0) {
    return false;
  }

  const size_t class_begin = prefix.size();
  const size_t class_end = abs_path.size() - suffix.size();
  const std::string extracted =
      abs_path.substr(class_begin, class_end - class_begin);
  if (extracted.empty() || extracted.find('/') != std::string::npos) {
    return false;
  }

  *class_name = extracted;
  return true;
}

static inline bool
vsyscall_virtual_class_exists(const std::string &class_name) {
  const auto roots = vfile_copy_vsysfs_root_list_for_snapshot();
  for (const auto &root : roots) {
    if (!root) {
      continue;
    }

    std::lock_guard<std::mutex> lock(root->mutex);
    if (root->class_name == class_name) {
      return true;
    }
  }

  return false;
}

static inline bool
vsyscall_try_get_virtual_class_subsystem_target(const std::string &abs_path,
                                                std::string *target) {
  if (!target) {
    return false;
  }

  std::string class_name;
  if (!vsyscall_extract_virtual_class_subsystem_name(abs_path, &class_name)) {
    return false;
  }
  if (!vsyscall_virtual_class_exists(class_name)) {
    return false;
  }

  *target = std::string("/sys/subsystem/") + class_name;
  return true;
}

static long dispatch_readlinkat(int dfd, const char *pathname, char *buf,
                                size_t bufsiz, int *forward) {
  if (!pathname) {
    return 0; /* forward to kernel */
  }

  const vsyscall_dispatch_path_result path_result =
      vsyscall_dispatch_resolve_path_for_dispatch(dfd, pathname, forward);
  if (path_result.status != vsyscall_dispatch_path_status::READY) {
    return path_result.ret;
  }

  vsyscall_dispatch_resolved_path resolved_path = path_result.path;

  /*
   * Mirror host sysfs convenience symlink. We keep this virtual so udev
   * traversal does not fail when it probes /sys/class/subsystem.
   */
  if (resolved_path.abs_path == "/sys/class/subsystem") {
    return vsyscall_return_readlink_target("/sys/class", buf, bufsiz, forward);
  }
  std::string class_subsystem_target;
  if (vsyscall_try_get_virtual_class_subsystem_target(
          resolved_path.abs_path, &class_subsystem_target)) {
    return vsyscall_return_readlink_target(class_subsystem_target, buf, bufsiz,
                                           forward);
  }

  const vsyscall_dispatch_node_result resolved =
      vsyscall_dispatch_lookup_node_from_resolved_path(resolved_path, forward,
                                                       ENOENT);
  if (resolved.status != vsyscall_dispatch_node_status::RESOLVED) {
    return resolved.ret;
  }

  std::string target;
  if (vsyscall_is_vsysfs_class_symlink_node(resolved.node, &target)) {
    std::string relative_target;
    if (vsyscall_make_readlink_relative_target(resolved.node.path, target,
                                               &relative_target)) {
      target = std::move(relative_target);
    }
    return vsyscall_return_readlink_target(target, buf, bufsiz, forward);
  }
  if (vsyscall_try_get_virtual_readlink_target(resolved.node, &target)) {
    return vsyscall_return_readlink_target(target, buf, bufsiz, forward);
  }

  return vsyscall_fail_virtual_readlink_unsupported(forward);
}

static inline bool vsyscall_group_list_contains(gid_t target_gid,
                                                const gid_t *groups,
                                                int count) noexcept {
  if (!groups || count <= 0) {
    return false;
  }
  for (int i = 0; i < count; ++i) {
    if (groups[i] == target_gid) {
      return true;
    }
  }
  return false;
}

static inline bool vsyscall_process_in_group(gid_t target_gid,
                                             bool use_effective_ids) noexcept {
  const gid_t primary_gid = use_effective_ids ? getegid() : getgid();
  if (target_gid == primary_gid) {
    return true;
  }

  bool saw_probe_einval = false;
  int last_einval_group_count = -1;
  bool should_retry = true;
  while (should_retry) {
    should_retry = false;
    const int group_count = getgroups(0, nullptr);
    if (group_count < 0) {
      if (errno == EINVAL) {
        // getgroups(0) should not normally return EINVAL. Retry only once.
        if (!saw_probe_einval) {
          saw_probe_einval = true;
          should_retry = true;
        }
        continue;
      }
      return false;
    }
    saw_probe_einval = false;
    if (group_count == 0) {
      return false;
    }

    const size_t groups_size = static_cast<size_t>(group_count);
    if (groups_size > (std::numeric_limits<size_t>::max() / sizeof(gid_t))) {
      return false;
    }
    std::unique_ptr<gid_t[]> groups(new (std::nothrow) gid_t[groups_size]);
    if (!groups) {
      return false;
    }

    const int fetched = getgroups(group_count, groups.get());
    if (fetched >= 0) {
      return vsyscall_group_list_contains(target_gid, groups.get(), fetched);
    }

    const int getgroups_errno = errno;
    if (getgroups_errno != EINVAL) {
      return false;
    }

    // Retry only while the sampled group-count is growing. If not, we are not
    // making progress and should fail closed instead of spinning indefinitely.
    if (group_count <= last_einval_group_count) {
      return false;
    }
    if (group_count >= kMaxGroups) {
      return false;
    }
    last_einval_group_count = group_count;
    should_retry = true;
  }
  return false;
}

static inline int vsyscall_check_access_mode(const struct stat &stbuf, int mode,
                                             bool use_effective_ids) noexcept {
  if ((mode & ~(R_OK | W_OK | X_OK | F_OK)) != 0) {
    return EINVAL;
  }
  if (mode == F_OK) {
    return 0;
  }

  const uid_t subject_uid = use_effective_ids ? geteuid() : getuid();
  if (subject_uid == 0) {
    if (((mode & X_OK) != 0) &&
        ((stbuf.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) == 0)) {
      return EACCES;
    }
    return 0;
  }

  unsigned int granted_bits = static_cast<unsigned int>(stbuf.st_mode) & 0x7U;
  if (subject_uid == stbuf.st_uid) {
    granted_bits = (static_cast<unsigned int>(stbuf.st_mode) >> 6U) & 0x7U;
  } else if (vsyscall_process_in_group(stbuf.st_gid, use_effective_ids)) {
    granted_bits = (static_cast<unsigned int>(stbuf.st_mode) >> 3U) & 0x7U;
  }

  if (((mode & R_OK) != 0) && ((granted_bits & 0x4U) == 0U)) {
    return EACCES;
  }
  if (((mode & W_OK) != 0) && ((granted_bits & 0x2U) == 0U)) {
    return EACCES;
  }
  if (((mode & X_OK) != 0) && ((granted_bits & 0x1U) == 0U)) {
    return EACCES;
  }
  return 0;
}

static long dispatch_faccessat_common(int dfd, const char *pathname, int mode,
                                      int flags, int *forward) {
  int supported_flags = vsyscall_supported_at_path_flags();
#ifdef AT_EACCESS
  supported_flags |= AT_EACCESS;
#endif
  const int flag_err = vsyscall_validate_known_flags(flags, supported_flags);
  if (flag_err != 0) {
    if (dfd != AT_FDCWD && vsyscall_fd_is_owned(dfd) &&
        (vsyscall_is_relative_path(pathname) ||
         (pathname && pathname[0] == '\0'))) {
      return vsyscall_consume_and_fail(forward, flag_err);
    }

    // For non-owned targets, forward unknown/new flags to the kernel.
    // For owned targets, keep kernel-compatible EINVAL behavior.
#ifdef AT_EMPTY_PATH
    const bool empty_path_request =
        ((flags & AT_EMPTY_PATH) != 0) &&
        ((pathname == nullptr) || (pathname[0] == '\0'));
    if (empty_path_request) {
      auto entry = vsyscall_acquire_owned_fd_for_dispatch(dfd, forward);
      if (!entry) {
        return 0; /* not ours */
      }
      vsyscall_active_op_guard release_guard(entry);
      return vfile_fail_with_errno(flag_err);
    }
#endif

    if (!pathname) {
      return 0; /* forward to kernel */
    }

    const vsyscall_dispatch_resolved_path resolved =
        vsyscall_dispatch_resolve_path(dfd, pathname);
    if (resolved.status == vsyscall_path_resolve_status::ERROR) {
      return vsyscall_consume_and_fail(forward, (resolved.resolve_errno > 0)
                                                    ? resolved.resolve_errno
                                                    : EINVAL);
    }
    if (resolved.status != vsyscall_path_resolve_status::RESOLVED) {
      return 0; /* not ours */
    }
    if (resolved.from_owned_dfd) {
      return vsyscall_consume_and_fail(forward, flag_err);
    }

    vfile_virtual_path_info node;
    int resolve_errno = 0;
    if (vfile_resolve_virtual_node(resolved.abs_path, &node, &resolve_errno)) {
      return vsyscall_consume_and_fail(forward, flag_err);
    }
    if (resolve_errno != 0) {
      return vsyscall_consume_and_fail(forward, resolve_errno);
    }
    return 0; /* not ours */
  }

  bool use_effective_ids = false;
#ifdef AT_EACCESS
  use_effective_ids = ((flags & AT_EACCESS) != 0);
#endif

#ifdef AT_EMPTY_PATH
  const bool empty_path_request =
      ((flags & AT_EMPTY_PATH) != 0) &&
      ((pathname == nullptr) || (pathname[0] == '\0'));
  if (empty_path_request) {
    auto entry = vsyscall_acquire_owned_fd_for_dispatch(dfd, forward);
    if (!entry) {
      return 0; /* not ours */
    }
    vsyscall_active_op_guard release_guard(entry);

    struct stat stbuf = {};
    const long stat_ret = vsyscall_build_stat_for_owned_entry(entry, &stbuf);
    if (stat_ret < 0) {
      return stat_ret;
    }
    const int access_errno =
        vsyscall_check_access_mode(stbuf, mode, use_effective_ids);
    if (access_errno != 0) {
      return vfile_fail_with_errno(access_errno);
    }
    return 0;
  }
#endif

  if (!pathname) {
    return 0; /* forward to kernel */
  }

  int access_errno = 0;
  const long lookup_ret = vsyscall_dispatch_lookup_stat_and_store(
      dfd, pathname, flags, forward,
      [&access_errno, mode, use_effective_ids](const struct stat &st) {
        access_errno = vsyscall_check_access_mode(st, mode, use_effective_ids);
      });
  if (lookup_ret != 0) {
    return lookup_ret;
  }
  if (access_errno != 0) {
    return vfile_fail_with_errno(access_errno);
  }
  return 0;
}

/*============================================================================
 * Syscall Dispatch
 *============================================================================*/

static inline void vsyscall_cleanup_vdev_after_open_bind_failure(
    const std::shared_ptr<pim_vdev_t> &vdev, int owner_fd) noexcept {
  if (!vdev || owner_fd < 0) {
    return;
  }
  const pim_vdev_close_fn close_fn =
      vsyscall_get_callback(vdev, &pim_vdev_t::close_fn);
  if (close_fn == nullptr) {
    return;
  }
  vsyscall_run_best_effort([&]() { (void)close_fn(vdev.get(), owner_fd); });
}

static long
vsyscall_dispatch_open_vdev_node(const vfile_virtual_path_info &node,
                                 long flags, int *forward) {
  if ((flags & O_DIRECTORY) != 0) {
    return vsyscall_consume_and_fail(forward, ENOTDIR);
  }

  const char *effective_path = node.path.c_str();
  std::shared_ptr<pim_vdev_t> vdev = node.vdev;
  pim_vdev_open_fn open_fn = nullptr;
  vsyscall_get_open_config(vdev, &open_fn);

  if (!open_fn) {
    /* No open callback: reserve a real fd for vdev session tracking. */
    return vsyscall_alloc_tracked_fd(
        vsyscall_make_device_entry(vdev, -1, effective_path,
                                   static_cast<int>(flags)),
        forward);
  }

  /*
   * open_fn parameters:
   *   - vdev.get(): matched virtual-device descriptor
   *   - pathname: resolved canonical path of this virtual node
   *   - static_cast<int>(flags): open flags (O_* bitmask)
   */
  int fd = open_fn(vdev.get(), effective_path, static_cast<int>(flags));
  if (fd < 0) {
    vsyscall_consume_syscall(forward);
    return vsyscall_wrapper_from_negative(fd, EIO);
  }

  auto entry = vsyscall_make_device_entry(vdev, fd, effective_path,
                                          static_cast<int>(flags));
  if (!vsyscall_bind_existing_fd_or_close(fd, entry, EBUSY)) {
    const int bind_errno = (errno > 0) ? errno : EBUSY;
    vsyscall_cleanup_vdev_after_open_bind_failure(vdev, fd);
    errno = bind_errno;
    return vsyscall_consume_and_fail(forward, errno);
  }
  vsyscall_consume_syscall(forward);
  return fd;
}

static inline long
vsyscall_dispatch_open_vsysfs_attr_node(const vfile_virtual_path_info &node,
                                        long flags, int *forward) {
  if ((flags & O_DIRECTORY) != 0) {
    return vsyscall_consume_and_fail(forward, ENOTDIR);
  }
  const bool path_only = ((flags & O_PATH) != 0);
  if (path_only) {
    return vsyscall_alloc_tracked_fd(
        vsyscall_make_vsysfs_entry(node.vsysfs, node.attr_name, node.path,
                                   static_cast<int>(flags)),
        forward);
  }
  const int access_mode = vsyscall_fd_access_mode(static_cast<int>(flags));
  if (access_mode != O_RDONLY && access_mode != O_WRONLY &&
      access_mode != O_RDWR) {
    return vsyscall_consume_and_fail(forward, EINVAL);
  }
  const bool wants_read = (access_mode == O_RDONLY || access_mode == O_RDWR);
  const bool wants_write = (access_mode == O_WRONLY || access_mode == O_RDWR);
  if (wants_read &&
      !vsyscall_vsysfs_attr_access_allows_read(node.attr_access)) {
    return vsyscall_consume_and_fail(forward, EACCES);
  }
  if (wants_write &&
      !vsyscall_vsysfs_attr_access_allows_write(node.attr_access)) {
    return vsyscall_consume_and_fail(forward, EACCES);
  }
  if ((flags & O_TRUNC) != 0 &&
      !vsyscall_fd_can_write(static_cast<int>(flags))) {
    return vsyscall_consume_and_fail(forward, EACCES);
  }
  if ((flags & O_TRUNC) != 0) {
    auto entry = vsyscall_make_vsysfs_entry(node.vsysfs, node.attr_name,
                                            node.path, static_cast<int>(flags));
    const long fd = vsyscall_alloc_tracked_fd(entry, forward);
    if (fd < 0) {
      return fd;
    }
    const int trunc_errno = vsyscall_vsysfs_truncate_attr_for_open(node);
    if (trunc_errno != 0) {
      vsyscall_fd_free(static_cast<int>(fd));
      (void)vsyscall_kernel_close_fd(static_cast<int>(fd));
      return vfile_fail_with_errno(trunc_errno);
    }
    return fd;
  }
  return vsyscall_alloc_tracked_fd(
      vsyscall_make_vsysfs_entry(node.vsysfs, node.attr_name, node.path,
                                 static_cast<int>(flags)),
      forward);
}

static inline long
vsyscall_dispatch_open_vmodule_attr_node(const vfile_virtual_path_info &node,
                                         long flags, int *forward) {
  if ((flags & O_DIRECTORY) != 0) {
    return vsyscall_consume_and_fail(forward, ENOTDIR);
  }
  const bool path_only = ((flags & O_PATH) != 0);
  if (path_only) {
    return vsyscall_alloc_tracked_fd(
        vsyscall_make_vmodule_entry(node.vmodule, node.attr_name, node.path,
                                    static_cast<int>(flags)),
        forward);
  }
  const int access_mode = vsyscall_fd_access_mode(static_cast<int>(flags));
  if (access_mode != O_RDONLY && access_mode != O_WRONLY &&
      access_mode != O_RDWR) {
    return vsyscall_consume_and_fail(forward, EINVAL);
  }
  const bool wants_read = (access_mode == O_RDONLY || access_mode == O_RDWR);
  const bool wants_write = (access_mode == O_WRONLY || access_mode == O_RDWR);
  if (wants_read &&
      !vsyscall_vsysfs_attr_access_allows_read(node.attr_access)) {
    return vsyscall_consume_and_fail(forward, EACCES);
  }
  if (wants_write &&
      !vsyscall_vsysfs_attr_access_allows_write(node.attr_access)) {
    return vsyscall_consume_and_fail(forward, EACCES);
  }
  if ((flags & O_TRUNC) != 0 &&
      !vsyscall_fd_can_write(static_cast<int>(flags))) {
    return vsyscall_consume_and_fail(forward, EACCES);
  }
  if ((flags & O_TRUNC) != 0) {
    auto entry = vsyscall_make_vmodule_entry(
        node.vmodule, node.attr_name, node.path, static_cast<int>(flags));
    const long fd = vsyscall_alloc_tracked_fd(entry, forward);
    if (fd < 0) {
      return fd;
    }
    const int trunc_errno = vsyscall_vmodule_truncate_attr_for_open(node);
    if (trunc_errno != 0) {
      vsyscall_fd_free(static_cast<int>(fd));
      (void)vsyscall_kernel_close_fd(static_cast<int>(fd));
      return vfile_fail_with_errno(trunc_errno);
    }
    return fd;
  }
  return vsyscall_alloc_tracked_fd(
      vsyscall_make_vmodule_entry(node.vmodule, node.attr_name, node.path,
                                  static_cast<int>(flags)),
      forward);
}

static inline long
vsyscall_dispatch_open_dir_node(const vfile_virtual_path_info &node, long flags,
                                int *forward) {
  const bool path_only = ((flags & O_PATH) != 0);
  if (path_only) {
    return vsyscall_alloc_tracked_fd(
        vdir_make_fd_entry(node.path, static_cast<int>(flags)), forward);
  }
  const int access_mode = static_cast<int>(flags) & O_ACCMODE;
  if (access_mode != O_RDONLY || (flags & (O_CREAT | O_TRUNC)) != 0) {
    return vsyscall_consume_and_fail(forward, EISDIR);
  }
  return vsyscall_alloc_tracked_fd(
      vdir_make_fd_entry(node.path, static_cast<int>(flags)), forward);
}

static inline long
vsyscall_dispatch_open_proc_fdinfo_node(int target_fd, const std::string &path,
                                        long flags, int *forward) {
  if ((flags & O_DIRECTORY) != 0) {
    return vsyscall_consume_and_fail(forward, ENOTDIR);
  }

  const bool path_only = ((flags & O_PATH) != 0);
  if (!path_only) {
    const int access_mode = vsyscall_fd_access_mode(static_cast<int>(flags));
    if (access_mode != O_RDONLY) {
      return vsyscall_consume_and_fail(forward, EACCES);
    }
  }

  const std::shared_ptr<vfile_fd_entry> target_entry =
      vsyscall_fd_lookup(target_fd);
  if (!target_entry) {
    return 0; /* not ours */
  }

  auto entry = vsyscall_make_proc_fdinfo_entry(target_fd, target_entry, path,
                                               static_cast<int>(flags));
  if (!entry) {
    return vsyscall_consume_and_fail(forward, ENOMEM);
  }
  return vsyscall_alloc_tracked_fd(std::move(entry), forward);
}

static inline vsyscall_dispatch_node_result
vsyscall_resolve_open_node_for_dispatch(int dfd, const char *pathname,
                                        int *forward) {
  const vsyscall_dispatch_path_result path_result =
      vsyscall_dispatch_resolve_path_for_dispatch(dfd, pathname, forward);
  if (path_result.status != vsyscall_dispatch_path_status::READY) {
    return vsyscall_dispatch_node_result{
        vsyscall_dispatch_node_status::STOP,
        {},
        -1,
        path_result.path.from_owned_dfd,
        path_result.ret,
    };
  }

  vsyscall_dispatch_resolved_path resolved_path = path_result.path;
  int proc_fdinfo_target_fd = -1;
  if (vsyscall_parse_proc_self_numeric_path(resolved_path.abs_path,
                                            "/proc/self/fdinfo/",
                                            &proc_fdinfo_target_fd) &&
      proc_fdinfo_target_fd >= 0 &&
      vsyscall_fd_is_owned(proc_fdinfo_target_fd)) {
    return vsyscall_dispatch_node_result{
        vsyscall_dispatch_node_status::PROC_FDINFO,
        {},
        proc_fdinfo_target_fd,
        resolved_path.from_owned_dfd,
        0,
    };
  }

  std::string rewritten_abs_path;
  if (vsyscall_rewrite_proc_self_fd_path_for_open(resolved_path.abs_path,
                                                  &rewritten_abs_path)) {
    resolved_path.abs_path = std::move(rewritten_abs_path);
  }

  return vsyscall_dispatch_lookup_node_from_resolved_path(resolved_path,
                                                          forward, ENOENT);
}

static inline long vsyscall_dispatch_open_resolved_node(
    const vsyscall_dispatch_node_result &resolved, long flags, int *forward) {
  if (resolved.status == vsyscall_dispatch_node_status::STOP) {
    return resolved.ret;
  }
  const int flag_err = vsyscall_validate_open_flags(flags);
  if (flag_err != 0) {
    return vsyscall_consume_and_fail(forward, flag_err);
  }
  if (resolved.status == vsyscall_dispatch_node_status::PROC_FDINFO) {
    const std::string fdinfo_path =
        resolved.node.path.empty()
            ? (std::string("/proc/self/fdinfo/") +
               std::to_string(resolved.proc_fdinfo_target_fd))
            : resolved.node.path;
    return vsyscall_dispatch_open_proc_fdinfo_node(
        resolved.proc_fdinfo_target_fd, fdinfo_path, flags, forward);
  }

  const vfile_virtual_path_info &node = resolved.node;
  switch (node.kind) {
  case vfile_virtual_path_kind::VDEV_NODE:
    if (!node.vdev) {
      return vsyscall_consume_and_fail(forward, EIO);
    }
    return vsyscall_dispatch_open_vdev_node(node, flags, forward);
  case vfile_virtual_path_kind::VSYSFS_ATTR:
    if (!node.vsysfs || node.attr_name.empty()) {
      return vsyscall_consume_and_fail(forward, EIO);
    }
    return vsyscall_dispatch_open_vsysfs_attr_node(node, flags, forward);
  case vfile_virtual_path_kind::VMODULE_ATTR:
    if (!node.vmodule || node.attr_name.empty()) {
      return vsyscall_consume_and_fail(forward, EIO);
    }
    return vsyscall_dispatch_open_vmodule_attr_node(node, flags, forward);
  case vfile_virtual_path_kind::VDIR: {
    if (node.path.empty()) {
      return vsyscall_consume_and_fail(forward, EIO);
    }

    std::string class_symlink_target;
    if (vsyscall_is_vsysfs_class_symlink_node(node, &class_symlink_target)) {
#ifdef O_NOFOLLOW
      if ((flags & O_NOFOLLOW) != 0) {
#ifdef O_PATH
        if ((flags & O_PATH) != 0) {
          // Linux keeps O_PATH|O_NOFOLLOW on the link itself.
          return vsyscall_dispatch_open_dir_node(node, flags, forward);
        }
#endif
        return vsyscall_consume_and_fail(forward, ELOOP);
      }
#endif
      vfile_virtual_path_info target_node;
      int resolve_errno = 0;
      if (!vfile_resolve_virtual_node(class_symlink_target, &target_node,
                                      &resolve_errno)) {
        return vsyscall_consume_and_fail(
            forward, (resolve_errno > 0) ? resolve_errno : ENOENT);
      }
      vsyscall_dispatch_node_result target_resolved;
      target_resolved.status = vsyscall_dispatch_node_status::RESOLVED;
      target_resolved.node = std::move(target_node);
      target_resolved.from_owned_dfd = resolved.from_owned_dfd;
      return vsyscall_dispatch_open_resolved_node(target_resolved, flags,
                                                  forward);
    }

    return vsyscall_dispatch_open_dir_node(node, flags, forward);
  }
  default:
    return vsyscall_return_not_ours_or_fail(resolved.from_owned_dfd, forward,
                                            ENOENT);
  }
}

/**
 * @brief Handle open/openat path dispatch to vdev/vsysfs
 */
static long dispatch_openat(long dfd, const char *pathname, long flags,
                            long mode, int *forward) {
  (void)mode;
  if (!pathname)
    return 0; /* forward to kernel */

  const vsyscall_dispatch_node_result resolved =
      vsyscall_resolve_open_node_for_dispatch(static_cast<int>(dfd), pathname,
                                              forward);
  return vsyscall_dispatch_open_resolved_node(resolved, flags, forward);
}

static inline long dispatch_name_to_handle_at(int dfd, const char *pathname,
                                              struct file_handle *handle,
                                              int *mount_id, int flags,
                                              int *forward) {
  (void)handle;
  (void)mount_id;
  (void)flags;

#ifdef AT_EMPTY_PATH
  const bool empty_path_request =
      ((flags & AT_EMPTY_PATH) != 0) &&
      ((pathname == nullptr) || (pathname[0] == '\0'));
  if (empty_path_request) {
    if (!vsyscall_fd_is_owned(dfd)) {
      return 0; /* not ours */
    }
    return vsyscall_consume_and_fail(forward, EOPNOTSUPP);
  }
#endif

  if (!pathname) {
    return 0; /* forward to kernel */
  }

  const vsyscall_dispatch_path_result path_result =
      vsyscall_dispatch_resolve_path_for_dispatch(dfd, pathname, forward);
  if (path_result.status != vsyscall_dispatch_path_status::READY) {
    return path_result.ret;
  }

  vsyscall_dispatch_resolved_path resolved_path = path_result.path;
  std::string rewritten_abs_path;
  if (vsyscall_rewrite_proc_self_fd_path_for_open(resolved_path.abs_path,
                                                  &rewritten_abs_path)) {
    resolved_path.abs_path = std::move(rewritten_abs_path);
  }

  const vsyscall_dispatch_node_result resolved =
      vsyscall_dispatch_lookup_node_from_resolved_path(resolved_path, forward,
                                                       ENOENT);
  if (resolved.status != vsyscall_dispatch_node_status::RESOLVED) {
    return resolved.ret;
  }

  return vsyscall_consume_and_fail(forward, EOPNOTSUPP);
}

enum class vsyscall_io_operation { READ, WRITE };

template <vsyscall_io_operation Op, typename BufferT>
static inline long
vsyscall_dispatch_io_entry(const std::shared_ptr<vfile_fd_entry> &entry,
                           BufferT buf, size_t count) {
  if (vsyscall_fd_opened_with_path_only(entry->flags)) {
    return vfile_fail_with_errno(EBADF);
  }
  switch (entry->type) {
  case vfile_fd_entry::FD_VDEV:
    (void)buf;
    (void)count;
    return vfile_fail_with_errno(EINVAL);
  case vfile_fd_entry::FD_VSYSFS:
  case vfile_fd_entry::FD_VMODULE:
  case vfile_fd_entry::FD_PROC_FDINFO:
    if constexpr (Op == vsyscall_io_operation::READ) {
      return dispatch_attr_read(entry, buf, count);
    }
    return dispatch_attr_write(entry, reinterpret_cast<const void *>(buf),
                               count);
  case vfile_fd_entry::FD_VDIR:
    if constexpr (Op == vsyscall_io_operation::READ) {
      return vfile_fail_with_errno(EISDIR);
    }
    return vfile_fail_with_errno(EBADF);
  }
  return vfile_fail_with_errno(EBADF);
}

template <vsyscall_io_operation Op, typename BufferT>
static inline long dispatch_io(int fd, BufferT buf, size_t count,
                               int *forward) {
  return vsyscall_dispatch_owned_fd(
      fd, forward, [buf, count](const std::shared_ptr<vfile_fd_entry> &entry) {
        return vsyscall_dispatch_io_entry<Op>(entry, buf, count);
      });
}

/**
 * @brief Handle SYS_mmap on an owned fd
 */
static long dispatch_mmap(void *addr, size_t length, int prot, int flags,
                          int fd, off_t offset, int *forward) {
  (void)addr;
  long ret = vsyscall_dispatch_owned_vdev_callback(
      fd, forward, &pim_vdev_t::mmap_fn, ENODEV, EBADF,
      [length, prot, flags, offset](pim_vdev_mmap_fn mmap_fn,
                                    pim_vdev_t *vsyscall_raw,
                                    int owner_fd) -> long {
        return reinterpret_cast<long>(
            mmap_fn(vsyscall_raw, owner_fd, length, prot, flags, offset));
      });
  if (ret >= 0) {
    vsyscall_track_mmap_region(reinterpret_cast<void *>(ret), length);
  }
  return ret;
}

#ifdef SYS_munmap
struct vsyscall_munmap_context {
  uintptr_t req_start{0};
  uintptr_t req_end{0};
  uintptr_t map_start{0};
  uintptr_t map_end{0};
  size_t map_len{0};
};

enum class vsyscall_munmap_prepare_status { READY, RETURN };

struct vsyscall_munmap_prepare_result {
  vsyscall_munmap_prepare_status status{vsyscall_munmap_prepare_status::RETURN};
  vsyscall_munmap_context ctx{};
  long ret{0};
};

static inline vsyscall_munmap_prepare_result
vsyscall_prepare_munmap_context(void *addr, size_t length, int *forward) {
  vsyscall_munmap_prepare_result result;
  if (addr == MAP_FAILED) {
    return result; /* not ours */
  }

  result.ctx.req_start = reinterpret_cast<uintptr_t>(addr);
  {
    std::lock_guard<std::mutex> lock(g_mmap_table_mutex);
    auto it = g_mmap_table.upper_bound(result.ctx.req_start);
    if (it == g_mmap_table.begin()) {
      return result; /* not ours */
    }
    --it;

    result.ctx.map_start = it->first;
    result.ctx.map_len = it->second;
    if (!vsyscall_range_end(result.ctx.map_start, result.ctx.map_len,
                            &result.ctx.map_end)) {
      g_mmap_table.erase(it);
      return result; /* invalid bookkeeping entry */
    }
    if (!vsyscall_range_end(result.ctx.req_start, length,
                            &result.ctx.req_end)) {
      // Mirror kernel behavior for invalid length/overflow.
      result.ret = vsyscall_consume_and_fail(forward, EINVAL);
      return result;
    }

    // Own only exact subranges of tracked mmap results.
    if (result.ctx.req_start < result.ctx.map_start ||
        result.ctx.req_start >= result.ctx.map_end ||
        result.ctx.req_end > result.ctx.map_end) {
      return result; /* not ours */
    }
  }

  result.status = vsyscall_munmap_prepare_status::READY;
  return result;
}

static inline void
vsyscall_update_mmap_table_after_munmap(const vsyscall_munmap_context &ctx) {
  std::lock_guard<std::mutex> lock(g_mmap_table_mutex);
  auto it = g_mmap_table.find(ctx.map_start);
  if (it == g_mmap_table.end() || it->second != ctx.map_len) {
    return; // best effort bookkeeping only
  }
  g_mmap_table.erase(it);
  vsyscall_run_best_effort([&]() {
    if (ctx.req_start > ctx.map_start) {
      g_mmap_table.emplace(ctx.map_start,
                           static_cast<size_t>(ctx.req_start - ctx.map_start));
    }
    if (ctx.req_end < ctx.map_end) {
      g_mmap_table.emplace(ctx.req_end,
                           static_cast<size_t>(ctx.map_end - ctx.req_end));
    }
  });
}

/**
 * @brief Handle SYS_munmap for mmap ranges previously returned by vdev mmap cb.
 *
 * There is no user munmap callback in the vdev API, so this path only
 * consumes munmap for tracked mmap results and delegates to libc munmap.
 */
static long dispatch_munmap(void *addr, size_t length, int *forward) {
  const vsyscall_munmap_prepare_result prepared =
      vsyscall_prepare_munmap_context(addr, length, forward);
  if (prepared.status != vsyscall_munmap_prepare_status::READY) {
    return prepared.ret;
  }

  vsyscall_consume_syscall(forward);
  if (::munmap(addr, length) < 0) {
    return -1;
  }

  vsyscall_update_mmap_table_after_munmap(prepared.ctx);
  return 0;
}

#endif

static inline bool vsyscall_fcntl_allowed_for_path_only(int cmd) noexcept {
  switch (cmd) {
  case F_GETFD:
  case F_SETFD:
  case F_GETFL:
    return true;
  default:
    return false;
  }
}

static long dispatch_fcntl(int fd, int cmd, long arg, int *forward) {
  return vsyscall_dispatch_owned_fd(
      fd, forward,
      [fd, cmd, arg](const std::shared_ptr<vfile_fd_entry> &entry) -> long {
        if (!entry) {
          return vfile_fail_with_errno(EBADF);
        }

        if (vsyscall_fd_opened_with_path_only(entry->flags) &&
            !vsyscall_fcntl_allowed_for_path_only(cmd)) {
          return vfile_fail_with_errno(EBADF);
        }

        switch (cmd) {
        case F_GETFL: {
          std::lock_guard<std::mutex> lock(entry->op_mutex);
          return static_cast<long>(entry->flags);
        }
        case F_SETFL: {
          const int requested_flags = static_cast<int>(arg);
          const int updatable_mask = vsyscall_fcntl_setfl_supported_mask();
          std::lock_guard<std::mutex> lock(entry->op_mutex);
          entry->flags = (entry->flags & ~updatable_mask) |
                         (requested_flags & updatable_mask);
          return 0;
        }
        case F_GETFD: {
#ifdef O_CLOEXEC
          std::lock_guard<std::mutex> lock(entry->op_mutex);
          return ((entry->flags & O_CLOEXEC) != 0) ? FD_CLOEXEC : 0;
#else
          return 0;
#endif
        }
        case F_SETFD: {
#ifdef O_CLOEXEC
          std::lock_guard<std::mutex> lock(entry->op_mutex);
          if ((static_cast<int>(arg) & FD_CLOEXEC) != 0) {
            entry->flags |= O_CLOEXEC;
          } else {
            entry->flags &= ~O_CLOEXEC;
          }
#endif
          return 0;
        }
        default:
          return static_cast<long>(fcntl(fd, cmd, arg));
        }
      });
}

#ifdef SYS_dup
static long dispatch_dup(int oldfd, int *forward) {
  auto old_entry = vsyscall_fd_lookup(oldfd);
  return vsyscall_dispatch_fd_mutation(
      static_cast<bool>(old_entry), forward,
      [oldfd]() -> long { return static_cast<long>(dup(oldfd)); },
      [old_entry](long kernel_ret) -> long {
        return vsyscall_rebind_fd_alias_after_dup(static_cast<int>(kernel_ret),
                                                  old_entry, true);
      });
}
#endif

#if defined(SYS_dup2) || defined(SYS_dup3)
static inline long vsyscall_dup2_or_dup3_kernel_call(int oldfd, int newfd,
                                                     int flags, bool is_dup3) {
  if (!is_dup3 && oldfd == newfd) {
    return static_cast<long>(newfd);
  }
  if (is_dup3 && oldfd == newfd) {
    return vfile_fail_with_errno(EINVAL);
  }

  if (is_dup3) {
#ifdef SYS_dup3
    // Use native dup3 to preserve kernel atomic semantics and keep fd-table
    // tracking consistent on errors.
    return static_cast<long>(dup3(oldfd, newfd, flags));
#else
    return vfile_fail_with_errno(ENOSYS);
#endif
  }

#ifdef SYS_dup2
  return static_cast<long>(dup2(oldfd, newfd));
#else
  return vfile_fail_with_errno(ENOSYS);
#endif
}

static inline long vsyscall_finalize_dup2_or_dup3_aliases(
    int oldfd, int newfd, bool is_dup3,
    const std::shared_ptr<vfile_fd_entry> &old_entry, long kernel_ret) {
  if (!is_dup3 && oldfd == newfd) {
    return kernel_ret;
  }
  return vsyscall_rebind_fd_alias_after_dup(newfd, old_entry, false);
}

static inline long vsyscall_dispatch_dup2_or_dup3_with_alias_tracking(
    int oldfd, int newfd, int flags, bool is_dup3,
    const vsyscall_dup2_prepare_result &prepare, int *forward) {
  const std::shared_ptr<vfile_fd_entry> old_entry = prepare.old_entry;
  return vsyscall_dispatch_fd_mutation(
      prepare.should_intercept, forward,
      [oldfd, newfd, flags, is_dup3]() -> long {
        return vsyscall_dup2_or_dup3_kernel_call(oldfd, newfd, flags, is_dup3);
      },
      [oldfd, newfd, is_dup3, old_entry](long kernel_ret) -> long {
        return vsyscall_finalize_dup2_or_dup3_aliases(oldfd, newfd, is_dup3,
                                                      old_entry, kernel_ret);
      });
}

static long dispatch_dup2_enter(int oldfd, int newfd, int flags, bool is_dup3,
                                int *forward) {
  const vsyscall_dup2_prepare_result prepare =
      vsyscall_prepare_dup2_context(oldfd, newfd);
  return vsyscall_dispatch_dup2_or_dup3_with_alias_tracking(
      oldfd, newfd, flags, is_dup3, prepare, forward);
}
#endif

/* liboverlaysys will care nested syscall interception. */
#ifdef SYS_close_range
static inline long vsyscall_sweep_close_range_stale_aliases(unsigned int first,
                                                            unsigned int last) {
  // close_range may close descriptors that were not in the pre-kernel
  // owned-fd snapshot (e.g., concurrent alias changes). Sweep current range
  // and detach only entries that are now invalid at the kernel level.
  std::vector<int> maybe_stale =
      vsyscall_collect_owned_fds_in_range(first, last);
  std::sort(maybe_stale.begin(), maybe_stale.end());
  for (int fd : maybe_stale) {
    if (fd < 0) {
      continue;
    }
    if (fcntl(fd, F_GETFD) >= 0) {
      continue;
    }
    if (errno != EBADF) {
      continue;
    }
    const long stale_ret = close_owned_fd(fd, false);
    if (stale_ret < 0) {
      return stale_ret;
    }
  }
  return 0;
}

static inline long
vsyscall_finalize_close_range_aliases(unsigned int first, unsigned int last,
                                      unsigned int flags,
                                      std::vector<int> owned_fds) {
#ifdef CLOSE_RANGE_CLOEXEC
  if ((flags & CLOSE_RANGE_CLOEXEC) != 0U) {
    return 0;
  }
#endif
  std::sort(owned_fds.begin(), owned_fds.end());
  const long close_ret = vsyscall_close_owned_fd_list(owned_fds, false);
  if (close_ret < 0) {
    return close_ret;
  }

  return vsyscall_sweep_close_range_stale_aliases(first, last);
}
#endif

static long dispatch_close_range(unsigned int first, unsigned int last,
                                 unsigned int flags, int *forward) {
#ifdef SYS_close_range
  std::vector<int> owned_fds = vsyscall_collect_owned_fds_in_range(first, last);
  return vsyscall_dispatch_fd_mutation(
      true, forward,
      [first, last, flags]() -> long {
        return static_cast<long>(close_range(first, last, flags));
      },
      [first, last, flags,
       owned_fds = std::move(owned_fds)](long kernel_ret) mutable -> long {
        (void)kernel_ret;
        return vsyscall_finalize_close_range_aliases(first, last, flags,
                                                     std::move(owned_fds));
      });
#else
  (void)first;
  (void)last;
  (void)flags;
  (void)forward;
  return 0;
#endif
}

bool pim_vfile_owns_fd(int fd) noexcept {
  return vsyscall_fd_lookup(fd) != nullptr;
}

void pim_vfile_clear(void) noexcept {
  vsyscall_close_all_owned_fds_best_effort();
  vsyscall_clear_mmap_tracking();
  vfile_clear_vmodule_state();
  vfile_clear_vsysfs_state();
  vfile_clear_vdev_state();
  vfile_notify_virtual_fs_changed();
}

long _pim_vsyscall_dispatch(long number, long a, long b, long c, long d, long e,
                            long f, int *forward) noexcept {
  long ret = 0;

  try {
#if !defined(SYS_mmap) && !defined(SYS_statx)
    (void)e;
#endif
#ifndef SYS_mmap
    (void)f;
#endif

    switch (number) {
#ifdef SYS_open
    case SYS_open:
      ret = dispatch_openat(AT_FDCWD, reinterpret_cast<const char *>(a), b, c,
                            forward);
      break;
#endif
#ifdef SYS_openat
    case SYS_openat:
      ret =
          dispatch_openat(a, reinterpret_cast<const char *>(b), c, d, forward);
      break;
#endif
#ifdef SYS_name_to_handle_at
    case SYS_name_to_handle_at:
      ret = dispatch_name_to_handle_at(
          static_cast<int>(a), reinterpret_cast<const char *>(b),
          reinterpret_cast<struct file_handle *>(c), reinterpret_cast<int *>(d),
          static_cast<int>(e), forward);
      break;
#endif
#ifdef SYS_access
    case SYS_access:
      ret =
          dispatch_faccessat_common(AT_FDCWD, reinterpret_cast<const char *>(a),
                                    static_cast<int>(b), 0, forward);
      break;
#endif
#ifdef SYS_faccessat
    case SYS_faccessat:
      ret = dispatch_faccessat_common(static_cast<int>(a),
                                      reinterpret_cast<const char *>(b),
                                      static_cast<int>(c), 0, forward);
      break;
#endif
#ifdef SYS_faccessat2
    case SYS_faccessat2:
      ret = dispatch_faccessat_common(
          static_cast<int>(a), reinterpret_cast<const char *>(b),
          static_cast<int>(c), static_cast<int>(d), forward);
      break;
#endif
#ifdef SYS_readlinkat
    case SYS_readlinkat:
      ret = dispatch_readlinkat(
          static_cast<int>(a), reinterpret_cast<const char *>(b),
          reinterpret_cast<char *>(c), static_cast<size_t>(d), forward);
      break;
#endif
#ifdef SYS_readlink
    case SYS_readlink:
      ret = dispatch_readlinkat(AT_FDCWD, reinterpret_cast<const char *>(a),
                                reinterpret_cast<char *>(b),
                                static_cast<size_t>(c), forward);
      break;
#endif
#ifdef SYS_fcntl
    case SYS_fcntl:
      ret =
          dispatch_fcntl(static_cast<int>(a), static_cast<int>(b), c, forward);
      break;
#endif
#ifdef SYS_fcntl64
    case SYS_fcntl64:
      ret =
          dispatch_fcntl(static_cast<int>(a), static_cast<int>(b), c, forward);
      break;
#endif
#ifdef SYS_ioctl
    case SYS_ioctl:
      ret = vsyscall_dispatch_owned_vdev_callback(
          static_cast<int>(a), forward, &pim_vdev_t::ioctl_fn, ENOTTY, ENOTTY,
          [cmd = static_cast<unsigned long>(b),
           arg = static_cast<unsigned long>(c)](pim_vdev_ioctl_fn ioctl_fn,
                                                pim_vdev_t *vsyscall_raw,
                                                int owner_fd) {
            return ioctl_fn(vsyscall_raw, owner_fd, cmd, arg);
          });
      break;
#endif
#ifdef SYS_read
    case SYS_read:
      ret = dispatch_io<vsyscall_io_operation::READ>(
          static_cast<int>(a), reinterpret_cast<void *>(b),
          static_cast<size_t>(c), forward);
      break;
#endif
#ifdef SYS_write
    case SYS_write:
      ret = dispatch_io<vsyscall_io_operation::WRITE>(
          static_cast<int>(a), reinterpret_cast<const void *>(b),
          static_cast<size_t>(c), forward);
      break;
#endif
#ifdef SYS_getdents64
    case SYS_getdents64:
      ret = vsyscall_dispatch_owned_fd(
          static_cast<int>(a), forward,
          [dirp = reinterpret_cast<void *>(b), count = static_cast<size_t>(c)](
              const std::shared_ptr<vfile_fd_entry> &entry) {
            return vdir_getdents64_from_entry(entry, dirp, count);
          });
      break;
#endif
#ifdef SYS_getdents
    case SYS_getdents:
      ret = vsyscall_dispatch_owned_fd(
          static_cast<int>(a), forward,
          [dirp = reinterpret_cast<void *>(b), count = static_cast<size_t>(c)](
              const std::shared_ptr<vfile_fd_entry> &entry) {
            return vdir_getdents_from_entry(entry, dirp, count);
          });
      break;
#endif
#ifdef SYS_mmap
    case SYS_mmap:
      ret = dispatch_mmap(reinterpret_cast<void *>(a), static_cast<size_t>(b),
                          static_cast<int>(c), static_cast<int>(d),
                          static_cast<int>(e), static_cast<off_t>(f), forward);
      break;
#endif
#ifdef SYS_munmap
    case SYS_munmap:
      ret = dispatch_munmap(reinterpret_cast<void *>(a), static_cast<size_t>(b),
                            forward);
      break;
#endif
#ifdef SYS_close
    case SYS_close:
      ret = vsyscall_dispatch_fd_mutation(
          vsyscall_fd_is_owned(static_cast<int>(a)), forward,
          []() -> long { return 0; },
          [fd = static_cast<int>(a)](long) -> long {
            return close_owned_fd(fd, true);
          });
      break;
#endif
#ifdef SYS_dup
    case SYS_dup:
      ret = dispatch_dup(static_cast<int>(a), forward);
      break;
#endif
#ifdef SYS_dup2
    case SYS_dup2:
      ret = dispatch_dup2_enter(static_cast<int>(a), static_cast<int>(b), 0,
                                false, forward);
      break;
#endif
#ifdef SYS_dup3
    case SYS_dup3:
      ret = dispatch_dup2_enter(static_cast<int>(a), static_cast<int>(b),
                                static_cast<int>(c), true, forward);
      break;
#endif
#ifdef SYS_close_range
    case SYS_close_range:
      ret = dispatch_close_range(static_cast<unsigned int>(a),
                                 static_cast<unsigned int>(b),
                                 static_cast<unsigned int>(c), forward);
      break;
#endif
#ifdef SYS_fstat
    case SYS_fstat:
      ret = vsyscall_dispatch_owned_fd(
          static_cast<int>(a), forward,
          [statbuf = reinterpret_cast<struct stat *>(b)](
              const std::shared_ptr<vfile_fd_entry> &entry) {
            return vsyscall_build_stat_for_owned_entry(entry, statbuf);
          });
      break;
#endif
#ifdef SYS_fstatfs
    case SYS_fstatfs:
      ret = dispatch_fstatfs_typed(
          static_cast<int>(a), reinterpret_cast<struct statfs *>(b), forward);
      break;
#endif
#ifdef SYS_newfstatat
    case SYS_newfstatat:
      ret = dispatch_newfstatat(
          static_cast<int>(a), reinterpret_cast<const char *>(b),
          reinterpret_cast<struct stat *>(c), static_cast<int>(d), forward);
      break;
#endif
#ifdef SYS_statfs
    case SYS_statfs:
      ret = vsyscall_dispatch_statfs_for_path(
          reinterpret_cast<const char *>(a),
          reinterpret_cast<struct statfs *>(b), forward);
      break;
#endif
#ifdef SYS_statfs64
    case SYS_statfs64:
      ret = dispatch_statfs64(reinterpret_cast<const char *>(a),
                              static_cast<size_t>(b),
                              reinterpret_cast<struct statfs64 *>(c), forward);
      break;
#endif
#ifdef SYS_statx
    case SYS_statx:
      ret =
          dispatch_statx(static_cast<int>(a), reinterpret_cast<const char *>(b),
                         static_cast<int>(c), static_cast<unsigned int>(d),
                         reinterpret_cast<struct statx *>(e), forward);
      break;
#endif
#ifdef SYS_lseek
    case SYS_lseek:
      ret = dispatch_lseek(static_cast<int>(a), static_cast<off_t>(b),
                           static_cast<int>(c), forward);
      break;
#endif

    } /* switch */

    return ret;
  } catch (const std::bad_alloc &) {
    if (forward) {
      *forward = 0;
    }
    errno = ENOMEM;
    return -1;
  } catch (...) {
    if (forward) {
      *forward = 0;
    }
    errno = EFAULT;
    return -1;
  }
}
