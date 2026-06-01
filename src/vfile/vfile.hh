/**
 * @file vfile.hh
 * @brief Internal shared declarations for vdev/vsysfs/vsyscall.
 */

#pragma once

#include "hostpimsim.h"

#include <condition_variable>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

/*============================================================================
 * Virtual Device Internal Structures
 *============================================================================*/

/**
 * @brief Cached virtual directory row for getdents/getdents64.
 */
struct vdev_dirent_row {
  std::string name;
  unsigned char dtype{0};
  uint64_t ino{0};
};

/**
 * @brief Per-fd session entry: links an owned fd to its owning virtual node.
 */
struct vfile_fd_entry {
  enum fd_type { FD_VDEV, FD_VSYSFS, FD_VMODULE, FD_PROC_FDINFO, FD_VDIR } type;
  std::shared_ptr<pim_vdev_t> vdev;
  std::shared_ptr<pim_vsysfs_t> vsysfs;
  std::shared_ptr<pim_vmodule_t> vmodule;
  int owner_fd{-1}; /**< Canonical real fd for this vdev open session */
  int flags{0};     /**< File descriptor flags tracked for this fd */

  /* For virtual attribute fds (vsysfs/vmodule): which attribute was opened */
  std::string sysfs_attr_name;
  size_t sysfs_read_offset{0};            /**< Current read cursor */
  std::string sysfs_cached_value;         /**< Last fetched attr payload */
  uint64_t sysfs_cached_attr_revision{0}; /**< Revision of cached payload */
  pim_access_mode_t sysfs_cached_attr_access{
      PIM_ACCESS_MODE_RW}; /**< Last fetched attr access mode */

  /* For virtual directory fds: canonical absolute path + readdir cursor */
  std::string virtual_path;
  size_t dir_read_offset{0};
  std::vector<vdev_dirent_row> dir_cached_rows;
  uint64_t dir_cached_generation{0};

  /*
   * Operation lifetime coordination:
   * - dispatch_* increments active_ops before using this entry
   * - close marks closing=true, waits active_ops==0, then runs close callback
   */
  std::mutex op_mutex;
  std::condition_variable op_cv;
  uint32_t active_ops{0};
  bool closing{false};
};

/**
 * @brief Virtual device root (/dev/<name_glob>)
 */
struct pim_vdev_root {
  std::string name_glob; /**< Relative name glob (e.g. "dpu_rank*") */
  std::mutex mutex;      /**< Protects callbacks + instance index */
  std::vector<std::weak_ptr<pim_vdev_t>>
      instances; /**< Registered instances under this root */

  /* Root callbacks (invoked with resolved pim_vdev_t instance) */
  pim_vdev_open_fn open_fn{nullptr};
  pim_vdev_ioctl_fn ioctl_fn{nullptr};
  pim_vdev_mmap_fn mmap_fn{nullptr};
  pim_vdev_close_fn close_fn{nullptr};
};

/**
 * @brief Virtual device instance (/dev/<name>)
 */
struct pim_vdev {
  std::weak_ptr<pim_vdev_root_t> root; /**< Owning root */
  std::string name;                    /**< Relative instance name */
  std::string path;                    /**< Absolute /dev path */
  unsigned int major{1};               /**< Device major for stat metadata */
  unsigned int minor{3};               /**< Device minor for stat metadata */
  std::mutex mutex; /**< Protects per-instance maps + callbacks */
  std::unordered_map<int, pim_device_t *>
      device_by_fd;         /**< fd -> associated PIM device (borrowed ptr) */
  void *user_data{nullptr}; /**< Caller-owned opaque pointer */
  pim_vdev_open_fn open_fn{nullptr};
  pim_vdev_ioctl_fn ioctl_fn{nullptr};
  pim_vdev_mmap_fn mmap_fn{nullptr};
  pim_vdev_close_fn close_fn{nullptr};
};

/**
 * @brief Virtual sysfs root
 *        (/sys/class/<class_glob>, /sys/devices/<devices_glob>)
 */
struct pim_vsysfs_root {
  std::weak_ptr<pim_vdev_root_t> vdev_root; /**< Owning vdev root */
  std::string class_name;         /**< Directory name under /sys/class */
  std::string kobject_name;       /**< Directory name under /sys/devices */
  std::string instance_name_glob; /**< Bus-instance glob */
  std::string class_glob;         /**< Relative glob under /sys/class */
  std::string devices_glob;       /**< Relative glob under /sys/devices */
  std::mutex mutex;               /**< Protects defaults + instances */
  std::map<std::string, std::string>
      default_attrs; /**< Default attrs for instances */
  std::map<std::string, pim_access_mode_t>
      default_attr_access; /**< Default attr access for instances */
  std::vector<std::weak_ptr<pim_vsysfs_t>>
      instances; /**< Registered instances under this root */
};

/**
 * @brief Virtual sysfs instance
 */
struct pim_vsysfs {
  std::weak_ptr<pim_vsysfs_root_t> root;    /**< Owning root */
  std::weak_ptr<pim_vdev_t> vdev;           /**< Linked vdev instance */
  std::string class_name;                   /**< Relative class-side name */
  std::string device_name;                  /**< Relative devices-side name */
  std::string class_dir;                    /**< Absolute /sys/class path */
  std::string devices_dir;                  /**< Absolute /sys/devices path */
  std::map<std::string, std::string> attrs; /**< name -> value */
  std::map<std::string, pim_access_mode_t> attr_access; /**< name -> RO/RW/WO */
  std::atomic<uint64_t> attrs_revision{
      1};           /**< Bumped on attr add/update/remove */
  std::mutex mutex; /**< Protects attrs + attachment */
  pim_device_t *attached_device{nullptr}; /**< Borrowed attached device */
};

/**
 * @brief Virtual module instance (/sys/module/<name>)
 */
struct pim_vmodule {
  std::string name;                         /**< Relative module name */
  std::string dir;                          /**< Absolute /sys/module path */
  std::map<std::string, std::string> attrs; /**< name -> value */
  std::set<std::pair<std::string, std::string>>
      registered_drivers; /**< (bus_name, driver_name) pairs */
  std::map<std::string, pim_access_mode_t> attr_access; /**< name -> RO/RW/WO */
  std::atomic<uint64_t> attrs_revision{
      1};           /**< Bumped on attr add/update/remove */
  std::mutex mutex; /**< Protects attrs */
};

// Classification of resolved virtual paths:
// - NONE: unresolved/empty node placeholder
// - VDEV_NODE: virtual device node matched from pim_vdev registration
// - VSYSFS_ATTR: virtual sysfs attribute file under a vsysfs directory
// - VMODULE_ATTR: virtual module attribute file under /sys/module/<name>
// - VDIR: virtual directory node (root, vsysfs dir, or synthetic parent dir)
enum class vfile_virtual_path_kind {
  NONE,
  VDEV_NODE,
  VSYSFS_ATTR,
  VMODULE_ATTR,
  VDIR
};

struct vfile_virtual_path_info {
  vfile_virtual_path_kind kind{vfile_virtual_path_kind::NONE};
  std::shared_ptr<pim_vdev_t> vdev;
  std::shared_ptr<pim_vsysfs_t> vsysfs;
  std::shared_ptr<pim_vmodule_t> vmodule;
  std::string attr_name;
  size_t attr_size{0};
  pim_access_mode_t attr_access{PIM_ACCESS_MODE_RW};
  std::string path;
};

/**
 * @brief Normalize an absolute path by collapsing ".", "..", and repeated '/'.
 *
 * Returns false if `input` is not absolute or output is invalid.
 */
static inline bool vfile_normalize_absolute_path(const std::string &input,
                                                 std::string *out) {
  if (!out || input.empty() || input[0] != '/') {
    return false;
  }

  std::vector<std::string> stack;
  size_t i = 0;
  const size_t n = input.size();
  while (i < n) {
    while (i < n && input[i] == '/') {
      ++i;
    }
    if (i >= n) {
      break;
    }

    size_t j = i;
    while (j < n && input[j] != '/') {
      ++j;
    }
    const std::string segment = input.substr(i, j - i);
    if (segment == ".") {
      // no-op
    } else if (segment == "..") {
      if (!stack.empty()) {
        stack.pop_back();
      }
    } else {
      stack.push_back(segment);
    }
    i = j;
  }

  std::string normalized("/");
  for (size_t idx = 0; idx < stack.size(); ++idx) {
    normalized.append(stack[idx]);
    if (idx + 1 < stack.size()) {
      normalized.push_back('/');
    }
  }
  *out = std::move(normalized);
  return true;
}

/**
 * @brief Remove trailing '/' while preserving single root "/".
 */
static inline std::string
vfile_strip_trailing_slashes_preserve_root(std::string path) noexcept {
  while (path.size() > 1U && path.back() == '/') {
    path.pop_back();
  }
  return path;
}

/**
 * @brief Strip prefix when `value` starts with `prefix`.
 */
static inline std::string
vfile_strip_known_prefix(std::string value, const char *prefix) noexcept {
  if (!prefix) {
    return value;
  }
  const size_t n = std::char_traits<char>::length(prefix);
  if (value.compare(0, n, prefix) == 0) {
    value.erase(0, n);
  }
  return value;
}

/**
 * @brief Return true if a path/name contains glob wildcard chars (* or ?).
 */
static inline bool
vfile_contains_glob_wildcards(const std::string &path) noexcept {
  return (path.find('*') != std::string::npos) ||
         (path.find('?') != std::string::npos);
}

/**
 * @brief Find the shared_ptr entry in `list` that matches `raw`.
 */
template <typename ObjT>
static inline std::shared_ptr<ObjT>
vfile_find_shared(const std::vector<std::shared_ptr<ObjT>> &list,
                  ObjT *raw) noexcept {
  if (!raw) {
    return nullptr;
  }
  for (const auto &entry : list) {
    if (entry.get() == raw) {
      return entry;
    }
  }
  return nullptr;
}

/**
 * @brief Set errno (with fallback) and return -1 for syscall-style failures.
 */
static inline long vfile_fail_with_errno(int err) noexcept {
  errno = (err > 0) ? err : EIO;
  return -1;
}

/**
 * @brief Run a callable and map exceptions to errno + false.
 */
template <typename Fn>
static inline bool vfile_try_or_false_with_errno(Fn &&fn, int err) noexcept {
  try {
    fn();
    return true;
  } catch (...) {
    errno = (err > 0) ? err : EIO;
    return false;
  }
}

/**
 * @brief Update a size_t cursor using SEEK_SET/SEEK_CUR with overflow checks.
 *
 * Returns 0 on success and stores the new cursor in `*cursor` and `*ret`.
 * Returns errno-style error on failure.
 */
static inline int vfile_seek_cursor_locked(size_t *cursor, off_t offset,
                                           int whence, long *ret) noexcept {
  if (!cursor || !ret) {
    return EINVAL;
  }

  long long base = 0;
  if (whence == SEEK_SET) {
    base = 0;
  } else if (whence == SEEK_CUR) {
    if (*cursor > static_cast<size_t>(std::numeric_limits<long long>::max())) {
      return EINVAL;
    }
    base = static_cast<long long>(*cursor);
  } else {
    return EINVAL;
  }

  const long long delta = static_cast<long long>(offset);
  if (delta > 0 && base > (std::numeric_limits<long long>::max() - delta)) {
    return EINVAL;
  }
  if (delta < 0 && base < (std::numeric_limits<long long>::min() - delta)) {
    return EINVAL;
  }
  const long long next = base + delta;
  if (next < 0) {
    return EINVAL;
  }

  *cursor = static_cast<size_t>(next);
  *ret = static_cast<long>(*cursor);
  return 0;
}

/**
 * @brief Bump a revision counter while keeping 0 as reserved/invalid sentinel.
 *
 * Returns the post-increment revision value (always non-zero).
 */
static inline uint64_t
vfile_bump_nonzero_revision(std::atomic<uint64_t> &revision) noexcept {
  uint64_t next = revision.fetch_add(1, std::memory_order_relaxed) + 1U;
  if (next == 0U) {
    next = 1U;
    revision.store(next, std::memory_order_relaxed);
  }
  return next;
}

/*============================================================================
 * Virtual Device Global State Accessors
 *============================================================================*/

/**
 * @brief Allocate/reserve a real fd and bind it in the global fd table.
 */
int vsyscall_fd_alloc(std::shared_ptr<vfile_fd_entry> entry);

/**
 * @brief Look up an owned fd entry (returns nullptr if not found).
 */
std::shared_ptr<vfile_fd_entry> vsyscall_fd_lookup(int fd);

/**
 * @brief Free an owned fd entry.
 */
void vsyscall_fd_free(int fd);

/**
 * @brief Simple glob match (supports * and ?).
 */
bool vdev_glob_match(const char *pattern, const char *str);

/**
 * @brief Notify syscall dispatch layer that virtual fs topology changed.
 */
void vfile_notify_virtual_fs_changed() noexcept;

/**
 * @brief Return current virtual fs topology generation.
 */
uint64_t vfile_get_virtual_fs_generation() noexcept;

/**
 * @brief Resolve virtual node metadata from absolute path.
 */
bool vfile_resolve_virtual_node(const std::string &abs_path,
                                vfile_virtual_path_info *node,
                                int *resolve_errno = nullptr);

/**
 * @brief Collect getdents rows for a virtual directory path.
 */
bool vfile_collect_dirents(const std::string &abs_dir_path,
                           std::vector<vdev_dirent_row> *rows);

/**
 * @brief Copy current virtual device instance list for snapshot building.
 */
std::vector<std::shared_ptr<pim_vdev_t>> vfile_copy_vdev_list_for_snapshot();

/**
 * @brief Look up a shared vdev root handle from a raw pointer.
 */
std::shared_ptr<pim_vdev_root_t>
vfile_lookup_vdev_root_shared(pim_vdev_root_t *root);

/**
 * @brief Look up a shared vdev instance handle from a raw pointer.
 */
std::shared_ptr<pim_vdev_t> vfile_lookup_vdev_shared(pim_vdev_t *vdev);

/**
 * @brief Copy current virtual sysfs instance list for snapshot building.
 */
std::vector<std::shared_ptr<pim_vsysfs_t>>
vfile_copy_vsysfs_list_for_snapshot();
std::vector<std::shared_ptr<pim_vsysfs_root_t>>
vfile_copy_vsysfs_root_list_for_snapshot();

/**
 * @brief Sync vsysfs->device attachment map after vdev fd-device changes.
 */
void vfile_on_vdev_device_mapping_changed(pim_vdev_t *vdev) noexcept;

/**
 * @brief Copy current virtual module instance list for snapshot building.
 */
std::vector<std::shared_ptr<pim_vmodule_t>>
vfile_copy_vmodule_list_for_snapshot();

/**
 * @brief Clear all registered virtual-device roots/instances.
 */
void vfile_clear_vdev_state() noexcept;

/**
 * @brief Clear all registered virtual-sysfs roots/instances and attachments.
 */
void vfile_clear_vsysfs_state() noexcept;

/**
 * @brief Clear all registered virtual-module instances.
 */
void vfile_clear_vmodule_state() noexcept;

/*============================================================================
 * Virtual Directory (VDIR) Helpers
 *============================================================================*/

/**
 * @brief Create a tracked FD entry for a virtual directory node.
 */
std::shared_ptr<vfile_fd_entry> vdir_make_fd_entry(std::string path, int flags);

/**
 * @brief Seek virtual-directory cursor (SEEK_SET/SEEK_CUR only).
 */
long vdir_seek_entry_cursor(const std::shared_ptr<vfile_fd_entry> &entry,
                            off_t offset, int whence);

#ifdef SYS_getdents64
/**
 * @brief Emit linux_dirent64 records from a virtual-directory entry.
 */
long vdir_getdents64_from_entry(const std::shared_ptr<vfile_fd_entry> &entry,
                                void *dirp, size_t count);
#endif

#ifdef SYS_getdents
/**
 * @brief Emit linux_dirent records from a virtual-directory entry.
 */
long vdir_getdents_from_entry(const std::shared_ptr<vfile_fd_entry> &entry,
                              void *dirp, size_t count);
#endif
