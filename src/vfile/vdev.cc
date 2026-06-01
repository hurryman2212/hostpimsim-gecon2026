/**
 * @file vdev.cc
 * @brief Virtual /dev root/instance registry and callback configuration
 *
 * This translation unit owns only virtual-device (/dev/<name>) metadata:
 *   - root creation/destruction
 *   - instance creation/destruction
 *   - callback propagation from root -> instances
 *   - per-instance fd->pim_device association map
 *   - /dev glob matcher and vdev snapshot source list
 *
 * Syscall interception, owned-fd lifecycle, virtual-fs snapshot/cache, and
 * vsysfs path dispatch live in `vsyscall.cc`.
 */

#include "vfile.hh"

#include <algorithm>
#include <mutex>
#include <unordered_set>

std::mutex g_vdev_list_mutex;
std::vector<std::shared_ptr<pim_vdev_t>> g_vdev_list;

std::mutex g_vdev_root_list_mutex;
std::vector<std::shared_ptr<pim_vdev_root_t>> g_vdev_root_list;

struct vdev_acquire_entry {
  std::shared_ptr<pim_vdev_t> hold;
  size_t refcount{0};
};

std::mutex g_vdev_acquire_mutex;
std::unordered_map<pim_vdev_t *, vdev_acquire_entry> g_vdev_acquire_map;

template <typename T, typename Fn>
static inline T *vdev_try_or_nullptr(Fn &&fn) noexcept {
  try {
    return fn();
  } catch (...) {
    return nullptr;
  }
}

static inline std::string vdev_normalize_name_glob(std::string value) {
  value = vfile_strip_trailing_slashes_preserve_root(std::move(value));
  value = vfile_strip_known_prefix(std::move(value), "/dev/");
  if (!value.empty() && value.front() == '/') {
    value.erase(0, 1);
  }
  return value;
}

static inline bool vdev_valid_instance_name(const std::string &name) noexcept {
  if (name.empty()) {
    return false;
  }
  if (name.find('/') != std::string::npos) {
    return false;
  }
  return !vfile_contains_glob_wildcards(name);
}

static inline uint64_t vdev_make_devnum_key(unsigned int major,
                                            unsigned int minor) noexcept {
  return (static_cast<uint64_t>(major) << 32U) | static_cast<uint64_t>(minor);
}

static inline bool vdev_increment_devnum_pair(unsigned int *major,
                                              unsigned int *minor) noexcept {
  if (!major || !minor) {
    return false;
  }

  const unsigned int kMax = std::numeric_limits<unsigned int>::max();
  if (*minor == kMax) {
    if (*major == kMax) {
      return false;
    }
    ++(*major);
    *minor = 0U;
    return true;
  }

  ++(*minor);
  return true;
}

std::shared_ptr<pim_vdev_root_t>
vfile_lookup_vdev_root_shared(pim_vdev_root_t *root) {
  std::lock_guard<std::mutex> lock(g_vdev_root_list_mutex);
  return vfile_find_shared(g_vdev_root_list, root);
}

std::shared_ptr<pim_vdev_t> vfile_lookup_vdev_shared(pim_vdev_t *vdev) {
  std::lock_guard<std::mutex> lock(g_vdev_list_mutex);
  return vfile_find_shared(g_vdev_list, vdev);
}

template <typename FnT>
static inline void
vdev_set_root_callback(pim_vdev_root_t *root, FnT pim_vdev_root_t::*root_member,
                       FnT pim_vdev_t::*instance_member, FnT fn) noexcept {
  std::shared_ptr<pim_vdev_root_t> root_shared =
      vfile_lookup_vdev_root_shared(root);
  if (!root_shared) {
    return;
  }

  std::vector<std::shared_ptr<pim_vdev_t>> instances;
  const bool snapshot_ok = vfile_try_or_false_with_errno(
      [&]() {
        std::lock_guard<std::mutex> lock(root_shared->mutex);
        root_shared.get()->*root_member = fn;
        instances.reserve(root_shared->instances.size());
        for (auto it = root_shared->instances.begin();
             it != root_shared->instances.end();) {
          if (auto v = it->lock()) {
            instances.push_back(std::move(v));
            ++it;
          } else {
            it = root_shared->instances.erase(it);
          }
        }
      },
      ENOMEM);
  if (!snapshot_ok) {
    std::lock_guard<std::mutex> lock(root_shared->mutex);
    root_shared.get()->*root_member = fn;
    return;
  }

  for (const auto &instance : instances) {
    std::lock_guard<std::mutex> lock(instance->mutex);
    instance.get()->*instance_member = fn;
  }
}

bool vdev_glob_match(const char *pattern, const char *str) {
  if (!pattern || !str) {
    return false;
  }

  const char *p = pattern;
  const char *s = str;
  const char *star_pattern = nullptr;
  const char *star_str = nullptr;

  while (*s != '\0') {
    if (*p == '*') {
      while (*p == '*') {
        ++p;
      }
      star_pattern = p;
      star_str = s;
      if (*p == '\0') {
        return true;
      }
      continue;
    }

    if ((*p == '?') || (*p == *s)) {
      ++p;
      ++s;
      continue;
    }

    if (!star_pattern) {
      return false;
    }

    p = star_pattern;
    s = ++star_str;
  }

  while (*p == '*') {
    ++p;
  }
  return (*p == '\0');
}

std::vector<std::shared_ptr<pim_vdev_t>> vfile_copy_vdev_list_for_snapshot() {
  std::lock_guard<std::mutex> lock(g_vdev_list_mutex);
  return g_vdev_list;
}

void vfile_clear_vdev_state() noexcept {
  {
    std::lock_guard<std::mutex> hold_lock(g_vdev_acquire_mutex);
    g_vdev_acquire_map.clear();
  }

  std::lock(g_vdev_list_mutex, g_vdev_root_list_mutex);
  std::lock_guard<std::mutex> list_lock(g_vdev_list_mutex, std::adopt_lock);
  std::lock_guard<std::mutex> root_lock(g_vdev_root_list_mutex,
                                        std::adopt_lock);
  g_vdev_list.clear();
  g_vdev_root_list.clear();
}

pim_vdev_root_t *pim_vdev_root_create(const char *name_glob) noexcept {
  if (!name_glob) {
    return nullptr;
  }

  return vdev_try_or_nullptr<pim_vdev_root_t>([&]() -> pim_vdev_root_t * {
    auto root = std::make_shared<pim_vdev_root_t>();
    root->name_glob = vdev_normalize_name_glob(std::string(name_glob));
    if (root->name_glob.empty()) {
      return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_vdev_root_list_mutex);
    g_vdev_root_list.push_back(root);
    return root.get();
  });
}

void pim_vdev_root_destroy(pim_vdev_root_t *root) noexcept {
  if (!root) {
    return;
  }

  std::lock_guard<std::mutex> lock(g_vdev_root_list_mutex);
  g_vdev_root_list.erase(
      std::remove_if(g_vdev_root_list.begin(), g_vdev_root_list.end(),
                     [root](const std::shared_ptr<pim_vdev_root_t> &entry) {
                       return !entry || entry.get() == root;
                     }),
      g_vdev_root_list.end());
}

const char *pim_vdev_root_get_name_glob(pim_vdev_root_t *root) noexcept {
  if (!root) {
    return nullptr;
  }
  return root->name_glob.c_str();
}

void pim_vdev_root_set_open_cb(pim_vdev_root_t *root,
                               pim_vdev_open_fn fn) noexcept {
  vdev_set_root_callback(root, &pim_vdev_root_t::open_fn, &pim_vdev_t::open_fn,
                         fn);
}

void pim_vdev_root_set_ioctl_cb(pim_vdev_root_t *root,
                                pim_vdev_ioctl_fn fn) noexcept {
  vdev_set_root_callback(root, &pim_vdev_root_t::ioctl_fn,
                         &pim_vdev_t::ioctl_fn, fn);
}

void pim_vdev_root_set_mmap_cb(pim_vdev_root_t *root,
                               pim_vdev_mmap_fn fn) noexcept {
  vdev_set_root_callback(root, &pim_vdev_root_t::mmap_fn, &pim_vdev_t::mmap_fn,
                         fn);
}

void pim_vdev_root_set_close_cb(pim_vdev_root_t *root,
                                pim_vdev_close_fn fn) noexcept {
  vdev_set_root_callback(root, &pim_vdev_root_t::close_fn,
                         &pim_vdev_t::close_fn, fn);
}

pim_vdev_t *pim_vdev_create(pim_vdev_root_t *root, const char *name, int major,
                            int minor) noexcept {
  if (!root || !name || major < -1 || minor < -1) {
    return nullptr;
  }

  const std::shared_ptr<pim_vdev_root_t> root_shared =
      vfile_lookup_vdev_root_shared(root);
  if (!root_shared) {
    return nullptr;
  }

  return vdev_try_or_nullptr<pim_vdev_t>([&]() -> pim_vdev_t * {
    std::string normalized_name(name);
    normalized_name =
        vfile_strip_trailing_slashes_preserve_root(std::move(normalized_name));
    normalized_name =
        vfile_strip_known_prefix(std::move(normalized_name), "/dev/");
    if (!vdev_valid_instance_name(normalized_name)) {
      return nullptr;
    }
    const std::string path = "/dev/" + normalized_name;

    auto instance = std::make_shared<pim_vdev_t>();
    instance->root = root_shared;
    instance->name = normalized_name;
    instance->path = path;

    {
      std::lock_guard<std::mutex> lock(g_vdev_list_mutex);
      std::unordered_set<uint64_t> used_devnums;
      used_devnums.reserve(g_vdev_list.size());
      bool has_used_devnum = false;
      unsigned int max_major = 0U;
      unsigned int max_minor = 0U;

      for (const auto &existing : g_vdev_list) {
        if (!existing) {
          continue;
        }
        std::lock_guard<std::mutex> existing_lock(existing->mutex);
        if (existing->path == path) {
          return nullptr;
        }

        used_devnums.insert(
            vdev_make_devnum_key(existing->major, existing->minor));
        if (!has_used_devnum || existing->major > max_major ||
            (existing->major == max_major && existing->minor > max_minor)) {
          has_used_devnum = true;
          max_major = existing->major;
          max_minor = existing->minor;
        }
      }

      unsigned int resolved_major = 0U;
      unsigned int resolved_minor = 0U;
      if (major >= 0) {
        resolved_major = static_cast<unsigned int>(major);
      }
      if (minor >= 0) {
        resolved_minor = static_cast<unsigned int>(minor);
      }

      const auto is_used = [&](unsigned int candidate_major,
                               unsigned int candidate_minor) noexcept -> bool {
        return used_devnums.find(vdev_make_devnum_key(
                   candidate_major, candidate_minor)) != used_devnums.end();
      };

      if (major >= 0 && minor == -1) {
        resolved_minor = 0U;
        while (is_used(resolved_major, resolved_minor)) {
          if (resolved_minor == std::numeric_limits<unsigned int>::max()) {
            return nullptr;
          }
          ++resolved_minor;
        }
      } else if (major == -1 && minor >= 0) {
        resolved_major = 0U;
        while (is_used(resolved_major, resolved_minor)) {
          if (resolved_major == std::numeric_limits<unsigned int>::max()) {
            return nullptr;
          }
          ++resolved_major;
        }
      } else if (major == -1 && minor == -1) {
        if (has_used_devnum) {
          resolved_major = max_major;
          resolved_minor = max_minor;
          if (!vdev_increment_devnum_pair(&resolved_major, &resolved_minor)) {
            return nullptr;
          }
        }
        while (is_used(resolved_major, resolved_minor)) {
          if (!vdev_increment_devnum_pair(&resolved_major, &resolved_minor)) {
            return nullptr;
          }
        }
      }

      instance->major = resolved_major;
      instance->minor = resolved_minor;
      g_vdev_list.push_back(instance);
    }

    const bool root_linked = vfile_try_or_false_with_errno(
        [&]() {
          std::lock_guard<std::mutex> lock(root_shared->mutex);
          instance->open_fn = root_shared->open_fn;
          instance->ioctl_fn = root_shared->ioctl_fn;
          instance->mmap_fn = root_shared->mmap_fn;
          instance->close_fn = root_shared->close_fn;
          root_shared->instances.push_back(instance);
        },
        ENOMEM);
    if (!root_linked) {
      std::lock_guard<std::mutex> lock(g_vdev_list_mutex);
      g_vdev_list.erase(
          std::remove_if(g_vdev_list.begin(), g_vdev_list.end(),
                         [&instance](const std::shared_ptr<pim_vdev_t> &entry) {
                           return !entry || entry.get() == instance.get();
                         }),
          g_vdev_list.end());
      return nullptr;
    }

    vfile_notify_virtual_fs_changed();
    return instance.get();
  });
}

void pim_vdev_destroy(pim_vdev_t *vdev) noexcept {
  if (!vdev) {
    return;
  }

  if (auto root_shared = vdev->root.lock()) {
    std::lock_guard<std::mutex> lock(root_shared->mutex);
    root_shared->instances.erase(
        std::remove_if(root_shared->instances.begin(),
                       root_shared->instances.end(),
                       [vdev](const std::weak_ptr<pim_vdev_t> &wk) {
                         auto sp = wk.lock();
                         return !sp || sp.get() == vdev;
                       }),
        root_shared->instances.end());
  }

  {
    std::lock_guard<std::mutex> lock(g_vdev_list_mutex);
    g_vdev_list.erase(
        std::remove_if(g_vdev_list.begin(), g_vdev_list.end(),
                       [vdev](const std::shared_ptr<pim_vdev_t> &entry) {
                         return !entry || entry.get() == vdev;
                       }),
        g_vdev_list.end());
  }

  vfile_notify_virtual_fs_changed();
}

const char *pim_vdev_get_name(pim_vdev_t *vdev) noexcept {
  if (!vdev) {
    return nullptr;
  }
  return vdev->name.c_str();
}

const char *pim_vdev_get_path(pim_vdev_t *vdev) noexcept {
  if (!vdev) {
    return nullptr;
  }
  return vdev->path.c_str();
}

int pim_vdev_get_major(pim_vdev_t *vdev) noexcept {
  if (!vdev || vdev->major >
                   static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    return -1;
  }
  return static_cast<int>(vdev->major);
}

int pim_vdev_get_minor(pim_vdev_t *vdev) noexcept {
  if (!vdev || vdev->minor >
                   static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    return -1;
  }
  return static_cast<int>(vdev->minor);
}

pim_vdev_root_t *pim_vdev_get_root(pim_vdev_t *vdev) noexcept {
  if (!vdev) {
    return nullptr;
  }
  auto root = vdev->root.lock();
  return root ? root.get() : nullptr;
}

void pim_vdev_set_user_data(pim_vdev_t *vdev, void *user_data) noexcept {
  if (!vdev) {
    return;
  }
  std::lock_guard<std::mutex> lock(vdev->mutex);
  vdev->user_data = user_data;
}

void *pim_vdev_get_user_data(pim_vdev_t *vdev) noexcept {
  if (!vdev) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(vdev->mutex);
  return vdev->user_data;
}

pim_error_t pim_vdev_lock(pim_vdev_t *vdev) noexcept {
  if (!vdev) {
    return PIM_ERR_INVALID_ARG;
  }

  {
    std::lock_guard<std::mutex> hold_lock(g_vdev_acquire_mutex);
    auto it = g_vdev_acquire_map.find(vdev);
    if (it != g_vdev_acquire_map.end()) {
      if (it->second.refcount == std::numeric_limits<size_t>::max()) {
        return PIM_ERR_BUSY;
      }
      ++it->second.refcount;
      return PIM_SUCCESS;
    }
  }

  std::shared_ptr<pim_vdev_t> shared_vdev;
  {
    std::lock_guard<std::mutex> list_lock(g_vdev_list_mutex);
    shared_vdev = vfile_find_shared(g_vdev_list, vdev);
  }
  if (!shared_vdev) {
    return PIM_ERR_NOT_FOUND;
  }

  try {
    std::lock_guard<std::mutex> hold_lock(g_vdev_acquire_mutex);
    auto [it, inserted] = g_vdev_acquire_map.try_emplace(vdev);
    if (inserted || it->second.refcount == 0) {
      it->second.hold = std::move(shared_vdev);
      it->second.refcount = 1;
      return PIM_SUCCESS;
    }
    if (it->second.refcount == std::numeric_limits<size_t>::max()) {
      return PIM_ERR_BUSY;
    }
    ++it->second.refcount;
  } catch (...) {
    return PIM_ERR_NO_MEMORY;
  }

  return PIM_SUCCESS;
}

pim_error_t pim_vdev_unlock(pim_vdev_t *vdev) noexcept {
  if (!vdev) {
    return PIM_ERR_INVALID_ARG;
  }

  std::lock_guard<std::mutex> hold_lock(g_vdev_acquire_mutex);
  auto it = g_vdev_acquire_map.find(vdev);
  if (it == g_vdev_acquire_map.end() || it->second.refcount == 0) {
    return PIM_ERR_NOT_FOUND;
  }

  --it->second.refcount;
  if (it->second.refcount == 0) {
    g_vdev_acquire_map.erase(it);
  }

  return PIM_SUCCESS;
}

pim_error_t pim_vdev_register_device(pim_vdev_t *vdev, int fd,
                                     pim_device_t *dev) noexcept {
  if (!vdev || fd < 0 || !dev) {
    return PIM_ERR_INVALID_ARG;
  }
  {
    std::lock_guard<std::mutex> lock(vdev->mutex);
    try {
      vdev->device_by_fd[fd] = dev;
    } catch (...) {
      return PIM_ERR_NO_MEMORY;
    }
  }
  vfile_on_vdev_device_mapping_changed(vdev);
  return PIM_SUCCESS;
}

pim_error_t pim_vdev_unregister_device(pim_vdev_t *vdev, int fd) noexcept {
  if (!vdev || fd < 0) {
    return PIM_ERR_INVALID_ARG;
  }
  {
    std::lock_guard<std::mutex> lock(vdev->mutex);
    auto it = vdev->device_by_fd.find(fd);
    if (it == vdev->device_by_fd.end()) {
      return PIM_ERR_NOT_FOUND;
    }
    vdev->device_by_fd.erase(it);
  }
  vfile_on_vdev_device_mapping_changed(vdev);
  return PIM_SUCCESS;
}

pim_device_t *pim_vdev_find_device(pim_vdev_t *vdev, int fd) noexcept {
  if (!vdev || fd < 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(vdev->mutex);
  auto it = vdev->device_by_fd.find(fd);
  return (it == vdev->device_by_fd.end()) ? nullptr : it->second;
}
