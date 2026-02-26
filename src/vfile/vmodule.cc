/**
 * @file vmodule.cc
 * @brief Virtual /sys/module instance lifecycle and attribute storage
 *
 * This translation unit provides virtual module metadata under
 * /sys/module/<name> and exposes snapshot source data used by vfile cache.
 */

#include "vfile.hh"

#include <algorithm>

static std::mutex g_vmodule_list_mutex;
static std::vector<std::shared_ptr<pim_vmodule_t>> g_vmodule_list;

template <typename Fn>
static inline pim_error_t vmodule_try_or_pim_no_memory(Fn &&fn) noexcept {
  try {
    fn();
    return PIM_SUCCESS;
  } catch (...) {
    return PIM_ERR_NO_MEMORY;
  }
}

template <typename T, typename Fn>
static inline T *vmodule_try_or_nullptr(Fn &&fn) noexcept {
  try {
    return fn();
  } catch (...) {
    return nullptr;
  }
}

static inline bool
vmodule_attr_access_valid(pim_access_mode_t access) noexcept {
  switch (access) {
  case PIM_ACCESS_MODE_NONE:
  case PIM_ACCESS_MODE_RO:
  case PIM_ACCESS_MODE_RW:
  case PIM_ACCESS_MODE_WO:
    return true;
  }
  return false;
}

static inline void vmodule_bump_attr_revision(pim_vmodule_t *vmodule) noexcept {
  if (!vmodule) {
    return;
  }
  (void)vfile_bump_nonzero_revision(vmodule->attrs_revision);
}

static inline bool
vmodule_valid_instance_name(const std::string &name) noexcept {
  if (name.empty() || name.find('/') != std::string::npos) {
    return false;
  }
  return !vfile_contains_glob_wildcards(name);
}

static inline std::string
vmodule_normalize_instance_name(std::string name) noexcept {
  name = vfile_strip_trailing_slashes_preserve_root(std::move(name));
  name = vfile_strip_known_prefix(std::move(name), "/sys/module/");
  if (!name.empty() && name.front() == '/') {
    name.erase(0, 1);
  }
  return name;
}

template <typename T>
static inline void
vmodule_unregister_object(std::vector<std::shared_ptr<T>> &list,
                          std::mutex &list_mutex, T *raw) noexcept {
  if (!raw) {
    return;
  }
  std::lock_guard<std::mutex> lock(list_mutex);
  list.erase(std::remove_if(list.begin(), list.end(),
                            [raw](const std::shared_ptr<T> &entry) {
                              return !entry || entry.get() == raw;
                            }),
             list.end());
}

std::vector<std::shared_ptr<pim_vmodule_t>>
vfile_copy_vmodule_list_for_snapshot() {
  std::lock_guard<std::mutex> lock(g_vmodule_list_mutex);
  return g_vmodule_list;
}

void vfile_clear_vmodule_state() noexcept {
  std::lock_guard<std::mutex> lock(g_vmodule_list_mutex);
  g_vmodule_list.clear();
}

pim_vmodule_t *pim_vmodule_create(const char *module_name) noexcept {
  if (!module_name) {
    return nullptr;
  }

  return vmodule_try_or_nullptr<pim_vmodule_t>([&]() -> pim_vmodule_t * {
    std::string normalized_name(module_name);
    normalized_name =
        vmodule_normalize_instance_name(std::move(normalized_name));
    if (!vmodule_valid_instance_name(normalized_name)) {
      return nullptr;
    }

    const std::string dir = "/sys/module/" + normalized_name;
    auto vmodule = std::make_shared<pim_vmodule_t>();
    vmodule->name = normalized_name;
    vmodule->dir = dir;

    {
      std::lock_guard<std::mutex> lock(g_vmodule_list_mutex);
      for (const auto &existing : g_vmodule_list) {
        if (!existing) {
          continue;
        }
        std::lock_guard<std::mutex> existing_lock(existing->mutex);
        if (existing->dir == dir) {
          return nullptr;
        }
      }
      g_vmodule_list.push_back(vmodule);
    }

    vfile_notify_virtual_fs_changed();
    return vmodule.get();
  });
}

void pim_vmodule_destroy(pim_vmodule_t *vmodule) noexcept {
  vmodule_unregister_object(g_vmodule_list, g_vmodule_list_mutex, vmodule);
  vfile_notify_virtual_fs_changed();
}

const char *pim_vmodule_get_name(pim_vmodule_t *vmodule) noexcept {
  if (!vmodule) {
    return nullptr;
  }
  return vmodule->name.c_str();
}

const char *pim_vmodule_get_dir(pim_vmodule_t *vmodule) noexcept {
  if (!vmodule) {
    return nullptr;
  }
  return vmodule->dir.c_str();
}

pim_error_t pim_vmodule_create_attr(pim_vmodule_t *vmodule, const char *name,
                                    const char *value,
                                    pim_access_mode_t access) noexcept {
  if (!vmodule || !name || !value || !vmodule_attr_access_valid(access)) {
    return PIM_ERR_INVALID_ARG;
  }

  bool had_attr = false;
  bool access_changed = false;
  const pim_error_t set_ret = vmodule_try_or_pim_no_memory([&]() {
    std::lock_guard<std::mutex> lock(vmodule->mutex);
    auto old_it = vmodule->attrs.find(name);
    had_attr = (old_it != vmodule->attrs.end());
    if (had_attr) {
      auto access_it = vmodule->attr_access.find(name);
      const pim_access_mode_t old_access =
          (access_it != vmodule->attr_access.end())
              ? access_it->second
              : PIM_ACCESS_MODE_RW;
      access_changed = (old_access != access);
    }
    vmodule->attrs[name] = value;
    vmodule->attr_access[name] = access;
    vmodule_bump_attr_revision(vmodule);
  });
  if (set_ret != PIM_SUCCESS) {
    return set_ret;
  }

  if (!had_attr || access_changed) {
    vfile_notify_virtual_fs_changed();
  }
  return PIM_SUCCESS;
}

pim_error_t pim_vmodule_remove_attr(pim_vmodule_t *vmodule,
                                    const char *name) noexcept {
  if (!vmodule || !name) {
    return PIM_ERR_INVALID_ARG;
  }

  std::lock_guard<std::mutex> lock(vmodule->mutex);
  auto it = vmodule->attrs.find(name);
  if (it == vmodule->attrs.end()) {
    return PIM_ERR_NOT_FOUND;
  }
  vmodule->attrs.erase(it);
  vmodule->attr_access.erase(name);
  vmodule_bump_attr_revision(vmodule);
  vfile_notify_virtual_fs_changed();
  return PIM_SUCCESS;
}

const char *pim_vmodule_find_attr(pim_vmodule_t *vmodule,
                                  const char *name) noexcept {
  if (!vmodule || !name) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(vmodule->mutex);
  auto it = vmodule->attrs.find(name);
  if (it == vmodule->attrs.end()) {
    return nullptr;
  }
  static thread_local std::string tls_value;
  const pim_error_t copy_ret =
      vmodule_try_or_pim_no_memory([&]() { tls_value = it->second; });
  if (copy_ret != PIM_SUCCESS) {
    return nullptr;
  }
  return tls_value.c_str();
}
