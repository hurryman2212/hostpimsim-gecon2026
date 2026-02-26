/**
 * @file vsysfs.cc
 * @brief Virtual sysfs root/instance lifecycle and attachment mapping
 */

#include "vfile.hh"

#include <algorithm>
#include <mutex>

static std::mutex g_vsysfs_list_mutex;
static std::vector<std::shared_ptr<pim_vsysfs_t>> g_vsysfs_list;

static std::mutex g_vsysfs_root_list_mutex;
static std::vector<std::shared_ptr<pim_vsysfs_root_t>> g_vsysfs_root_list;

static std::mutex g_device_vsysfs_map_mutex;
static std::unordered_map<pim_device_t *,
                          std::vector<std::weak_ptr<pim_vsysfs_t>>>
    g_device_vsysfs_map;

template <typename Fn>
static inline void vsysfs_run_best_effort(Fn &&fn) noexcept {
  try {
    fn();
  } catch (...) {
    // Best effort only.
  }
}

template <typename Fn>
static inline pim_error_t vsysfs_try_or_pim_no_memory(Fn &&fn) noexcept {
  try {
    fn();
    return PIM_SUCCESS;
  } catch (...) {
    return PIM_ERR_NO_MEMORY;
  }
}

template <typename T, typename Fn>
static inline T *vsysfs_try_or_nullptr(Fn &&fn) noexcept {
  try {
    return fn();
  } catch (...) {
    return nullptr;
  }
}

static inline bool vsysfs_attr_access_valid(pim_access_mode_t access) noexcept {
  switch (access) {
  case PIM_ACCESS_MODE_NONE:
  case PIM_ACCESS_MODE_RO:
  case PIM_ACCESS_MODE_RW:
  case PIM_ACCESS_MODE_WO:
    return true;
  }
  return false;
}

static inline pim_access_mode_t vsysfs_default_attr_access_mode() noexcept {
  return PIM_ACCESS_MODE_RW;
}

static inline bool
vsysfs_valid_instance_name(const std::string &name) noexcept {
  if (name.empty() || name.find('/') != std::string::npos) {
    return false;
  }
  return !vfile_contains_glob_wildcards(name);
}

static inline void vsysfs_bump_attr_revision(pim_vsysfs_t *vsysfs) noexcept;

static inline std::string
vsysfs_normalize_component(std::string value) noexcept {
  value = vfile_strip_trailing_slashes_preserve_root(std::move(value));
  if (!value.empty() && value.front() == '/') {
    value.erase(0, 1);
  }
  return value;
}

static inline bool vsysfs_upsert_parent_device_attrs_locked(
    const std::shared_ptr<pim_vsysfs_t> &target, const std::string &class_dir,
    const std::string &devices_dir, const std::string &subsystem_value,
    const std::string &device_value, const std::string &uevent_value) noexcept {
  if (!target) {
    return false;
  }
  bool changed = false;
  std::lock_guard<std::mutex> lock(target->mutex);
  if (target->class_dir.empty() && !class_dir.empty()) {
    target->class_dir = class_dir;
    changed = true;
  }
  if (target->devices_dir.empty() && !devices_dir.empty()) {
    target->devices_dir = devices_dir;
    changed = true;
  }
  auto subsystem_it = target->attrs.find("subsystem");
  if (subsystem_it == target->attrs.end()) {
    target->attrs.emplace("subsystem", subsystem_value);
    target->attr_access.emplace("subsystem", PIM_ACCESS_MODE_RO);
    changed = true;
  }
  auto device_it = target->attrs.find("device");
  if (device_it == target->attrs.end()) {
    target->attrs.emplace("device", device_value);
    target->attr_access.emplace("device", PIM_ACCESS_MODE_RO);
    changed = true;
  }
  auto uevent_it = target->attrs.find("uevent");
  if (uevent_it == target->attrs.end()) {
    target->attrs.emplace("uevent", uevent_value);
    target->attr_access.emplace("uevent", PIM_ACCESS_MODE_RO);
    changed = true;
  }
  if (changed) {
    vsysfs_bump_attr_revision(target.get());
  }
  return changed;
}

static inline bool vsysfs_ensure_parent_device_node(
    const std::shared_ptr<pim_vsysfs_root_t> &root_shared,
    const std::string &parent_devices_dir,
    const std::string &instance_name) noexcept {
  if (!root_shared || parent_devices_dir.empty() || instance_name.empty()) {
    return false;
  }

  std::string subsystem_value;
  std::string class_dir;
  std::string device_value;
  std::string uevent_value;
  try {
    subsystem_value = "/sys/class/dpu_region";
    class_dir = "/sys/class/dpu_region/" + instance_name;
    device_value = parent_devices_dir;
    uevent_value = std::string("DEVPATH=/devices/") + root_shared->bus_name +
                   "/" + instance_name +
                   "\nSUBSYSTEM=dpu_region\nDEVNAME=" + instance_name + "\n";
  } catch (...) {
    return false;
  }

  std::shared_ptr<pim_vsysfs_t> existing;
  {
    std::lock_guard<std::mutex> lock(g_vsysfs_list_mutex);
    for (const auto &candidate : g_vsysfs_list) {
      if (!candidate) {
        continue;
      }
      std::lock_guard<std::mutex> candidate_lock(candidate->mutex);
      if (candidate->devices_dir == parent_devices_dir) {
        existing = candidate;
        break;
      }
    }

    if (!existing) {
      try {
        auto parent = std::make_shared<pim_vsysfs_t>();
        parent->root = root_shared;
        parent->class_name = instance_name;
        parent->device_name = instance_name;
        parent->class_dir = class_dir;
        parent->devices_dir = parent_devices_dir;
        parent->attrs.emplace("subsystem", subsystem_value);
        parent->attr_access.emplace("subsystem", PIM_ACCESS_MODE_RO);
        parent->attrs.emplace("device", device_value);
        parent->attr_access.emplace("device", PIM_ACCESS_MODE_RO);
        parent->attrs.emplace("uevent", uevent_value);
        parent->attr_access.emplace("uevent", PIM_ACCESS_MODE_RO);
        g_vsysfs_list.push_back(parent);
      } catch (...) {
        return false;
      }
      return true;
    }
  }

  return vsysfs_upsert_parent_device_attrs_locked(
      existing, class_dir, parent_devices_dir, subsystem_value, device_value,
      uevent_value);
}

static inline std::string
vsysfs_normalize_class_component(std::string value) noexcept {
  value = vfile_strip_known_prefix(std::move(value), "/sys/class/");
  return vsysfs_normalize_component(std::move(value));
}

static inline std::string
vsysfs_normalize_devices_component(std::string value) noexcept {
  value = vfile_strip_known_prefix(std::move(value), "/sys/devices/");
  return vsysfs_normalize_component(std::move(value));
}

static inline bool
vsysfs_root_component_valid(const std::string &value) noexcept {
  return !value.empty() && value.find('*') == std::string::npos &&
         value.find('?') == std::string::npos;
}

static inline std::shared_ptr<pim_vsysfs_root_t>
vsysfs_lookup_root_shared(pim_vsysfs_root_t *root) {
  std::lock_guard<std::mutex> lock(g_vsysfs_root_list_mutex);
  return vfile_find_shared(g_vsysfs_root_list, root);
}

static inline void vsysfs_bump_attr_revision(pim_vsysfs_t *vsysfs) noexcept {
  if (!vsysfs) {
    return;
  }
  (void)vfile_bump_nonzero_revision(vsysfs->attrs_revision);
}

template <typename T>
static inline void
vsysfs_unregister_object(std::vector<std::shared_ptr<T>> &list,
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

std::vector<std::shared_ptr<pim_vsysfs_t>>
vfile_copy_vsysfs_list_for_snapshot() {
  std::lock_guard<std::mutex> lock(g_vsysfs_list_mutex);
  return g_vsysfs_list;
}

void vfile_clear_vsysfs_state() noexcept {
  std::lock(g_vsysfs_list_mutex, g_vsysfs_root_list_mutex,
            g_device_vsysfs_map_mutex);
  std::lock_guard<std::mutex> list_lock(g_vsysfs_list_mutex, std::adopt_lock);
  std::lock_guard<std::mutex> root_lock(g_vsysfs_root_list_mutex,
                                        std::adopt_lock);
  std::lock_guard<std::mutex> map_lock(g_device_vsysfs_map_mutex,
                                       std::adopt_lock);
  g_device_vsysfs_map.clear();
  g_vsysfs_list.clear();
  g_vsysfs_root_list.clear();
}

/*----------------------------------------------------------------------------
 * vsysfs root
 *----------------------------------------------------------------------------*/

pim_vsysfs_root_t *
pim_vsysfs_root_create(pim_vdev_root_t *vdev_root, const char *class_name,
                       const char *bus_name,
                       const char *instance_name_glob) noexcept {
  if (!vdev_root || !class_name || !bus_name || !instance_name_glob) {
    return nullptr;
  }

  std::shared_ptr<pim_vdev_root_t> vdev_root_shared =
      vfile_lookup_vdev_root_shared(vdev_root);
  if (!vdev_root_shared) {
    return nullptr;
  }

  return vsysfs_try_or_nullptr<pim_vsysfs_root_t>([&]() -> pim_vsysfs_root_t * {
    auto root = std::make_shared<pim_vsysfs_root_t>();

    std::string vdev_name_glob;
    {
      std::lock_guard<std::mutex> vdev_root_lock(vdev_root_shared->mutex);
      vdev_name_glob = vdev_root_shared->name_glob;
    }

    root->vdev_root = vdev_root_shared;
    root->class_name =
        vsysfs_normalize_class_component(std::string(class_name));
    root->bus_name = vsysfs_normalize_devices_component(std::string(bus_name));
    root->instance_name_glob =
        vsysfs_normalize_component(std::string(instance_name_glob));

    if (!vsysfs_root_component_valid(root->class_name) ||
        !vsysfs_root_component_valid(root->bus_name) ||
        root->instance_name_glob.empty() || vdev_name_glob.empty()) {
      return nullptr;
    }

    root->class_glob = root->class_name + "/" + vdev_name_glob;
    root->devices_glob = root->bus_name + "/" + root->instance_name_glob;

    std::lock_guard<std::mutex> lock(g_vsysfs_root_list_mutex);
    g_vsysfs_root_list.push_back(root);
    return root.get();
  });
}

void pim_vsysfs_root_destroy(pim_vsysfs_root_t *root) noexcept {
  vsysfs_unregister_object(g_vsysfs_root_list, g_vsysfs_root_list_mutex, root);
}

const char *pim_vsysfs_root_get_class_glob(pim_vsysfs_root_t *root) noexcept {
  if (!root) {
    return nullptr;
  }
  return root->class_glob.c_str();
}

const char *pim_vsysfs_root_get_devices_glob(pim_vsysfs_root_t *root) noexcept {
  if (!root) {
    return nullptr;
  }
  return root->devices_glob.c_str();
}

pim_error_t
pim_vsysfs_root_create_default_attr(pim_vsysfs_root_t *root, const char *name,
                                    const char *value,
                                    pim_access_mode_t access) noexcept {
  if (!root || !name || !value || !vsysfs_attr_access_valid(access)) {
    return PIM_ERR_INVALID_ARG;
  }
  std::shared_ptr<pim_vsysfs_root_t> root_shared =
      vsysfs_lookup_root_shared(root);
  if (!root_shared) {
    return PIM_ERR_INVALID_ARG;
  }

  bool root_had_attr = false;
  std::string root_old_value;
  pim_access_mode_t root_old_access = vsysfs_default_attr_access_mode();
  std::vector<std::shared_ptr<pim_vsysfs_t>> instances;
  {
    std::lock_guard<std::mutex> lock(root_shared->mutex);
    auto root_it = root_shared->default_attrs.find(name);
    root_had_attr = (root_it != root_shared->default_attrs.end());
    if (root_had_attr) {
      const pim_error_t copy_ret = vsysfs_try_or_pim_no_memory(
          [&]() { root_old_value = root_it->second; });
      if (copy_ret != PIM_SUCCESS) {
        return copy_ret;
      }
      auto access_it = root_shared->default_attr_access.find(name);
      if (access_it != root_shared->default_attr_access.end()) {
        root_old_access = access_it->second;
      }
    }

    const pim_error_t collect_ret = vsysfs_try_or_pim_no_memory([&]() {
      instances.reserve(root_shared->instances.size());
      for (auto it = root_shared->instances.begin();
           it != root_shared->instances.end();) {
        if (auto vs = it->lock()) {
          instances.push_back(std::move(vs));
          ++it;
        } else {
          it = root_shared->instances.erase(it);
        }
      }
    });
    if (collect_ret != PIM_SUCCESS) {
      return collect_ret;
    }
  }

  struct vsysfs_prev_state {
    std::shared_ptr<pim_vsysfs_t> vs;
    bool had_attr{false};
    std::string old_value;
    pim_access_mode_t old_access{PIM_ACCESS_MODE_RW};
  };
  std::vector<vsysfs_prev_state> prev_states;
  const pim_error_t reserve_prev_ret = vsysfs_try_or_pim_no_memory(
      [&]() { prev_states.reserve(instances.size()); });
  if (reserve_prev_ret != PIM_SUCCESS) {
    return reserve_prev_ret;
  }

  for (const auto &vs : instances) {
    vsysfs_prev_state state;
    state.vs = vs;
    {
      std::lock_guard<std::mutex> lock(vs->mutex);
      auto it = vs->attrs.find(name);
      state.had_attr = (it != vs->attrs.end());
      if (state.had_attr) {
        const pim_error_t copy_ret = vsysfs_try_or_pim_no_memory(
            [&]() { state.old_value = it->second; });
        if (copy_ret != PIM_SUCCESS) {
          return copy_ret;
        }
        auto access_it = vs->attr_access.find(name);
        state.old_access = (access_it != vs->attr_access.end())
                               ? access_it->second
                               : vsysfs_default_attr_access_mode();
      }
    }
    const pim_error_t push_ret = vsysfs_try_or_pim_no_memory(
        [&]() { prev_states.push_back(std::move(state)); });
    if (push_ret != PIM_SUCCESS) {
      return push_ret;
    }
  }

  auto rollback_root = [&]() noexcept {
    std::lock_guard<std::mutex> lock(root_shared->mutex);
    if (root_had_attr) {
      vsysfs_run_best_effort([&]() {
        root_shared->default_attrs[name] = root_old_value;
        root_shared->default_attr_access[name] = root_old_access;
      });
    } else {
      root_shared->default_attrs.erase(name);
      root_shared->default_attr_access.erase(name);
    }
  };

  auto rollback_instances = [&](size_t applied_count) noexcept {
    for (size_t i = 0; i < applied_count; ++i) {
      const auto &state = prev_states[i];
      if (!state.vs) {
        continue;
      }
      std::lock_guard<std::mutex> lock(state.vs->mutex);
      if (state.had_attr) {
        vsysfs_run_best_effort([&]() {
          state.vs->attrs[name] = state.old_value;
          state.vs->attr_access[name] = state.old_access;
        });
      } else {
        state.vs->attrs.erase(name);
        state.vs->attr_access.erase(name);
      }
      vsysfs_bump_attr_revision(state.vs.get());
    }
  };

  {
    std::lock_guard<std::mutex> lock(root_shared->mutex);
    const pim_error_t root_set_ret = vsysfs_try_or_pim_no_memory([&]() {
      root_shared->default_attrs[name] = value;
      root_shared->default_attr_access[name] = access;
    });
    if (root_set_ret != PIM_SUCCESS) {
      if (root_had_attr) {
        vsysfs_run_best_effort([&]() {
          root_shared->default_attrs[name] = root_old_value;
          root_shared->default_attr_access[name] = root_old_access;
        });
      } else {
        root_shared->default_attrs.erase(name);
        root_shared->default_attr_access.erase(name);
      }
      return root_set_ret;
    }
  }

  size_t applied_instances = 0;
  for (; applied_instances < prev_states.size(); ++applied_instances) {
    auto &state = prev_states[applied_instances];
    std::lock_guard<std::mutex> lock(state.vs->mutex);
    const pim_error_t set_ret = vsysfs_try_or_pim_no_memory([&]() {
      state.vs->attrs[name] = value;
      state.vs->attr_access[name] = access;
    });
    if (set_ret != PIM_SUCCESS) {
      rollback_instances(applied_instances + 1U);
      rollback_root();
      return set_ret;
    }
  }

  bool any_new_attr = false;
  for (const auto &state : prev_states) {
    any_new_attr = any_new_attr || !state.had_attr;
    vsysfs_bump_attr_revision(state.vs.get());
  }
  if (any_new_attr) {
    vfile_notify_virtual_fs_changed();
  }
  return PIM_SUCCESS;
}

const char *pim_vsysfs_root_find_default_attr(pim_vsysfs_root_t *root,
                                              const char *name) noexcept {
  if (!root || !name) {
    return nullptr;
  }
  std::shared_ptr<pim_vsysfs_root_t> root_shared =
      vsysfs_lookup_root_shared(root);
  if (!root_shared) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(root_shared->mutex);
  auto it = root_shared->default_attrs.find(name);
  if (it == root_shared->default_attrs.end()) {
    return nullptr;
  }
  static thread_local std::string tls_value;
  const pim_error_t copy_ret =
      vsysfs_try_or_pim_no_memory([&]() { tls_value = it->second; });
  if (copy_ret != PIM_SUCCESS) {
    return nullptr;
  }
  return tls_value.c_str();
}

pim_error_t pim_vsysfs_root_remove_default_attr(pim_vsysfs_root_t *root,
                                                const char *name) noexcept {
  if (!root || !name) {
    return PIM_ERR_INVALID_ARG;
  }
  std::shared_ptr<pim_vsysfs_root_t> root_shared =
      vsysfs_lookup_root_shared(root);
  if (!root_shared) {
    return PIM_ERR_INVALID_ARG;
  }

  std::vector<std::shared_ptr<pim_vsysfs_t>> instances;
  {
    std::lock_guard<std::mutex> lock(root_shared->mutex);
    auto it = root_shared->default_attrs.find(name);
    if (it == root_shared->default_attrs.end()) {
      return PIM_ERR_NOT_FOUND;
    }

    const pim_error_t collect_ret = vsysfs_try_or_pim_no_memory([&]() {
      instances.reserve(root_shared->instances.size());
      for (auto inst_it = root_shared->instances.begin();
           inst_it != root_shared->instances.end();) {
        if (auto vs = inst_it->lock()) {
          instances.push_back(std::move(vs));
          ++inst_it;
        } else {
          inst_it = root_shared->instances.erase(inst_it);
        }
      }
    });
    if (collect_ret != PIM_SUCCESS) {
      return collect_ret;
    }

    root_shared->default_attrs.erase(it);
    root_shared->default_attr_access.erase(name);
  }

  bool any_removed = false;
  for (const auto &vs : instances) {
    std::lock_guard<std::mutex> lock(vs->mutex);
    auto attr_it = vs->attrs.find(name);
    if (attr_it == vs->attrs.end()) {
      continue;
    }
    vs->attrs.erase(attr_it);
    vs->attr_access.erase(name);
    vsysfs_bump_attr_revision(vs.get());
    any_removed = true;
  }

  if (any_removed) {
    vfile_notify_virtual_fs_changed();
  }
  return PIM_SUCCESS;
}

/*----------------------------------------------------------------------------
 * vsysfs instance
 *----------------------------------------------------------------------------*/

pim_vsysfs_t *pim_vsysfs_create(pim_vsysfs_root_t *root, pim_vdev_t *vdev,
                                const char *instance_name) noexcept {
  if (!root || !vdev || !instance_name) {
    return nullptr;
  }
  std::shared_ptr<pim_vsysfs_root_t> root_shared =
      vsysfs_lookup_root_shared(root);
  std::shared_ptr<pim_vdev_t> vdev_shared = vfile_lookup_vdev_shared(vdev);
  if (!root_shared || !vdev_shared) {
    return nullptr;
  }

  const std::shared_ptr<pim_vdev_root_t> root_vdev_root =
      root_shared->vdev_root.lock();
  const std::shared_ptr<pim_vdev_root_t> vdev_root = vdev_shared->root.lock();
  if (!root_vdev_root || !vdev_root ||
      root_vdev_root.get() != vdev_root.get()) {
    return nullptr;
  }

  return vsysfs_try_or_nullptr<pim_vsysfs_t>([&]() -> pim_vsysfs_t * {
    std::string cls;
    {
      std::lock_guard<std::mutex> vdev_lock(vdev_shared->mutex);
      cls = vdev_shared->name;
    }
    std::string inst(instance_name);
    inst = vsysfs_normalize_component(std::move(inst));
    if (!vsysfs_valid_instance_name(cls) || !vsysfs_valid_instance_name(inst)) {
      return nullptr;
    }

    const std::string class_dir =
        std::string("/sys/class/") + root_shared->class_name + "/" + cls;

    const std::string devices_parent_dir =
        std::string("/sys/devices/") + root_shared->bus_name + "/" + inst;
    const bool has_parent_child_layout = (inst != cls);
    std::string devices_dir = devices_parent_dir;
    if (has_parent_child_layout) {
      devices_dir += "/" + cls;
    }

    auto vsysfs = std::make_shared<pim_vsysfs_t>();
    vsysfs->root = root_shared;
    vsysfs->vdev = vdev_shared;
    vsysfs->class_name = cls;
    vsysfs->device_name = inst;
    vsysfs->class_dir = class_dir;
    vsysfs->devices_dir = devices_dir;

    {
      // Atomic duplicate-check + insert under one list lock.
      std::lock_guard<std::mutex> lock(g_vsysfs_list_mutex);
      for (const auto &existing : g_vsysfs_list) {
        if (!existing) {
          continue;
        }
        std::lock_guard<std::mutex> existing_lock(existing->mutex);
        if (existing->class_dir == class_dir ||
            existing->devices_dir == devices_dir) {
          return nullptr;
        }
      }
      g_vsysfs_list.push_back(vsysfs);
    }

    // Root-link must succeed; otherwise rollback registration.
    const pim_error_t root_link_ret = vsysfs_try_or_pim_no_memory([&]() {
      std::lock_guard<std::mutex> lock(root_shared->mutex);
      // Keep default-attr snapshot and root-link atomic w.r.t. root default
      // updates so a new instance cannot miss concurrent root mutations.
      vsysfs->attrs = root_shared->default_attrs;
      vsysfs->attr_access = root_shared->default_attr_access;
      for (const auto &kv : vsysfs->attrs) {
        if (vsysfs->attr_access.find(kv.first) == vsysfs->attr_access.end()) {
          vsysfs->attr_access[kv.first] = vsysfs_default_attr_access_mode();
        }
      }
      root_shared->instances.push_back(vsysfs);
    });
    if (root_link_ret != PIM_SUCCESS) {
      std::lock_guard<std::mutex> lock(g_vsysfs_list_mutex);
      g_vsysfs_list.erase(
          std::remove_if(g_vsysfs_list.begin(), g_vsysfs_list.end(),
                         [&vsysfs](const std::shared_ptr<pim_vsysfs_t> &entry) {
                           return !entry || entry.get() == vsysfs.get();
                         }),
          g_vsysfs_list.end());
      return nullptr;
    }

    // Auto-attach to currently mapped device for this vdev (if present).
    pim_device_t *mapped_dev = nullptr;
    {
      std::lock_guard<std::mutex> vdev_lock(vdev_shared->mutex);
      for (const auto &entry : vdev_shared->device_by_fd) {
        if (entry.second) {
          mapped_dev = entry.second;
          break;
        }
      }
    }

    if (mapped_dev) {
      std::lock_guard<std::mutex> map_lock(g_device_vsysfs_map_mutex);
      auto &items = g_device_vsysfs_map[mapped_dev];
      bool exists = false;
      for (const auto &wk : items) {
        if (wk.lock().get() == vsysfs.get()) {
          exists = true;
          break;
        }
      }
      if (!exists) {
        try {
          items.push_back(vsysfs);
        } catch (...) {
          // Best effort: instance remains valid even if map cache misses.
        }
      }
      std::lock_guard<std::mutex> vs_lock(vsysfs->mutex);
      vsysfs->attached_device = mapped_dev;
    }

    if (has_parent_child_layout) {
      (void)vsysfs_ensure_parent_device_node(root_shared, devices_parent_dir,
                                             inst);
    }

    vfile_notify_virtual_fs_changed();
    return vsysfs.get();
  });
}

void pim_vsysfs_destroy(pim_vsysfs_t *vsysfs) noexcept {
  if (!vsysfs) {
    return;
  }
  if (auto root_shared = vsysfs->root.lock()) {
    std::lock_guard<std::mutex> root_lock(root_shared->mutex);
    root_shared->instances.erase(
        std::remove_if(root_shared->instances.begin(),
                       root_shared->instances.end(),
                       [vsysfs](const std::weak_ptr<pim_vsysfs_t> &wk) {
                         auto sp = wk.lock();
                         return !sp || sp.get() == vsysfs;
                       }),
        root_shared->instances.end());
  }
  {
    std::lock_guard<std::mutex> lock(g_device_vsysfs_map_mutex);
    for (auto it = g_device_vsysfs_map.begin();
         it != g_device_vsysfs_map.end();) {
      auto &entries = it->second;
      entries.erase(
          std::remove_if(entries.begin(), entries.end(),
                         [vsysfs](const std::weak_ptr<pim_vsysfs_t> &wk) {
                           auto sp = wk.lock();
                           return !sp || sp.get() == vsysfs;
                         }),
          entries.end());
      if (entries.empty()) {
        it = g_device_vsysfs_map.erase(it);
      } else {
        ++it;
      }
    }
  }
  vsysfs_unregister_object(g_vsysfs_list, g_vsysfs_list_mutex, vsysfs);
  vfile_notify_virtual_fs_changed();
}

const char *pim_vsysfs_get_class_name(pim_vsysfs_t *vsysfs) noexcept {
  if (!vsysfs) {
    return nullptr;
  }
  return vsysfs->class_name.c_str();
}

const char *pim_vsysfs_get_device_name(pim_vsysfs_t *vsysfs) noexcept {
  if (!vsysfs) {
    return nullptr;
  }
  return vsysfs->device_name.c_str();
}

const char *pim_vsysfs_get_class_dir(pim_vsysfs_t *vsysfs) noexcept {
  if (!vsysfs) {
    return nullptr;
  }
  return vsysfs->class_dir.c_str();
}

const char *pim_vsysfs_get_devices_dir(pim_vsysfs_t *vsysfs) noexcept {
  if (!vsysfs) {
    return nullptr;
  }
  return vsysfs->devices_dir.c_str();
}

pim_error_t pim_vsysfs_create_attr(pim_vsysfs_t *vsysfs, const char *name,
                                   const char *value,
                                   pim_access_mode_t access) noexcept {
  if (!vsysfs || !name || !value || !vsysfs_attr_access_valid(access)) {
    return PIM_ERR_INVALID_ARG;
  }

  bool had_attr = false;
  std::string old_value;
  bool had_access = false;
  pim_access_mode_t old_access = vsysfs_default_attr_access_mode();
  bool access_changed = false;
  {
    std::lock_guard<std::mutex> lock(vsysfs->mutex);
    auto old_it = vsysfs->attrs.find(name);
    had_attr = (old_it != vsysfs->attrs.end());
    if (had_attr) {
      const pim_error_t copy_ret =
          vsysfs_try_or_pim_no_memory([&]() { old_value = old_it->second; });
      if (copy_ret != PIM_SUCCESS) {
        return copy_ret;
      }
      auto access_it = vsysfs->attr_access.find(name);
      had_access = (access_it != vsysfs->attr_access.end());
      old_access =
          had_access ? access_it->second : vsysfs_default_attr_access_mode();
      access_changed = (old_access != access);
    }

    try {
      vsysfs->attrs.insert_or_assign(name, value);
      vsysfs->attr_access.insert_or_assign(name, access);
    } catch (...) {
      vsysfs_run_best_effort([&]() {
        if (had_attr) {
          vsysfs->attrs.insert_or_assign(name, old_value);
        } else {
          vsysfs->attrs.erase(name);
        }
        if (had_access) {
          vsysfs->attr_access.insert_or_assign(name, old_access);
        } else {
          vsysfs->attr_access.erase(name);
        }
      });
      return PIM_ERR_NO_MEMORY;
    }

    vsysfs_bump_attr_revision(vsysfs);
  }
  if (!had_attr || access_changed) {
    vfile_notify_virtual_fs_changed();
  }
  return PIM_SUCCESS;
}

pim_error_t pim_vsysfs_remove_attr(pim_vsysfs_t *vsysfs,
                                   const char *name) noexcept {
  if (!vsysfs || !name) {
    return PIM_ERR_INVALID_ARG;
  }

  {
    std::lock_guard<std::mutex> lock(vsysfs->mutex);
    auto it = vsysfs->attrs.find(name);
    if (it == vsysfs->attrs.end()) {
      return PIM_ERR_NOT_FOUND;
    }
    vsysfs->attrs.erase(it);
    vsysfs->attr_access.erase(name);
    vsysfs_bump_attr_revision(vsysfs);
  }
  vfile_notify_virtual_fs_changed();
  return PIM_SUCCESS;
}

const char *pim_vsysfs_find_attr(pim_vsysfs_t *vsysfs,
                                 const char *name) noexcept {
  if (!vsysfs || !name) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(vsysfs->mutex);
  auto it = vsysfs->attrs.find(name);
  if (it == vsysfs->attrs.end()) {
    return nullptr;
  }
  static thread_local std::string tls_value;
  const pim_error_t copy_ret =
      vsysfs_try_or_pim_no_memory([&]() { tls_value = it->second; });
  if (copy_ret != PIM_SUCCESS) {
    return nullptr;
  }
  return tls_value.c_str();
}

static inline pim_device_t *
vsysfs_pick_mapped_device_from_vdev_locked(const pim_vdev_t *vdev) noexcept {
  if (!vdev) {
    return nullptr;
  }
  for (const auto &entry : vdev->device_by_fd) {
    if (entry.second) {
      return entry.second;
    }
  }
  return nullptr;
}

static inline void vsysfs_prune_device_vsysfs_items_locked(
    std::vector<std::weak_ptr<pim_vsysfs_t>> *items) noexcept {
  if (!items) {
    return;
  }
  items->erase(std::remove_if(items->begin(), items->end(),
                              [](const std::weak_ptr<pim_vsysfs_t> &wk) {
                                return wk.expired();
                              }),
               items->end());
}

static inline bool vsysfs_device_vsysfs_items_contains_locked(
    const std::vector<std::weak_ptr<pim_vsysfs_t>> &items,
    const pim_vsysfs_t *vsysfs) noexcept {
  if (!vsysfs) {
    return false;
  }
  for (const auto &wk : items) {
    if (wk.lock().get() == vsysfs) {
      return true;
    }
  }
  return false;
}

static inline void vsysfs_add_to_device_map_locked(
    pim_device_t *dev, const std::shared_ptr<pim_vsysfs_t> &vsysfs) noexcept {
  if (!dev || !vsysfs) {
    return;
  }
  try {
    auto &items = g_device_vsysfs_map[dev];
    vsysfs_prune_device_vsysfs_items_locked(&items);
    if (!vsysfs_device_vsysfs_items_contains_locked(items, vsysfs.get())) {
      items.push_back(vsysfs);
    }
  } catch (...) {
    // Best effort cache only.
  }
}

static inline void
vsysfs_remove_from_device_map_locked(pim_device_t *dev,
                                     const pim_vsysfs_t *vsysfs) noexcept {
  if (!dev || !vsysfs) {
    return;
  }
  auto it = g_device_vsysfs_map.find(dev);
  if (it == g_device_vsysfs_map.end()) {
    return;
  }
  auto &items = it->second;
  items.erase(std::remove_if(items.begin(), items.end(),
                             [vsysfs](const std::weak_ptr<pim_vsysfs_t> &wk) {
                               auto sp = wk.lock();
                               return !sp || sp.get() == vsysfs;
                             }),
              items.end());
  if (items.empty()) {
    g_device_vsysfs_map.erase(it);
  }
}

void vfile_on_vdev_device_mapping_changed(pim_vdev_t *vdev) noexcept {
  if (!vdev) {
    return;
  }

  const std::shared_ptr<pim_vdev_t> vdev_shared =
      vfile_lookup_vdev_shared(vdev);
  if (!vdev_shared) {
    return;
  }

  std::vector<std::shared_ptr<pim_vsysfs_t>> linked_instances;
  {
    std::lock_guard<std::mutex> list_lock(g_vsysfs_list_mutex);
    for (const auto &vs : g_vsysfs_list) {
      if (!vs) {
        continue;
      }
      if (vs->vdev.lock().get() == vdev) {
        linked_instances.push_back(vs);
      }
    }
  }
  if (linked_instances.empty()) {
    return;
  }

  pim_device_t *mapped_dev = nullptr;
  {
    std::lock_guard<std::mutex> vdev_lock(vdev_shared->mutex);
    mapped_dev = vsysfs_pick_mapped_device_from_vdev_locked(vdev_shared.get());
  }

  std::lock_guard<std::mutex> map_lock(g_device_vsysfs_map_mutex);
  for (const auto &vs : linked_instances) {
    if (!vs) {
      continue;
    }

    pim_device_t *old_dev = nullptr;
    {
      std::lock_guard<std::mutex> vs_lock(vs->mutex);
      old_dev = vs->attached_device;
      vs->attached_device = mapped_dev;
    }

    if (old_dev && old_dev != mapped_dev) {
      vsysfs_remove_from_device_map_locked(old_dev, vs.get());
    }
    if (mapped_dev) {
      vsysfs_add_to_device_map_locked(mapped_dev, vs);
    }
  }
}

pim_vsysfs_t *pim_vdev_get_vsysfs(pim_vdev_t *vdev) noexcept {
  if (!vdev) {
    return nullptr;
  }

  const std::shared_ptr<pim_vdev_t> vdev_shared =
      vfile_lookup_vdev_shared(vdev);
  if (!vdev_shared) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(g_vsysfs_list_mutex);
  for (const auto &vs : g_vsysfs_list) {
    if (!vs) {
      continue;
    }
    if (vs->vdev.lock().get() == vdev_shared.get()) {
      return vs.get();
    }
  }

  return nullptr;
}
