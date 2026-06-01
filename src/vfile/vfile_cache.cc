/**
 * @file vfile_cache.cc
 * @brief Virtual topology cache for the unified vfile namespace (/dev + /sys)
 *
 * Purpose:
 * - Builds and caches an immutable in-memory snapshot of the virtual file
 *   namespace exposed by the vfile layer.
 * - The snapshot merges:
 *   - vdev instances (virtual character nodes under /dev/<glob>)
 *   - vsysfs instances (virtual directories + attribute files under
 * /sys/<glob>)
 *   - vmodule instances (virtual module directories + attrs under
 * /sys/module/<name>)
 * - Provides normalized path lookup and directory-entry materialization for
 *   syscall dispatch and virtual directory traversal.
 *
 * Public cache API implemented here:
 * - vfile_notify_virtual_fs_changed():
 *   Bumps topology generation and invalidates cached snapshot.
 * - vfile_get_virtual_fs_generation():
 *   Returns current topology generation for stale-cache detection.
 * - vfile_resolve_virtual_node():
 *   Resolves absolute path -> virtual node metadata
 *   (VDEV_NODE / VSYSFS_ATTR / VDIR).
 * - vfile_collect_dirents():
 *   Emits "."/".." plus synthetic child entries for getdents/getdents64.
 *
 * Affected components:
 * - vdev.cc:
 *   vdev create/destroy mutates /dev namespace and calls
 *   vfile_notify_virtual_fs_changed().
 * - vsysfs.cc:
 *   vsysfs create/destroy and attribute add/remove/update/access-mode changes
 *   mutate /sys namespace and call vfile_notify_virtual_fs_changed().
 * - vmodule.cc:
 *   vmodule create/destroy and attribute add/remove/update/access-mode changes
 *   mutate /sys/module namespace and call vfile_notify_virtual_fs_changed().
 * - vsyscall.cc:
 *   open/stat/access/readlink/path-based dispatch uses
 *   vfile_resolve_virtual_node() to classify virtual paths.
 * - vdir.cc:
 *   getdents/getdents64 uses vfile_collect_dirents(); directory-fd cache
 *   refresh is generation-aware via vfile_get_virtual_fs_generation().
 *
 * Non-target path behavior (normal kernel files):
 * - Path-based intercepted syscalls still pass through virtual-node lookup.
 * - This cache makes "not ours" checks cheap (snapshot/map lookup), so
 *   non-target paths are rejected quickly.
 * - The call still has interceptor overhead versus direct kernel syscall
 *   without interception; this cache minimizes that overhead but cannot remove
 *   it.
 *
 * Not handled here:
 * - FD ownership table, alias tracking, and syscall execution flow
 *   (vsyscall.cc)
 * - vdev callback invocation/session lifecycle (vdev.cc + vsyscall.cc)
 * - vsysfs attribute storage/mutation logic (vsysfs.cc)
 */

#include "vfile.hh"

#include <dirent.h>

static std::mutex g_virtual_fs_cache_mutex;
static uint64_t g_virtual_fs_generation{1};
static uint64_t g_virtual_fs_cached_generation{0};
static bool g_virtual_fs_cache_valid{false};

static inline pim_access_mode_t
vfile_vsysfs_get_attr_access_locked(const pim_vsysfs_t *vsysfs,
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
vfile_vmodule_get_attr_access_locked(const pim_vmodule_t *vmodule,
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

enum class vfile_attr_owner_kind { VSYSFS, VMODULE };

struct vfile_attr_binding {
  vfile_attr_owner_kind owner{vfile_attr_owner_kind::VSYSFS};
  std::shared_ptr<pim_vsysfs_t> vsysfs;
  std::shared_ptr<pim_vmodule_t> vmodule;
  std::string attr_name;
};

struct vfile_virtual_fs_snapshot {
  std::map<std::string, std::map<std::string, unsigned char>> dirs;
  std::unordered_map<std::string, std::shared_ptr<pim_vdev_t>> vfile_nodes;
  std::unordered_map<std::string, std::shared_ptr<pim_vsysfs_t>> vsysfs_dirs;
  std::unordered_map<std::string, std::shared_ptr<pim_vmodule_t>> vmodule_dirs;
  std::unordered_map<std::string, vfile_attr_binding> attr_bindings;
};

using vfile_virtual_fs_snapshot_handle =
    std::shared_ptr<const vfile_virtual_fs_snapshot>;

static vfile_virtual_fs_snapshot_handle g_virtual_fs_cached_snapshot;

static inline vfile_virtual_fs_snapshot_handle
vfile_get_cached_virtual_fs_snapshot_locked(uint64_t generation) noexcept {
  if (g_virtual_fs_cache_valid && g_virtual_fs_cached_snapshot &&
      g_virtual_fs_cached_generation == generation) {
    return g_virtual_fs_cached_snapshot;
  }
  return {};
}

static inline void
vfile_snapshot_ensure_dir(vfile_virtual_fs_snapshot *snapshot,
                          const std::string &dir_path) {
  if (!snapshot) {
    return;
  }
  auto &entries = snapshot->dirs[dir_path];
  (void)entries;
}

static inline void vfile_snapshot_add_child(vfile_virtual_fs_snapshot *snapshot,
                                            const std::string &parent,
                                            const std::string &name,
                                            unsigned char dtype) {
  if (!snapshot || name.empty()) {
    return;
  }
  auto &children = snapshot->dirs[parent];
  auto it = children.find(name);
  if (it == children.end()) {
    children.emplace(name, dtype);
  } else if (it->second == DT_UNKNOWN) {
    it->second = dtype;
  }
}

static inline void vfile_snapshot_add_path(vfile_virtual_fs_snapshot *snapshot,
                                           const std::string &abs_path,
                                           unsigned char leaf_type) {
  if (!snapshot) {
    return;
  }

  std::string normalized;
  if (!vfile_normalize_absolute_path(abs_path, &normalized)) {
    return;
  }
  if (normalized == "/") {
    vfile_snapshot_ensure_dir(snapshot, "/");
    return;
  }

  vfile_snapshot_ensure_dir(snapshot, "/");

  std::string current = "/";
  size_t start = 1;
  while (start < normalized.size()) {
    size_t slash = normalized.find('/', start);
    const bool is_last = (slash == std::string::npos);
    const size_t end = is_last ? normalized.size() : slash;
    const std::string name = normalized.substr(start, end - start);
    const unsigned char dtype = is_last ? leaf_type : DT_DIR;
    vfile_snapshot_add_child(snapshot, current, name, dtype);

    if (is_last) {
      if (leaf_type == DT_DIR) {
        std::string child =
            (current == "/") ? "/" + name : current + "/" + name;
        vfile_snapshot_ensure_dir(snapshot, child);
      }
      break;
    }

    current = (current == "/") ? "/" + name : current + "/" + name;
    vfile_snapshot_ensure_dir(snapshot, current);
    start = slash + 1;
  }
}

static bool vfile_get_vsysfs_attr_properties(
    const std::shared_ptr<pim_vsysfs_t> &vsysfs, const std::string &attr_name,
    size_t *value_size, pim_access_mode_t *attr_access) {
  if (!vsysfs || !value_size || !attr_access) {
    return false;
  }
  std::lock_guard<std::mutex> attrs_lock(vsysfs->mutex);
  auto it = vsysfs->attrs.find(attr_name);
  if (it == vsysfs->attrs.end()) {
    return false;
  }
  *value_size = it->second.size();
  *attr_access = vfile_vsysfs_get_attr_access_locked(vsysfs.get(), attr_name);
  return true;
}

static bool vfile_get_vmodule_attr_properties(
    const std::shared_ptr<pim_vmodule_t> &vmodule, const std::string &attr_name,
    size_t *value_size, pim_access_mode_t *attr_access) {
  if (!vmodule || !value_size || !attr_access) {
    return false;
  }
  std::lock_guard<std::mutex> attrs_lock(vmodule->mutex);
  auto it = vmodule->attrs.find(attr_name);
  if (it == vmodule->attrs.end()) {
    return false;
  }
  *value_size = it->second.size();
  *attr_access = vfile_vmodule_get_attr_access_locked(vmodule.get(), attr_name);
  return true;
}

struct vfile_snapshot_vsysfs_seed {
  std::vector<std::string> attrs;
  std::vector<std::string> concrete_dirs;
};

struct vfile_snapshot_vmodule_seed {
  std::vector<std::string> attrs;
  std::vector<std::pair<std::string, std::string>> registered_drivers;
  std::string dir;
};

static inline bool
vfile_snapshot_prepare_vsysfs_seed(const std::shared_ptr<pim_vsysfs_t> &vsysfs,
                                   vfile_snapshot_vsysfs_seed *seed) {
  if (!vsysfs || !seed) {
    return false;
  }

  std::string class_dir;
  std::string devices_dir;
  {
    std::lock_guard<std::mutex> attrs_lock(vsysfs->mutex);
    seed->attrs.clear();
    seed->attrs.reserve(vsysfs->attrs.size());
    for (const auto &kv : vsysfs->attrs) {
      seed->attrs.push_back(kv.first);
    }
    class_dir = vsysfs->class_dir;
    devices_dir = vsysfs->devices_dir;
  }

  seed->concrete_dirs.clear();
  if (!class_dir.empty()) {
    seed->concrete_dirs.push_back(class_dir);
  }
  if (!devices_dir.empty() && devices_dir != class_dir) {
    seed->concrete_dirs.push_back(devices_dir);
  }
  return !seed->concrete_dirs.empty();
}

static inline bool vfile_snapshot_prepare_vmodule_seed(
    const std::shared_ptr<pim_vmodule_t> &vmodule,
    vfile_snapshot_vmodule_seed *seed) {
  if (!vmodule || !seed) {
    return false;
  }

  std::lock_guard<std::mutex> attrs_lock(vmodule->mutex);
  seed->attrs.clear();
  seed->attrs.reserve(vmodule->attrs.size());
  for (const auto &kv : vmodule->attrs) {
    seed->attrs.push_back(kv.first);
  }
  seed->registered_drivers.clear();
  seed->registered_drivers.reserve(vmodule->registered_drivers.size());
  for (const auto &entry : vmodule->registered_drivers) {
    seed->registered_drivers.push_back(entry);
  }
  seed->dir = vmodule->dir;
  return !seed->dir.empty();
}

class vfile_snapshot_builder {
public:
  vfile_snapshot_builder() { vfile_snapshot_ensure_dir(&snapshot_, "/"); }

  void add_vdev(const std::shared_ptr<pim_vdev_t> &vdev) {
    if (!vdev) {
      return;
    }

    std::string path;
    {
      std::lock_guard<std::mutex> vfile_lock(vdev->mutex);
      path = vdev->path;
    }
    if (path.empty()) {
      return;
    }
    add_vdev_node(vdev, path);
  }

  void add_vsysfs(const std::shared_ptr<pim_vsysfs_t> &vsysfs) {
    if (!vsysfs) {
      return;
    }

    vfile_snapshot_vsysfs_seed seed;
    if (!vfile_snapshot_prepare_vsysfs_seed(vsysfs, &seed)) {
      return;
    }

    for (const auto &sample_dir : seed.concrete_dirs) {
      add_vsysfs_dir(vsysfs, sample_dir, seed.attrs);
    }
  }

  void add_vmodule(const std::shared_ptr<pim_vmodule_t> &vmodule) {
    if (!vmodule) {
      return;
    }

    vfile_snapshot_vmodule_seed seed;
    if (!vfile_snapshot_prepare_vmodule_seed(vmodule, &seed)) {
      return;
    }
    add_vmodule_dir(vmodule, seed.dir, seed.attrs);
    add_vmodule_registered_drivers(seed.registered_drivers);
  }

  vfile_virtual_fs_snapshot build() { return std::move(snapshot_); }

private:
  void add_vdev_node(const std::shared_ptr<pim_vdev_t> &vdev,
                     const std::string &path) {
    vfile_snapshot_add_path(&snapshot_, path, DT_CHR);
    snapshot_.vfile_nodes[path] = vdev;
  }

  void bind_vsysfs_attr(const std::shared_ptr<pim_vsysfs_t> &vsysfs,
                        const std::string &attr_path,
                        const std::string &attr_name) {
    vfile_snapshot_add_path(&snapshot_, attr_path, DT_REG);
    snapshot_.attr_bindings.emplace(attr_path,
                                    vfile_attr_binding{
                                        vfile_attr_owner_kind::VSYSFS,
                                        vsysfs,
                                        nullptr,
                                        attr_name,
                                    });
  }

  void bind_vsysfs_attrs_for_dir(const std::shared_ptr<pim_vsysfs_t> &vsysfs,
                                 const std::string &sample_dir,
                                 const std::vector<std::string> &attrs) {
    for (const auto &attr_name : attrs) {
      bind_vsysfs_attr(vsysfs, sample_dir + "/" + attr_name, attr_name);
    }
  }

  void add_vsysfs_dir(const std::shared_ptr<pim_vsysfs_t> &vsysfs,
                      const std::string &sample_dir,
                      const std::vector<std::string> &attrs) {
    bool root_uses_distinct_globs = true;
    if (vsysfs) {
      if (auto root = vsysfs->root.lock()) {
        std::lock_guard<std::mutex> root_lock(root->mutex);
        root_uses_distinct_globs = (root->class_glob != root->devices_glob);
      }
    }

    unsigned char leaf_type = DT_DIR;
    if (vsysfs && !vsysfs->class_dir.empty() && !vsysfs->devices_dir.empty() &&
        root_uses_distinct_globs && (sample_dir == vsysfs->class_dir) &&
        (vsysfs->class_dir != vsysfs->devices_dir)) {
      leaf_type = DT_LNK;
    }

    vfile_snapshot_add_path(&snapshot_, sample_dir, leaf_type);
    snapshot_.vsysfs_dirs[sample_dir] = vsysfs;
    bind_vsysfs_attrs_for_dir(vsysfs, sample_dir, attrs);
  }

  void bind_vmodule_attr(const std::shared_ptr<pim_vmodule_t> &vmodule,
                         const std::string &attr_path,
                         const std::string &attr_name) {
    vfile_snapshot_add_path(&snapshot_, attr_path, DT_REG);
    snapshot_.attr_bindings.emplace(attr_path,
                                    vfile_attr_binding{
                                        vfile_attr_owner_kind::VMODULE,
                                        nullptr,
                                        vmodule,
                                        attr_name,
                                    });
  }

  void bind_vmodule_attrs_for_dir(const std::shared_ptr<pim_vmodule_t> &vmodule,
                                  const std::string &dir_path,
                                  const std::vector<std::string> &attrs) {
    for (const auto &attr_name : attrs) {
      bind_vmodule_attr(vmodule, dir_path + "/" + attr_name, attr_name);
    }
  }

  void add_vmodule_dir(const std::shared_ptr<pim_vmodule_t> &vmodule,
                       const std::string &dir_path,
                       const std::vector<std::string> &attrs) {
    vfile_snapshot_add_path(&snapshot_, dir_path, DT_DIR);
    snapshot_.vmodule_dirs[dir_path] = vmodule;
    bind_vmodule_attrs_for_dir(vmodule, dir_path, attrs);
  }

  void add_bus_driver_dir(const std::string &bus_name,
                          const std::string &driver_name) {
    if (bus_name.empty() || driver_name.empty()) {
      return;
    }
    const std::string path = "/sys/bus/" + bus_name + "/drivers/" + driver_name;
    vfile_snapshot_add_path(&snapshot_, path, DT_DIR);
  }

  void add_vmodule_registered_drivers(
      const std::vector<std::pair<std::string, std::string>>
          &registered_drivers) {
    for (const auto &entry : registered_drivers) {
      add_bus_driver_dir(entry.first, entry.second);
    }
  }

  vfile_virtual_fs_snapshot snapshot_;
};

static vfile_virtual_fs_snapshot vfile_build_virtual_fs_snapshot_uncached() {
  vfile_snapshot_builder builder;

  std::vector<std::shared_ptr<pim_vdev_t>> vdevs =
      vfile_copy_vdev_list_for_snapshot();
  for (const auto &vdev : vdevs) {
    builder.add_vdev(vdev);
  }

  std::vector<std::shared_ptr<pim_vsysfs_t>> vsysfs_list =
      vfile_copy_vsysfs_list_for_snapshot();
  for (const auto &vsysfs : vsysfs_list) {
    builder.add_vsysfs(vsysfs);
  }

  std::vector<std::shared_ptr<pim_vmodule_t>> vmodule_list =
      vfile_copy_vmodule_list_for_snapshot();
  for (const auto &vmodule : vmodule_list) {
    builder.add_vmodule(vmodule);
  }

  return builder.build();
}

static vfile_virtual_fs_snapshot_handle vfile_build_virtual_fs_snapshot() {
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(g_virtual_fs_cache_mutex);
    generation = g_virtual_fs_generation;
    if (auto cached = vfile_get_cached_virtual_fs_snapshot_locked(generation)) {
      return cached;
    }
  }

  auto snapshot = std::make_shared<vfile_virtual_fs_snapshot>(
      vfile_build_virtual_fs_snapshot_uncached());

  std::lock_guard<std::mutex> lock(g_virtual_fs_cache_mutex);
  if (generation == g_virtual_fs_generation) {
    g_virtual_fs_cached_snapshot = std::move(snapshot);
    g_virtual_fs_cached_generation = generation;
    g_virtual_fs_cache_valid = true;
    return g_virtual_fs_cached_snapshot;
  }

  if (auto cached = vfile_get_cached_virtual_fs_snapshot_locked(
          g_virtual_fs_generation)) {
    return cached;
  }

  // Topology changed during build and no stable cached snapshot was available.
  // Return best effort for this call without publishing stale cache state.
  return snapshot;
}

static inline bool vfile_bind_vsysfs_attr_node(
    const std::string &path, std::shared_ptr<pim_vsysfs_t> vsysfs,
    std::string attr_name, vfile_virtual_path_info *node) {
  if (!node || !vsysfs) {
    return false;
  }
  size_t attr_size = 0;
  pim_access_mode_t attr_access = PIM_ACCESS_MODE_RW;
  if (!vfile_get_vsysfs_attr_properties(vsysfs, attr_name, &attr_size,
                                        &attr_access)) {
    return false;
  }
  node->kind = vfile_virtual_path_kind::VSYSFS_ATTR;
  node->path = path;
  node->vsysfs = std::move(vsysfs);
  node->attr_name = std::move(attr_name);
  node->attr_size = attr_size;
  node->attr_access = attr_access;
  return true;
}

static inline bool vfile_bind_vmodule_attr_node(
    const std::string &path, std::shared_ptr<pim_vmodule_t> vmodule,
    std::string attr_name, vfile_virtual_path_info *node) {
  if (!node || !vmodule) {
    return false;
  }
  size_t attr_size = 0;
  pim_access_mode_t attr_access = PIM_ACCESS_MODE_RW;
  if (!vfile_get_vmodule_attr_properties(vmodule, attr_name, &attr_size,
                                         &attr_access)) {
    return false;
  }
  node->kind = vfile_virtual_path_kind::VMODULE_ATTR;
  node->path = path;
  node->vmodule = std::move(vmodule);
  node->attr_name = std::move(attr_name);
  node->attr_size = attr_size;
  node->attr_access = attr_access;
  return true;
}

static inline bool
vfile_try_resolve_attr_node(const std::string &path,
                            const vfile_virtual_fs_snapshot &snapshot,
                            vfile_virtual_path_info *node) {
  auto attr_it = snapshot.attr_bindings.find(path);
  if (attr_it == snapshot.attr_bindings.end()) {
    return false;
  }
  if (attr_it->second.owner == vfile_attr_owner_kind::VSYSFS) {
    return vfile_bind_vsysfs_attr_node(path, attr_it->second.vsysfs,
                                       attr_it->second.attr_name, node);
  }
  if (attr_it->second.owner == vfile_attr_owner_kind::VMODULE) {
    return vfile_bind_vmodule_attr_node(path, attr_it->second.vmodule,
                                        attr_it->second.attr_name, node);
  }
  return false;
}

static inline bool
vfile_try_resolve_vdev_node(const std::string &path,
                            const vfile_virtual_fs_snapshot &snapshot,
                            vfile_virtual_path_info *node) {
  if (!node) {
    return false;
  }
  auto vfile_it = snapshot.vfile_nodes.find(path);
  if (vfile_it == snapshot.vfile_nodes.end()) {
    return false;
  }
  node->kind = vfile_virtual_path_kind::VDEV_NODE;
  node->path = path;
  node->vdev = vfile_it->second;
  return true;
}

static inline bool
vfile_try_resolve_non_dir_node(const std::string &path,
                               const vfile_virtual_fs_snapshot &snapshot,
                               vfile_virtual_path_info *node) {
  if (vfile_try_resolve_attr_node(path, snapshot, node)) {
    return true;
  }
  return vfile_try_resolve_vdev_node(path, snapshot, node);
}

static inline bool
vfile_try_resolve_dir_node(const std::string &path,
                           const vfile_virtual_fs_snapshot &snapshot,
                           vfile_virtual_path_info *node) {
  if (!node) {
    return false;
  }
  if (snapshot.dirs.find(path) != snapshot.dirs.end()) {
    node->kind = vfile_virtual_path_kind::VDIR;
    node->path = path;
    auto dir_it = snapshot.vsysfs_dirs.find(path);
    if (dir_it != snapshot.vsysfs_dirs.end()) {
      node->vsysfs = dir_it->second;
    }
    auto module_dir_it = snapshot.vmodule_dirs.find(path);
    if (module_dir_it != snapshot.vmodule_dirs.end()) {
      node->vmodule = module_dir_it->second;
    }
    return true;
  }
  return false;
}

void vfile_notify_virtual_fs_changed() noexcept {
  std::lock_guard<std::mutex> lock(g_virtual_fs_cache_mutex);
  ++g_virtual_fs_generation;
  if (g_virtual_fs_generation == 0) {
    g_virtual_fs_generation = 1;
  }
  g_virtual_fs_cache_valid = false;
}

uint64_t vfile_get_virtual_fs_generation() noexcept {
  std::lock_guard<std::mutex> lock(g_virtual_fs_cache_mutex);
  return g_virtual_fs_generation;
}

bool vfile_resolve_virtual_node(const std::string &abs_path,
                                vfile_virtual_path_info *node,
                                int *resolve_errno) {
  if (resolve_errno) {
    *resolve_errno = 0;
  }
  if (!node) {
    if (resolve_errno) {
      *resolve_errno = EINVAL;
    }
    return false;
  }
  *node = {};

  const vfile_virtual_fs_snapshot_handle snapshot =
      vfile_build_virtual_fs_snapshot();
  if (!snapshot) {
    if (resolve_errno) {
      *resolve_errno = ENOMEM;
    }
    return false;
  }

  std::string normalized;
  if (!vfile_normalize_absolute_path(abs_path, &normalized)) {
    if (resolve_errno) {
      *resolve_errno = EINVAL;
    }
    return false;
  }

  if (vfile_try_resolve_non_dir_node(normalized, *snapshot, node)) {
    return true;
  }
  return vfile_try_resolve_dir_node(normalized, *snapshot, node);
}

bool vfile_collect_dirents(const std::string &abs_dir_path,
                           std::vector<vdev_dirent_row> *rows) {
  if (!rows) {
    return false;
  }
  rows->clear();

  std::string normalized_dir_path;
  if (!vfile_normalize_absolute_path(abs_dir_path, &normalized_dir_path)) {
    return false;
  }

  vfile_virtual_fs_snapshot_handle snapshot = vfile_build_virtual_fs_snapshot();
  if (!snapshot) {
    return false;
  }

  auto it = snapshot->dirs.find(normalized_dir_path);
  rows->push_back(
      vdev_dirent_row{".", DT_DIR,
                      static_cast<uint64_t>(std::hash<std::string>{}(
                          normalized_dir_path + "/" + "."))});
  rows->push_back(
      vdev_dirent_row{"..", DT_DIR,
                      static_cast<uint64_t>(std::hash<std::string>{}(
                          normalized_dir_path + "/" + ".."))});

  if (it != snapshot->dirs.end()) {
    for (const auto &entry : it->second) {
      rows->push_back(
          vdev_dirent_row{entry.first, entry.second,
                          static_cast<uint64_t>(std::hash<std::string>{}(
                              normalized_dir_path + "/" + entry.first))});
    }
    return true;
  }
  rows->clear();
  return false;
}
