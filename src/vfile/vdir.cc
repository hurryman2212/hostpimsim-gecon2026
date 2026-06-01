/**
 * @file vdir.cc
 * @brief Virtual-directory (VDIR) fd helpers for cursor and getdents.
 */

#include "vfile.hh"

#include <cstring>

static inline bool vdir_fd_opened_with_path_only(int flags) noexcept {
  return (flags & O_PATH) != 0;
}

std::shared_ptr<vfile_fd_entry> vdir_make_fd_entry(std::string path,
                                                   int flags) {
  auto entry = std::make_shared<vfile_fd_entry>();
  entry->type = vfile_fd_entry::FD_VDIR;
  entry->flags = flags;
  entry->virtual_path = std::move(path);
  entry->dir_read_offset = 0;
  entry->dir_cached_rows.clear();
  entry->dir_cached_generation = 0;
  return entry;
}

long vdir_seek_entry_cursor(const std::shared_ptr<vfile_fd_entry> &entry,
                            off_t offset, int whence) {
  if (!entry || entry->type != vfile_fd_entry::FD_VDIR) {
    return vfile_fail_with_errno(EBADF);
  }
  if (vdir_fd_opened_with_path_only(entry->flags)) {
    return vfile_fail_with_errno(EBADF);
  }

  long ret = -1;
  int seek_errno = EINVAL;
  {
    std::lock_guard<std::mutex> fd_lock(entry->op_mutex);
    seek_errno =
        vfile_seek_cursor_locked(&entry->dir_read_offset, offset, whence, &ret);
  }
  if (seek_errno != 0) {
    return vfile_fail_with_errno(seek_errno);
  }
  return ret;
}

static inline size_t vdir_dirent_aligned_reclen(size_t base, size_t name_len,
                                                size_t suffix_len,
                                                size_t align) noexcept {
  if (align == 0U || ((align & (align - 1U)) != 0U)) {
    return 0U;
  }

  size_t raw_reclen = base;
  if (name_len > (std::numeric_limits<size_t>::max() - raw_reclen)) {
    return 0U;
  }
  raw_reclen += name_len;
  if (suffix_len > (std::numeric_limits<size_t>::max() - raw_reclen)) {
    return 0U;
  }
  raw_reclen += suffix_len;

  const size_t align_mask = align - 1U;
  if (raw_reclen > (std::numeric_limits<size_t>::max() - align_mask)) {
    return 0U;
  }
  const size_t reclen = (raw_reclen + align_mask) & ~align_mask;
  if (reclen == 0U ||
      reclen >
          static_cast<size_t>(std::numeric_limits<unsigned short>::max())) {
    return 0U;
  }
  return reclen;
}

template <typename DirentT, typename InoT, typename OffT>
static inline void vdir_fill_dirent_common(DirentT *dent, size_t reclen,
                                           const vdev_dirent_row &row,
                                           size_t index) noexcept {
  if (!dent) {
    return;
  }
  memset(dent, 0, reclen);
  dent->d_ino = static_cast<InoT>(row.ino);
  dent->d_off = static_cast<OffT>(index + 1U);
  dent->d_reclen = static_cast<unsigned short>(reclen);
  memcpy(dent->d_name, row.name.c_str(), row.name.size());
  dent->d_name[row.name.size()] = '\0';
}

static inline long
vdir_refresh_cache_if_needed(const std::shared_ptr<vfile_fd_entry> &entry,
                             uint64_t generation) {
  bool need_refresh = false;
  std::string dir_path;
  {
    std::lock_guard<std::mutex> fd_lock(entry->op_mutex);
    need_refresh = (entry->dir_cached_generation != generation) ||
                   entry->dir_cached_rows.empty();
    if (need_refresh) {
      dir_path = entry->virtual_path;
    }
  }
  if (!need_refresh) {
    return 0;
  }

  std::vector<vdev_dirent_row> rows;
  if (!vfile_collect_dirents(dir_path, &rows)) {
    std::lock_guard<std::mutex> fd_lock(entry->op_mutex);
    entry->dir_cached_rows.clear();
    entry->dir_cached_generation = 0;
    return vfile_fail_with_errno(ENOENT);
  }

  {
    std::lock_guard<std::mutex> fd_lock(entry->op_mutex);
    if ((entry->dir_cached_generation != generation) ||
        entry->dir_cached_rows.empty()) {
      const bool assigned = vfile_try_or_false_with_errno(
          [&]() { entry->dir_cached_rows = std::move(rows); }, ENOMEM);
      if (!assigned) {
        return vfile_fail_with_errno(ENOMEM);
      }
      entry->dir_cached_generation = generation;
    }
  }
  return 0;
}

template <typename ReclenFn, typename FillFn>
static inline long
vdir_emit_cached_dirents_locked(vfile_fd_entry *entry, void *dirp, size_t count,
                                ReclenFn &&calc_reclen, FillFn &&fill_record) {
  if (!entry) {
    return vfile_fail_with_errno(EFAULT);
  }

  const std::vector<vdev_dirent_row> &rows = entry->dir_cached_rows;
  size_t used = 0;
  size_t cursor = entry->dir_read_offset;
  if (cursor >= rows.size()) {
    return 0; // EOF
  }

  char *out = reinterpret_cast<char *>(dirp);
  for (size_t i = cursor; i < rows.size(); ++i) {
    const size_t reclen = calc_reclen(rows[i]);
    if ((reclen == 0) || (reclen > (count - used))) {
      if (used == 0) {
        return vfile_fail_with_errno(EINVAL);
      }
      break;
    }

    fill_record(out + used, reclen, rows[i], i);
    used += reclen;
    cursor = i + 1;
  }
  entry->dir_read_offset = cursor;
  return static_cast<long>(used);
}

template <typename ReclenFn, typename FillFn>
static inline long
vdir_getdents_try_once(const std::shared_ptr<vfile_fd_entry> &entry, void *dirp,
                       size_t count, uint64_t generation, ReclenFn &calc_reclen,
                       FillFn &fill_record, bool *retry) {
  if (retry) {
    *retry = false;
  }

  const long refresh_ret = vdir_refresh_cache_if_needed(entry, generation);
  if (refresh_ret < 0) {
    return refresh_ret;
  }

  std::lock_guard<std::mutex> fd_lock(entry->op_mutex);
  if ((entry->dir_cached_generation != generation) ||
      entry->dir_cached_rows.empty()) {
    if (retry) {
      *retry = true;
    }
    return 0;
  }
  return vdir_emit_cached_dirents_locked(entry.get(), dirp, count, calc_reclen,
                                         fill_record);
}

template <typename ReclenFn, typename FillFn>
static long
vdir_dispatch_getdents_common(const std::shared_ptr<vfile_fd_entry> &entry,
                              void *dirp, size_t count, ReclenFn &&calc_reclen,
                              FillFn &&fill_record) {
  if (!entry || entry->type != vfile_fd_entry::FD_VDIR) {
    return vfile_fail_with_errno(ENOTDIR);
  }
  if (vdir_fd_opened_with_path_only(entry->flags)) {
    return vfile_fail_with_errno(EBADF);
  }
  if (!dirp) {
    return vfile_fail_with_errno(EFAULT);
  }

  auto calc = calc_reclen;
  auto fill = fill_record;
  uint64_t generation = vfile_get_virtual_fs_generation();
  bool retry = false;
  long ret = vdir_getdents_try_once(entry, dirp, count, generation, calc, fill,
                                    &retry);
  if (!retry) {
    return ret;
  }

  const uint64_t next_generation = vfile_get_virtual_fs_generation();
  if (next_generation == generation) {
    // Avoid spinning forever when cache competition is local to this fd while
    // global topology generation did not progress.
    return 0;
  }

  retry = false;
  ret = vdir_getdents_try_once(entry, dirp, count, next_generation, calc, fill,
                               &retry);
  if (!retry) {
    return ret;
  }

  // Topology kept changing across both attempts. Keep getdents-compatible
  // behavior and let caller retry.
  return 0;
}

template <typename DirentT> struct vdir_getdents_traits;

template <typename DirentT>
static inline size_t vdir_getdents_reclen(const vdev_dirent_row &row) noexcept {
  using traits = vdir_getdents_traits<DirentT>;
  return vdir_dirent_aligned_reclen(offsetof(DirentT, d_name), row.name.size(),
                                    traits::kSuffixLen, traits::kAlign);
}

template <typename DirentT>
static inline void vdir_fill_getdents_record(char *dst, size_t reclen,
                                             const vdev_dirent_row &row,
                                             size_t index) noexcept {
  auto *dent = reinterpret_cast<DirentT *>(dst);
  using traits = vdir_getdents_traits<DirentT>;
  vdir_fill_dirent_common<DirentT, typename traits::ino_type,
                          typename traits::off_type>(dent, reclen, row, index);
  traits::finalize(dent, reclen, row);
}

template <typename DirentT>
static inline long
vdir_getdents_typed(const std::shared_ptr<vfile_fd_entry> &entry, void *dirp,
                    size_t count) {
  return vdir_dispatch_getdents_common(entry, dirp, count,
                                       vdir_getdents_reclen<DirentT>,
                                       vdir_fill_getdents_record<DirentT>);
}

#ifdef SYS_getdents64
struct vdir_linux_dirent64 {
  uint64_t d_ino;
  int64_t d_off;
  unsigned short d_reclen;
  unsigned char d_type;
  char d_name[];
};

template <> struct vdir_getdents_traits<vdir_linux_dirent64> {
  using ino_type = uint64_t;
  using off_type = int64_t;

  // linux_dirent64 emits d_name with a trailing NUL byte.
  static constexpr size_t kSuffixLen = sizeof(char);
  // linux_dirent64 record lengths are aligned to an 8-byte boundary.
  static constexpr size_t kAlign = alignof(uint64_t);

  static inline void finalize(vdir_linux_dirent64 *dent, size_t reclen,
                              const vdev_dirent_row &row) noexcept {
    (void)reclen;
    dent->d_type = row.dtype;
  }
};

long vdir_getdents64_from_entry(const std::shared_ptr<vfile_fd_entry> &entry,
                                void *dirp, size_t count) {
  return vdir_getdents_typed<vdir_linux_dirent64>(entry, dirp, count);
}
#endif

#ifdef SYS_getdents
struct vdir_linux_dirent {
  unsigned long d_ino;
  unsigned long d_off;
  unsigned short d_reclen;
  char d_name[];
};

template <> struct vdir_getdents_traits<vdir_linux_dirent> {
  using ino_type = unsigned long;
  using off_type = unsigned long;

  // linux_dirent (legacy) uses trailing NUL + d_type byte.
  static constexpr size_t kSuffixLen = sizeof(char) + sizeof(unsigned char);
  // linux_dirent (legacy) record lengths are aligned to machine word size.
  static constexpr size_t kAlign = sizeof(unsigned long);

  static inline void finalize(vdir_linux_dirent *dent, size_t reclen,
                              const vdev_dirent_row &row) noexcept {
    unsigned char *bytes = reinterpret_cast<unsigned char *>(dent);
    bytes[reclen - 1U] = row.dtype;
  }
};

long vdir_getdents_from_entry(const std::shared_ptr<vfile_fd_entry> &entry,
                              void *dirp, size_t count) {
  return vdir_getdents_typed<vdir_linux_dirent>(entry, dirp, count);
}
#endif
