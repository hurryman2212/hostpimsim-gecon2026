#include "cb.hh"

#include <sys/ioctl.h>
#include <sys/mman.h>

int vdev_open_cb(pim_vdev_t *vdev, const char *path, int flags) {
  (void)path;

  if (!vdev) {
    errno = EINVAL;
    return -1;
  }

  pim_vdev_lock(vdev);

  int ret = -1, fd, errno_save = 0;
  UPMEMPIMRank *rank = nullptr;
  upmem_vdev_user_data::fd_type type;
  upmem_vdev_user_data *vdev_ud =
      static_cast<upmem_vdev_user_data *>(pim_vdev_get_user_data(vdev));
  if (!vdev_ud || vdev_ud->opened) {
    errno_save = EINVAL;
    goto out;
  }
  type = vdev_ud->type;

  fd = pim_vfile_open(flags);
  if (fd < 0) {
    errno_save = errno; // `errno` is set by `pim_vfile_open()`.
    goto out;
  }

  if (type == upmem_vdev_user_data::FD_RANK) {
    rank = upmem_create_rank(vdev);
    if (!rank) {
      errno_save = (errno == 0) ? ENOMEM : errno;
      goto out_close;
    }
    vdev_ud->rank = rank;

    pim_device_t *device = rank->get_device();
    if (pim_vdev_register_device(vdev, fd, device) != PIM_SUCCESS) {
      errno_save = EBUSY;
      goto out_destroy_rank;
    }
  }

  vdev_ud->opened = true;
  ret = fd;
  goto out;

out_destroy_rank:
  if (rank) {
    upmem_destroy_rank(rank);
  }
out_close:
  (void)close(fd);
out:
  pim_vdev_unlock(vdev);
  errno = errno_save;
  return ret;
}

static constexpr char kDpuRankIoctlMagic = 'd';
static constexpr unsigned long DPU_RANK_IOCTL_WRITE_TO_RANK =
    _IOW(kDpuRankIoctlMagic, 0, struct dpu_transfer_mram_abi *);
static constexpr unsigned long DPU_RANK_IOCTL_READ_FROM_RANK =
    _IOW(kDpuRankIoctlMagic, 1, struct dpu_transfer_mram_abi *);
static constexpr unsigned long DPU_RANK_IOCTL_COMMIT_COMMANDS =
    _IOW(kDpuRankIoctlMagic, 2, uint64_t *);
static constexpr unsigned long DPU_RANK_IOCTL_UPDATE_COMMANDS =
    _IOR(kDpuRankIoctlMagic, 3, uint64_t *);
static constexpr unsigned long DPU_RANK_IOCTL_DEBUG_MODE =
    _IOW(kDpuRankIoctlMagic, 4, uint8_t *);
static constexpr unsigned long DPU_RANK_IOCTL_SLICE_INFO =
    _IOW(kDpuRankIoctlMagic, 5, void *);
long rank_ioctl_cb(pim_vdev_t *vdev, int fd, unsigned long cmd,
                   unsigned long arg) {
  if (!vdev || fd < 0) {
    errno = EINVAL;
    return -1;
  }

  pim_vdev_lock(vdev);
  long ret = -1;

  upmem_vdev_user_data *vdev_ud =
      static_cast<upmem_vdev_user_data *>(pim_vdev_get_user_data(vdev));
  if (!vdev_ud || !vdev_ud->rank) {
    errno = ENODEV;
    pim_vdev_unlock(vdev);
    return -1;
  }

  UPMEMPIMRank *rank = vdev_ud->rank;
  switch (cmd) {
  case DPU_RANK_IOCTL_WRITE_TO_RANK:
    ret = upmem_ioctl_write_to_rank(rank, arg);
    break;
  case DPU_RANK_IOCTL_READ_FROM_RANK:
    ret = upmem_ioctl_read_from_rank(rank, arg);
    break;
  case DPU_RANK_IOCTL_COMMIT_COMMANDS:
    ret = upmem_ioctl_commit_commands(rank, arg);
    break;
  case DPU_RANK_IOCTL_UPDATE_COMMANDS:
    ret = upmem_ioctl_update_commands(rank, arg);
    break;
  case DPU_RANK_IOCTL_DEBUG_MODE:
    ret = upmem_ioctl_debug_mode(rank, arg);
    break;
  case DPU_RANK_IOCTL_SLICE_INFO:
    ret = upmem_ioctl_slice_info(rank, arg);
    break;
  default:
    errno = ENOTTY;
    ret = -1;
    break;
  }

  pim_vdev_unlock(vdev);
  return ret;
}

long dax_ioctl_cb(pim_vdev_t *vdev, int fd, unsigned long cmd,
                  unsigned long arg) {
  (void)vdev;
  (void)fd;
  (void)cmd;
  (void)arg;

  errno = ENOTTY;
  return -1;
}

void *rank_mmap_cb(pim_vdev_t *vdev, int fd, size_t length, int prot, int flags,
                   off_t offset) {
  (void)vdev;
  (void)fd;
  (void)length;
  (void)prot;
  (void)flags;
  (void)offset;

  errno = ENODEV;
  return MAP_FAILED;
}

void *dax_mmap_cb(pim_vdev_t *vdev, int fd, size_t length, int prot, int flags,
                  off_t offset) {
  (void)vdev;
  (void)fd;
  (void)length;
  (void)prot;
  (void)flags;
  (void)offset;

  /*
   * DAX path is intentionally stub-only for now.
   * Kernel parity (dpu_dax_mmap): when PERF capability is unavailable,
   * mmap fails from UNDEFINED mode with -EINVAL.
   */
  errno = EINVAL;
  return MAP_FAILED;
}

int vdev_close_cb(pim_vdev_t *vdev, int fd) {
  if (!vdev || fd < 0) {
    errno = EINVAL;
    return -1;
  }

  pim_vdev_lock(vdev);
  int ret = 0;

  upmem_vdev_user_data *vdev_ud =
      static_cast<upmem_vdev_user_data *>(pim_vdev_get_user_data(vdev));
  pim_device_t *device = pim_vdev_find_device(vdev, fd);
  if (device) {
    if (vdev_ud && vdev_ud->rank) {
      UPMEMPIMRank *rank = vdev_ud->rank;
      upmem_destroy_rank(rank);
      vdev_ud->rank = nullptr;
    }

    (void)pim_vdev_unregister_device(vdev, fd);
  }

  // DAX opens do not register a backing pim_device_t, but they still need to
  // release the single-open latch so a later dpu_alloc can reopen the node.
  if (vdev_ud) {
    vdev_ud->opened = false;
  }

  pim_vdev_unlock(vdev);
  return ret;
}
