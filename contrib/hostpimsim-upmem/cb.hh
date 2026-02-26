#pragma once

#include <hostpimsim.h>

struct upmem_vdev_user_data {
  enum fd_type { FD_RANK = 0, FD_DAX = 1 } type;
  pid_t owner_pid{-1};
  void *state{nullptr};
};

int vdev_open_cb(pim_vdev_t *vdev, const char *path, int flags);

long rank_ioctl_cb(pim_vdev_t *vdev, int fd, unsigned long cmd,
                   unsigned long arg);
long dax_ioctl_cb(pim_vdev_t *vdev, int fd, unsigned long cmd,
                  unsigned long arg);

void *rank_mmap_cb(pim_vdev_t *vdev, int fd, size_t length, int prot, int flags,
                   off_t offset);
void *dax_mmap_cb(pim_vdev_t *vdev, int fd, size_t length, int prot, int flags,
                  off_t offset);

int vdev_close_cb(pim_vdev_t *vdev, int fd);
