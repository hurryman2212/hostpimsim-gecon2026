#include "cb.hh"

#include <overlaysys.h>

static void setup_vfile_tree(void) {
  pim_vmodule_t *dpu_vmodule = pim_vmodule_create("dpu");
  pim_vmodule_create_attr(dpu_vmodule, "version", "7.1", PIM_ACCESS_MODE_RO);

  pim_vmodule_register_driver(dpu_vmodule, "platform", "dpu_region_mem");

  pim_vdev_root_t *rank_root = pim_vdev_root_create("dpu_rank*");
  pim_vdev_root_t *dax_root = pim_vdev_root_create("dax*.*");

  pim_vdev_root_set_open_cb(rank_root, vdev_open_cb);
  pim_vdev_root_set_ioctl_cb(rank_root, rank_ioctl_cb);
  pim_vdev_root_set_mmap_cb(rank_root, rank_mmap_cb);
  pim_vdev_root_set_close_cb(rank_root, vdev_close_cb);

  pim_vdev_root_set_open_cb(dax_root, vdev_open_cb);
  pim_vdev_root_set_ioctl_cb(dax_root, dax_ioctl_cb);
  pim_vdev_root_set_mmap_cb(dax_root, dax_mmap_cb);
  pim_vdev_root_set_close_cb(dax_root, vdev_close_cb);

  pim_vdev_t *rank_vdev0 = pim_vdev_create(rank_root, "dpu_rank0", -1, -1);
  upmem_vdev_user_data *rank_ud0 = new upmem_vdev_user_data;
  rank_ud0->type = upmem_vdev_user_data::FD_RANK;
  pim_vdev_set_user_data(rank_vdev0, rank_ud0);

  pim_vdev_t *dax_vdev0 =
      pim_vdev_create(dax_root, "dax0.0", pim_vdev_get_major(rank_vdev0), -1);
  upmem_vdev_user_data *dax_ud0 = new upmem_vdev_user_data;
  dax_ud0->type = upmem_vdev_user_data::FD_DAX;
  pim_vdev_set_user_data(dax_vdev0, dax_ud0);

  pim_vsysfs_root_t *rank_vsysfs_root = pim_vsysfs_root_create(
      rank_root, "dpu_rank", "platform", "dpu_region_mem.*");
  pim_vsysfs_root_t *dax_vsysfs_root = pim_vsysfs_root_create(
      dax_root, "dpu_dax", "platform", "dpu_region_mem.*");

  pim_vsysfs_t *rank_vsysfs0 =
      pim_vsysfs_create(rank_vsysfs_root, rank_vdev0, "dpu_region_mem.0");
  pim_vsysfs_t *dax_vsysfs0 =
      pim_vsysfs_create(dax_vsysfs_root, dax_vdev0, "dpu_region_mem.0");

  pim_vsysfs_write_attr(rank_vsysfs0, "rank_id", "0", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "rank_index", "0", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "channel_id", "0", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "backend_id", "0", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "dpu_chip_id", "8", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "nb_ci", "8", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "nb_dpus_per_ci", "8",
                        PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "ci_mask", "255", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "mram_size", "67108864",
                        PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "clock_division", "1",
                        PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "fck_frequency", "400",
                        PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "numa_node", "0", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(
      rank_vsysfs0, "byte_order",
      "0x000103FF0F8FCFEF 0x000103FF0F8FCFEF 0x000103FF0F8FCFEF "
      "0x000103FF0F8FCFEF 0x000103FF0F8FCFEF 0x000103FF0F8FCFEF "
      "0x000103FF0F8FCFEF 0x000103FF0F8FCFEF",
      PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "mode", "2", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "capabilities", "7", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "region_size", "8589934592",
                        PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(rank_vsysfs0, "device",
                        "/sys/devices/platform/dpu_region_mem.0",
                        PIM_ACCESS_MODE_RO);

  pim_vsysfs_write_attr(dax_vsysfs0, "rank_index", "0", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(dax_vsysfs0, "size", "8589934592", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(dax_vsysfs0, "numa", "0", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(dax_vsysfs0, "numa_node", "0", PIM_ACCESS_MODE_RW);
  pim_vsysfs_write_attr(dax_vsysfs0, "device",
                        "/sys/devices/platform/dpu_region_mem.0",
                        PIM_ACCESS_MODE_RO);
}

__attribute__((constructor)) static void hostpimsim_upmem_init(void) {
  if (!overlaysys_is_allowed()) {
    return;
  }

  if (setenv("UPMEM_PROFILE", "backend=hw,regionMode=safe", 1)) {
    return;
  }

  setup_vfile_tree();

  overlaysys_syscall_hook = _pim_vsyscall_dispatch;
}
