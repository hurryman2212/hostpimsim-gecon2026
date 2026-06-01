#pragma once

#include "ci/ci.hh"
#include "dpu/dpu.hh"

/* This instance is used `user_data` for the first-level DPU handler. */
class UPMEMPIMChip {
public:
  /*
   * Execute the committed CI command through the MMIO-dispatch path and cache
   * the decoded payload/metadata in CIState.
   */
  void propagate_ci(uint64_t ci_word);

  CI &ci();
  const CI &ci() const;

  UPMEMDPU &dpu(size_t dpu_idx);
  const UPMEMDPU &dpu(size_t dpu_idx) const;

  size_t get_nr_dpu() const;

private:
  CI ci_{};
  std::array<UPMEMDPU, kChipNumDpus> dpus_{};
};

static constexpr size_t kRankNumDpus = kRankNumBanks * kChipNumDpus;
struct dpu_transfer_mram_abi {
  void *ptr[kRankNumDpus];
  uint32_t offset_in_mram;
  uint32_t size;
};

class UPMEMPIMRank {
public:
  struct MRAMBankTransferRequest {
    uint8_t dpu_local_idx{0};
    uint8_t active_bank_mask{0};
    uint8_t active_bank_indices[kRankNumBanks]{};
    uint8_t active_bank_count{0};
  };

  long host_copy_to_rank(const dpu_transfer_mram_abi *tm);
  long host_copy_from_rank(const dpu_transfer_mram_abi *tm);

  bool host_copy_to_mram_sparse(const MRAMBankTransferRequest &request,
                                const dpu_transfer_mram_abi &transfer);
  bool host_copy_from_mram_sparse(const MRAMBankTransferRequest &request,
                                  const dpu_transfer_mram_abi &transfer);
  bool host_copy_to_mram_bank_group(
      uint8_t dpu_local_idx, uint8_t active_bank_mask, uint32_t offset_in_mram,
      const std::array<const uint8_t *, kRankNumBanks> &src_by_bank,
      size_t size);
  bool host_copy_from_mram_bank_group(
      uint8_t dpu_local_idx, uint8_t active_bank_mask, uint32_t offset_in_mram,
      std::array<uint8_t *, kRankNumBanks> &dst_by_bank, size_t size);

  UPMEMPIMChip &chip(size_t bank_idx);
  const UPMEMPIMChip &chip(size_t bank_idx) const;

  UPMEMDPU &dpu_by_global(size_t dpu_global_idx);
  const UPMEMDPU &dpu_by_global(size_t dpu_global_idx) const;

  pim_device_t *get_device();
  const pim_device_t *get_device() const;
  void set_device(pim_device_t *device);

private:
  /* One chip per bank slot in the rank model. */
  std::array<UPMEMPIMChip, kRankNumBanks> chips_{};

  pim_device_t *device_{nullptr};
};

/* Rank management implementations */

UPMEMPIMRank *upmem_create_rank(pim_vdev_t *vdev);
void upmem_destroy_rank(UPMEMPIMRank *rank);

/* ioctl() implementations */

long upmem_ioctl_write_to_rank(UPMEMPIMRank *rank, unsigned long arg);
long upmem_ioctl_read_from_rank(UPMEMPIMRank *rank, unsigned long arg);
long upmem_ioctl_commit_commands(UPMEMPIMRank *rank, unsigned long arg);
long upmem_ioctl_update_commands(UPMEMPIMRank *rank, unsigned long arg);
long upmem_ioctl_debug_mode(UPMEMPIMRank *rank, unsigned long arg);
long upmem_ioctl_slice_info(UPMEMPIMRank *rank, unsigned long arg);
