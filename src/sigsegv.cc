/**
 * @file sigsegv.cc
 * @brief SIGSEGV-based Write Protection Monitoring Implementation
 *
 * This file contains:
 * - SIGSEGV handler for CTRL_MMIO regions
 * - Instruction decoding for x86_64 and ARM64
 * - Async-signal-safe callback thread using futex
 */

#include "internal.hh"

#include <sys/mman.h>

#include <linux/futex.h>

static constexpr size_t kSigsegvLogBufferSize = 320;

/*============================================================================
 * Async-Signal-Safe Logging Helpers
 *============================================================================*/

static inline size_t sigsegv_log_append_str(char *buf, size_t cap, size_t pos,
                                            const char *s) noexcept {
  while (*s && pos < cap) {
    buf[pos++] = *s++;
  }
  return pos;
}

static inline size_t sigsegv_log_append_hex_uintptr(char *buf, size_t cap,
                                                    size_t pos,
                                                    uintptr_t value) noexcept {
  if (pos < cap) {
    buf[pos++] = '0';
  }
  if (pos < cap) {
    buf[pos++] = 'x';
  }

  bool started = false;
  for (int shift = static_cast<int>(sizeof(uintptr_t) * 8) - 4; shift >= 0;
       shift -= 4) {
    const uint8_t nibble = static_cast<uint8_t>((value >> shift) & 0xF);
    if (nibble != 0 || started || shift == 0) {
      if (pos < cap) {
        buf[pos++] = "0123456789abcdef"[nibble];
      }
      started = true;
    }
  }
  return pos;
}

static void sigsegv_log_decode_miss(uintptr_t fault_addr, uintptr_t ip,
                                    pim_region_t *region) noexcept {
  char buf[kSigsegvLogBufferSize]; // Truncate if overflow
  size_t pos = 0;

  pos = sigsegv_log_append_str(buf, sizeof(buf), pos,
                               "hostpimsim: SIGSEGV decode miss; forwarding");

  pos = sigsegv_log_append_str(buf, sizeof(buf), pos, " ip=");
  pos = sigsegv_log_append_hex_uintptr(buf, sizeof(buf), pos, ip);
  pos = sigsegv_log_append_str(buf, sizeof(buf), pos, " fault=");
  pos = sigsegv_log_append_hex_uintptr(buf, sizeof(buf), pos, fault_addr);

  if (region) {
    pos = sigsegv_log_append_str(buf, sizeof(buf), pos, " type=");
    switch (region->type) {
    case PIM_REGION_CTRL_MMIO:
      pos = sigsegv_log_append_str(buf, sizeof(buf), pos, "CTRL_MMIO");
      break;
    case PIM_REGION_CTRL_RW:
      pos = sigsegv_log_append_str(buf, sizeof(buf), pos, "CTRL_RW");
      break;
    case PIM_REGION_RAM:
      pos = sigsegv_log_append_str(buf, sizeof(buf), pos, "RAM");
      break;
    default:
      pos = sigsegv_log_append_str(buf, sizeof(buf), pos, "UNKNOWN");
      break;
    }

    const uintptr_t region_start = reinterpret_cast<uintptr_t>(region->ptr);
    if (fault_addr >= region_start) {
      pos = sigsegv_log_append_str(buf, sizeof(buf), pos, " region_off=");
      pos = sigsegv_log_append_hex_uintptr(buf, sizeof(buf), pos,
                                           fault_addr - region_start);
    }
  }

  if (pos < sizeof(buf)) {
    buf[pos++] = '\n';
  }

  (void)write(STDERR_FILENO, buf, pos);
}

/*============================================================================
 * SIGSEGV Callback Thread (async-signal-safe design)
 *============================================================================*/

/**
 * @brief Callback thread that waits on futex and invokes handlers
 *
 * This thread waits on a futex word. When the SIGSEGV handler detects a write,
 * it sets the callback data and wakes this thread via futex. This avoids
 * calling non-async-signal-safe functions from the signal handler.
 */
static void sigsegv_callback_thread_func(pim_device_t *dev,
                                         sigsegv_callback_ctx *ctx) {
  while (true) {
    // Wait for work or exit signal
    // FUTEX_WAIT only waits if futex_word == kSigsegvFutexWait.
    while (ctx->futex_word.load(std::memory_order_acquire) ==
           kSigsegvFutexWait) {
      futex(reinterpret_cast<void *>(&ctx->futex_word), FUTEX_WAIT,
            kSigsegvFutexWait, nullptr, nullptr, 0);
    }

    // Read signal value (but don't reset yet - need to copy data first)
    uint32_t signal = ctx->futex_word.load(std::memory_order_acquire);

    // Check for exit signal
    if (signal == kSigsegvFutexExit) {
      return;
    }

    // Copy callback payload locally BEFORE resetting futex_word
    // (otherwise a new signal could overwrite data before we read it)
    pim_region_t *region = ctx->region;
    size_t offset = ctx->offset.load(std::memory_order_relaxed);
    uint64_t value = ctx->value.load(std::memory_order_relaxed);

    // NOW reset futex_word - signal handler can start writing new data
    ctx->futex_word.store(kSigsegvFutexWait, std::memory_order_release);

    // Reuse common dispatch path for pool/direct fallback semantics.
    if (region) {
      // CRUCIAL: SIGSEGV MMIO path intentionally does not commit MMIO memory.
      // Never advance cached bytes here; doing so can create synthetic
      // memory!=snapshot diffs in mixed mode (monitor + SIGSEGV).
      (void)pim_invoke_dpu_handler_internal(dev, region, offset, value, false,
                                            0);
    }
  }
}

/*============================================================================
 * Global State (Stable Linked List)
 *============================================================================*/

/**
 * @brief A node in the device list that is allocated ONCE and never freed.
 *
 * This guarantees that a signal handler holding a pointer to this node
 * will never encounter a Use-After-Free. Disabled devices become "tombstones"
 * (dev == nullptr) that can be reused.
 */
struct SigsegvNode {
  std::atomic<pim_device_t *> dev{nullptr};
  // Number of signal handlers currently scanning this node with a live dev.
  std::atomic<uint32_t> active_with_dev{0};
  std::atomic<SigsegvNode *> next{nullptr};
};

// The head of the device list
static std::atomic<SigsegvNode *> g_device_list_head{nullptr};

struct PageAlignedRange {
  uintptr_t start{0};
  size_t size{0};
};

static inline bool compute_page_aligned_range(uintptr_t addr, size_t span,
                                              size_t page_size,
                                              PageAlignedRange *out) noexcept {
  if (!out || span == 0 || page_size == 0 || (page_size & (page_size - 1))) {
    return false;
  }
  if (addr > std::numeric_limits<uintptr_t>::max() - span) {
    return false;
  }

  const uintptr_t end = addr + span;
  const uintptr_t aligned_start = addr & ~(page_size - 1);
  if (end > std::numeric_limits<uintptr_t>::max() - (page_size - 1)) {
    return false;
  }
  const uintptr_t aligned_end = (end + page_size - 1) & ~(page_size - 1);
  if (aligned_end < aligned_start) {
    return false;
  }

  const uintptr_t range_size_u = aligned_end - aligned_start;
  if (range_size_u > std::numeric_limits<size_t>::max()) {
    return false;
  }

  out->start = aligned_start;
  out->size = static_cast<size_t>(range_size_u);
  return true;
}

static inline void sigsegv_node_release(SigsegvNode *node) {
  if (node) {
    node->active_with_dev.fetch_sub(1, std::memory_order_acq_rel);
  }
}

static inline pim_region_t *find_ctrl_region_for_fault(pim_device_t *dev,
                                                       uintptr_t fault_addr) {
  pim_region_t *ctrl_rw_match = nullptr;
  for (const auto &entry : dev->regions) {
    auto *region = entry.second;
    if (!region) {
      continue;
    }
    const uintptr_t region_start = reinterpret_cast<uintptr_t>(region->ptr);
    const uintptr_t region_end = region_start + region->size;
    if (fault_addr < region_start || fault_addr >= region_end) {
      continue;
    }

    if (region->type == PIM_REGION_CTRL_MMIO) {
      return region;
    }
    if (region->type == PIM_REGION_CTRL_RW && !ctrl_rw_match) {
      ctrl_rw_match = region;
    }
  }
  return ctrl_rw_match;
}

// Tombstone device node in-place and return that node, or nullptr if not found.
static SigsegvNode *sigsegv_tombstone_device_node(pim_device_t *dev) {
  SigsegvNode *node = g_device_list_head.load(std::memory_order_acquire);
  while (node) {
    pim_device_t *current = node->dev.load(std::memory_order_relaxed);
    if (current == dev) {
      node->dev.store(nullptr, std::memory_order_release);
      return node;
    }
    node = node->next.load(std::memory_order_relaxed);
  }
  return nullptr;
}

static void sigsegv_wait_node_quiescent(SigsegvNode *node) {
  while (node && node->active_with_dev.load(std::memory_order_acquire) != 0) {
    std::this_thread::yield();
  }
}

static void sigsegv_signal_callback_threads_exit(
    const std::vector<std::unique_ptr<sigsegv_callback_ctx>> &callbacks) {
  for (const auto &ctx : callbacks) {
    if (!ctx) {
      continue;
    }
    ctx->futex_word.store(kSigsegvFutexExit, std::memory_order_release);
    futex(reinterpret_cast<void *>(&ctx->futex_word), FUTEX_WAKE, 1, nullptr,
          nullptr, 0);
  }
}

static void sigsegv_join_callback_threads(
    const std::vector<std::unique_ptr<sigsegv_callback_ctx>> &callbacks) {
  for (const auto &ctx : callbacks) {
    if (!ctx) {
      continue;
    }
    if (ctx->thread.joinable()) {
      ctx->thread.join();
    }
  }
}

static void sigsegv_clear_callback_threads(
    std::vector<std::unique_ptr<sigsegv_callback_ctx>> &callbacks) {
  callbacks.clear();
}

static inline bool sigsegv_collect_mmio_regions(
    pim_device_t *dev, std::vector<pim_region_t *> *mmio_regions) noexcept {
  if (!dev || !mmio_regions) {
    return false;
  }
  mmio_regions->clear();
  try {
    mmio_regions->reserve(dev->regions.size());
    for (const auto &entry : dev->regions) {
      auto *region = entry.second;
      if (region && region->type == PIM_REGION_CTRL_MMIO) {
        mmio_regions->push_back(region);
      }
    }
  } catch (...) {
    mmio_regions->clear();
    return false;
  }
  return true;
}

static inline void sigsegv_rollback_disarm_readonly(
    const std::vector<pim_region_t *> &mmio_regions,
    size_t processed_mmio) noexcept {
  const size_t limit = (processed_mmio < mmio_regions.size())
                           ? processed_mmio
                           : mmio_regions.size();
  for (size_t i = 0; i < limit; ++i) {
    auto *rollback_region = mmio_regions[i];
    const uintptr_t rb_start =
        reinterpret_cast<uintptr_t>(rollback_region->ptr);
    const size_t page_size = rollback_region->page_size;
    if (page_size == 0) {
      continue;
    }
    PageAlignedRange rollback_range;
    if (compute_page_aligned_range(rb_start, rollback_region->size, page_size,
                                   &rollback_range)) {
      (void)mprotect(reinterpret_cast<void *>(rollback_range.start),
                     rollback_range.size, PROT_READ);
    }
  }
}

/**
 * @brief Disarm MMIO page protections while holding `dev->mutex`.
 */
static pim_error_t
sigsegv_disarm_mmio_regions_locked(pim_device_t *dev) noexcept {
  std::vector<pim_region_t *> mmio_regions;
  if (!sigsegv_collect_mmio_regions(dev, &mmio_regions)) {
    return PIM_ERR_NO_MEMORY;
  }

  size_t processed_mmio = 0;
  for (auto *region : mmio_regions) {
    uintptr_t region_start = reinterpret_cast<uintptr_t>(region->ptr);
    const size_t page_size = region->page_size;
    if (page_size == 0) {
      errno = EINVAL;
      return PIM_ERR_MPROTECT;
    }
    PageAlignedRange region_range;
    if (!compute_page_aligned_range(region_start, region->size, page_size,
                                    &region_range)) {
      errno = EINVAL;
      return PIM_ERR_MPROTECT;
    }

    if (mprotect(reinterpret_cast<void *>(region_range.start),
                 region_range.size, PROT_READ | PROT_WRITE) < 0) {
      const int mprotect_errno = (errno > 0) ? errno : EIO;
      // Keep SIGSEGV mode intact on disarm failure by rolling back pages we
      // already disarmed in this attempt.
      sigsegv_rollback_disarm_readonly(mmio_regions, processed_mmio);
      errno = mprotect_errno;
      return PIM_ERR_MPROTECT;
    }
    ++processed_mmio;
  }

  // Publish disabled after disarming protections.
  dev->sigsegv_enabled.store(false, std::memory_order_release);
  return PIM_SUCCESS;
}

// Other global state
static std::atomic<bool> g_sigsegv_chain_handler{true};
static struct sigaction g_old_sigsegv_action;
// CRUCIAL: Exported (non-static) so topology/handler mutations in
// hostpimsim.cc can serialize with SIGSEGV enable/disable transitions.
std::mutex g_sigsegv_mutex;

/*============================================================================
 * Instruction Decoding
 *============================================================================*/

#if defined(__x86_64__) || defined(_M_X64)
static constexpr int kX86RegMap[16] = {
    REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
    REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};
static inline uint64_t read_x86_reg_value(ucontext_t *ctx, uint8_t reg,
                                          size_t write_size) noexcept {
  greg_t *gregs = ctx->uc_mcontext.gregs;
  uint64_t val = static_cast<uint64_t>(gregs[kX86RegMap[reg]]);
  if (write_size == 1) {
    return val & 0xFF;
  }
  if (write_size == 2) {
    return val & 0xFFFF;
  }
  if (write_size == 4) {
    return val & 0xFFFFFFFF;
  }
  return val;
}

static inline size_t x86_modrm_len(uint8_t modrm, bool rex_b) noexcept {
  const uint8_t mod = (modrm >> 6) & 0x03;
  const uint8_t rm = (modrm & 0x07) | (rex_b ? 8 : 0);
  size_t len = 1;
  if (mod == 0 && (rm & 7) == 5) {
    len += 4;
  } else if (mod == 1) {
    len += 1;
  } else if (mod == 2) {
    len += 4;
  }
  if ((rm & 7) == 4) {
    len += 1;
  }
  return len;
}

/**
 * @brief Decode x86_64 instruction and extract write value
 *
 * Currently handles these store instructions: MOV and MOVNTI.
 * Returns true if successfully decoded, false otherwise.
 *
 * @param rip         Instruction pointer
 * @param ctx         CPU context with register values
 * @param write_size  Output: size of the write (1, 2, 4, or 8 bytes)
 * @param write_value Output: value being written
 * @param insn_len    Output: instruction length for skipping
 */
static bool decode_x86_write(const uint8_t *rip, ucontext_t *ctx,
                             size_t *write_size, uint64_t *write_value,
                             size_t *insn_len) {
  const uint8_t *p = rip;
  bool rex_w = false;  // 64-bit operand
  bool rex_r = false;  // REG extension
  bool rex_b = false;  // R/M extension
  bool has_66 = false; // 16-bit operand override

  // Parse prefixes
  while (true) {
    if (*p == 0x66) {
      has_66 = true;
      p++;
    } else if (*p == 0x67 || *p == 0xF2 || *p == 0xF3 || *p == 0x2E ||
               *p == 0x36 || *p == 0x3E || *p == 0x26 || *p == 0x64 ||
               *p == 0x65) {
      p++; // Skip other prefixes
    } else if ((*p & 0xF0) == 0x40) {
      // REX prefix
      rex_w = (*p & 0x08) != 0;
      rex_r = (*p & 0x04) != 0;
      rex_b = (*p & 0x01) != 0;
      p++;
    } else {
      break;
    }
  }

  // Determine operand size
  size_t opsize = 4; // Default 32-bit
  if (rex_w) {
    opsize = 8;
  } else if (has_66) {
    opsize = 2;
  }

  // Decode opcode
  uint8_t opcode = *p++;

  // Two-byte opcodes (0x0F prefix)
  if (opcode == 0x0F) {
    uint8_t opcode2 = *p++;

    // MOVNTI [mem], reg: 0x0F 0xC3 /r
    if (opcode2 == 0xC3) {
      uint8_t modrm = *p++;
      uint8_t reg = ((modrm >> 3) & 0x07) | (rex_r ? 8 : 0);
      *write_size = opsize;
      *write_value = read_x86_reg_value(ctx, reg, opsize);

      const size_t modrmlen = x86_modrm_len(modrm, rex_b);
      *insn_len = (p - rip) + modrmlen - 1;
      return true;
    }

    return false;
  }

  // MOV [mem], reg (8-bit): 0x88
  if (opcode == 0x88) {
    uint8_t modrm = *p++;
    uint8_t reg = ((modrm >> 3) & 0x07) | (rex_r ? 8 : 0);
    *write_size = 1;
    *write_value = read_x86_reg_value(ctx, reg, 1);

    const size_t modrmlen = x86_modrm_len(modrm, rex_b);
    *insn_len = (p - rip) + modrmlen - 1;
    return true;
  }

  // MOV [mem], reg (16/32/64-bit): 0x89
  if (opcode == 0x89) {
    uint8_t modrm = *p++;
    uint8_t reg = ((modrm >> 3) & 0x07) | (rex_r ? 8 : 0);
    *write_size = opsize;
    *write_value = read_x86_reg_value(ctx, reg, opsize);

    const size_t modrmlen = x86_modrm_len(modrm, rex_b);
    *insn_len = (p - rip) + modrmlen - 1;
    return true;
  }

  // MOV [mem], imm8: 0xC6 /0
  if (opcode == 0xC6) {
    uint8_t modrm = *p++;
    if (((modrm >> 3) & 0x07) != 0)
      return false; // Not MOV

    *write_size = 1;
    const size_t modrmlen = x86_modrm_len(modrm, rex_b);

    const uint8_t *imm_ptr = rip + (p - rip) + modrmlen - 1;
    *write_value = *imm_ptr;
    *insn_len = (p - rip) + modrmlen - 1 + 1;
    return true;
  }

  // MOV [mem], imm16/32: 0xC7 /0
  if (opcode == 0xC7) {
    uint8_t modrm = *p++;
    if (((modrm >> 3) & 0x07) != 0)
      return false;

    *write_size = opsize;
    // Note: imm is 32-bit even for 64-bit operand (sign-extended)
    size_t imm_size = (opsize == 2) ? 2 : 4;
    const size_t modrmlen = x86_modrm_len(modrm, rex_b);

    const uint8_t *imm_ptr = rip + (p - rip) + modrmlen - 1;
    if (imm_size == 2) {
      *write_value = *reinterpret_cast<const uint16_t *>(imm_ptr);
    } else {
      int32_t imm32 = *reinterpret_cast<const int32_t *>(imm_ptr);
      if (opsize == 8) {
        *write_value = static_cast<uint64_t>(static_cast<int64_t>(imm32));
      } else {
        *write_value = static_cast<uint32_t>(imm32);
      }
    }
    *insn_len = (p - rip) + modrmlen - 1 + imm_size;
    return true;
  }

  return false; // Unrecognized instruction
}
#endif // __x86_64__

#if defined(__aarch64__)
/**
 * @brief Decode ARM64 store instruction and extract write value
 */
static bool decode_arm64_write(uint32_t insn, ucontext_t *ctx,
                               size_t *write_size, uint64_t *write_value,
                               size_t *insn_len) {
  *insn_len = 4; // All ARM64 instructions are 4 bytes

  // STR (immediate) - unsigned offset: 1x111001 00xxxxxx xxxxxxxx xxxRtttt
  // STRB: 00111001 00...
  // STRH: 01111001 00...
  // STR 32: 10111001 00...
  // STR 64: 11111001 00...
  uint32_t opc = (insn >> 30) & 0x03;
  uint32_t op1 = (insn >> 22) & 0x3FF;

  if ((op1 & 0x3F9) == 0x1C1) { // Store unsigned offset pattern
    uint32_t rt = insn & 0x1F;
    uint64_t xreg = ctx->uc_mcontext.regs[rt];

    switch (opc) {
    case 0:
      *write_size = 1;
      *write_value = xreg & 0xFF;
      return true;
    case 1:
      *write_size = 2;
      *write_value = xreg & 0xFFFF;
      return true;
    case 2:
      *write_size = 4;
      *write_value = xreg & 0xFFFFFFFF;
      return true;
    case 3:
      *write_size = 8;
      *write_value = xreg;
      return true;
    }
  }

  // STUR (unscaled immediate): xx111000 000xxxxx xxxx00xx xxxRtttt
  if ((op1 & 0x3F8) == 0x1C0 && ((insn >> 10) & 0x03) == 0) {
    uint32_t rt = insn & 0x1F;
    uint64_t xreg = ctx->uc_mcontext.regs[rt];

    switch (opc) {
    case 0:
      *write_size = 1;
      *write_value = xreg & 0xFF;
      return true;
    case 1:
      *write_size = 2;
      *write_value = xreg & 0xFFFF;
      return true;
    case 2:
      *write_size = 4;
      *write_value = xreg & 0xFFFFFFFF;
      return true;
    case 3:
      *write_size = 8;
      *write_value = xreg;
      return true;
    }
  }

  return false;
}
#endif // __aarch64__

/*============================================================================
 * SIGSEGV Handler
 *============================================================================*/

/**
 * @brief SIGSEGV handler - detects writes to CTRL_MMIO regions
 *
 * For MMIO-style control regions, we:
 * 1. Decode the write instruction to get the value
 * 2. Skip the instruction (no actual write - avoids double-processing)
 * 3. Wake the callback thread via futex to invoke the handler
 */
bool _pim_sigsegv_handler_func(siginfo_t *info, void *ucontext) noexcept {
  {
    uintptr_t fault_addr = reinterpret_cast<uintptr_t>(info->si_addr);
    auto *ctx = static_cast<ucontext_t *>(ucontext);

    // Find which device and control region this address belongs to
    pim_device_t *dev = nullptr;
    pim_region_t *target_region = nullptr;
    SigsegvNode *held_node = nullptr;

    // Walk the linked list (lock-free, allocation-free, async-signal-safe)
    SigsegvNode *node = g_device_list_head.load(std::memory_order_acquire);

    while (node) {
      // First snapshot of device pointer from this node.
      pim_device_t *d = node->dev.load(std::memory_order_acquire);
      if (!d) {
        node = node->next.load(std::memory_order_acquire);
        continue;
      }

      // CRUCIAL: Acquire node guard BEFORE touching d->regions, then re-check
      // node->dev. This closes the race with disable/destroy (tombstone/free).
      node->active_with_dev.fetch_add(1, std::memory_order_acq_rel);
      pim_device_t *stable = node->dev.load(std::memory_order_acquire);
      if (stable != d || !stable ||
          !stable->sigsegv_enabled.load(std::memory_order_relaxed)) {
        sigsegv_node_release(node);
        node = node->next.load(std::memory_order_acquire);
        continue;
      }
      d = stable;

      // Single pass with priority:
      // - Prefer CTRL_MMIO (handler-capable)
      // - Fallback to CTRL_RW (status-write replay only)
      target_region = find_ctrl_region_for_fault(d, fault_addr);

      if (target_region) {
        dev = d;
        held_node = node;
        break;
      }

      sigsegv_node_release(node);
      node = node->next.load(std::memory_order_acquire);
    }

    if (!target_region) {
      sigsegv_node_release(held_node);
      goto chain_handler;
    }

    // Decode the write instruction
    size_t write_size = 0;
    uint64_t write_value = 0;
    size_t insn_len = 0;

#if defined(__x86_64__) || defined(_M_X64)
    const uint8_t *rip =
        reinterpret_cast<const uint8_t *>(ctx->uc_mcontext.gregs[REG_RIP]);
    if (!decode_x86_write(rip, ctx, &write_size, &write_value, &insn_len)) {
      // Can't decode - fall through to chain
      sigsegv_log_decode_miss(fault_addr, reinterpret_cast<uintptr_t>(rip),
                              target_region);
      sigsegv_node_release(held_node);
      goto chain_handler;
    }
#elif defined(__aarch64__)
    const uint8_t *pc = reinterpret_cast<const uint8_t *>(ctx->uc_mcontext.pc);
    uint32_t insn = *reinterpret_cast<const uint32_t *>(pc);
    if (!decode_arm64_write(insn, ctx, &write_size, &write_value, &insn_len)) {
      sigsegv_log_decode_miss(fault_addr, reinterpret_cast<uintptr_t>(pc),
                              target_region);
      sigsegv_node_release(held_node);
      goto chain_handler;
    }
#else
    sigsegv_log_decode_miss(fault_addr, 0, target_region);
    sigsegv_node_release(held_node);
    goto chain_handler;
#endif

    // Calculate offset within region
    uintptr_t region_start = reinterpret_cast<uintptr_t>(target_region->ptr);
    size_t offset_in_region = fault_addr - region_start;

    // Find handler for this offset (only CTRL_MMIO can have handlers)
    bool has_handler = false;
    size_t handler_offset = 0;

    if (target_region->type == PIM_REGION_CTRL_MMIO) {
      for (auto &[offset, entry] : target_region->handlers) {
        if (offset_in_region >= offset &&
            offset_in_region < offset + entry.register_size) {
          has_handler = true;
          handler_offset = offset;
          break;
        }
      }
    }

    // Apply the write to CTRL_RW regions so the host can read status back.
    // CTRL_MMIO writes are NOT applied - the handler will process the value
    // passed via futex, and not writing prevents polling monitor from
    // triggering duplicate handler invocations.
    const bool should_write = (target_region->type == PIM_REGION_CTRL_RW);
    if (should_write && write_size > 0) {
      const size_t page_size = target_region->page_size;
      if (page_size == 0) {
        sigsegv_node_release(held_node);
        goto chain_handler;
      }
      uintptr_t target_addr = region_start + offset_in_region;
      PageAlignedRange write_range;
      if (!compute_page_aligned_range(target_addr, write_size, page_size,
                                      &write_range)) {
        sigsegv_node_release(held_node);
        goto chain_handler;
      }

      // CRUCIAL: Unprotect the full write span, not just the first page.
      // Unaligned 4/8-byte stores can cross page boundaries.
      if (mprotect(reinterpret_cast<void *>(write_range.start),
                   write_range.size, PROT_READ | PROT_WRITE) == 0) {
        // Byte stores avoid UB on unaligned addresses in signal context.
        volatile uint8_t *dst =
            reinterpret_cast<volatile uint8_t *>(target_addr);
        const uint8_t *src = reinterpret_cast<const uint8_t *>(&write_value);
        for (size_t i = 0; i < write_size; ++i) {
          dst[i] = src[i];
        }
        mprotect(reinterpret_cast<void *>(write_range.start), write_range.size,
                 PROT_READ);
      } else {
        // CRUCIAL: If we cannot replay the write, do not skip instruction.
        // Forward to chain/default SIGSEGV path instead of silently dropping.
        sigsegv_node_release(held_node);
        goto chain_handler;
      }
    }

    // Skip the faulting instruction so execution resumes at the next one.
    // - CTRL_MMIO: value NOT written, handler invoked via futex
    // - CTRL_RW: value written above, no handler (for status readback)
#if defined(__x86_64__) || defined(_M_X64)
    ctx->uc_mcontext.gregs[REG_RIP] += static_cast<greg_t>(insn_len);
#elif defined(__aarch64__)
    ctx->uc_mcontext.pc += insn_len;
#endif

    // Call handler if found (async-signal-safe: use futex to wake callback
    // thread)
    //
    // NOTE: If multiple SIGSEGVs arrive before the callback thread processes
    // the first one, only the last write's handler invocation will occur.
    // This is intentional: applications must ensure proper ordering of
    // CTRL_MMIO writes (e.g., using memory barriers or waiting for completion)
    // just as they would with real hardware. Rapid fire-and-forget writes
    // are coalesced to the final value, matching polling monitor behavior.
    if (has_handler) {
      // Find the callback context for this region
      for (const auto &cb_ctx : dev->sigsegv_callbacks) {
        if (!cb_ctx || cb_ctx->region != target_region) {
          continue;
        }
        // Set callback data (all atomic stores)
        cb_ctx->offset.store(handler_offset, std::memory_order_relaxed);
        cb_ctx->value.store(write_value, std::memory_order_relaxed);

        // Signal the callback thread (async-signal-safe)
        cb_ctx->futex_word.store(kSigsegvFutexWork, std::memory_order_release);
        futex(reinterpret_cast<void *>(&cb_ctx->futex_word), FUTEX_WAKE, 1,
              nullptr, nullptr, 0);
        break;
      }
    }

    sigsegv_node_release(held_node);
    return false; // Processed - do not forward
  }

chain_handler:
  // Not our fault - return true to indicate should forward to next handler
  return true;
}
static void sigsegv_handler(int sig, siginfo_t *info, void *ucontext) {
  if (!_pim_sigsegv_handler_func(info, ucontext)) {
    // Processed - do not forward
    return;
  }

  // Not processed - check if we should chain to old handler
  if (!g_sigsegv_chain_handler.load(std::memory_order_relaxed)) {
    // CRUCIAL: Returning from a synchronous SIGSEGV handler retries the same
    // faulting instruction and can spin forever. Force default SIGSEGV action.
    signal(SIGSEGV, SIG_DFL);
    raise(SIGSEGV);
    return; // Unreachable in normal cases, but keeps control-flow explicit.
  }

  // Chain to old handler
  if (g_old_sigsegv_action.sa_flags & SA_SIGINFO) {
    g_old_sigsegv_action.sa_sigaction(sig, info, ucontext);
  } else if (g_old_sigsegv_action.sa_handler != SIG_IGN &&
             g_old_sigsegv_action.sa_handler != SIG_DFL) {
    g_old_sigsegv_action.sa_handler(sig);
  } else {
    signal(SIGSEGV, SIG_DFL);
    raise(SIGSEGV);
  }
}

static inline pim_error_t
sigsegv_query_install_state(bool *is_installed) noexcept {
  if (!is_installed) {
    return PIM_ERR_INVALID_ARG;
  }
  struct sigaction current_sa;
  if (sigaction(SIGSEGV, nullptr, &current_sa) < 0) {
    return PIM_ERR_SIGNAL;
  }
  *is_installed = (current_sa.sa_flags & SA_SIGINFO) &&
                  (current_sa.sa_sigaction == sigsegv_handler);
  return PIM_SUCCESS;
}

/**
 * @brief Validate device pointer and transition gate for per-device SIGSEGV
 * ops.
 *
 * CRUCIAL: Caller must hold `g_sigsegv_mutex`.
 */
static inline pim_error_t
sigsegv_validate_device_transition_locked(pim_device_t *dev) noexcept {
  if (!dev) {
    return PIM_ERR_INVALID_ARG;
  }
  if (dev->sigsegv_transition_in_progress.load(std::memory_order_acquire)) {
    return PIM_ERR_BUSY;
  }
  return PIM_SUCCESS;
}

/*============================================================================
 * Public API
 *============================================================================*/

pim_error_t pim_g_enable_sigsegv_handler(bool enable_chain) noexcept {
  std::lock_guard<std::mutex> lock(g_sigsegv_mutex);

  // Set chain handler setting
  g_sigsegv_chain_handler.store(enable_chain, std::memory_order_relaxed);

  bool installed = false;
  pim_error_t state_err = sigsegv_query_install_state(&installed);
  if (state_err != PIM_SUCCESS) {
    return state_err;
  }
  if (installed) {
    return PIM_SUCCESS;
  }

  // Install SIGSEGV handler
  struct sigaction sa;
  sa.sa_sigaction = sigsegv_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO | SA_RESTART;

  if (sigaction(SIGSEGV, &sa, &g_old_sigsegv_action) < 0) {
    return PIM_ERR_SIGNAL;
  }

  return PIM_SUCCESS;
}

pim_error_t pim_g_disable_sigsegv_handler(void) noexcept {
  std::lock_guard<std::mutex> lock(g_sigsegv_mutex);

  // Ensure all devices are disabled first (check for any non-tombstone nodes)
  SigsegvNode *node = g_device_list_head.load(std::memory_order_relaxed);
  while (node) {
    if (node->dev.load(std::memory_order_relaxed) != nullptr) {
      return PIM_ERR_BUSY;
    }
    node = node->next.load(std::memory_order_relaxed);
  }

  bool installed = false;
  pim_error_t state_err = sigsegv_query_install_state(&installed);
  if (state_err != PIM_SUCCESS) {
    return state_err;
  }
  if (!installed) {
    return PIM_SUCCESS;
  }

  // Restore old SIGSEGV handler
  sigaction(SIGSEGV, &g_old_sigsegv_action, nullptr);

  // Note: We intentionally do NOT free the linked list nodes.
  // They are kept permanently to avoid UAF in signal handlers.
  // This is a small, bounded memory "leak" that ensures safety.

  return PIM_SUCCESS;
}

pim_error_t pim_device_enable_sigsegv(pim_device_t *dev) noexcept {
  std::lock_guard<std::mutex> lock(g_sigsegv_mutex);

  pim_error_t transition_err = sigsegv_validate_device_transition_locked(dev);
  if (transition_err != PIM_SUCCESS) {
    return transition_err;
  }

  if (dev->teardown_in_progress.load(std::memory_order_acquire)) {
    return PIM_ERR_BUSY;
  }

  // Do NOT check global enabled flag here! (for LD_PRELOAD usage)

  // Check if device already enabled
  if (dev->sigsegv_enabled.load()) {
    return PIM_ERR_ALREADY_EXISTS;
  }

  std::vector<pim_region_t *> mmio_regions;
  std::vector<std::pair<uintptr_t, size_t>> protected_ranges;
  bool listed_in_global = false;

  auto cleanup_after_failure = [&]() {
    // Revert page protections applied before the failure.
    for (const auto &range : protected_ranges) {
      mprotect(reinterpret_cast<void *>(range.first), range.second,
               PROT_READ | PROT_WRITE);
    }

    // CRUCIAL: Tombstone + wait before freeing callback contexts. This avoids
    // racing signal-handler iteration over dev->sigsegv_callbacks.
    if (listed_in_global) {
      SigsegvNode *owner_node = sigsegv_tombstone_device_node(dev);
      sigsegv_wait_node_quiescent(owner_node);
    }

    sigsegv_signal_callback_threads_exit(dev->sigsegv_callbacks);
    sigsegv_join_callback_threads(dev->sigsegv_callbacks);
    sigsegv_clear_callback_threads(dev->sigsegv_callbacks);
    dev->sigsegv_enabled.store(false, std::memory_order_release);
  };

  try {
    // 1. Prepare: Spawn callback threads for each CTRL_MMIO region
    {
      std::lock_guard<std::mutex> dlock(dev->mutex);
      if (!sigsegv_collect_mmio_regions(dev, &mmio_regions)) {
        return PIM_ERR_NO_MEMORY;
      }
      for (auto *region : mmio_regions) {

        auto ctx = std::make_unique<sigsegv_callback_ctx>();
        ctx->region = region;
        ctx->futex_word.store(kSigsegvFutexWait);
        ctx->thread = std::thread(sigsegv_callback_thread_func, dev, ctx.get());
        try {
          dev->sigsegv_callbacks.push_back(std::move(ctx));
        } catch (...) {
          ctx->futex_word.store(kSigsegvFutexExit, std::memory_order_release);
          futex(reinterpret_cast<void *>(&ctx->futex_word), FUTEX_WAKE, 1,
                nullptr, nullptr, 0);
          if (ctx->thread.joinable()) {
            ctx->thread.join();
          }
          throw;
        }
      }
    }

    if (mmio_regions.empty()) {
      return PIM_ERR_NOT_FOUND;
    }

    // 2. Add device to linked list (try to reuse tombstone, else prepend)
    {
      bool stored = false;

      // First, try to reuse a tombstone slot (node with dev == nullptr)
      SigsegvNode *node = g_device_list_head.load(std::memory_order_acquire);
      while (node) {
        pim_device_t *expected = nullptr;
        if (node->dev.compare_exchange_strong(expected, dev,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
          stored = true;
          listed_in_global = true;
          break;
        }
        node = node->next.load(std::memory_order_relaxed);
      }

      // If no tombstone found, prepend a new node
      if (!stored) {
        SigsegvNode *new_node = new SigsegvNode();
        new_node->dev.store(dev, std::memory_order_relaxed);

        // Prepend to list: new_node->next = old_head, then head = new_node
        SigsegvNode *old_head =
            g_device_list_head.load(std::memory_order_relaxed);
        new_node->next.store(old_head, std::memory_order_relaxed);
        g_device_list_head.store(new_node, std::memory_order_release);
        listed_in_global = true;
      }
    }

    // 3. Arm: Write-protect all CTRL_MMIO regions
    bool any_failed = false;
    int mprotect_errno = 0;
    {
      for (auto *region : mmio_regions) {
        uintptr_t region_start = reinterpret_cast<uintptr_t>(region->ptr);
        const size_t page_size = region->page_size;
        if (page_size == 0) {
          mprotect_errno = EINVAL;
          any_failed = true;
          break;
        }
        PageAlignedRange region_range;
        if (!compute_page_aligned_range(region_start, region->size, page_size,
                                        &region_range)) {
          mprotect_errno = EINVAL;
          any_failed = true;
          break;
        }

        if (mprotect(reinterpret_cast<void *>(region_range.start),
                     region_range.size, PROT_READ) < 0) {
          mprotect_errno = (errno > 0) ? errno : EIO;
          any_failed = true;
          break;
        }
        protected_ranges.emplace_back(region_range.start, region_range.size);
      }

      if (!any_failed) {
        dev->sigsegv_enabled.store(true, std::memory_order_release);
      }
    }

    if (any_failed) {
      const int fail_errno = (mprotect_errno > 0) ? mprotect_errno : EIO;
      cleanup_after_failure();
      errno = fail_errno;
      return PIM_ERR_MPROTECT;
    }

    return PIM_SUCCESS;
  } catch (...) {
    // CRUCIAL: C API must not leak C++ exceptions across the ABI boundary.
    cleanup_after_failure();
    return PIM_ERR_NO_MEMORY;
  }
}

pim_error_t pim_device_disable_sigsegv(pim_device_t *dev) noexcept {
  std::unique_lock<std::mutex> lock(g_sigsegv_mutex);

  pim_error_t transition_err = sigsegv_validate_device_transition_locked(dev);
  if (transition_err != PIM_SUCCESS) {
    return transition_err;
  }

  // CRUCIAL: A SIGSEGV callback worker cannot disable+join itself.
  // Require a control thread to perform disable transition.
  if (pim_current_thread_in_sigsegv_callbacks_locked(dev->sigsegv_callbacks)) {
    return PIM_ERR_BUSY;
  }
  if (!dev->sigsegv_enabled.load()) {
    return PIM_SUCCESS; // Already disabled
  }

  // Block concurrent enable/disable while this disable transitions and joins.
  dev->sigsegv_transition_in_progress.store(true, std::memory_order_release);

  // 1. Disarm first: avoid a window where pages are still read-only but the
  // device is already tombstoned (which would turn valid writes into crashes).
  {
    std::lock_guard<std::mutex> dlock(dev->mutex);
    pim_error_t disarm_err = sigsegv_disarm_mmio_regions_locked(dev);
    if (disarm_err != PIM_SUCCESS) {
      dev->sigsegv_transition_in_progress.store(false,
                                                std::memory_order_release);
      return disarm_err;
    }
  }

  // 2. Tombstone and wait for any in-flight signal handlers on this node.
  SigsegvNode *owner_node = sigsegv_tombstone_device_node(dev);
  sigsegv_wait_node_quiescent(owner_node);

  // 3. Signal callback threads to exit while still under global transition
  // lock, then release the lock and join outside to avoid lock inversion
  // with user callbacks that also acquire g_sigsegv_mutex.
  sigsegv_signal_callback_threads_exit(dev->sigsegv_callbacks);
  lock.unlock();

  sigsegv_join_callback_threads(dev->sigsegv_callbacks);

  lock.lock();
  sigsegv_clear_callback_threads(dev->sigsegv_callbacks);
  dev->sigsegv_transition_in_progress.store(false, std::memory_order_release);

  return PIM_SUCCESS;
}

bool pim_is_sigsegv_enabled(pim_device_t *dev) noexcept {
  if (!dev) {
    return false;
  }
  return dev->sigsegv_enabled.load();
}

void pim_g_set_sigsegv_chain_handler(bool chain) noexcept {
  std::lock_guard<std::mutex> lock(g_sigsegv_mutex);
  g_sigsegv_chain_handler.store(chain, std::memory_order_relaxed);
}

bool pim_g_is_sigsegv_chain_handler_enabled(void) noexcept {
  std::lock_guard<std::mutex> lock(g_sigsegv_mutex);
  return g_sigsegv_chain_handler.load(std::memory_order_relaxed);
}

bool pim_g_is_sigsegv_handler_enabled(void) noexcept {
  std::lock_guard<std::mutex> lock(g_sigsegv_mutex);
  bool is_installed = false;
  const pim_error_t ret = sigsegv_query_install_state(&is_installed);
  if (ret != PIM_SUCCESS) {
    return false;
  }
  return is_installed;
}
