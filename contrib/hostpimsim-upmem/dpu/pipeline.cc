#include "dpu.hh"
#include "../ci/ci.hh"
#include "runtime.hh"

/* Merged runtime/decode/execute/control/ci pipeline translation unit. */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <vector>

constexpr uint8_t kNumWorkRegistersPerThread = 24;

/* Keep this aligned with host shim's MRAM bounds checks. */
constexpr size_t kWramSizeBytes = WRAM_SIZE;
constexpr size_t kWramWords = kWramSizeBytes / sizeof(uint32_t);

constexpr size_t kIramWriteWordBytes = 6;
constexpr size_t kIramSlotBytes = 8;
constexpr uint32_t kTaskletMask = (1u << kNumTasklets) - 1u;
constexpr size_t kDefaultMaxExecSteps = 1000000000u;

static_assert(WRAM_OFFSET + WRAM_SIZE <= PRIVATE_MEM_SIZE,
              "WRAM region must fit private memory");

static inline size_t get_max_exec_steps() {
  static const size_t value = [] {
    size_t parsed_value = kDefaultMaxExecSteps;
    if (const char *env = std::getenv("HOSTPIMSIM_UPMEM_MAX_EXEC_STEPS")) {
      char *end = nullptr;
      const unsigned long long parsed = std::strtoull(env, &end, 10);
      if (end != env && *end == '\0' && parsed > 0ull) {
        parsed_value = static_cast<size_t>(parsed);
      }
    }
    return parsed_value;
  }();

  return value;
}

constexpr uint64_t kCiIdentity = 0x01FF000000000000ULL;
constexpr uint64_t kCiByteOrder = 0x7777777777777777ULL;

union dpu_reg {
  uint64_t d;
  uint32_t r[2];
};

struct dpu_regfile {
  union dpu_reg regs[12];
  uint32_t thread_id;

  bool ZF;
  bool CF;

  uint32_t pc;

  uint32_t perf_counter;
  uint8_t perf_counter_mode;

  bool stopped;

  uint32_t replay_pending_wmask;
  uint8_t replay_src0;
  uint8_t replay_src1;

  static constexpr uint32_t zero = 0;
  static constexpr uint32_t one = 1;
  static constexpr uint32_t lneg = 0xffffffff;
  static constexpr uint32_t mneg = 0x80000000;
  uint32_t id() const { return thread_id; }
  uint32_t id2() const { return thread_id * 2; }
  uint32_t id4() const { return thread_id * 4; }
  uint32_t id8() const { return thread_id * 8; }

  uint64_t &d0() { return regs[0].d; }
  uint64_t &d2() { return regs[1].d; }
  uint64_t &d4() { return regs[2].d; }
  uint64_t &d6() { return regs[3].d; }
  uint64_t &d8() { return regs[4].d; }
  uint64_t &d10() { return regs[5].d; }
  uint64_t &d12() { return regs[6].d; }
  uint64_t &d14() { return regs[7].d; }
  uint64_t &d16() { return regs[8].d; }
  uint64_t &d18() { return regs[9].d; }
  uint64_t &d20() { return regs[10].d; }
  uint64_t &d22() { return regs[11].d; }

  uint32_t &r0() { return regs[0].r[0]; }
  uint32_t &r1() { return regs[0].r[1]; }
  uint32_t &r2() { return regs[1].r[0]; }
  uint32_t &r3() { return regs[1].r[1]; }
  uint32_t &r4() { return regs[2].r[0]; }
  uint32_t &r5() { return regs[2].r[1]; }
  uint32_t &r6() { return regs[3].r[0]; }
  uint32_t &r7() { return regs[3].r[1]; }
  uint32_t &r8() { return regs[4].r[0]; }
  uint32_t &r9() { return regs[4].r[1]; }
  uint32_t &r10() { return regs[5].r[0]; }
  uint32_t &r11() { return regs[5].r[1]; }
  uint32_t &r12() { return regs[6].r[0]; }
  uint32_t &r13() { return regs[6].r[1]; }
  uint32_t &r14() { return regs[7].r[0]; }
  uint32_t &r15() { return regs[7].r[1]; }
  uint32_t &r16() { return regs[8].r[0]; }
  uint32_t &r17() { return regs[8].r[1]; }
  uint32_t &r18() { return regs[9].r[0]; }
  uint32_t &r19() { return regs[9].r[1]; }
  uint32_t &r20() { return regs[10].r[0]; }
  uint32_t &r21() { return regs[10].r[1]; }
  uint32_t &r22() { return regs[11].r[0]; }
  uint32_t &r23() { return regs[11].r[1]; }

  uint32_t read_reg(uint8_t idx) const {
    if (idx < 24) {
      return regs[idx / 2].r[idx % 2];
    }
    switch (idx) {
    case 24:
      return zero;
    case 25:
      return one;
    case 26:
      return lneg;
    case 27:
      return mneg;
    case 28:
      return id();
    case 29:
      return id2();
    case 30:
      return id4();
    case 31:
      return id8();
    default:
      return 0;
    }
  }

  void write_reg(uint8_t idx, uint32_t val) {
    if (idx < 24) {
      regs[idx / 2].r[idx % 2] = val;
    }
  }

  uint64_t read_dreg(uint8_t idx) const {
    if (idx < 12) {
      const uint64_t hi = static_cast<uint64_t>(regs[idx].r[0]);
      const uint64_t lo = static_cast<uint64_t>(regs[idx].r[1]);
      return (hi << 32) | lo;
    }
    return 0;
  }

  void write_dreg(uint8_t idx, uint64_t val) {
    if (idx < 12) {
      regs[idx].r[0] = static_cast<uint32_t>((val >> 32) & 0xFFFFFFFFu);
      regs[idx].r[1] = static_cast<uint32_t>(val & 0xFFFFFFFFu);
    }
  }

  void clear_conditions() {
    ZF = false;
    CF = false;
  }
};

enum dpu_opcode : uint8_t {
  OP_ACQUIRE = 0x00,
  OP_RELEASE = 0x01,
  OP_ADD = 0x02,
  OP_ADDC = 0x03,
  OP_AND = 0x04,
  OP_ANDN = 0x05,
  OP_ASR = 0x06,
  OP_CAO = 0x07,
  OP_CLO = 0x08,
  OP_CLS = 0x09,
  OP_CLZ = 0x0A,
  OP_CMPB4 = 0x0B,
  OP_DIV_STEP = 0x0C,
  OP_EXTSB = 0x0D,
  OP_EXTSH = 0x0E,
  OP_EXTUB = 0x0F,
  OP_EXTUH = 0x10,
  OP_LSL = 0x11,
  OP_LSL_ADD = 0x12,
  OP_LSL_SUB = 0x13,
  OP_LSL1 = 0x14,
  OP_LSL1X = 0x15,
  OP_LSLX = 0x16,
  OP_LSR = 0x17,
  OP_LSR_ADD = 0x18,
  OP_LSR1 = 0x19,
  OP_LSR1X = 0x1A,
  OP_LSRX = 0x1B,
  OP_MUL_SH_SH = 0x1C,
  OP_MUL_SH_SL = 0x1D,
  OP_MUL_SH_UH = 0x1E,
  OP_MUL_SH_UL = 0x1F,
  OP_MUL_SL_SH = 0x20,
  OP_MUL_SL_SL = 0x21,
  OP_MUL_SL_UH = 0x22,
  OP_MUL_SL_UL = 0x23,
  OP_MUL_STEP = 0x24,
  OP_MUL_UH_UH = 0x25,
  OP_MUL_UH_UL = 0x26,
  OP_MUL_UL_UH = 0x27,
  OP_MUL_UL_UL = 0x28,
  OP_NAND = 0x29,
  OP_NOR = 0x2A,
  OP_NXOR = 0x2B,
  OP_OR = 0x2C,
  OP_ORN = 0x2D,
  OP_ROL = 0x2E,
  OP_ROL_ADD = 0x2F,
  OP_ROR = 0x30,
  OP_RSUB = 0x31,
  OP_RSUBC = 0x32,
  OP_SUB = 0x33,
  OP_SUBC = 0x34,
  OP_XOR = 0x35,
  OP_BOOT = 0x36,
  OP_RESUME = 0x37,
  OP_STOP = 0x38,
  OP_CALL = 0x39,
  OP_FAULT = 0x3A,
  OP_NOP = 0x3B,
  OP_SATS = 0x3C,
  OP_MOVD = 0x3D,
  OP_SWAPD = 0x3E,
  OP_HASH = 0x3F,
  OP_TIME = 0x40,
  OP_TIME_CFG = 0x41,
  OP_LBS = 0x42,
  OP_LBU = 0x43,
  OP_LD = 0x44,
  OP_LHS = 0x45,
  OP_LHU = 0x46,
  OP_LW = 0x47,
  OP_SB = 0x48,
  OP_SB_ID = 0x49,
  OP_SD = 0x4A,
  OP_SD_ID = 0x4B,
  OP_SH = 0x4C,
  OP_SH_ID = 0x4D,
  OP_SW = 0x4E,
  OP_SW_ID = 0x4F,
  OP_LDMA = 0x50,
  OP_LDMAI = 0x51,
  OP_SDMA = 0x52,
};

enum dpu_cond : uint8_t {
  CC_TRUE = 0,
  CC_FALSE = 1,
  CC_Z = 2,
  CC_NZ = 3,
  CC_E = 4,
  CC_O = 5,
  CC_PL = 6,
  CC_MI = 7,
  CC_OV = 8,
  CC_NOV = 9,
  CC_C = 10,
  CC_NC = 11,
  CC_SZ = 12,
  CC_SNZ = 13,
  CC_SPL = 14,
  CC_SMI = 15,
  CC_SO = 16,
  CC_SE = 17,
  CC_NC5 = 18,
  CC_NC6 = 19,
  CC_NC7 = 20,
  CC_NC8 = 21,
  CC_NC9 = 22,
  CC_NC10 = 23,
  CC_NC11 = 24,
  CC_NC12 = 25,
  CC_NC13 = 26,
  CC_NC14 = 27,
  CC_MAX = 28,
  CC_NMAX = 29,
  CC_SH32 = 30,
  CC_NSH32 = 31,
  CC_EQ = 32,
  CC_NEQ = 33,
  CC_LTU = 34,
  CC_LEU = 35,
  CC_GTU = 36,
  CC_GEU = 37,
  CC_LTS = 38,
  CC_LES = 39,
  CC_GTS = 40,
  CC_GES = 41,
  CC_XZ = 42,
  CC_XNZ = 43,
  CC_XLEU = 44,
  CC_XGTU = 45,
  CC_XLES = 46,
  CC_XGTS = 47,
  CC_SMALL = 48,
  CC_LARGE = 49,
};

enum dpu_suffix : uint8_t {
  SUFFIX_RICI = 0,
  SUFFIX_RRI = 1,
  SUFFIX_RRIC = 2,
  SUFFIX_RRICI = 3,
  SUFFIX_RRIF = 4,
  SUFFIX_RRR = 5,
  SUFFIX_RRRC = 6,
  SUFFIX_RRRCI = 7,
  SUFFIX_ZRI = 8,
  SUFFIX_ZRIC = 9,
  SUFFIX_ZRICI = 10,
  SUFFIX_ZRIF = 11,
  SUFFIX_ZRR = 12,
  SUFFIX_ZRRC = 13,
  SUFFIX_ZRRCI = 14,
  SUFFIX_S_RRI = 15,
  SUFFIX_S_RRIC = 16,
  SUFFIX_S_RRICI = 17,
  SUFFIX_S_RRIF = 18,
  SUFFIX_S_RRR = 19,
  SUFFIX_S_RRRC = 20,
  SUFFIX_S_RRRCI = 21,
  SUFFIX_U_RRI = 22,
  SUFFIX_U_RRIC = 23,
  SUFFIX_U_RRICI = 24,
  SUFFIX_U_RRIF = 25,
  SUFFIX_U_RRR = 26,
  SUFFIX_U_RRRC = 27,
  SUFFIX_U_RRRCI = 28,
  SUFFIX_RR = 29,
  SUFFIX_RRC = 30,
  SUFFIX_RRCI = 31,
  SUFFIX_ZR = 32,
  SUFFIX_ZRC = 33,
  SUFFIX_ZRCI = 34,
  SUFFIX_S_RR = 35,
  SUFFIX_S_RRC = 36,
  SUFFIX_S_RRCI = 37,
  SUFFIX_U_RR = 38,
  SUFFIX_U_RRC = 39,
  SUFFIX_U_RRCI = 40,
  SUFFIX_DRDICI = 41,
  SUFFIX_RRRI = 42,
  SUFFIX_RRRICI = 43,
  SUFFIX_ZRRI = 44,
  SUFFIX_ZRRICI = 45,
  SUFFIX_S_RRRI = 46,
  SUFFIX_S_RRRICI = 47,
  SUFFIX_U_RRRI = 48,
  SUFFIX_U_RRRICI = 49,
  SUFFIX_RIR = 50,
  SUFFIX_RIRC = 51,
  SUFFIX_RIRCI = 52,
  SUFFIX_ZIR = 53,
  SUFFIX_ZIRC = 54,
  SUFFIX_ZIRCI = 55,
  SUFFIX_S_RIRC = 56,
  SUFFIX_S_RIRCI = 57,
  SUFFIX_U_RIRC = 58,
  SUFFIX_U_RIRCI = 59,
  SUFFIX_R = 60,
  SUFFIX_RCI = 61,
  SUFFIX_Z = 62,
  SUFFIX_ZCI = 63,
  SUFFIX_S_R = 64,
  SUFFIX_S_RCI = 65,
  SUFFIX_U_R = 66,
  SUFFIX_U_RCI = 67,
  SUFFIX_CI = 68,
  SUFFIX_I = 69,
  SUFFIX_DDCI = 70,
  SUFFIX_ERRI = 71,
  SUFFIX_S_ERRI = 72,
  SUFFIX_U_ERRI = 73,
  SUFFIX_EDRI = 74,
  SUFFIX_ERII = 75,
  SUFFIX_ERIR = 76,
  SUFFIX_ERID = 77,
  SUFFIX_DMA_RRI = 78,
};

static constexpr int NUM_TASKLETS = static_cast<int>(kNumTasklets);

struct dpu_state {
  dpu_regfile tasklets[NUM_TASKLETS];
  uint64_t run_bits;
  uint64_t sleep_bits;
  uint64_t replay_bits;

  uint32_t perf_counter_raw;
  uint8_t perf_counter_mode;

  bool is_running(int tid) const { return (run_bits & (1ULL << tid)) != 0; }
  void set_running(int tid) {
    run_bits |= (1ULL << tid);
    sleep_bits &= ~(1ULL << tid);
  }
  void clear_running(int tid) { run_bits &= ~(1ULL << tid); }

  void set_sleeping(int tid) {
    run_bits &= ~(1ULL << tid);
    sleep_bits |= (1ULL << tid);
    replay_bits &= ~(1ULL << tid);
  }
  bool is_sleeping(int tid) const { return (sleep_bits & (1ULL << tid)) != 0; }

  bool is_replaying(int tid) const {
    return (replay_bits & (1ULL << tid)) != 0;
  }
  void set_replaying(int tid) { replay_bits |= (1ULL << tid); }
  void clear_replaying(int tid) { replay_bits &= ~(1ULL << tid); }

  uint32_t alive_count() const {
    uint32_t count = 0;
    for (int i = 0; i < NUM_TASKLETS; i++) {
      if ((run_bits | sleep_bits) & (1ULL << i)) {
        count++;
      }
    }
    return count;
  }

  uint32_t active_count() const {
    uint32_t count = 0;
    for (int i = 0; i < NUM_TASKLETS; i++) {
      if (is_running(i)) {
        count++;
      }
    }
    return count;
  }
};

Pipeline::Pipeline(size_t tasklet_count) : tasklet_count_(tasklet_count) {}

size_t Pipeline::tasklet_count() const { return tasklet_count_; }

const Pipeline &Pipeline::instance() {
  static const Pipeline pipeline{kNumTasklets};
  return pipeline;
}

uint32_t Pipeline::low_u32_impl(uint64_t value) const {
  return static_cast<uint32_t>(value & 0xFFFFFFFFULL);
}

uint8_t Pipeline::byte_impl(uint64_t value, unsigned shift) const {
  return static_cast<uint8_t>((value >> shift) & 0xFFu);
}

uint8_t Pipeline::dreg_slot_impl(int reg_num) const {
  if (reg_num < 0) {
    return 0;
  }
  return static_cast<uint8_t>(reg_num / register_set_.dreg_divisor);
}

uint32_t Pipeline::perf_counter_read_32_impl(const dpu_state &state) const {
  return static_cast<uint32_t>(state.perf_counter_raw >> 4u);
}

uint32_t Pipeline::perf_counter_config_impl(dpu_state &state,
                                            uint32_t config_bits) const {
  const uint32_t old_value = perf_counter_read_32_impl(state);
  const uint8_t packed = static_cast<uint8_t>(config_bits & 0x7u);
  const bool should_reset = (packed & 0x1u) != 0;
  const uint8_t mode_bits = static_cast<uint8_t>((packed >> 1) & 0x3u);

  if (mode_bits == register_set_.perf_mode_same) {
    /* keep current mode */
  } else if (mode_bits == register_set_.perf_mode_cycles) {
    state.perf_counter_mode = register_set_.perf_mode_cycles;
  } else if (mode_bits == register_set_.perf_mode_instructions) {
    state.perf_counter_mode = register_set_.perf_mode_instructions;
  } else if (mode_bits == register_set_.perf_mode_nothing) {
    state.perf_counter_mode = register_set_.perf_mode_nothing;
  }

  if (should_reset) {
    state.perf_counter_raw =
        (state.perf_counter_mode == register_set_.perf_mode_nothing)
            ? 0u
            : static_cast<uint32_t>(~0u);
  }

  return old_value;
}

void Pipeline::perf_counter_retire_step_impl(dpu_state &state,
                                             bool is_replay_step) const {
  if (state.perf_counter_mode == register_set_.perf_mode_cycles) {
    state.perf_counter_raw += 1u;
    return;
  }

  if (state.perf_counter_mode == register_set_.perf_mode_instructions) {
    if (!is_replay_step) {
      state.perf_counter_raw += 1u;
    }
    return;
  }
}

uint32_t Pipeline::low_u32(uint64_t value) {
  return instance().low_u32_impl(value);
}

uint8_t Pipeline::byte(uint64_t value, unsigned shift) {
  return instance().byte_impl(value, shift);
}

uint8_t Pipeline::dreg_slot(int reg_num) {
  return instance().dreg_slot_impl(reg_num);
}

uint32_t Pipeline::perf_counter_read_32(const dpu_state &state) {
  return instance().perf_counter_read_32_impl(state);
}

uint32_t Pipeline::perf_counter_config(dpu_state &state, uint32_t config_bits) {
  return instance().perf_counter_config_impl(state, config_bits);
}

void Pipeline::perf_counter_retire_step(dpu_state &state, bool is_replay_step) {
  instance().perf_counter_retire_step_impl(state, is_replay_step);
}

/* DMAEngine and MRAM implementations are split into dma.cc / mram.cc. */

namespace upmem_perf {

inline uint32_t read_counter_32(const dpu_state &state) {
  return Pipeline::perf_counter_read_32(state);
}

inline uint32_t config_counter(dpu_state &state, uint32_t config_bits) {
  return Pipeline::perf_counter_config(state, config_bits);
}

inline void retire_step(dpu_state &state, bool is_replay_step) {
  Pipeline::perf_counter_retire_step(state, is_replay_step);
}

inline bool is_dma_opcode(uint8_t opcode) {
  return DMAEngine::is_opcode(opcode);
}

inline bool is_dma_signature(std::string_view signature) {
  return DMAEngine::is_signature(signature);
}

} // namespace upmem_perf

enum class ThreadCmdKind : uint8_t {
  Boot = 0,
  Resume = 1,
  ClearRun = 2,
  ReadRun = 3,
};

struct DecodedInst48;
struct DecodedProgram48CacheEntry;
std::shared_ptr<const DecodedProgram48CacheEntry>
get_decoded_program_cache_48(DpuState &dpu);
void invalidate_decoded_program_cache_48(const DpuState *dpu);

bool replay_strict_sig_allowed_48(const std::string &sig);
bool replay_should_fire_from_sources_48(const DecodedInst48 &ins,
                                        const dpu_regfile &rf);
void replay_update_sources_48(const DecodedInst48 &ins, dpu_regfile &rf);

bool execute_launch_program_48(DpuState &dpu);
void execute_launch_program(DpuState &dpu);

/* dpu_index is declared in dpu/runtime.hh */

static inline uint64_t encode_frame_from_x(uint64_t x, uint8_t tag) {
  uint64_t cmd = 0x3300000000000000ULL;
  cmd |= ((x >> (2 * 8)) & 0xFFULL) << 0;
  cmd |= ((x >> (5 * 8)) & 0xFFULL) << 8;
  cmd |= ((x >> (4 * 8)) & 0xFFULL) << 16;
  cmd |= ((x >> (0 * 8)) & 0xFFULL) << 24;
  cmd |= ((x >> (1 * 8)) & 0xFFULL) << 32;
  cmd |= ((x >> (3 * 8)) & 0xFFULL) << 40;
  cmd |= static_cast<uint64_t>(tag) << 48;
  return cmd;
}

static inline uint64_t thread_frame_command(uint64_t base, uint8_t thread_id) {
  uint64_t x = base;
  x |= (static_cast<uint64_t>(24u & 31u) << 34);
  x |= (static_cast<uint64_t>(thread_id & 0xFu) << 20);
  x |= (static_cast<uint64_t>((thread_id >> 4) & 0xFu) << 16);
  return encode_frame_from_x(x, 0x98u);
}

static inline uint64_t ci_wram_read_word_frame_for_addr(uint32_t address) {
  uint64_t x = 0x700344000000ULL;
  const uint64_t a2 = static_cast<uint64_t>(address) << 2;

  x |= (static_cast<uint64_t>(24u & 31u) << 34);
  x |= ((a2 >> 0) & 0xFULL) << 20;
  x |= ((a2 >> 4) & 0xFULL) << 16;
  x |= ((a2 >> 8) & 0x1ULL) << 15;
  x |= ((a2 >> 9) & 0x1ULL) << 14;
  x |= ((a2 >> 10) & 0x1ULL) << 13;
  x |= ((a2 >> 11) & 0x1ULL) << 12;
  x |= ((a2 >> 12) & 0xFFFULL) << 0;

  return encode_frame_from_x(x, 0x99u);
}

static inline uint64_t ci_wram_write_addr_x(uint32_t address) {
  uint64_t x = 0x7c0044000000ULL;
  const uint64_t a2 = static_cast<uint64_t>(address) << 2;

  x |= (static_cast<uint64_t>(24u & 31u) << 34);
  x |= ((a2 >> 0) & 0xFULL) << 20;
  x |= ((a2 >> 4) & 0x7ULL) << 39;
  x |= ((a2 >> 7) & 0x1ULL) << 24;
  x |= ((a2 >> 8) & 0x1ULL) << 15;
  x |= ((a2 >> 9) & 0x1ULL) << 14;
  x |= ((a2 >> 10) & 0x1ULL) << 13;
  x |= ((a2 >> 11) & 0x1ULL) << 12;
  x |= ((a2 >> 12) & 0xFFFULL) << 0;

  return x;
}

static inline uint64_t ci_wram_write_word_structure_for_addr(uint32_t address) {
  const uint64_t x = ci_wram_write_addr_x(address);

  uint64_t cmd = 0x1100000000000000ULL;
  cmd |= (0xFFULL << 0);
  cmd |= (0x03ULL << 8);
  cmd |= ((x >> (0 * 8)) & 0xFFULL) << 16;
  cmd |= ((x >> (1 * 8)) & 0xFFULL) << 24;
  cmd |= ((x >> (3 * 8)) & 0xFFULL) << 32;
  cmd |= (0x99ULL << 40);
  return cmd;
}

static inline bool decode_wram_write_word_structure(uint64_t cmd,
                                                    uint16_t &address) {
  static thread_local uint64_t cached_cmd = UINT64_MAX;
  static thread_local uint16_t cached_addr = 0;

  if (cmd == cached_cmd) {
    address = cached_addr;
    return true;
  }

  for (uint32_t candidate = 0; candidate < kWramWords; ++candidate) {
    if (ci_wram_write_word_structure_for_addr(candidate) == cmd) {
      cached_cmd = cmd;
      cached_addr = static_cast<uint16_t>(candidate);
      address = cached_addr;
      return true;
    }
  }

  return false;
}

static inline uint64_t
ci_wram_write_word_frame_upper_for_addr(uint32_t address) {
  const uint64_t x = ci_wram_write_addr_x(address);
  uint64_t cmd = 0x3300000000000000ULL;
  cmd |= ((x >> (2 * 8)) & 0xFFULL) << 32;
  cmd |= ((x >> (5 * 8)) & 0xFFULL) << 40;
  cmd |= ((x >> (4 * 8)) & 0xFFULL) << 48;
  return cmd;
}

static inline bool decode_wram_write_word_frame_low5(uint64_t cmd,
                                                     uint8_t &low5) {
  static thread_local uint64_t cached_key = UINT64_MAX;
  static thread_local uint8_t cached_low5 = 0;

  const uint64_t key = cmd & 0xFFFFFFFF00000000ULL;
  if (key == cached_key) {
    low5 = cached_low5;
    return true;
  }

  for (uint32_t candidate = 0; candidate < 32u; ++candidate) {
    if (ci_wram_write_word_frame_upper_for_addr(candidate) == key) {
      cached_key = key;
      cached_low5 = static_cast<uint8_t>(candidate);
      low5 = cached_low5;
      return true;
    }
  }

  return false;
}

static inline bool decode_wram_read_word_frame(uint64_t cmd,
                                               uint32_t &address) {
  static thread_local uint64_t cached_cmd = UINT64_MAX;
  static thread_local uint32_t cached_addr = 0;

  if (cmd == cached_cmd) {
    address = cached_addr;
    return true;
  }

  for (uint32_t candidate = 0; candidate < kWramWords; ++candidate) {
    if (ci_wram_read_word_frame_for_addr(candidate) == cmd) {
      cached_cmd = cmd;
      cached_addr = candidate;
      address = cached_addr;
      return true;
    }
  }

  return false;
}

static inline bool decode_iram_write_structure(uint64_t cmd,
                                               uint16_t &addr_hi) {
  const uint8_t op = static_cast<uint8_t>((cmd >> 56) & 0xFFu);
  if (op != 0x11u) {
    return false;
  }

  const uint8_t b0 = static_cast<uint8_t>((cmd >> 0) & 0xFFu);
  const uint8_t b1 = static_cast<uint8_t>((cmd >> 8) & 0xFFu);
  const uint8_t b2 = static_cast<uint8_t>((cmd >> 16) & 0xFFu);
  const uint8_t b3 = static_cast<uint8_t>((cmd >> 24) & 0xFFu);
  const uint8_t b4 = static_cast<uint8_t>((cmd >> 32) & 0xFFu);
  const uint8_t b5 = static_cast<uint8_t>((cmd >> 40) & 0xFFu);

  if (b0 != 0xFFu || b1 != 0x00u || b3 != 0x47u || b4 != 0x02u || b5 != 0x00u) {
    return false;
  }

  addr_hi = b2;
  return true;
}

static inline bool decode_thread_command(const upmem_runtime *rt, uint64_t cmd,
                                         ThreadCmdKind &kind,
                                         uint8_t &thread_id) {
  if (!rt || !rt->thread_cmd_cache_ready) {
    return false;
  }

  for (size_t k = 0; k < rt->thread_cmd_cache.size(); ++k) {
    for (size_t t = 0; t < kNumTasklets; ++t) {
      if (rt->thread_cmd_cache[k][t] == cmd) {
        kind = static_cast<ThreadCmdKind>(k);
        thread_id = static_cast<uint8_t>(t);
        return true;
      }
    }
  }

  return false;
}

static inline uint8_t b(uint64_t x, unsigned shift) {
  return Pipeline::byte(x, shift);
}

static inline bool exec_trace_enabled() {
  static bool initialized = false;
  static bool enabled = false;

  if (!initialized) {
    const char *env = std::getenv("HOSTPIMSIM_UPMEM_TRACE_EXEC");
    enabled = (env && *env && std::strcmp(env, "0") != 0);
    initialized = true;
  }

  return enabled;
}

static inline bool exec_profile_sig_enabled() {
  static bool initialized = false;
  static bool enabled = false;

  if (!initialized) {
    const char *env = std::getenv("HOSTPIMSIM_UPMEM_PROFILE_SIG");
    enabled = (env && *env && std::strcmp(env, "0") != 0);
    initialized = true;
  }

  return enabled;
}

static inline size_t exec_profile_sig_limit() {
  static bool initialized = false;
  static size_t value = 24u;

  if (!initialized) {
    if (const char *env = std::getenv("HOSTPIMSIM_UPMEM_PROFILE_SIG_LIMIT")) {
      char *end = nullptr;
      const unsigned long long parsed = std::strtoull(env, &end, 10);
      if (end != env && *end == '\0' && parsed > 0ull) {
        value = static_cast<size_t>(parsed);
      }
    }
    initialized = true;
  }

  return value;
}

static inline bool exec_launch_timing_enabled() {
  static bool initialized = false;
  static bool enabled = false;

  if (!initialized) {
    const char *env = std::getenv("HOSTPIMSIM_UPMEM_LAUNCH_TIMING");
    enabled = (env && *env && std::strcmp(env, "0") != 0);
    initialized = true;
  }

  return enabled;
}

enum class ReplayModelMode : uint8_t {
  kDisabled = 0,
  // Conservative mode: keep the low-active-thread guard.
  kGuarded = 1,
  // Fsim-like mode: no active-thread guard (or scoped strict; see scope env).
  kStrict = 2,
};

enum class ReplayStrictScope : uint8_t {
  kAll = 0,
  kRdOnly = 1,
};

enum ReplayStrictRdSigMask : uint32_t {
  kReplayStrictRdDivStep = 1u << 0,
  kReplayStrictRdMulStep = 1u << 1,
  kReplayStrictRdSd = 1u << 2,
  kReplayStrictRdAll =
      kReplayStrictRdDivStep | kReplayStrictRdMulStep | kReplayStrictRdSd,
};

static inline ReplayModelMode replay_model_mode() {
  static bool initialized = false;
  static ReplayModelMode mode = ReplayModelMode::kDisabled;

  if (!initialized) {
    const char *env = std::getenv("HOSTPIMSIM_UPMEM_REPLAY_MODEL");
    if (env && *env && std::strcmp(env, "0") != 0) {
      if (std::strcmp(env, "2") == 0 || std::strcmp(env, "strict") == 0) {
        mode = ReplayModelMode::kStrict;
      } else {
        mode = ReplayModelMode::kGuarded;
      }
    }
    initialized = true;
  }

  return mode;
}

static inline ReplayStrictScope replay_strict_scope() {
  static bool initialized = false;
  static ReplayStrictScope scope = ReplayStrictScope::kAll;

  if (!initialized) {
    const char *env = std::getenv("HOSTPIMSIM_UPMEM_REPLAY_STRICT_SCOPE");
    if (env && *env) {
      if (std::strcmp(env, "rd") == 0 || std::strcmp(env, "rd-only") == 0 ||
          std::strcmp(env, "rd_only") == 0) {
        scope = ReplayStrictScope::kRdOnly;
      }
    }
    initialized = true;
  }

  return scope;
}

static inline uint32_t replay_strict_rd_sig_mask() {
  static bool initialized = false;
  static uint32_t mask = kReplayStrictRdAll;

  if (!initialized) {
    const char *env = std::getenv("HOSTPIMSIM_UPMEM_REPLAY_STRICT_RD_SIGS");
    if (env && *env) {
      uint32_t parsed = 0u;
      std::stringstream ss(env);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (tok.empty()) {
          continue;
        }
        // trim spaces
        const auto begin = tok.find_first_not_of(" \t\n\r");
        if (begin == std::string::npos) {
          continue;
        }
        const auto end = tok.find_last_not_of(" \t\n\r");
        const std::string key = tok.substr(begin, end - begin + 1);

        if (key == "all") {
          parsed = kReplayStrictRdAll;
          break;
        }
        if (key == "div" || key == "div_step" || key == "div_step:rrri") {
          parsed |= kReplayStrictRdDivStep;
          continue;
        }
        if (key == "mul" || key == "mul_step" || key == "mul_step:rrrici") {
          parsed |= kReplayStrictRdMulStep;
          continue;
        }
        if (key == "sd" || key == "sd:rir") {
          parsed |= kReplayStrictRdSd;
          continue;
        }
      }
      mask = parsed;
    }
    initialized = true;
  }

  return mask;
}

static inline void exec_tracef(const char *fmt, ...) {
  if (!exec_trace_enabled() || !fmt) {
    return;
  }

  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "[hostpimsim-upmem-exec] ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
}

bool wram_word_in_bounds(uint32_t word_addr);
void write_wram_word(DpuState &dpu, uint32_t word_addr, uint32_t value);
uint32_t read_wram_word(const DpuState &dpu, uint32_t word_addr);

bool iram_slot_in_bounds(uint16_t iram_slot);
void write_iram_word(DpuState &dpu, uint16_t iram_slot, uint64_t value48);

static inline void clear_dpu_state(DpuState &dpu) {
  std::fill(dpu.private_mem.begin(), dpu.private_mem.end(), 0u);
  invalidate_decoded_program_cache_48(&dpu);

  /* Initialize stdout runtime metadata to a sane default. */
  const uint32_t stdout_buffer_size = 0x00100000u; // 1MB (__stdout_buffer)
  const size_t stdout_size_off = WRAM_OFFSET + 0x11C8u;
  if (stdout_size_off + sizeof(stdout_buffer_size) <= dpu.private_mem.size()) {
    std::memcpy(dpu.private_mem.data() + stdout_size_off, &stdout_buffer_size,
                sizeof(stdout_buffer_size));
  }

  dpu.running_tasklets = 0u;
  dpu.launch_pending = false;
}

static inline uint8_t first_selected_dpu(uint8_t mask) {
  for (uint8_t dpu = 0; dpu < kNumDpusPerCi; ++dpu) {
    if (mask & (1u << dpu)) {
      return dpu;
    }
  }
  return 0;
}

enum FastOp48 : uint16_t {
  FAST_OP_NONE = 0,
  FAST_OP_MOVE_RI_RR,
  FAST_OP_MOVE_RICI_RRCI,
  FAST_OP_ADD_RRI,
  FAST_OP_ADD_RRR,
  FAST_OP_SUB_RRR,
  FAST_OP_SUB_RRRC,
  FAST_OP_NEG_RR,
  FAST_OP_AND_RRI,
  FAST_OP_AND_RRR,
  FAST_OP_ADDC_RRR,
  FAST_OP_MOVE_U_RR,
  FAST_OP_MOVE_S_RI_RR,
  FAST_OP_CLZ_RR,
  FAST_OP_CLZ_RRCI,
  FAST_OP_XOR_RRR,
  FAST_OP_OR_RRR,
  FAST_OP_OR_RRIF,
  FAST_OP_JEQ_RRI,
  FAST_OP_JEQ_RII,
  FAST_OP_JNEQ_RRI,
  FAST_OP_JNEQ_RII,
  FAST_OP_JZ_RI,
  FAST_OP_JGTS_RRI,
  FAST_OP_JLTU_RRI,
  FAST_OP_JGTU_RRI,
  FAST_OP_JGEU_RRI,
  FAST_OP_JLEU_RRI,
  FAST_OP_MUL_STEP_RRRICI,
  FAST_OP_CALL_ANY,
  FAST_OP_JUMP_ANY,
  FAST_OP_LW_RRI,
  FAST_OP_LBU_RRI,
  FAST_OP_LBS_RRI,
  FAST_OP_SW_RII,
  FAST_OP_SW_RIR,
  FAST_OP_SB_RIR,
  FAST_OP_LD_RRI,
  FAST_OP_SD_RIR,
  FAST_OP_LDMA_RRI,
  FAST_OP_SDMA_RRI,
  FAST_OP_LSR_ADD_RRRI,
  FAST_OP_LSR_RRI,
  FAST_OP_LSR_RRICI,
  FAST_OP_LSL_RRI,
  FAST_OP_LSL_RRR,
  FAST_OP_DIV_STEP_RRRI,
  FAST_OP_MUL_UL_UL_RRR,
  FAST_OP_MUL_UL_UH_RRR,
  FAST_OP_MUL_UH_UL_RRR,
  FAST_OP_MUL_UH_UH_RRR,
  FAST_OP_LSL_ADD_RRRI,
  FAST_OP_LSL_ADD_RRRICI,
  FAST_OP_LSL_SUB_RRRI,
  FAST_OP_LSL_SUB_RRRICI,
  FAST_OP_SUB_ZRR,
  FAST_OP_SUB_ZRICI,
  FAST_OP_SUB_ZRRCI,
  FAST_OP_SUBC_ZRRCI,
  FAST_OP_ACQUIRE_RICI,
  FAST_OP_RELEASE_RICI,
};

struct DecodedInst48 {
  bool valid = false;
  bool perf_is_dma = false;
  uint16_t fast_op = FAST_OP_NONE;
  std::string signature;
  std::string cc_name;

  bool has_ra = false;
  bool has_rb = false;
  bool has_rc = false;
  bool has_dc = false;
  bool has_db = false;

  int ra = 0;
  int rb = 0;
  int rc = 0;
  int dc = 0;
  int db = 0;

  uint32_t replay_read_mask = 0;
  uint32_t replay_write_mask = 0;

  int imm = 0;
  int off = 0;
  int pc = 0;
  int shift = 0;
  int immDma = 0;
  int endian = 0;
};

struct DecodedProgram48CacheEntry {
  std::vector<DecodedInst48> decoded;
};

static inline bool fast_op_needs_rb(uint16_t fast_op) {
  switch (fast_op) {
  case FAST_OP_ADD_RRR:
  case FAST_OP_SUB_RRR:
  case FAST_OP_SUB_RRRC:
  case FAST_OP_AND_RRR:
  case FAST_OP_ADDC_RRR:
  case FAST_OP_OR_RRR:
  case FAST_OP_XOR_RRR:
  case FAST_OP_JEQ_RRI:
  case FAST_OP_JNEQ_RRI:
  case FAST_OP_JGTS_RRI:
  case FAST_OP_JLTU_RRI:
  case FAST_OP_JGTU_RRI:
  case FAST_OP_JGEU_RRI:
  case FAST_OP_JLEU_RRI:
  case FAST_OP_CALL_ANY:
  case FAST_OP_SW_RIR:
  case FAST_OP_SB_RIR:
  case FAST_OP_LDMA_RRI:
  case FAST_OP_SDMA_RRI:
  case FAST_OP_LSR_ADD_RRRI:
  case FAST_OP_LSL_RRR:
  case FAST_OP_LSL_ADD_RRRI:
  case FAST_OP_LSL_ADD_RRRICI:
  case FAST_OP_LSL_SUB_RRRI:
  case FAST_OP_LSL_SUB_RRRICI:
  case FAST_OP_SUB_ZRR:
  case FAST_OP_SUB_ZRRCI:
  case FAST_OP_SUBC_ZRRCI:
  case FAST_OP_MUL_UL_UL_RRR:
  case FAST_OP_MUL_UL_UH_RRR:
  case FAST_OP_MUL_UH_UL_RRR:
  case FAST_OP_MUL_UH_UH_RRR:
    return true;
  default:
    return false;
  }
}

static inline uint16_t fast_op_from_signature(const std::string &sig) {
  if (sig == "move:ri" || sig == "move:rr" || sig == "or:rri") {
    return FAST_OP_MOVE_RI_RR;
  }
  if (sig == "move:rici" || sig == "move:rrci") {
    return FAST_OP_MOVE_RICI_RRCI;
  }
  if (sig == "move.u:rr") {
    return FAST_OP_MOVE_U_RR;
  }
  if (sig == "add:rri") {
    return FAST_OP_ADD_RRI;
  }
  if (sig == "add:rrr") {
    return FAST_OP_ADD_RRR;
  }
  if (sig == "sub:rrr") {
    return FAST_OP_SUB_RRR;
  }
  if (sig == "sub:rrrc") {
    return FAST_OP_SUB_RRRC;
  }
  if (sig == "neg:rr") {
    return FAST_OP_NEG_RR;
  }
  if (sig == "and:rri") {
    return FAST_OP_AND_RRI;
  }
  if (sig == "and:rrr") {
    return FAST_OP_AND_RRR;
  }
  if (sig == "addc:rrr") {
    return FAST_OP_ADDC_RRR;
  }
  if (sig == "move.s:ri" || sig == "move.s:rr") {
    return FAST_OP_MOVE_S_RI_RR;
  }
  if (sig == "clz:rr") {
    return FAST_OP_CLZ_RR;
  }
  if (sig == "clz:rrci") {
    return FAST_OP_CLZ_RRCI;
  }
  if (sig == "xor:rrr") {
    return FAST_OP_XOR_RRR;
  }
  if (sig == "or:rrr") {
    return FAST_OP_OR_RRR;
  }
  if (sig == "or:rrif") {
    return FAST_OP_OR_RRIF;
  }
  if (sig == "jeq:rri") {
    return FAST_OP_JEQ_RRI;
  }
  if (sig == "jeq:rii") {
    return FAST_OP_JEQ_RII;
  }
  if (sig == "jneq:rri") {
    return FAST_OP_JNEQ_RRI;
  }
  if (sig == "jneq:rii") {
    return FAST_OP_JNEQ_RII;
  }
  if (sig == "jz:ri") {
    return FAST_OP_JZ_RI;
  }
  if (sig == "jgts:rri") {
    return FAST_OP_JGTS_RRI;
  }
  if (sig == "jltu:rri") {
    return FAST_OP_JLTU_RRI;
  }
  if (sig == "jgtu:rri") {
    return FAST_OP_JGTU_RRI;
  }
  if (sig == "jgeu:rri") {
    return FAST_OP_JGEU_RRI;
  }
  if (sig == "jleu:rri") {
    return FAST_OP_JLEU_RRI;
  }
  if (sig == "mul_step:rrrici") {
    return FAST_OP_MUL_STEP_RRRICI;
  }
  if (sig == "call:ri" || sig == "call:rri" || sig == "call:rr" ||
      sig == "call:rrr" || sig == "call:zri" || sig == "call:zrr") {
    return FAST_OP_CALL_ANY;
  }
  if (sig == "jump:i" || sig == "jump:r" || sig == "jump:ri") {
    return FAST_OP_JUMP_ANY;
  }
  if (sig == "lw:rri" || sig == "lw:erri") {
    return FAST_OP_LW_RRI;
  }
  if (sig == "lbu:rri" || sig == "lbu:erri") {
    return FAST_OP_LBU_RRI;
  }
  if (sig == "lbs:rri" || sig == "lbs:erri") {
    return FAST_OP_LBS_RRI;
  }
  if (sig == "sw:rii") {
    return FAST_OP_SW_RII;
  }
  if (sig == "sw:rir" || sig == "sw:erir") {
    return FAST_OP_SW_RIR;
  }
  if (sig == "sb:rir" || sig == "sb:erir") {
    return FAST_OP_SB_RIR;
  }
  if (sig == "ld:rri" || sig == "ld:erri") {
    return FAST_OP_LD_RRI;
  }
  if (sig == "sd:rir" || sig == "sd:erir") {
    return FAST_OP_SD_RIR;
  }
  if (sig == "ldma:rri") {
    return FAST_OP_LDMA_RRI;
  }
  if (sig == "sdma:rri") {
    return FAST_OP_SDMA_RRI;
  }
  if (sig == "lsr_add:rrri") {
    return FAST_OP_LSR_ADD_RRRI;
  }
  if (sig == "lsr:rri") {
    return FAST_OP_LSR_RRI;
  }
  if (sig == "lsr:rrici") {
    return FAST_OP_LSR_RRICI;
  }
  if (sig == "lsl:rri") {
    return FAST_OP_LSL_RRI;
  }
  if (sig == "lsl:rrr") {
    return FAST_OP_LSL_RRR;
  }
  if (sig == "div_step:rrri") {
    return FAST_OP_DIV_STEP_RRRI;
  }
  if (sig == "mul_ul_ul:rrr") {
    return FAST_OP_MUL_UL_UL_RRR;
  }
  if (sig == "mul_ul_uh:rrr") {
    return FAST_OP_MUL_UL_UH_RRR;
  }
  if (sig == "mul_uh_ul:rrr") {
    return FAST_OP_MUL_UH_UL_RRR;
  }
  if (sig == "mul_uh_uh:rrr") {
    return FAST_OP_MUL_UH_UH_RRR;
  }
  if (sig == "lsl_add:rrri") {
    return FAST_OP_LSL_ADD_RRRI;
  }
  if (sig == "lsl_add:rrrici") {
    return FAST_OP_LSL_ADD_RRRICI;
  }
  if (sig == "lsl_sub:rrri") {
    return FAST_OP_LSL_SUB_RRRI;
  }
  if (sig == "lsl_sub:rrrici") {
    return FAST_OP_LSL_SUB_RRRICI;
  }
  if (sig == "sub:zrr") {
    return FAST_OP_SUB_ZRR;
  }
  if (sig == "sub:zrici") {
    return FAST_OP_SUB_ZRICI;
  }
  if (sig == "sub:zrrci") {
    return FAST_OP_SUB_ZRRCI;
  }
  if (sig == "subc:zrrci") {
    return FAST_OP_SUBC_ZRRCI;
  }
  if (sig == "acquire:rici") {
    return FAST_OP_ACQUIRE_RICI;
  }
  if (sig == "release:rici") {
    return FAST_OP_RELEASE_RICI;
  }
  return FAST_OP_NONE;
}

bool wram_rel_in_bounds(uint32_t addr, size_t size);
bool wram_load(DpuState &dpu, uint32_t addr, void *dst, size_t size);
bool wram_store(DpuState &dpu, uint32_t addr, const void *src, size_t size);
uint8_t wram_load_u8(DpuState &dpu, uint32_t addr);
uint32_t wram_load_u32(DpuState &dpu, uint32_t addr, bool big_endian);
uint64_t wram_load_u64(DpuState &dpu, uint32_t addr, bool big_endian);
void wram_store_u8(DpuState &dpu, uint32_t addr, uint8_t value);
void wram_store_u16(DpuState &dpu, uint32_t addr, uint16_t value,
                    bool big_endian);
void wram_store_u32(DpuState &dpu, uint32_t addr, uint32_t value,
                    bool big_endian);
void wram_store_u64(DpuState &dpu, uint32_t addr, uint64_t value,
                    bool big_endian);

static inline uint8_t dreg_slot(int reg_num) {
  return Pipeline::dreg_slot(reg_num);
}

static inline bool eval_cc_name(const std::string &cc_name,
                                const dpu_regfile &rf, uint32_t value_for_z) {
  if (cc_name == "true") {
    return true;
  }
  if (cc_name == "false" || cc_name.empty()) {
    return false;
  }

  if (cc_name == "z" || cc_name == "eq") {
    return value_for_z == 0;
  }
  if (cc_name == "nz" || cc_name == "neq") {
    return value_for_z != 0;
  }

  if (cc_name == "c" || cc_name == "ltu") {
    return rf.CF;
  }
  if (cc_name == "nc" || cc_name == "geu") {
    return !rf.CF;
  }
  if (cc_name == "leu") {
    return rf.CF || (value_for_z == 0);
  }
  if (cc_name == "gtu") {
    return (!rf.CF) && (value_for_z != 0);
  }

  if (cc_name == "pl" || cc_name == "spl") {
    return static_cast<int32_t>(value_for_z) >= 0;
  }
  if (cc_name == "mi" || cc_name == "smi") {
    return static_cast<int32_t>(value_for_z) < 0;
  }
  if (cc_name == "lts") {
    return static_cast<int32_t>(value_for_z) < 0;
  }
  if (cc_name == "les") {
    return static_cast<int32_t>(value_for_z) <= 0;
  }
  if (cc_name == "gts") {
    return static_cast<int32_t>(value_for_z) > 0;
  }
  if (cc_name == "ges") {
    return static_cast<int32_t>(value_for_z) >= 0;
  }

  if (cc_name == "sz") {
    return value_for_z == 0;
  }
  if (cc_name == "snz") {
    return value_for_z != 0;
  }

  if (cc_name == "e" || cc_name == "se") {
    return (value_for_z & 1u) == 0;
  }
  if (cc_name == "o" || cc_name == "so") {
    return (value_for_z & 1u) != 0;
  }

  if (cc_name == "max") {
    return value_for_z == 32u;
  }
  if (cc_name == "nmax") {
    return value_for_z != 32u;
  }

  if (cc_name == "xz") {
    return rf.ZF && (value_for_z == 0);
  }
  if (cc_name == "xnz") {
    return !(rf.ZF && (value_for_z == 0));
  }

  return false;
}

static inline bool eval_subc_cc(const std::string &cc_name,
                                const dpu_regfile &rf, uint32_t value,
                                bool old_zf, bool overflow) {
  if (cc_name == "xz") {
    return (value == 0) && old_zf;
  }
  if (cc_name == "xnz") {
    return !((value == 0) && old_zf);
  }
  if (cc_name == "xleu") {
    return rf.CF || old_zf;
  }
  if (cc_name == "xgtu") {
    return rf.CF && !old_zf;
  }
  const bool signed_less = (static_cast<int32_t>(value) < 0) || overflow;
  const bool equal64 = old_zf && (value == 0);
  if (cc_name == "xles") {
    return signed_less || equal64;
  }
  if (cc_name == "xgts") {
    return !signed_less && !equal64;
  }
  return eval_cc_name(cc_name, rf, value);
}

static inline bool eval_add_cc_name(const std::string &cc_name,
                                    const dpu_regfile &rf, uint32_t lhs,
                                    uint32_t rhs, uint32_t result) {
  if (cc_name == "ov") {
    const bool s1 = ((lhs >> 31) & 1u) != 0;
    const bool s2 = ((rhs >> 31) & 1u) != 0;
    const bool sr = ((result >> 31) & 1u) != 0;
    return (s1 == s2) && (s1 != sr);
  }
  if (cc_name == "nov") {
    const bool s1 = ((lhs >> 31) & 1u) != 0;
    const bool s2 = ((rhs >> 31) & 1u) != 0;
    const bool sr = ((result >> 31) & 1u) != 0;
    return !((s1 == s2) && (s1 != sr));
  }

  if (cc_name.size() >= 3 && cc_name[0] == 'n' && cc_name[1] == 'c') {
    char *end = nullptr;
    const long bit = std::strtol(cc_name.c_str() + 2, &end, 10);
    if (end && *end == '\0' && bit > 0 && bit < 32) {
      const uint64_t width = static_cast<uint64_t>(bit);
      const uint64_t mask = (1ULL << width) - 1ULL;
      const uint64_t sum =
          static_cast<uint64_t>(lhs & mask) + static_cast<uint64_t>(rhs & mask);
      const bool carry = (sum >> width) != 0ULL;
      return !carry;
    }
  }

  return eval_cc_name(cc_name, rf, result);
}

/* ---- merged from decode48_generated.hh ---- */
// Auto-generated from UPMEM disasm.py / isa.py by OpenClaw.
#include <cstdint>
#include <string_view>

struct RawDecoded48 {
  bool valid = false;
  const char *signature = nullptr;
  bool has_ra = false;
  int32_t ra = 0;
  bool has_rb = false;
  int32_t rb = 0;
  bool has_rc = false;
  int32_t rc = 0;
  bool has_dc = false;
  int32_t dc = 0;
  bool has_db = false;
  int32_t db = 0;
  bool has_imm = false;
  int32_t imm = 0;
  bool has_off = false;
  int32_t off = 0;
  bool has_pc = false;
  int32_t pc = 0;
  bool has_shift = false;
  int32_t shift = 0;
  bool has_immDma = false;
  int32_t immDma = 0;
  bool has_endian = false;
  int32_t endian = 0;
  bool has_cc = false;
  int32_t cc = 0;
};

static inline const char *decode_cc_map_0(int cc) {
  switch (cc) {
  case 1:
    return "true";
  case 2:
    return "z";
  case 3:
    return "nz";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_1(int cc) {
  switch (cc) {
  case 6:
    return "z";
  case 7:
    return "nz";
  case 10:
    return "xz";
  case 11:
    return "xnz";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_2(int cc) {
  switch (cc) {
  case 1:
    return "true";
  case 2:
    return "z";
  case 3:
    return "nz";
  case 4:
    return "xz";
  case 5:
    return "xnz";
  case 8:
    return "pl";
  case 9:
    return "mi";
  case 12:
    return "sz";
  case 13:
    return "snz";
  case 14:
    return "spl";
  case 15:
    return "smi";
  case 18:
    return "ov";
  case 19:
    return "nov";
  case 20:
    return "c";
  case 21:
    return "nc";
  case 22:
    return "nc5";
  case 23:
    return "nc6";
  case 24:
    return "nc7";
  case 25:
    return "nc8";
  case 26:
    return "nc9";
  case 27:
    return "nc10";
  case 28:
    return "nc11";
  case 29:
    return "nc12";
  case 30:
    return "nc13";
  case 31:
    return "nc14";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_3(int cc) {
  switch (cc) {
  case 0:
    return "false";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_4(int cc) {
  switch (cc) {
  case 1:
    return "true";
  case 2:
    return "z";
  case 3:
    return "nz";
  case 4:
    return "xz";
  case 5:
    return "xnz";
  case 8:
    return "pl";
  case 9:
    return "mi";
  case 12:
    return "sz";
  case 13:
    return "snz";
  case 14:
    return "spl";
  case 15:
    return "smi";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_5(int cc) {
  switch (cc) {
  case 1:
    return "true";
  case 2:
    return "z";
  case 3:
    return "nz";
  case 4:
    return "xz";
  case 5:
    return "xnz";
  case 8:
    return "pl";
  case 9:
    return "mi";
  case 12:
    return "sz";
  case 13:
    return "snz";
  case 14:
    return "spl";
  case 15:
    return "smi";
  case 24:
    return "e";
  case 25:
    return "o";
  case 30:
    return "se";
  case 31:
    return "so";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_6(int cc) {
  switch (cc) {
  case 1:
    return "true";
  case 2:
    return "z";
  case 3:
    return "nz";
  case 4:
    return "xz";
  case 5:
    return "xnz";
  case 8:
    return "pl";
  case 9:
    return "mi";
  case 12:
    return "sz";
  case 13:
    return "snz";
  case 14:
    return "spl";
  case 15:
    return "smi";
  case 24:
    return "e";
  case 25:
    return "o";
  case 28:
    return "nsh32";
  case 29:
    return "sh32";
  case 30:
    return "se";
  case 31:
    return "so";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_7(int cc) {
  switch (cc) {
  case 0:
    return "false";
  case 1:
    return "true";
  case 2:
    return "z";
  case 3:
    return "nz";
  case 4:
    return "xz";
  case 5:
    return "xnz";
  case 12:
    return "sz";
  case 13:
    return "snz";
  case 14:
    return "spl";
  case 15:
    return "smi";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_8(int cc) {
  switch (cc) {
  case 1:
    return "true";
  case 2:
    return "z";
  case 3:
    return "nz";
  case 4:
    return "xz";
  case 5:
    return "xnz";
  case 8:
    return "max";
  case 9:
    return "nmax";
  case 12:
    return "sz";
  case 13:
    return "snz";
  case 14:
    return "spl";
  case 15:
    return "smi";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_9(int cc) {
  switch (cc) {
  case 0:
    return "false";
  case 1:
    return "true";
  case 12:
    return "sz";
  case 13:
    return "snz";
  case 14:
    return "spl";
  case 15:
    return "smi";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_10(int cc) {
  switch (cc) {
  case 1:
    return "true";
  case 12:
    return "sz";
  case 13:
    return "snz";
  case 14:
    return "spl";
  case 15:
    return "smi";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_11(int cc) {
  switch (cc) {
  case 0:
    return "false";
  case 1:
    return "true";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_12(int cc) {
  switch (cc) {
  case 1:
    return "true";
  case 2:
    return "z";
  case 3:
    return "nz";
  case 4:
    return "xz";
  case 5:
    return "xnz";
  case 12:
    return "sz";
  case 13:
    return "snz";
  case 14:
    return "spl";
  case 15:
    return "smi";
  case 30:
    return "small";
  case 31:
    return "large";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_13(int cc) {
  switch (cc) {
  case 1:
    return "true";
  case 2:
    return "z";
  case 3:
    return "nz";
  case 4:
    return "xz";
  case 5:
    return "xnz";
  case 8:
    return "pl";
  case 9:
    return "mi";
  case 12:
    return "sz";
  case 13:
    return "snz";
  case 14:
    return "spl";
  case 15:
    return "smi";
  case 18:
    return "ov";
  case 19:
    return "nov";
  case 20:
    return "ltu";
  case 21:
    return "geu";
  case 22:
    return "lts";
  case 23:
    return "ges";
  case 24:
    return "les";
  case 25:
    return "gts";
  case 26:
    return "leu";
  case 27:
    return "gtu";
  case 28:
    return "xles";
  case 29:
    return "xgts";
  case 30:
    return "xleu";
  case 31:
    return "xgtu";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_14(int cc) {
  switch (cc) {
  case 0:
    return "nz";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_15(int cc) {
  switch (cc) {
  case 6:
    return "z";
  case 7:
    return "nz";
  case 10:
    return "xz";
  case 11:
    return "xnz";
  case 33:
    return "true";
  case 34:
    return "z";
  case 35:
    return "nz";
  case 36:
    return "xz";
  case 37:
    return "xnz";
  case 40:
    return "pl";
  case 41:
    return "mi";
  case 44:
    return "sz";
  case 45:
    return "snz";
  case 46:
    return "spl";
  case 47:
    return "smi";
  case 50:
    return "ov";
  case 51:
    return "nov";
  case 52:
    return "ltu";
  case 53:
    return "geu";
  case 54:
    return "lts";
  case 55:
    return "ges";
  case 56:
    return "les";
  case 57:
    return "gts";
  case 58:
    return "leu";
  case 59:
    return "gtu";
  case 60:
    return "xles";
  case 61:
    return "xgts";
  case 62:
    return "xleu";
  case 63:
    return "xgtu";
  default:
    return "";
  }
}

static inline const char *decode_cc_map_16(int cc) {
  switch (cc) {
  case 1:
    return "true";
  default:
    return "";
  }
}

static inline const char *
decode_cc_name_for_signature_48(std::string_view signature, int cc) {
  if (cc < 0 || cc > 63)
    return "";
  if (signature == "acquire:rici")
    return decode_cc_map_0(cc);
  if (signature == "add.s:rric" || signature == "add.s:rrrc" ||
      signature == "add.u:rric" || signature == "add.u:rrrc" ||
      signature == "add:rric" || signature == "add:rrrc" ||
      signature == "add:zric" || signature == "add:zrrc" ||
      signature == "addc.s:rric" || signature == "addc.s:rrrc" ||
      signature == "addc.u:rric" || signature == "addc.u:rrrc" ||
      signature == "addc:rric" || signature == "addc:rrrc" ||
      signature == "addc:zric" || signature == "addc:zrrc" ||
      signature == "and.s:rric" || signature == "and.s:rrrc" ||
      signature == "and.u:rric" || signature == "and.u:rrrc" ||
      signature == "and:rric" || signature == "and:rrrc" ||
      signature == "and:zric" || signature == "and:zrrc" ||
      signature == "andn.s:rric" || signature == "andn.s:rrrc" ||
      signature == "andn.u:rric" || signature == "andn.u:rrrc" ||
      signature == "andn:rric" || signature == "andn:rrrc" ||
      signature == "andn:zric" || signature == "andn:zrrc" ||
      signature == "asr.s:rric" || signature == "asr.s:rrrc" ||
      signature == "asr.u:rric" || signature == "asr.u:rrrc" ||
      signature == "asr:rric" || signature == "asr:rrrc" ||
      signature == "asr:zric" || signature == "asr:zrrc" ||
      signature == "cao.s:rrc" || signature == "cao.u:rrc" ||
      signature == "cao:rrc" || signature == "cao:zrc" ||
      signature == "clo.s:rrc" || signature == "clo.u:rrc" ||
      signature == "clo:rrc" || signature == "clo:zrc" ||
      signature == "cls.s:rrc" || signature == "cls.u:rrc" ||
      signature == "cls:rrc" || signature == "cls:zrc" ||
      signature == "clz.s:rrc" || signature == "clz.u:rrc" ||
      signature == "clz:rrc" || signature == "clz:zrc" ||
      signature == "cmpb4.s:rrrc" || signature == "cmpb4.u:rrrc" ||
      signature == "cmpb4:rrrc" || signature == "cmpb4:zrrc" ||
      signature == "extsb.s:rrc" || signature == "extsb:rrc" ||
      signature == "extsb:zrc" || signature == "extsh.s:rrc" ||
      signature == "extsh:rrc" || signature == "extsh:zrc" ||
      signature == "extub.u:rrc" || signature == "extub:rrc" ||
      signature == "extub:zrc" || signature == "extuh.u:rrc" ||
      signature == "extuh:rrc" || signature == "extuh:zrc" ||
      signature == "hash.s:rric" || signature == "hash.s:rrrc" ||
      signature == "hash.u:rric" || signature == "hash.u:rrrc" ||
      signature == "hash:rric" || signature == "hash:rrrc" ||
      signature == "hash:zric" || signature == "hash:zrrc" ||
      signature == "lsl.s:rric" || signature == "lsl.s:rrrc" ||
      signature == "lsl.u:rric" || signature == "lsl.u:rrrc" ||
      signature == "lsl1.s:rric" || signature == "lsl1.s:rrrc" ||
      signature == "lsl1.u:rric" || signature == "lsl1.u:rrrc" ||
      signature == "lsl1:rric" || signature == "lsl1:rrrc" ||
      signature == "lsl1:zric" || signature == "lsl1:zrrc" ||
      signature == "lsl1x.s:rric" || signature == "lsl1x.s:rrrc" ||
      signature == "lsl1x.u:rric" || signature == "lsl1x.u:rrrc" ||
      signature == "lsl1x:rric" || signature == "lsl1x:rrrc" ||
      signature == "lsl1x:zric" || signature == "lsl1x:zrrc" ||
      signature == "lsl:rric" || signature == "lsl:rrrc" ||
      signature == "lsl:zric" || signature == "lsl:zrrc" ||
      signature == "lslx.s:rric" || signature == "lslx.s:rrrc" ||
      signature == "lslx.u:rric" || signature == "lslx.u:rrrc" ||
      signature == "lslx:rric" || signature == "lslx:rrrc" ||
      signature == "lslx:zric" || signature == "lslx:zrrc" ||
      signature == "lsr.s:rric" || signature == "lsr.s:rrrc" ||
      signature == "lsr.u:rric" || signature == "lsr.u:rrrc" ||
      signature == "lsr1.s:rric" || signature == "lsr1.s:rrrc" ||
      signature == "lsr1.u:rric" || signature == "lsr1.u:rrrc" ||
      signature == "lsr1:rric" || signature == "lsr1:rrrc" ||
      signature == "lsr1:zric" || signature == "lsr1:zrrc" ||
      signature == "lsr1x.s:rric" || signature == "lsr1x.s:rrrc" ||
      signature == "lsr1x.u:rric" || signature == "lsr1x.u:rrrc" ||
      signature == "lsr1x:rric" || signature == "lsr1x:rrrc" ||
      signature == "lsr1x:zric" || signature == "lsr1x:zrrc" ||
      signature == "lsr:rric" || signature == "lsr:rrrc" ||
      signature == "lsr:zric" || signature == "lsr:zrrc" ||
      signature == "lsrx.s:rric" || signature == "lsrx.s:rrrc" ||
      signature == "lsrx.u:rric" || signature == "lsrx.u:rrrc" ||
      signature == "lsrx:rric" || signature == "lsrx:rrrc" ||
      signature == "lsrx:zric" || signature == "lsrx:zrrc" ||
      signature == "mul_sh_sh.s:rrrc" || signature == "mul_sh_sh:rrrc" ||
      signature == "mul_sh_sh:zrrc" || signature == "mul_sh_sl.s:rrrc" ||
      signature == "mul_sh_sl:rrrc" || signature == "mul_sh_sl:zrrc" ||
      signature == "mul_sh_uh.s:rrrc" || signature == "mul_sh_uh:rrrc" ||
      signature == "mul_sh_uh:zrrc" || signature == "mul_sh_ul.s:rrrc" ||
      signature == "mul_sh_ul:rrrc" || signature == "mul_sh_ul:zrrc" ||
      signature == "mul_sl_sh.s:rrrc" || signature == "mul_sl_sh:rrrc" ||
      signature == "mul_sl_sh:zrrc" || signature == "mul_sl_sl.s:rrrc" ||
      signature == "mul_sl_sl:rrrc" || signature == "mul_sl_sl:zrrc" ||
      signature == "mul_sl_uh.s:rrrc" || signature == "mul_sl_uh:rrrc" ||
      signature == "mul_sl_uh:zrrc" || signature == "mul_sl_ul.s:rrrc" ||
      signature == "mul_sl_ul:rrrc" || signature == "mul_sl_ul:zrrc" ||
      signature == "mul_uh_uh.u:rrrc" || signature == "mul_uh_uh:rrrc" ||
      signature == "mul_uh_uh:zrrc" || signature == "mul_uh_ul.u:rrrc" ||
      signature == "mul_uh_ul:rrrc" || signature == "mul_uh_ul:zrrc" ||
      signature == "mul_ul_uh.u:rrrc" || signature == "mul_ul_uh:rrrc" ||
      signature == "mul_ul_uh:zrrc" || signature == "mul_ul_ul.u:rrrc" ||
      signature == "mul_ul_ul:rrrc" || signature == "mul_ul_ul:zrrc" ||
      signature == "nand.s:rric" || signature == "nand.s:rrrc" ||
      signature == "nand.u:rric" || signature == "nand.u:rrrc" ||
      signature == "nand:rric" || signature == "nand:rrrc" ||
      signature == "nand:zric" || signature == "nand:zrrc" ||
      signature == "nor.s:rric" || signature == "nor.s:rrrc" ||
      signature == "nor.u:rric" || signature == "nor.u:rrrc" ||
      signature == "nor:rric" || signature == "nor:rrrc" ||
      signature == "nor:zric" || signature == "nor:zrrc" ||
      signature == "nxor.s:rric" || signature == "nxor.s:rrrc" ||
      signature == "nxor.u:rric" || signature == "nxor.u:rrrc" ||
      signature == "nxor:rric" || signature == "nxor:rrrc" ||
      signature == "nxor:zric" || signature == "nxor:zrrc" ||
      signature == "or.s:rric" || signature == "or.s:rrrc" ||
      signature == "or.u:rric" || signature == "or.u:rrrc" ||
      signature == "or:rric" || signature == "or:rrrc" ||
      signature == "or:zric" || signature == "or:zrrc" ||
      signature == "orn.s:rric" || signature == "orn.s:rrrc" ||
      signature == "orn.u:rric" || signature == "orn.u:rrrc" ||
      signature == "orn:rric" || signature == "orn:rrrc" ||
      signature == "orn:zric" || signature == "orn:zrrc" ||
      signature == "rol.s:rric" || signature == "rol.s:rrrc" ||
      signature == "rol.u:rric" || signature == "rol.u:rrrc" ||
      signature == "rol:rric" || signature == "rol:rrrc" ||
      signature == "rol:zric" || signature == "rol:zrrc" ||
      signature == "ror.s:rric" || signature == "ror.s:rrrc" ||
      signature == "ror.u:rric" || signature == "ror.u:rrrc" ||
      signature == "ror:rric" || signature == "ror:rrrc" ||
      signature == "ror:zric" || signature == "ror:zrrc" ||
      signature == "rsub.s:rrrc" || signature == "rsub.u:rrrc" ||
      signature == "rsub:rrrc" || signature == "rsub:zrrc" ||
      signature == "rsubc.s:rrrc" || signature == "rsubc.u:rrrc" ||
      signature == "rsubc:rrrc" || signature == "rsubc:zrrc" ||
      signature == "sats.s:rrc" || signature == "sats.u:rrc" ||
      signature == "sats:rrc" || signature == "sats:zrc" ||
      signature == "sub.s:rirc" || signature == "sub.u:rirc" ||
      signature == "sub:rirc" || signature == "sub:zirc" ||
      signature == "subc.s:rirc" || signature == "subc.u:rirc" ||
      signature == "subc:rirc" || signature == "subc:zirc" ||
      signature == "xor.s:rric" || signature == "xor.s:rrrc" ||
      signature == "xor.u:rric" || signature == "xor.u:rrrc" ||
      signature == "xor:rric" || signature == "xor:rrrc" ||
      signature == "xor:zric" || signature == "xor:zrrc")
    return decode_cc_map_1(cc);
  if (signature == "add.s:rrici" || signature == "add.s:rrrci" ||
      signature == "add.u:rrici" || signature == "add.u:rrrci" ||
      signature == "add:rrici" || signature == "add:rrrci" ||
      signature == "add:zrici" || signature == "add:zrrci" ||
      signature == "addc.s:rrici" || signature == "addc.s:rrrci" ||
      signature == "addc.u:rrici" || signature == "addc.u:rrrci" ||
      signature == "addc:rrici" || signature == "addc:rrrci" ||
      signature == "addc:zrici" || signature == "addc:zrrci")
    return decode_cc_map_2(cc);
  if (signature == "add.s:rrif" || signature == "add.u:rrif" ||
      signature == "add:rrif" || signature == "add:zrif" ||
      signature == "addc.s:rrif" || signature == "addc.u:rrif" ||
      signature == "addc:rrif" || signature == "addc:zrif" ||
      signature == "and.s:rrif" || signature == "and.u:rrif" ||
      signature == "and:rrif" || signature == "and:zrif" ||
      signature == "andn.s:rrif" || signature == "andn.u:rrif" ||
      signature == "andn:rrif" || signature == "andn:zrif" ||
      signature == "hash.s:rrif" || signature == "hash.u:rrif" ||
      signature == "hash:rrif" || signature == "hash:zrif" ||
      signature == "nand.s:rrif" || signature == "nand.u:rrif" ||
      signature == "nand:rrif" || signature == "nand:zrif" ||
      signature == "nor.s:rrif" || signature == "nor.u:rrif" ||
      signature == "nor:rrif" || signature == "nor:zrif" ||
      signature == "nxor.s:rrif" || signature == "nxor.u:rrif" ||
      signature == "nxor:rrif" || signature == "nxor:zrif" ||
      signature == "or.s:rrif" || signature == "or.u:rrif" ||
      signature == "or:rrif" || signature == "or:zrif" ||
      signature == "orn.s:rrif" || signature == "orn.u:rrif" ||
      signature == "orn:rrif" || signature == "orn:zrif" ||
      signature == "sub.s:rirf" || signature == "sub.s:rrif" ||
      signature == "sub.u:rirf" || signature == "sub.u:rrif" ||
      signature == "sub:rirf" || signature == "sub:rrif" ||
      signature == "sub:zirf" || signature == "sub:zrif" ||
      signature == "subc.s:rirf" || signature == "subc.s:rrif" ||
      signature == "subc.u:rirf" || signature == "subc.u:rrif" ||
      signature == "subc:rirf" || signature == "subc:rrif" ||
      signature == "subc:zirf" || signature == "subc:zrif" ||
      signature == "xor.s:rrif" || signature == "xor.u:rrif" ||
      signature == "xor:rrif" || signature == "xor:zrif")
    return decode_cc_map_3(cc);
  if (signature == "and.s:rrici" || signature == "and.s:rrrci" ||
      signature == "and.u:rrici" || signature == "and.u:rrrci" ||
      signature == "and:rrici" || signature == "and:rrrci" ||
      signature == "and:zrici" || signature == "and:zrrci" ||
      signature == "andn.s:rrici" || signature == "andn.s:rrrci" ||
      signature == "andn.u:rrici" || signature == "andn.u:rrrci" ||
      signature == "andn:rrici" || signature == "andn:rrrci" ||
      signature == "andn:zrici" || signature == "andn:zrrci" ||
      signature == "cmpb4.s:rrrci" || signature == "cmpb4.u:rrrci" ||
      signature == "cmpb4:rrrci" || signature == "cmpb4:zrrci" ||
      signature == "extsb.s:rrci" || signature == "extsb:rrci" ||
      signature == "extsb:zrci" || signature == "extsh.s:rrci" ||
      signature == "extsh:rrci" || signature == "extsh:zrci" ||
      signature == "extub.u:rrci" || signature == "extub:rrci" ||
      signature == "extub:zrci" || signature == "extuh.u:rrci" ||
      signature == "extuh:rrci" || signature == "extuh:zrci" ||
      signature == "hash.s:rrici" || signature == "hash.s:rrrci" ||
      signature == "hash.u:rrici" || signature == "hash.u:rrrci" ||
      signature == "hash:rrici" || signature == "hash:rrrci" ||
      signature == "hash:zrici" || signature == "hash:zrrci" ||
      signature == "move.s:rici" || signature == "move.s:rrci" ||
      signature == "move.u:rici" || signature == "move.u:rrci" ||
      signature == "move:rici" || signature == "move:rrci" ||
      signature == "nand.s:rrici" || signature == "nand.s:rrrci" ||
      signature == "nand.u:rrici" || signature == "nand.u:rrrci" ||
      signature == "nand:rrici" || signature == "nand:rrrci" ||
      signature == "nand:zrici" || signature == "nand:zrrci" ||
      signature == "nor.s:rrici" || signature == "nor.s:rrrci" ||
      signature == "nor.u:rrici" || signature == "nor.u:rrrci" ||
      signature == "nor:rrici" || signature == "nor:rrrci" ||
      signature == "nor:zrici" || signature == "nor:zrrci" ||
      signature == "not:rci" || signature == "not:rrci" ||
      signature == "nxor.s:rrici" || signature == "nxor.s:rrrci" ||
      signature == "nxor.u:rrici" || signature == "nxor.u:rrrci" ||
      signature == "nxor:rrici" || signature == "nxor:rrrci" ||
      signature == "nxor:zrici" || signature == "nxor:zrrci" ||
      signature == "or.s:rrici" || signature == "or.s:rrrci" ||
      signature == "or.u:rrici" || signature == "or.u:rrrci" ||
      signature == "or:rrici" || signature == "or:rrrci" ||
      signature == "or:zrici" || signature == "or:zrrci" ||
      signature == "orn.s:rrici" || signature == "orn.s:rrrci" ||
      signature == "orn.u:rrici" || signature == "orn.u:rrrci" ||
      signature == "orn:rrici" || signature == "orn:rrrci" ||
      signature == "orn:zrici" || signature == "orn:zrrci" ||
      signature == "sats.s:rrci" || signature == "sats.u:rrci" ||
      signature == "sats:rrci" || signature == "sats:zrci" ||
      signature == "xor.s:rrici" || signature == "xor.s:rrrci" ||
      signature == "xor.u:rrici" || signature == "xor.u:rrrci" ||
      signature == "xor:rrici" || signature == "xor:rrrci" ||
      signature == "xor:zrici" || signature == "xor:zrrci")
    return decode_cc_map_4(cc);
  if (signature == "asr.s:rrici" || signature == "asr.u:rrici" ||
      signature == "asr:rrici" || signature == "asr:zrici" ||
      signature == "lsl.s:rrici" || signature == "lsl.u:rrici" ||
      signature == "lsl1.s:rrici" || signature == "lsl1.u:rrici" ||
      signature == "lsl1:rrici" || signature == "lsl1:zrici" ||
      signature == "lsl1x.s:rrici" || signature == "lsl1x.u:rrici" ||
      signature == "lsl1x:rrici" || signature == "lsl1x:zrici" ||
      signature == "lsl:rrici" || signature == "lsl:zrici" ||
      signature == "lslx.s:rrici" || signature == "lslx.u:rrici" ||
      signature == "lslx:rrici" || signature == "lslx:zrici" ||
      signature == "lsr.s:rrici" || signature == "lsr.u:rrici" ||
      signature == "lsr1.s:rrici" || signature == "lsr1.u:rrici" ||
      signature == "lsr1:rrici" || signature == "lsr1:zrici" ||
      signature == "lsr1x.s:rrici" || signature == "lsr1x.u:rrici" ||
      signature == "lsr1x:rrici" || signature == "lsr1x:zrici" ||
      signature == "lsr:rrici" || signature == "lsr:zrici" ||
      signature == "lsrx.s:rrici" || signature == "lsrx.u:rrici" ||
      signature == "lsrx:rrici" || signature == "lsrx:zrici" ||
      signature == "rol.s:rrici" || signature == "rol.u:rrici" ||
      signature == "rol:rrici" || signature == "rol:zrici" ||
      signature == "ror.s:rrici" || signature == "ror.u:rrici" ||
      signature == "ror:rrici" || signature == "ror:zrici")
    return decode_cc_map_5(cc);
  if (signature == "asr.s:rrrci" || signature == "asr.u:rrrci" ||
      signature == "asr:rrrci" || signature == "asr:zrrci" ||
      signature == "lsl.s:rrrci" || signature == "lsl.u:rrrci" ||
      signature == "lsl1.s:rrrci" || signature == "lsl1.u:rrrci" ||
      signature == "lsl1:rrrci" || signature == "lsl1:zrrci" ||
      signature == "lsl1x.s:rrrci" || signature == "lsl1x.u:rrrci" ||
      signature == "lsl1x:rrrci" || signature == "lsl1x:zrrci" ||
      signature == "lsl:rrrci" || signature == "lsl:zrrci" ||
      signature == "lslx.s:rrrci" || signature == "lslx.u:rrrci" ||
      signature == "lslx:rrrci" || signature == "lslx:zrrci" ||
      signature == "lsr.s:rrrci" || signature == "lsr.u:rrrci" ||
      signature == "lsr1.s:rrrci" || signature == "lsr1.u:rrrci" ||
      signature == "lsr1:rrrci" || signature == "lsr1:zrrci" ||
      signature == "lsr1x.s:rrrci" || signature == "lsr1x.u:rrrci" ||
      signature == "lsr1x:rrrci" || signature == "lsr1x:zrrci" ||
      signature == "lsr:rrrci" || signature == "lsr:zrrci" ||
      signature == "lsrx.s:rrrci" || signature == "lsrx.u:rrrci" ||
      signature == "lsrx:rrrci" || signature == "lsrx:zrrci" ||
      signature == "rol.s:rrrci" || signature == "rol.u:rrrci" ||
      signature == "rol:rrrci" || signature == "rol:zrrci" ||
      signature == "ror.s:rrrci" || signature == "ror.u:rrrci" ||
      signature == "ror:rrrci" || signature == "ror:zrrci")
    return decode_cc_map_6(cc);
  if (signature == "boot:rici" || signature == "clr_run:rici" ||
      signature == "mul_step:rrrici" || signature == "read_run:rici" ||
      signature == "resume:rici" || signature == "stop:ci")
    return decode_cc_map_7(cc);
  if (signature == "cao.s:rrci" || signature == "cao.u:rrci" ||
      signature == "cao:rrci" || signature == "cao:zrci" ||
      signature == "clo.s:rrci" || signature == "clo.u:rrci" ||
      signature == "clo:rrci" || signature == "clo:zrci" ||
      signature == "cls.s:rrci" || signature == "cls.u:rrci" ||
      signature == "cls:rrci" || signature == "cls:zrci" ||
      signature == "clz.s:rrci" || signature == "clz.u:rrci" ||
      signature == "clz:rrci" || signature == "clz:zrci")
    return decode_cc_map_8(cc);
  if (signature == "div_step:rrrici")
    return decode_cc_map_9(cc);
  if (signature == "lsl_add.s:rrrici" || signature == "lsl_add.u:rrrici" ||
      signature == "lsl_add:rrrici" || signature == "lsl_add:zrrici" ||
      signature == "lsl_sub.s:rrrici" || signature == "lsl_sub.u:rrrici" ||
      signature == "lsl_sub:rrrici" || signature == "lsl_sub:zrrici" ||
      signature == "lsr_add.s:rrrici" || signature == "lsr_add.u:rrrici" ||
      signature == "lsr_add:rrrici" || signature == "lsr_add:zrrici" ||
      signature == "rol_add.s:rrrici" || signature == "rol_add.u:rrrici" ||
      signature == "rol_add:rrrici" || signature == "rol_add:zrrici")
    return decode_cc_map_10(cc);
  if (signature == "movd:rrci" || signature == "swapd:rrci")
    return decode_cc_map_11(cc);
  if (signature == "mul_sh_sh.s:rrrci" || signature == "mul_sh_sh:rrrci" ||
      signature == "mul_sh_sh:zrrci" || signature == "mul_sh_sl.s:rrrci" ||
      signature == "mul_sh_sl:rrrci" || signature == "mul_sh_sl:zrrci" ||
      signature == "mul_sh_uh.s:rrrci" || signature == "mul_sh_uh:rrrci" ||
      signature == "mul_sh_uh:zrrci" || signature == "mul_sh_ul.s:rrrci" ||
      signature == "mul_sh_ul:rrrci" || signature == "mul_sh_ul:zrrci" ||
      signature == "mul_sl_sh.s:rrrci" || signature == "mul_sl_sh:rrrci" ||
      signature == "mul_sl_sh:zrrci" || signature == "mul_sl_sl.s:rrrci" ||
      signature == "mul_sl_sl:rrrci" || signature == "mul_sl_sl:zrrci" ||
      signature == "mul_sl_uh.s:rrrci" || signature == "mul_sl_uh:rrrci" ||
      signature == "mul_sl_uh:zrrci" || signature == "mul_sl_ul.s:rrrci" ||
      signature == "mul_sl_ul:rrrci" || signature == "mul_sl_ul:zrrci" ||
      signature == "mul_uh_uh.u:rrrci" || signature == "mul_uh_uh:rrrci" ||
      signature == "mul_uh_uh:zrrci" || signature == "mul_uh_ul.u:rrrci" ||
      signature == "mul_uh_ul:rrrci" || signature == "mul_uh_ul:zrrci" ||
      signature == "mul_ul_uh.u:rrrci" || signature == "mul_ul_uh:rrrci" ||
      signature == "mul_ul_uh:zrrci" || signature == "mul_ul_ul.u:rrrci" ||
      signature == "mul_ul_ul:rrrci" || signature == "mul_ul_ul:zrrci")
    return decode_cc_map_12(cc);
  if (signature == "neg:rrci" || signature == "rsub.s:rrrci" ||
      signature == "rsub.u:rrrci" || signature == "rsub:rrrci" ||
      signature == "rsub:zrrci" || signature == "rsubc.s:rrrci" ||
      signature == "rsubc.u:rrrci" || signature == "rsubc:rrrci" ||
      signature == "rsubc:zrrci" || signature == "sub.s:rirci" ||
      signature == "sub.s:rrici" || signature == "sub.s:rrrci" ||
      signature == "sub.u:rirci" || signature == "sub.u:rrici" ||
      signature == "sub.u:rrrci" || signature == "sub:rirci" ||
      signature == "sub:rrici" || signature == "sub:rrrci" ||
      signature == "sub:zirci" || signature == "sub:zrici" ||
      signature == "sub:zrrci" || signature == "subc.s:rirci" ||
      signature == "subc.s:rrici" || signature == "subc.s:rrrci" ||
      signature == "subc.u:rirci" || signature == "subc.u:rrici" ||
      signature == "subc.u:rrrci" || signature == "subc:rirci" ||
      signature == "subc:rrici" || signature == "subc:rrrci" ||
      signature == "subc:zirci" || signature == "subc:zrici" ||
      signature == "subc:zrrci")
    return decode_cc_map_13(cc);
  if (signature == "release:rici")
    return decode_cc_map_14(cc);
  if (signature == "sub.s:rric" || signature == "sub.s:rrrc" ||
      signature == "sub.u:rric" || signature == "sub.u:rrrc" ||
      signature == "sub:rric" || signature == "sub:rrrc" ||
      signature == "sub:zric" || signature == "sub:zrrc" ||
      signature == "subc.s:rric" || signature == "subc.s:rrrc" ||
      signature == "subc.u:rric" || signature == "subc.u:rrrc" ||
      signature == "subc:rric" || signature == "subc:rrrc" ||
      signature == "subc:zric" || signature == "subc:zrrc")
    return decode_cc_map_15(cc);
  if (signature == "time.s:rci" || signature == "time.u:rci" ||
      signature == "time:rci" || signature == "time:zci" ||
      signature == "time_cfg.s:rrci" || signature == "time_cfg.u:rrci" ||
      signature == "time_cfg:rrci" || signature == "time_cfg:zrci")
    return decode_cc_map_16(cc);
  return "";
}

static inline bool decode_raw_word_48(uint64_t instruction, RawDecoded48 &out) {
  out = RawDecoded48{};
  if ((((instruction >> 44) & 0xf)) == (0x7)) {
    if ((((instruction >> 32) & 0x3)) == (0x3)) {
      if ((((instruction >> 42) & 0x3)) == (0x3)) {
        if (((((instruction >> 28) & 0xf)) & (0x8)) == (0x8)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
          out.has_pc = true;
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 16) & 15) << 4) |
                                         (((instruction >> 26) & 31) << 8) |
                                         (((instruction >> 39) & 7) << 13));
          out.has_imm = true;
          if ((((instruction >> 24) & 0x3)) != (0x0)) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 3) << 0));
            out.has_cc = true;
            out.valid = true;
            out.signature = "acquire:rici";
            return true;
          }
          if ((((instruction >> 24) & 0x3)) == (0x0)) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 3) << 0));
            out.has_cc = true;
            out.valid = true;
            out.signature = "release:rici";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
          if (((((instruction >> 39) & 0x1f)) & (0x1b)) == (0x1b)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if (((((instruction >> 24) & 0xf)) < (0x6)) ||
                ((((instruction >> 24) & 0xf)) > (0xb))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "boot:rici";
              return true;
            }
            return false;
          }
          if (((((instruction >> 39) & 0x1f)) & (0x1f)) == (0x19)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if (((((instruction >> 24) & 0xf)) < (0x6)) ||
                ((((instruction >> 24) & 0xf)) > (0xb))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "clr_run:rici";
              return true;
            }
            return false;
          }
          if (((((instruction >> 39) & 0x1f)) & (0x1f)) == (0x1c)) {
            if ((((instruction >> 34) & 0x1f)) == (0x18)) {
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.imm =
                    static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 16) & 15) << 4) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12));
                out.has_imm = true;
                out.valid = true;
                out.signature = "fault:i";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 39) & 0x1f)) & (0x1f)) == (0x18)) {
            if (((((instruction >> 28) & 0xf)) & (0xf)) == (0x3)) {
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if (((((instruction >> 24) & 0xf)) < (0x6)) ||
                  ((((instruction >> 24) & 0xf)) > (0xb))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.imm =
                    static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 16) & 15) << 4));
                out.has_imm = true;
                out.valid = true;
                out.signature = "read_run:rici";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 39) & 0x1f)) & (0x1b)) == (0x1a)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if (((((instruction >> 24) & 0xf)) < (0x6)) ||
                ((((instruction >> 24) & 0xf)) > (0xb))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "resume:rici";
              return true;
            }
            return false;
          }
          if (((((instruction >> 39) & 0x1f)) & (0x1f)) == (0x1d)) {
            if ((((instruction >> 34) & 0x1f)) == (0x1c)) {
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if (((((instruction >> 24) & 0xf)) < (0x6)) ||
                  ((((instruction >> 24) & 0xf)) > (0xb))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "stop:ci";
                return true;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xf)) == (0x0)) {
          if (((((instruction >> 39) & 0x1f)) & (0x1f)) == (0x18)) {
            if ((((instruction >> 34) & 0x1f)) == (0x18)) {
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x0)) {
                  if (((((instruction >> 0) & 0xffff)) & (0xffff)) == (0x0)) {
                    out.valid = true;
                    out.signature = "nop:";
                    return true;
                  }
                  return false;
                }
                return false;
              }
              return false;
            }
            if ((((instruction >> 37) & 0x3)) != (0x3)) {
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.off =
                    static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 16) & 15) << 4) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12));
                out.has_off = true;
                out.valid = true;
                out.signature = "tell:ri";
                return true;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
          if ((((instruction >> 28) & 0x1)) == (0x0)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            out.off = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 39) & 7) << 4) |
                                           (((instruction >> 24) & 1) << 7) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11));
            out.has_off = true;
            out.endian = static_cast<int32_t>((((instruction >> 27) & 1) << 0));
            out.has_endian = true;
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x0)) {
              out.imm = static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                             (((instruction >> 0) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sb:erii";
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x6)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sd:erii";
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x2)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sh:erii";
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sw:erii";
              return true;
            }
            return false;
          }
          if ((((instruction >> 28) & 0x1)) == (0x1)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            out.off = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 39) & 7) << 4) |
                                           (((instruction >> 24) & 1) << 7) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11));
            out.has_off = true;
            out.endian = static_cast<int32_t>((((instruction >> 27) & 1) << 0));
            out.has_endian = true;
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x0)) {
              out.imm = static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                             (((instruction >> 0) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sb:esii";
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x6)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sd:esii";
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x2)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sh:esii";
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sw:esii";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          out.off = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 39) & 7) << 4) |
                                         (((instruction >> 24) & 1) << 7) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11));
          out.has_off = true;
          out.endian = static_cast<int32_t>((((instruction >> 27) & 1) << 0));
          out.has_endian = true;
          if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                           (((instruction >> 0) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sb_id:erii";
            return true;
          }
          if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x6)) {
            out.imm = static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                           (((instruction >> 0) & 4095) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sd_id:erii";
            return true;
          }
          if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x2)) {
            out.imm = static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                           (((instruction >> 0) & 4095) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sh_id:erii";
            return true;
          }
          if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
            out.imm = static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                           (((instruction >> 0) & 4095) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sw_id:erii";
            return true;
          }
          return false;
        }
        return false;
      }
      if ((((instruction >> 42) & 0x3)) != (0x3)) {
        if ((((instruction >> 25) & 0x3)) != (0x3)) {
          if (((((instruction >> 28) & 0xf)) & (0xc)) == (0x4)) {
            if ((((instruction >> 28) & 0x1)) == (0x0)) {
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              out.off =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_off = true;
              out.endian =
                  static_cast<int32_t>((((instruction >> 27) & 1) << 0));
              out.has_endian = true;
              if (((((instruction >> 24) & 0xf)) & (0x7)) == (0x1)) {
                if (((((instruction >> 29) & 0x1)) == (0x1)) &&
                    ((((instruction >> 39) & 0x1)) == (0x1))) {
                  out.dc =
                      static_cast<int32_t>((((instruction >> 40) & 15) << 1));
                  out.has_dc = true;
                  out.valid = true;
                  out.signature = "lbs.s:erri";
                  return true;
                }
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  out.signature = "lbs:erri";
                  return true;
                }
                return false;
              }
              if (((((instruction >> 24) & 0xf)) & (0x7)) == (0x0)) {
                if (((((instruction >> 29) & 0x1)) == (0x1)) &&
                    ((((instruction >> 39) & 0x1)) == (0x0))) {
                  out.dc =
                      static_cast<int32_t>((((instruction >> 40) & 15) << 1));
                  out.has_dc = true;
                  out.valid = true;
                  out.signature = "lbu.u:erri";
                  return true;
                }
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  out.signature = "lbu:erri";
                  return true;
                }
                return false;
              }
              if (((((instruction >> 24) & 0xf)) & (0x7)) == (0x3)) {
                if (((((instruction >> 29) & 0x1)) == (0x1)) &&
                    ((((instruction >> 39) & 0x1)) == (0x1))) {
                  out.dc =
                      static_cast<int32_t>((((instruction >> 40) & 15) << 1));
                  out.has_dc = true;
                  out.valid = true;
                  out.signature = "lhs.s:erri";
                  return true;
                }
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  out.signature = "lhs:erri";
                  return true;
                }
                return false;
              }
              if (((((instruction >> 24) & 0xf)) & (0x7)) == (0x2)) {
                if (((((instruction >> 29) & 0x1)) == (0x1)) &&
                    ((((instruction >> 39) & 0x1)) == (0x0))) {
                  out.dc =
                      static_cast<int32_t>((((instruction >> 40) & 15) << 1));
                  out.has_dc = true;
                  out.valid = true;
                  out.signature = "lhu.u:erri";
                  return true;
                }
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  out.signature = "lhu:erri";
                  return true;
                }
                return false;
              }
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
                if (((((instruction >> 29) & 0x1)) == (0x1)) &&
                    ((((instruction >> 39) & 0x1)) == (0x1))) {
                  out.dc =
                      static_cast<int32_t>((((instruction >> 40) & 15) << 1));
                  out.has_dc = true;
                  out.valid = true;
                  out.signature = "lw.s:erri";
                  return true;
                }
                if (((((instruction >> 29) & 0x1)) == (0x1)) &&
                    ((((instruction >> 39) & 0x1)) == (0x0))) {
                  out.dc =
                      static_cast<int32_t>((((instruction >> 40) & 15) << 1));
                  out.has_dc = true;
                  out.valid = true;
                  out.signature = "lw.u:erri";
                  return true;
                }
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  out.signature = "lw:erri";
                  return true;
                }
                return false;
              }
              return false;
            }
            if ((((instruction >> 28) & 0x1)) == (0x1)) {
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              out.off =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_off = true;
              out.endian =
                  static_cast<int32_t>((((instruction >> 27) & 1) << 0));
              out.has_endian = true;
              if (((((instruction >> 24) & 0xf)) & (0x7)) == (0x1)) {
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  out.signature = "lbs:ersi";
                  return true;
                }
                return false;
              }
              if (((((instruction >> 24) & 0xf)) & (0x7)) == (0x0)) {
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  out.signature = "lbu:ersi";
                  return true;
                }
                return false;
              }
              if (((((instruction >> 24) & 0xf)) & (0x7)) == (0x3)) {
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  out.signature = "lhs:ersi";
                  return true;
                }
                return false;
              }
              if (((((instruction >> 24) & 0xf)) & (0x7)) == (0x2)) {
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  out.signature = "lhu:ersi";
                  return true;
                }
                return false;
              }
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  out.signature = "lw:ersi";
                  return true;
                }
                return false;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        if ((((instruction >> 25) & 0x3)) == (0x3)) {
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
            if ((((instruction >> 28) & 0x1)) == (0x0)) {
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              out.off =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_off = true;
              out.endian =
                  static_cast<int32_t>((((instruction >> 27) & 1) << 0));
              out.has_endian = true;
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x6)) {
                out.dc =
                    static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                out.has_dc = true;
                out.valid = true;
                out.signature = "ld:erri";
                return true;
              }
              return false;
            }
            if ((((instruction >> 28) & 0x1)) == (0x1)) {
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              out.off =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_off = true;
              out.endian =
                  static_cast<int32_t>((((instruction >> 27) & 1) << 0));
              out.has_endian = true;
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x6)) {
                out.dc =
                    static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                out.has_dc = true;
                out.valid = true;
                out.signature = "ld:ersi";
                return true;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      return false;
    }
    if ((((instruction >> 32) & 0x3)) != (0x3)) {
      if (((((instruction >> 39) & 0x1f)) & (0x1f)) == (0x0)) {
        out.immDma = static_cast<int32_t>((((instruction >> 24) & 255) << 0));
        out.has_immDma = true;
        out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                      (((instruction >> 32) & 3) << 3));
        out.has_rb = true;
        out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
        out.has_ra = true;
        if (((((instruction >> 0) & 0xffff)) & (0xffff)) == (0x0)) {
          out.valid = true;
          out.signature = "ldma:rri";
          return true;
        }
        if (((((instruction >> 0) & 0xffff)) & (0xffff)) == (0x1)) {
          out.valid = true;
          out.signature = "ldmai:rri";
          return true;
        }
        if (((((instruction >> 0) & 0xffff)) & (0xffff)) == (0x2)) {
          out.valid = true;
          out.signature = "sdma:rri";
          return true;
        }
        return false;
      }
      if ((((instruction >> 42) & 0x3)) == (0x3)) {
        if ((((instruction >> 16) & 0x1)) == (0x0)) {
          out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                        (((instruction >> 32) & 3) << 3));
          out.has_rb = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 28) & 0xf)) & (0xc)) == (0x4)) {
            if ((((instruction >> 28) & 0x1)) == (0x0)) {
              out.off =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 39) & 7) << 4) |
                                       (((instruction >> 24) & 1) << 7) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_off = true;
              out.endian =
                  static_cast<int32_t>((((instruction >> 27) & 1) << 0));
              out.has_endian = true;
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x0)) {
                out.valid = true;
                out.signature = "sb:erir";
                return true;
              }
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x2)) {
                out.valid = true;
                out.signature = "sh:erir";
                return true;
              }
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
                out.valid = true;
                out.signature = "sw:erir";
                return true;
              }
              return false;
            }
            if ((((instruction >> 28) & 0x1)) == (0x1)) {
              out.off =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 39) & 7) << 4) |
                                       (((instruction >> 24) & 1) << 7) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_off = true;
              out.endian =
                  static_cast<int32_t>((((instruction >> 27) & 1) << 0));
              out.has_endian = true;
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x0)) {
                out.valid = true;
                out.signature = "sb:esir";
                return true;
              }
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x2)) {
                out.valid = true;
                out.signature = "sh:esir";
                return true;
              }
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
                out.valid = true;
                out.signature = "sw:esir";
                return true;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        if ((((instruction >> 16) & 0x1)) == (0x1)) {
          out.db = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                        (((instruction >> 32) & 3) << 3));
          out.has_db = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 28) & 0xf)) & (0xc)) == (0x4)) {
            if ((((instruction >> 28) & 0x1)) == (0x0)) {
              out.off =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 39) & 7) << 4) |
                                       (((instruction >> 24) & 1) << 7) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_off = true;
              out.endian =
                  static_cast<int32_t>((((instruction >> 27) & 1) << 0));
              out.has_endian = true;
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x6)) {
                out.valid = true;
                out.signature = "sd:erir";
                return true;
              }
              return false;
            }
            if ((((instruction >> 28) & 0x1)) == (0x1)) {
              out.off =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 39) & 7) << 4) |
                                       (((instruction >> 24) & 1) << 7) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_off = true;
              out.endian =
                  static_cast<int32_t>((((instruction >> 27) & 1) << 0));
              out.has_endian = true;
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x6)) {
                out.valid = true;
                out.signature = "sd:esir";
                return true;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      return false;
    }
    return false;
  }
  if (((((instruction >> 45) & 0x7)) == (0x0)) &&
      ((((instruction >> 42) & 0x3)) != (0x3))) {
    if (((((instruction >> 32) & 0x3)) != (0x3)) &&
        ((((instruction >> 16) & 0x1)) == (0x0))) {
      out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                    (((instruction >> 32) & 3) << 3));
      out.has_rb = true;
      if (((((instruction >> 42) & 0x3)) != (0x3)) &&
          ((((instruction >> 39) & 0x1)) == (0x1))) {
        out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
        out.has_dc = true;
        if ((((instruction >> 37) & 0x3)) == (0x3)) {
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 34) & 7) << 4) |
                                         (((instruction >> 44) & 1) << 7) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "add.s:rri";
          return true;
        }
        return false;
      }
      if (((((instruction >> 42) & 0x3)) != (0x3)) &&
          ((((instruction >> 39) & 0x1)) == (0x0))) {
        out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
        out.has_dc = true;
        if ((((instruction >> 37) & 0x3)) == (0x3)) {
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 34) & 7) << 4) |
                                         (((instruction >> 44) & 1) << 7) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "add.u:rri";
          return true;
        }
        return false;
      }
      return false;
    }
    if ((((instruction >> 32) & 0x3)) == (0x3)) {
      if (((((instruction >> 44) & 0xf)) == (0x0)) &&
          ((((instruction >> 42) & 0x3)) != (0x3))) {
        if ((((instruction >> 42) & 0x3)) != (0x3)) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 16) & 15) << 4) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "add:rri";
          return true;
        }
        return false;
      }
      if (((((instruction >> 44) & 0xf)) == (0x1)) &&
          ((((instruction >> 42) & 0x3)) != (0x3))) {
        if ((((instruction >> 42) & 0x3)) != (0x3)) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 16) & 15) << 4) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "addc:rri";
          return true;
        }
        return false;
      }
      return false;
    }
    return false;
  }
  if ((((((instruction >> 47) & 0x1)) == (0x0)) &&
       ((((instruction >> 44) & 0x7)) != (0x7))) &&
      ((((instruction >> 42) & 0x3)) == (0x3))) {
    if ((((instruction >> 32) & 0x3)) == (0x3)) {
      if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
           ((((instruction >> 46) & 0x1)) == (0x1))) &&
          ((((instruction >> 39) & 0x1)) == (0x1))) {
        out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                      (((instruction >> 44) & 3) << 3));
        out.has_dc = true;
        out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
        out.has_ra = true;
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x0)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                (((((instruction >> 25) & 0x1)) == (0x1)) &&
                 ((((instruction >> 28) & 0x1)) == (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add.s:rric";
              return true;
            }
            if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                 (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                  (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                ((((instruction >> 24) & 0x1f)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add.s:rrici";
              return true;
            }
            if ((((instruction >> 24) & 0x1f)) == (0x0)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add.s:rrif";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                (((((instruction >> 25) & 0x1)) == (0x1)) &&
                 ((((instruction >> 28) & 0x1)) == (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc.s:rric";
              return true;
            }
            if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                 (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                  (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                ((((instruction >> 24) & 0x1f)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc.s:rrici";
              return true;
            }
            if ((((instruction >> 24) & 0x1f)) == (0x0)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc.s:rrif";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              (((((instruction >> 25) & 0x1)) == (0x1)) &&
               ((((instruction >> 28) & 0x1)) == (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub.s:rirc";
            return true;
          }
          if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
               (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
              ((((instruction >> 24) & 0x1f)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub.s:rirci";
            return true;
          }
          if ((((instruction >> 24) & 0x1f)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub.s:rirf";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xa)) == (0x8)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                 ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                (((((instruction >> 30) & 0x1)) != (0x1)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub.s:rric";
              return true;
            }
            if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                 (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                  (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                   (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                (((((instruction >> 30) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub.s:rrici";
              return true;
            }
            if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                ((((instruction >> 30) & 0x1)) == (0x0))) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub.s:rrif";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              (((((instruction >> 25) & 0x1)) == (0x1)) &&
               ((((instruction >> 28) & 0x1)) == (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc.s:rirc";
            return true;
          }
          if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
               (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
              ((((instruction >> 24) & 0x1f)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc.s:rirci";
            return true;
          }
          if ((((instruction >> 24) & 0x1f)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc.s:rirf";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xa)) == (0xa)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                 ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                (((((instruction >> 30) & 0x1)) != (0x1)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc.s:rric";
              return true;
            }
            if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                 (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                  (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                   (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                (((((instruction >> 30) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc.s:rrici";
              return true;
            }
            if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                ((((instruction >> 30) & 0x1)) == (0x0))) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc.s:rrif";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
           ((((instruction >> 46) & 0x1)) == (0x1))) &&
          ((((instruction >> 39) & 0x1)) == (0x0))) {
        out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                      (((instruction >> 44) & 3) << 3));
        out.has_dc = true;
        out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
        out.has_ra = true;
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x0)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                (((((instruction >> 25) & 0x1)) == (0x1)) &&
                 ((((instruction >> 28) & 0x1)) == (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add.u:rric";
              return true;
            }
            if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                 (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                  (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                ((((instruction >> 24) & 0x1f)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add.u:rrici";
              return true;
            }
            if ((((instruction >> 24) & 0x1f)) == (0x0)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add.u:rrif";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                (((((instruction >> 25) & 0x1)) == (0x1)) &&
                 ((((instruction >> 28) & 0x1)) == (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc.u:rric";
              return true;
            }
            if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                 (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                  (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                ((((instruction >> 24) & 0x1f)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc.u:rrici";
              return true;
            }
            if ((((instruction >> 24) & 0x1f)) == (0x0)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc.u:rrif";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              (((((instruction >> 25) & 0x1)) == (0x1)) &&
               ((((instruction >> 28) & 0x1)) == (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub.u:rirc";
            return true;
          }
          if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
               (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
              ((((instruction >> 24) & 0x1f)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub.u:rirci";
            return true;
          }
          if ((((instruction >> 24) & 0x1f)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub.u:rirf";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xa)) == (0x8)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                 ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                (((((instruction >> 30) & 0x1)) != (0x1)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub.u:rric";
              return true;
            }
            if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                 (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                  (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                   (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                (((((instruction >> 30) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub.u:rrici";
              return true;
            }
            if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                ((((instruction >> 30) & 0x1)) == (0x0))) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub.u:rrif";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              (((((instruction >> 25) & 0x1)) == (0x1)) &&
               ((((instruction >> 28) & 0x1)) == (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc.u:rirc";
            return true;
          }
          if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
               (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
              ((((instruction >> 24) & 0x1f)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc.u:rirci";
            return true;
          }
          if ((((instruction >> 24) & 0x1f)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc.u:rirf";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xa)) == (0xa)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                 ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                (((((instruction >> 30) & 0x1)) != (0x1)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc.u:rric";
              return true;
            }
            if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                 (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                  (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                   (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                (((((instruction >> 30) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc.u:rrici";
              return true;
            }
            if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                ((((instruction >> 30) & 0x1)) == (0x0))) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc.u:rrif";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 44) & 0x3)) != (0x3)) &&
          ((((instruction >> 46) & 0x1)) == (0x0))) {
        out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                      (((instruction >> 44) & 3) << 3));
        out.has_rc = true;
        out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
        out.has_ra = true;
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x0)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                (((((instruction >> 25) & 0x1)) == (0x1)) &&
                 ((((instruction >> 28) & 0x1)) == (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add:rric";
              return true;
            }
            if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                 (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                  (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                ((((instruction >> 24) & 0x1f)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add:rrici";
              return true;
            }
            if ((((instruction >> 24) & 0x1f)) == (0x0)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add:rrif";
              return true;
            }
            return false;
          }
          if (((((instruction >> 28) & 0x1)) == (0x1)) &&
              ((((instruction >> 24) & 0xf)) == (0x0))) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 31) << 12) |
                                           (((instruction >> 5) & 1) << 16) |
                                           (((instruction >> 6) & 1) << 16) |
                                           (((instruction >> 7) & 1) << 16) |
                                           (((instruction >> 8) & 1) << 16) |
                                           (((instruction >> 9) & 1) << 16) |
                                           (((instruction >> 10) & 1) << 16) |
                                           (((instruction >> 11) & 1) << 16));
            out.has_imm = true;
            out.valid = true;
            out.signature = "add:ssi";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                (((((instruction >> 25) & 0x1)) == (0x1)) &&
                 ((((instruction >> 28) & 0x1)) == (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc:rric";
              return true;
            }
            if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                 (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                  (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                ((((instruction >> 24) & 0x1f)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc:rrici";
              return true;
            }
            if ((((instruction >> 24) & 0x1f)) == (0x0)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc:rrif";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              (((((instruction >> 25) & 0x1)) == (0x1)) &&
               ((((instruction >> 28) & 0x1)) == (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub:rirc";
            return true;
          }
          if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
               (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
              ((((instruction >> 24) & 0x1f)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub:rirci";
            return true;
          }
          if ((((instruction >> 24) & 0x1f)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub:rirf";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xa)) == (0x8)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                 ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                (((((instruction >> 30) & 0x1)) != (0x1)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub:rric";
              return true;
            }
            if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                 (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                  (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                   (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                (((((instruction >> 30) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub:rrici";
              return true;
            }
            if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                ((((instruction >> 30) & 0x1)) == (0x0))) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub:rrif";
              return true;
            }
            return false;
          }
          if (((((instruction >> 28) & 0x1)) == (0x1)) &&
              ((((instruction >> 24) & 0xf)) == (0x0))) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 31) << 12) |
                                           (((instruction >> 5) & 1) << 16) |
                                           (((instruction >> 6) & 1) << 16) |
                                           (((instruction >> 7) & 1) << 16) |
                                           (((instruction >> 8) & 1) << 16) |
                                           (((instruction >> 9) & 1) << 16) |
                                           (((instruction >> 10) & 1) << 16) |
                                           (((instruction >> 11) & 1) << 16));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub:ssi";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              (((((instruction >> 25) & 0x1)) == (0x1)) &&
               ((((instruction >> 28) & 0x1)) == (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc:rirc";
            return true;
          }
          if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
               (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
              ((((instruction >> 24) & 0x1f)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc:rirci";
            return true;
          }
          if ((((instruction >> 24) & 0x1f)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc:rirf";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xa)) == (0xa)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                 ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                (((((instruction >> 30) & 0x1)) != (0x1)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc:rric";
              return true;
            }
            if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                 (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                  (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                   (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                (((((instruction >> 30) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc:rrici";
              return true;
            }
            if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                ((((instruction >> 30) & 0x1)) == (0x0))) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc:rrif";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if ((((instruction >> 44) & 0x3)) == (0x3)) {
        out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
        out.has_ra = true;
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x0)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                (((((instruction >> 25) & 0x1)) == (0x1)) &&
                 ((((instruction >> 28) & 0x1)) == (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12) |
                                       (((instruction >> 39) & 7) << 24));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add:zric";
              return true;
            }
            if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                 (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                  (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                ((((instruction >> 24) & 0x1f)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4) |
                                             (((instruction >> 39) & 7) << 8));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add:zrici";
              return true;
            }
            if ((((instruction >> 24) & 0x1f)) == (0x0)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12) |
                                       (((instruction >> 39) & 7) << 24));
              out.has_imm = true;
              out.valid = true;
              out.signature = "add:zrif";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                (((((instruction >> 25) & 0x1)) == (0x1)) &&
                 ((((instruction >> 28) & 0x1)) == (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12) |
                                       (((instruction >> 39) & 7) << 24));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc:zric";
              return true;
            }
            if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                 (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                  (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                ((((instruction >> 24) & 0x1f)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4) |
                                             (((instruction >> 39) & 7) << 8));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc:zrici";
              return true;
            }
            if ((((instruction >> 24) & 0x1f)) == (0x0)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12) |
                                       (((instruction >> 39) & 7) << 24));
              out.has_imm = true;
              out.valid = true;
              out.signature = "addc:zrif";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              (((((instruction >> 25) & 0x1)) == (0x1)) &&
               ((((instruction >> 28) & 0x1)) == (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub:zirc";
            return true;
          }
          if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
               (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
              ((((instruction >> 24) & 0x1f)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub:zirci";
            return true;
          }
          if ((((instruction >> 24) & 0x1f)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24));
            out.has_imm = true;
            out.valid = true;
            out.signature = "sub:zirf";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xa)) == (0x8)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                 ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                (((((instruction >> 30) & 0x1)) != (0x1)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12) |
                                       (((instruction >> 39) & 7) << 24));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub:zric";
              return true;
            }
            if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                 (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                  (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                   (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                (((((instruction >> 30) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4) |
                                             (((instruction >> 39) & 7) << 8));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub:zrici";
              return true;
            }
            if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                ((((instruction >> 30) & 0x1)) == (0x0))) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12) |
                                       (((instruction >> 39) & 7) << 24));
              out.has_imm = true;
              out.valid = true;
              out.signature = "sub:zrif";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              (((((instruction >> 25) & 0x1)) == (0x1)) &&
               ((((instruction >> 28) & 0x1)) == (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc:zirc";
            return true;
          }
          if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
               (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
              ((((instruction >> 24) & 0x1f)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc:zirci";
            return true;
          }
          if ((((instruction >> 24) & 0x1f)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24));
            out.has_imm = true;
            out.valid = true;
            out.signature = "subc:zirf";
            return true;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0xa)) == (0xa)) {
          if (((((instruction >> 28) & 0x1)) != (0x1)) ||
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                 ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                (((((instruction >> 30) & 0x1)) != (0x1)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12) |
                                       (((instruction >> 39) & 7) << 24));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc:zric";
              return true;
            }
            if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                 (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                  (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                   (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                (((((instruction >> 30) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0x1f)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                            (((instruction >> 30) & 1) << 5));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 16) & 15) << 4) |
                                             (((instruction >> 39) & 7) << 8));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc:zrici";
              return true;
            }
            if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                ((((instruction >> 30) & 0x1)) == (0x0))) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 16) & 15) << 4) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12) |
                                       (((instruction >> 39) & 7) << 24));
              out.has_imm = true;
              out.valid = true;
              out.signature = "subc:zrif";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      return false;
    }
    if (((((instruction >> 32) & 0x3)) != (0x3)) &&
        ((((instruction >> 16) & 0x1)) == (0x0))) {
      if (((((instruction >> 20) & 0xf)) & (0xc)) == (0xc)) {
        if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
             ((((instruction >> 46) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                        (((instruction >> 44) & 3) << 3));
          out.has_dc = true;
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x0)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "add.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "add.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "add.s:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "addc.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "addc.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "addc.s:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "rsub.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "rsub.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "rsub.s:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "rsubc.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "rsubc.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "rsubc.s:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xa)) == (0x8)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                  ((((instruction >> 30) & 0x1)) == (0x0))) {
                out.valid = true;
                out.signature = "sub.s:rrr";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                   ((((((instruction >> 26) & 0x1)) ^
                      (((instruction >> 27) & 0x1))) == (0x1)) &&
                    (((((instruction >> 25) & 0x1)) == (0x1)) &&
                     ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x1)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.valid = true;
                out.signature = "sub.s:rrrc";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                   (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                    (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                     (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x0)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "sub.s:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xa)) == (0xa)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                  ((((instruction >> 30) & 0x1)) == (0x0))) {
                out.valid = true;
                out.signature = "subc.s:rrr";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                   ((((((instruction >> 26) & 0x1)) ^
                      (((instruction >> 27) & 0x1))) == (0x1)) &&
                    (((((instruction >> 25) & 0x1)) == (0x1)) &&
                     ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x1)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.valid = true;
                out.signature = "subc.s:rrrc";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                   (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                    (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                     (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x0)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "subc.s:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
             ((((instruction >> 46) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                        (((instruction >> 44) & 3) << 3));
          out.has_dc = true;
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x0)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "add.u:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "add.u:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "add.u:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "addc.u:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "addc.u:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "addc.u:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "rsub.u:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "rsub.u:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "rsub.u:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "rsubc.u:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "rsubc.u:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "rsubc.u:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xa)) == (0x8)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                  ((((instruction >> 30) & 0x1)) == (0x0))) {
                out.valid = true;
                out.signature = "sub.u:rrr";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                   ((((((instruction >> 26) & 0x1)) ^
                      (((instruction >> 27) & 0x1))) == (0x1)) &&
                    (((((instruction >> 25) & 0x1)) == (0x1)) &&
                     ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x1)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.valid = true;
                out.signature = "sub.u:rrrc";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                   (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                    (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                     (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x0)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "sub.u:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xa)) == (0xa)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                  ((((instruction >> 30) & 0x1)) == (0x0))) {
                out.valid = true;
                out.signature = "subc.u:rrr";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                   ((((((instruction >> 26) & 0x1)) ^
                      (((instruction >> 27) & 0x1))) == (0x1)) &&
                    (((((instruction >> 25) & 0x1)) == (0x1)) &&
                     ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x1)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.valid = true;
                out.signature = "subc.u:rrrc";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                   (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                    (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                     (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x0)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "subc.u:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 44) & 0x3)) != (0x3)) &&
            ((((instruction >> 46) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                        (((instruction >> 44) & 3) << 3));
          out.has_rc = true;
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x0)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "add:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "add:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "add:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 28) & 0x1)) == (0x1)) &&
                ((((instruction >> 24) & 0xf)) == (0x0))) {
              if ((((instruction >> 37) & 0x3)) != (0x3)) {
                out.rb =
                    static_cast<int32_t>((((instruction >> 34) & 31) << 0));
                out.has_rb = true;
                out.ra = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                              (((instruction >> 32) & 3) << 3));
                out.has_ra = true;
                out.valid = true;
                out.signature = "add:sss";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "addc:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "addc:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "addc:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "rsub:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "rsub:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "rsub:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 28) & 0x1)) == (0x1)) &&
                ((((instruction >> 24) & 0xf)) == (0x0))) {
              if ((((instruction >> 37) & 0x3)) != (0x3)) {
                out.rb =
                    static_cast<int32_t>((((instruction >> 34) & 31) << 0));
                out.has_rb = true;
                out.ra = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                              (((instruction >> 32) & 3) << 3));
                out.has_ra = true;
                out.valid = true;
                out.signature = "sub:sss";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "rsubc:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "rsubc:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "rsubc:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xa)) == (0x8)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                  ((((instruction >> 30) & 0x1)) == (0x0))) {
                out.valid = true;
                out.signature = "sub:rrr";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                   ((((((instruction >> 26) & 0x1)) ^
                      (((instruction >> 27) & 0x1))) == (0x1)) &&
                    (((((instruction >> 25) & 0x1)) == (0x1)) &&
                     ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x1)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.valid = true;
                out.signature = "sub:rrrc";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                   (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                    (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                     (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x0)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "sub:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xa)) == (0xa)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                  ((((instruction >> 30) & 0x1)) == (0x0))) {
                out.valid = true;
                out.signature = "subc:rrr";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                   ((((((instruction >> 26) & 0x1)) ^
                      (((instruction >> 27) & 0x1))) == (0x1)) &&
                    (((((instruction >> 25) & 0x1)) == (0x1)) &&
                     ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x1)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.valid = true;
                out.signature = "subc:rrrc";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                   (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                    (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                     (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x0)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "subc:rrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        if ((((instruction >> 44) & 0x3)) == (0x3)) {
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x0)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "add:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "add:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "add:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "addc:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "addc:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "addc:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "rsub:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "rsub:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "rsub:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((instruction >> 24) & 0x1f)) == (0x0)) {
                out.valid = true;
                out.signature = "rsubc:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  (((((instruction >> 25) & 0x1)) == (0x1)) &&
                   ((((instruction >> 28) & 0x1)) == (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "rsubc:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0x1f)) < (0x6)) ||
                   (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                    (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "rsubc:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xa)) == (0x8)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                  ((((instruction >> 30) & 0x1)) == (0x0))) {
                out.valid = true;
                out.signature = "sub:zrr";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                   ((((((instruction >> 26) & 0x1)) ^
                      (((instruction >> 27) & 0x1))) == (0x1)) &&
                    (((((instruction >> 25) & 0x1)) == (0x1)) &&
                     ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x1)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.valid = true;
                out.signature = "sub:zrrc";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                   (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                    (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                     (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x0)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "sub:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 28) & 0xf)) & (0xa)) == (0xa)) {
            if (((((instruction >> 28) & 0x1)) != (0x1)) ||
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if (((((instruction >> 24) & 0x1f)) == (0x0)) &&
                  ((((instruction >> 30) & 0x1)) == (0x0))) {
                out.valid = true;
                out.signature = "subc:zrr";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x1)) ||
                   ((((((instruction >> 26) & 0x1)) ^
                      (((instruction >> 27) & 0x1))) == (0x1)) &&
                    (((((instruction >> 25) & 0x1)) == (0x1)) &&
                     ((((instruction >> 28) & 0x1)) == (0x0))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x1)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.valid = true;
                out.signature = "subc:zrrc";
                return true;
              }
              if ((((((instruction >> 30) & 0x1)) == (0x0)) &&
                   (((((instruction >> 24) & 0x1f)) < (0x6)) ||
                    (((((instruction >> 24) & 0x1f)) > (0xb)) ||
                     (((((instruction >> 24) & 0x1f)) & (0x1e)) == (0x8))))) &&
                  (((((instruction >> 30) & 0x1)) != (0x0)) ||
                   ((((instruction >> 24) & 0x1f)) != (0x0)))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0) |
                                         (((instruction >> 30) & 1) << 5));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "subc:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 20) & 0xf)) & (0xc)) != (0xc)) {
        if (((((instruction >> 28) & 0xf)) & (0x2)) == (0x0)) {
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x7)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x1))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_sh.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_sh.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_sh.s:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_sh:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_sh:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_sh:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_sh:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_sh:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_sh:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x6)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x1))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_sl.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_sl.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_sl.s:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_sl:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_sl:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_sl:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_sl:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_sl:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_sl:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xb)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x1))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_uh.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_uh.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_uh.s:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_uh:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_uh:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_uh:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_uh:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_uh:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_uh:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xa)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x1))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_ul.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_ul.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_ul.s:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_ul:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_ul:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_ul:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sh_ul:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sh_ul:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sh_ul:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x5)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x1))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_sh.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_sh.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_sh.s:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_sh:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_sh:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_sh:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_sh:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_sh:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_sh:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x4)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x1))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_sl.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_sl.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_sl.s:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_sl:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_sl:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_sl:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_sl:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_sl:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_sl:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x9)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x1))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_uh.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_uh.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_uh.s:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_uh:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_uh:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_uh:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_uh:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_uh:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_uh:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x8)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x1))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_ul.s:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_ul.s:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_ul.s:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_ul:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_ul:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_ul:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_sl_ul:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_sl_ul:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_sl_ul:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x3)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x0))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_uh_uh.u:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_uh_uh.u:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_uh_uh.u:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_uh_uh:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_uh_uh:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_uh_uh:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_uh_uh:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_uh_uh:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_uh_uh:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x2)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x0))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_uh_ul.u:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_uh_ul.u:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_uh_ul.u:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_uh_ul:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_uh_ul:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_uh_ul:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_uh_ul:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_uh_ul:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_uh_ul:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x1)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x0))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_ul_uh.u:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_ul_uh.u:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_ul_uh.u:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_ul_uh:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_ul_uh:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_ul_uh:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_ul_uh:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_ul_uh:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_ul_uh:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x0)) {
            out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                          (((instruction >> 32) & 3) << 3));
            out.has_rb = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                 ((((instruction >> 46) & 0x1)) == (0x1))) &&
                ((((instruction >> 39) & 0x1)) == (0x0))) {
              out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_dc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_ul_ul.u:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_ul_ul.u:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_ul_ul.u:rrrci";
                return true;
              }
              return false;
            }
            if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                ((((instruction >> 46) & 0x1)) == (0x0))) {
              out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                            (((instruction >> 44) & 3) << 3));
              out.has_rc = true;
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_ul_ul:rrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_ul_ul:rrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_ul_ul:rrrci";
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                out.signature = "mul_ul_ul:zrr";
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                out.signature = "mul_ul_ul:zrrc";
                return true;
              }
              if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                  ((((instruction >> 24) & 0x1f)) != (0x0))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 31) << 0));
                out.has_cc = true;
                out.pc =
                    static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                out.has_pc = true;
                out.valid = true;
                out.signature = "mul_ul_ul:zrrci";
                return true;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 28) & 0xf)) & (0x3)) == (0x2)) {
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x8)) {
            if ((((instruction >> 34) & 0x1f)) == (0x18)) {
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                   ((((instruction >> 46) & 0x1)) == (0x1))) &&
                  ((((instruction >> 39) & 0x1)) == (0x1))) {
                out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                              (((instruction >> 44) & 3) << 3));
                out.has_dc = true;
                if ((((instruction >> 24) & 0xf)) == (0x0)) {
                  out.valid = true;
                  out.signature = "time.s:r";
                  return true;
                }
                if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                     ((((instruction >> 24) & 0xf)) > (0xb))) &&
                    ((((instruction >> 24) & 0xf)) != (0x0))) {
                  out.cc =
                      static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                  out.has_cc = true;
                  out.pc =
                      static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                  out.has_pc = true;
                  out.valid = true;
                  out.signature = "time.s:rci";
                  return true;
                }
                return false;
              }
              if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                   ((((instruction >> 46) & 0x1)) == (0x1))) &&
                  ((((instruction >> 39) & 0x1)) == (0x0))) {
                out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                              (((instruction >> 44) & 3) << 3));
                out.has_dc = true;
                if ((((instruction >> 24) & 0xf)) == (0x0)) {
                  out.valid = true;
                  out.signature = "time.u:r";
                  return true;
                }
                if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                     ((((instruction >> 24) & 0xf)) > (0xb))) &&
                    ((((instruction >> 24) & 0xf)) != (0x0))) {
                  out.cc =
                      static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                  out.has_cc = true;
                  out.pc =
                      static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                  out.has_pc = true;
                  out.valid = true;
                  out.signature = "time.u:rci";
                  return true;
                }
                return false;
              }
              if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                  ((((instruction >> 46) & 0x1)) == (0x0))) {
                out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                              (((instruction >> 44) & 3) << 3));
                out.has_rc = true;
                if ((((instruction >> 24) & 0xf)) == (0x0)) {
                  out.valid = true;
                  out.signature = "time:r";
                  return true;
                }
                if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                     ((((instruction >> 24) & 0xf)) > (0xb))) &&
                    ((((instruction >> 24) & 0xf)) != (0x0))) {
                  out.cc =
                      static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                  out.has_cc = true;
                  out.pc =
                      static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                  out.has_pc = true;
                  out.valid = true;
                  out.signature = "time:rci";
                  return true;
                }
                return false;
              }
              if ((((instruction >> 44) & 0x3)) == (0x3)) {
                if ((((instruction >> 24) & 0xf)) == (0x0)) {
                  out.valid = true;
                  out.signature = "time:z";
                  return true;
                }
                if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                     ((((instruction >> 24) & 0xf)) > (0xb))) &&
                    ((((instruction >> 24) & 0xf)) != (0x0))) {
                  out.cc =
                      static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                  out.has_cc = true;
                  out.pc =
                      static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                  out.has_pc = true;
                  out.valid = true;
                  out.signature = "time:zci";
                  return true;
                }
                return false;
              }
              return false;
            }
            if ((((instruction >> 34) & 0x1f)) == (0x19)) {
              out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                            (((instruction >> 32) & 3) << 3));
              out.has_rb = true;
              out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
              out.has_ra = true;
              if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                   ((((instruction >> 46) & 0x1)) == (0x1))) &&
                  ((((instruction >> 39) & 0x1)) == (0x1))) {
                out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                              (((instruction >> 44) & 3) << 3));
                out.has_dc = true;
                if ((((instruction >> 24) & 0xf)) == (0x0)) {
                  out.valid = true;
                  out.signature = "time_cfg.s:rr";
                  return true;
                }
                if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                     ((((instruction >> 24) & 0xf)) > (0xb))) &&
                    ((((instruction >> 24) & 0xf)) != (0x0))) {
                  out.cc =
                      static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                  out.has_cc = true;
                  out.pc =
                      static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                  out.has_pc = true;
                  out.valid = true;
                  out.signature = "time_cfg.s:rrci";
                  return true;
                }
                return false;
              }
              if ((((((instruction >> 44) & 0x3)) != (0x3)) &&
                   ((((instruction >> 46) & 0x1)) == (0x1))) &&
                  ((((instruction >> 39) & 0x1)) == (0x0))) {
                out.dc = static_cast<int32_t>((((instruction >> 40) & 3) << 1) |
                                              (((instruction >> 44) & 3) << 3));
                out.has_dc = true;
                if ((((instruction >> 24) & 0xf)) == (0x0)) {
                  out.valid = true;
                  out.signature = "time_cfg.u:rr";
                  return true;
                }
                if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                     ((((instruction >> 24) & 0xf)) > (0xb))) &&
                    ((((instruction >> 24) & 0xf)) != (0x0))) {
                  out.cc =
                      static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                  out.has_cc = true;
                  out.pc =
                      static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                  out.has_pc = true;
                  out.valid = true;
                  out.signature = "time_cfg.u:rrci";
                  return true;
                }
                return false;
              }
              if (((((instruction >> 44) & 0x3)) != (0x3)) &&
                  ((((instruction >> 46) & 0x1)) == (0x0))) {
                out.rc = static_cast<int32_t>((((instruction >> 39) & 7) << 0) |
                                              (((instruction >> 44) & 3) << 3));
                out.has_rc = true;
                if ((((instruction >> 24) & 0xf)) == (0x0)) {
                  out.valid = true;
                  out.signature = "time_cfg:rr";
                  return true;
                }
                if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                     ((((instruction >> 24) & 0xf)) > (0xb))) &&
                    ((((instruction >> 24) & 0xf)) != (0x0))) {
                  out.cc =
                      static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                  out.has_cc = true;
                  out.pc =
                      static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                  out.has_pc = true;
                  out.valid = true;
                  out.signature = "time_cfg:rrci";
                  return true;
                }
                return false;
              }
              if ((((instruction >> 44) & 0x3)) == (0x3)) {
                if ((((instruction >> 24) & 0xf)) == (0x0)) {
                  out.valid = true;
                  out.signature = "time_cfg:zr";
                  return true;
                }
                if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                     ((((instruction >> 24) & 0xf)) > (0xb))) &&
                    ((((instruction >> 24) & 0xf)) != (0x0))) {
                  out.cc =
                      static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                  out.has_cc = true;
                  out.pc =
                      static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
                  out.has_pc = true;
                  out.valid = true;
                  out.signature = "time_cfg:zrci";
                  return true;
                }
                return false;
              }
              return false;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      return false;
    }
    return false;
  }
  if (((((instruction >> 44) & 0xf)) == (0x6)) &&
      ((((instruction >> 42) & 0x3)) != (0x3))) {
    if (((((instruction >> 32) & 0x3)) != (0x3)) &&
        ((((instruction >> 16) & 0x1)) == (0x0))) {
      out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                    (((instruction >> 32) & 3) << 3));
      out.has_rb = true;
      if ((((instruction >> 37) & 0x3)) == (0x3)) {
        out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 34) & 7) << 4) |
                                       (((instruction >> 39) & 1) << 7) |
                                       (((instruction >> 15) & 1) << 8) |
                                       (((instruction >> 14) & 1) << 9) |
                                       (((instruction >> 13) & 1) << 10) |
                                       (((instruction >> 12) & 1) << 11) |
                                       (((instruction >> 0) & 4095) << 12) |
                                       (((instruction >> 24) & 255) << 24));
        out.has_imm = true;
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0x0)) {
          out.valid = true;
          out.signature = "add:zri";
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0x2)) {
          out.valid = true;
          out.signature = "addc:zri";
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0xa)) {
          out.valid = true;
          out.signature = "and:zri";
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0xc)) {
          out.valid = true;
          out.signature = "or:zri";
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0x4)) {
          out.valid = true;
          out.signature = "sub:zir";
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0x6)) {
          out.valid = true;
          out.signature = "subc:zir";
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0x8)) {
          out.valid = true;
          out.signature = "xor:zri";
          return true;
        }
        return false;
      }
      return false;
    }
    if ((((instruction >> 32) & 0x3)) == (0x3)) {
      out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
      out.has_ra = true;
      out.imm = static_cast<int32_t>(
          (((instruction >> 20) & 15) << 0) |
          (((instruction >> 16) & 15) << 4) | (((instruction >> 15) & 1) << 8) |
          (((instruction >> 14) & 1) << 9) | (((instruction >> 13) & 1) << 10) |
          (((instruction >> 12) & 1) << 11) |
          (((instruction >> 0) & 4095) << 12) |
          (((instruction >> 24) & 255) << 24));
      out.has_imm = true;
      if ((((instruction >> 42) & 0x3)) != (0x3)) {
        out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
        out.has_rc = true;
        out.valid = true;
        out.signature = "or:rri";
        return true;
      }
      return false;
    }
    return false;
  }
  if (((((instruction >> 45) & 0x7)) == (0x2)) &&
      ((((instruction >> 42) & 0x3)) != (0x3))) {
    if ((((instruction >> 32) & 0x3)) == (0x3)) {
      if (((((instruction >> 44) & 0xf)) == (0x5)) &&
          ((((instruction >> 42) & 0x3)) != (0x3))) {
        if ((((instruction >> 37) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 24) & 255) << 24));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and.s:rki";
            return true;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 24) & 255) << 24));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and.u:rki";
            return true;
          }
          return false;
        }
        if ((((instruction >> 37) & 0x3)) != (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 42) & 0x3)) != (0x3)) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 24) & 255) << 24));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and:rri";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 44) & 0xf)) == (0x4)) &&
          ((((instruction >> 42) & 0x3)) != (0x3))) {
        if ((((instruction >> 42) & 0x3)) != (0x3)) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 16) & 15) << 4) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "xor:rri";
          return true;
        }
        return false;
      }
      return false;
    }
    if (((((instruction >> 32) & 0x3)) != (0x3)) &&
        ((((instruction >> 16) & 0x1)) == (0x0))) {
      out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                    (((instruction >> 32) & 3) << 3));
      out.has_rb = true;
      if (((((instruction >> 42) & 0x3)) != (0x3)) &&
          ((((instruction >> 39) & 0x1)) == (0x1))) {
        out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
        out.has_dc = true;
        if ((((instruction >> 37) & 0x3)) == (0x3)) {
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 34) & 7) << 4) |
                                         (((instruction >> 44) & 1) << 7) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "or.s:rri";
          return true;
        }
        return false;
      }
      if (((((instruction >> 42) & 0x3)) != (0x3)) &&
          ((((instruction >> 39) & 0x1)) == (0x0))) {
        out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
        out.has_dc = true;
        if ((((instruction >> 37) & 0x3)) == (0x3)) {
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 34) & 7) << 4) |
                                         (((instruction >> 44) & 1) << 7) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "or.u:rri";
          return true;
        }
        return false;
      }
      return false;
    }
    return false;
  }
  if (((((instruction >> 45) & 0x7)) == (0x1)) &&
      ((((instruction >> 42) & 0x3)) != (0x3))) {
    if (((((instruction >> 32) & 0x3)) != (0x3)) &&
        ((((instruction >> 16) & 0x1)) == (0x0))) {
      out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                    (((instruction >> 32) & 3) << 3));
      out.has_rb = true;
      if (((((instruction >> 42) & 0x3)) != (0x3)) &&
          ((((instruction >> 39) & 0x1)) == (0x1))) {
        out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
        out.has_dc = true;
        if ((((instruction >> 37) & 0x3)) == (0x3)) {
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 34) & 7) << 4) |
                                         (((instruction >> 44) & 1) << 7) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "and.s:rri";
          return true;
        }
        return false;
      }
      if (((((instruction >> 42) & 0x3)) != (0x3)) &&
          ((((instruction >> 39) & 0x1)) == (0x0))) {
        out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
        out.has_dc = true;
        if ((((instruction >> 37) & 0x3)) == (0x3)) {
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 34) & 7) << 4) |
                                         (((instruction >> 44) & 1) << 7) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "and.u:rri";
          return true;
        }
        return false;
      }
      return false;
    }
    if ((((instruction >> 32) & 0x3)) == (0x3)) {
      if (((((instruction >> 44) & 0xf)) == (0x2)) &&
          ((((instruction >> 42) & 0x3)) != (0x3))) {
        if ((((instruction >> 42) & 0x3)) != (0x3)) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 16) & 15) << 4) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "sub:rir";
          return true;
        }
        return false;
      }
      if (((((instruction >> 44) & 0xf)) == (0x3)) &&
          ((((instruction >> 42) & 0x3)) != (0x3))) {
        if ((((instruction >> 42) & 0x3)) != (0x3)) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                         (((instruction >> 16) & 15) << 4) |
                                         (((instruction >> 15) & 1) << 8) |
                                         (((instruction >> 14) & 1) << 9) |
                                         (((instruction >> 13) & 1) << 10) |
                                         (((instruction >> 12) & 1) << 11) |
                                         (((instruction >> 0) & 4095) << 12) |
                                         (((instruction >> 24) & 255) << 24));
          out.has_imm = true;
          out.valid = true;
          out.signature = "subc:rir";
          return true;
        }
        return false;
      }
      return false;
    }
    return false;
  }
  if ((((instruction >> 45) & 0x7)) == (0x4)) {
    if ((((instruction >> 32) & 0x3)) == (0x3)) {
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0xa)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and.s:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and.s:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and.s:rrif";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and.u:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and.u:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and.u:rrif";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and:rrif";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and:zric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8) |
                                           (((instruction >> 44) & 1) << 11));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and:zrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "and:zrif";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0xc)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn.s:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn.s:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn.s:rrif";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn.u:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn.u:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn.u:rrif";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn:rrif";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn:zric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8) |
                                           (((instruction >> 44) & 1) << 11));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn:zrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "andn:zrif";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xc)) == (0x4)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 24) & 0xf)) == (0x0)) &&
              ((((instruction >> 29) & 0x1)) == (0x0))) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.valid = true;
              out.signature = "asr.s:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              out.signature = "lsl.s:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              out.signature = "lsl1.s:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              out.signature = "lsl1x.s:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              out.signature = "lslx.s:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              out.signature = "lsr.s:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              out.signature = "lsr1.s:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              out.signature = "lsr1x.s:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              out.signature = "lsrx.s:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              out.signature = "rol.s:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
              out.signature = "ror.s:rri";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.valid = true;
              out.signature = "asr.s:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              out.signature = "lsl.s:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              out.signature = "lsl1.s:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              out.signature = "lsl1x.s:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              out.signature = "lslx.s:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              out.signature = "lsr.s:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              out.signature = "lsr1.s:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              out.signature = "lsr1x.s:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              out.signature = "lsrx.s:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              out.signature = "rol.s:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
              out.signature = "ror.s:rric";
              return true;
            }
            return false;
          }
          if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                (((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
               (((((instruction >> 29) & 0x1)) == (0x1)) &&
                (((((instruction >> 24) & 0xf)) > (0xd)) ||
                 (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
              (((((instruction >> 29) & 0x1)) != (0x0)) ||
               ((((instruction >> 24) & 0xf)) != (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                          (((instruction >> 29) & 1) << 4));
            out.has_cc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "asr.s:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl.s:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1.s:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1x.s:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lslx.s:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr.s:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1.s:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1x.s:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsrx.s:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "rol.s:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "ror.s:rrici";
              return true;
            }
            return false;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 24) & 0xf)) == (0x0)) &&
              ((((instruction >> 29) & 0x1)) == (0x0))) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.valid = true;
              out.signature = "asr.u:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              out.signature = "lsl.u:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              out.signature = "lsl1.u:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              out.signature = "lsl1x.u:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              out.signature = "lslx.u:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              out.signature = "lsr.u:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              out.signature = "lsr1.u:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              out.signature = "lsr1x.u:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              out.signature = "lsrx.u:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              out.signature = "rol.u:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
              out.signature = "ror.u:rri";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.valid = true;
              out.signature = "asr.u:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              out.signature = "lsl.u:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              out.signature = "lsl1.u:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              out.signature = "lsl1x.u:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              out.signature = "lslx.u:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              out.signature = "lsr.u:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              out.signature = "lsr1.u:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              out.signature = "lsr1x.u:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              out.signature = "lsrx.u:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              out.signature = "rol.u:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
              out.signature = "ror.u:rric";
              return true;
            }
            return false;
          }
          if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                (((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
               (((((instruction >> 29) & 0x1)) == (0x1)) &&
                (((((instruction >> 24) & 0xf)) > (0xd)) ||
                 (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
              (((((instruction >> 29) & 0x1)) != (0x0)) ||
               ((((instruction >> 24) & 0xf)) != (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                          (((instruction >> 29) & 1) << 4));
            out.has_cc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "asr.u:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl.u:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1.u:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1x.u:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lslx.u:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr.u:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1.u:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1x.u:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsrx.u:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "rol.u:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "ror.u:rrici";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 24) & 0xf)) == (0x0)) &&
              ((((instruction >> 29) & 0x1)) == (0x0))) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.valid = true;
              out.signature = "asr:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              out.signature = "lsl1:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              out.signature = "lsl1x:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              out.signature = "lsl:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              out.signature = "lslx:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              out.signature = "lsr1:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              out.signature = "lsr1x:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              out.signature = "lsr:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              out.signature = "lsrx:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              out.signature = "rol:rri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
              out.signature = "ror:rri";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.valid = true;
              out.signature = "asr:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              out.signature = "lsl1:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              out.signature = "lsl1x:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              out.signature = "lsl:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              out.signature = "lslx:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              out.signature = "lsr1:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              out.signature = "lsr1x:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              out.signature = "lsr:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              out.signature = "lsrx:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              out.signature = "rol:rric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
              out.signature = "ror:rric";
              return true;
            }
            return false;
          }
          if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                (((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
               (((((instruction >> 29) & 0x1)) == (0x1)) &&
                (((((instruction >> 24) & 0xf)) > (0xd)) ||
                 (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
              (((((instruction >> 29) & 0x1)) != (0x0)) ||
               ((((instruction >> 24) & 0xf)) != (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                          (((instruction >> 29) & 1) << 4));
            out.has_cc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "asr:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1x:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lslx:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1x:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsrx:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "rol:rrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "ror:rrici";
              return true;
            }
            return false;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 24) & 0xf)) == (0x0)) &&
              ((((instruction >> 29) & 0x1)) == (0x0))) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.valid = true;
              out.signature = "asr:zri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              out.signature = "lsl1:zri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              out.signature = "lsl1x:zri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              out.signature = "lsl:zri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              out.signature = "lslx:zri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              out.signature = "lsr1:zri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              out.signature = "lsr1x:zri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              out.signature = "lsr:zri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              out.signature = "lsrx:zri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              out.signature = "rol:zri";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
              out.signature = "ror:zri";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.valid = true;
              out.signature = "asr:zric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              out.signature = "lsl1:zric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              out.signature = "lsl1x:zric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              out.signature = "lsl:zric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              out.signature = "lslx:zric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              out.signature = "lsr1:zric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              out.signature = "lsr1x:zric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              out.signature = "lsr:zric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              out.signature = "lsrx:zric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              out.signature = "rol:zric";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
              out.signature = "ror:zric";
              return true;
            }
            return false;
          }
          if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                (((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
               (((((instruction >> 29) & 0x1)) == (0x1)) &&
                (((((instruction >> 24) & 0xf)) > (0xd)) ||
                 (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
              (((((instruction >> 29) & 0x1)) != (0x0)) ||
               ((((instruction >> 24) & 0xf)) != (0x0)))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                          (((instruction >> 29) & 1) << 4));
            out.has_cc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x8)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "asr:zrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1:zrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1x:zrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl:zrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lslx:zrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1:zrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1x:zrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr:zrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsrx:zrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "rol:zrici";
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "ror:zrici";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0x0)) {
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.off = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_off = true;
            out.valid = true;
            out.signature = "call:rri";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.off = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_off = true;
            out.valid = true;
            out.signature = "call:zri";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0x3)) {
        out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
        out.has_ra = true;
        if (((((instruction >> 20) & 0xf)) & (0x7)) == (0x0)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "cao.s:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "cao.s:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "cao.s:rrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "cao.u:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "cao.u:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "cao.u:rrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "cao:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "cao:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "cao:rrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "cao:zr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "cao:zrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "cao:zrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0x7)) == (0x1)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "clo.s:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "clo.s:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "clo.s:rrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "clo.u:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "clo.u:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "clo.u:rrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "clo:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "clo:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "clo:rrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "clo:zr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "clo:zrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "clo:zrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0x7)) == (0x2)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "cls.s:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "cls.s:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "cls.s:rrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "cls.u:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "cls.u:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "cls.u:rrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "cls:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "cls:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "cls:rrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "cls:zr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "cls:zrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "cls:zrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0x7)) == (0x3)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "clz.s:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "clz.s:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "clz.s:rrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "clz.u:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "clz.u:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "clz.u:rrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "clz:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "clz:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "clz:rrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "clz:zr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "clz:zrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "clz:zrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0x7)) == (0x5)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extsb.s:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extsb.s:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extsb.s:rrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extsb:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extsb:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extsb:rrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extsb:zr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extsb:zrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extsb:zrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0x7)) == (0x7)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extsh.s:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extsh.s:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extsh.s:rrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extsh:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extsh:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extsh:rrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extsh:zr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extsh:zrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extsh:zrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0x7)) == (0x4)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extub.u:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extub.u:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extub.u:rrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extub:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extub:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extub:rrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extub:zr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extub:zrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extub:zrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0x7)) == (0x6)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extuh.u:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extuh.u:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extuh.u:rrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extuh:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extuh:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extuh:rrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "extuh:zr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "extuh:zrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "extuh:zrci";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0x1)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash.s:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash.s:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash.s:rrif";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash.u:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash.u:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash.u:rrif";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash:rrif";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash:zric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8) |
                                           (((instruction >> 44) & 1) << 11));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash:zrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "hash:zrif";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0xf)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand.s:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand.s:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand.s:rrif";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand.u:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand.u:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand.u:rrif";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand:rrif";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand:zric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8) |
                                           (((instruction >> 44) & 1) << 11));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand:zrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nand:zrif";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0xe)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor.s:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor.s:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor.s:rrif";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor.u:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor.u:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor.u:rrif";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor:rrif";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor:zric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8) |
                                           (((instruction >> 44) & 1) << 11));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor:zrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nor:zrif";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0x9)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor.s:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor.s:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor.s:rrif";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor.u:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor.u:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor.u:rrif";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor:rrif";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor:zric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8) |
                                           (((instruction >> 44) & 1) << 11));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor:zrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "nxor:zrif";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0xb)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or.s:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or.s:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or.s:rrif";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or.u:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or.u:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or.u:rrif";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or:rrif";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or:zric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8) |
                                           (((instruction >> 44) & 1) << 11));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or:zrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "or:zrif";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0xd)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn.s:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn.s:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn.s:rrif";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn.u:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn.u:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn.u:rrif";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn:rrif";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn:zric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8) |
                                           (((instruction >> 44) & 1) << 11));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn:zrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "orn:zrif";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0x2)) {
        out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
        out.has_ra = true;
        if (((((instruction >> 20) & 0xf)) & (0xc)) == (0x0)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "sats.s:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "sats.s:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "sats.s:rrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "sats.u:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "sats.u:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "sats.u:rrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "sats:rr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "sats:rrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "sats:rrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "sats:zr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "sats:zrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "sats:zrci";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0x8)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor.s:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor.s:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor.s:rrif";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor.u:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor.u:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor.u:rrif";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor:rric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor:rrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor:rrif";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor:zric";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 39) & 7) << 8) |
                                           (((instruction >> 44) & 1) << 11));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor:zrici";
            return true;
          }
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.imm = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                           (((instruction >> 16) & 15) << 4) |
                                           (((instruction >> 15) & 1) << 8) |
                                           (((instruction >> 14) & 1) << 9) |
                                           (((instruction >> 13) & 1) << 10) |
                                           (((instruction >> 12) & 1) << 11) |
                                           (((instruction >> 0) & 4095) << 12) |
                                           (((instruction >> 39) & 7) << 24) |
                                           (((instruction >> 44) & 1) << 27));
            out.has_imm = true;
            out.valid = true;
            out.signature = "xor:zrif";
            return true;
          }
          return false;
        }
        return false;
      }
      return false;
    }
    if (((((instruction >> 32) & 0x3)) != (0x3)) &&
        ((((instruction >> 16) & 0x1)) == (0x0))) {
      out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                    (((instruction >> 32) & 3) << 3));
      out.has_rb = true;
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0xc)) {
        if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xa)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "and.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "and.s:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "and.s:rrrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "and.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "and.u:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "and.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "and:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "and:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "and:rrrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "and:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "and:zrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "and:zrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xc)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "andn.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "andn.s:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "andn.s:rrrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "andn.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "andn.u:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "andn.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "andn:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "andn:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "andn:rrrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "andn:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "andn:zrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "andn:zrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x0)) {
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "call:rrr";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "call:zrr";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x1)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "hash.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "hash.s:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "hash.s:rrrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "hash.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "hash.u:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "hash.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "hash:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "hash:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "hash:rrrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "hash:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "hash:zrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "hash:zrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xf)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nand.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nand.s:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nand.s:rrrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nand.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nand.u:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nand.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nand:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nand:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nand:rrrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nand:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nand:zrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nand:zrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xe)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nor.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nor.s:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nor.s:rrrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nor.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nor.u:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nor.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nor:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nor:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nor:rrrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nor:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nor:zrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nor:zrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x9)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nxor.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nxor.s:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nxor.s:rrrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nxor.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nxor.u:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nxor.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nxor:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nxor:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nxor:rrrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "nxor:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "nxor:zrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "nxor:zrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xb)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "or.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "or.s:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "or.s:rrrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "or.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "or.u:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "or.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "or:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "or:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "or:rrrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "or:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "or:zrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "or:zrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xd)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "orn.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "orn.s:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "orn.s:rrrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "orn.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "orn.u:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "orn.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "orn:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "orn:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "orn:rrrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "orn:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "orn:zrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "orn:zrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x8)) {
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "xor.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "xor.s:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "xor.s:rrrci";
              return true;
            }
            return false;
          }
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x0))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "xor.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "xor.u:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "xor.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 42) & 0x3)) != (0x3)) &&
              ((((instruction >> 44) & 0x1)) == (0x0))) {
            out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
            out.has_rc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "xor:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "xor:rrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "xor:rrrci";
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              out.signature = "xor:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "xor:zrrc";
              return true;
            }
            if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
                 (((((instruction >> 24) & 0xf)) > (0xb)) ||
                  (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
                ((((instruction >> 24) & 0xf)) != (0x0))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "xor:zrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xd)) == (0xd)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x8)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "asr.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "asr.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "asr.s:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x4)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl.s:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x6)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl1.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl1.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1.s:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x7)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl1x.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl1x.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1x.s:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x5)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lslx.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lslx.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lslx.s:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xc)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr.s:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xe)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr1.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr1.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1.s:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xf)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr1x.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr1x.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1x.s:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xd)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsrx.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsrx.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsrx.s:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x2)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "rol.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "rol.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "rol.s:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xa)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "ror.s:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "ror.s:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "ror.s:rrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x8)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "asr.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "asr.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "asr.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x4)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x6)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl1.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl1.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x7)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl1x.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl1x.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1x.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x5)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lslx.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lslx.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lslx.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xc)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xe)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr1.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr1.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xf)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr1x.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr1x.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1x.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xd)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsrx.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsrx.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsrx.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x2)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "rol.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "rol.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "rol.u:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xa)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "ror.u:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "ror.u:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "ror.u:rrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x8)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "asr:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "asr:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "asr:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x6)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl1:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl1:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x7)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl1x:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl1x:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1x:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x4)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x5)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lslx:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lslx:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lslx:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xe)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr1:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr1:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xf)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr1x:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr1x:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1x:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xc)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xd)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsrx:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsrx:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsrx:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x2)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "rol:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "rol:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "rol:rrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xa)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "ror:rrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "ror:rrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "ror:rrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x8)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "asr:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "asr:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "asr:zrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x6)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl1:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl1:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1:zrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x7)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl1x:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl1x:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl1x:zrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x4)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsl:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsl:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsl:zrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x5)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lslx:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lslx:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lslx:zrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xe)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr1:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr1:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1:zrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xf)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr1x:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr1x:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr1x:zrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xc)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsr:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsr:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsr:zrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xd)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "lsrx:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "lsrx:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "lsrx:zrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x2)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "rol:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "rol:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "rol:zrrci";
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xa)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              out.signature = "ror:zrr";
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              out.signature = "ror:zrrc";
              return true;
            }
            if (((((((instruction >> 29) & 0x1)) == (0x0)) &&
                  (((((instruction >> 24) & 0xf)) < (0x6)) ||
                   (((((instruction >> 24) & 0xf)) > (0xb)) ||
                    (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) ||
                 (((((instruction >> 29) & 0x1)) == (0x1)) &&
                  (((((instruction >> 24) & 0xf)) > (0xb)) ||
                   (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8))))) &&
                (((((instruction >> 29) & 0x1)) != (0x0)) ||
                 ((((instruction >> 24) & 0xf)) != (0x0)))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0) |
                                            (((instruction >> 29) & 1) << 4));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "ror:zrrci";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xc)) == (0x8)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.valid = true;
            out.signature = "cmpb4.s:rrr";
            return true;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.valid = true;
            out.signature = "cmpb4.s:rrrc";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.valid = true;
            out.signature = "cmpb4.s:rrrci";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.valid = true;
            out.signature = "cmpb4.u:rrr";
            return true;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.valid = true;
            out.signature = "cmpb4.u:rrrc";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.valid = true;
            out.signature = "cmpb4.u:rrrci";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.valid = true;
            out.signature = "cmpb4:rrr";
            return true;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.valid = true;
            out.signature = "cmpb4:rrrc";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.valid = true;
            out.signature = "cmpb4:rrrci";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.valid = true;
            out.signature = "cmpb4:zrr";
            return true;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.valid = true;
            out.signature = "cmpb4:zrrc";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x6)) ||
               (((((instruction >> 24) & 0xf)) > (0xb)) ||
                (((((instruction >> 24) & 0xf)) & (0xe)) == (0x8)))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.valid = true;
            out.signature = "cmpb4:zrrci";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_add.s:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_add.s:rrrici";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_add.u:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_add.u:rrrici";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_add:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_add:rrrici";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_add:zrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_add:zrrici";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_sub.s:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_sub.s:rrrici";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_sub.u:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_sub.u:rrrici";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_sub:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_sub:rrrici";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_sub:zrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsl_sub:zrrici";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsr_add.s:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsr_add.s:rrrici";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsr_add.u:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsr_add.u:rrrici";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsr_add:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsr_add:rrrici";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsr_add:zrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "lsr_add:zrrici";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x0)) {
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x1))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "rol_add.s:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "rol_add.s:rrrici";
            return true;
          }
          return false;
        }
        if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
             ((((instruction >> 44) & 0x1)) == (0x1))) &&
            ((((instruction >> 39) & 0x1)) == (0x0))) {
          out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
          out.has_dc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "rol_add.u:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "rol_add.u:rrrici";
            return true;
          }
          return false;
        }
        if (((((instruction >> 42) & 0x3)) != (0x3)) &&
            ((((instruction >> 44) & 0x1)) == (0x0))) {
          out.rc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_rc = true;
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "rol_add:rrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "rol_add:rrrici";
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "rol_add:zrri";
            return true;
          }
          if ((((((instruction >> 24) & 0xf)) < (0x2)) ||
               ((((instruction >> 24) & 0xf)) > (0xb))) &&
              ((((instruction >> 24) & 0xf)) != (0x0))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "rol_add:zrrici";
            return true;
          }
          return false;
        }
        return false;
      }
      return false;
    }
    if (((((instruction >> 32) & 0x3)) != (0x3)) &&
        ((((instruction >> 16) & 0x1)) == (0x1))) {
      out.db = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                    (((instruction >> 32) & 3) << 3));
      out.has_db = true;
      if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
        if ((((instruction >> 42) & 0x3)) != (0x3)) {
          out.dc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_dc = true;
          if (((((instruction >> 24) & 0xf)) < (0x2)) ||
              ((((instruction >> 24) & 0xf)) > (0xb))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            out.pc = static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
            out.has_pc = true;
            out.shift = static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                             (((instruction >> 28) & 1) << 4));
            out.has_shift = true;
            out.valid = true;
            out.signature = "div_step:rrrici";
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0x2)) {
        if ((((instruction >> 42) & 0x3)) != (0x3)) {
          out.dc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_dc = true;
          if ((((instruction >> 34) & 0x1f)) == (0x18)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if (((((instruction >> 24) & 0xf)) < (0x6)) ||
                ((((instruction >> 24) & 0xf)) > (0xb))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "movd:rrci";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
        if ((((instruction >> 42) & 0x3)) != (0x3)) {
          out.dc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_dc = true;
          if ((((instruction >> 34) & 0x1f)) != (0x18)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if (((((instruction >> 24) & 0xf)) < (0x6)) ||
                ((((instruction >> 24) & 0xf)) > (0xb))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.shift =
                  static_cast<int32_t>((((instruction >> 20) & 15) << 0) |
                                       (((instruction >> 28) & 1) << 4));
              out.has_shift = true;
              out.valid = true;
              out.signature = "mul_step:rrrici";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xf)) == (0xa)) {
        if ((((instruction >> 42) & 0x3)) != (0x3)) {
          out.dc = static_cast<int32_t>((((instruction >> 39) & 31) << 0));
          out.has_dc = true;
          if ((((instruction >> 34) & 0x1f)) == (0x18)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if (((((instruction >> 24) & 0xf)) < (0x6)) ||
                ((((instruction >> 24) & 0xf)) > (0xb))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              out.signature = "swapd:rrci";
              return true;
            }
            return false;
          }
          return false;
        }
        return false;
      }
      return false;
    }
    return false;
  }
  return false;
}

/* ---- merged from decode.cc ---- */

static std::unordered_map<const DpuState *,
                          std::shared_ptr<const DecodedProgram48CacheEntry>> &
program_cache_48() {
  static std::unordered_map<const DpuState *,
                            std::shared_ptr<const DecodedProgram48CacheEntry>>
      cache;
  return cache;
}

static std::mutex &program_cache_48_mutex() {
  static std::mutex cache_mutex;
  return cache_mutex;
}

void invalidate_decoded_program_cache_48(const DpuState *dpu) {
  if (!dpu) {
    return;
  }
  std::lock_guard<std::mutex> lock(program_cache_48_mutex());
  program_cache_48().erase(dpu);
}

static inline uint64_t hash_iram_words(const std::vector<uint64_t> &words) {
  uint64_t h = 1469598103934665603ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;

  for (uint64_t w : words) {
    for (size_t i = 0; i < 8; ++i) {
      const uint8_t byte = static_cast<uint8_t>((w >> (i * 8)) & 0xFFu);
      h ^= static_cast<uint64_t>(byte);
      h *= kPrime;
    }
  }

  return h;
}

static inline const char *decode_cache_dir_48() {
  return "/tmp/hostpimsim-upmem-decode-cache";
}

static inline void ensure_decode_cache_dir_48() {
  static std::once_flag initialized;
  std::call_once(initialized,
                 [] { (void)::mkdir(decode_cache_dir_48(), 0755); });
}

static inline std::string decode_cache_path_48(uint64_t key) {
  char path[128];
  std::snprintf(path, sizeof(path), "%s/%016llx.txt", decode_cache_dir_48(),
                static_cast<unsigned long long>(key));
  return std::string(path);
}

static inline bool set_decoded_field(DecodedInst48 &inst,
                                     const std::string &key,
                                     const std::string &value) {
  if (key.empty()) {
    return false;
  }

  if (key == "cc_name") {
    inst.cc_name = value;
    return true;
  }

  char *end = nullptr;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (!end || *end != '\0') {
    return false;
  }

  const int v = static_cast<int>(parsed);
  if (key == "ra") {
    inst.has_ra = true;
    inst.ra = v;
  } else if (key == "rb") {
    inst.has_rb = true;
    inst.rb = v;
  } else if (key == "rc") {
    inst.has_rc = true;
    inst.rc = v;
  } else if (key == "dc") {
    inst.has_dc = true;
    inst.dc = v;
  } else if (key == "db") {
    inst.has_db = true;
    inst.db = v;
  } else if (key == "imm") {
    inst.imm = v;
  } else if (key == "off") {
    inst.off = v;
  } else if (key == "pc") {
    inst.pc = v;
  } else if (key == "shift") {
    inst.shift = v;
  } else if (key == "immDma") {
    inst.immDma = v;
  } else if (key == "endian") {
    inst.endian = v;
  } else {
    return false;
  }

  return true;
}

static inline uint32_t replay_reg_bit(int reg_num) {
  if (reg_num < 0 || reg_num >= 24) {
    return 0u;
  }
  return (1u << static_cast<uint32_t>(reg_num));
}

static inline bool sig_has_prefix(const std::string &sig, const char *prefix) {
  const size_t n = std::strlen(prefix);
  return sig.size() >= n && sig.compare(0, n, prefix) == 0;
}

static inline bool sig_has_suffix(const std::string &sig, const char *suffix) {
  const size_t n = std::strlen(suffix);
  return sig.size() >= n && sig.compare(sig.size() - n, n, suffix) == 0;
}

static inline int32_t sign_extend_24(uint32_t value) {
  value &= 0x00FFFFFFu;
  if ((value & 0x00800000u) != 0u) {
    value |= 0xFF000000u;
  }
  return static_cast<int32_t>(value);
}

static inline bool signature_has_external_off24(const std::string &sig) {
  return sig_has_suffix(sig, ":erri") || sig_has_suffix(sig, ":erir") ||
         sig_has_suffix(sig, ":erii") || sig_has_suffix(sig, ":edri") ||
         sig_has_suffix(sig, ":ersi") || sig_has_suffix(sig, ":esir") ||
         sig_has_suffix(sig, ":esii");
}

static inline void normalize_decoded_offset_48(DecodedInst48 &inst) {
  if (!signature_has_external_off24(inst.signature)) {
    return;
  }
  inst.off = static_cast<int>(
      sign_extend_24(static_cast<uint32_t>(static_cast<int32_t>(inst.off))));
}

static inline bool signature_writes_rc_48(const std::string &sig) {
  if (sig.empty()) {
    return false;
  }

  // Branch/control and DMA/synchronization signatures do not produce a
  // destination GPR write in our execute path.
  if (sig_has_prefix(sig, "j") || sig_has_prefix(sig, "jump:") ||
      sig_has_prefix(sig, "call:") || sig_has_prefix(sig, "boot:") ||
      sig_has_prefix(sig, "resume:") || sig_has_prefix(sig, "stop:") ||
      sig_has_prefix(sig, "ldma:") || sig_has_prefix(sig, "ldmai:") ||
      sig_has_prefix(sig, "sdma:") || sig_has_prefix(sig, "acquire:") ||
      sig_has_prefix(sig, "release:") || sig == "nop:" || sig == "fault:i") {
    return false;
  }

  return true;
}

static inline void infer_replay_masks_48(DecodedInst48 &inst) {
  uint32_t read_mask = 0u;
  uint32_t write_mask = 0u;

  if (inst.has_ra) {
    read_mask |= replay_reg_bit(inst.ra);
  }
  if (inst.has_rb) {
    read_mask |= replay_reg_bit(inst.rb);
  }
  if (inst.has_db) {
    read_mask |= replay_reg_bit(inst.db);
    read_mask |= replay_reg_bit(inst.db + 1);
  }

  if (inst.has_rc && signature_writes_rc_48(inst.signature)) {
    write_mask |= replay_reg_bit(inst.rc);
  }
  if (inst.has_dc) {
    write_mask |= replay_reg_bit(inst.dc);
    write_mask |= replay_reg_bit(inst.dc + 1);
  }

  inst.replay_read_mask = read_mask;
  inst.replay_write_mask = write_mask;
}

static inline uint8_t replay_normalize_reg_index(int idx) {
  if (idx < 0 || idx >= static_cast<int>(kNumWorkRegistersPerThread)) {
    return 0xFFu;
  }
  return static_cast<uint8_t>(idx);
}

static inline uint8_t replay_parity_class(uint8_t reg_idx) {
  const uint8_t r9 = static_cast<uint8_t>(reg_idx >> 7);
  const uint8_t ecx = static_cast<uint8_t>(((r9 + reg_idx) & 0x1u) - r9);
  return static_cast<uint8_t>(ecx & 0x1u);
}

static inline bool replay_result_provoke_again_rr(uint8_t read1, uint8_t read2,
                                                  uint8_t source) {
  if (read1 == 0xFFu) {
    return false;
  }

  if (source != read2) {
    const uint8_t max_src = std::max(source, read2);
    if (!(kNumWorkRegistersPerThread > max_src)) {
      return false;
    }
  }

  if ((source & 0x1u) != (read2 & 0x1u)) {
    return false;
  }
  if (replay_parity_class(read1) != (read2 & 0x1u)) {
    return false;
  }

  if (read1 == read2) {
    return false;
  }
  if (source == read1) {
    return false;
  }

  return source != read2;
}

static inline bool replay_result_provoke_again_rd(uint8_t read1, uint8_t read2,
                                                  uint8_t read3,
                                                  uint8_t source) {
  if (read1 == 0xFFu || read2 == 0xFFu || read3 == 0xFFu) {
    return false;
  }
  if (source == 0xFFu) {
    return false;
  }

  const uint8_t max_read = std::max(read1, std::max(read2, read3));
  if (!(kNumWorkRegistersPerThread > max_read)) {
    return false;
  }

  if (read1 == read2 || read1 == read3) {
    return false;
  }

  if (replay_parity_class(source) != (read1 & 0x1u)) {
    return false;
  }

  if (source == read1 || source == read2 || source == read3) {
    return false;
  }

  return true;
}

static inline bool signature_uses_replay_rd_48(const std::string &sig) {
  // Call-site verified in libdpufsim.so.2025.1:
  // _handler_div_step_rrrici, _handler_mul_step_rrrici,
  // _handler_sd_erir, _handler_sd_esir -> _must_replay_rd.
  return sig == "div_step:rrri" || sig == "mul_step:rrrici" || sig == "sd:rir";
}

bool replay_strict_sig_allowed_48(const std::string &sig) {
  switch (replay_strict_scope()) {
  case ReplayStrictScope::kAll:
    return true;
  case ReplayStrictScope::kRdOnly: {
    const uint32_t mask = replay_strict_rd_sig_mask();
    if (sig == "div_step:rrri") {
      return (mask & kReplayStrictRdDivStep) != 0u;
    }
    if (sig == "mul_step:rrrici") {
      return (mask & kReplayStrictRdMulStep) != 0u;
    }
    if (sig == "sd:rir") {
      return (mask & kReplayStrictRdSd) != 0u;
    }
    return false;
  }
  }
  return true;
}

bool replay_should_fire_from_sources_48(const DecodedInst48 &ins,
                                        const dpu_regfile &rf) {
  if (!ins.has_ra && !ins.has_rb) {
    return false;
  }

  const uint8_t src0 = rf.replay_src0;
  const uint8_t src1 = rf.replay_src1;

  if (ins.has_db && signature_uses_replay_rd_48(ins.signature)) {
    const uint8_t read1 = ins.has_ra ? replay_normalize_reg_index(ins.ra)
                                     : replay_normalize_reg_index(ins.rb);
    const uint8_t read2 = replay_normalize_reg_index(ins.db);
    const uint8_t read3 = replay_normalize_reg_index(ins.db + 1);

    if (replay_result_provoke_again_rd(read1, read2, read3, src0)) {
      return true;
    }
    if (src0 != src1 &&
        replay_result_provoke_again_rd(read1, read2, read3, src1)) {
      return true;
    }
    return false;
  }

  const uint8_t read1 = ins.has_ra ? replay_normalize_reg_index(ins.ra) : 0xFFu;
  const uint8_t read2 = ins.has_rb ? replay_normalize_reg_index(ins.rb) : 0xFFu;

  if (replay_result_provoke_again_rr(read1, read2, src0)) {
    return true;
  }
  if (src0 != src1 && replay_result_provoke_again_rr(read1, read2, src1)) {
    return true;
  }
  return false;
}

void replay_update_sources_48(const DecodedInst48 &ins, dpu_regfile &rf) {
  if (ins.has_dc) {
    rf.replay_src0 = replay_normalize_reg_index(ins.dc);
    rf.replay_src1 = replay_normalize_reg_index(ins.dc + 1);
    return;
  }

  if (ins.has_rc && signature_writes_rc_48(ins.signature)) {
    const uint8_t r = replay_normalize_reg_index(ins.rc);
    rf.replay_src0 = r;
    rf.replay_src1 = r;
    return;
  }

  rf.replay_src0 = 0xFFu;
  rf.replay_src1 = 0xFFu;
}

static inline bool parse_decoded_line_48(const std::string &line,
                                         std::vector<DecodedInst48> &out) {
  if (line.empty()) {
    return true;
  }

  const size_t p1 = line.find('|');
  const size_t p2 =
      (p1 == std::string::npos) ? std::string::npos : line.find('|', p1 + 1);
  if (p1 == std::string::npos || p2 == std::string::npos) {
    return false;
  }

  const std::string idx_str = line.substr(0, p1);
  char *end = nullptr;
  const long idx_long = std::strtol(idx_str.c_str(), &end, 10);
  if (!end || *end != '\0' || idx_long < 0) {
    return false;
  }

  const size_t idx = static_cast<size_t>(idx_long);
  if (idx >= out.size()) {
    return false;
  }

  DecodedInst48 inst;
  const std::string sig = line.substr(p1 + 1, p2 - p1 - 1);
  if (sig == "?" || sig.empty()) {
    out[idx] = inst;
    return true;
  }

  inst.valid = true;
  inst.signature = sig;
  inst.fast_op = fast_op_from_signature(sig);
  inst.perf_is_dma = DMAEngine::is_signature(sig);

  const std::string fields = line.substr(p2 + 1);
  size_t pos = 0;
  while (pos < fields.size()) {
    size_t sep = fields.find(';', pos);
    if (sep == std::string::npos) {
      sep = fields.size();
    }

    const std::string kv = fields.substr(pos, sep - pos);
    const size_t eq = kv.find('=');
    if (eq != std::string::npos && eq > 0 && eq + 1 < kv.size()) {
      const std::string key_name = kv.substr(0, eq);
      const std::string value = kv.substr(eq + 1);
      (void)set_decoded_field(inst, key_name, value);
    }

    pos = sep + 1;
  }

  normalize_decoded_offset_48(inst);
  infer_replay_masks_48(inst);
  out[idx] = inst;
  return true;
}

static inline void populate_decoded_from_raw_48(const RawDecoded48 &raw,
                                                DecodedInst48 &inst) {
  inst = DecodedInst48{};
  if (!raw.valid || !raw.signature) {
    return;
  }

  inst.valid = true;
  inst.signature = raw.signature;
  inst.fast_op = fast_op_from_signature(inst.signature);
  inst.perf_is_dma = DMAEngine::is_signature(inst.signature);

  if (raw.has_ra) {
    inst.has_ra = true;
    inst.ra = static_cast<int>(raw.ra);
  }
  if (raw.has_rb) {
    inst.has_rb = true;
    inst.rb = static_cast<int>(raw.rb);
  }
  if (raw.has_rc) {
    inst.has_rc = true;
    inst.rc = static_cast<int>(raw.rc);
  }
  if (raw.has_dc) {
    inst.has_dc = true;
    inst.dc = static_cast<int>(raw.dc);
  }
  if (raw.has_db) {
    inst.has_db = true;
    inst.db = static_cast<int>(raw.db);
  }
  if (raw.has_imm) {
    inst.imm = static_cast<int>(raw.imm);
  }
  if (raw.has_off) {
    inst.off = static_cast<int>(raw.off);
  }
  if (raw.has_pc) {
    inst.pc = static_cast<int>(raw.pc);
  }
  if (raw.has_shift) {
    inst.shift = static_cast<int>(raw.shift);
  }
  if (raw.has_immDma) {
    inst.immDma = static_cast<int>(raw.immDma);
  }
  if (raw.has_endian) {
    inst.endian = static_cast<int>(raw.endian);
  }
  if (raw.has_cc) {
    inst.cc_name = decode_cc_name_for_signature_48(inst.signature,
                                                   static_cast<int>(raw.cc));
  }

  normalize_decoded_offset_48(inst);
  infer_replay_masks_48(inst);
}

static inline std::string serialize_decoded_line_48(size_t idx,
                                                    const DecodedInst48 &inst) {
  std::ostringstream out;
  out << idx << "|";

  if (!inst.valid || inst.signature.empty()) {
    out << "?|";
    return out.str();
  }

  out << inst.signature << "|";

  bool first = true;
  const auto append_kv = [&](const char *key, const std::string &value,
                             bool enabled, bool *first_field,
                             std::ostringstream *stream) {
    if (!enabled) {
      return;
    }
    if (!*first_field) {
      *stream << ';';
    }
    *first_field = false;
    *stream << key << '=' << value;
  };

  append_kv("cc_name", inst.cc_name, !inst.cc_name.empty(), &first, &out);
  append_kv("ra", std::to_string(inst.ra), inst.has_ra, &first, &out);
  append_kv("rb", std::to_string(inst.rb), inst.has_rb, &first, &out);
  append_kv("rc", std::to_string(inst.rc), inst.has_rc, &first, &out);
  append_kv("dc", std::to_string(inst.dc), inst.has_dc, &first, &out);
  append_kv("db", std::to_string(inst.db), inst.has_db, &first, &out);
  append_kv("imm", std::to_string(inst.imm), true, &first, &out);
  append_kv("off", std::to_string(inst.off), true, &first, &out);
  append_kv("pc", std::to_string(inst.pc), true, &first, &out);
  append_kv("shift", std::to_string(inst.shift), true, &first, &out);
  append_kv("immDma", std::to_string(inst.immDma), true, &first, &out);
  append_kv("endian", std::to_string(inst.endian), true, &first, &out);

  return out.str();
}

static inline bool decode_iram_program_48(const uint8_t *iram_bytes,
                                          std::vector<DecodedInst48> &decoded,
                                          std::vector<uint64_t> &words_out) {
  decoded.clear();
  words_out.clear();

  if (!iram_bytes) {
    return false;
  }

  const size_t max_slots = IRAM_SIZE / kIramSlotBytes;
  words_out.resize(max_slots, 0);

  size_t last_non_zero = 0;
  bool has_non_zero = false;
  for (size_t i = 0; i < max_slots; ++i) {
    uint64_t w = 0;
    const size_t base = i * kIramSlotBytes;
    for (size_t bidx = 0; bidx < kIramWriteWordBytes; ++bidx) {
      w |= static_cast<uint64_t>(iram_bytes[base + bidx]) << (8 * bidx);
    }
    words_out[i] = w;
    if (w != 0) {
      last_non_zero = i;
      has_non_zero = true;
    }
  }

  if (!has_non_zero) {
    return false;
  }

  words_out.resize(last_non_zero + 1);

  const uint64_t key = hash_iram_words(words_out);
  static std::unordered_map<uint64_t, std::vector<DecodedInst48>> cache;
  static std::mutex cache_mutex;

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      decoded = it->second;
      return true;
    }
  }

  ensure_decode_cache_dir_48();
  const std::string cache_path = decode_cache_path_48(key);

  {
    FILE *cache_fp = std::fopen(cache_path.c_str(), "r");
    if (cache_fp) {
      std::vector<DecodedInst48> out(words_out.size());
      char line[4096];
      bool ok_file = true;
      while (std::fgets(line, sizeof(line), cache_fp)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
          s.pop_back();
        }
        if (!parse_decoded_line_48(s, out)) {
          ok_file = false;
          break;
        }
      }
      std::fclose(cache_fp);

      if (ok_file) {
        decoded = std::move(out);
        {
          std::lock_guard<std::mutex> lock(cache_mutex);
          cache.emplace(key, decoded);
        }
        return true;
      }
    }
  }

  std::vector<DecodedInst48> out(words_out.size());
  for (size_t i = 0; i < words_out.size(); ++i) {
    RawDecoded48 raw;
    (void)decode_raw_word_48(words_out[i], raw);
    populate_decoded_from_raw_48(raw, out[i]);
  }

  if (FILE *cache_fp = std::fopen(cache_path.c_str(), "w")) {
    for (size_t i = 0; i < out.size(); ++i) {
      const std::string line = serialize_decoded_line_48(i, out[i]);
      std::fputs(line.c_str(), cache_fp);
      std::fputc('\n', cache_fp);
    }
    std::fclose(cache_fp);
  }

  decoded = std::move(out);
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache.emplace(key, decoded);
  }
  return true;
}

std::shared_ptr<const DecodedProgram48CacheEntry>
get_decoded_program_cache_48(DpuState &dpu) {
  {
    std::lock_guard<std::mutex> lock(program_cache_48_mutex());
    auto &cache = program_cache_48();
    const auto it = cache.find(&dpu);
    if (it != cache.end()) {
      return it->second;
    }
  }

  auto entry = std::make_shared<DecodedProgram48CacheEntry>();
  std::vector<uint64_t> words;
  if (!decode_iram_program_48(dpu.private_mem.data() + IRAM_OFFSET,
                              entry->decoded, words)) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(program_cache_48_mutex());
  auto &cache = program_cache_48();
  const auto it = cache.find(&dpu);
  if (it != cache.end()) {
    return it->second;
  }
  cache.emplace(&dpu, entry);
  return entry;
}

/* ---- merged from execute.cc ---- */

static inline bool try_execute_fast_op_48(DpuState &dpu, dpu_state &state,
                                          int t, dpu_regfile &rf,
                                          const DecodedInst48 &ins, uint32_t ra,
                                          uint32_t rb, uint32_t imm_u32,
                                          uint32_t &next_pc,
                                          bool &keep_running) {
  switch (ins.fast_op) {
  case FAST_OP_MOVE_RI_RR: {
    const uint32_t x = ra | imm_u32;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_MOVE_RICI_RRCI: {
    const uint32_t x = ra | imm_u32;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    if (eval_cc_name(ins.cc_name, rf, x)) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_ADD_RRI: {
    const uint64_t sum =
        static_cast<uint64_t>(ra) + static_cast<uint64_t>(imm_u32);
    const uint32_t x = static_cast<uint32_t>(sum);
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    rf.CF = (sum >> 32) != 0;
    return true;
  }
  case FAST_OP_ADD_RRR: {
    const uint64_t sum = static_cast<uint64_t>(ra) + static_cast<uint64_t>(rb);
    const uint32_t x = static_cast<uint32_t>(sum);
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    rf.CF = (sum >> 32) != 0;
    return true;
  }
  case FAST_OP_SUB_RRR: {
    const uint32_t x = static_cast<uint32_t>(ra - rb);
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    rf.CF = (ra < rb);
    return true;
  }
  case FAST_OP_SUB_RRRC: {
    const uint32_t x = static_cast<uint32_t>(ra - rb);
    rf.ZF = (x == 0);
    rf.CF = (ra < rb);

    bool cond = false;
    if (ins.cc_name == "ltu" || ins.cc_name == "c") {
      cond = rf.CF;
    } else if (ins.cc_name == "geu" || ins.cc_name == "nc") {
      cond = !rf.CF;
    } else if (ins.cc_name == "z") {
      cond = (x == 0);
    } else if (ins.cc_name == "nz") {
      cond = (x != 0);
    } else {
      cond = eval_cc_name(ins.cc_name, rf, x);
    }

    rf.write_reg(static_cast<uint8_t>(ins.rc), cond ? 1u : 0u);
    return true;
  }
  case FAST_OP_NEG_RR: {
    const uint32_t x = static_cast<uint32_t>(imm_u32 - ra);
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    rf.CF = (imm_u32 < ra);
    return true;
  }
  case FAST_OP_AND_RRI: {
    const uint32_t x = ra & imm_u32;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_AND_RRR: {
    const uint32_t x = ra & rb;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_ADDC_RRR: {
    const uint64_t sum = static_cast<uint64_t>(ra) + static_cast<uint64_t>(rb) +
                         static_cast<uint64_t>(rf.CF ? 1u : 0u);
    const uint32_t x = static_cast<uint32_t>(sum);
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    rf.CF = (sum >> 32) != 0;
    return true;
  }
  case FAST_OP_MOVE_U_RR: {
    const uint32_t x = ra | imm_u32;
    rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_MOVE_S_RI_RR: {
    int32_t x = 0;
    if (ins.signature == "move.s:ri") {
      x = static_cast<int32_t>(ra & imm_u32);
    } else {
      x = static_cast<int32_t>(ra | imm_u32);
    }
    rf.write_dreg(dreg_slot(ins.dc),
                  static_cast<uint64_t>(static_cast<int64_t>(x)));
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_CLZ_RR: {
    const uint32_t x =
        (ra == 0) ? 32u : static_cast<uint32_t>(__builtin_clz(ra));
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_CLZ_RRCI: {
    const uint32_t x =
        (ra == 0) ? 32u : static_cast<uint32_t>(__builtin_clz(ra));
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    if (eval_cc_name(ins.cc_name, rf, x)) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_XOR_RRR: {
    const uint32_t x = ra ^ rb;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_OR_RRR: {
    const uint32_t x = ra | rb;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_OR_RRIF: {
    const uint32_t x = ra | imm_u32;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_JEQ_RRI: {
    const uint32_t x = static_cast<uint32_t>(ra - rb);
    rf.ZF = (x == 0);
    rf.CF = (ra < rb);
    if (ra == rb) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_JEQ_RII: {
    const uint32_t x = static_cast<uint32_t>(ra - imm_u32);
    rf.ZF = (x == 0);
    rf.CF = (ra < imm_u32);
    if (ra == imm_u32) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_JNEQ_RRI: {
    const uint32_t x = static_cast<uint32_t>(ra - rb);
    rf.ZF = (x == 0);
    rf.CF = (ra < rb);
    if (ra != rb) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_JNEQ_RII: {
    const uint32_t x = static_cast<uint32_t>(ra - imm_u32);
    rf.ZF = (x == 0);
    rf.CF = (ra < imm_u32);
    if (ra != imm_u32) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_JZ_RI: {
    const uint32_t x = static_cast<uint32_t>(ra - imm_u32);
    rf.ZF = (x == 0);
    rf.CF = (ra < imm_u32);
    if (x == 0) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_JGTS_RRI: {
    const int32_t lhs = static_cast<int32_t>(ra);
    const int32_t rhs = static_cast<int32_t>(rb);
    const uint32_t x = static_cast<uint32_t>(lhs - rhs);
    rf.ZF = (x == 0);
    if (lhs > rhs) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_JLTU_RRI: {
    const uint32_t x = static_cast<uint32_t>(ra - rb);
    rf.ZF = (x == 0);
    rf.CF = (ra < rb);
    if (ra < rb) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_JGTU_RRI: {
    const uint32_t x = static_cast<uint32_t>(ra - rb);
    rf.ZF = (x == 0);
    rf.CF = (ra < rb);
    if (ra > rb) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_JGEU_RRI: {
    const uint32_t x = static_cast<uint32_t>(ra - rb);
    rf.ZF = (x == 0);
    rf.CF = (ra < rb);
    if (ra >= rb) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_JLEU_RRI: {
    const uint32_t x = static_cast<uint32_t>(ra - rb);
    rf.ZF = (x == 0);
    rf.CF = (ra < rb);
    if (ra <= rb) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_MUL_STEP_RRRICI: {
    const uint32_t ra_val = ra;
    const uint8_t db_even_reg = static_cast<uint8_t>(ins.db);
    const uint8_t db_odd_reg = static_cast<uint8_t>(ins.db + 1);
    const uint8_t dc_even_reg = static_cast<uint8_t>(ins.dc);
    const uint8_t dc_odd_reg = static_cast<uint8_t>(ins.dc + 1);

    const uint32_t db_even = rf.read_reg(db_even_reg);
    const uint32_t db_odd = rf.read_reg(db_odd_reg);
    const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;

    const uint32_t new_even = db_even >> 1;
    uint32_t new_odd = rf.read_reg(dc_odd_reg);
    if (db_even & 1u) {
      new_odd = static_cast<uint32_t>(db_odd + (ra_val << sh));
    }

    rf.write_reg(dc_even_reg, new_even);
    rf.write_reg(dc_odd_reg, new_odd);
    rf.ZF = (new_even == 0);
    rf.CF = false;

    if (eval_cc_name(ins.cc_name, rf, new_even)) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_CALL_ANY: {
    int64_t target = 0;
    if (ins.signature == "call:rr") {
      target = static_cast<int64_t>(static_cast<int32_t>(ra));
    } else if (ins.signature == "call:rrr" || ins.signature == "call:zrr") {
      target = static_cast<int64_t>(static_cast<int32_t>(ra)) +
               static_cast<int64_t>(static_cast<int32_t>(rb));
    } else {
      target = static_cast<int64_t>(static_cast<int32_t>(ra)) +
               static_cast<int64_t>(ins.off);
    }
    if (ins.has_rc) {
      rf.write_reg(static_cast<uint8_t>(ins.rc), rf.pc + 1);
    }
    next_pc = static_cast<uint32_t>(target);
    return true;
  }
  case FAST_OP_JUMP_ANY: {
    const int64_t target = static_cast<int64_t>(static_cast<int32_t>(ra)) +
                           static_cast<int64_t>(ins.off);
    next_pc = static_cast<uint32_t>(target);
    return true;
  }
  case FAST_OP_LW_RRI: {
    const uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(ra) +
                                                static_cast<int32_t>(ins.off));
    const uint32_t v = wram_load_u32(dpu, addr, ins.endian != 0);
    rf.write_reg(static_cast<uint8_t>(ins.rc), v);
    rf.ZF = (v == 0);
    return true;
  }
  case FAST_OP_LBU_RRI: {
    const uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(ra) +
                                                static_cast<int32_t>(ins.off));
    const uint8_t v = wram_load_u8(dpu, addr);
    rf.write_reg(static_cast<uint8_t>(ins.rc), static_cast<uint32_t>(v));
    rf.ZF = (v == 0);
    return true;
  }
  case FAST_OP_LBS_RRI: {
    const uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(ra) +
                                                static_cast<int32_t>(ins.off));
    const int8_t v = static_cast<int8_t>(wram_load_u8(dpu, addr));
    rf.write_reg(static_cast<uint8_t>(ins.rc),
                 static_cast<uint32_t>(static_cast<int32_t>(v)));
    rf.ZF = (v == 0);
    return true;
  }
  case FAST_OP_SW_RII: {
    const uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(ra) +
                                                static_cast<int32_t>(ins.off));
    const uint32_t val = static_cast<uint32_t>(ins.imm);
    wram_store_u32(dpu, addr, val, ins.endian != 0);
    return true;
  }
  case FAST_OP_SW_RIR: {
    const uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(ra) +
                                                static_cast<int32_t>(ins.off));
    wram_store_u32(dpu, addr, rb, ins.endian != 0);
    return true;
  }
  case FAST_OP_SB_RIR: {
    const uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(ra) +
                                                static_cast<int32_t>(ins.off));
    wram_store_u8(dpu, addr, static_cast<uint8_t>(rb & 0xFFu));
    return true;
  }
  case FAST_OP_LD_RRI: {
    const uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(ra) +
                                                static_cast<int32_t>(ins.off));
    const uint64_t v = wram_load_u64(dpu, addr, ins.endian != 0);
    rf.write_dreg(dreg_slot(ins.dc), v);
    rf.ZF = (v == 0);
    return true;
  }
  case FAST_OP_SD_RIR: {
    const uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(ra) +
                                                static_cast<int32_t>(ins.off));
    const uint64_t v = rf.read_dreg(dreg_slot(ins.db));
    wram_store_u64(dpu, addr, v, ins.endian != 0);
    return true;
  }
  case FAST_OP_LDMA_RRI:
  case FAST_OP_SDMA_RRI: {
    const uint32_t ra_val = ra;
    const uint32_t rb_val = rb;
    const uint32_t w = ra_val & 0xFFFFF8u;
    const uint32_t m = rb_val & 0xFFFFFFF8u;
    const uint32_t n =
        (1u + ((static_cast<uint32_t>(ins.immDma) + ((ra_val >> 24) & 0xFFu)) &
               0xFFu))
        << 3;

    const bool load_to_wram = (ins.fast_op == FAST_OP_LDMA_RRI);
    (void)DMAEngine::transfer(dpu, w, m, static_cast<size_t>(n), load_to_wram);
    return true;
  }
  case FAST_OP_LSR_ADD_RRRI: {
    const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
    const uint32_t shifted = ra >> sh;
    const uint32_t x = rb + shifted;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (shifted == 0);
    return true;
  }
  case FAST_OP_LSR_RRI: {
    const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
    const uint32_t x = ra >> sh;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_LSR_RRICI: {
    const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
    const uint32_t x = ra >> sh;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    if (eval_cc_name(ins.cc_name, rf, x)) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_LSL_RRI: {
    const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
    const uint32_t x = ra << sh;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_LSL_RRR: {
    const uint32_t sh = rb & 31u;
    const uint32_t x = ra << sh;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_DIV_STEP_RRRI: {
    const uint32_t db_even_reg = static_cast<uint8_t>(ins.db);
    const uint32_t db_odd_reg = static_cast<uint8_t>(ins.db + 1);
    const uint32_t dc_even_reg = static_cast<uint8_t>(ins.dc);
    const uint32_t dc_odd_reg = static_cast<uint8_t>(ins.dc + 1);

    const uint32_t db_even = rf.read_reg(db_even_reg);
    const uint32_t db_odd = rf.read_reg(db_odd_reg);
    const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
    const uint32_t shifted = ra << sh;
    const uint32_t trial = static_cast<uint32_t>(db_odd - shifted);

    uint32_t new_even = 0;
    uint32_t new_odd = 0;
    if (db_odd >= shifted) {
      new_even = static_cast<uint32_t>((db_even << 1) | 1u);
      new_odd = trial;
    } else {
      new_even = static_cast<uint32_t>(db_even << 1);
      new_odd = rf.read_reg(dc_odd_reg);
    }

    rf.write_reg(static_cast<uint8_t>(dc_even_reg), new_even);
    rf.write_reg(static_cast<uint8_t>(dc_odd_reg), new_odd);

    rf.ZF = (trial == 0u);
    rf.CF = false;
    return true;
  }
  case FAST_OP_MUL_UL_UL_RRR: {
    const uint32_t x = (ra & 0xFFu) * (rb & 0xFFu);
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_MUL_UL_UH_RRR: {
    const uint32_t x = (ra & 0xFFu) * ((rb >> 8) & 0xFFu);
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_MUL_UH_UL_RRR: {
    const uint32_t x = ((ra >> 8) & 0xFFu) * (rb & 0xFFu);
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_MUL_UH_UH_RRR: {
    const uint32_t x = ((ra >> 8) & 0xFFu) * ((rb >> 8) & 0xFFu);
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (x == 0);
    return true;
  }
  case FAST_OP_LSL_ADD_RRRI: {
    const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
    const uint32_t shifted = ra << sh;
    const uint32_t x = rb + shifted;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (shifted == 0);
    return true;
  }
  case FAST_OP_LSL_ADD_RRRICI: {
    const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
    const uint32_t shifted = ra << sh;
    const uint32_t x = rb + shifted;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (shifted == 0);
    if (eval_cc_name(ins.cc_name, rf, x)) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_LSL_SUB_RRRI: {
    const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
    const uint32_t shifted = ra << sh;
    const uint32_t x = rb - shifted;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (shifted == 0);
    rf.CF = (rb < shifted);
    return true;
  }
  case FAST_OP_LSL_SUB_RRRICI: {
    const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
    const uint32_t shifted = ra << sh;
    const uint32_t x = rb - shifted;
    rf.write_reg(static_cast<uint8_t>(ins.rc), x);
    rf.ZF = (shifted == 0);
    rf.CF = (rb < shifted);
    if (eval_cc_name(ins.cc_name, rf, x)) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_SUB_ZRR: {
    const uint32_t x = static_cast<uint32_t>(ra - rb);
    rf.ZF = (x == 0);
    rf.CF = (ra < rb);
    return true;
  }
  case FAST_OP_SUB_ZRICI: {
    const uint32_t rhs = imm_u32;
    const uint32_t x = static_cast<uint32_t>(ra - rhs);
    rf.ZF = (x == 0);
    rf.CF = (ra < rhs);
    if (eval_cc_name(ins.cc_name, rf, x)) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_SUB_ZRRCI: {
    const uint32_t x = static_cast<uint32_t>(ra - rb);
    rf.ZF = (x == 0);
    rf.CF = (ra < rb);
    if (eval_cc_name(ins.cc_name, rf, x)) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_SUBC_ZRRCI: {
    const bool old_zf = rf.ZF;
    const uint32_t borrow = rf.CF ? 1u : 0u;
    const uint32_t rhs = static_cast<uint32_t>(rb + borrow);
    const bool rhs_wrap = rhs < rb;
    const uint32_t x = static_cast<uint32_t>(ra - rhs);
    const bool s1 = ((ra >> 31) & 1u) != 0;
    const bool s2 = ((rb >> 31) & 1u) != 0;
    const bool sr = ((x >> 31) & 1u) != 0;
    const bool overflow = (s1 && !s2 && !sr) || (!s1 && s2 && sr);

    rf.ZF = (x == 0);
    rf.CF = (ra < rhs) || rhs_wrap;

    if (eval_subc_cc(ins.cc_name, rf, x, old_zf, overflow)) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_ACQUIRE_RICI: {
    const uint32_t addr = static_cast<uint32_t>(ra + imm_u32);
    const uint8_t index =
        static_cast<uint8_t>(((addr >> 8) & 0xFFu) ^ (addr & 0xFFu));
    const uint8_t byte_idx = static_cast<uint8_t>(index / 8);
    const uint8_t bit_idx = static_cast<uint8_t>(index % 8);
    const uint8_t mask = static_cast<uint8_t>(1u << bit_idx);

    auto *atomic = dpu.private_mem.data() + ATOMIC_OFFSET;
    const bool was_free = (atomic[byte_idx] & mask) == 0;
    atomic[byte_idx] = static_cast<uint8_t>(atomic[byte_idx] | mask);

    const uint32_t result = was_free ? 0u : 1u;
    rf.ZF = (result == 0);

    bool cond = false;
    if (ins.cc_name == "nz") {
      cond = (result != 0u);
    } else if (ins.cc_name == "z") {
      cond = (result == 0u);
    } else if (ins.cc_name == "true") {
      cond = true;
    } else if (ins.cc_name == "false" || ins.cc_name.empty()) {
      cond = false;
    } else {
      cond = eval_cc_name(ins.cc_name, rf, result);
    }

    if (cond) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  case FAST_OP_RELEASE_RICI: {
    const uint32_t addr = static_cast<uint32_t>(ra + imm_u32);
    const uint8_t index =
        static_cast<uint8_t>(((addr >> 8) & 0xFFu) ^ (addr & 0xFFu));
    const uint8_t byte_idx = static_cast<uint8_t>(index / 8);
    const uint8_t bit_idx = static_cast<uint8_t>(index % 8);
    const uint8_t mask = static_cast<uint8_t>(1u << bit_idx);

    auto *atomic = dpu.private_mem.data() + ATOMIC_OFFSET;
    const bool was_held = (atomic[byte_idx] & mask) != 0;
    atomic[byte_idx] = static_cast<uint8_t>(atomic[byte_idx] & ~mask);

    const uint32_t result = was_held ? 0u : 1u;
    rf.ZF = (result == 0u);

    bool cond = false;
    if (ins.cc_name == "nz") {
      cond = (result != 0u);
    } else if (ins.cc_name == "z") {
      cond = (result == 0u);
    } else if (ins.cc_name == "true") {
      cond = true;
    } else if (ins.cc_name == "false" || ins.cc_name.empty()) {
      cond = false;
    } else {
      cond = eval_cc_name(ins.cc_name, rf, result);
    }

    if (cond) {
      next_pc = static_cast<uint32_t>(ins.pc);
    }
    return true;
  }
  default:
    break;
  }

  (void)state;
  (void)t;
  (void)keep_running;
  return false;
}

bool execute_launch_program_48(DpuState &dpu) {
  if (!dpu.launch_pending) {
    return true;
  }

  if (!MRAM::is_bound(dpu)) {
    exec_tracef("launch skipped: mram not bound");
    dpu.running_tasklets = 0u;
    dpu.launch_pending = false;
    return true;
  }

  using launch_clock = std::chrono::steady_clock;
  const bool launch_timing = exec_launch_timing_enabled();
  const auto launch_t0 =
      launch_timing ? launch_clock::now() : launch_clock::time_point{};

  const std::shared_ptr<const DecodedProgram48CacheEntry> decoded_cache =
      get_decoded_program_cache_48(dpu);
  if (!decoded_cache) {
    return false;
  }

  const std::vector<DecodedInst48> &program = decoded_cache->decoded;
  const auto launch_t_decode_done =
      launch_timing ? launch_clock::now() : launch_clock::time_point{};

  exec_tracef("launch48 begin: run_mask=0x%08x insts=%zu mram[0..3]=%02x %02x "
              "%02x %02x",
              dpu.running_tasklets, program.size(),
              static_cast<unsigned>(dpu.mram_base[0]),
              static_cast<unsigned>(dpu.mram_base[1]),
              static_cast<unsigned>(dpu.mram_base[2]),
              static_cast<unsigned>(dpu.mram_base[3]));
  if (exec_trace_enabled()) {
    const size_t wb = WRAM_OFFSET + 0xF8u;
    if (wb + 8u <= dpu.private_mem.size()) {
      exec_tracef(
          "launch48 wram[0xf8..0xff]=%02x %02x %02x %02x %02x %02x %02x %02x",
          static_cast<unsigned>(dpu.private_mem[wb + 0]),
          static_cast<unsigned>(dpu.private_mem[wb + 1]),
          static_cast<unsigned>(dpu.private_mem[wb + 2]),
          static_cast<unsigned>(dpu.private_mem[wb + 3]),
          static_cast<unsigned>(dpu.private_mem[wb + 4]),
          static_cast<unsigned>(dpu.private_mem[wb + 5]),
          static_cast<unsigned>(dpu.private_mem[wb + 6]),
          static_cast<unsigned>(dpu.private_mem[wb + 7]));
    }
  }

  dpu_state state{};
  for (int t = 0; t < NUM_TASKLETS; ++t) {
    auto &rf = state.tasklets[t];
    rf.thread_id = static_cast<uint32_t>(t);
    rf.ZF = false;
    rf.CF = false;
    rf.pc = 0;
    rf.perf_counter = 0;
    rf.perf_counter_mode = 0;
    rf.stopped = false;
    rf.replay_pending_wmask = 0u;
    rf.replay_src0 = 0xFFu;
    rf.replay_src1 = 0xFFu;

    if (dpu.running_tasklets & (1u << static_cast<uint32_t>(t))) {
      state.set_running(t);
    }
  }

  if (state.run_bits == 0) {
    state.set_running(0);
  }

  state.perf_counter_raw = 0u;
  state.perf_counter_mode = 0u;

  std::memset(dpu.private_mem.data() + ATOMIC_OFFSET, 0, ATOMIC_SIZE);

  const size_t max_pc = program.size();
  const size_t max_exec_steps = get_max_exec_steps();
  const bool trace_enabled = exec_trace_enabled();
  const bool profile_sig_enabled = exec_profile_sig_enabled();
  const ReplayModelMode replay_mode = replay_model_mode();
  const bool replay_model = (replay_mode != ReplayModelMode::kDisabled);
  std::unordered_map<std::string, size_t> sig_counts;
  if (profile_sig_enabled) {
    sig_counts.reserve(512);
  }
  size_t steps = 0;

  while (((state.run_bits | state.sleep_bits) &
          static_cast<uint64_t>(kTaskletMask)) != 0u &&
         steps < max_exec_steps) {
    if ((state.run_bits & static_cast<uint64_t>(kTaskletMask)) == 0u) {
      break;
    }

    uint32_t run_mask_iter = static_cast<uint32_t>(
        state.run_bits & static_cast<uint64_t>(kTaskletMask));
    while (run_mask_iter != 0u && steps < max_exec_steps) {
      const int t = __builtin_ctz(run_mask_iter);
      run_mask_iter &= (run_mask_iter - 1u);

      auto &rf = state.tasklets[t];
      state.clear_replaying(t);
      if (rf.pc >= max_pc) {
        state.set_sleeping(t);
        continue;
      }

      const DecodedInst48 &ins = program[rf.pc];
      if (!ins.valid) {
        exec_tracef("launch48 invalid decode at pc=%u (treated as nop)", rf.pc);
        rf.pc += 1;
        ++steps;
        continue;
      }

      const uint32_t pending_wmask = rf.replay_pending_wmask;
      rf.replay_pending_wmask = 0u;

      const uint32_t active_threads_now =
          static_cast<uint32_t>(__builtin_popcountll(
              state.run_bits & static_cast<uint64_t>(kTaskletMask)));
      const bool strict_sig_allowed =
          (replay_mode != ReplayModelMode::kStrict) ||
          replay_strict_sig_allowed_48(ins.signature);
      const bool replay_gate_ok =
          (replay_mode == ReplayModelMode::kStrict && strict_sig_allowed) ||
          (active_threads_now <= 2u);
      const uint8_t prior_src0 = rf.replay_src0;
      const uint8_t prior_src1 = rf.replay_src1;
      const bool raw_dep = (pending_wmask != 0u) &&
                           ((ins.replay_read_mask & pending_wmask) != 0u);
      const bool source_dep = replay_should_fire_from_sources_48(ins, rf);
      if (replay_model && replay_gate_ok && (raw_dep || source_dep)) {
        state.set_replaying(t);
        rf.replay_src0 = 0xFFu;
        rf.replay_src1 = 0xFFu;
        if (trace_enabled && steps < 400) {
          exec_tracef(
              "replay48 step=%zu t=%d pc=%u sig=%s mode=%u strict_ok=%d raw=%d "
              "src=%d read=0x%08x pend=0x%08x srcbytes=%02x/%02x",
              steps, t, rf.pc, ins.signature.c_str(),
              static_cast<unsigned>(replay_mode), strict_sig_allowed ? 1 : 0,
              raw_dep ? 1 : 0, source_dep ? 1 : 0, ins.replay_read_mask,
              pending_wmask, static_cast<unsigned>(prior_src0),
              static_cast<unsigned>(prior_src1));
        }
        Pipeline::perf_counter_retire_step(state, true);
        ++steps;
        continue;
      }

      if (profile_sig_enabled) {
        ++sig_counts[ins.signature];
      }

      if (trace_enabled && steps < 400) {
        exec_tracef("step48=%zu t=%d pc=%u sig=%s", steps, t, rf.pc,
                    ins.signature.c_str());
      }
      uint32_t next_pc = rf.pc + 1;
      bool keep_running = true;

      const uint32_t imm_u32 = static_cast<uint32_t>(ins.imm);
      bool fast_handled = false;
      uint32_t fast_ra = 0;
      uint32_t fast_rb = 0;

      if (!trace_enabled && ins.fast_op != FAST_OP_NONE) {
        fast_ra = rf.read_reg(static_cast<uint8_t>(ins.ra));
        if (fast_op_needs_rb(ins.fast_op)) {
          fast_rb = rf.read_reg(static_cast<uint8_t>(ins.rb));
        }
        fast_handled =
            try_execute_fast_op_48(dpu, state, t, rf, ins, fast_ra, fast_rb,
                                   imm_u32, next_pc, keep_running);
      }

      if (fast_handled) {
        /* handled by fast dispatch */
      } else {
        const uint32_t ra = rf.read_reg(static_cast<uint8_t>(ins.ra));
        const uint32_t rb = rf.read_reg(static_cast<uint8_t>(ins.rb));

        if (ins.signature == "nop:") {
          /* no-op */
        } else if (ins.signature == "move:ri" || ins.signature == "move:rr") {
          const uint32_t x = ra | imm_u32;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "move.u:rr") {
          const uint32_t x = ra | imm_u32;
          rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
          rf.ZF = (x == 0);
        } else if (ins.signature == "move.u:ri") {
          const uint32_t x = ra & imm_u32;
          rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
          rf.ZF = (x == 0);
        } else if (ins.signature == "move.s:rr") {
          const int32_t x = static_cast<int32_t>(ra | imm_u32);
          rf.write_dreg(dreg_slot(ins.dc),
                        static_cast<uint64_t>(static_cast<int64_t>(x)));
          rf.ZF = (x == 0);
        } else if (ins.signature == "move.s:ri") {
          const int32_t x = static_cast<int32_t>(ra & imm_u32);
          rf.write_dreg(dreg_slot(ins.dc),
                        static_cast<uint64_t>(static_cast<int64_t>(x)));
          rf.ZF = (x == 0);
        } else if (ins.signature == "movd:rr") {
          const uint64_t v = rf.read_dreg(dreg_slot(ins.db));
          rf.write_dreg(dreg_slot(ins.dc), v);
        } else if (ins.signature == "movd:rrci") {
          const uint64_t v = rf.read_dreg(dreg_slot(ins.db));
          rf.write_dreg(dreg_slot(ins.dc), v);
          if (eval_cc_name(ins.cc_name, rf, static_cast<uint32_t>(v))) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "move:rici" ||
                   ins.signature == "move:rrci" ||
                   ins.signature == "or:rrici") {
          const uint32_t x = ra | imm_u32;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "or.u:rrici") {
          const uint32_t x = ra | imm_u32;
          rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
          rf.ZF = (x == 0);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "or:rri") {
          const uint32_t x = ra | imm_u32;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "or.u:rri") {
          const uint32_t x = ra | imm_u32;
          rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
          rf.ZF = (x == 0);
        } else if (ins.signature == "or:rrr") {
          const uint32_t x = ra | rb;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "or.u:rrr") {
          const uint32_t x = ra | rb;
          rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
          rf.ZF = (x == 0);
        } else if (ins.signature == "or:rrrci") {
          const uint32_t x = ra | rb;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "or.u:rrrci") {
          const uint32_t x = ra | rb;
          rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
          rf.ZF = (x == 0);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "xor:rrr") {
          const uint32_t x = ra ^ rb;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "move.u:rrci") {
          const uint32_t x = ra | imm_u32;
          rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
          rf.ZF = (x == 0);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "move.s:rrci") {
          const int32_t x = static_cast<int32_t>(ra | imm_u32);
          rf.write_dreg(dreg_slot(ins.dc),
                        static_cast<uint64_t>(static_cast<int64_t>(x)));
          rf.ZF = (x == 0);
          if (eval_cc_name(ins.cc_name, rf, static_cast<uint32_t>(x))) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "not:rr") {
          const uint32_t x = ra ^ imm_u32;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "neg:rr") {
          const uint32_t x = static_cast<uint32_t>(imm_u32 - ra);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (imm_u32 < ra);
        } else if (ins.signature == "and:rri") {
          const uint32_t x = ra & imm_u32;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "and:rrr") {
          const uint32_t x = ra & rb;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "and:rrici") {
          const uint32_t x = ra & imm_u32;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "add:rri") {
          const uint64_t sum =
              static_cast<uint64_t>(ra) + static_cast<uint64_t>(imm_u32);
          const uint32_t x = static_cast<uint32_t>(sum);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (sum >> 32) != 0;
        } else if (ins.signature == "add.u:rri") {
          const uint64_t sum =
              static_cast<uint64_t>(rb) + static_cast<uint64_t>(imm_u32);
          rf.write_dreg(dreg_slot(ins.dc), sum);
          rf.ZF = (sum == 0);
          rf.CF = false;
        } else if (ins.signature == "add:rrr") {
          const uint64_t sum =
              static_cast<uint64_t>(ra) + static_cast<uint64_t>(rb);
          const uint32_t x = static_cast<uint32_t>(sum);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (sum >> 32) != 0;
        } else if (ins.signature == "addc:rri") {
          const uint64_t sum = static_cast<uint64_t>(ra) +
                               static_cast<uint64_t>(imm_u32) +
                               static_cast<uint64_t>(rf.CF ? 1u : 0u);
          const uint32_t x = static_cast<uint32_t>(sum);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (sum >> 32) != 0;
        } else if (ins.signature == "addc:rrr") {
          const uint64_t sum = static_cast<uint64_t>(ra) +
                               static_cast<uint64_t>(rb) +
                               static_cast<uint64_t>(rf.CF ? 1u : 0u);
          const uint32_t x = static_cast<uint32_t>(sum);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (sum >> 32) != 0;
        } else if (ins.signature == "add:rrrci") {
          const uint64_t sum =
              static_cast<uint64_t>(ra) + static_cast<uint64_t>(rb);
          const uint32_t x = static_cast<uint32_t>(sum);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (sum >> 32) != 0;
          if (eval_add_cc_name(ins.cc_name, rf, ra, rb, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "add:rrici") {
          const int32_t imm_s8 =
              static_cast<int32_t>(static_cast<int8_t>(ins.imm & 0xFF));
          const uint32_t rhs = static_cast<uint32_t>(imm_s8);
          const uint64_t sum =
              static_cast<uint64_t>(ra) + static_cast<uint64_t>(rhs);
          const uint32_t x = static_cast<uint32_t>(sum);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (sum >> 32) != 0;
          if (eval_add_cc_name(ins.cc_name, rf, ra, rhs, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "sub:rrif") {
          const uint32_t rhs = imm_u32;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs);
        } else if (ins.signature == "sub:rrr") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);
        } else if (ins.signature == "sub:rrrci") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "sub:zrr") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);
        } else if (ins.signature == "sub:zrici") {
          const uint32_t rhs = imm_u32;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "sub:zrrci") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "sub:rrrc") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);

          bool cond = false;
          if (ins.cc_name == "ltu" || ins.cc_name == "c") {
            cond = rf.CF;
          } else if (ins.cc_name == "geu" || ins.cc_name == "nc") {
            cond = !rf.CF;
          } else if (ins.cc_name == "z") {
            cond = (x == 0);
          } else if (ins.cc_name == "nz") {
            cond = (x != 0);
          } else {
            cond = eval_cc_name(ins.cc_name, rf, x);
          }

          rf.write_reg(static_cast<uint8_t>(ins.rc), cond ? 1u : 0u);
        } else if (ins.signature == "subc:rrr") {
          const uint32_t borrow = rf.CF ? 1u : 0u;
          const uint32_t rhs = static_cast<uint32_t>(rb + borrow);
          const bool rhs_wrap = rhs < rb;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);

          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs) || rhs_wrap;
        } else if (ins.signature == "subc:zrr") {
          const uint32_t borrow = rf.CF ? 1u : 0u;
          const uint32_t rhs = static_cast<uint32_t>(rb + borrow);
          const bool rhs_wrap = rhs < rb;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);

          rf.ZF = (x == 0);
          rf.CF = (ra < rhs) || rhs_wrap;
        } else if (ins.signature == "subc:zrrci") {
          const bool old_zf = rf.ZF;
          const uint32_t borrow = rf.CF ? 1u : 0u;
          const uint32_t rhs = static_cast<uint32_t>(rb + borrow);
          const bool rhs_wrap = rhs < rb;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          const bool s1 = ((ra >> 31) & 1u) != 0;
          const bool s2 = ((rb >> 31) & 1u) != 0;
          const bool sr = ((x >> 31) & 1u) != 0;
          const bool overflow = (s1 && !s2 && !sr) || (!s1 && s2 && sr);

          rf.ZF = (x == 0);
          rf.CF = (ra < rhs) || rhs_wrap;

          const bool cond = eval_subc_cc(ins.cc_name, rf, x, old_zf, overflow);
          if (cond) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "subc:rrrc") {
          const bool old_zf = rf.ZF;
          const uint32_t borrow = rf.CF ? 1u : 0u;
          const uint32_t rhs = static_cast<uint32_t>(rb + borrow);
          const bool rhs_wrap = rhs < rb;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          const bool s1 = ((ra >> 31) & 1u) != 0;
          const bool s2 = ((rb >> 31) & 1u) != 0;
          const bool sr = ((x >> 31) & 1u) != 0;
          const bool overflow = (s1 && !s2 && !sr) || (!s1 && s2 && sr);

          rf.ZF = (x == 0);
          rf.CF = (ra < rhs) || rhs_wrap;

          const bool cond = eval_subc_cc(ins.cc_name, rf, x, old_zf, overflow);
          rf.write_reg(static_cast<uint8_t>(ins.rc), cond ? 1u : 0u);
        } else if (ins.signature == "jeq:rii") {
          const uint32_t rhs = imm_u32;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs);
          if (ra == rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jeq:rri") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);
          if (ra == rb) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jneq:rii") {
          const uint32_t rhs = imm_u32;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs);
          if (ra != rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jneq:rri") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);
          if (ra != rb) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jz:ri") {
          const uint32_t rhs = imm_u32;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs);
          if (x == 0) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jnz:ri") {
          const uint32_t rhs = imm_u32;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs);
          if (x != 0) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jltu:rii") {
          const uint32_t rhs = imm_u32;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs);
          if (ra < rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jltu:rri") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);
          if (ra < rb) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jleu:rri") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);
          if (ra <= rb) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jgtu:rri") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);
          if (ra > rb) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jleu:rii") {
          const uint32_t rhs = imm_u32;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs);
          if (ra <= rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jgtu:rii") {
          const uint32_t rhs = imm_u32;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs);
          if (ra > rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jgeu:rri") {
          const uint32_t x = static_cast<uint32_t>(ra - rb);
          rf.ZF = (x == 0);
          rf.CF = (ra < rb);
          if (ra >= rb) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jgeu:rii") {
          const uint32_t rhs = imm_u32;
          const uint32_t x = static_cast<uint32_t>(ra - rhs);
          rf.ZF = (x == 0);
          rf.CF = (ra < rhs);
          if (ra >= rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jlts:rri") {
          const int32_t lhs = static_cast<int32_t>(ra);
          const int32_t rhs = static_cast<int32_t>(rb);
          const uint32_t x = static_cast<uint32_t>(lhs - rhs);
          rf.ZF = (x == 0);
          if (lhs < rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jlts:rii") {
          const int32_t lhs = static_cast<int32_t>(ra);
          const int32_t rhs = static_cast<int32_t>(ins.imm);
          const uint32_t x = static_cast<uint32_t>(lhs - rhs);
          rf.ZF = (x == 0);
          if (lhs < rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jles:rii") {
          const int32_t lhs = static_cast<int32_t>(ra);
          const int32_t rhs = static_cast<int32_t>(ins.imm);
          const uint32_t x = static_cast<uint32_t>(lhs - rhs);
          rf.ZF = (x == 0);
          if (lhs <= rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jgts:rii") {
          const int32_t lhs = static_cast<int32_t>(ra);
          const int32_t rhs = static_cast<int32_t>(ins.imm);
          const uint32_t x = static_cast<uint32_t>(lhs - rhs);
          rf.ZF = (x == 0);
          if (lhs > rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jgts:rri") {
          const int32_t lhs = static_cast<int32_t>(ra);
          const int32_t rhs = static_cast<int32_t>(rb);
          const uint32_t x = static_cast<uint32_t>(lhs - rhs);
          rf.ZF = (x == 0);
          if (lhs > rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jges:rri") {
          const int32_t lhs = static_cast<int32_t>(ra);
          const int32_t rhs = static_cast<int32_t>(rb);
          const uint32_t x = static_cast<uint32_t>(lhs - rhs);
          rf.ZF = (x == 0);
          if (lhs >= rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jges:rii") {
          const int32_t lhs = static_cast<int32_t>(ra);
          const int32_t rhs = static_cast<int32_t>(ins.imm);
          const uint32_t x = static_cast<uint32_t>(lhs - rhs);
          rf.ZF = (x == 0);
          if (lhs >= rhs) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "jump:i" || ins.signature == "jump:r" ||
                   ins.signature == "jump:ri") {
          const int64_t target =
              static_cast<int64_t>(static_cast<int32_t>(ra)) +
              static_cast<int64_t>(ins.off);
          next_pc = static_cast<uint32_t>(target);
        } else if (ins.signature == "call:ri" || ins.signature == "call:rri" ||
                   ins.signature == "call:rr" || ins.signature == "call:rrr" ||
                   ins.signature == "call:zri" || ins.signature == "call:zrr") {
          int64_t target = 0;
          if (ins.signature == "call:rr") {
            target = static_cast<int64_t>(static_cast<int32_t>(ra));
          } else if (ins.signature == "call:rrr" ||
                     ins.signature == "call:zrr") {
            target = static_cast<int64_t>(static_cast<int32_t>(ra)) +
                     static_cast<int64_t>(static_cast<int32_t>(rb));
          } else {
            target = static_cast<int64_t>(static_cast<int32_t>(ra)) +
                     static_cast<int64_t>(ins.off);
          }
          if (ins.has_rc) {
            rf.write_reg(static_cast<uint8_t>(ins.rc), rf.pc + 1);
          }
          next_pc = static_cast<uint32_t>(target);
        } else if (ins.signature == "boot:ri" || ins.signature == "boot:rici") {
          const uint32_t addr = static_cast<uint32_t>(ra + imm_u32);
          const uint32_t tid = ((addr >> 8) & 0xFFu) ^ (addr & 0xFFu);

          uint32_t result = 1;
          if (tid < NUM_TASKLETS && !state.is_running(static_cast<int>(tid)) &&
              !state.is_sleeping(static_cast<int>(tid))) {
            auto &boot_rf = state.tasklets[tid];
            boot_rf.thread_id = tid;
            boot_rf.ZF = false;
            boot_rf.CF = false;
            boot_rf.pc = 0;
            boot_rf.stopped = false;
            state.set_running(static_cast<int>(tid));
            result = 0;
          }

          rf.ZF = (result == 0);
          if (eval_cc_name(ins.cc_name, rf, result)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "stop:ci") {
          if (eval_cc_name(ins.cc_name, rf, 1)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
          keep_running = false;
        } else if (ins.signature == "stop:") {
          keep_running = false;
        } else if (ins.signature == "acquire:rici") {
          const uint32_t addr = static_cast<uint32_t>(ra + imm_u32);
          const uint8_t index =
              static_cast<uint8_t>(((addr >> 8) & 0xFFu) ^ (addr & 0xFFu));
          const uint8_t byte_idx = static_cast<uint8_t>(index / 8);
          const uint8_t bit_idx = static_cast<uint8_t>(index % 8);
          const uint8_t mask = static_cast<uint8_t>(1u << bit_idx);

          auto *atomic = dpu.private_mem.data() + ATOMIC_OFFSET;
          const bool was_free = (atomic[byte_idx] & mask) == 0;
          atomic[byte_idx] = static_cast<uint8_t>(atomic[byte_idx] | mask);

          const uint32_t result = was_free ? 0u : 1u;
          rf.ZF = (result == 0);

          bool cond = false;
          if (ins.cc_name == "nz") {
            cond = (result != 0u);
          } else if (ins.cc_name == "z") {
            cond = (result == 0u);
          } else if (ins.cc_name == "true") {
            cond = true;
          } else if (ins.cc_name == "false" || ins.cc_name.empty()) {
            cond = false;
          } else {
            cond = eval_cc_name(ins.cc_name, rf, result);
          }

          if (cond) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "release:rici") {
          const uint32_t addr = static_cast<uint32_t>(ra + imm_u32);
          const uint8_t index =
              static_cast<uint8_t>(((addr >> 8) & 0xFFu) ^ (addr & 0xFFu));
          const uint8_t byte_idx = static_cast<uint8_t>(index / 8);
          const uint8_t bit_idx = static_cast<uint8_t>(index % 8);
          const uint8_t mask = static_cast<uint8_t>(1u << bit_idx);

          auto *atomic = dpu.private_mem.data() + ATOMIC_OFFSET;
          const bool was_held = (atomic[byte_idx] & mask) != 0;
          atomic[byte_idx] = static_cast<uint8_t>(atomic[byte_idx] & ~mask);

          const uint32_t result = was_held ? 0u : 1u;
          rf.ZF = (result == 0u);

          bool cond = false;
          if (ins.cc_name == "nz") {
            cond = (result != 0u);
          } else if (ins.cc_name == "z") {
            cond = (result == 0u);
          } else if (ins.cc_name == "true") {
            cond = true;
          } else if (ins.cc_name == "false" || ins.cc_name.empty()) {
            cond = false;
          } else {
            cond = eval_cc_name(ins.cc_name, rf, result);
          }

          if (cond) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "resume:rici") {
          const int32_t tid =
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.imm);
          bool can_resume = false;
          if (tid >= 0 && tid < NUM_TASKLETS &&
              state.is_sleeping(static_cast<int>(tid))) {
            state.set_running(static_cast<int>(tid));
            can_resume = true;
          }

          const uint32_t result = can_resume ? 0u : 1u;
          rf.ZF = (result == 0u);
          if (eval_cc_name(ins.cc_name, rf, result)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "lsl:rri") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t x = ra << sh;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "lsl:rrr") {
          const uint32_t sh = rb & 31u;
          const uint32_t x = ra << sh;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "lsr:rrr") {
          const uint32_t sh = rb & 31u;
          const uint32_t x = ra >> sh;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "lslx:rri") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t x = (sh == 0) ? 0u : (ra >> (32u - sh));
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "lsr:rri") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t x = ra >> sh;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "lsr:rrici") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t x = ra >> sh;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "lsrx:rri") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t x = (sh == 0) ? 0u : (ra << (32u - sh));
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "lsr.u:rri") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t x = ra >> sh;
          rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
          rf.ZF = (x == 0);
        } else if (ins.signature == "lsl_add:rrri") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t shifted = ra << sh;
          const uint32_t x = rb + shifted;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (shifted == 0);
        } else if (ins.signature == "lsl_add:rrrici") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t shifted = ra << sh;
          const uint32_t x = rb + shifted;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (shifted == 0);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "lsl_sub:rrri") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t shifted = ra << sh;
          const uint32_t x = rb - shifted;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (shifted == 0);
          rf.CF = (rb < shifted);
        } else if (ins.signature == "lsl_sub:rrrici") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t shifted = ra << sh;
          const uint32_t x = rb - shifted;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (shifted == 0);
          rf.CF = (rb < shifted);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "lsr_add:rrri" ||
                   ins.signature == "lsr_add:rrrici" ||
                   ins.signature == "lsr_add.s:rrri" ||
                   ins.signature == "lsr_add.s:rrrici" ||
                   ins.signature == "lsr_add.u:rrri" ||
                   ins.signature == "lsr_add.u:rrrici" ||
                   ins.signature == "lsr_add:zrrici") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t shifted = ra >> sh;
          const uint32_t x = rb + shifted;
          rf.ZF = (shifted == 0);

          if (ins.signature == "lsr_add.s:rrri" ||
              ins.signature == "lsr_add.s:rrrici") {
            const int64_t sx = static_cast<int64_t>(static_cast<int32_t>(x));
            rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(sx));
          } else if (ins.signature == "lsr_add.u:rrri" ||
                     ins.signature == "lsr_add.u:rrrici") {
            rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
          } else if (ins.signature != "lsr_add:zrrici") {
            rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          }

          if (ins.signature == "lsr_add:rrrici" ||
              ins.signature == "lsr_add.s:rrrici" ||
              ins.signature == "lsr_add.u:rrrici" ||
              ins.signature == "lsr_add:zrrici") {
            if (eval_cc_name(ins.cc_name, rf, x)) {
              next_pc = static_cast<uint32_t>(ins.pc);
            }
          }
        } else if (ins.signature == "rol_add:rrri" ||
                   ins.signature == "rol_add:rrrici" ||
                   ins.signature == "rol_add.s:rrri" ||
                   ins.signature == "rol_add.u:rrrici") {
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t rotated =
              (sh == 0) ? ra : ((ra << sh) | (ra >> (32u - sh)));
          const uint32_t x = rb + rotated;
          rf.ZF = (rotated == 0);

          if (ins.signature == "rol_add.s:rrri") {
            const int64_t sx = static_cast<int64_t>(static_cast<int32_t>(x));
            rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(sx));
          } else if (ins.signature == "rol_add.u:rrrici") {
            rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
            if (eval_cc_name(ins.cc_name, rf, x)) {
              next_pc = static_cast<uint32_t>(ins.pc);
            }
          } else {
            rf.write_reg(static_cast<uint8_t>(ins.rc), x);
            if (ins.signature == "rol_add:rrrici" &&
                eval_cc_name(ins.cc_name, rf, x)) {
              next_pc = static_cast<uint32_t>(ins.pc);
            }
          }
        } else if (ins.signature == "mul_ul_ul:rrr") {
          const uint32_t x = (ra & 0xFFu) * (rb & 0xFFu);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "mul_ul_uh:rrr") {
          const uint32_t x = (ra & 0xFFu) * ((rb >> 8) & 0xFFu);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "mul_uh_ul:rrr") {
          const uint32_t x = ((ra >> 8) & 0xFFu) * (rb & 0xFFu);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "mul_uh_uh:rrr") {
          const uint32_t x = ((ra >> 8) & 0xFFu) * ((rb >> 8) & 0xFFu);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "mul_step:rrrici") {
          const uint32_t ra_val = ra;
          const uint8_t db_even_reg = static_cast<uint8_t>(ins.db);
          const uint8_t db_odd_reg = static_cast<uint8_t>(ins.db + 1);
          const uint8_t dc_even_reg = static_cast<uint8_t>(ins.dc);
          const uint8_t dc_odd_reg = static_cast<uint8_t>(ins.dc + 1);

          const uint32_t db_even = rf.read_reg(db_even_reg);
          const uint32_t db_odd = rf.read_reg(db_odd_reg);
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;

          const uint32_t new_even = db_even >> 1;
          uint32_t new_odd = rf.read_reg(dc_odd_reg);
          if (db_even & 1u) {
            new_odd = static_cast<uint32_t>(db_odd + (ra_val << sh));
          }

          rf.write_reg(dc_even_reg, new_even);
          rf.write_reg(dc_odd_reg, new_odd);
          rf.ZF = (new_even == 0);
          rf.CF = false;

          if (exec_trace_enabled()) {
            static size_t mul_step_trace = 0;
            if (mul_step_trace < 5000) {
              exec_tracef("mul_step t=%d dc=%d db=%d ra=r%d(0x%08x) sh=%u "
                          "db_even=0x%08x db_odd=0x%08x -> ne=0x%08x no=0x%08x",
                          t, ins.dc, ins.db, ins.ra, ra_val, sh, db_even,
                          db_odd, new_even, new_odd);
              ++mul_step_trace;
            }
          }

          if (eval_cc_name(ins.cc_name, rf, new_even)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "div_step:rrri" ||
                   ins.signature == "div_step:rrrici") {
          const uint32_t db_even_reg = static_cast<uint8_t>(ins.db);
          const uint32_t db_odd_reg = static_cast<uint8_t>(ins.db + 1);
          const uint32_t dc_even_reg = static_cast<uint8_t>(ins.dc);
          const uint32_t dc_odd_reg = static_cast<uint8_t>(ins.dc + 1);

          const uint32_t db_even = rf.read_reg(db_even_reg);
          const uint32_t db_odd = rf.read_reg(db_odd_reg);
          const uint32_t sh = static_cast<uint32_t>(ins.shift) & 31u;
          const uint32_t shifted = ra << sh;
          const uint32_t trial = static_cast<uint32_t>(db_odd - shifted);

          uint32_t new_even = 0;
          uint32_t new_odd = 0;
          if (db_odd >= shifted) {
            new_even = static_cast<uint32_t>((db_even << 1) | 1u);
            new_odd = trial;
          } else {
            new_even = static_cast<uint32_t>(db_even << 1);
            new_odd = rf.read_reg(dc_odd_reg);
          }

          rf.write_reg(static_cast<uint8_t>(dc_even_reg), new_even);
          rf.write_reg(static_cast<uint8_t>(dc_odd_reg), new_odd);

          rf.ZF = (trial == 0u);
          rf.CF = false;

          if (ins.signature == "div_step:rrrici" &&
              eval_cc_name(ins.cc_name, rf, trial)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "xor:rri") {
          const uint32_t x = ra ^ imm_u32;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "or:rrif") {
          const uint32_t x = ra | imm_u32;
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "or.u:rrif") {
          const uint32_t x = ra | imm_u32;
          rf.write_dreg(dreg_slot(ins.dc), static_cast<uint64_t>(x));
          rf.ZF = (x == 0);
        } else if (ins.signature == "sub:rir") {
          const uint32_t x = static_cast<uint32_t>(imm_u32 - ra);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          rf.CF = (imm_u32 < ra);
        } else if (ins.signature == "clz:rr") {
          const uint32_t x =
              (ra == 0) ? 32u : static_cast<uint32_t>(__builtin_clz(ra));
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
        } else if (ins.signature == "clz:rrci") {
          const uint32_t x =
              (ra == 0) ? 32u : static_cast<uint32_t>(__builtin_clz(ra));
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
          rf.ZF = (x == 0);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "clo:zrci") {
          const uint32_t x = (ra == 0xFFFFFFFFu)
                                 ? 32u
                                 : static_cast<uint32_t>(__builtin_clz(~ra));
          rf.ZF = (x == 0);
          if (eval_cc_name(ins.cc_name, rf, x)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "lbu:rri" || ins.signature == "lbu:erri") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const uint8_t v = wram_load_u8(dpu, addr);
          rf.write_reg(static_cast<uint8_t>(ins.rc), static_cast<uint32_t>(v));
          rf.ZF = (v == 0);
        } else if (ins.signature == "lbs:rri" || ins.signature == "lbs:erri") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const int8_t v = static_cast<int8_t>(wram_load_u8(dpu, addr));
          rf.write_reg(static_cast<uint8_t>(ins.rc),
                       static_cast<uint32_t>(static_cast<int32_t>(v)));
          rf.ZF = (v == 0);
        } else if (ins.signature == "lw:rri" || ins.signature == "lw:erri") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const uint32_t v = wram_load_u32(dpu, addr, ins.endian != 0);
          rf.write_reg(static_cast<uint8_t>(ins.rc), v);
          rf.ZF = (v == 0);
          if (exec_trace_enabled()) {
            static size_t lw_trace = 0;
            if (lw_trace < 5000) {
              exec_tracef(
                  "lw t=%d pc=%u rc=r%d addr=0x%08x ra=0x%08x off=%d v=0x%08x",
                  t, rf.pc, ins.rc, addr, ra, ins.off, v);
              ++lw_trace;
            }
          }
        } else if (ins.signature == "lw.s:rri") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const int32_t v =
              static_cast<int32_t>(wram_load_u32(dpu, addr, ins.endian != 0));
          rf.write_dreg(dreg_slot(ins.dc),
                        static_cast<uint64_t>(static_cast<int64_t>(v)));
          rf.ZF = (v == 0);
        } else if (ins.signature == "ld:rri" || ins.signature == "ld:erri") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const uint64_t v = wram_load_u64(dpu, addr, ins.endian != 0);
          rf.write_dreg(dreg_slot(ins.dc), v);
          rf.ZF = (v == 0);
          if (exec_trace_enabled()) {
            static size_t ld_trace = 0;
            if (ld_trace < 5000) {
              exec_tracef(
                  "ld t=%d dc=d%d addr=0x%08x ra=0x%08x off=%d v=0x%016llx", t,
                  ins.dc, addr, ra, ins.off,
                  static_cast<unsigned long long>(v));
              ++ld_trace;
            }
          }
        } else if (ins.signature == "sw:rii" || ins.signature == "sw:erii") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const uint32_t val = static_cast<uint32_t>(ins.imm);
          wram_store_u32(dpu, addr, val, ins.endian != 0);
          if (exec_trace_enabled()) {
            static size_t store_trace = 0;
            if (store_trace < 5000 && addr >= WRAM_SIZE) {
              exec_tracef("sw:rii t=%d addr=0x%08x v=0x%08x", t, addr, val);
              ++store_trace;
            }
          }
        } else if (ins.signature == "sw:rir" || ins.signature == "sw:erir") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          wram_store_u32(dpu, addr, rb, ins.endian != 0);
          if (exec_trace_enabled()) {
            static size_t store_trace = 0;
            if (store_trace < 5000 && addr >= WRAM_SIZE) {
              exec_tracef("sw:rir t=%d addr=0x%08x v=0x%08x", t, addr, rb);
              ++store_trace;
            }
          }
        } else if (ins.signature == "sb:rii" || ins.signature == "sb:erii") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const uint8_t val = static_cast<uint8_t>(ins.imm & 0xFF);
          wram_store_u8(dpu, addr, val);
        } else if (ins.signature == "sb:rir" || ins.signature == "sb:erir") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const uint8_t val = static_cast<uint8_t>(rb & 0xFFu);
          wram_store_u8(dpu, addr, val);
        } else if (ins.signature == "sh:rii" || ins.signature == "sh:erii") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const uint16_t val = static_cast<uint16_t>(imm_u32 & 0xFFFFu);
          wram_store_u16(dpu, addr, val, ins.endian != 0);
        } else if (ins.signature == "sh:rir" || ins.signature == "sh:erir") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const uint16_t val = static_cast<uint16_t>(rb & 0xFFFFu);
          wram_store_u16(dpu, addr, val, ins.endian != 0);
        } else if (ins.signature == "sd:rii" || ins.signature == "sd:erii") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const uint64_t v =
              static_cast<uint64_t>(static_cast<int64_t>(ins.imm));
          wram_store_u64(dpu, addr, v, ins.endian != 0);
          if (exec_trace_enabled()) {
            static size_t store_trace = 0;
            if (store_trace < 5000 && addr >= WRAM_SIZE) {
              exec_tracef("sd:rii t=%d addr=0x%08x v=0x%016llx", t, addr,
                          static_cast<unsigned long long>(v));
              ++store_trace;
            }
          }
        } else if (ins.signature == "sd:rir" || ins.signature == "sd:erir") {
          const uint32_t addr = static_cast<uint32_t>(
              static_cast<int32_t>(ra) + static_cast<int32_t>(ins.off));
          const uint64_t v = rf.read_dreg(dreg_slot(ins.db));
          wram_store_u64(dpu, addr, v, ins.endian != 0);
          if (exec_trace_enabled()) {
            static size_t store_trace = 0;
            if (store_trace < 5000 && addr >= WRAM_SIZE) {
              exec_tracef("sd:rir t=%d addr=0x%08x v=0x%016llx", t, addr,
                          static_cast<unsigned long long>(v));
              ++store_trace;
            }
          }
        } else if (ins.signature == "ldma:rri" || ins.signature == "sdma:rri") {
          const uint32_t ra_val = ra;
          const uint32_t rb_val = rb;
          const uint32_t w = ra_val & 0xFFFFF8u;
          const uint32_t m = rb_val & 0xFFFFFFF8u;
          const uint32_t n =
              (1u +
               ((static_cast<uint32_t>(ins.immDma) + ((ra_val >> 24) & 0xFFu)) &
                0xFFu))
              << 3;

          size_t copy_bytes = static_cast<size_t>(n);
          if (w >= WRAM_SIZE || m >= dpu.mram_size || !dpu.mram_base) {
            copy_bytes = 0;
          } else {
            copy_bytes =
                std::min(copy_bytes, static_cast<size_t>(WRAM_SIZE - w));
            copy_bytes =
                std::min(copy_bytes, static_cast<size_t>(dpu.mram_size - m));
          }

          const bool load_to_wram = (ins.signature == "ldma:rri");
          (void)DMAEngine::transfer(dpu, w, m, copy_bytes, load_to_wram);

          if (exec_trace_enabled()) {
            static size_t dma_trace = 0;
            if (dma_trace < 256) {
              exec_tracef(
                  "%s t=%d w=0x%08x m=0x%08x n=%u copy=%zu ra=0x%08x rb=0x%08x",
                  ins.signature.c_str(), t, w, m, n, copy_bytes, ra_val,
                  rb_val);
              ++dma_trace;
            }
          }
        } else if (ins.signature == "time:r") {
          const uint32_t x = Pipeline::perf_counter_read_32(state);
          rf.write_reg(static_cast<uint8_t>(ins.rc), x);
        } else if (ins.signature == "time_cfg:rr" ||
                   ins.signature == "time_cfg:rrci") {
          const uint32_t y = Pipeline::perf_counter_config(state, rb);
          rf.write_reg(static_cast<uint8_t>(ins.rc), y);
          rf.ZF = (y == 0);
          if (ins.signature == "time_cfg:rrci" &&
              eval_cc_name(ins.cc_name, rf, y)) {
            next_pc = static_cast<uint32_t>(ins.pc);
          }
        } else if (ins.signature == "fault:i") {
          keep_running = false;
        } else {
          const char *warn_unsup =
              std::getenv("HOSTPIMSIM_UPMEM_WARN_UNSUPPORTED");
          if (warn_unsup && *warn_unsup && std::strcmp(warn_unsup, "0") != 0) {
            std::fprintf(
                stderr,
                "[hostpimsim-upmem-exec] launch48 unsupported signature: %s\n",
                ins.signature.c_str());
          }
          exec_tracef("launch48 unsupported signature: %s",
                      ins.signature.c_str());
          keep_running = false;
        }
      }

      rf.pc = next_pc;
      if (!keep_running) {
        state.set_sleeping(t);
      }

      if (replay_model) {
        rf.replay_pending_wmask = ins.replay_write_mask;
        replay_update_sources_48(ins, rf);
      }

      Pipeline::perf_counter_retire_step(state, state.is_replaying(t));
      ++steps;
    }
  }

  if (steps >= max_exec_steps && state.alive_count() > 0) {
    const char *dump_env = std::getenv("HOSTPIMSIM_UPMEM_DUMP_ON_STEP_CAP");
    if (dump_env && *dump_env && std::strcmp(dump_env, "0") != 0) {
      std::fprintf(stderr,
                   "[hostpimsim-upmem-cap] launch48 step cap hit: steps=%zu "
                   "max=%zu run_bits=0x%08llx active=%u\n",
                   steps, max_exec_steps,
                   static_cast<unsigned long long>(
                       state.run_bits & static_cast<uint64_t>(kTaskletMask)),
                   state.active_count());
      for (int t = 0; t < NUM_TASKLETS; ++t) {
        if (!state.is_running(t)) {
          continue;
        }
        const auto &rf = state.tasklets[t];
        const uint32_t pc = rf.pc;
        const char *sig = "<oob>";
        if (pc < program.size() && program[pc].valid) {
          sig = program[pc].signature.c_str();
        }
        std::fprintf(stderr,
                     "[hostpimsim-upmem-cap] t=%d pc=%u sig=%s zf=%d cf=%d "
                     "r0=0x%08x r1=0x%08x r2=0x%08x r3=0x%08x "
                     "r9=0x%08x r10=0x%08x r11=0x%08x r12=0x%08x "
                     "r13=0x%08x r14=0x%08x r15=0x%08x r16=0x%08x\n",
                     t, pc, sig, rf.ZF ? 1 : 0, rf.CF ? 1 : 0, rf.read_reg(0),
                     rf.read_reg(1), rf.read_reg(2), rf.read_reg(3),
                     rf.read_reg(9), rf.read_reg(10), rf.read_reg(11),
                     rf.read_reg(12), rf.read_reg(13), rf.read_reg(14),
                     rf.read_reg(15), rf.read_reg(16));
      }
    }
  }

  dpu.running_tasklets = static_cast<uint32_t>(
      state.run_bits & static_cast<uint64_t>(kTaskletMask));
  dpu.launch_pending = false;

  /* Keep dpulog reader stable even when runtime printf emulation is partial. */
  wram_store_u32(dpu, 0x10u, 0u, false);   // __stdout_buffer_state[0]
  wram_store_u32(dpu, 0x14u, 0u, false);   // __stdout_buffer_state[1]
  wram_store_u32(dpu, 0x10F0u, 0u, false); // __stdout_cache_write_pointer
  wram_store_u32(dpu, 0x10F4u, 0u, false); // __stdout_nr_of_writes

  uint32_t wram_result =
      read_wram_word(dpu, static_cast<uint32_t>(0x11D0u >> 2));
  exec_tracef("launch48 end: steps=%zu run_mask=0x%08x wram_result0=0x%08x",
              steps, dpu.running_tasklets, wram_result);

  if (profile_sig_enabled && !sig_counts.empty()) {
    std::vector<std::pair<std::string, size_t>> ordered;
    ordered.reserve(sig_counts.size());
    for (const auto &it : sig_counts) {
      ordered.emplace_back(it.first, it.second);
    }

    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
      if (a.second != b.second) {
        return a.second > b.second;
      }
      return a.first < b.first;
    });

    const size_t configured_limit = exec_profile_sig_limit();
    std::fprintf(
        stderr,
        "[hostpimsim-upmem-prof] launch48 signature profile (top %zu/%zu)\n",
        std::min<size_t>(ordered.size(), configured_limit), ordered.size());
    const size_t limit = std::min<size_t>(ordered.size(), configured_limit);
    for (size_t i = 0; i < limit; ++i) {
      std::fprintf(stderr, "[hostpimsim-upmem-prof] #%zu %s = %zu\n", i + 1,
                   ordered[i].first.c_str(), ordered[i].second);
    }
  }

  if (launch_timing) {
    const auto launch_t1 = launch_clock::now();
    const auto decode_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            launch_t_decode_done - launch_t0)
            .count();
    const auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              launch_t1 - launch_t0)
                              .count();
    const auto exec_us = total_us - decode_us;
    const double mips =
        (exec_us > 0)
            ? (static_cast<double>(steps) / static_cast<double>(exec_us))
            : 0.0;
    std::fprintf(stderr,
                 "[hostpimsim-upmem-time] launch48 insts=%zu steps=%zu "
                 "decode_us=%lld exec_us=%lld total_us=%lld mips=%.3f\n",
                 program.size(), steps, static_cast<long long>(decode_us),
                 static_cast<long long>(exec_us),
                 static_cast<long long>(total_us), mips);
  }

  return true;
}

void execute_launch_program(DpuState &dpu) {
  if (execute_launch_program_48(dpu)) {
    return;
  }

  exec_tracef("launch48 unavailable; clearing runnable mask");
  dpu.running_tasklets = 0u;
  dpu.launch_pending = false;
}

/* ---- merged from control.cc ---- */

static inline void ensure_thread_cmd_cache(upmem_runtime *rt) {
  if (!rt || rt->thread_cmd_cache_ready) {
    return;
  }

  constexpr uint64_t kBootBase = 0x7d8320000000ULL;
  constexpr uint64_t kResumeBase = 0x7d0320000000ULL;
  constexpr uint64_t kClearBase = 0x7c8320000000ULL;
  constexpr uint64_t kReadBase = 0x7c0330000000ULL;

  for (uint8_t tid = 0; tid < kNumTasklets; ++tid) {
    rt->thread_cmd_cache[0][tid] = thread_frame_command(kBootBase, tid);
    rt->thread_cmd_cache[1][tid] = thread_frame_command(kResumeBase, tid);
    rt->thread_cmd_cache[2][tid] = thread_frame_command(kClearBase, tid);
    rt->thread_cmd_cache[3][tid] = thread_frame_command(kReadBase, tid);
  }

  rt->thread_cmd_cache_ready = true;
}

/* MRAM::bind implementation is split into mram.cc. */

struct upmem_runtime *upmem_runtime_create(void) {
  auto *rt = new upmem_runtime();
  ensure_thread_cmd_cache(rt);
  upmem_runtime_reset(rt);
  return rt;
}

void upmem_runtime_destroy(struct upmem_runtime *rt) { delete rt; }

void upmem_runtime_reset(struct upmem_runtime *rt) {
  if (!rt) {
    return;
  }

  for (size_t ci = 0; ci < kNumCis; ++ci) {
    CI::reset_state(rt, ci);
  }

  for (auto &dpu : rt->dpus) {
    clear_dpu_state(dpu);
  }
}

void upmem_runtime_bind_mram(struct upmem_runtime *rt, size_t dpu_global_index,
                             void *mram_base, size_t mram_size) {
  MRAM::bind(rt, dpu_global_index, mram_base, mram_size);
}

/* ---- merged from ci.cc ---- */

void Pipeline::reset_ci(CiState &ci) {
  ci.selected_mask = 0xFFu;
  ci.group_masks.fill(0u);
  ci.group_masks[0] = 0x01u;

  ci.pc_mode = 0x04u;
  ci.dma_mux_status.fill(0x00u);
  for (auto &regs : ci.dma_ctrl_regs) {
    regs.fill(0x00u);
  }
  ci.dma_ctrl_read_register = 0x00u;
  ci.stack_up_mask = 0x00u;

  ci.structure = 0;
  ci.iram_write_structure_valid = false;
  ci.iram_write_addr_hi = 0;
  ci.wram_write_structure_valid = false;
  ci.wram_write_addr = 0;
}

static inline void mark_launch_complete(DpuState &dpu) {
  execute_launch_program(dpu);
}

/* CI reset/payload entry points are exposed through class CI (ci.cc). */

uint32_t Pipeline::payload_for_command(struct upmem_runtime *rt, size_t ci,
                                       uint64_t cmd_word,
                                       bool *needs_mask_fuzz) {
  if (needs_mask_fuzz) {
    *needs_mask_fuzz = false;
  }

  if (!rt || ci >= kNumCis) {
    return 0u;
  }

  auto &ci_state = rt->cis[ci];

  if (cmd_word == kCiIdentity) {
    return 0x00000001u;
  }

  if ((cmd_word & 0xFFFF00000000FF00ULL) == 0x01FF00000000FF00ULL) {
    Pipeline::reset_ci(ci_state);
    for (uint8_t dpu = 0; dpu < kNumDpusPerCi; ++dpu) {
      clear_dpu_state(rt->dpus[dpu_index(ci, dpu)]);
    }
    return 0x00000000u;
  }

  if (cmd_word == kCiByteOrder) {
    return 0x0F8FCFEFu;
  }

  const uint8_t opcode = b(cmd_word, 56);
  const uint8_t tag = b(cmd_word, 48);

  if (opcode == 0x11u) {
    ci_state.structure = cmd_word;

    uint16_t addr_hi = 0;
    ci_state.iram_write_structure_valid =
        decode_iram_write_structure(cmd_word, addr_hi);
    ci_state.iram_write_addr_hi = addr_hi;

    uint16_t wram_addr = 0;
    ci_state.wram_write_structure_valid =
        decode_wram_write_word_structure(cmd_word, wram_addr);
    ci_state.wram_write_addr = wram_addr;

    return 0x000000FFu;
  }

  if (opcode != 0x33u) {
    return 0u;
  }

  const uint8_t b0 = b(cmd_word, 0);
  const uint8_t b1 = b(cmd_word, 8);
  const uint8_t b2 = b(cmd_word, 16);
  const uint8_t b3 = b(cmd_word, 24);
  const uint8_t b4 = b(cmd_word, 32);
  const uint8_t b5 = b(cmd_word, 40);

  /* IRAM write frame under CI_IRAM_WRITE_INSTRUCTION_STRUCT */
  if (ci_state.iram_write_structure_valid) {
    if (needs_mask_fuzz) {
      *needs_mask_fuzz = true;
    }
    const uint16_t addr = static_cast<uint16_t>(
        (static_cast<uint16_t>(ci_state.iram_write_addr_hi) << 8) | tag);

    uint64_t data = 0;
    data |= static_cast<uint64_t>(b0) << 0;
    data |= static_cast<uint64_t>(b1) << 8;
    data |= static_cast<uint64_t>(b2) << 16;
    data |= static_cast<uint64_t>(b3) << 24;
    data |= static_cast<uint64_t>(b4) << 32;
    data |= static_cast<uint64_t>(b5) << 40;

    for (uint8_t dpu = 0; dpu < kNumDpusPerCi; ++dpu) {
      const uint8_t dpu_bit = static_cast<uint8_t>(1u << dpu);
      if ((ci_state.selected_mask & dpu_bit) == 0) {
        continue;
      }
      auto &state = rt->dpus[dpu_index(ci, dpu)];
      write_iram_word(state, addr, data);
    }

    return ci_state.selected_mask;
  }

  /* WRAM write frame under CI_WRAM_WRITE_WORD_STRUCT */
  if (ci_state.wram_write_structure_valid) {
    if (needs_mask_fuzz) {
      *needs_mask_fuzz = true;
    }
    uint16_t addr = ci_state.wram_write_addr;
    const uint32_t data = Pipeline::low_u32(cmd_word);

    uint8_t low5 = 0;
    if (decode_wram_write_word_frame_low5(cmd_word, low5)) {
      addr = static_cast<uint16_t>((addr & static_cast<uint16_t>(~0x1Fu)) |
                                   static_cast<uint16_t>(low5));
    }

    if (exec_trace_enabled()) {
      static size_t wram_write_trace = 0;
      if (wram_write_trace < 512) {
        exec_tracef(
            "wram_write ci=%zu addr=0x%04x data=0x%08x sel=0x%02x tag=0x%02x",
            ci, static_cast<unsigned>(addr), data,
            static_cast<unsigned>(ci_state.selected_mask),
            static_cast<unsigned>(tag));
        ++wram_write_trace;
      }
    }

    for (uint8_t dpu = 0; dpu < kNumDpusPerCi; ++dpu) {
      const uint8_t dpu_bit = static_cast<uint8_t>(1u << dpu);
      if ((ci_state.selected_mask & dpu_bit) == 0) {
        continue;
      }
      auto &state = rt->dpus[dpu_index(ci, dpu)];
      write_wram_word(state, addr, data);
    }

    return ci_state.selected_mask;
  }

  /* Selection/group commands */
  if (tag == 0x00u) {
    if (b0 == 0x08u && b1 == 0xFFu) {
      ci_state.selected_mask = 0xFFu;
      return ci_state.selected_mask;
    }
    if (b0 == 0x0Au) {
      const uint8_t dpu_id = static_cast<uint8_t>(b1 & 0x7u);
      ci_state.selected_mask = static_cast<uint8_t>(1u << dpu_id);
      return ci_state.selected_mask;
    }
    if (b0 == 0x09u) {
      const uint8_t group_id = static_cast<uint8_t>(b1 & 0x7u);
      ci_state.selected_mask = ci_state.group_masks[group_id];
      return ci_state.selected_mask;
    }
    if (b0 == 0x0Cu) {
      const uint8_t group_id = static_cast<uint8_t>(b1 & 0x7u);
      ci_state.group_masks[group_id] = ci_state.selected_mask;
      return ci_state.selected_mask;
    }

    if (b0 == 0x16u && b1 == 0x02u) {
      return ci_state.pc_mode;
    }
    if (b0 == 0x10u && b1 == 0x02u) {
      const uint8_t selected = first_selected_dpu(ci_state.selected_mask);
      if (selected >= kNumDpusPerCi) {
        return 0u;
      }

      if (ci_state.dma_ctrl_read_register == 0x02u) {
        return ci_state.dma_mux_status[selected];
      }

      return ci_state.dma_ctrl_regs[selected][ci_state.dma_ctrl_read_register];
    }
    if (b0 == 0x84u && b1 == 0x02u) {
      uint8_t run_mask = 0;
      for (uint8_t dpu = 0; dpu < kNumDpusPerCi; ++dpu) {
        auto &state = rt->dpus[dpu_index(ci, dpu)];
        if (state.launch_pending) {
          if (!state.mram_base && rt->fallback_mram_base) {
            state.mram_base = rt->fallback_mram_base;
            state.mram_size = rt->fallback_mram_size;
          }
          mark_launch_complete(state);
        }
        if (state.running_tasklets != 0) {
          run_mask |= static_cast<uint8_t>(1u << dpu);
        }
      }
      return run_mask;
    }

    if (b0 == 0xF2u && b1 == 0x02u) {
      const uint8_t previous = ci_state.stack_up_mask;
      ci_state.stack_up_mask = 0xFFu;
      return previous;
    }
    if (b0 == 0xF0u && b1 == 0x02u) {
      const uint8_t previous = ci_state.stack_up_mask;
      ci_state.stack_up_mask = 0x00u;
      return previous;
    }

    if (b1 == 0x02u) {
      switch (b0) {
      case 0x80u: /* CI_DPU_FAULT_STATE_READ_FRAME */
      case 0x82u: /* CI_DMA_FAULT_READ_AND_CLR_FRAME */
      case 0xB0u: /* CI_BKP_FAULT_READ_FRAME */
      case 0xB4u: /* CI_POISON_FAULT_READ_FRAME */
      case 0xF4u: /* CI_MEM_FAULT_READ_AND_CLR_FRAME */
        return 0u;
      default:
        break;
      }

      /* Generic CI read-style frame fallback. */
      return ci_state.selected_mask;
    }
  }

  /* DMA control */
  if (tag == 0xA7u) {
    /* CI_DMA_CTRL_WRITE_FRAME nibble-encoded address/data */
    if (((b0 & 0xF0u) == 0x60u) && ((b1 & 0xF0u) == 0x60u) &&
        ((b2 & 0xF0u) == 0x60u) && ((b3 & 0xF0u) == 0x60u)) {
      const uint8_t address =
          static_cast<uint8_t>(((b0 & 0x0Fu) << 4) | (b1 & 0x0Fu));
      const uint8_t data =
          static_cast<uint8_t>(((b2 & 0x0Fu) << 4) | (b3 & 0x0Fu));

      if (address == 0xFFu) {
        ci_state.dma_ctrl_read_register = data;
      }

      for (uint8_t dpu = 0; dpu < kNumDpusPerCi; ++dpu) {
        const uint8_t dpu_bit = static_cast<uint8_t>(1u << dpu);
        if ((ci_state.selected_mask & dpu_bit) == 0) {
          continue;
        }

        ci_state.dma_ctrl_regs[dpu][address] = data;

        if (address == 0x80u || address == 0x82u || address == 0x84u) {
          ci_state.dma_mux_status[dpu] = (data == 0u) ? 0x00u : 0x03u;
        }
      }

      return 0x000000FFu;
    }

    if (needs_mask_fuzz) {
      *needs_mask_fuzz = true;
    }
    return ci_state.selected_mask;
  }

  if (tag == 0xA6u) {
    /* CI_DMA_CTRL_CLEAR_FRAME: pipeline flush, not a state reset. */
    if (b0 == 0 && b1 == 0 && b2 == 0 && b3 == 0 && b4 == 0 && b5 == 0) {
      return 0x000000FFu;
    }

    if (needs_mask_fuzz) {
      *needs_mask_fuzz = true;
    }
    return ci_state.selected_mask;
  }

  if (tag == 0xA4u) {
    /* CI_PC_MODE_WRITE_FRAME */
    if (b1 == 0x00u && b2 == 0x00u && b3 == 0x00u && b4 == 0x00u &&
        b5 == 0x00u) {
      ci_state.pc_mode = b0;
      return 0x000000FFu;
    }

    if (needs_mask_fuzz) {
      *needs_mask_fuzz = true;
    }
    return ci_state.selected_mask;
  }

  /* Thread commands (BOOT / RESUME / CLR / READ_RUN). */
  ThreadCmdKind kind{};
  uint8_t thread_id = 0;
  if (decode_thread_command(rt, cmd_word, kind, thread_id)) {
    const uint32_t thread_bit =
        (thread_id < 32u) ? (1u << static_cast<uint32_t>(thread_id)) : 0u;
    uint8_t payload = 0;

    for (uint8_t dpu = 0; dpu < kNumDpusPerCi; ++dpu) {
      const uint8_t dpu_bit = static_cast<uint8_t>(1u << dpu);
      if ((ci_state.selected_mask & dpu_bit) == 0) {
        continue;
      }

      auto &state = rt->dpus[dpu_index(ci, dpu)];
      if (!MRAM::is_bound(state)) {
        continue;
      }

      const bool was_running = (state.running_tasklets & thread_bit) != 0u;
      if (was_running) {
        payload |= dpu_bit;
      }

      switch (kind) {
      case ThreadCmdKind::Boot:
      case ThreadCmdKind::Resume:
        state.running_tasklets |= thread_bit;
        if (thread_id == 0u) {
          state.launch_pending = true;
        }
        break;
      case ThreadCmdKind::ClearRun:
        state.running_tasklets &= ~thread_bit;
        break;
      case ThreadCmdKind::ReadRun:
        break;
      }
    }

    return payload;
  }

  /* WRAM read frame. */
  if (tag == 0x99u) {
    uint32_t word_addr = 0;
    if (decode_wram_read_word_frame(cmd_word, word_addr)) {
      const uint8_t selected = first_selected_dpu(ci_state.selected_mask);
      if (selected >= kNumDpusPerCi) {
        return 0u;
      }

      const auto &state = rt->dpus[dpu_index(ci, selected)];
      return read_wram_word(state, word_addr);
    }
  }

  /* Conservative fallback: treat as write-like / collision-prone frame. */
  if (needs_mask_fuzz) {
    *needs_mask_fuzz = true;
  }
  return ci_state.selected_mask;
}

uint32_t upmem_pipeline_payload_for_command(struct upmem_runtime *rt, size_t ci,
                                            uint64_t cmd_word,
                                            bool *needs_mask_fuzz) {
  return Pipeline::payload_for_command(rt, ci, cmd_word, needs_mask_fuzz);
}

uint32_t upmem_pipeline_run_state_for_dpu(struct upmem_runtime *rt, size_t ci,
                                          uint8_t dpu_local) {
  if (!rt || ci >= kNumCis || dpu_local >= kNumDpusPerCi) {
    return 0u;
  }

  auto &state = rt->dpus[dpu_index(ci, dpu_local)];
  if (state.launch_pending) {
    if (!state.mram_base && rt->fallback_mram_base) {
      state.mram_base = rt->fallback_mram_base;
      state.mram_size = rt->fallback_mram_size;
    }
    mark_launch_complete(state);
  }

  if (state.running_tasklets == 0u) {
    return 0u;
  }
  return static_cast<uint32_t>(1u << dpu_local);
}

/* IRAM/WRAM implementations are split into iram.cc / wram.cc. */
