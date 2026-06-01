#include "dpu.hh"

#include <limits>

static constexpr size_t kDecodedInstCacheLineCount = 4096u;

/* Simplified opcode map aligned to the backup runtime operation families. */
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

static inline int32_t sign_extend16(uint16_t value) {
  return static_cast<int32_t>(static_cast<int16_t>(value));
}

static inline int32_t sign_extend24(uint32_t value) {
  const uint32_t masked = value & 0x00FFFFFFu;
  if ((masked & 0x00800000u) != 0u) {
    return static_cast<int32_t>(masked | 0xFF000000u);
  }
  return static_cast<int32_t>(masked);
}

static inline int32_t sign_extend11(uint32_t value) {
  const uint32_t masked = value & 0x7FFu;
  if ((masked & 0x400u) != 0u) {
    return static_cast<int32_t>(masked | 0xFFFFF800u);
  }
  return static_cast<int32_t>(masked);
}

static inline uint8_t atomic_bit_index(uint32_t value) {
  return static_cast<uint8_t>(value & 0xFFu);
}

static inline uint32_t rotl32(uint32_t value, uint32_t shift) {
  const uint32_t s = shift & 31u;
  if (s == 0u) {
    return value;
  }
  return static_cast<uint32_t>((value << s) | (value >> (32u - s)));
}

static inline uint32_t rotr32(uint32_t value, uint32_t shift) {
  const uint32_t s = shift & 31u;
  if (s == 0u) {
    return value;
  }
  return static_cast<uint32_t>((value >> s) | (value << (32u - s)));
}

static inline uint32_t count_leading_ones_u32(uint32_t value) {
  return (value == 0xFFFFFFFFu) ? 32u
                                : static_cast<uint32_t>(__builtin_clz(~value));
}

static inline uint32_t count_leading_sign_u32(uint32_t value) {
  if (value == 0u || value == 0xFFFFFFFFu) {
    return 31u;
  }

  const uint32_t leading = ((value & 0x80000000u) != 0u)
                               ? static_cast<uint32_t>(__builtin_clz(~value))
                               : static_cast<uint32_t>(__builtin_clz(value));
  return (leading == 0u) ? 0u : (leading - 1u);
}

static inline uint32_t saturating_add_s32(uint32_t lhs, uint32_t rhs) {
  const int64_t a = static_cast<int64_t>(static_cast<int32_t>(lhs));
  const int64_t b = static_cast<int64_t>(static_cast<int32_t>(rhs));
  const int64_t sum = a + b;

  const int64_t min_v =
      static_cast<int64_t>(std::numeric_limits<int32_t>::min());
  const int64_t max_v =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max());
  if (sum < min_v) {
    return static_cast<uint32_t>(std::numeric_limits<int32_t>::min());
  }
  if (sum > max_v) {
    return static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
  }
  return static_cast<uint32_t>(static_cast<int32_t>(sum));
}

/*
 * Hardware-style PERF counter readout.
 *
 * The raw counter is modeled with low 4 bits reserved; visible value is
 * `raw >> 4` to match expected register semantics.
 */
uint32_t Pipeline::perf_counter_read32() const {
  return static_cast<uint32_t>(register_file_.perf_counter_raw_ >> 4u);
}

/* PERF mode encoding used by `perf_counter_config()`. */
static constexpr uint8_t kPerfModeSame = 0;
static constexpr uint8_t kPerfModeCycles = 1;
static constexpr uint8_t kPerfModeInstructions = 2;
static constexpr uint8_t kPerfModeNothing = 3;

/*
 * Configure the performance counter.
 *
 * config_bits layout (packed 3 bits):
 *   bit0   : reset request
 *   bits2:1: mode (same/cycles/instructions/nothing)
 *
 * Returns previous visible counter value.
 */
uint32_t Pipeline::perf_counter_config(uint32_t config_bits) {
  const uint32_t old = perf_counter_read32();
  const uint8_t packed = static_cast<uint8_t>(config_bits & 0x7u);
  const bool should_reset = (packed & 0x1u) != 0;
  const uint8_t mode_bits = static_cast<uint8_t>((packed >> 1) & 0x3u);

  if (mode_bits == kPerfModeSame) {
    // Keep current mode unchanged.
  } else {
    register_file_.perf_counter_mode_ = mode_bits;
  }

  if (should_reset) {
    register_file_.perf_counter_raw_ =
        (register_file_.perf_counter_mode_ == kPerfModeNothing)
            ? 0u
            : static_cast<uint32_t>(~0u);
  }

  return old;
}

/*
 * Retire one simulated pipeline step for PERF accounting.
 *
 * - Cycles mode: increment every step.
 * - Instructions mode: increment only non-replay retirement.
 */
void Pipeline::perf_counter_retire_step(bool is_replay_step) {
  if (register_file_.perf_counter_mode_ == kPerfModeCycles) {
    register_file_.perf_counter_raw_ += 1u;
    return;
  }

  if (register_file_.perf_counter_mode_ == kPerfModeInstructions &&
      !is_replay_step) {
    register_file_.perf_counter_raw_ += 1u;
  }
}

/* Utility: lower 32 bits extraction helper. */
uint32_t Pipeline::low_u32(uint64_t value) {
  return static_cast<uint32_t>(value & 0xFFFFFFFFULL);
}

/* Utility: byte extraction helper with explicit bit shift. */
uint8_t Pipeline::byte(uint64_t value, unsigned shift) {
  return static_cast<uint8_t>((value >> shift) & 0xFFu);
}

/* Translate scalar register index to D-register slot index. */
uint8_t Pipeline::dreg_slot(int reg_num) {
  if (reg_num < 0) {
    return 0;
  }

  /*
   * DREG mapping divisor.
   *
   * UPMEM-style model maps pairs of 32-bit scalar registers into one 64-bit
   * D-register slot, so slot = reg_num / 2.
   */
  constexpr uint8_t kDregDivisor = 2;
  return static_cast<uint8_t>(reg_num / kDregDivisor);
}

struct RawDecoded48 {
  bool valid = false;
  enum class HotOp : uint8_t {
    None = 0,
    Acquire,
    Release,
    Or,
    OrS,
    OrU,
    Add,
    Addc,
    And,
    AndS,
    AndU,
    Sub,
    Subc,
    MulStep,
    DivStep,
    Resume,
    Boot,
    Call,
    Ldma,
    Sdma,
    Lsl,
    LslS,
    LslU,
    Lslx,
    LslxS,
    LslxU,
    Lsr,
    LsrS,
    LsrU,
    Lsrx,
    LsrxS,
    LsrxU,
    Movd,
    Swapd,
    Clz,
    Clo,
    MulUlUl,
    MulUlUh,
    MulUhUl,
    MulUhUh,
    LslAdd,
    LslAddS,
    LslAddU,
    LslSub,
    LslSubS,
    LslSubU,
    LsrAdd,
    LsrAddS,
    LsrAddU,
    Lbu,
    Lbs,
    Lw,
    LwS,
    Ld,
    Sw,
    Sb,
    Sh,
    Sd,
    Time,
    TimeCfg,
    Stop,
    Nop,
    Fault,
  } hot_op{HotOp::None};
  enum class HotFormat : uint8_t {
    None = 0,
    Ci,
    Erii,
    Erir,
    Erri,
    Ersi,
    Esii,
    Esir,
    I,
    R,
    Rci,
    Rici,
    Rir,
    Rirc,
    Rirci,
    Rirf,
    Rki,
    Rr,
    Rrc,
    Rrci,
    Rri,
    Rric,
    Rrici,
    Rrif,
    Rrr,
    Rrrc,
    Rrrci,
    Rrri,
    Rrrici,
    Ssi,
    Sss,
    Z,
    Zci,
    Zir,
    Zirc,
    Zirci,
    Zirf,
    Zr,
    Zrc,
    Zrci,
    Zri,
    Zric,
    Zrici,
    Zrif,
    Zrr,
    Zrrc,
    Zrrci,
    Zrri,
    Zrrici,
  } hot_fmt{HotFormat::None};
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
  bool off_uses_sign_extend24 = false;
  enum class CcKind : uint8_t {
    None = 0,
    False,
    True,
    Zero,
    NonZero,
    Carry,
    NoCarry,
    LessEqualUnsigned,
    GreaterUnsigned,
    LessSigned,
    LessEqualSigned,
    GreaterSigned,
    GreaterEqualSigned,
    XZero,
    XNonZero,
    Max,
    NotMax,
    Even,
    Odd,
    XLessEqualUnsigned,
    XGreaterUnsigned,
    XLessEqualSigned,
    XGreaterSigned,
    SubwordNoCarry,
    Unknown,
  } cc_kind{CcKind::None};
  uint8_t cc_subword_bits = 0;
  enum class WritebackMode : uint8_t {
    None = 0,
    DregSign32,
    DregZero32,
  } writeback_mode{WritebackMode::None};
  enum class MulBytePairMode : uint8_t {
    None = 0,
    UlUl,
    UlUh,
    UhUl,
    UhUh,
  } mul_byte_pair_mode{MulBytePairMode::None};
  enum class StoreSourceMode : uint8_t {
    None = 0,
    Imm32,
    Reg32,
    Dreg64,
  } store_source_mode{StoreSourceMode::None};
  bool hot_supported = false;
  bool imm_is_signed8 = false;
  bool imm_passthrough = false;
  bool suppress_scalar_write = false;
  bool acquire_is_release = false;
  bool dma_to_mram = false;
  bool movd_is_swap = false;
  bool clz_invert_input = false;
  bool time_cfg_uses_ra = false;
  bool sub_lhs_is_imm = false;
  bool sub_rhs_is_reg = false;
  bool sub_is_rrrc = false;
  bool sub_is_zrrc = false;
  bool subc_lhs_is_imm = false;
  bool subc_rhs_is_reg = false;
  bool subc_is_rrrc = false;
  bool subc_is_zrrc = false;
  enum class CcMapKind : uint8_t {
    None = 0,
    Map0,
    Map1,
    Map2,
    Map3,
    Map4,
    Map5,
    Map6,
    Map7,
    Map8,
    Map9,
    Map10,
    Map11,
    Map13,
    Map14,
    Map15,
  } cc_map_kind{CcMapKind::None};
};

static inline void set_hot_metadata(RawDecoded48 &decoded,
                                    RawDecoded48::HotOp op,
                                    RawDecoded48::HotFormat fmt) {
  decoded.hot_op = op;
  decoded.hot_fmt = fmt;
}

static inline void decode_cc_map_0_metadata(int cc,
                                            RawDecoded48::CcKind &cc_kind,
                                            uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 2:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 3:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_1_metadata(int cc,
                                            RawDecoded48::CcKind &cc_kind,
                                            uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 6:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 7:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 10:
    cc_kind = RawDecoded48::CcKind::XZero;
    return;
  case 11:
    cc_kind = RawDecoded48::CcKind::XNonZero;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_2_metadata(int cc,
                                            RawDecoded48::CcKind &cc_kind,
                                            uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 2:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 3:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 4:
    cc_kind = RawDecoded48::CcKind::XZero;
    return;
  case 5:
    cc_kind = RawDecoded48::CcKind::XNonZero;
    return;
  case 8:
  case 14:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 9:
  case 15:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  case 12:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 13:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 18:
  case 19:
    cc_kind = RawDecoded48::CcKind::Unknown;
    return;
  case 20:
    cc_kind = RawDecoded48::CcKind::Carry;
    return;
  case 21:
    cc_kind = RawDecoded48::CcKind::NoCarry;
    return;
  case 22:
    cc_kind = RawDecoded48::CcKind::SubwordNoCarry;
    cc_subword_bits = 5;
    return;
  case 23:
    cc_kind = RawDecoded48::CcKind::SubwordNoCarry;
    cc_subword_bits = 6;
    return;
  case 24:
    cc_kind = RawDecoded48::CcKind::SubwordNoCarry;
    cc_subword_bits = 7;
    return;
  case 25:
    cc_kind = RawDecoded48::CcKind::SubwordNoCarry;
    cc_subword_bits = 8;
    return;
  case 26:
    cc_kind = RawDecoded48::CcKind::SubwordNoCarry;
    cc_subword_bits = 9;
    return;
  case 27:
    cc_kind = RawDecoded48::CcKind::SubwordNoCarry;
    cc_subword_bits = 10;
    return;
  case 28:
    cc_kind = RawDecoded48::CcKind::SubwordNoCarry;
    cc_subword_bits = 11;
    return;
  case 29:
    cc_kind = RawDecoded48::CcKind::SubwordNoCarry;
    cc_subword_bits = 12;
    return;
  case 30:
    cc_kind = RawDecoded48::CcKind::SubwordNoCarry;
    cc_subword_bits = 13;
    return;
  case 31:
    cc_kind = RawDecoded48::CcKind::SubwordNoCarry;
    cc_subword_bits = 14;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_3_metadata(int cc,
                                            RawDecoded48::CcKind &cc_kind,
                                            uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  if (cc == 0) {
    cc_kind = RawDecoded48::CcKind::False;
  }
}

static inline void decode_cc_map_4_metadata(int cc,
                                            RawDecoded48::CcKind &cc_kind,
                                            uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 2:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 3:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 4:
    cc_kind = RawDecoded48::CcKind::XZero;
    return;
  case 5:
    cc_kind = RawDecoded48::CcKind::XNonZero;
    return;
  case 8:
  case 14:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 9:
  case 15:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  case 12:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 13:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_5_metadata(int cc,
                                            RawDecoded48::CcKind &cc_kind,
                                            uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 2:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 3:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 4:
    cc_kind = RawDecoded48::CcKind::XZero;
    return;
  case 5:
    cc_kind = RawDecoded48::CcKind::XNonZero;
    return;
  case 8:
  case 14:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 9:
  case 15:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  case 12:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 13:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 24:
  case 30:
    cc_kind = RawDecoded48::CcKind::Even;
    return;
  case 25:
  case 31:
    cc_kind = RawDecoded48::CcKind::Odd;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_6_metadata(int cc,
                                            RawDecoded48::CcKind &cc_kind,
                                            uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 2:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 3:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 4:
    cc_kind = RawDecoded48::CcKind::XZero;
    return;
  case 5:
    cc_kind = RawDecoded48::CcKind::XNonZero;
    return;
  case 8:
  case 14:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 9:
  case 15:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  case 12:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 13:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 24:
  case 30:
    cc_kind = RawDecoded48::CcKind::Even;
    return;
  case 25:
  case 31:
    cc_kind = RawDecoded48::CcKind::Odd;
    return;
  case 28:
  case 29:
    cc_kind = RawDecoded48::CcKind::Unknown;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_7_metadata(int cc,
                                            RawDecoded48::CcKind &cc_kind,
                                            uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 0:
    cc_kind = RawDecoded48::CcKind::False;
    return;
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 2:
  case 12:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 3:
  case 13:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 4:
    cc_kind = RawDecoded48::CcKind::XZero;
    return;
  case 5:
    cc_kind = RawDecoded48::CcKind::XNonZero;
    return;
  case 14:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 15:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_8_metadata(int cc,
                                            RawDecoded48::CcKind &cc_kind,
                                            uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 2:
  case 12:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 3:
  case 13:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 4:
    cc_kind = RawDecoded48::CcKind::XZero;
    return;
  case 5:
    cc_kind = RawDecoded48::CcKind::XNonZero;
    return;
  case 8:
    cc_kind = RawDecoded48::CcKind::Max;
    return;
  case 9:
    cc_kind = RawDecoded48::CcKind::NotMax;
    return;
  case 14:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 15:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_9_metadata(int cc,
                                            RawDecoded48::CcKind &cc_kind,
                                            uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 0:
    cc_kind = RawDecoded48::CcKind::False;
    return;
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 12:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 13:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 14:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 15:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_10_metadata(int cc,
                                             RawDecoded48::CcKind &cc_kind,
                                             uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 12:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 13:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 14:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 15:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_11_metadata(int cc,
                                             RawDecoded48::CcKind &cc_kind,
                                             uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 0:
    cc_kind = RawDecoded48::CcKind::False;
    return;
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_13_metadata(int cc,
                                             RawDecoded48::CcKind &cc_kind,
                                             uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 1:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 2:
  case 12:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 3:
  case 13:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 4:
    cc_kind = RawDecoded48::CcKind::XZero;
    return;
  case 5:
    cc_kind = RawDecoded48::CcKind::XNonZero;
    return;
  case 8:
  case 14:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 9:
  case 15:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  case 18:
  case 19:
    cc_kind = RawDecoded48::CcKind::Unknown;
    return;
  case 20:
    cc_kind = RawDecoded48::CcKind::Carry;
    return;
  case 21:
    cc_kind = RawDecoded48::CcKind::NoCarry;
    return;
  case 22:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  case 23:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 24:
    cc_kind = RawDecoded48::CcKind::LessEqualSigned;
    return;
  case 25:
    cc_kind = RawDecoded48::CcKind::GreaterSigned;
    return;
  case 26:
    cc_kind = RawDecoded48::CcKind::LessEqualUnsigned;
    return;
  case 27:
    cc_kind = RawDecoded48::CcKind::GreaterUnsigned;
    return;
  case 28:
    cc_kind = RawDecoded48::CcKind::XLessEqualSigned;
    return;
  case 29:
    cc_kind = RawDecoded48::CcKind::XGreaterSigned;
    return;
  case 30:
    cc_kind = RawDecoded48::CcKind::XLessEqualUnsigned;
    return;
  case 31:
    cc_kind = RawDecoded48::CcKind::XGreaterUnsigned;
    return;
  default:
    return;
  }
}

static inline void decode_cc_map_14_metadata(int cc,
                                             RawDecoded48::CcKind &cc_kind,
                                             uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  if (cc == 0) {
    cc_kind = RawDecoded48::CcKind::NonZero;
  }
}

static inline void decode_cc_map_15_metadata(int cc,
                                             RawDecoded48::CcKind &cc_kind,
                                             uint8_t &cc_subword_bits) {
  cc_kind = RawDecoded48::CcKind::False;
  cc_subword_bits = 0;

  switch (cc) {
  case 6:
  case 34:
  case 44:
    cc_kind = RawDecoded48::CcKind::Zero;
    return;
  case 7:
  case 35:
  case 45:
    cc_kind = RawDecoded48::CcKind::NonZero;
    return;
  case 10:
  case 36:
    cc_kind = RawDecoded48::CcKind::XZero;
    return;
  case 11:
  case 37:
    cc_kind = RawDecoded48::CcKind::XNonZero;
    return;
  case 33:
    cc_kind = RawDecoded48::CcKind::True;
    return;
  case 40:
  case 46:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 41:
  case 47:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  case 50:
  case 51:
    cc_kind = RawDecoded48::CcKind::Unknown;
    return;
  case 52:
    cc_kind = RawDecoded48::CcKind::Carry;
    return;
  case 53:
    cc_kind = RawDecoded48::CcKind::NoCarry;
    return;
  case 54:
    cc_kind = RawDecoded48::CcKind::LessSigned;
    return;
  case 55:
    cc_kind = RawDecoded48::CcKind::GreaterEqualSigned;
    return;
  case 56:
    cc_kind = RawDecoded48::CcKind::LessEqualSigned;
    return;
  case 57:
    cc_kind = RawDecoded48::CcKind::GreaterSigned;
    return;
  case 58:
    cc_kind = RawDecoded48::CcKind::LessEqualUnsigned;
    return;
  case 59:
    cc_kind = RawDecoded48::CcKind::GreaterUnsigned;
    return;
  case 60:
    cc_kind = RawDecoded48::CcKind::XLessEqualSigned;
    return;
  case 61:
    cc_kind = RawDecoded48::CcKind::XGreaterSigned;
    return;
  case 62:
    cc_kind = RawDecoded48::CcKind::XLessEqualUnsigned;
    return;
  case 63:
    cc_kind = RawDecoded48::CcKind::XGreaterUnsigned;
    return;
  default:
    return;
  }
}

static inline RawDecoded48::CcMapKind
select_cc_map_48(RawDecoded48::HotOp op, RawDecoded48::HotFormat fmt) {
  switch (op) {
  case RawDecoded48::HotOp::Acquire:
    return (fmt == RawDecoded48::HotFormat::Rici)
               ? RawDecoded48::CcMapKind::Map0
               : RawDecoded48::CcMapKind::None;
  case RawDecoded48::HotOp::Release:
    return (fmt == RawDecoded48::HotFormat::Rici)
               ? RawDecoded48::CcMapKind::Map14
               : RawDecoded48::CcMapKind::None;
  case RawDecoded48::HotOp::Add:
  case RawDecoded48::HotOp::Addc:
    switch (fmt) {
    case RawDecoded48::HotFormat::Rric:
    case RawDecoded48::HotFormat::Rrrc:
    case RawDecoded48::HotFormat::Zric:
    case RawDecoded48::HotFormat::Zrrc:
      return RawDecoded48::CcMapKind::Map1;
    case RawDecoded48::HotFormat::Rrici:
    case RawDecoded48::HotFormat::Rrrci:
    case RawDecoded48::HotFormat::Zrici:
    case RawDecoded48::HotFormat::Zrrci:
      return RawDecoded48::CcMapKind::Map2;
    case RawDecoded48::HotFormat::Rrif:
    case RawDecoded48::HotFormat::Zrif:
      return RawDecoded48::CcMapKind::Map3;
    default:
      return RawDecoded48::CcMapKind::None;
    }
  case RawDecoded48::HotOp::Or:
  case RawDecoded48::HotOp::OrS:
  case RawDecoded48::HotOp::OrU:
  case RawDecoded48::HotOp::And:
  case RawDecoded48::HotOp::AndS:
  case RawDecoded48::HotOp::AndU:
    switch (fmt) {
    case RawDecoded48::HotFormat::Rric:
    case RawDecoded48::HotFormat::Rrrc:
    case RawDecoded48::HotFormat::Zric:
    case RawDecoded48::HotFormat::Zrrc:
      return RawDecoded48::CcMapKind::Map1;
    case RawDecoded48::HotFormat::Rrici:
    case RawDecoded48::HotFormat::Rrrci:
    case RawDecoded48::HotFormat::Zrici:
    case RawDecoded48::HotFormat::Zrrci:
      return RawDecoded48::CcMapKind::Map4;
    case RawDecoded48::HotFormat::Rrif:
    case RawDecoded48::HotFormat::Zrif:
      return RawDecoded48::CcMapKind::Map3;
    default:
      return RawDecoded48::CcMapKind::None;
    }
  case RawDecoded48::HotOp::Sub:
  case RawDecoded48::HotOp::Subc:
    switch (fmt) {
    case RawDecoded48::HotFormat::Rirc:
    case RawDecoded48::HotFormat::Zirc:
      return RawDecoded48::CcMapKind::Map1;
    case RawDecoded48::HotFormat::Rirci:
    case RawDecoded48::HotFormat::Rrici:
    case RawDecoded48::HotFormat::Rrrci:
    case RawDecoded48::HotFormat::Zirci:
    case RawDecoded48::HotFormat::Zrici:
    case RawDecoded48::HotFormat::Zrrci:
      return RawDecoded48::CcMapKind::Map13;
    case RawDecoded48::HotFormat::Rirf:
    case RawDecoded48::HotFormat::Rrif:
    case RawDecoded48::HotFormat::Zirf:
    case RawDecoded48::HotFormat::Zrif:
      return RawDecoded48::CcMapKind::Map3;
    case RawDecoded48::HotFormat::Rric:
    case RawDecoded48::HotFormat::Rrrc:
    case RawDecoded48::HotFormat::Zric:
    case RawDecoded48::HotFormat::Zrrc:
      return RawDecoded48::CcMapKind::Map15;
    default:
      return RawDecoded48::CcMapKind::None;
    }
  case RawDecoded48::HotOp::Boot:
  case RawDecoded48::HotOp::Resume:
    return (fmt == RawDecoded48::HotFormat::Rici)
               ? RawDecoded48::CcMapKind::Map7
               : RawDecoded48::CcMapKind::None;
  case RawDecoded48::HotOp::Stop:
    return (fmt == RawDecoded48::HotFormat::Ci) ? RawDecoded48::CcMapKind::Map7
                                                : RawDecoded48::CcMapKind::None;
  case RawDecoded48::HotOp::Lsl:
  case RawDecoded48::HotOp::LslS:
  case RawDecoded48::HotOp::LslU:
  case RawDecoded48::HotOp::Lslx:
  case RawDecoded48::HotOp::LslxS:
  case RawDecoded48::HotOp::LslxU:
  case RawDecoded48::HotOp::Lsr:
  case RawDecoded48::HotOp::LsrS:
  case RawDecoded48::HotOp::LsrU:
  case RawDecoded48::HotOp::Lsrx:
  case RawDecoded48::HotOp::LsrxS:
  case RawDecoded48::HotOp::LsrxU:
    switch (fmt) {
    case RawDecoded48::HotFormat::Rric:
    case RawDecoded48::HotFormat::Rrrc:
    case RawDecoded48::HotFormat::Zric:
    case RawDecoded48::HotFormat::Zrrc:
      return RawDecoded48::CcMapKind::Map1;
    case RawDecoded48::HotFormat::Rrici:
    case RawDecoded48::HotFormat::Zrici:
      return RawDecoded48::CcMapKind::Map5;
    case RawDecoded48::HotFormat::Rrrci:
    case RawDecoded48::HotFormat::Zrrci:
      return RawDecoded48::CcMapKind::Map6;
    default:
      return RawDecoded48::CcMapKind::None;
    }
  case RawDecoded48::HotOp::Movd:
  case RawDecoded48::HotOp::Swapd:
    return (fmt == RawDecoded48::HotFormat::Rrci)
               ? RawDecoded48::CcMapKind::Map11
               : RawDecoded48::CcMapKind::None;
  case RawDecoded48::HotOp::Clz:
  case RawDecoded48::HotOp::Clo:
    switch (fmt) {
    case RawDecoded48::HotFormat::Rrc:
    case RawDecoded48::HotFormat::Zrc:
      return RawDecoded48::CcMapKind::Map1;
    case RawDecoded48::HotFormat::Rrci:
    case RawDecoded48::HotFormat::Zrci:
      return RawDecoded48::CcMapKind::Map8;
    default:
      return RawDecoded48::CcMapKind::None;
    }
  case RawDecoded48::HotOp::MulStep:
    return (fmt == RawDecoded48::HotFormat::Rrrici)
               ? RawDecoded48::CcMapKind::Map7
               : RawDecoded48::CcMapKind::None;
  case RawDecoded48::HotOp::DivStep:
    return (fmt == RawDecoded48::HotFormat::Rrrici)
               ? RawDecoded48::CcMapKind::Map9
               : RawDecoded48::CcMapKind::None;
  case RawDecoded48::HotOp::LslAdd:
  case RawDecoded48::HotOp::LslAddS:
  case RawDecoded48::HotOp::LslAddU:
  case RawDecoded48::HotOp::LslSub:
  case RawDecoded48::HotOp::LslSubS:
  case RawDecoded48::HotOp::LslSubU:
  case RawDecoded48::HotOp::LsrAdd:
  case RawDecoded48::HotOp::LsrAddS:
  case RawDecoded48::HotOp::LsrAddU:
    return (fmt == RawDecoded48::HotFormat::Rrrici ||
            fmt == RawDecoded48::HotFormat::Zrrici)
               ? RawDecoded48::CcMapKind::Map10
               : RawDecoded48::CcMapKind::None;
  default:
    return RawDecoded48::CcMapKind::None;
  }
}

static inline void decode_cc_metadata_48(RawDecoded48::CcMapKind cc_map_kind,
                                         int cc, RawDecoded48::CcKind &cc_kind,
                                         uint8_t &cc_subword_bits) {
  if (cc < 0 || cc > 63) {
    cc_kind = RawDecoded48::CcKind::False;
    cc_subword_bits = 0;
    return;
  }

  switch (cc_map_kind) {
  case RawDecoded48::CcMapKind::Map0:
    decode_cc_map_0_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map1:
    decode_cc_map_1_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map2:
    decode_cc_map_2_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map3:
    decode_cc_map_3_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map4:
    decode_cc_map_4_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map5:
    decode_cc_map_5_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map6:
    decode_cc_map_6_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map7:
    decode_cc_map_7_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map8:
    decode_cc_map_8_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map9:
    decode_cc_map_9_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map10:
    decode_cc_map_10_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map11:
    decode_cc_map_11_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map13:
    decode_cc_map_13_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map14:
    decode_cc_map_14_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::Map15:
    decode_cc_map_15_metadata(cc, cc_kind, cc_subword_bits);
    return;
  case RawDecoded48::CcMapKind::None:
  default:
    cc_kind = RawDecoded48::CcKind::False;
    cc_subword_bits = 0;
    return;
  }
}

static inline bool is_hot_format_one_of(RawDecoded48::HotFormat fmt,
                                        RawDecoded48::HotFormat a,
                                        RawDecoded48::HotFormat b) {
  return fmt == a || fmt == b;
}

static inline bool is_hot_format_one_of(RawDecoded48::HotFormat fmt,
                                        RawDecoded48::HotFormat a,
                                        RawDecoded48::HotFormat b,
                                        RawDecoded48::HotFormat c) {
  return fmt == a || fmt == b || fmt == c;
}

static inline bool is_hot_format_one_of(RawDecoded48::HotFormat fmt,
                                        RawDecoded48::HotFormat a,
                                        RawDecoded48::HotFormat b,
                                        RawDecoded48::HotFormat c,
                                        RawDecoded48::HotFormat d) {
  return fmt == a || fmt == b || fmt == c || fmt == d;
}

static inline bool
is_hot_format_one_of(RawDecoded48::HotFormat fmt, RawDecoded48::HotFormat a,
                     RawDecoded48::HotFormat b, RawDecoded48::HotFormat c,
                     RawDecoded48::HotFormat d, RawDecoded48::HotFormat e) {
  return fmt == a || fmt == b || fmt == c || fmt == d || fmt == e;
}

static inline bool
is_hot_format_one_of(RawDecoded48::HotFormat fmt, RawDecoded48::HotFormat a,
                     RawDecoded48::HotFormat b, RawDecoded48::HotFormat c,
                     RawDecoded48::HotFormat d, RawDecoded48::HotFormat e,
                     RawDecoded48::HotFormat f) {
  return fmt == a || fmt == b || fmt == c || fmt == d || fmt == e || fmt == f;
}

static inline bool
is_hot_format_one_of(RawDecoded48::HotFormat fmt, RawDecoded48::HotFormat a,
                     RawDecoded48::HotFormat b, RawDecoded48::HotFormat c,
                     RawDecoded48::HotFormat d, RawDecoded48::HotFormat e,
                     RawDecoded48::HotFormat f, RawDecoded48::HotFormat g) {
  return fmt == a || fmt == b || fmt == c || fmt == d || fmt == e || fmt == f ||
         fmt == g;
}

static inline void populate_hot_metadata_48(RawDecoded48 &decoded) {
  decoded.off_uses_sign_extend24 = false;
  decoded.cc_kind = RawDecoded48::CcKind::None;
  decoded.cc_subword_bits = 0;
  decoded.cc_map_kind = RawDecoded48::CcMapKind::None;
  decoded.writeback_mode = RawDecoded48::WritebackMode::None;
  decoded.mul_byte_pair_mode = RawDecoded48::MulBytePairMode::None;
  decoded.store_source_mode = RawDecoded48::StoreSourceMode::None;
  decoded.hot_supported = false;
  decoded.imm_is_signed8 = false;
  decoded.imm_passthrough = false;
  decoded.suppress_scalar_write = false;
  decoded.acquire_is_release = false;
  decoded.dma_to_mram = false;
  decoded.movd_is_swap = false;
  decoded.clz_invert_input = false;
  decoded.time_cfg_uses_ra = false;
  decoded.sub_lhs_is_imm = false;
  decoded.sub_rhs_is_reg = false;
  decoded.sub_is_rrrc = false;
  decoded.sub_is_zrrc = false;
  decoded.subc_lhs_is_imm = false;
  decoded.subc_rhs_is_reg = false;
  decoded.subc_is_rrrc = false;
  decoded.subc_is_zrrc = false;
  if (!decoded.valid || decoded.hot_op == RawDecoded48::HotOp::None) {
    return;
  }

  const auto fmt = decoded.hot_fmt;
  decoded.off_uses_sign_extend24 = (fmt == RawDecoded48::HotFormat::Erri ||
                                    fmt == RawDecoded48::HotFormat::Erir ||
                                    fmt == RawDecoded48::HotFormat::Erii ||
                                    fmt == RawDecoded48::HotFormat::Esii ||
                                    fmt == RawDecoded48::HotFormat::Esir);
  if (decoded.has_cc) {
    decoded.cc_map_kind = select_cc_map_48(decoded.hot_op, fmt);
    decode_cc_metadata_48(decoded.cc_map_kind, decoded.cc, decoded.cc_kind,
                          decoded.cc_subword_bits);
  }

  switch (decoded.hot_op) {
  case RawDecoded48::HotOp::Or:
  case RawDecoded48::HotOp::OrS:
  case RawDecoded48::HotOp::OrU:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rri, RawDecoded48::HotFormat::Rrr,
        RawDecoded48::HotFormat::Rrif, RawDecoded48::HotFormat::Rrici,
        RawDecoded48::HotFormat::Rrrci, RawDecoded48::HotFormat::Zrr,
        RawDecoded48::HotFormat::Zrrci);
    decoded.suppress_scalar_write =
        decoded.hot_op == RawDecoded48::HotOp::Or &&
        is_hot_format_one_of(fmt, RawDecoded48::HotFormat::Zrr,
                             RawDecoded48::HotFormat::Zrrci);
    if (decoded.hot_op == RawDecoded48::HotOp::OrS) {
      decoded.writeback_mode = RawDecoded48::WritebackMode::DregSign32;
    } else if (decoded.hot_op == RawDecoded48::HotOp::OrU) {
      decoded.writeback_mode = RawDecoded48::WritebackMode::DregZero32;
    }
    return;
  case RawDecoded48::HotOp::Add:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rri, RawDecoded48::HotFormat::Rrr,
        RawDecoded48::HotFormat::Rrici, RawDecoded48::HotFormat::Rrrci);
    decoded.imm_is_signed8 = (fmt == RawDecoded48::HotFormat::Rrici);
    return;
  case RawDecoded48::HotOp::Addc:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rri, RawDecoded48::HotFormat::Rrr,
        RawDecoded48::HotFormat::Zrr);
    return;
  case RawDecoded48::HotOp::And:
  case RawDecoded48::HotOp::AndS:
  case RawDecoded48::HotOp::AndU:
    if (decoded.hot_op == RawDecoded48::HotOp::And) {
      decoded.hot_supported = is_hot_format_one_of(
          fmt, RawDecoded48::HotFormat::Rri, RawDecoded48::HotFormat::Rrr,
          RawDecoded48::HotFormat::Rrici);
    } else {
      decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Rki);
      decoded.imm_passthrough = decoded.hot_supported;
      decoded.writeback_mode = (decoded.hot_op == RawDecoded48::HotOp::AndS)
                                   ? RawDecoded48::WritebackMode::DregSign32
                                   : RawDecoded48::WritebackMode::DregZero32;
    }
    return;
  case RawDecoded48::HotOp::Sub:
    decoded.hot_supported =
        is_hot_format_one_of(
            fmt, RawDecoded48::HotFormat::Rir, RawDecoded48::HotFormat::Rrr,
            RawDecoded48::HotFormat::Zrr, RawDecoded48::HotFormat::Rrrc,
            RawDecoded48::HotFormat::Zrrc, RawDecoded48::HotFormat::Rrrci,
            RawDecoded48::HotFormat::Zrrci) ||
        is_hot_format_one_of(fmt, RawDecoded48::HotFormat::Rrif,
                             RawDecoded48::HotFormat::Rrici,
                             RawDecoded48::HotFormat::Zrici);
    decoded.sub_lhs_is_imm = (fmt == RawDecoded48::HotFormat::Rir);
    decoded.sub_rhs_is_reg = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rrr, RawDecoded48::HotFormat::Zrr,
        RawDecoded48::HotFormat::Rrrc, RawDecoded48::HotFormat::Zrrc,
        RawDecoded48::HotFormat::Rrrci, RawDecoded48::HotFormat::Zrrci);
    decoded.sub_is_rrrc = (fmt == RawDecoded48::HotFormat::Rrrc);
    decoded.sub_is_zrrc = (fmt == RawDecoded48::HotFormat::Zrrc);
    return;
  case RawDecoded48::HotOp::Subc:
    decoded.hot_supported =
        is_hot_format_one_of(
            fmt, RawDecoded48::HotFormat::Rir, RawDecoded48::HotFormat::Rirci,
            RawDecoded48::HotFormat::Zirci, RawDecoded48::HotFormat::Rrr,
            RawDecoded48::HotFormat::Zrr, RawDecoded48::HotFormat::Rrrc,
            RawDecoded48::HotFormat::Zrrc) ||
        is_hot_format_one_of(
            fmt, RawDecoded48::HotFormat::Rrrci, RawDecoded48::HotFormat::Zrrci,
            RawDecoded48::HotFormat::Rric, RawDecoded48::HotFormat::Zric,
            RawDecoded48::HotFormat::Rrici, RawDecoded48::HotFormat::Zrici);
    decoded.subc_lhs_is_imm = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rir, RawDecoded48::HotFormat::Rirci,
        RawDecoded48::HotFormat::Zirci);
    decoded.subc_rhs_is_reg = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rrr, RawDecoded48::HotFormat::Zrr,
        RawDecoded48::HotFormat::Rrrc, RawDecoded48::HotFormat::Zrrc,
        RawDecoded48::HotFormat::Rrrci, RawDecoded48::HotFormat::Zrrci);
    decoded.subc_is_rrrc = (fmt == RawDecoded48::HotFormat::Rrrc);
    decoded.subc_is_zrrc = (fmt == RawDecoded48::HotFormat::Zrrc);
    return;
  case RawDecoded48::HotOp::MulStep:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Rrrici ||
                             fmt == RawDecoded48::HotFormat::Rrri);
    return;
  case RawDecoded48::HotOp::DivStep:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Rrrici);
    return;
  case RawDecoded48::HotOp::Acquire:
  case RawDecoded48::HotOp::Release:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Rici);
    decoded.acquire_is_release =
        (decoded.hot_op == RawDecoded48::HotOp::Release);
    return;
  case RawDecoded48::HotOp::Resume:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Rici);
    return;
  case RawDecoded48::HotOp::Boot:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Rici);
    return;
  case RawDecoded48::HotOp::Call:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rri, RawDecoded48::HotFormat::Zri);
    return;
  case RawDecoded48::HotOp::Ldma:
  case RawDecoded48::HotOp::Sdma:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Rri);
    decoded.dma_to_mram = (decoded.hot_op == RawDecoded48::HotOp::Sdma);
    return;
  case RawDecoded48::HotOp::Lsl:
  case RawDecoded48::HotOp::LslS:
  case RawDecoded48::HotOp::LslU:
  case RawDecoded48::HotOp::Lslx:
  case RawDecoded48::HotOp::LslxS:
  case RawDecoded48::HotOp::LslxU:
  case RawDecoded48::HotOp::Lsr:
  case RawDecoded48::HotOp::LsrS:
  case RawDecoded48::HotOp::LsrU:
  case RawDecoded48::HotOp::Lsrx:
  case RawDecoded48::HotOp::LsrxS:
  case RawDecoded48::HotOp::LsrxU:
    decoded.hot_supported =
        is_hot_format_one_of(
            fmt, RawDecoded48::HotFormat::Rri, RawDecoded48::HotFormat::Rric,
            RawDecoded48::HotFormat::Rrici, RawDecoded48::HotFormat::Rrr,
            RawDecoded48::HotFormat::Rrrc, RawDecoded48::HotFormat::Rrrci,
            RawDecoded48::HotFormat::Zri) ||
        is_hot_format_one_of(
            fmt, RawDecoded48::HotFormat::Zric, RawDecoded48::HotFormat::Zrici,
            RawDecoded48::HotFormat::Zrr, RawDecoded48::HotFormat::Zrrc,
            RawDecoded48::HotFormat::Zrrci);
    return;
  case RawDecoded48::HotOp::Movd:
  case RawDecoded48::HotOp::Swapd:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Rrci);
    decoded.movd_is_swap = (decoded.hot_op == RawDecoded48::HotOp::Swapd);
    return;
  case RawDecoded48::HotOp::Clz:
  case RawDecoded48::HotOp::Clo:
    decoded.hot_supported =
        (decoded.hot_op == RawDecoded48::HotOp::Clz &&
         is_hot_format_one_of(fmt, RawDecoded48::HotFormat::Rr,
                              RawDecoded48::HotFormat::Rrci)) ||
        (decoded.hot_op == RawDecoded48::HotOp::Clo &&
         fmt == RawDecoded48::HotFormat::Zrci);
    decoded.clz_invert_input = (decoded.hot_op == RawDecoded48::HotOp::Clo);
    return;
  case RawDecoded48::HotOp::MulUlUl:
  case RawDecoded48::HotOp::MulUlUh:
  case RawDecoded48::HotOp::MulUhUl:
  case RawDecoded48::HotOp::MulUhUh:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Rrr);
    switch (decoded.hot_op) {
    case RawDecoded48::HotOp::MulUlUl:
      decoded.mul_byte_pair_mode = RawDecoded48::MulBytePairMode::UlUl;
      return;
    case RawDecoded48::HotOp::MulUlUh:
      decoded.mul_byte_pair_mode = RawDecoded48::MulBytePairMode::UlUh;
      return;
    case RawDecoded48::HotOp::MulUhUl:
      decoded.mul_byte_pair_mode = RawDecoded48::MulBytePairMode::UhUl;
      return;
    case RawDecoded48::HotOp::MulUhUh:
      decoded.mul_byte_pair_mode = RawDecoded48::MulBytePairMode::UhUh;
      return;
    default:
      return;
    }
  case RawDecoded48::HotOp::LslAdd:
  case RawDecoded48::HotOp::LslAddS:
  case RawDecoded48::HotOp::LslAddU:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rrri, RawDecoded48::HotFormat::Rrrici,
        RawDecoded48::HotFormat::Zrri, RawDecoded48::HotFormat::Zrrici);
    decoded.suppress_scalar_write =
        decoded.hot_op == RawDecoded48::HotOp::LslAdd &&
        is_hot_format_one_of(fmt, RawDecoded48::HotFormat::Zrri,
                             RawDecoded48::HotFormat::Zrrici);
    if (decoded.hot_op == RawDecoded48::HotOp::LslAddS) {
      decoded.writeback_mode = RawDecoded48::WritebackMode::DregSign32;
    } else if (decoded.hot_op == RawDecoded48::HotOp::LslAddU) {
      decoded.writeback_mode = RawDecoded48::WritebackMode::DregZero32;
    }
    return;
  case RawDecoded48::HotOp::LslSub:
  case RawDecoded48::HotOp::LslSubS:
  case RawDecoded48::HotOp::LslSubU:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rrri, RawDecoded48::HotFormat::Rrrici,
        RawDecoded48::HotFormat::Zrri, RawDecoded48::HotFormat::Zrrici);
    decoded.suppress_scalar_write =
        decoded.hot_op == RawDecoded48::HotOp::LslSub &&
        is_hot_format_one_of(fmt, RawDecoded48::HotFormat::Zrri,
                             RawDecoded48::HotFormat::Zrrici);
    if (decoded.hot_op == RawDecoded48::HotOp::LslSubS) {
      decoded.writeback_mode = RawDecoded48::WritebackMode::DregSign32;
    } else if (decoded.hot_op == RawDecoded48::HotOp::LslSubU) {
      decoded.writeback_mode = RawDecoded48::WritebackMode::DregZero32;
    }
    return;
  case RawDecoded48::HotOp::LsrAdd:
  case RawDecoded48::HotOp::LsrAddS:
  case RawDecoded48::HotOp::LsrAddU:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rrri, RawDecoded48::HotFormat::Rrrici,
        RawDecoded48::HotFormat::Zrrici);
    decoded.suppress_scalar_write =
        decoded.hot_op == RawDecoded48::HotOp::LsrAdd &&
        fmt == RawDecoded48::HotFormat::Zrrici;
    if (decoded.hot_op == RawDecoded48::HotOp::LsrAddS) {
      decoded.writeback_mode = RawDecoded48::WritebackMode::DregSign32;
    } else if (decoded.hot_op == RawDecoded48::HotOp::LsrAddU) {
      decoded.writeback_mode = RawDecoded48::WritebackMode::DregZero32;
    }
    return;
  case RawDecoded48::HotOp::Lbu:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Erri);
    return;
  case RawDecoded48::HotOp::Lbs:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Erri);
    return;
  case RawDecoded48::HotOp::Lw:
  case RawDecoded48::HotOp::LwS:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Erri);
    decoded.writeback_mode = (decoded.hot_op == RawDecoded48::HotOp::LwS)
                                 ? RawDecoded48::WritebackMode::DregSign32
                                 : RawDecoded48::WritebackMode::DregZero32;
    return;
  case RawDecoded48::HotOp::Ld:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Erri);
    return;
  case RawDecoded48::HotOp::Sw:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Erii, RawDecoded48::HotFormat::Erir);
    decoded.store_source_mode = (fmt == RawDecoded48::HotFormat::Erii)
                                    ? RawDecoded48::StoreSourceMode::Imm32
                                    : RawDecoded48::StoreSourceMode::Reg32;
    return;
  case RawDecoded48::HotOp::Sb:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Erii, RawDecoded48::HotFormat::Esii,
        RawDecoded48::HotFormat::Erir, RawDecoded48::HotFormat::Esir);
    decoded.store_source_mode =
        is_hot_format_one_of(fmt, RawDecoded48::HotFormat::Erii,
                             RawDecoded48::HotFormat::Esii)
            ? RawDecoded48::StoreSourceMode::Imm32
            : RawDecoded48::StoreSourceMode::Reg32;
    return;
  case RawDecoded48::HotOp::Sh:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Erii, RawDecoded48::HotFormat::Esii,
        RawDecoded48::HotFormat::Erir, RawDecoded48::HotFormat::Esir);
    decoded.store_source_mode =
        is_hot_format_one_of(fmt, RawDecoded48::HotFormat::Erii,
                             RawDecoded48::HotFormat::Esii)
            ? RawDecoded48::StoreSourceMode::Imm32
            : RawDecoded48::StoreSourceMode::Reg32;
    return;
  case RawDecoded48::HotOp::Sd:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Erii, RawDecoded48::HotFormat::Erir);
    decoded.store_source_mode = (fmt == RawDecoded48::HotFormat::Erii)
                                    ? RawDecoded48::StoreSourceMode::Imm32
                                    : RawDecoded48::StoreSourceMode::Dreg64;
    return;
  case RawDecoded48::HotOp::Time:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::R);
    return;
  case RawDecoded48::HotOp::TimeCfg:
    decoded.hot_supported = is_hot_format_one_of(
        fmt, RawDecoded48::HotFormat::Rr, RawDecoded48::HotFormat::Zr);
    decoded.time_cfg_uses_ra = (fmt == RawDecoded48::HotFormat::Zr);
    return;
  case RawDecoded48::HotOp::Stop:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::Ci);
    return;
  case RawDecoded48::HotOp::Nop:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::None);
    return;
  case RawDecoded48::HotOp::Fault:
    decoded.hot_supported = (fmt == RawDecoded48::HotFormat::I);
    return;
  case RawDecoded48::HotOp::None:
  default:
    return;
  }
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
            set_hot_metadata(out, RawDecoded48::HotOp::Acquire,
                             RawDecoded48::HotFormat::Rici);
            return true;
          }
          if ((((instruction >> 24) & 0x3)) == (0x0)) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 3) << 0));
            out.has_cc = true;
            out.valid = true;
            set_hot_metadata(out, RawDecoded48::HotOp::Release,
                             RawDecoded48::HotFormat::Rici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Boot,
                               RawDecoded48::HotFormat::Rici);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Fault,
                                 RawDecoded48::HotFormat::I);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Resume,
                               RawDecoded48::HotFormat::Rici);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Stop,
                                 RawDecoded48::HotFormat::Ci);
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
                    set_hot_metadata(out, RawDecoded48::HotOp::Nop,
                                     RawDecoded48::HotFormat::None);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Sb,
                               RawDecoded48::HotFormat::Erii);
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x6)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Sd,
                               RawDecoded48::HotFormat::Erii);
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x2)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Sh,
                               RawDecoded48::HotFormat::Erii);
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Sw,
                               RawDecoded48::HotFormat::Erii);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Sb,
                               RawDecoded48::HotFormat::Esii);
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x6)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Sd,
                               RawDecoded48::HotFormat::Esii);
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x2)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Sh,
                               RawDecoded48::HotFormat::Esii);
              return true;
            }
            if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
              out.imm =
                  static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                       (((instruction >> 0) & 4095) << 4));
              out.has_imm = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Sw,
                               RawDecoded48::HotFormat::Esii);
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
            return true;
          }
          if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x6)) {
            out.imm = static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                           (((instruction >> 0) & 4095) << 4));
            out.has_imm = true;
            out.valid = true;
            return true;
          }
          if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x2)) {
            out.imm = static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                           (((instruction >> 0) & 4095) << 4));
            out.has_imm = true;
            out.valid = true;
            return true;
          }
          if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
            out.imm = static_cast<int32_t>((((instruction >> 16) & 15) << 0) |
                                           (((instruction >> 0) & 4095) << 4));
            out.has_imm = true;
            out.valid = true;
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
                  return true;
                }
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  set_hot_metadata(out, RawDecoded48::HotOp::Lbs,
                                   RawDecoded48::HotFormat::Erri);
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
                  return true;
                }
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  set_hot_metadata(out, RawDecoded48::HotOp::Lbu,
                                   RawDecoded48::HotFormat::Erri);
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
                  return true;
                }
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
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
                  return true;
                }
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
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
                  set_hot_metadata(out, RawDecoded48::HotOp::LwS,
                                   RawDecoded48::HotFormat::Erri);
                  return true;
                }
                if (((((instruction >> 29) & 0x1)) == (0x1)) &&
                    ((((instruction >> 39) & 0x1)) == (0x0))) {
                  out.dc =
                      static_cast<int32_t>((((instruction >> 40) & 15) << 1));
                  out.has_dc = true;
                  out.valid = true;
                  return true;
                }
                if ((((instruction >> 29) & 0x1)) == (0x0)) {
                  out.rc =
                      static_cast<int32_t>((((instruction >> 39) & 31) << 0));
                  out.has_rc = true;
                  out.valid = true;
                  set_hot_metadata(out, RawDecoded48::HotOp::Lw,
                                   RawDecoded48::HotFormat::Erri);
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
                  set_hot_metadata(out, RawDecoded48::HotOp::Lbs,
                                   RawDecoded48::HotFormat::Ersi);
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
                  set_hot_metadata(out, RawDecoded48::HotOp::Lbu,
                                   RawDecoded48::HotFormat::Ersi);
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
                  set_hot_metadata(out, RawDecoded48::HotOp::Lw,
                                   RawDecoded48::HotFormat::Ersi);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Ld,
                                 RawDecoded48::HotFormat::Erri);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Ld,
                                 RawDecoded48::HotFormat::Ersi);
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
          set_hot_metadata(out, RawDecoded48::HotOp::Ldma,
                           RawDecoded48::HotFormat::Rri);
          return true;
        }
        if (((((instruction >> 0) & 0xffff)) & (0xffff)) == (0x1)) {
          out.valid = true;
          return true;
        }
        if (((((instruction >> 0) & 0xffff)) & (0xffff)) == (0x2)) {
          out.valid = true;
          set_hot_metadata(out, RawDecoded48::HotOp::Sdma,
                           RawDecoded48::HotFormat::Rri);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sb,
                                 RawDecoded48::HotFormat::Erir);
                return true;
              }
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x2)) {
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::Sh,
                                 RawDecoded48::HotFormat::Erir);
                return true;
              }
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::Sw,
                                 RawDecoded48::HotFormat::Erir);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sb,
                                 RawDecoded48::HotFormat::Esir);
                return true;
              }
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x2)) {
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::Sh,
                                 RawDecoded48::HotFormat::Esir);
                return true;
              }
              if (((((instruction >> 24) & 0xf)) & (0x6)) == (0x4)) {
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::Sw,
                                 RawDecoded48::HotFormat::Esir);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sd,
                                 RawDecoded48::HotFormat::Erir);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sd,
                                 RawDecoded48::HotFormat::Esir);
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
          set_hot_metadata(out, RawDecoded48::HotOp::Add,
                           RawDecoded48::HotFormat::Rri);
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
          set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                           RawDecoded48::HotFormat::Rri);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Add,
                               RawDecoded48::HotFormat::Rric);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Add,
                               RawDecoded48::HotFormat::Rrici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Add,
                               RawDecoded48::HotFormat::Rrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Add,
                             RawDecoded48::HotFormat::Ssi);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                               RawDecoded48::HotFormat::Rric);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                               RawDecoded48::HotFormat::Rrici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                               RawDecoded48::HotFormat::Rrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                             RawDecoded48::HotFormat::Rirc);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                             RawDecoded48::HotFormat::Rirci);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                             RawDecoded48::HotFormat::Rirf);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                               RawDecoded48::HotFormat::Rric);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                               RawDecoded48::HotFormat::Rrici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                               RawDecoded48::HotFormat::Rrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                             RawDecoded48::HotFormat::Ssi);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                             RawDecoded48::HotFormat::Rirc);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                             RawDecoded48::HotFormat::Rirci);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                             RawDecoded48::HotFormat::Rirf);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                               RawDecoded48::HotFormat::Rric);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                               RawDecoded48::HotFormat::Rrici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                               RawDecoded48::HotFormat::Rrif);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Add,
                               RawDecoded48::HotFormat::Zric);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Add,
                               RawDecoded48::HotFormat::Zrici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Add,
                               RawDecoded48::HotFormat::Zrif);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                               RawDecoded48::HotFormat::Zric);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                               RawDecoded48::HotFormat::Zrici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                               RawDecoded48::HotFormat::Zrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                             RawDecoded48::HotFormat::Zirc);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                             RawDecoded48::HotFormat::Zirci);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                             RawDecoded48::HotFormat::Zirf);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                               RawDecoded48::HotFormat::Zric);
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
              out.imm = sign_extend11(
                  static_cast<uint32_t>((((instruction >> 20) & 15) << 0) |
                                        (((instruction >> 16) & 15) << 4) |
                                        (((instruction >> 39) & 7) << 8)));
              out.has_imm = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                               RawDecoded48::HotFormat::Zrici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                               RawDecoded48::HotFormat::Zrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                             RawDecoded48::HotFormat::Zirc);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                             RawDecoded48::HotFormat::Zirci);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                             RawDecoded48::HotFormat::Zirf);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                               RawDecoded48::HotFormat::Zric);
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
              out.imm = sign_extend11(
                  static_cast<uint32_t>((((instruction >> 20) & 15) << 0) |
                                        (((instruction >> 16) & 15) << 4) |
                                        (((instruction >> 39) & 7) << 8)));
              out.has_imm = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                               RawDecoded48::HotFormat::Zrici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                               RawDecoded48::HotFormat::Zrif);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Add,
                                 RawDecoded48::HotFormat::Rrr);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Add,
                                 RawDecoded48::HotFormat::Rrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Add,
                                 RawDecoded48::HotFormat::Rrrci);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Add,
                                 RawDecoded48::HotFormat::Sss);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                                 RawDecoded48::HotFormat::Rrr);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                                 RawDecoded48::HotFormat::Rrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                                 RawDecoded48::HotFormat::Rrrci);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                                 RawDecoded48::HotFormat::Sss);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                                 RawDecoded48::HotFormat::Rrr);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                                 RawDecoded48::HotFormat::Rrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                                 RawDecoded48::HotFormat::Rrrci);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                                 RawDecoded48::HotFormat::Rrr);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                                 RawDecoded48::HotFormat::Rrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                                 RawDecoded48::HotFormat::Rrrci);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Add,
                                 RawDecoded48::HotFormat::Zrr);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Add,
                                 RawDecoded48::HotFormat::Zrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Add,
                                 RawDecoded48::HotFormat::Zrrci);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                                 RawDecoded48::HotFormat::Zrr);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                                 RawDecoded48::HotFormat::Zrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                                 RawDecoded48::HotFormat::Zrrci);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                                 RawDecoded48::HotFormat::Zrr);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                                 RawDecoded48::HotFormat::Zrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                                 RawDecoded48::HotFormat::Zrrci);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                                 RawDecoded48::HotFormat::Zrr);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                                 RawDecoded48::HotFormat::Zrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                                 RawDecoded48::HotFormat::Zrrci);
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUh,
                                 RawDecoded48::HotFormat::Rrr);
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUh,
                                 RawDecoded48::HotFormat::Rrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUh,
                                 RawDecoded48::HotFormat::Rrrci);
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUh,
                                 RawDecoded48::HotFormat::Zrr);
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUh,
                                 RawDecoded48::HotFormat::Zrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUh,
                                 RawDecoded48::HotFormat::Zrrci);
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUl,
                                 RawDecoded48::HotFormat::Rrr);
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUl,
                                 RawDecoded48::HotFormat::Rrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUl,
                                 RawDecoded48::HotFormat::Rrrci);
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUl,
                                 RawDecoded48::HotFormat::Zrr);
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUl,
                                 RawDecoded48::HotFormat::Zrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUhUl,
                                 RawDecoded48::HotFormat::Zrrci);
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUh,
                                 RawDecoded48::HotFormat::Rrr);
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUh,
                                 RawDecoded48::HotFormat::Rrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUh,
                                 RawDecoded48::HotFormat::Rrrci);
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUh,
                                 RawDecoded48::HotFormat::Zrr);
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUh,
                                 RawDecoded48::HotFormat::Zrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUh,
                                 RawDecoded48::HotFormat::Zrrci);
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
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUl,
                                 RawDecoded48::HotFormat::Rrr);
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUl,
                                 RawDecoded48::HotFormat::Rrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUl,
                                 RawDecoded48::HotFormat::Rrrci);
                return true;
              }
              return false;
            }
            if ((((instruction >> 44) & 0x3)) == (0x3)) {
              if ((((instruction >> 24) & 0xf)) == (0x0)) {
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUl,
                                 RawDecoded48::HotFormat::Zrr);
                return true;
              }
              if ((((((instruction >> 26) & 0x1)) ^
                    (((instruction >> 27) & 0x1))) == (0x1)) &&
                  ((((instruction >> 25) & 0x1)) == (0x1))) {
                out.cc =
                    static_cast<int32_t>((((instruction >> 24) & 15) << 0));
                out.has_cc = true;
                out.valid = true;
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUl,
                                 RawDecoded48::HotFormat::Zrrc);
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
                set_hot_metadata(out, RawDecoded48::HotOp::MulUlUl,
                                 RawDecoded48::HotFormat::Zrrci);
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
                  set_hot_metadata(out, RawDecoded48::HotOp::Time,
                                   RawDecoded48::HotFormat::R);
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
                  set_hot_metadata(out, RawDecoded48::HotOp::Time,
                                   RawDecoded48::HotFormat::Rci);
                  return true;
                }
                return false;
              }
              if ((((instruction >> 44) & 0x3)) == (0x3)) {
                if ((((instruction >> 24) & 0xf)) == (0x0)) {
                  out.valid = true;
                  set_hot_metadata(out, RawDecoded48::HotOp::Time,
                                   RawDecoded48::HotFormat::Z);
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
                  set_hot_metadata(out, RawDecoded48::HotOp::Time,
                                   RawDecoded48::HotFormat::Zci);
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
                  set_hot_metadata(out, RawDecoded48::HotOp::TimeCfg,
                                   RawDecoded48::HotFormat::Rr);
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
                  set_hot_metadata(out, RawDecoded48::HotOp::TimeCfg,
                                   RawDecoded48::HotFormat::Rrci);
                  return true;
                }
                return false;
              }
              if ((((instruction >> 44) & 0x3)) == (0x3)) {
                if ((((instruction >> 24) & 0xf)) == (0x0)) {
                  out.valid = true;
                  set_hot_metadata(out, RawDecoded48::HotOp::TimeCfg,
                                   RawDecoded48::HotFormat::Zr);
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
                  set_hot_metadata(out, RawDecoded48::HotOp::TimeCfg,
                                   RawDecoded48::HotFormat::Zrci);
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
          set_hot_metadata(out, RawDecoded48::HotOp::Add,
                           RawDecoded48::HotFormat::Zri);
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0x2)) {
          out.valid = true;
          set_hot_metadata(out, RawDecoded48::HotOp::Addc,
                           RawDecoded48::HotFormat::Zri);
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0xa)) {
          out.valid = true;
          set_hot_metadata(out, RawDecoded48::HotOp::And,
                           RawDecoded48::HotFormat::Zri);
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0xc)) {
          out.valid = true;
          set_hot_metadata(out, RawDecoded48::HotOp::Or,
                           RawDecoded48::HotFormat::Zri);
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0x4)) {
          out.valid = true;
          set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                           RawDecoded48::HotFormat::Zir);
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0x6)) {
          out.valid = true;
          set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                           RawDecoded48::HotFormat::Zir);
          return true;
        }
        if (((((instruction >> 39) & 0x1f)) & (0x1e)) == (0x8)) {
          out.valid = true;
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
        set_hot_metadata(out, RawDecoded48::HotOp::Or,
                         RawDecoded48::HotFormat::Rri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::AndS,
                             RawDecoded48::HotFormat::Rki);
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
            set_hot_metadata(out, RawDecoded48::HotOp::AndU,
                             RawDecoded48::HotFormat::Rki);
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
            set_hot_metadata(out, RawDecoded48::HotOp::And,
                             RawDecoded48::HotFormat::Rri);
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
          set_hot_metadata(out, RawDecoded48::HotOp::OrS,
                           RawDecoded48::HotFormat::Rri);
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
          set_hot_metadata(out, RawDecoded48::HotOp::OrU,
                           RawDecoded48::HotFormat::Rri);
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
          set_hot_metadata(out, RawDecoded48::HotOp::AndS,
                           RawDecoded48::HotFormat::Rri);
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
          set_hot_metadata(out, RawDecoded48::HotOp::AndU,
                           RawDecoded48::HotFormat::Rri);
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
          set_hot_metadata(out, RawDecoded48::HotOp::Sub,
                           RawDecoded48::HotFormat::Rir);
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
          set_hot_metadata(out, RawDecoded48::HotOp::Subc,
                           RawDecoded48::HotFormat::Rir);
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
            set_hot_metadata(out, RawDecoded48::HotOp::AndS,
                             RawDecoded48::HotFormat::Rric);
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
            set_hot_metadata(out, RawDecoded48::HotOp::AndS,
                             RawDecoded48::HotFormat::Rrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::AndS,
                             RawDecoded48::HotFormat::Rrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::AndU,
                             RawDecoded48::HotFormat::Rric);
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
            set_hot_metadata(out, RawDecoded48::HotOp::AndU,
                             RawDecoded48::HotFormat::Rrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::AndU,
                             RawDecoded48::HotFormat::Rrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::And,
                             RawDecoded48::HotFormat::Rric);
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
            set_hot_metadata(out, RawDecoded48::HotOp::And,
                             RawDecoded48::HotFormat::Rrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::And,
                             RawDecoded48::HotFormat::Rrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::And,
                             RawDecoded48::HotFormat::Zric);
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
            set_hot_metadata(out, RawDecoded48::HotOp::And,
                             RawDecoded48::HotFormat::Zrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::And,
                             RawDecoded48::HotFormat::Zrif);
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslS,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslxS,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrS,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxS,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslS,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslxS,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrS,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxS,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslS,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslxS,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrS,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxS,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslU,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslxU,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrU,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxU,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslU,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslxU,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrU,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxU,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslU,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslxU,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrU,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxU,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Rri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Rric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Rrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Zri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Zri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Zri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Zri);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Zric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Zric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Zric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Zric);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.valid = true;
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
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x6)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x7)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x4)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Zrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x5)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Zrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xe)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xf)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xc)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Zrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xd)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Zrici);
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0x2)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
              return true;
            }
            if (((((instruction >> 16) & 0xf)) & (0xf)) == (0xa)) {
              out.pc =
                  static_cast<int32_t>((((instruction >> 0) & 65535) << 0));
              out.has_pc = true;
              out.valid = true;
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
            set_hot_metadata(out, RawDecoded48::HotOp::Call,
                             RawDecoded48::HotFormat::Rri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Call,
                             RawDecoded48::HotFormat::Zri);
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              set_hot_metadata(out, RawDecoded48::HotOp::Clo,
                               RawDecoded48::HotFormat::Rr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Clo,
                               RawDecoded48::HotFormat::Rrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Clo,
                               RawDecoded48::HotFormat::Rrci);
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Clo,
                               RawDecoded48::HotFormat::Zr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Clo,
                               RawDecoded48::HotFormat::Zrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Clo,
                               RawDecoded48::HotFormat::Zrci);
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              set_hot_metadata(out, RawDecoded48::HotOp::Clz,
                               RawDecoded48::HotFormat::Rr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Clz,
                               RawDecoded48::HotFormat::Rrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Clz,
                               RawDecoded48::HotFormat::Rrci);
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Clz,
                               RawDecoded48::HotFormat::Zr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Clz,
                               RawDecoded48::HotFormat::Zrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Clz,
                               RawDecoded48::HotFormat::Zrci);
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
            set_hot_metadata(out, RawDecoded48::HotOp::OrS,
                             RawDecoded48::HotFormat::Rric);
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
            set_hot_metadata(out, RawDecoded48::HotOp::OrS,
                             RawDecoded48::HotFormat::Rrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::OrS,
                             RawDecoded48::HotFormat::Rrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::OrU,
                             RawDecoded48::HotFormat::Rric);
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
            set_hot_metadata(out, RawDecoded48::HotOp::OrU,
                             RawDecoded48::HotFormat::Rrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::OrU,
                             RawDecoded48::HotFormat::Rrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Or,
                             RawDecoded48::HotFormat::Rric);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Or,
                             RawDecoded48::HotFormat::Rrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Or,
                             RawDecoded48::HotFormat::Rrif);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Or,
                             RawDecoded48::HotFormat::Zric);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Or,
                             RawDecoded48::HotFormat::Zrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::Or,
                             RawDecoded48::HotFormat::Zrif);
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              set_hot_metadata(out, RawDecoded48::HotOp::AndS,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::AndS,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::AndS,
                               RawDecoded48::HotFormat::Rrrci);
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
              set_hot_metadata(out, RawDecoded48::HotOp::AndU,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::AndU,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::AndU,
                               RawDecoded48::HotFormat::Rrrci);
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
              set_hot_metadata(out, RawDecoded48::HotOp::And,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::And,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::And,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::And,
                               RawDecoded48::HotFormat::Zrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::And,
                               RawDecoded48::HotFormat::Zrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::And,
                               RawDecoded48::HotFormat::Zrrci);
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              set_hot_metadata(out, RawDecoded48::HotOp::Call,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Call,
                               RawDecoded48::HotFormat::Zrr);
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
          if ((((((instruction >> 42) & 0x3)) != (0x3)) &&
               ((((instruction >> 44) & 0x1)) == (0x1))) &&
              ((((instruction >> 39) & 0x1)) == (0x1))) {
            out.dc = static_cast<int32_t>((((instruction >> 40) & 15) << 1));
            out.has_dc = true;
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::OrS,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::OrS,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::OrS,
                               RawDecoded48::HotFormat::Rrrci);
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
              set_hot_metadata(out, RawDecoded48::HotOp::OrU,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::OrU,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::OrU,
                               RawDecoded48::HotFormat::Rrrci);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Or,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Or,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Or,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Or,
                               RawDecoded48::HotFormat::Zrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Or,
                               RawDecoded48::HotFormat::Zrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Or,
                               RawDecoded48::HotFormat::Zrrci);
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if ((((instruction >> 42) & 0x3)) == (0x3)) {
            out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
            out.has_ra = true;
            if ((((instruction >> 24) & 0xf)) == (0x0)) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x4)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslS,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslS,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::LslS,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x6)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x7)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x5)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslxS,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslxS,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::LslxS,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xc)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrS,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrS,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::LsrS,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xe)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xf)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xd)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxS,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxS,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxS,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x2)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xa)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x4)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslU,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslU,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::LslU,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x6)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x7)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x5)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslxU,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LslxU,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::LslxU,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xc)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrU,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrU,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::LsrU,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xe)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xf)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xd)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxU,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxU,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::LsrxU,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x2)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xa)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x6)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x7)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x4)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x5)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xe)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xf)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xc)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xd)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Rrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Rrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Rrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x2)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xa)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x6)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x7)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x4)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Zrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Zrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Lsl,
                               RawDecoded48::HotFormat::Zrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x5)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Zrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Zrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Lslx,
                               RawDecoded48::HotFormat::Zrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xe)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xf)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xc)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Zrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Zrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Lsr,
                               RawDecoded48::HotFormat::Zrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xd)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Zrr);
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Zrrc);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Lsrx,
                               RawDecoded48::HotFormat::Zrrci);
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0x2)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
              return true;
            }
            return false;
          }
          if (((((instruction >> 20) & 0xf)) & (0xf)) == (0xa)) {
            if (((((instruction >> 24) & 0xf)) == (0x0)) &&
                ((((instruction >> 29) & 0x1)) == (0x0))) {
              out.valid = true;
              return true;
            }
            if ((((((instruction >> 26) & 0x1)) ^
                  (((instruction >> 27) & 0x1))) == (0x1)) &&
                ((((instruction >> 25) & 0x1)) == (0x1))) {
              out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
              out.has_cc = true;
              out.valid = true;
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
            return true;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.valid = true;
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
            return true;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.valid = true;
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
            return true;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.valid = true;
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
            return true;
          }
          return false;
        }
        if ((((instruction >> 42) & 0x3)) == (0x3)) {
          out.ra = static_cast<int32_t>((((instruction >> 34) & 31) << 0));
          out.has_ra = true;
          if ((((instruction >> 24) & 0xf)) == (0x0)) {
            out.valid = true;
            return true;
          }
          if ((((((instruction >> 26) & 0x1)) ^
                (((instruction >> 27) & 0x1))) == (0x1)) &&
              ((((instruction >> 25) & 0x1)) == (0x1))) {
            out.cc = static_cast<int32_t>((((instruction >> 24) & 15) << 0));
            out.has_cc = true;
            out.valid = true;
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
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x4)) {
        out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                      (((instruction >> 32) & 3) << 3));
        out.has_rb = true;
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslAddS,
                             RawDecoded48::HotFormat::Rrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslAddS,
                             RawDecoded48::HotFormat::Rrrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslAddU,
                             RawDecoded48::HotFormat::Rrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslAddU,
                             RawDecoded48::HotFormat::Rrrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslAdd,
                             RawDecoded48::HotFormat::Rrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslAdd,
                             RawDecoded48::HotFormat::Rrrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslAdd,
                             RawDecoded48::HotFormat::Zrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslAdd,
                             RawDecoded48::HotFormat::Zrrici);
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x6)) {
        out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                      (((instruction >> 32) & 3) << 3));
        out.has_rb = true;
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslSubS,
                             RawDecoded48::HotFormat::Rrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslSubS,
                             RawDecoded48::HotFormat::Rrrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslSubU,
                             RawDecoded48::HotFormat::Rrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslSubU,
                             RawDecoded48::HotFormat::Rrrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslSub,
                             RawDecoded48::HotFormat::Rrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslSub,
                             RawDecoded48::HotFormat::Rrrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslSub,
                             RawDecoded48::HotFormat::Zrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LslSub,
                             RawDecoded48::HotFormat::Zrrici);
            return true;
          }
          return false;
        }
        return false;
      }
      if (((((instruction >> 28) & 0xf)) & (0xe)) == (0x2)) {
        out.rb = static_cast<int32_t>((((instruction >> 17) & 7) << 0) |
                                      (((instruction >> 32) & 3) << 3));
        out.has_rb = true;
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
            set_hot_metadata(out, RawDecoded48::HotOp::LsrAddS,
                             RawDecoded48::HotFormat::Rrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LsrAddS,
                             RawDecoded48::HotFormat::Rrrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LsrAddU,
                             RawDecoded48::HotFormat::Rrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LsrAddU,
                             RawDecoded48::HotFormat::Rrrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LsrAdd,
                             RawDecoded48::HotFormat::Rrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LsrAdd,
                             RawDecoded48::HotFormat::Rrrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LsrAdd,
                             RawDecoded48::HotFormat::Zrri);
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
            set_hot_metadata(out, RawDecoded48::HotOp::LsrAdd,
                             RawDecoded48::HotFormat::Zrrici);
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
            set_hot_metadata(out, RawDecoded48::HotOp::DivStep,
                             RawDecoded48::HotFormat::Rrrici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Movd,
                               RawDecoded48::HotFormat::Rrci);
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
              set_hot_metadata(out, RawDecoded48::HotOp::MulStep,
                               RawDecoded48::HotFormat::Rrrici);
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
              set_hot_metadata(out, RawDecoded48::HotOp::Swapd,
                               RawDecoded48::HotFormat::Rrci);
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

static inline bool eval_cc_kind_simple(RawDecoded48::CcKind cc_kind, bool zf,
                                       bool cf, uint32_t value_for_z) {
  (void)zf;

  switch (cc_kind) {
  case RawDecoded48::CcKind::True:
    return true;
  case RawDecoded48::CcKind::Zero:
    return value_for_z == 0u;
  case RawDecoded48::CcKind::NonZero:
    return value_for_z != 0u;
  case RawDecoded48::CcKind::Carry:
    return cf;
  case RawDecoded48::CcKind::NoCarry:
    return !cf;
  case RawDecoded48::CcKind::LessEqualUnsigned:
    return cf || (value_for_z == 0u);
  case RawDecoded48::CcKind::GreaterUnsigned:
    return !cf && (value_for_z != 0u);
  case RawDecoded48::CcKind::LessSigned:
    return static_cast<int32_t>(value_for_z) < 0;
  case RawDecoded48::CcKind::LessEqualSigned:
    return static_cast<int32_t>(value_for_z) <= 0;
  case RawDecoded48::CcKind::GreaterSigned:
    return static_cast<int32_t>(value_for_z) > 0;
  case RawDecoded48::CcKind::GreaterEqualSigned:
    return static_cast<int32_t>(value_for_z) >= 0;
  case RawDecoded48::CcKind::XZero:
    return zf && (value_for_z == 0u);
  case RawDecoded48::CcKind::XNonZero:
    return !(zf && (value_for_z == 0u));
  case RawDecoded48::CcKind::Max:
    return value_for_z == 32u;
  case RawDecoded48::CcKind::NotMax:
    return value_for_z != 32u;
  case RawDecoded48::CcKind::Even:
    return (value_for_z & 1u) == 0u;
  case RawDecoded48::CcKind::Odd:
    return (value_for_z & 1u) != 0u;
  default:
    return false;
  }
}

static inline bool eval_add_subword_no_carry(uint8_t bits, uint32_t lhs,
                                             uint32_t rhs) {
  if (bits < 5u || bits > 31u) {
    return false;
  }
  const uint64_t limit = 1ULL << bits;
  const uint64_t mask = limit - 1ULL;
  const uint64_t partial_sum =
      (static_cast<uint64_t>(lhs) & mask) + (static_cast<uint64_t>(rhs) & mask);
  return partial_sum < limit;
}

static inline bool eval_subc_cc_simple(RawDecoded48::CcKind cc_kind, bool zf,
                                       bool cf, uint32_t value, bool old_zf,
                                       bool overflow) {
  if (cc_kind == RawDecoded48::CcKind::XZero) {
    return (value == 0u) && old_zf;
  }
  if (cc_kind == RawDecoded48::CcKind::XNonZero) {
    return !((value == 0u) && old_zf);
  }
  if (cc_kind == RawDecoded48::CcKind::XLessEqualUnsigned) {
    return cf || old_zf;
  }
  if (cc_kind == RawDecoded48::CcKind::XGreaterUnsigned) {
    return cf && !old_zf;
  }

  const bool signed_less = (static_cast<int32_t>(value) < 0) || overflow;
  const bool equal64 = old_zf && (value == 0u);
  if (cc_kind == RawDecoded48::CcKind::XLessEqualSigned) {
    return signed_less || equal64;
  }
  if (cc_kind == RawDecoded48::CcKind::XGreaterSigned) {
    return !signed_less && !equal64;
  }

  return eval_cc_kind_simple(cc_kind, zf, cf, value);
}

/*
 * Execute one tasklet step.
 *
 * The current model intentionally keeps a compact fixed-field instruction
 * decode (opcode/rc/ra/rb/imm) while broadening opcode-family behavior:
 * ALU, shifts, control/flow, load/store, DMA, and perf counter ops.
 */
bool Pipeline::execute_once(IRAM &iram, WRAM &wram, DMAEngine &dma_engine,
                            MRAM &mram, int tasklet_idx) {
  const size_t tasklet_count =
      std::min(nr_tasklet_, static_cast<size_t>(kNumTasklets));
  if (tasklet_idx < 0 || static_cast<size_t>(tasklet_idx) >= tasklet_count) {
    return false;
  }

  const size_t t = static_cast<size_t>(tasklet_idx);
  const uint64_t tasklet_bit = 1ULL << static_cast<uint64_t>(tasklet_idx);
  if ((run_bits_ & tasklet_bit) == 0ULL ||
      (sleep_bits_ & tasklet_bit) != 0ULL) {
    return false;
  }

  const uint64_t pc = register_file_.pc_regs_[t];
  const size_t iram_slot64 = static_cast<size_t>(pc / 8u);
  if (iram_slot64 > static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
    return false;
  }

  const uint16_t iram_slot = static_cast<uint16_t>(iram_slot64);
  if (!iram.slot_in_bounds(iram_slot)) {
    return false;
  }

  const uint64_t inst48 = iram.read_word(iram_slot);
  if (inst48 == 0u) {
    return false;
  }

  const uint8_t opcode = byte(inst48, 40u);
  const uint8_t rc = static_cast<uint8_t>(byte(inst48, 32u) & 0x1Fu);
  const uint8_t ra = static_cast<uint8_t>(byte(inst48, 24u) & 0x1Fu);
  const uint8_t rb = static_cast<uint8_t>(byte(inst48, 16u) & 0x1Fu);
  const uint8_t imm8 = byte(inst48, 8u);
  const uint16_t imm16 = static_cast<uint16_t>(inst48 & 0xFFFFu);
  const int32_t imm16s = sign_extend16(imm16);
  const int32_t imm24s =
      sign_extend24(static_cast<uint32_t>(inst48 & 0xFFFFFFu));

  auto read_reg = [&](uint8_t idx) -> uint32_t {
    if (idx < 24u) {
      return register_file_.gp_regs_[t][idx];
    }

    switch (idx) {
    case 24u:
      return 0u;
    case 25u:
      return 1u;
    case 26u:
      return 0xFFFFFFFFu;
    case 27u:
      return 0x80000000u;
    case 28u:
      return static_cast<uint32_t>(tasklet_idx);
    case 29u:
      return static_cast<uint32_t>(tasklet_idx * 2);
    case 30u:
      return static_cast<uint32_t>(tasklet_idx * 4);
    case 31u:
      return static_cast<uint32_t>(tasklet_idx * 8);
    default:
      return 0u;
    }
  };

  auto write_reg = [&](uint8_t idx, uint32_t value) {
    if (idx < 24u) {
      register_file_.gp_regs_[t][idx] = value;
    }
  };

  auto read_dreg = [&](uint8_t reg_idx) -> uint64_t {
    const uint8_t slot = dreg_slot(static_cast<int>(reg_idx));
    if (slot >= 12u) {
      return 0u;
    }

    const uint8_t hi = static_cast<uint8_t>(slot * 2u);
    const uint8_t lo = static_cast<uint8_t>(hi + 1u);
    return (static_cast<uint64_t>(read_reg(hi)) << 32) |
           static_cast<uint64_t>(read_reg(lo));
  };

  auto write_dreg = [&](uint8_t reg_idx, uint64_t value) {
    const uint8_t slot = dreg_slot(static_cast<int>(reg_idx));
    if (slot >= 12u) {
      return;
    }

    const uint8_t hi = static_cast<uint8_t>(slot * 2u);
    const uint8_t lo = static_cast<uint8_t>(hi + 1u);
    write_reg(hi, static_cast<uint32_t>((value >> 32) & 0xFFFFFFFFu));
    write_reg(lo, static_cast<uint32_t>(value & 0xFFFFFFFFu));
  };

  auto set_zf = [&](uint32_t value) {
    register_file_.zf_regs_[t] = static_cast<uint8_t>(value == 0u);
  };

  auto set_cf = [&](bool value) {
    register_file_.cf_regs_[t] = static_cast<uint8_t>(value ? 1u : 0u);
  };

  auto addr_add = [&](uint32_t base, int32_t off) -> uint32_t {
    return static_cast<uint32_t>(
        static_cast<int64_t>(static_cast<int32_t>(base)) +
        static_cast<int64_t>(off));
  };

  const uint32_t ra_val = read_reg(ra);
  const uint32_t rb_val = read_reg(rb);
  const uint32_t imm_u32 = static_cast<uint32_t>(imm16s);
  const uint32_t op_b = (rb == 24u) ? imm_u32 : rb_val;

  uint64_t next_pc = pc + 8u;
  bool keep_running = true;

  RawDecoded48 decoded{};
  bool decoded_ok = false;
  {
    struct DecodedInstCacheEntry {
      const IRAM *iram{nullptr};
      uint16_t iram_slot{0u};
      uint64_t inst48{0u};
      RawDecoded48 decoded{};
    };

    // Keep per-worker decode caches alive for process lifetime so pool-worker
    // teardown does not pay large thread-local destruction costs.
    static thread_local auto *decode_cache =
        new std::array<DecodedInstCacheEntry, kDecodedInstCacheLineCount>();

    const size_t cache_idx = ((reinterpret_cast<uintptr_t>(&iram) >> 3u) ^
                              static_cast<size_t>(iram_slot)) &
                             (kDecodedInstCacheLineCount - 1u);
    DecodedInstCacheEntry &cache_entry = (*decode_cache)[cache_idx];
    if (cache_entry.iram == &iram && cache_entry.iram_slot == iram_slot &&
        cache_entry.inst48 == inst48) {
      decoded = cache_entry.decoded;
      decoded_ok = decoded.valid;
    } else {
      decoded_ok = decode_raw_word_48(inst48, decoded);
      if (decoded_ok) {
        populate_hot_metadata_48(decoded);
      }

      cache_entry.iram = &iram;
      cache_entry.iram_slot = iram_slot;
      cache_entry.inst48 = inst48;
      cache_entry.decoded = decoded;
    }
  }
  if (decoded_ok && decoded.valid) {
    const uint8_t dra = decoded.has_ra ? static_cast<uint8_t>(decoded.ra & 0x1F)
                                       : static_cast<uint8_t>(24u);
    const uint8_t drb = decoded.has_rb ? static_cast<uint8_t>(decoded.rb & 0x1F)
                                       : static_cast<uint8_t>(24u);
    const uint8_t drc = decoded.has_rc ? static_cast<uint8_t>(decoded.rc & 0x1F)
                                       : static_cast<uint8_t>(24u);
    const uint8_t ddc = decoded.has_dc ? static_cast<uint8_t>(decoded.dc & 0x1F)
                                       : static_cast<uint8_t>(0u);
    const uint8_t ddb = decoded.has_db ? static_cast<uint8_t>(decoded.db & 0x1F)
                                       : static_cast<uint8_t>(0u);

    const uint32_t vra = read_reg(dra);
    const uint32_t vrb = read_reg(drb);
    const uint32_t vimm = static_cast<uint32_t>(decoded.imm);
    int32_t voff = decoded.off;
    if (decoded.off_uses_sign_extend24) {
      voff = sign_extend24(static_cast<uint32_t>(decoded.off));
    }
    const bool big_endian = decoded.has_endian && (decoded.endian != 0);
    const auto cc_kind = decoded.cc_kind;

    auto take_cond_branch = [&](uint32_t value_for_cc) {
      if (decoded.has_cc && decoded.has_pc &&
          eval_cc_kind_simple(cc_kind, register_file_.zf_regs_[t] != 0u,
                              register_file_.cf_regs_[t] != 0u, value_for_cc)) {
        next_pc = static_cast<uint64_t>(static_cast<uint32_t>(decoded.pc)) * 8u;
      }
    };

    auto write_ext_dreg32 = [&](uint32_t value) {
      if (!decoded.has_dc) {
        return;
      }

      if (decoded.writeback_mode == RawDecoded48::WritebackMode::DregSign32) {
        write_dreg(ddc, static_cast<uint64_t>(
                            static_cast<int64_t>(static_cast<int32_t>(value))));
      } else if (decoded.writeback_mode ==
                 RawDecoded48::WritebackMode::DregZero32) {
        write_dreg(ddc, static_cast<uint64_t>(value));
      }
    };

    bool handled = false;
    if (decoded.hot_supported) {
      switch (decoded.hot_op) {
      case RawDecoded48::HotOp::Or:
      case RawDecoded48::HotOp::OrS:
      case RawDecoded48::HotOp::OrU: {
        const uint32_t rhs = decoded.has_rb ? vrb : vimm;
        const uint32_t x = vra | rhs;
        if (decoded.writeback_mode != RawDecoded48::WritebackMode::None) {
          write_ext_dreg32(x);
        } else if (!decoded.suppress_scalar_write && decoded.has_rc) {
          write_reg(drc, x);
        }
        set_zf(x);
        take_cond_branch(x);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Add: {
        uint32_t rhs = decoded.has_rb ? vrb : vimm;
        if (decoded.imm_is_signed8) {
          const int32_t imm_s8 =
              static_cast<int32_t>(static_cast<int8_t>(decoded.imm & 0xFF));
          rhs = static_cast<uint32_t>(imm_s8);
        }
        const uint64_t sum =
            static_cast<uint64_t>(vra) + static_cast<uint64_t>(rhs);
        const uint32_t x = static_cast<uint32_t>(sum);
        if (decoded.has_rc) {
          write_reg(drc, x);
        }
        set_zf(x);
        set_cf((sum >> 32u) != 0u);
        const bool add_cond =
            decoded.has_cc && decoded.has_pc &&
            decoded.cc_kind == RawDecoded48::CcKind::SubwordNoCarry &&
            eval_add_subword_no_carry(decoded.cc_subword_bits, vra, rhs);
        if (decoded.has_cc && decoded.has_pc && add_cond) {
          next_pc =
              static_cast<uint64_t>(static_cast<uint32_t>(decoded.pc)) * 8u;
        } else {
          take_cond_branch(x);
        }
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Addc: {
        const uint32_t rhs = decoded.has_rb ? vrb : vimm;
        const uint32_t carry_in = (register_file_.cf_regs_[t] != 0u) ? 1u : 0u;
        const uint64_t sum = static_cast<uint64_t>(vra) +
                             static_cast<uint64_t>(rhs) +
                             static_cast<uint64_t>(carry_in);
        const uint32_t x = static_cast<uint32_t>(sum);
        if (decoded.has_rc) {
          write_reg(drc, x);
        }
        set_zf(x);
        set_cf((sum >> 32u) != 0u);
        take_cond_branch(x);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::And:
      case RawDecoded48::HotOp::AndS:
      case RawDecoded48::HotOp::AndU: {
        const uint32_t rhs = decoded.has_rb ? vrb : vimm;
        const uint32_t x = decoded.imm_passthrough ? vimm : (vra & rhs);
        if (decoded.writeback_mode != RawDecoded48::WritebackMode::None) {
          write_ext_dreg32(x);
        } else if (decoded.has_rc) {
          write_reg(drc, x);
        }
        set_zf(x);
        take_cond_branch(x);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Sub: {
        const uint32_t lhs = decoded.sub_lhs_is_imm ? vimm : vra;
        const uint32_t rhs = decoded.sub_rhs_is_reg
                                 ? vrb
                                 : (decoded.sub_lhs_is_imm ? vra : vimm);
        const uint32_t x = static_cast<uint32_t>(lhs - rhs);
        set_zf(x);
        set_cf(lhs < rhs);

        if (decoded.sub_is_rrrc) {
          const bool cond =
              eval_cc_kind_simple(cc_kind, register_file_.zf_regs_[t] != 0u,
                                  register_file_.cf_regs_[t] != 0u, x);
          if (decoded.has_rc) {
            write_reg(drc, cond ? 1u : 0u);
          }
        } else if (decoded.has_rc) {
          write_reg(drc, x);
        }

        if (decoded.has_cc && decoded.has_pc && !decoded.sub_is_rrrc &&
            !decoded.sub_is_zrrc) {
          take_cond_branch(x);
        }
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Subc: {
        const uint32_t lhs = decoded.subc_lhs_is_imm ? vimm : vra;
        const uint32_t rhs_base = decoded.subc_rhs_is_reg
                                      ? vrb
                                      : (decoded.subc_lhs_is_imm ? vra : vimm);
        const bool old_zf = register_file_.zf_regs_[t] != 0u;
        const uint32_t borrow = (register_file_.cf_regs_[t] != 0u) ? 1u : 0u;
        const uint32_t rhs = static_cast<uint32_t>(rhs_base + borrow);
        const bool rhs_wrap = rhs < rhs_base;
        const uint32_t x = static_cast<uint32_t>(lhs - rhs);
        const bool s1 = ((lhs >> 31u) & 1u) != 0u;
        const bool s2 = ((rhs_base >> 31u) & 1u) != 0u;
        const bool sr = ((x >> 31u) & 1u) != 0u;
        const bool overflow = (s1 && !s2 && !sr) || (!s1 && s2 && sr);

        set_zf(x);
        set_cf((lhs < rhs) || rhs_wrap);

        if (decoded.subc_is_rrrc) {
          const bool cond = eval_subc_cc_simple(
              cc_kind, register_file_.zf_regs_[t] != 0u,
              register_file_.cf_regs_[t] != 0u, x, old_zf, overflow);
          if (decoded.has_rc) {
            write_reg(drc, cond ? 1u : 0u);
          }
        } else if (decoded.has_rc) {
          write_reg(drc, x);
        }

        if (decoded.has_cc && decoded.has_pc && !decoded.subc_is_rrrc &&
            !decoded.subc_is_zrrc) {
          const bool cond = eval_subc_cc_simple(
              cc_kind, register_file_.zf_regs_[t] != 0u,
              register_file_.cf_regs_[t] != 0u, x, old_zf, overflow);
          if (cond) {
            next_pc =
                static_cast<uint64_t>(static_cast<uint32_t>(decoded.pc)) * 8u;
          }
        }
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::MulStep: {
        if (decoded.hot_fmt == RawDecoded48::HotFormat::Rrrici) {
          const uint8_t db_even_reg = decoded.has_db ? ddb : ddc;
          const uint8_t db_odd_reg = static_cast<uint8_t>(db_even_reg + 1u);
          const uint8_t dc_even_reg = ddc;
          const uint8_t dc_odd_reg = static_cast<uint8_t>(dc_even_reg + 1u);

          const uint32_t db_even = read_reg(db_even_reg);
          const uint32_t db_odd = read_reg(db_odd_reg);
          const uint32_t sh = static_cast<uint32_t>(decoded.shift) & 31u;

          const uint32_t new_even = db_even >> 1u;
          uint32_t new_odd = read_reg(dc_odd_reg);
          if ((db_even & 1u) != 0u) {
            new_odd = static_cast<uint32_t>(db_odd + (vra << sh));
          }

          write_reg(dc_even_reg, new_even);
          write_reg(dc_odd_reg, new_odd);
          set_zf(new_even);
          set_cf(false);
          take_cond_branch(new_even);
        } else {
          const uint8_t src_even = static_cast<uint8_t>(ddb & 0x1Eu);
          const uint8_t src_odd = static_cast<uint8_t>(src_even + 1u);
          const uint8_t dst_even = static_cast<uint8_t>(ddc & 0x1Eu);
          const uint8_t dst_odd = static_cast<uint8_t>(dst_even + 1u);
          const uint32_t dbe = read_reg(src_even);
          const uint32_t dbo = read_reg(src_odd);
          const uint32_t sh = static_cast<uint32_t>(decoded.shift) & 31u;
          const uint32_t x = dbe >> 1u;
          uint32_t y = dbo;
          if ((dbe & 1u) != 0u) {
            y = static_cast<uint32_t>(dbo + (vra << sh));
          }

          write_reg(dst_even, x);
          write_reg(dst_odd, y);
          set_zf(x);
        }
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::DivStep: {
        const uint8_t db_even_reg = decoded.has_db ? ddb : ddc;
        const uint8_t db_odd_reg = static_cast<uint8_t>(db_even_reg + 1u);
        const uint8_t dc_even_reg = ddc;
        const uint8_t dc_odd_reg = static_cast<uint8_t>(dc_even_reg + 1u);

        const uint32_t db_even = read_reg(db_even_reg);
        const uint32_t db_odd = read_reg(db_odd_reg);
        const uint32_t sh = static_cast<uint32_t>(decoded.shift) & 31u;
        const uint32_t shifted = static_cast<uint32_t>(vra << sh);
        const uint32_t trial = static_cast<uint32_t>(db_odd - shifted);

        uint32_t new_even = 0u;
        uint32_t new_odd = 0u;
        if (db_odd >= shifted) {
          new_even = static_cast<uint32_t>((db_even << 1u) | 1u);
          new_odd = trial;
        } else {
          new_even = static_cast<uint32_t>(db_even << 1u);
          new_odd = read_reg(dc_odd_reg);
        }

        write_reg(dc_even_reg, new_even);
        write_reg(dc_odd_reg, new_odd);
        set_zf(trial);
        set_cf(false);
        take_cond_branch(trial);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Acquire:
      case RawDecoded48::HotOp::Release: {
        const uint32_t addr = static_cast<uint32_t>(vra + vimm);
        const uint8_t index = atomic_bit_index(addr);
        const uint8_t byte_idx = static_cast<uint8_t>(index / 8u);
        const uint8_t bit_idx = static_cast<uint8_t>(index % 8u);
        const uint8_t mask = static_cast<uint8_t>(1u << bit_idx);
        uint32_t result = 0u;
        if (!decoded.acquire_is_release) {
          const bool was_free = (atomic_bits_[byte_idx] & mask) == 0u;
          atomic_bits_[byte_idx] =
              static_cast<uint8_t>(atomic_bits_[byte_idx] | mask);
          result = was_free ? 0u : 1u;
        } else {
          const bool was_held = (atomic_bits_[byte_idx] & mask) != 0u;
          atomic_bits_[byte_idx] =
              static_cast<uint8_t>(atomic_bits_[byte_idx] & ~mask);
          result = was_held ? 0u : 1u;
        }
        if (decoded.has_rc) {
          write_reg(drc, result);
        }
        set_zf(result);
        take_cond_branch(result);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Resume: {
        const uint32_t addr = static_cast<uint32_t>(vra + vimm);
        const uint8_t index =
            static_cast<uint8_t>(((addr >> 8u) & 0xFFu) ^ (addr & 0xFFu));

        uint32_t result = 1u;
        if (index < tasklet_count) {
          const uint64_t bit = 1ULL << static_cast<uint64_t>(index);
          const bool is_sleeping = (sleep_bits_ & bit) != 0u;
          const bool is_stopped = (run_bits_ & bit) == 0u;
          if (is_sleeping || is_stopped) {
            sleep_bits_ &= ~bit;
            run_bits_ |= bit;
            replay_bits_ &= ~bit;
            result = 0u;
          }
        }

        if (decoded.has_rc) {
          write_reg(drc, result);
        }
        set_zf(result);
        take_cond_branch(result);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Boot: {
        const uint32_t addr = static_cast<uint32_t>(vra + vimm);
        const uint8_t index =
            static_cast<uint8_t>(((addr >> 8u) & 0xFFu) ^ (addr & 0xFFu));

        uint32_t result = 1u;
        if (index < tasklet_count) {
          const uint64_t bit = 1ULL << static_cast<uint64_t>(index);
          const bool is_running = (run_bits_ & bit) != 0u;
          const bool is_sleeping = (sleep_bits_ & bit) != 0u;
          if (!is_running && !is_sleeping) {
            run_bits_ |= bit;
            sleep_bits_ &= ~bit;
            replay_bits_ &= ~bit;
            register_file_.gp_regs_[index].fill(0u);
            register_file_.pc_regs_[index] = 0u;
            register_file_.zf_regs_[index] = 0u;
            register_file_.cf_regs_[index] = 0u;
            result = 0u;
          }
        }

        if (decoded.has_rc) {
          write_reg(drc, result);
        }
        set_zf(result);
        take_cond_branch(result);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Call: {
        const int64_t target = static_cast<int64_t>(static_cast<int32_t>(vra)) +
                               static_cast<int64_t>(voff);
        if (decoded.has_rc) {
          write_reg(drc, static_cast<uint32_t>(pc / 8u + 1u));
        }
        next_pc = static_cast<uint64_t>(static_cast<uint32_t>(target)) * 8u;
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Ldma:
      case RawDecoded48::HotOp::Sdma: {
        const uint32_t w = vra & 0xFFFFF8u;
        const uint32_t m = vrb & 0xFFFFFFF8u;
        const uint32_t n =
            (1u +
             ((static_cast<uint32_t>(decoded.immDma) + ((vra >> 24u) & 0xFFu)) &
              0xFFu))
            << 3u;
        if (!decoded.dma_to_mram) {
          (void)dma_engine.dma_to_wram_from_mram(wram, mram, n, w, m);
        } else {
          (void)dma_engine.dma_to_mram_from_wram(mram, wram, n, m, w);
        }
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Lsl:
      case RawDecoded48::HotOp::LslS:
      case RawDecoded48::HotOp::LslU:
      case RawDecoded48::HotOp::Lslx:
      case RawDecoded48::HotOp::LslxS:
      case RawDecoded48::HotOp::LslxU:
      case RawDecoded48::HotOp::Lsr:
      case RawDecoded48::HotOp::LsrS:
      case RawDecoded48::HotOp::LsrU:
      case RawDecoded48::HotOp::Lsrx:
      case RawDecoded48::HotOp::LsrxS:
      case RawDecoded48::HotOp::LsrxU: {
        const uint32_t sh = decoded.has_rb
                                ? (vrb & 31u)
                                : (static_cast<uint32_t>(decoded.shift) & 31u);
        uint32_t x = 0u;
        if (decoded.hot_op == RawDecoded48::HotOp::Lsl ||
            decoded.hot_op == RawDecoded48::HotOp::LslS ||
            decoded.hot_op == RawDecoded48::HotOp::LslU) {
          x = static_cast<uint32_t>(vra << sh);
        } else if (decoded.hot_op == RawDecoded48::HotOp::Lslx ||
                   decoded.hot_op == RawDecoded48::HotOp::LslxS ||
                   decoded.hot_op == RawDecoded48::HotOp::LslxU) {
          x = (sh == 0u) ? 0u : static_cast<uint32_t>(vra >> (32u - sh));
        } else if (decoded.hot_op == RawDecoded48::HotOp::Lsrx ||
                   decoded.hot_op == RawDecoded48::HotOp::LsrxS ||
                   decoded.hot_op == RawDecoded48::HotOp::LsrxU) {
          x = (sh == 0u) ? 0u : static_cast<uint32_t>(vra << (32u - sh));
        } else {
          x = static_cast<uint32_t>(vra >> sh);
        }

        const bool write_bool = decoded.has_cc && !decoded.has_pc &&
                                (decoded.has_rc || decoded.has_dc);
        bool cond = false;
        if (write_bool) {
          cond = eval_cc_kind_simple(cc_kind, x == 0u,
                                     register_file_.cf_regs_[t] != 0u, x);
        }

        if (decoded.has_dc) {
          if (write_bool) {
            write_dreg(ddc, cond ? 1u : 0u);
          } else {
            const bool sign_extend =
                (decoded.hot_op == RawDecoded48::HotOp::LslS ||
                 decoded.hot_op == RawDecoded48::HotOp::LslxS ||
                 decoded.hot_op == RawDecoded48::HotOp::LsrS ||
                 decoded.hot_op == RawDecoded48::HotOp::LsrxS);
            const uint64_t ext =
                sign_extend ? static_cast<uint64_t>(
                                  static_cast<int64_t>(static_cast<int32_t>(x)))
                            : static_cast<uint64_t>(x);
            write_dreg(ddc, ext);
          }
        } else if (decoded.has_rc) {
          write_reg(drc, write_bool ? (cond ? 1u : 0u) : x);
        }

        set_zf(x);
        if (decoded.has_cc && decoded.has_pc) {
          take_cond_branch(x);
        }
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Movd:
      case RawDecoded48::HotOp::Swapd: {
        const uint64_t rhs = read_dreg(ddb);
        if (!decoded.movd_is_swap) {
          write_dreg(ddc, rhs);
        } else {
          const uint64_t lhs = read_dreg(ddc);
          write_dreg(ddc, rhs);
          write_dreg(ddb, lhs);
        }

        const uint32_t cc_value = (rhs == 0u) ? 0u : 1u;
        set_zf(cc_value);
        take_cond_branch(cc_value);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Clz:
      case RawDecoded48::HotOp::Clo: {
        uint32_t src = vra;
        if (decoded.clz_invert_input) {
          src = ~src;
        }
        const uint32_t x =
            (src == 0u) ? 32u : static_cast<uint32_t>(__builtin_clz(src));
        if (decoded.has_rc) {
          write_reg(drc, x);
        }
        set_zf(x);
        take_cond_branch(x);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::MulUlUl:
      case RawDecoded48::HotOp::MulUlUh:
      case RawDecoded48::HotOp::MulUhUl:
      case RawDecoded48::HotOp::MulUhUh: {
        const uint8_t a_low = static_cast<uint8_t>(vra & 0xFFu);
        const uint8_t a_high = static_cast<uint8_t>((vra >> 8u) & 0xFFu);
        const uint8_t b_low = static_cast<uint8_t>(vrb & 0xFFu);
        const uint8_t b_high = static_cast<uint8_t>((vrb >> 8u) & 0xFFu);

        uint32_t x = 0u;
        switch (decoded.mul_byte_pair_mode) {
        case RawDecoded48::MulBytePairMode::UlUl:
          x = static_cast<uint32_t>(a_low) * static_cast<uint32_t>(b_low);
          break;
        case RawDecoded48::MulBytePairMode::UlUh:
          x = static_cast<uint32_t>(a_low) * static_cast<uint32_t>(b_high);
          break;
        case RawDecoded48::MulBytePairMode::UhUl:
          x = static_cast<uint32_t>(a_high) * static_cast<uint32_t>(b_low);
          break;
        case RawDecoded48::MulBytePairMode::UhUh:
          x = static_cast<uint32_t>(a_high) * static_cast<uint32_t>(b_high);
          break;
        default:
          break;
        }

        if (decoded.has_rc) {
          write_reg(drc, x);
        }
        set_zf(x);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::LslAdd:
      case RawDecoded48::HotOp::LslAddS:
      case RawDecoded48::HotOp::LslAddU: {
        const uint32_t sh = static_cast<uint32_t>(decoded.shift) & 31u;
        const uint32_t shifted = static_cast<uint32_t>(vra << sh);
        const uint32_t x = static_cast<uint32_t>(vrb + shifted);

        if (decoded.writeback_mode != RawDecoded48::WritebackMode::None) {
          write_ext_dreg32(x);
        } else if (!decoded.suppress_scalar_write && decoded.has_rc) {
          write_reg(drc, x);
        }
        set_zf(shifted);
        take_cond_branch(x);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::LslSub:
      case RawDecoded48::HotOp::LslSubS:
      case RawDecoded48::HotOp::LslSubU: {
        const uint32_t sh = static_cast<uint32_t>(decoded.shift) & 31u;
        const uint32_t shifted = static_cast<uint32_t>(vra << sh);
        const uint32_t x = static_cast<uint32_t>(vrb - shifted);
        if (decoded.writeback_mode != RawDecoded48::WritebackMode::None) {
          write_ext_dreg32(x);
        } else if (!decoded.suppress_scalar_write && decoded.has_rc) {
          write_reg(drc, x);
        }
        set_zf(shifted);
        set_cf(vrb < shifted);
        take_cond_branch(x);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::LsrAdd:
      case RawDecoded48::HotOp::LsrAddS:
      case RawDecoded48::HotOp::LsrAddU: {
        const uint32_t sh = static_cast<uint32_t>(decoded.shift) & 31u;
        const uint32_t shifted = static_cast<uint32_t>(vra >> sh);
        const uint32_t x = static_cast<uint32_t>(vrb + shifted);

        if (decoded.writeback_mode != RawDecoded48::WritebackMode::None) {
          write_ext_dreg32(x);
        } else if (!decoded.suppress_scalar_write && decoded.has_rc) {
          write_reg(drc, x);
        }

        set_zf(shifted);
        take_cond_branch(x);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Lbu: {
        const uint32_t addr = addr_add(vra, voff);
        const uint32_t x = static_cast<uint32_t>(wram.load_u8(addr));
        if (decoded.has_rc) {
          write_reg(drc, x);
        }
        set_zf(x);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Lbs: {
        const uint32_t addr = addr_add(vra, voff);
        const int8_t signed_byte = static_cast<int8_t>(wram.load_u8(addr));
        const uint32_t x =
            static_cast<uint32_t>(static_cast<int32_t>(signed_byte));
        if (decoded.has_rc) {
          write_reg(drc, x);
        }
        set_zf(x);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Lw:
      case RawDecoded48::HotOp::LwS: {
        const uint32_t addr = addr_add(vra, voff);
        const uint32_t raw32 = wram.load_u32(addr, big_endian);
        if (decoded.has_rc) {
          write_reg(drc, raw32);
        }
        write_ext_dreg32(raw32);
        set_zf(raw32);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Ld: {
        const uint32_t addr = addr_add(vra, voff);
        const uint64_t x = wram.load_u64(addr, big_endian);
        if (decoded.has_dc) {
          write_dreg(ddc, x);
        }
        set_zf(static_cast<uint32_t>((x | (x >> 32u)) & 0xFFFFFFFFu));
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Sw: {
        const uint32_t addr = addr_add(vra, voff);
        const uint32_t x =
            (decoded.store_source_mode == RawDecoded48::StoreSourceMode::Imm32)
                ? static_cast<uint32_t>(decoded.imm)
                : vrb;
        wram.store_u32(addr, x, big_endian);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Sb: {
        const uint32_t addr = addr_add(vra, voff);
        const uint32_t x =
            (decoded.store_source_mode == RawDecoded48::StoreSourceMode::Imm32)
                ? vimm
                : vrb;
        wram.store_u8(addr, static_cast<uint8_t>(x & 0xFFu));
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Sh: {
        const uint32_t addr = addr_add(vra, voff);
        const uint32_t x =
            (decoded.store_source_mode == RawDecoded48::StoreSourceMode::Imm32)
                ? vimm
                : vrb;
        wram.store_u16(addr, static_cast<uint16_t>(x & 0xFFFFu), false);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Sd: {
        const uint32_t addr = addr_add(vra, voff);
        const uint64_t x =
            (decoded.store_source_mode == RawDecoded48::StoreSourceMode::Imm32)
                ? static_cast<uint64_t>(static_cast<uint32_t>(decoded.imm))
                : read_dreg(ddb);
        wram.store_u64(addr, x, big_endian);
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Time:
        if (decoded.has_rc) {
          write_reg(drc, perf_counter_read32());
        }
        handled = true;
        break;
      case RawDecoded48::HotOp::TimeCfg: {
        const uint32_t cfg_value = decoded.time_cfg_uses_ra ? vra : vrb;
        if (decoded.has_rc) {
          write_reg(drc, perf_counter_config(cfg_value));
        } else {
          (void)perf_counter_config(cfg_value);
        }
        handled = true;
        break;
      }
      case RawDecoded48::HotOp::Stop:
        take_cond_branch(1u);
        keep_running = false;
        handled = true;
        break;
      case RawDecoded48::HotOp::Nop:
        handled = true;
        break;
      case RawDecoded48::HotOp::Fault:
        keep_running = false;
        handled = true;
        break;
      case RawDecoded48::HotOp::None:
      default:
        break;
      }
    }

    if (handled) {
      register_file_.pc_regs_[t] = next_pc;
      if (!keep_running) {
        run_bits_ &= ~tasklet_bit;
        sleep_bits_ &= ~tasklet_bit;
        replay_bits_ &= ~tasklet_bit;
      }
      perf_counter_retire_step(false);
      return keep_running;
    }
  }

  switch (opcode) {
  case OP_ACQUIRE: {
    const uint32_t addr = static_cast<uint32_t>(ra_val + imm_u32);
    const uint8_t index = atomic_bit_index(addr);
    const uint8_t byte_idx = static_cast<uint8_t>(index / 8u);
    const uint8_t bit_idx = static_cast<uint8_t>(index % 8u);
    const uint8_t mask = static_cast<uint8_t>(1u << bit_idx);

    const bool was_free = (atomic_bits_[byte_idx] & mask) == 0u;
    atomic_bits_[byte_idx] =
        static_cast<uint8_t>(atomic_bits_[byte_idx] | mask);

    const uint32_t result = was_free ? 0u : 1u;
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_RELEASE: {
    const uint32_t addr = static_cast<uint32_t>(ra_val + imm_u32);
    const uint8_t index = atomic_bit_index(addr);
    const uint8_t byte_idx = static_cast<uint8_t>(index / 8u);
    const uint8_t bit_idx = static_cast<uint8_t>(index % 8u);
    const uint8_t mask = static_cast<uint8_t>(1u << bit_idx);

    const bool was_held = (atomic_bits_[byte_idx] & mask) != 0u;
    atomic_bits_[byte_idx] =
        static_cast<uint8_t>(atomic_bits_[byte_idx] & ~mask);

    const uint32_t result = was_held ? 0u : 1u;
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_ADD: {
    const uint64_t sum =
        static_cast<uint64_t>(ra_val) + static_cast<uint64_t>(op_b);
    const uint32_t result = static_cast<uint32_t>(sum);
    write_reg(rc, result);
    set_zf(result);
    set_cf((sum >> 32u) != 0u);
    break;
  }
  case OP_ADDC: {
    const uint64_t sum =
        static_cast<uint64_t>(ra_val) + static_cast<uint64_t>(op_b) +
        static_cast<uint64_t>(register_file_.cf_regs_[t] != 0u);
    const uint32_t result = static_cast<uint32_t>(sum);
    write_reg(rc, result);
    set_zf(result);
    set_cf((sum >> 32u) != 0u);
    break;
  }
  case OP_AND: {
    const uint32_t result = ra_val & op_b;
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_ANDN: {
    const uint32_t result = ra_val & ~op_b;
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_ASR: {
    const uint32_t sh = op_b & 31u;
    const uint32_t result =
        static_cast<uint32_t>(static_cast<int32_t>(ra_val) >> sh);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_CAO: {
    const uint32_t result = static_cast<uint32_t>(__builtin_popcount(ra_val));
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_CLO: {
    const uint32_t result = count_leading_ones_u32(ra_val);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_CLS: {
    const uint32_t result = count_leading_sign_u32(ra_val);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_CLZ: {
    const uint32_t result =
        (ra_val == 0u) ? 32u : static_cast<uint32_t>(__builtin_clz(ra_val));
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_CMPB4: {
    uint32_t matches = 0u;
    for (uint32_t b = 0u; b < 4u; ++b) {
      const uint8_t lhs = static_cast<uint8_t>((ra_val >> (8u * b)) & 0xFFu);
      const uint8_t rhs = static_cast<uint8_t>((op_b >> (8u * b)) & 0xFFu);
      if (lhs == rhs) {
        ++matches;
      }
    }
    write_reg(rc, matches);
    set_zf(matches);
    break;
  }
  case OP_DIV_STEP: {
    const uint32_t result =
        (op_b == 0u) ? 0u : static_cast<uint32_t>(ra_val / op_b);
    write_reg(rc, result);
    set_zf(result);
    set_cf(op_b == 0u);
    break;
  }
  case OP_EXTSB: {
    const int32_t value =
        static_cast<int32_t>(static_cast<int8_t>(ra_val & 0xFFu));
    const uint32_t result = static_cast<uint32_t>(value);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_EXTSH: {
    const int32_t value =
        static_cast<int32_t>(static_cast<int16_t>(ra_val & 0xFFFFu));
    const uint32_t result = static_cast<uint32_t>(value);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_EXTUB: {
    const uint32_t result = ra_val & 0xFFu;
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_EXTUH: {
    const uint32_t result = ra_val & 0xFFFFu;
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_LSL: {
    const uint32_t result = static_cast<uint32_t>(ra_val << (op_b & 31u));
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_LSL_ADD: {
    const uint32_t shifted = static_cast<uint32_t>(ra_val << (imm8 & 31u));
    const uint32_t result = static_cast<uint32_t>(rb_val + shifted);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_LSL_SUB: {
    const uint32_t shifted = static_cast<uint32_t>(ra_val << (imm8 & 31u));
    const uint32_t result = static_cast<uint32_t>(rb_val - shifted);
    write_reg(rc, result);
    set_zf(result);
    set_cf(rb_val < shifted);
    break;
  }
  case OP_LSL1: {
    const uint32_t result = static_cast<uint32_t>(ra_val << 1u);
    write_reg(rc, result);
    set_zf(result);
    set_cf((ra_val & 0x80000000u) != 0u);
    break;
  }
  case OP_LSL1X: {
    const bool old_cf = register_file_.cf_regs_[t] != 0u;
    const uint32_t result =
        static_cast<uint32_t>((ra_val << 1u) | static_cast<uint32_t>(old_cf));
    write_reg(rc, result);
    set_zf(result);
    set_cf((ra_val & 0x80000000u) != 0u);
    break;
  }
  case OP_LSLX: {
    const uint32_t sh = imm8 & 31u;
    const uint32_t result =
        (sh == 0u) ? 0u : static_cast<uint32_t>(ra_val >> (32u - sh));
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_LSR: {
    const uint32_t result = static_cast<uint32_t>(ra_val >> (op_b & 31u));
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_LSR_ADD: {
    const uint32_t shifted = static_cast<uint32_t>(ra_val >> (imm8 & 31u));
    const uint32_t result = static_cast<uint32_t>(rb_val + shifted);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_LSR1: {
    const uint32_t result = static_cast<uint32_t>(ra_val >> 1u);
    write_reg(rc, result);
    set_zf(result);
    set_cf((ra_val & 0x1u) != 0u);
    break;
  }
  case OP_LSR1X: {
    const bool old_cf = register_file_.cf_regs_[t] != 0u;
    const uint32_t result = static_cast<uint32_t>(
        (ra_val >> 1u) | (static_cast<uint32_t>(old_cf) << 31u));
    write_reg(rc, result);
    set_zf(result);
    set_cf((ra_val & 0x1u) != 0u);
    break;
  }
  case OP_LSRX: {
    const uint32_t sh = imm8 & 31u;
    const uint32_t result =
        (sh == 0u) ? 0u : static_cast<uint32_t>(ra_val << (32u - sh));
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_MUL_SH_SH:
  case OP_MUL_SH_SL:
  case OP_MUL_SH_UH:
  case OP_MUL_SH_UL:
  case OP_MUL_SL_SH:
  case OP_MUL_SL_SL:
  case OP_MUL_SL_UH:
  case OP_MUL_SL_UL:
  case OP_MUL_UH_UH:
  case OP_MUL_UH_UL:
  case OP_MUL_UL_UH:
  case OP_MUL_UL_UL: {
    const int8_t ra_sl = static_cast<int8_t>(ra_val & 0xFFu);
    const int8_t ra_sh = static_cast<int8_t>((ra_val >> 8u) & 0xFFu);
    const int8_t rb_sl = static_cast<int8_t>(op_b & 0xFFu);
    const int8_t rb_sh = static_cast<int8_t>((op_b >> 8u) & 0xFFu);
    const uint8_t ra_ul = static_cast<uint8_t>(ra_val & 0xFFu);
    const uint8_t ra_uh = static_cast<uint8_t>((ra_val >> 8u) & 0xFFu);
    const uint8_t rb_ul = static_cast<uint8_t>(op_b & 0xFFu);
    const uint8_t rb_uh = static_cast<uint8_t>((op_b >> 8u) & 0xFFu);

    uint32_t result = 0u;
    switch (opcode) {
    case OP_MUL_SH_SH:
      result = static_cast<uint32_t>(static_cast<int32_t>(ra_sh) *
                                     static_cast<int32_t>(rb_sh));
      break;
    case OP_MUL_SH_SL:
      result = static_cast<uint32_t>(static_cast<int32_t>(ra_sh) *
                                     static_cast<int32_t>(rb_sl));
      break;
    case OP_MUL_SH_UH:
      result = static_cast<uint32_t>(static_cast<int32_t>(ra_sh) *
                                     static_cast<int32_t>(rb_uh));
      break;
    case OP_MUL_SH_UL:
      result = static_cast<uint32_t>(static_cast<int32_t>(ra_sh) *
                                     static_cast<int32_t>(rb_ul));
      break;
    case OP_MUL_SL_SH:
      result = static_cast<uint32_t>(static_cast<int32_t>(ra_sl) *
                                     static_cast<int32_t>(rb_sh));
      break;
    case OP_MUL_SL_SL:
      result = static_cast<uint32_t>(static_cast<int32_t>(ra_sl) *
                                     static_cast<int32_t>(rb_sl));
      break;
    case OP_MUL_SL_UH:
      result = static_cast<uint32_t>(static_cast<int32_t>(ra_sl) *
                                     static_cast<int32_t>(rb_uh));
      break;
    case OP_MUL_SL_UL:
      result = static_cast<uint32_t>(static_cast<int32_t>(ra_sl) *
                                     static_cast<int32_t>(rb_ul));
      break;
    case OP_MUL_UH_UH:
      result = static_cast<uint32_t>(ra_uh) * static_cast<uint32_t>(rb_uh);
      break;
    case OP_MUL_UH_UL:
      result = static_cast<uint32_t>(ra_uh) * static_cast<uint32_t>(rb_ul);
      break;
    case OP_MUL_UL_UH:
      result = static_cast<uint32_t>(ra_ul) * static_cast<uint32_t>(rb_uh);
      break;
    case OP_MUL_UL_UL:
      result = static_cast<uint32_t>(ra_ul) * static_cast<uint32_t>(rb_ul);
      break;
    default:
      break;
    }

    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_MUL_STEP: {
    const uint64_t prod =
        static_cast<uint64_t>(ra_val) * static_cast<uint64_t>(op_b);
    const uint32_t result = static_cast<uint32_t>(prod);
    write_reg(rc, result);
    set_zf(result);
    set_cf((prod >> 32u) != 0u);
    break;
  }
  case OP_NAND: {
    const uint32_t result = ~(ra_val & op_b);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_NOR: {
    const uint32_t result = ~(ra_val | op_b);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_NXOR: {
    const uint32_t result = ~(ra_val ^ op_b);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_OR: {
    const uint32_t result = ra_val | op_b;
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_ORN: {
    const uint32_t result = ra_val | ~op_b;
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_ROL: {
    const uint32_t result = rotl32(ra_val, op_b & 31u);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_ROL_ADD: {
    const uint32_t rotated = rotl32(ra_val, imm8 & 31u);
    const uint32_t result = static_cast<uint32_t>(rb_val + rotated);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_ROR: {
    const uint32_t result = rotr32(ra_val, op_b & 31u);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_RSUB: {
    const uint32_t result = static_cast<uint32_t>(op_b - ra_val);
    write_reg(rc, result);
    set_zf(result);
    set_cf(op_b < ra_val);
    break;
  }
  case OP_RSUBC: {
    const uint32_t borrow = (register_file_.cf_regs_[t] != 0u) ? 1u : 0u;
    const uint32_t rhs = static_cast<uint32_t>(ra_val + borrow);
    const uint32_t result = static_cast<uint32_t>(op_b - rhs);
    write_reg(rc, result);
    set_zf(result);
    set_cf(op_b < rhs);
    break;
  }
  case OP_SUB: {
    const uint32_t result = static_cast<uint32_t>(ra_val - op_b);
    write_reg(rc, result);
    set_zf(result);
    set_cf(ra_val < op_b);
    break;
  }
  case OP_SUBC: {
    const uint32_t borrow = (register_file_.cf_regs_[t] != 0u) ? 1u : 0u;
    const uint32_t rhs = static_cast<uint32_t>(op_b + borrow);
    const uint32_t result = static_cast<uint32_t>(ra_val - rhs);
    write_reg(rc, result);
    set_zf(result);
    set_cf(ra_val < rhs);
    break;
  }
  case OP_XOR: {
    const uint32_t result = ra_val ^ op_b;
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_BOOT: {
    const uint32_t addr = static_cast<uint32_t>(ra_val + imm_u32);
    const uint8_t tid =
        static_cast<uint8_t>(((addr >> 8u) & 0xFFu) ^ (addr & 0xFFu));
    uint32_t result = 1u;
    if (tid < tasklet_count) {
      const uint64_t other_bit = 1ULL << tid;
      if ((run_bits_ & other_bit) == 0u && (sleep_bits_ & other_bit) == 0u) {
        register_file_.gp_regs_[tid].fill(0u);
        register_file_.pc_regs_[tid] = 0u;
        register_file_.zf_regs_[tid] = 0u;
        register_file_.cf_regs_[tid] = 0u;
        run_bits_ |= other_bit;
        result = 0u;
      }
    }
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_RESUME: {
    const int32_t tid = static_cast<int32_t>(ra_val) + imm16s;
    uint32_t result = 1u;
    if (tid >= 0 && static_cast<size_t>(tid) < tasklet_count) {
      const uint64_t other_bit = 1ULL << static_cast<uint64_t>(tid);
      const bool is_sleeping = (sleep_bits_ & other_bit) != 0u;
      const bool is_stopped = (run_bits_ & other_bit) == 0u;
      if (is_sleeping || is_stopped) {
        sleep_bits_ &= ~other_bit;
        run_bits_ |= other_bit;
        replay_bits_ &= ~other_bit;
        result = 0u;
      }
    }
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_STOP:
    keep_running = false;
    break;
  case OP_CALL: {
    int64_t target_slots = 0;
    if (rb != 24u) {
      target_slots = static_cast<int64_t>(static_cast<int32_t>(ra_val)) +
                     static_cast<int64_t>(static_cast<int32_t>(rb_val));
    } else {
      target_slots = static_cast<int64_t>(static_cast<int32_t>(ra_val)) +
                     static_cast<int64_t>(imm24s);
    }

    write_reg(rc, static_cast<uint32_t>((pc / 8u) + 1u));
    if (target_slots < 0) {
      keep_running = false;
      break;
    }
    next_pc = static_cast<uint64_t>(static_cast<uint32_t>(target_slots)) * 8u;
    break;
  }
  case OP_FAULT:
    keep_running = false;
    break;
  case OP_NOP:
    break;
  case OP_SATS: {
    const uint32_t result = saturating_add_s32(ra_val, op_b);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_MOVD: {
    const uint64_t value = read_dreg(ra);
    write_dreg(rc, value);
    set_zf(static_cast<uint32_t>((value == 0u) ? 0u : 1u));
    break;
  }
  case OP_SWAPD: {
    const uint64_t lhs = read_dreg(rc);
    const uint64_t rhs = read_dreg(ra);
    write_dreg(rc, rhs);
    write_dreg(ra, lhs);
    set_zf(static_cast<uint32_t>((rhs == 0u) ? 0u : 1u));
    break;
  }
  case OP_HASH: {
    uint32_t x = ra_val;
    x ^= op_b + 0x9E3779B9u + (x << 6u) + (x >> 2u);
    write_reg(rc, x);
    set_zf(x);
    break;
  }
  case OP_TIME: {
    const uint32_t value = perf_counter_read32();
    write_reg(rc, value);
    set_zf(value);
    break;
  }
  case OP_TIME_CFG: {
    const uint32_t old = perf_counter_config(op_b & 0x7u);
    write_reg(rc, old);
    set_zf(old);
    break;
  }
  case OP_LBS: {
    const uint32_t addr = addr_add(ra_val, imm16s);
    const int8_t value = static_cast<int8_t>(wram.load_u8(addr));
    const uint32_t result = static_cast<uint32_t>(static_cast<int32_t>(value));
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_LBU: {
    const uint32_t addr = addr_add(ra_val, imm16s);
    const uint32_t result = static_cast<uint32_t>(wram.load_u8(addr));
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_LD: {
    const uint32_t addr = addr_add(ra_val, imm16s);
    const uint64_t value = wram.load_u64(addr, false);
    write_dreg(rc, value);
    set_zf(static_cast<uint32_t>(value == 0u ? 0u : 1u));
    break;
  }
  case OP_LHS: {
    const uint32_t addr = addr_add(ra_val, imm16s);
    const int16_t value = static_cast<int16_t>(wram.load_u16(addr, false));
    const uint32_t result = static_cast<uint32_t>(static_cast<int32_t>(value));
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_LHU: {
    const uint32_t addr = addr_add(ra_val, imm16s);
    const uint32_t result = wram.load_u16(addr, false);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_LW: {
    const uint32_t addr = addr_add(ra_val, imm16s);
    const uint32_t result = wram.load_u32(addr, false);
    write_reg(rc, result);
    set_zf(result);
    break;
  }
  case OP_SB:
  case OP_SB_ID: {
    uint32_t addr = addr_add(ra_val, imm16s);
    if (opcode == OP_SB_ID) {
      addr = static_cast<uint32_t>(addr + static_cast<uint32_t>(tasklet_idx));
    }
    const uint8_t value =
        static_cast<uint8_t>(((rb == 24u) ? imm_u32 : rb_val) & 0xFFu);
    wram.store_u8(addr, value);
    break;
  }
  case OP_SD:
  case OP_SD_ID: {
    uint32_t addr = addr_add(ra_val, imm16s);
    if (opcode == OP_SD_ID) {
      addr = static_cast<uint32_t>(
          addr + static_cast<uint32_t>(tasklet_idx *
                                       static_cast<int>(sizeof(uint64_t))));
    }
    const uint64_t value =
        (rb == 24u) ? static_cast<uint64_t>(static_cast<int64_t>(imm16s))
                    : read_dreg(rb);
    wram.store_u64(addr, value, false);
    break;
  }
  case OP_SH:
  case OP_SH_ID: {
    uint32_t addr = addr_add(ra_val, imm16s);
    if (opcode == OP_SH_ID) {
      addr = static_cast<uint32_t>(
          addr + static_cast<uint32_t>(tasklet_idx *
                                       static_cast<int>(sizeof(uint16_t))));
    }
    const uint16_t value =
        static_cast<uint16_t>(((rb == 24u) ? imm_u32 : rb_val) & 0xFFFFu);
    wram.store_u16(addr, value, false);
    break;
  }
  case OP_SW:
  case OP_SW_ID: {
    uint32_t addr = addr_add(ra_val, imm16s);
    if (opcode == OP_SW_ID) {
      addr = static_cast<uint32_t>(
          addr + static_cast<uint32_t>(tasklet_idx *
                                       static_cast<int>(sizeof(uint32_t))));
    }
    const uint32_t value = (rb == 24u) ? imm_u32 : rb_val;
    wram.store_u32(addr, value, false);
    break;
  }
  case OP_LDMA:
  case OP_LDMAI:
  case OP_SDMA: {
    const uint32_t wram_addr = ra_val & 0xFFFFF8u;
    const uint32_t mram_addr = rb_val & 0xFFFFFFF8u;
    const uint32_t chunks =
        (1u +
         ((static_cast<uint32_t>(imm8) + ((ra_val >> 24u) & 0xFFu)) & 0xFFu));
    const size_t dma_bytes = static_cast<size_t>(chunks << 3u);
    if (opcode == OP_SDMA) {
      (void)dma_engine.dma_to_mram_from_wram(mram, wram, dma_bytes, mram_addr,
                                             wram_addr);
    } else {
      (void)dma_engine.dma_to_wram_from_mram(wram, mram, dma_bytes, wram_addr,
                                             mram_addr);
    }
    break;
  }
  default:
    // Unknown instructions are treated as NOPs to keep forward progress.
    break;
  }

  register_file_.pc_regs_[t] = next_pc;

  const bool is_replay_step = (replay_bits_ & tasklet_bit) != 0ULL;
  replay_bits_ &= ~tasklet_bit;
  perf_counter_retire_step(is_replay_step);

  if (!keep_running) {
    return false;
  }

  const size_t next_slot64 =
      static_cast<size_t>(register_file_.pc_regs_[t] / 8u);
  if (next_slot64 > static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
    return false;
  }

  const uint16_t next_slot = static_cast<uint16_t>(next_slot64);
  return iram.slot_in_bounds(next_slot);
}
