#include "internal.hh"

#include <array>

/*
 * Build a CI thread-control frame for one tasklet.
 *
 * - `base` selects operation family (boot/resume/clear/read).
 * - `thread_id` is split across two nibbles per hardware encoding.
 * - tag 0x98 marks thread control commands.
 */
static uint64_t thread_frame_command(uint64_t base, uint8_t thread_id) {
  uint64_t x = base;
  x |= (static_cast<uint64_t>(24u & 31u) << 34);
  x |= (static_cast<uint64_t>(thread_id & 0xFu) << 20);
  x |= (static_cast<uint64_t>((thread_id >> 4) & 0xFu) << 16);
  return encode_frame_from_x(x, 0x98u);
}

/*
 * Cache all valid thread command encodings once.
 *
 * cache[k][tid] where:
 *   k=0: boot, k=1: resume, k=2: clear-run, k=3: read-run.
 * This avoids recomputing frame encodings for every decode attempt.
 */
static const std::array<std::array<uint64_t, kNumTasklets>, 4> &
thread_cmd_cache() {
  static const std::array<std::array<uint64_t, kNumTasklets>, 4> cache = [] {
    std::array<std::array<uint64_t, kNumTasklets>, 4> local{};

    constexpr uint64_t kBootBase = 0x7d8320000000ULL;
    constexpr uint64_t kResumeBase = 0x7d0320000000ULL;
    constexpr uint64_t kClearBase = 0x7c8320000000ULL;
    constexpr uint64_t kReadBase = 0x7c0330000000ULL;
    for (uint8_t tid = 0; tid < kNumTasklets; ++tid) {
      local[0][tid] = thread_frame_command(kBootBase, tid);
      local[1][tid] = thread_frame_command(kResumeBase, tid);
      local[2][tid] = thread_frame_command(kClearBase, tid);
      local[3][tid] = thread_frame_command(kReadBase, tid);
    }

    return local;
  }();

  return cache;
}

/*
 * Decode a raw CI command into (kind, tasklet-id).
 * Returns true only if the command matches one of the canonical cached frames.
 */
bool decode_thread_command(uint64_t cmd, ThreadCmdKind &kind,
                           uint8_t &thread_id) {
  const auto &cache = thread_cmd_cache();
  for (size_t k = 0; k < cache.size(); ++k) {
    for (size_t t = 0; t < kNumTasklets; ++t) {
      if (cache[k][t] == cmd) {
        kind = static_cast<ThreadCmdKind>(k);
        thread_id = static_cast<uint8_t>(t);
        return true;
      }
    }
  }

  return false;
}
