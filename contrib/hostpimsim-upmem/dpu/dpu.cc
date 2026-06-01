#include "dpu.hh"

static inline bool is_run_state_read_command(uint64_t ci_cmd) {
  const uint8_t opcode = static_cast<uint8_t>((ci_cmd >> 56) & 0xFFu);
  const uint8_t tag = static_cast<uint8_t>((ci_cmd >> 48) & 0xFFu);
  const uint8_t b0 = static_cast<uint8_t>((ci_cmd >> 0) & 0xFFu);
  const uint8_t b1 = static_cast<uint8_t>((ci_cmd >> 8) & 0xFFu);
  return opcode == 0x33u && tag == 0x00u && b0 == 0x84u && b1 == 0x02u;
}

/*
 * Control function to execute a single CI command.
 *
 * In case of an execution command, each tasklet is fine-grained
 * multithreaded, which is time-sliced in a round-robin manner.
 *
 * Updates this DPU's cached 32-bit control payload internally
 * (last_control_result_ / generation).
 */
void UPMEMDPU::control(uint64_t ci_cmd) {
  if (is_run_state_read_command(ci_cmd)) {
    if (!mram_.get_addr() || mram_.get_size() == 0u) {
      pipeline_.set_run_bits(0u);
      pipeline_.set_sleep_bits(0u);
      pipeline_.set_replay_bits(0u);
      set_last_control_result(0u);
      return;
    }

    const size_t tasklet_count =
        std::min(pipeline_.get_nr_tasklet(), static_cast<size_t>(kNumTasklets));

    if (tasklet_count > 0u) {
      static constexpr size_t kProgressBudget = 8388608u;
      for (size_t budget = 0; budget < kProgressBudget; ++budget) {
        if (pipeline_.get_run_bits() == 0u) {
          break;
        }

        for (size_t tid = 0; tid < tasklet_count; ++tid) {
          const uint64_t tasklet_bit = 1ULL << tid;
          const auto run_bits_now = pipeline_.get_run_bits();
          const auto sleep_bits_now = pipeline_.get_sleep_bits();
          if ((run_bits_now & tasklet_bit) == 0ULL ||
              (sleep_bits_now & tasklet_bit) != 0ULL) {
            continue;
          }

          const bool keep_running = pipeline_.execute_once(
              iram_, wram_, dma_engine_, mram_, static_cast<int>(tid));
          if (!keep_running) {
            auto run_bits = pipeline_.get_run_bits();
            auto sleep_bits = pipeline_.get_sleep_bits();
            auto replay_bits = pipeline_.get_replay_bits();
            run_bits &= ~tasklet_bit;
            sleep_bits &= ~tasklet_bit;
            replay_bits &= ~tasklet_bit;
            pipeline_.set_run_bits(run_bits);
            pipeline_.set_sleep_bits(sleep_bits);
            pipeline_.set_replay_bits(replay_bits);
          }
        }
      }
    }

    const auto run_bits = pipeline_.get_run_bits();
    set_last_control_result((run_bits != 0u) ? 1u : 0u);
    return;
  }

  ThreadCmdKind kind{};
  uint8_t thread_id = 0;
  if (!decode_thread_command(ci_cmd, kind, thread_id)) {
    set_last_control_result(0u);
    return;
  }

  auto run_bits = pipeline_.get_run_bits();
  auto sleep_bits = pipeline_.get_sleep_bits();
  auto replay_bits = pipeline_.get_replay_bits();

  auto &register_file = pipeline_.register_file();
  auto &atomic_bits = pipeline_.atomic_bits();

  const auto nr_tasklet = pipeline_.get_nr_tasklet();

  const uint64_t bit =
      (thread_id < 64) ? (1ULL << static_cast<uint64_t>(thread_id)) : 0;
  const bool was_running = (run_bits & bit) != 0;

  switch (kind) {
  case ThreadCmdKind::Boot:
    run_bits |= bit;
    sleep_bits &= ~bit;
    replay_bits &= ~bit;
    if (thread_id < nr_tasklet) {
      register_file.gp_regs_[thread_id].fill(0);
      register_file.pc_regs_[thread_id] = 0;
      register_file.zf_regs_[thread_id] = 0;
      register_file.cf_regs_[thread_id] = 0;
      if (thread_id == 0) {
        atomic_bits.fill(0);
      }
    }
    break;
  case ThreadCmdKind::Resume:
    run_bits |= bit;
    sleep_bits &= ~bit;
    break;
  case ThreadCmdKind::ClearRun:
    run_bits &= ~bit;
    sleep_bits &= ~bit;
    replay_bits &= ~bit;
    break;
  case ThreadCmdKind::ReadRun:
    break;
  }

  pipeline_.set_run_bits(run_bits);
  pipeline_.set_sleep_bits(sleep_bits);
  pipeline_.set_replay_bits(replay_bits);

  set_last_control_result(was_running ? 1u : 0u);
  return;
}
