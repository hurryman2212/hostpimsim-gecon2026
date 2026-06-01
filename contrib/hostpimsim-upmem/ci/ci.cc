#include "ci.hh"

// CI protocol state is shared by multiple control paths (monitor thread,
// callback/handler paths, and host API calls). We keep all protocol fields
// under lock_ so that each read/write observes a coherent state snapshot.
//
// Note: `updated_` is intentionally atomic and exposed via load/store helpers.
// It is used as a low-overhead publication channel for CI words that may be
// sampled frequently without taking the full protocol-state mutex.
/*
 * Reset the emulated CI protocol state to power-on defaults.
 *
 * This function clears all command-decoder latches and restores selection/
 * grouping control fields so subsequent CI frames are interpreted from a
 * known baseline after allocation, reset, or software-reset commands.
 */
void CIState::reset_protocol_state() {
  std::lock_guard<std::mutex> guard(lock_);
  // Power-on defaults expected by the CI protocol emulator.
  pc_mode_ = 0x04u;
  stack_up_mask_ = 0x00u;
  selected_mask_ = 0xFFu;

  // Group slot 0 is selected by default; the rest start disabled.
  group_mask_.fill(0u);
  group_mask_[0] = 0xFFu;

  // Clear sticky protocol-side latches/stateful decoder context.
  dma_ctrl_read_register_ = 0x00u;
  structure_ = 0;
  iram_write_structure_valid_ = false;
  iram_write_addr_hi_ = 0;
  wram_write_structure_valid_ = false;
  wram_write_addr_ = 0;

  dispatched_payload_ = 0u;
  dispatched_needs_mask_fuzz_ = false;

  dispatch_pending_.expected_set = false;
  dispatch_pending_.dispatch_generation_before = 0;
  dispatch_pending_.active = false;

  control_pending_.expected_set = false;
  control_pending_.dispatch_generation_before = 0;
  control_pending_.selected_mask = 0u;
  control_pending_.execution_mask = 0u;
  control_pending_.payload = 0u;
  control_pending_.control_generations.fill(0u);
  control_pending_.active = false;

  run_state_pending_.expected_set = false;
  run_state_pending_.dispatch_generation_before = 0;
  run_state_pending_.selected_mask = 0u;
  run_state_pending_.execution_mask = 0u;
  run_state_pending_.payload = 0u;
  run_state_pending_.control_generations.fill(0u);
  run_state_pending_.active = false;
}

/*
 * Return the most recently committed raw CI command word.
 *
 * Access is mutex-protected because commit/update paths and MMIO callback
 * paths may read/write this field concurrently.
 */
uint64_t CIState::get_committed() const {
  std::lock_guard<std::mutex> guard(lock_);
  return committed_;
}

/*
 * Publish the latest committed raw CI command word.
 *
 * The committed word is the authoritative command consumed by the dispatch
 * path and must stay coherent with the rest of protocol state.
 */
void CIState::set_committed(uint64_t word) {
  std::lock_guard<std::mutex> guard(lock_);
  committed_ = word;
}

/*
 * Atomically load the last updated CI response word.
 *
 * This is intentionally lock-free because polling paths may read this value
 * at high frequency; the caller chooses the memory-order contract.
 */
uint64_t CIState::load_updated(std::memory_order order) const {
  // Fast-path read used by polling paths where taking lock_ for every sample
  // would add unnecessary contention.
  return updated_.load(order);
}

/*
 * Atomically store the latest CI response word.
 *
 * Writers publish through this channel so update/polling consumers can read
 * without taking the full state mutex.
 */
void CIState::store_updated(uint64_t word, std::memory_order order) {
  // Paired with load_updated(order) to publish the latest CI word with caller-
  // chosen memory-order semantics.
  updated_.store(word, order);
}

/*
 * Get the current per-chip selected-DPU bitmask.
 *
 * The selected mask drives which local DPUs are targeted by many CI frames.
 */
uint8_t CIState::get_selected_mask() const {
  std::lock_guard<std::mutex> guard(lock_);
  return selected_mask_;
}

/*
 * Replace the current selected-DPU bitmask.
 *
 * Callers use this to emulate CI select-all/select-one/select-group commands.
 */
void CIState::set_selected_mask(uint8_t mask) {
  std::lock_guard<std::mutex> guard(lock_);
  selected_mask_ = mask;
}

uint8_t CIState::get_enabled_mask() const {
  std::lock_guard<std::mutex> guard(lock_);
  return enabled_mask_;
}

void CIState::set_enabled_mask(uint8_t mask) {
  std::lock_guard<std::mutex> guard(lock_);
  enabled_mask_ = mask;
}

/*
 * Read a saved group-selection mask.
 *
 * Out-of-range slots are treated as empty groups and return 0 for robust
 * behavior against malformed inputs.
 */
uint8_t CIState::get_group_mask(size_t group_slot_nr) const {
  std::lock_guard<std::mutex> guard(lock_);
  if (group_slot_nr >= group_mask_.size()) {
    return 0u;
  }
  return group_mask_[group_slot_nr];
}

/*
 * Store a group-selection mask into one group slot.
 *
 * Invalid slot indices are ignored to keep protocol handling defensive.
 */
void CIState::set_group_mask(size_t group_slot_nr, uint8_t mask) {
  std::lock_guard<std::mutex> guard(lock_);
  if (group_slot_nr >= group_mask_.size()) {
    return;
  }
  group_mask_[group_slot_nr] = mask;
}

/*
 * Read the emulated PC mode register value.
 *
 * This mirrors CI control commands that query execution mode state.
 */
uint8_t CIState::get_pc_mode() const {
  std::lock_guard<std::mutex> guard(lock_);
  return pc_mode_;
}

/*
 * Update the emulated PC mode register value.
 *
 * This is written by CI programming commands (for example tag 0xA4 frames).
 */
void CIState::set_pc_mode(uint8_t mode) {
  std::lock_guard<std::mutex> guard(lock_);
  pc_mode_ = mode;
}

/*
 * Atomically swap the stack-up mask and return the previous value.
 *
 * CI protocol commands use read-modify-write semantics for this field, so a
 * single locked exchange keeps behavior deterministic.
 */
uint8_t CIState::exchange_stack_up_mask(uint8_t value) {
  std::lock_guard<std::mutex> guard(lock_);
  const uint8_t old = stack_up_mask_;
  stack_up_mask_ = value;
  return old;
}

/*
 * Get which DMA control register should be exposed by subsequent read frames.
 *
 * This models protocol behavior where one command selects a register and
 * another command fetches it.
 */
uint8_t CIState::get_dma_ctrl_read_register() const {
  std::lock_guard<std::mutex> guard(lock_);
  return dma_ctrl_read_register_;
}

/*
 * Set which DMA control register should be returned on read commands.
 *
 * This is updated by DMA-control programming frames (tag 0xA7).
 */
void CIState::set_dma_ctrl_read_register(uint8_t value) {
  std::lock_guard<std::mutex> guard(lock_);
  dma_ctrl_read_register_ = value;
}

/*
 * Return the most recent structure/setup command word.
 *
 * Structure commands provide context used to decode subsequent frame words.
 */
uint64_t CIState::get_structure() const {
  std::lock_guard<std::mutex> guard(lock_);
  return structure_;
}

/*
 * Cache the most recent structure/setup command word.
 *
 * Later frame decoding may depend on this latched structure context.
 */
void CIState::set_structure(uint64_t value) {
  std::lock_guard<std::mutex> guard(lock_);
  structure_ = value;
}

/*
 * Report whether the currently latched structure describes an IRAM write.
 *
 * Dispatch code checks this flag before interpreting opcode-0x33 frames as
 * IRAM payload writes.
 */
bool CIState::is_iram_write_structure_valid() const {
  std::lock_guard<std::mutex> guard(lock_);
  return iram_write_structure_valid_;
}

/*
 * Mark whether the latched structure corresponds to a valid IRAM-write form.
 */
void CIState::set_iram_write_structure_valid(bool valid) {
  std::lock_guard<std::mutex> guard(lock_);
  iram_write_structure_valid_ = valid;
}

/*
 * Get cached high IRAM address bits from the last valid structure command.
 *
 * The low address byte is carried by the frame tag and combined at dispatch.
 */
uint16_t CIState::get_iram_write_addr_hi() const {
  std::lock_guard<std::mutex> guard(lock_);
  return iram_write_addr_hi_;
}

/*
 * Set cached high IRAM address bits extracted from structure commands.
 */
void CIState::set_iram_write_addr_hi(uint16_t addr_hi) {
  std::lock_guard<std::mutex> guard(lock_);
  iram_write_addr_hi_ = addr_hi;
}

/*
 * Report whether the currently latched structure describes a WRAM write.
 */
bool CIState::is_wram_write_structure_valid() const {
  std::lock_guard<std::mutex> guard(lock_);
  return wram_write_structure_valid_;
}

/*
 * Mark whether the latched structure is a valid WRAM-write descriptor.
 */
void CIState::set_wram_write_structure_valid(bool valid) {
  std::lock_guard<std::mutex> guard(lock_);
  wram_write_structure_valid_ = valid;
}

/*
 * Get the cached WRAM word address derived from structure decoding.
 *
 * Some frame forms may later override low bits, but this remains the base.
 */
uint16_t CIState::get_wram_write_addr() const {
  std::lock_guard<std::mutex> guard(lock_);
  return wram_write_addr_;
}

/*
 * Set the cached WRAM word address extracted from structure commands.
 */
void CIState::set_wram_write_addr(uint16_t addr) {
  std::lock_guard<std::mutex> guard(lock_);
  wram_write_addr_ = addr;
}

/*
 * Cache the decoded dispatch result for the latest committed command.
 *
 * `payload` is the 32-bit CI payload field and `needs_mask_fuzz` selects the
 * response-envelope style used when commit/update words are assembled.
 */
void CIState::set_dispatched_result(uint32_t payload, bool needs_mask_fuzz) {
  {
    std::lock_guard<std::mutex> guard(lock_);
    dispatched_payload_ = payload;
    dispatched_needs_mask_fuzz_ = needs_mask_fuzz;
  }
  (void)dispatched_generation_.fetch_add(1u, std::memory_order_release);
}

/*
 * Return the cached 32-bit dispatch payload for the latest command.
 */
uint32_t CIState::get_dispatched_payload() const {
  std::lock_guard<std::mutex> guard(lock_);
  return dispatched_payload_;
}

/*
 * Return whether the latest command requires nop/mask-style response framing.
 */
bool CIState::get_dispatched_needs_mask_fuzz() const {
  std::lock_guard<std::mutex> guard(lock_);
  return dispatched_needs_mask_fuzz_;
}

uint64_t CIState::load_dispatched_generation(std::memory_order order) const {
  return dispatched_generation_.load(order);
}

void CIState::set_dispatch_pending(bool expected_set,
                                   uint64_t dispatch_generation_before) {
  std::lock_guard<std::mutex> guard(lock_);
  dispatch_pending_.expected_set = expected_set;
  dispatch_pending_.dispatch_generation_before = dispatch_generation_before;
  dispatch_pending_.active = true;
}

bool CIState::get_dispatch_pending(bool &expected_set,
                                   uint64_t &dispatch_generation_before) const {
  std::lock_guard<std::mutex> guard(lock_);
  expected_set = dispatch_pending_.expected_set;
  dispatch_generation_before = dispatch_pending_.dispatch_generation_before;
  return dispatch_pending_.active;
}

bool CIState::has_dispatch_pending() const {
  std::lock_guard<std::mutex> guard(lock_);
  return dispatch_pending_.active;
}

void CIState::clear_dispatch_pending() {
  std::lock_guard<std::mutex> guard(lock_);
  dispatch_pending_.expected_set = false;
  dispatch_pending_.dispatch_generation_before = 0;
  dispatch_pending_.active = false;
}

void CIState::set_control_pending(
    bool expected_set, uint64_t dispatch_generation_before,
    uint8_t selected_mask, uint8_t execution_mask,
    const std::array<uint64_t, kChipNumDpus> &control_generations,
    uint8_t payload) {
  std::lock_guard<std::mutex> guard(lock_);
  control_pending_.expected_set = expected_set;
  control_pending_.dispatch_generation_before = dispatch_generation_before;
  control_pending_.selected_mask = selected_mask;
  control_pending_.execution_mask = execution_mask;
  control_pending_.payload = payload;
  control_pending_.control_generations = control_generations;
  control_pending_.active = true;
}

bool CIState::get_control_pending(bool &expected_set,
                                  uint64_t &dispatch_generation_before,
                                  uint8_t &selected_mask,
                                  uint8_t &execution_mask) const {
  std::lock_guard<std::mutex> guard(lock_);
  expected_set = control_pending_.expected_set;
  dispatch_generation_before = control_pending_.dispatch_generation_before;
  selected_mask = control_pending_.selected_mask;
  execution_mask = control_pending_.execution_mask;
  return control_pending_.active;
}

uint8_t CIState::get_control_pending_payload() const {
  std::lock_guard<std::mutex> guard(lock_);
  return control_pending_.payload;
}

uint64_t CIState::get_control_pending_generation(uint8_t dpu_idx) const {
  std::lock_guard<std::mutex> guard(lock_);
  if (dpu_idx >= kChipNumDpus) {
    return 0u;
  }
  return control_pending_.control_generations[dpu_idx];
}

bool CIState::has_control_pending() const {
  std::lock_guard<std::mutex> guard(lock_);
  return control_pending_.active;
}

void CIState::clear_control_pending() {
  std::lock_guard<std::mutex> guard(lock_);
  control_pending_.expected_set = false;
  control_pending_.dispatch_generation_before = 0;
  control_pending_.selected_mask = 0u;
  control_pending_.execution_mask = 0u;
  control_pending_.payload = 0u;
  control_pending_.control_generations.fill(0u);
  control_pending_.active = false;
}

void CIState::set_run_state_pending(
    bool expected_set, uint64_t dispatch_generation_before,
    uint8_t selected_mask, uint8_t execution_mask,
    const std::array<uint64_t, kChipNumDpus> &control_generations,
    uint8_t payload) {
  std::lock_guard<std::mutex> guard(lock_);
  run_state_pending_.expected_set = expected_set;
  run_state_pending_.dispatch_generation_before = dispatch_generation_before;
  run_state_pending_.selected_mask = selected_mask;
  run_state_pending_.execution_mask = execution_mask;
  run_state_pending_.payload = payload;
  run_state_pending_.control_generations = control_generations;
  run_state_pending_.active = true;
}

bool CIState::get_run_state_pending(bool &expected_set,
                                    uint64_t &dispatch_generation_before,
                                    uint8_t &selected_mask,
                                    uint8_t &execution_mask) const {
  std::lock_guard<std::mutex> guard(lock_);
  expected_set = run_state_pending_.expected_set;
  dispatch_generation_before = run_state_pending_.dispatch_generation_before;
  selected_mask = run_state_pending_.selected_mask;
  execution_mask = run_state_pending_.execution_mask;
  return run_state_pending_.active;
}

uint8_t CIState::get_run_state_pending_payload() const {
  std::lock_guard<std::mutex> guard(lock_);
  return run_state_pending_.payload;
}

uint64_t
CIState::get_run_state_pending_control_generation(uint8_t dpu_idx) const {
  std::lock_guard<std::mutex> guard(lock_);
  if (dpu_idx >= kChipNumDpus) {
    return 0u;
  }
  return run_state_pending_.control_generations[dpu_idx];
}

bool CIState::has_run_state_pending() const {
  std::lock_guard<std::mutex> guard(lock_);
  return run_state_pending_.active;
}

void CIState::clear_run_state_pending() {
  std::lock_guard<std::mutex> guard(lock_);
  run_state_pending_.expected_set = false;
  run_state_pending_.dispatch_generation_before = 0;
  run_state_pending_.selected_mask = 0u;
  run_state_pending_.execution_mask = 0u;
  run_state_pending_.payload = 0u;
  run_state_pending_.control_generations.fill(0u);
  run_state_pending_.active = false;
}

/*
 * Expose mutable CIState so protocol handlers can update state in-place.
 */
CIState &CI::state() { return state_; }

/*
 * Expose read-only CIState for const call paths.
 */
const CIState &CI::state() const { return state_; }

/*
 * Return the MMIO write region bound to this CI instance.
 */
pim_region_t *CI::get_write_region() { return region_write_; }

/*
 * Const overload of write-region accessor.
 */
const pim_region_t *CI::get_write_region() const { return region_write_; }

// Region bindings are intentionally lightweight pointer assignments.
// Lifetime/ownership is managed by the owning DPU/device topology.
/*
 * Bind (or rebind) the MMIO write region used by this CI.
 *
 * Ownership is external; this class only stores the pointer.
 */
void CI::set_write_region(pim_region_t *region) { region_write_ = region; }

/*
 * Return the MMIO read/result region bound to this CI instance.
 */
pim_region_t *CI::get_read_region() { return region_read_; }

/*
 * Const overload of read-region accessor.
 */
const pim_region_t *CI::get_read_region() const { return region_read_; }

/*
 * Bind (or rebind) the MMIO read/result region used by this CI.
 *
 * Ownership is external; this class only stores the pointer.
 */
void CI::set_read_region(pim_region_t *region) { region_read_ = region; }
