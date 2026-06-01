# hostpimsim-upmem

A **userspace UPMEM rank emulator** built on top of `libhostpimsim` and `liboverlaysys`. It installs virtual `/dev/dpu_rank0`, `/dev/dax0.0`, and the matching sysfs nodes inside the target process, translates UPMEM SDK ioctls into CI/MMIO activity, and executes a functional DPU core for one emulated rank (8 banks x 8 DPUs).

> **This is not a standalone benchmark runner.** `hostpimsim-upmem` is a preloadable shared library (`libhostpimsim-upmem-preload.so`) that is loaded into an application which already expects the UPMEM driver stack.

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ Application process                                                           │
│                                                                               │
│  open("/dev/dpu_rank0") / ioctl() / read() / stat() / access()                │
└───────────────────────────────┬───────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ liboverlaysys syscall hook                                                    │
│                                                                               │
│  overlaysys_syscall_hook = _pim_vsyscall_dispatch                             │
└───────────────────────────────┬───────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ hostpimsim-upmem preload library                                              │
│                                                                               │
│  - creates virtual /dev/dpu_rank0 and /dev/dax0.0                             │
│  - publishes sysfs metadata for dpu_region_mem.0                              │
│  - allocates one anonymous 8 GB rank aperture on first open                   │
│  - routes rank ioctls to UPMEMPIMRank                                         │
└───────────────────────────────┬───────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ UPMEMPIMRank                                                                  │
│                                                                               │
│  8 bank slots                                                                 │
│  ┌────────────┐ ┌────────────┐        ┌────────────┐                          │
│  │  chip 0    │ │  chip 1    │  ...   │  chip 7    │                          │
│  │ CI + 8 DPU │ │ CI + 8 DPU │        │ CI + 8 DPU │                          │
│  └─────┬──────┘ └─────┬──────┘        └─────┬──────┘                          │
└────────┼──────────────┼─────────────────────┼─────────────────────────────────┘
         │              │                     │
         ▼              ▼                     ▼
   ┌───────────┐  ┌───────────┐         ┌───────────┐
   │ UPMEMDPU  │  │ UPMEMDPU  │   ...   │ UPMEMDPU  │
   └─────┬─────┘  └─────┬─────┘         └─────┬─────┘
         │              │                     │
         ▼              ▼                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Per-DPU functional core                                                       │
│                                                                               │
│  IRAM (24 KB)   WRAM (64 KB)   MRAM (64 MB logical aperture)  DMAEngine       │
│  Pipeline       24 tasklets    GP regs / PC / CF / ZF         perf counter    │
│                                                                               │
│  CI thread-control frames and run-state reads drive execution.                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### DPU Resources (per DPU)

| Resource     | Size                 | Description                                                |
| ------------ | -------------------- | ---------------------------------------------------------- |
| Atomic slots | 256 entries          | Software lock slots used by `ACQUIRE` / `RELEASE`          |
| WRAM         | 64 KB                | Shared scratchpad for all tasklets in one DPU              |
| IRAM         | 24 KB                | 3072 instruction slots, 48-bit payload stored in 8 bytes   |
| MRAM         | 64 MB                | Logical main memory with host-aperture address translation |
| Registers    | 24 x 32-bit          | Mutable GP registers per tasklet                           |
| Tasklets     | 24                   | Independent execution contexts per DPU                     |
| Perf counter | 32-bit visible value | DPU-level counter with mode configuration                  |

### Register File

Each tasklet exposes 24 x 32-bit GP registers. Even/odd pairs are also used as one 64-bit D-register by `LD`, `SD`, `MOVD`, `SWAPD`, and the `S_` / `U_` writeback families.

```
64-bit view: d0,  d2,  d4,  d6,  d8,  d10, d12, d14, d16, d18, d20, d22
32-bit view: r0   r2   r4   r6   r8   r10  r12  r14  r16  r18  r20  r22
             r1   r3   r5   r7   r9   r11  r13  r15  r17  r19  r21  r23
             `-------------- each dn spans {rN, rN+1} ------------------`
```

### Special Registers (Read-Only, indices 24-31)

| Index | Name   | Value         | Description                 |
| ----- | ------ | ------------- | --------------------------- |
| 24    | `zero` | 0             | Constant zero               |
| 25    | `one`  | 1             | Constant one                |
| 26    | `lneg` | 0xFFFFFFFF    | All bits set                |
| 27    | `mneg` | 0x80000000    | Minimum signed 32-bit value |
| 28    | `id`   | tasklet id    | Current tasklet index       |
| 29    | `id2`  | tasklet id x2 | Current tasklet index x 2   |
| 30    | `id4`  | tasklet id x4 | Current tasklet index x 4   |
| 31    | `id8`  | tasklet id x8 | Current tasklet index x 8   |

### Flags and Program Counter

| Register     | Width  | Description                                                      |
| ------------ | ------ | ---------------------------------------------------------------- |
| CF           | 1-bit  | Carry flag per tasklet                                           |
| ZF           | 1-bit  | Zero flag per tasklet                                            |
| PC           | 64-bit | IRAM program counter per tasklet                                 |
| perf_counter | 32-bit | DPU-level visible counter value (`raw >> 4`)                     |
| run_bits     | 64-bit | Shared runnable-bit mask for tasklets 0-23                       |
| sleep_bits   | 64-bit | Shared sleep-bit mask for tasklets 0-23                          |
| replay_bits  | 64-bit | Shared replay tracking for instruction retirement / perf updates |

### Tasklet Execution Model

- Each DPU tracks 24 tasklets with independent GP registers, flags, and PC.
- Execution is functional, not cycle-accurate. `Pipeline::execute_once()` runs one tasklet step at a time.
- A run-state read command (`0x33 / tag 0x00 / b0=0x84 / b1=0x02`) acts as the execution trigger. It advances runnable tasklets in round-robin order until no runnable tasklet remains or the progress budget is exhausted.
- CI thread-control frames (`tag 0x98`) implement `Boot`, `Resume`, `ClearRun`, and `ReadRun`.
- In-program `BOOT`, `RESUME`, and `STOP` opcodes update the same run/sleep state tracked by the pipeline.
- Booting tasklet 0 also clears the DPU's atomic-slot array.

## Memory Map (hostpimsim-upmem Rank Aperture)

Each opened rank allocates one anonymous 8 GB aperture. Most of that space is reserved for lazily materialized MRAM bank-line windows; fixed CI and color state regions are carved out at known offsets.

```
Rank aperture (size = 0x200000000 bytes = 8 GB)

  0x00000000  ┌─────────────────────────────────────────────────────────────┐
              │ MRAM aperture windows (lazy-created RAM regions)            │
              │ - logical size: 64 MB per DPU                               │
              │ - topology: 8 banks x 8 local DPUs                          │
              │ - layout: translated / strided host aperture, not linear    │
              ├─────────────────────────────────────────────────────────────┤
  0x00020000  │ CI MMIO lanes (8 x 8-byte command registers)                │
              ├─────────────────────────────────────────────────────────────┤
  0x00028000  │ CI RW lanes (8 x 8-byte response registers)                 │
              ├─────────────────────────────────────────────────────────────┤
      ...     │ Remaining MRAM aperture windows                             │
              ├─────────────────────────────────────────────────────────────┤
  0x1FFFFFFF8 │ CI color state bytes (8 x 1 byte, one per bank)             │
  0x200000000 └─────────────────────────────────────────────────────────────┘
```

### Shared Memory Layout

`hostpimsim-upmem` does **not** expose a shared `/dev/shm/pim` layout. The runtime instead creates one private anonymous mapping per opened rank and uses libhostpimsim regions on top of it.

```
Process-local rank backing

┌───────────────────────────────────────────────────────────────────────────┐
│ mmap(PROT_NONE, 8 GB)                                                     │
├───────────────────────────────────────────────────────────────────────────┤
│ Lazy RAM regions                                                          │
│ - MRAM bank-line windows are created on demand                            │
│ - CI MMIO / RW lanes are created on first bank access                     │
│ - CI color-state bytes are created on first use                           │
├───────────────────────────────────────────────────────────────────────────┤
│ Virtual device front-end                                                  │
│ - /dev/dpu_rank0 uses ioctl dispatch                                      │
│ - /dev/dax0.0 exists for discovery only                                   │
│ - sysfs metadata is published through the vdev / vsysfs layer             │
└───────────────────────────────────────────────────────────────────────────┘

Address calculation notes:
  - local-DPU bank-line stride: 0x40000 bytes
  - CI MMIO base: 0x20000
  - CI RW base:   0x28000
  - color-state base: aperture_end - 8
```

## Control Registers

### CTRL_MMIO Page (0x20000 - 0x2003F)

Only eight 64-bit bank lanes are used. Writing one CI word into a lane routes the command to the corresponding bank's `UPMEMPIMChip`.

| Offset                   | Name     | Width  | Access | Description                   |
| ------------------------ | -------- | ------ | ------ | ----------------------------- |
| `0x20000 + bank_idx * 8` | `CI_CMD` | 64-bit | W      | Raw committed CI command word |

### CTRL_RW Page (0x28000 - 0x2803F)

Each bank has one 64-bit response lane that publishes the last updated CI word.

| Offset                   | Name     | Width  | Access | Description                           |
| ------------------------ | -------- | ------ | ------ | ------------------------------------- |
| `0x28000 + bank_idx * 8` | `CI_RSP` | 64-bit | R/W    | Last updated CI response / poll value |

### CMD Values

Instead of a custom command enum, this implementation consumes raw UPMEM CI frames. The following families are handled explicitly in `UPMEMPIMChip`:

| Family                   | Kind          | Description                                           |
| ------------------------ | ------------- | ----------------------------------------------------- |
| Software reset           | CI word       | Resets protocol state for one bank                    |
| Byte-order probe         | CI word       | Returns the expected byte-order response              |
| Select all / one / group | `tag 0x00`    | Updates per-bank selected mask and group masks        |
| IRAM / WRAM structure    | `opcode 0x11` | Latches address context for subsequent payload frames |
| IRAM / WRAM payload      | `opcode 0x33` | Writes instruction words or WRAM words                |
| Run-state read           | `tag 0x00`    | Advances runnable tasklets and returns run status     |
| Thread control           | `tag 0x98`    | Boot / Resume / ClearRun / ReadRun                    |
| WRAM read                | `tag 0x99`    | Reads one WRAM word from the first selected DPU       |
| DMA control read / write | `tag 0xA7`    | Selects or updates DMA control state                  |

### Status Transitions

`hostpimsim-upmem` models the UPMEM rank protocol as a commit / update cycle instead of a standalone device-specific status register.

```
open("/dev/dpu_rank0")
        │
        ▼
  upmem_create_rank()
        │
        ▼
ioctl(DPU_RANK_IOCTL_COMMIT_COMMANDS)
        │
        ├── decode CI word
        ├── maybe dispatch per-bank / per-DPU handlers
        └── publish immediate response or mark pending
        │
        ▼
ioctl(DPU_RANK_IOCTL_UPDATE_COMMANDS)
        │
        ├── poll pending control generations
        ├── finalize run-state / thread-control responses
        └── publish ready CI response word
        │
        ▼
read updated CI word from bank RW lane
```

### STATUS Values

There is no dedicated `STATUS` register. Observable protocol state is encoded in `CIState` and surfaced through the updated CI response word.

| State slot           | Meaning                                                   |
| -------------------- | --------------------------------------------------------- |
| `committed_`         | Last CI word written into one bank's MMIO lane            |
| `updated_`           | Last response word published for polling / reads          |
| `dispatch_pending_`  | Command was committed but final response is not ready     |
| `control_pending_`   | Waiting for per-DPU control generations to advance        |
| `run_state_pending_` | Waiting for run-state completion or deferred control path |

### ERROR Values

Errors are reported through normal Unix return values and `errno`.

| Error    | Typical source                                             |
| -------- | ---------------------------------------------------------- |
| `EINVAL` | Null pointers, malformed ioctl arguments, unsupported mmap |
| `ENOMEM` | Rank allocation or lazy population of aperture regions     |
| `EFAULT` | MRAM copy failure, invalid transfer buffers, OOB access    |
| `ENODEV` | Rank device not opened or missing backing device           |
| `ENOTTY` | Unsupported ioctl on rank or DAX virtual device            |
| `EBUSY`  | Virtual device registration conflict during open           |

## Instruction Set Architecture

`hostpimsim-upmem` executes the DPU ISA implemented in `dpu/pipeline.cc`. The model is functional and uses the same opcode space (`0x00` through `0x52`) as current UPMEM DPU programs, but it does not attempt cycle-accurate timing.

### Instruction Format (48-bit, Little-Endian)

IRAM stores one 48-bit instruction payload in the low 6 bytes of each 8-byte
slot. The high 2 bytes are kept zero when writing through the CI path.

```
IRAM slot (8 bytes):

  byte 0 .. byte 5 : instruction payload (little-endian, 48 bits)
  byte 6 .. byte 7 : zero-filled padding

inst48 [47:0]:
  [6:0]   opcode
  [7]     variant / suffix bit
  [47:8]  register, condition, immediate, and offset fields
```

### Instruction Suffixes and `S_`/`U_` 64-bit Extensions

The decoder distinguishes writeback style, condition handling, and 64-bit pair register behavior from the instruction format.

| Suffix family                      | Write behavior                                        | Examples                  |
| ---------------------------------- | ----------------------------------------------------- | ------------------------- |
| Plain register writeback           | Write 32-bit result into one GP register              | `RRI`, `RRR`, `RRIF`      |
| `S_` prefixed writeback            | Sign-extend 32-bit result into a D-register pair      | `S_RRI`, `S_RRRC`         |
| `U_` prefixed writeback            | Zero-extend 32-bit result into a D-register pair      | `U_RRI`, `U_RRRC`         |
| Zero-write / compare-only variants | Evaluate flags / condition without keeping the result | `ZRI`, `ZRRCI`            |
| Condition-set variants             | Materialize a boolean 0 / 1 result                    | `RRRC`, `RRIC`, `RIRC`    |
| Branch variants                    | Test a decoded condition code and redirect PC         | `RRRCI`, `ZRRCI`, `ZRICI` |

### Condition Codes

Condition handling is implemented in the pipeline decoder and covers the common UPMEM condition families.

| Family           | Examples                 | Meaning                                             |
| ---------------- | ------------------------ | --------------------------------------------------- |
| Always / never   | `TRUE`, `FALSE`          | Unconditional or suppressed branch / writeback      |
| Zero / non-zero  | `Z`, `NZ`, `EQ`, `NEQ`   | Based on the zero flag or compare result            |
| Carry / no-carry | `C`, `NC`, `LTU`, `GEU`  | Based on carry / borrow state                       |
| Signed relation  | `MI`, `PL`, `LTS`, `GES` | Based on sign and signed compare result             |
| Even / odd       | `E`, `O`, `SE`, `SO`     | Tests low-bit parity of source or result            |
| Extended compare | `XZ`, `XNZ`, `XLEU`      | Cross-instruction compare helpers for multiword ops |
| Small / large    | `SMALL`, `LARGE`         | Size-class checks on the computed result            |

### Opcode Reference (83 opcodes: 0x00-0x52)

The tables below summarize the functional opcode groups executed by the pipeline.

#### Synchronization (Atomic Mutex)

| Opcode | Mnemonic  | Description                      |
| ------ | --------- | -------------------------------- |
| 0x00   | `ACQUIRE` | Acquire one software atomic slot |
| 0x01   | `RELEASE` | Release one software atomic slot |

#### ALU Operations

| Opcode | Mnemonic   | Description             |
| ------ | ---------- | ----------------------- |
| 0x02   | `ADD`      | Integer add             |
| 0x03   | `ADDC`     | Integer add with carry  |
| 0x04   | `AND`      | Bitwise and             |
| 0x05   | `ANDN`     | Bitwise and-not         |
| 0x06   | `ASR`      | Arithmetic right shift  |
| 0x07   | `CAO`      | Count alternating bits  |
| 0x08   | `CLO`      | Count leading ones      |
| 0x09   | `CLS`      | Count leading sign bits |
| 0x0A   | `CLZ`      | Count leading zeros     |
| 0x0B   | `CMPB4`    | Byte-wise compare       |
| 0x0C   | `DIV_STEP` | Division step helper    |
| 0x0D   | `EXTSB`    | Sign-extend byte        |
| 0x0E   | `EXTSH`    | Sign-extend halfword    |
| 0x0F   | `EXTUB`    | Zero-extend byte        |
| 0x10   | `EXTUH`    | Zero-extend halfword    |

#### Shift Operations

| Opcode | Mnemonic  | Description               |
| ------ | --------- | ------------------------- |
| 0x11   | `LSL`     | Logical shift left        |
| 0x12   | `LSL_ADD` | Shift-left plus add       |
| 0x13   | `LSL_SUB` | Shift-left plus subtract  |
| 0x14   | `LSL1`    | Shift left by one         |
| 0x15   | `LSL1X`   | Shift left through carry  |
| 0x16   | `LSLX`    | Extended logical left     |
| 0x17   | `LSR`     | Logical shift right       |
| 0x18   | `LSR_ADD` | Shift-right plus add      |
| 0x19   | `LSR1`    | Shift right by one        |
| 0x1A   | `LSR1X`   | Shift right through carry |
| 0x1B   | `LSRX`    | Extended logical right    |

#### Multiply Operations

| Opcode | Mnemonic    | Description                 |
| ------ | ----------- | --------------------------- |
| 0x1C   | `MUL_SH_SH` | Signed high-half multiply   |
| 0x1D   | `MUL_SH_SL` | Mixed signed multiply       |
| 0x1E   | `MUL_SH_UH` | Mixed signed/unsigned mult  |
| 0x1F   | `MUL_SH_UL` | Mixed signed/unsigned mult  |
| 0x20   | `MUL_SL_SH` | Mixed signed multiply       |
| 0x21   | `MUL_SL_SL` | Signed low-half multiply    |
| 0x22   | `MUL_SL_UH` | Mixed signed/unsigned mult  |
| 0x23   | `MUL_SL_UL` | Mixed signed/unsigned mult  |
| 0x24   | `MUL_STEP`  | Multiply step helper        |
| 0x25   | `MUL_UH_UH` | Unsigned high-half multiply |
| 0x26   | `MUL_UH_UL` | Unsigned mixed multiply     |
| 0x27   | `MUL_UL_UH` | Unsigned mixed multiply     |
| 0x28   | `MUL_UL_UL` | Unsigned low-half multiply  |

#### Logic Operations

| Opcode | Mnemonic  | Description     |
| ------ | --------- | --------------- |
| 0x29   | `NAND`    | Bitwise nand    |
| 0x2A   | `NOR`     | Bitwise nor     |
| 0x2B   | `NXOR`    | Bitwise xnor    |
| 0x2C   | `OR`      | Bitwise or      |
| 0x2D   | `ORN`     | Bitwise or-not  |
| 0x2E   | `ROL`     | Rotate left     |
| 0x2F   | `ROL_ADD` | Rotate-left add |
| 0x30   | `ROR`     | Rotate right    |

#### Subtract Operations

| Opcode | Mnemonic | Description                 |
| ------ | -------- | --------------------------- |
| 0x31   | `RSUB`   | Reverse subtract            |
| 0x32   | `RSUBC`  | Reverse subtract with carry |
| 0x33   | `SUB`    | Subtract                    |
| 0x34   | `SUBC`   | Subtract with carry         |
| 0x35   | `XOR`    | Bitwise xor                 |

#### Control Flow

| Opcode | Mnemonic | Description                 |
| ------ | -------- | --------------------------- |
| 0x36   | `BOOT`   | Start another tasklet       |
| 0x37   | `RESUME` | Wake a sleeping tasklet     |
| 0x38   | `STOP`   | Stop current tasklet        |
| 0x39   | `CALL`   | Call with return PC save    |
| 0x3A   | `FAULT`  | Trigger a fault / stop path |
| 0x3B   | `NOP`    | No operation                |

#### Special Operations

| Opcode | Mnemonic   | Description               |
| ------ | ---------- | ------------------------- |
| 0x3C   | `SATS`     | Signed saturation helper  |
| 0x3D   | `MOVD`     | Move one D-register pair  |
| 0x3E   | `SWAPD`    | Swap two D-register pairs |
| 0x3F   | `HASH`     | Hash helper               |
| 0x40   | `TIME`     | Read perf counter         |
| 0x41   | `TIME_CFG` | Configure perf counter    |

#### Load Operations (WRAM)

| Opcode | Mnemonic | Description            |
| ------ | -------- | ---------------------- |
| 0x42   | `LBS`    | Load signed byte       |
| 0x43   | `LBU`    | Load unsigned byte     |
| 0x44   | `LD`     | Load 64-bit pair       |
| 0x45   | `LHS`    | Load signed halfword   |
| 0x46   | `LHU`    | Load unsigned halfword |
| 0x47   | `LW`     | Load 32-bit word       |

#### Store Operations (WRAM)

| Opcode | Mnemonic | Description               |
| ------ | -------- | ------------------------- |
| 0x48   | `SB`     | Store byte                |
| 0x49   | `SB_ID`  | Store byte with id form   |
| 0x4A   | `SD`     | Store 64-bit pair         |
| 0x4B   | `SD_ID`  | Store 64-bit pair id form |
| 0x4C   | `SH`     | Store halfword            |
| 0x4D   | `SH_ID`  | Store halfword id form    |
| 0x4E   | `SW`     | Store 32-bit word         |
| 0x4F   | `SW_ID`  | Store word id form        |

#### DMA Operations (MRAM <-> WRAM/IRAM)

| Opcode | Mnemonic | Description           |
| ------ | -------- | --------------------- |
| 0x50   | `LDMA`   | MRAM to WRAM DMA copy |
| 0x51   | `LDMAI`  | MRAM to IRAM DMA copy |
| 0x52   | `SDMA`   | WRAM to MRAM DMA copy |

## Usage

### Prerequisites

This component is intended to be loaded into an application that already uses the UPMEM SDK ABI. The expected runtime pieces are:

```
hostpimsim/
  build/
    lib/
      libhostpimsim.so
      libhostpimsim-upmem-preload.so

Target process:
  - linked against liboverlaysys
  - launched with LD_PRELOAD=.../libhostpimsim-upmem-preload.so
  - opens /dev/dpu_rank0 and uses rank ioctls
```

### Command Line

```bash
# On hostpimsim root directory

# Run an application with the virtual UPMEM rank enabled
LD_PRELOAD=build/lib/libhostpimsim-upmem-preload.so ./your_upmem_app

# Inspect virtual-device activity while the SDK code runs
LD_PRELOAD=build/lib/libhostpimsim-upmem-preload.so strace -e openat,ioctl ./your_upmem_app
```

### Command Line Arguments

There is no standalone `hostpimsim-upmem` CLI. The launched application keeps
its normal arguments; the preload library is configured by process environment.

| Argument                         | Required | Default                      | Description                                   |
| -------------------------------- | -------- | ---------------------------- | --------------------------------------------- |
| `LD_PRELOAD`                     | Yes      |                              | Must include `libhostpimsim-upmem-preload.so` |
| `UPMEM_PROFILE`                  | No       | `backend=hw,regionMode=safe` | Forced by the preload constructor             |
| Application path / app arguments | Yes      |                              | Passed through unchanged to the target app    |

### Execution Flow

1. The preload constructor registers virtual `/dev` and sysfs nodes and hooks `liboverlaysys` syscall dispatch.
1. The first open of `/dev/dpu_rank0` allocates one `UPMEMPIMRank` and an 8GB anonymous backing aperture.
1. Rank ioctls (`WRITE_TO_RANK`, `READ_FROM_RANK`, `COMMIT_COMMANDS`, `UPDATE_COMMANDS`, `DEBUG_MODE`, `SLICE_INFO`) are forwarded into `upmem.cc` translation unit.
1. CI structure / payload frames program IRAM, WRAM, DMA control state, and per-bank selection masks.
1. Run-state reads and thread-control frames dispatch work to per-DPU handlers, which execute tasklets functionally through `Pipeline::execute_once()`.
1. Closing the rank fd destroys the rank object and unregisters the virtual backing device for that fd.

## Building

```bash
# On hostpimsim root directory
mkdir build
cd hostpimsim/build
cmake ..
cmake --build . --target libhostpimsim-upmem-preload.so
```

The shared library will be written to `build/lib/libhostpimsim-upmem-preload.so`.

## Comparison

| Feature            | hostpimsim-upmem                    | UPMEM driver + hardware        | Standalone ISA simulators     |
| ------------------ | ----------------------------------- | ------------------------------ | ----------------------------- |
| Form factor        | LD_PRELOAD shared library           | Kernel driver + DIMM           | Separate executable / library |
| Rank topology      | Fixed 1 rank, 8 banks x 8 DPUs      | Hardware-defined               | Simulator-defined             |
| IRAM / WRAM / MRAM | 24 KB / 64 KB / 64 MB per DPU       | 24 KB / 64 KB / 64 MB per DPU  | Varies                        |
| Control interface  | Virtual `/dev/dpu_rank0` + ioctls   | Real `/dev/dpu_rank*` + ioctls | Usually custom CLI / API      |
| Sysfs exposure     | Virtual in-process                  | Real kernel sysfs              | Usually none                  |
| Execution model    | Functional                          | Real hardware                  | Functional or cycle-accurate  |
| DAX mmap           | Node exists, mmap is stubbed        | Supported by driver / platform | Usually not applicable        |
| Binary compatible  | Yes, for SDK apps using rank ioctls | Yes                            | Usually no                    |
| Process sharing    | In-process only                     | System-wide device nodes       | Process-local                 |
| Kernel dependency  | None                                | Required                       | None                          |

## Implementation Notes

- The preload constructor always creates exactly one virtual rank node (`/dev/dpu_rank0`) and one DAX node (`/dev/dax0.0`) for the process.
- DAX support is discovery-only for now. `dax_ioctl_cb()` returns `ENOTTY` and `dax_mmap_cb()` intentionally fails with `EINVAL`.
- Rank `mmap` is also not exposed through the vdev layer; MRAM is kept as an internal logical-linear mapping.
- MRAM host transfers and DMA use logical offsets directly, so rank ioctl copies can stay contiguous.
- Sparse MRAM rank transfers are optimized when the request touches the same local DPU across multiple banks.
- `upmem_ioctl_debug_mode()` is currently a no-op that returns success.
- `upmem_ioctl_slice_info()` is supported and updates per-bank enabled/selected masks from the SDK-provided slice configuration.
- The performance counter is modeled, but timing remains functional rather than cycle-accurate.
