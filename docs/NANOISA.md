# NanoISA Virtual Machine Architecture

I provide a complete virtual machine backend as an alternative to C transpilation. I compile my source code to a custom bytecode format (.nvm) and execute it in a stack-based virtual machine with process-isolated FFI.

## Overview

```
.nano source
  |
  v
nano_virt (compiler)
  |
  +---> .nvm bytecode file (--emit-nvm)
  |       |
  |       v
  |     nano_vm (VM executor)
  |       |
  |       +---> nano_cop (FFI co-process, on demand)
  |       +---> nano_vmd (VM daemon, optional)
  |
  +---> native binary (default)
          embeds .nvm + VM runtime
```

## Binaries

| Binary | Description |
|--------|-------------|
| `nano_virt` | My compiler: .nano source to .nvm bytecode or native binary |
| `nano_vm` | My VM executor: loads and runs .nvm files |
| `nano_cop` | My FFI co-process: isolates external function calls |
| `nano_vmd` | My VM daemon: persistent VM process for reduced startup latency |

### nano_virt (Compiler)

```bash
nano_virt input.nano [-o output] [--run] [--emit-nvm] [--strip-debug] [--daemon-wrapper] [-v]
```

- `-o <path>`: Output file (native binary or .nvm)
- `--run`: Execute immediately after compilation (in-process VM)
- `--emit-nvm`: Write raw .nvm bytecode instead of native binary
- `--strip-debug`: Remove debug/source-map data from emitted module
- `--daemon-wrapper`: Generate thin daemon-mode binary (requires nano_vmd)
- `-v`: Verbose output

### nano_vm (Executor)

```bash
nano_vm [--daemon] [--isolate-ffi] <file.nvm>
```

- `--daemon`: Send .nvm to nano_vmd daemon (lazy-launches if needed)
- `--isolate-ffi`: Route FFI calls through nano_cop co-process

### nano_vmd (Daemon)

My daemon listens on a Unix domain socket (`/tmp/nanolang_vm_<uid>.sock`) and accepts .nvm blobs from clients. I use this to reduce startup cost when repeatedly running programs.

## Instruction Set Architecture

### Architecture Model

- **Local/stack hybrid** with an operand stack, indexed frame locals, and an instruction pointer
- **Runtime-typed values**: 16 bytes per value (1-byte tag + 15-byte payload)
- **Variable-length encoding**: 1-byte opcode + 0-4 operands
- **Little-endian** byte order
- **Two-plane opcode space**: primary identifiers occupy `0x00..0xFE`; the byte `0xFF` is a reserved *extension prefix*, never an instruction by itself

### Module Linking Model

I keep separately linked `.nvm` modules. I do not flatten dependency code,
functions, strings, or globals into the root file. The root module's
`MODULE_REFS` section stores dependency names in order; that order is the
module index used by `CALL_MODULE`. A host loads each file independently and
links it with `vm_link_named_module`, which rejects a missing, extra, or
out-of-order name before execution. I then resolve each `(module, function)`
pair to a callable handle once and preserve the module boundary on call frames.

This keeps serialized files independent while making their link order part of
the checked module format rather than an agreement hidden in host code.

### Value Types

| Tag | Code | Description |
|-----|------|-------------|
| `TAG_VOID` | 0x00 | No value |
| `TAG_INT` | 0x01 | 64-bit signed integer |
| `TAG_U8` | 0x02 | Unsigned byte |
| `TAG_FLOAT` | 0x03 | 64-bit IEEE 754 double |
| `TAG_BOOL` | 0x04 | true/false |
| `TAG_STRING` | 0x05 | Heap-allocated, GC-managed, immutable |
| `TAG_BSTRING` | 0x06 | Binary string with length |
| `TAG_ARRAY` | 0x07 | Dynamic array, GC-managed |
| `TAG_STRUCT` | 0x08 | Named struct with fields |
| `TAG_ENUM` | 0x09 | Integer variant index |
| `TAG_UNION` | 0x0A | Tagged union |
| `TAG_FUNCTION` | 0x0B | Function table index + optional closure env |
| `TAG_TUPLE` | 0x0C | Fixed-size heterogeneous container |
| `TAG_HASHMAP` | 0x0D | Key-value map |
| `TAG_OPAQUE` | 0x0E | RPC proxy ID (handle to co-process object) |

### Opcodes (178 total)

**Stack & Constants (0x00-0x0F):**
`NOP`, `PUSH_I64`, `PUSH_F64`, `PUSH_BOOL`, `PUSH_STR`, `PUSH_VOID`, `PUSH_U8`, `DUP`, `POP`, `SWAP`, `ROT3`

**Variable Access (0x10-0x1F):**
`LOAD_LOCAL`, `STORE_LOCAL` (FP-relative), `LOAD_GLOBAL`, `STORE_GLOBAL`, `LOAD_UPVALUE`, `STORE_UPVALUE` (closure captures)

**Arithmetic (0x20-0x27):**
`ADD` (also string concatenation), `SUB`, `MUL`, `DIV` (0 on div-by-zero), `MOD`, `NEG`

**Comparison (0x28-0x2F):** `EQ`, `NE`, `LT`, `LE`, `GT`, `GE`

**Logic (0x30-0x37):** `AND`, `OR`, `NOT`

**Control Flow (0x38-0x3F):**
`JMP` (relative i32), `JMP_TRUE`, `JMP_FALSE`, `CALL`, `CALL_INDIRECT`, `CALL_EXTERN` (FFI via RPC), `CALL_MODULE` (cross-module), `RET`

**String Ops (0x40-0x4F):**
`STR_LEN`, `STR_CONCAT`, `STR_SUBSTR`, `STR_CONTAINS`, `STR_EQ`, `STR_CHAR_AT`, `STR_FROM_INT`, `STR_FROM_FLOAT`

The retained string primitives are `STR_LEN`, `STR_CONCAT`, `STR_CHAR_AT`, `STR_EQ`, `STR_FROM_INT`, and `STR_FROM_FLOAT`, each justified by string representation or measured cost. Higher-level string algorithms (`STR_SUBSTR`, `STR_CONTAINS`, `STR_TRIM`, `STR_TO_LOWER`, `STR_TO_UPPER`, `STR_STARTS_WITH`, `STR_ENDS_WITH`, `STR_SPLIT`, `STR_REPLACE`) compose from those primitives and move to runtime libraries; the opcode values remain only for assembler compatibility. Aggregate primitives `AGG_PACK`, `AGG_GET`, `AGG_SET`, and `AGG_TAG` are all representation-justified. See [NanoISA Primitive String and Aggregate Operations](superpowers/specs/2026-09-01-nanoisa-primitive-string-aggregate-ops.md).

**Array Ops (0x50-0x5F):**
`ARR_NEW`, `ARR_PUSH`, `ARR_POP`, `ARR_GET`, `ARR_SET`, `ARR_LEN`, `ARR_SLICE`, `ARR_REMOVE`, `ARR_LITERAL`

**Struct Ops (0x60-0x67):**
`STRUCT_NEW`, `STRUCT_GET`, `STRUCT_SET`, `STRUCT_LITERAL`

**Union/Enum Ops (0x68-0x6F):**
`UNION_CONSTRUCT`, `UNION_TAG`, `UNION_FIELD`, `MATCH_TAG`, `ENUM_VAL`

**Tuple Ops (0x70-0x77):** `TUPLE_NEW`, `TUPLE_GET`

**Hashmap Ops (0x78-0x7F):**
`HM_NEW`, `HM_GET`, `HM_SET`, `HM_HAS`, `HM_DELETE`, `HM_KEYS`, `HM_VALUES`, `HM_LEN`

**GC/Memory (0x80-0x87):**
`GC_RETAIN`, `GC_RELEASE`, `GC_SCOPE_ENTER`, `GC_SCOPE_EXIT`

**Type Casts (0x88-0x8F):**
`CAST_INT`, `CAST_FLOAT`, `CAST_BOOL`, `CAST_STRING`, `TYPE_CHECK`

**Closures (0x90-0x97):**
`CLOSURE_NEW` (fn_idx + capture_count), `CLOSURE_CALL`

**I/O & legacy debug (0xA0-0xAF):**
`PRINT`, `ASSERT`, `DEBUG_LINE`, `HALT`. NanoVirt records source locations in
the debug side table and does not emit `DEBUG_LINE`; the opcode remains only
for old hand-written assembly until NanoISA v2 removes it. NanoISA v2 folds
these I/O opcodes into the typed `trap` family (see Typed Traps below).

**Opaque Proxy (0xB0-0xBF):**
`OPAQUE_NULL`, `OPAQUE_VALID`

### v2 Compact Encodings (design)

NanoISA v2 (`spec/nanoisa.yaml`, `instruction_families`) adds compact encodings
that shrink common instructions without adding new instruction meanings. Each
compact form is an *encoding-only* alias of a canonical family instruction: it
keeps the same mnemonic family, stack effect, and ownership, so assembly text
stays regular. Assemblers pick a compact encoding when the operand value fits
its bounded range; disassemblers always render the canonical operand.

- **Compact constants** — `const.i64.small` stores a small signed integer
  (range −64..63) inline instead of a full `sleb` immediate; it decodes to
  `const.i64`.
- **Short local forms** — `local.get.short` / `local.set.short` address the
  first 16 frame locals with an inline nibble instead of a `uleb` operand; they
  decode to `local.get` / `local.set`.
- **Compact general operands** — a single-byte `compact` operand (range 0..255)
  reused by any family instruction whose sole operand is a small count or index:
  `pick.compact`, `roll.compact`, `global.get.compact`, `global.set.compact`,
  `aggregate.get.compact`, and `aggregate.set.compact`. Each decodes to its
  canonical `uleb`-operand instruction with an identical stack effect.

Compact operand kinds (`small-constant`, `local-short`, `compact`) declare the
canonical variable-length encoding they decode to and their permitted value
range, so a compact form never changes what an instruction means.

### Extended Opcode Space

Opcode bytes are identifiers, not a running instruction count. The primary
plane uses one byte per opcode across `0x00..0xFE`
(`NANOISA_PRIMARY_OPCODE_LIMIT` is the exclusive upper bound of that range).

The byte `0xFF` is the **extension prefix** (`OP_EXTENSION_PREFIX` /
`NANOISA_EXTENSION_PREFIX`). A decoder that reads it must not treat it as an
instruction: it reads one more byte and resolves the pair through the extended
plane (`nanoisa_extended_opcodes[]`, `isa_get_extended_info`). This gives a
clean, non-overlapping way to grow the instruction set — up to 256 additional
opcodes — without renumbering existing opcodes or conflating an opcode value
with a count. The extended plane is defined in `spec/nanoisa.yaml`
(`extended_opcodes`) and is currently empty; families migrate into it as
NanoISA v2 lands.
### Portable ISA design (v2)

My next-generation instruction set lives in `spec/nanoisa.yaml`, and I generate
`src/nanoisa/generated_schema.h` from it so the design and the code never drift.
That schema follows two rules I hold myself to.

First, every portable instruction has one comprehensible meaning. I do not
overload an opcode to do two jobs depending on its operands. `i64.add` adds
integers; `f64.add` adds floats; `const.i64` pushes an integer literal. Each
family entry carries a one-line `meaning`, and I refuse to generate the schema
if any meaning is missing or shared with another instruction. If you read the
mnemonic, you know what it does.

Second, I keep operand forms symmetric. Every operand names a kind declared in
`operand_kinds`; no instruction hides a raw encoding inline. Matching
instructions take matching operands: `local.get` and `local.set` both take a
`local`, `global.get` and `global.set` both take a `global`, and every `mem.*`
load and store takes the same `[offset, align]` pair. When two instructions are
mirror images, their operand lists are too.

I enforce both rules in `scripts/gen_nanoisa_schema.py`. The generator validates
the schema before it emits anything, so a design that breaks either rule fails
`make schema-check` instead of shipping.

## .nvm Binary Format

### Header (32 bytes)

```
[magic: "NVM\x01" (4B)] [version (4B)] [flags (4B)] [entry_point (4B)]
[section_count (4B)] [string_pool_offset (4B)] [string_pool_length (4B)] [CRC32 checksum (4B)]
```

### Sections

Each section has a 12-byte directory entry: `[type (4B)] [offset (4B)] [size (4B)]`

| Section | Type | Contents |
|---------|------|----------|
| `CODE` | 0x0001 | Bytecode instructions |
| `STRINGS` | 0x0002 | String constant pool (deduplicated) |
| `FUNCTIONS` | 0x0003 | Function table (name, arity, code offset, locals, upvalues) |
| `STRUCTS` | 0x0004 | Struct type definitions |
| `ENUMS` | 0x0005 | Enum type definitions |
| `UNIONS` | 0x0006 | Union type definitions |
| `GLOBALS` | 0x0007 | Global variable declarations |
| `IMPORTS` | 0x0008 | Extern function stubs for FFI |
| `DEBUG` | 0x0009 | Source maps (bytecode offset to source line) |
| `METADATA` | 0x000A | Module name, version |
| `MODULE_REFS` | 0x000B | Referenced module names for cross-module linking |

### String Pool

Variable-length entries: `[length: u32] [utf8_bytes: length]`. I deduplicate strings at compile time.

### Function Entries (18 bytes each)

```
[name_idx (4B)] [arity (2B)] [code_offset (4B)] [code_length (4B)] [local_count (2B)] [upvalue_count (2B)]
```

## Execution Model

### Execution Representations

I keep three separate representations of a program so that serialization,
verification, and dispatch each stay simple and independently testable:

1. **Compact serialized bytecode** (`NvmModule`, `src/nanoisa/nvm_format.*`) is
   the on-disk and on-wire form: variable-length, byte-addressed instructions
   plus the string pool, function table, and section layout.
2. **Verified instruction IR** (`VmDecodedModule`, `src/nanovm/vm_decode.*`) is
   the result of one decode pass per function. It establishes instruction
   boundaries and resolves every branch and direct call against a verified
   boundary map. This is the representation the verifier reasons about; it is
   byte-offset addressed.
3. **Optimized dispatch IR** (`VmDispatchModule`, `src/nanovm/vm_dispatch.*`) is
   a projection of the verified IR shaped for the hot fetch loop. Instructions
   live in a flat, instruction-indexed array, the linear-path successor is a
   precomputed instruction index, and branch and call targets are precomputed
   as dispatch indices. It is derived from — and validated against — the
   verified IR and is rebuilt in lockstep whenever the verified IR is rebuilt.

`vm_core_execute` executes representation 3. A dispatch cursor advances by
instruction index on the linear path and consults a byte-offset map only to
re-enter the stream after a jump, call, or return, which keeps the byte-addressed
`ip` contract the frames, traps, and returns depend on.

### Trap Architecture

My VM separates pure computation from side effects.

My **pure core** (`vm_core_execute`) handles 83+ opcodes:
- All arithmetic, logic, comparison
- Stack manipulation, variable access
- Data structure operations
- Control flow (jumps, calls, returns)

When my core encounters a side-effecting operation, it returns a **trap descriptor**:

| Trap | Trigger | Handler Action |
|------|---------|----------------|
| `TRAP_EXTERN_CALL` | `OP_CALL_EXTERN` | Route to co-process FFI |
| `TRAP_PRINT` | `OP_PRINT` | Write to stdout |
| `TRAP_ASSERT` | `OP_ASSERT` | Check boolean, abort if false |
| `TRAP_HALT` | `OP_HALT` | Stop execution |
| `TRAP_ERROR` | Runtime error | Report and terminate |

My **harness** (`vm_execute`) dispatches traps and resumes the core. I chose this separation to enable potential FPGA implementation of the pure-compute core.

#### Typed Traps (NanoISA v2)

NanoISA v1 exposed side effects as special opcodes (`PRINT`, `PRINTLN`,
`ASSERT`, `HALT`, `CALL_EXTERN`). NanoISA v2 replaces them with a regular
`trap` instruction family so that every side effect is one composable
instruction carrying explicit stack effects and `trap` ownership. The core
still suspends on a trap and the harness resumes it, so the FPGA-friendly
pure/effect split is preserved while the opcode space stays regular.

| Typed trap | v1 opcode | Stack effect | Handler action |
|------------|-----------|--------------|----------------|
| `trap.print` | `PRINT` | pops value | Write value to stdout |
| `trap.println` | `PRINTLN` | pops value | Write value plus newline to stdout |
| `trap.assert` | `ASSERT` | pops bool | Abort if the condition is false |
| `trap.halt` | `HALT` | none | Stop execution |
| `trap.host` | `CALL_EXTERN` | signature args to result | Route to the FFI co-process |
| `trap.dispatch` | reserved | operand-defined | Generic escape for future host traps |

The normative definitions live in `spec/nanoisa.yaml` under the `trap`
instruction family and are generated into `src/nanoisa/generated_schema.h`.

### Memory Management

I use reference-counted GC with scope-based auto-release:
- `OP_GC_RETAIN` / `OP_GC_RELEASE` - Manual reference counting
- `OP_GC_SCOPE_ENTER` / `OP_GC_SCOPE_EXIT` - Automatic release on scope exit

I insert scope markers for let-bindings at compile time.

### Call Frames

Each function call I execute pushes a frame with:
- Function index and return address
- Stack base (where locals begin)
- Local count (including parameters)
- Closure environment (if closure call)
- Module reference (for cross-module calls)

## Co-Process FFI Protocol

I isolate external function calls in a `nano_cop` co-process. I use a **shared-memory mailbox** as the fast path, with a pipe-based fallback for large payloads.

### Fast Path: Shared-Memory Mailbox

I `mmap` a `CopMailbox` region with `MAP_SHARED | MAP_ANON` before `fork()`. Because I fork without `exec`, the mailbox pointer is valid in both parent and child address spaces at zero copy cost.

**Per-call cost: 2 syscalls** — one 1-byte write to wake the child, one 1-byte read to receive the ack. All payload data lives in the shared region; no kernel copies of arguments or results.

**Mailbox layout:**

```
Request slot (written by parent, read by child):
  req_import_idx (u32)
  req_argc       (u16)
  req_data_size  (u16)
  req_data       [4096 bytes]   ← serialized args

Response slot (written by child, read by parent):
  resp_is_error  (u8)           ← 0=result, 1=error string
  resp_data_size (u32)
  resp_data      [4096 bytes]   ← serialized result
  resp_error     [256 bytes]    ← error message if resp_is_error=1
```

**Per-call timeout** — I use `poll()` with a configurable timeout (default 5000 ms, override with `COP_TIMEOUT_MS` env var). If the child does not ack in time, I kill and restart it.

### Pipe Fallback (large payloads)

When serialized args exceed 4096 bytes, I fall back to the original pipe protocol:

8-byte header: `[version (1B)] [msg_type (1B)] [reserved (2B)] [payload_len (4B)]`

**VM to Co-Process:** `COP_MSG_FFI_REQ` (0x02) — import index + serialized args  
**Co-Process to VM:** `COP_MSG_FFI_RESULT` (0x10) — serialized return value; `COP_MSG_FFI_ERROR` (0x11) — error string

### Value Serialization

| Type | Encoding |
|------|----------|
| INT | i64 (8 bytes, little-endian) |
| FLOAT | f64 (8 bytes, IEEE 754) |
| BOOL | u8 (0 or 1) |
| STRING | length (u32) + UTF-8 data |
| ARRAY | elem_type (u8) + count (u32) + serialized elements |
| OPAQUE | i64 proxy ID |
| VOID | 0 bytes |

### Lifecycle

1. I `mmap` a `CopMailbox` and create two 1-byte signal pipes
2. I `fork()` without `exec` — the child calls `cop_child_main()` directly
3. The child initializes FFI and writes a 1-byte ready signal
4. For each FFI call: I write 1 byte to wake the child, the child dispatches, writes 1 byte ack
5. On shutdown: I close the send pipe (child sees EOF and exits cleanly)

If my co-process crashes, I detect it via `waitpid(WNOHANG)` and recover. I isolate FFI crashes from my execution.

## Compiler Backend (Codegen)

My `nano_virt` compiler translates my AST to NanoISA bytecode in three passes.

**Pass 1 - Type Registration:** I register all struct, enum, union definitions. I register function signatures and globals.

**Pass 1.5 - Global Initialization:** I compile the `__init__` function for global variable initializers.

**Pass 2 - Function Bodies:** I use two sub-passes per function:
- 2a: I scan for loop/branch labels
- 2b: I generate bytecode with forward jump patching

### Native Binary Generation

My wrapper generator (`wrapper_gen.c`) produces standalone native executables:

1. **Full wrapper** (default): I embed .nvm bytecode and link my full VM runtime. I support all features including closures, cross-module calls, and FFI.
2. **Daemon wrapper** (`--daemon-wrapper`): I create a thin binary that connects to my `nano_vmd` daemon. This footprint is smaller but requires the daemon to be running.

### Symbolic Assembly Operands

I accept names where an instruction otherwise takes a function, import, field,
type/layout, string constant, or branch target index. Function declarations and
named strings declare their own symbols. `.symbol` names indices supplied by
other module sections:

```text
.string greeting "hello"
.symbol import write 0
.symbol type Point 0
.symbol field x 0
.function main 0 0 0 void 0
start:
  PUSH_STR greeting
  CALL_EXTERN write
  STRUCT_NEW Point
  STRUCT_GET x
  JMP start
.end
.entry main
```

Numeric operands remain valid. Symbol kinds are separate, so a function and a
field may have the same name without ambiguity.

## Source Files

### NanoISA (`src/nanoisa/`)

| File | Lines | Purpose |
|------|-------|---------|
| `isa.h` / `isa.c` | 401 | My instruction set definition, encode/decode |
| `nvm_format.h` / `nvm_format.c` | 618 | My binary format serialization, CRC32 |
| `assembler.h` / `assembler.c` | 736 | My two-pass text assembler |
| `disassembler.h` / `disassembler.c` | 246 | My binary to text with label reconstruction |

### NanoVM (`src/nanovm/`)

| File | Lines | Purpose |
|------|-------|---------|
| `vm.h` / `vm.c` | 1,844 | My core switch-dispatch interpreter |
| `vm_decode.h` / `vm_decode.c` | ~300 | My verified instruction IR: one-pass decode with boundary and branch/call resolution |
| `vm_dispatch.h` / `vm_dispatch.c` | ~230 | My optimized dispatch IR projected from the verified IR for the hot fetch loop |
| `value.h` / `value.c` | 225 | My NanoValue constructors, type checking |
| `heap.h` / `heap.c` | 595 | My reference-counting GC |
| `vm_builtins.c` | 297 | My runtime builtins |
| `vm_ffi.h` / `vm_ffi.c` | ~700 | My FFI lifecycle: shared-memory mailbox fast path, pipe fallback, per-call timeout |
| `cop_protocol.h` / `cop_protocol.c` | ~350 | My co-process wire protocol and `cop_child_main` service loop |
| `cop_main.c` | ~175 | My `nano_cop` binary (pipe-protocol main loop for legacy/standalone use) |
| `vmd_protocol.h` / `vmd_protocol.c` | 150 | My daemon wire protocol |
| `vmd_client.c` | 275 | My daemon client connector |
| `vmd_server.c` | 430 | My daemon server handler |
| `vmd_main.c` | 52 | My `nano_vmd` binary |
| `main.c` | 214 | My `nano_vm` binary |

### NanoVirt (`src/nanovirt/`)

| File | Lines | Purpose |
|------|-------|---------|
| `codegen.h` / `codegen.c` | 3,083 | My AST to bytecode compiler |
| `wrapper_gen.h` / `wrapper_gen.c` | 574 | My native executable generator |
| `main.c` | 331 | My `nano_virt` binary |

**Total: ~11,000 lines of C**

## Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/nanoisa/test_nanoisa.c` | 470 | I test ISA encoding, NVM format, assembler, disassembler |
| `tests/nanovm/test_vm.c` | 150 | I test all VM operations, GC, cross-module calls, daemon, co-process |
| `tests/nanovm/test_codegen.c` | 62 | I test bytecode generation from AST |

## Design Decisions

- **Division by zero produces 0** - I do this to match my Coq formal semantics (total division).
- **ADD is polymorphic** - I handle both integer addition and string concatenation with this opcode.
- **Relative jump offsets** - I use signed i32 offsets, relative to the start of the instruction.
- **String pool deduplication** - I deduplicate strings at compile time to save .nvm file size.
- **Per-frame module tracking** - Each call frame I create records its module for cross-module resolution.
- **Link-time callable handles** - I resolve every `CALL_MODULE` (module index, function index) operand pair into a direct callable handle during linking, so dispatch follows a resolved module/function pointer instead of re-indexing the module and function tables and repeating bounds checks on every call. Relinking or rebuilding a module re-resolves the handles.
- **Opaque proxy values** - I represent FFI objects as integer IDs. I keep the actual handles in my co-process address space.
