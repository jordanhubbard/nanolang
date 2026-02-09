# NanoISA Virtual Machine Architecture

NanoLang includes a complete virtual machine backend as an alternative to C transpilation. The system compiles NanoLang source to a custom bytecode format (.nvm) and executes it in a stack-based virtual machine with process-isolated FFI.

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
| `nano_virt` | Compiler: .nano source to .nvm bytecode or native binary |
| `nano_vm` | VM executor: loads and runs .nvm files |
| `nano_cop` | FFI co-process: isolates external function calls |
| `nano_vmd` | VM daemon: persistent VM process for reduced startup latency |

### nano_virt (Compiler)

```bash
nano_virt input.nano [-o output] [--run] [--emit-nvm] [--daemon-wrapper] [-v]
```

- `-o <path>`: Output file (native binary or .nvm)
- `--run`: Execute immediately after compilation (in-process VM)
- `--emit-nvm`: Write raw .nvm bytecode instead of native binary
- `--daemon-wrapper`: Generate thin daemon-mode binary (requires nano_vmd)
- `-v`: Verbose output

### nano_vm (Executor)

```bash
nano_vm [--daemon] [--isolate-ffi] <file.nvm>
```

- `--daemon`: Send .nvm to nano_vmd daemon (lazy-launches if needed)
- `--isolate-ffi`: Route FFI calls through nano_cop co-process

### nano_vmd (Daemon)

Listens on a Unix domain socket (`/tmp/nanolang_vm_<uid>.sock`) and accepts .nvm blobs from clients. Useful for reducing startup cost when repeatedly running programs.

## Instruction Set Architecture

### Architecture Model

- **Stack machine** with 5 virtual registers: SP (stack pointer), FP (frame pointer), IP (instruction pointer), R0 (accumulator), R1 (scratch)
- **Runtime-typed values**: 16 bytes per value (1-byte tag + 15-byte payload)
- **Variable-length encoding**: 1-byte opcode + 0-4 operands
- **Little-endian** byte order

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

**I/O & Debug (0xA0-0xAF):**
`PRINT`, `ASSERT`, `DEBUG_LINE`, `HALT`

**Opaque Proxy (0xB0-0xBF):**
`OPAQUE_NULL`, `OPAQUE_VALID`

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

Variable-length entries: `[length: u32] [utf8_bytes: length]`. Strings are deduplicated at compile time.

### Function Entries (18 bytes each)

```
[name_idx (4B)] [arity (2B)] [code_offset (4B)] [code_length (4B)] [local_count (2B)] [upvalue_count (2B)]
```

## Execution Model

### Trap Architecture

The VM separates pure computation from side effects:

**Pure core** (`vm_core_execute`) handles 83+ opcodes:
- All arithmetic, logic, comparison
- Stack manipulation, variable access
- Data structure operations
- Control flow (jumps, calls, returns)

When the core encounters a side-effecting operation, it returns a **trap descriptor**:

| Trap | Trigger | Handler Action |
|------|---------|----------------|
| `TRAP_EXTERN_CALL` | `OP_CALL_EXTERN` | Route to co-process FFI |
| `TRAP_PRINT` | `OP_PRINT` | Write to stdout |
| `TRAP_ASSERT` | `OP_ASSERT` | Check boolean, abort if false |
| `TRAP_HALT` | `OP_HALT` | Stop execution |
| `TRAP_ERROR` | Runtime error | Report and terminate |

The **harness** (`vm_execute`) dispatches traps and resumes the core. This separation enables potential FPGA implementation of the pure-compute core.

### Memory Management

Reference-counted GC with scope-based auto-release:
- `OP_GC_RETAIN` / `OP_GC_RELEASE` - Manual reference counting
- `OP_GC_SCOPE_ENTER` / `OP_GC_SCOPE_EXIT` - Automatic release on scope exit
- The compiler inserts scope markers for let-bindings

### Call Frames

Each function call pushes a frame with:
- Function index and return address
- Stack base (where locals begin)
- Local count (including parameters)
- Closure environment (if closure call)
- Module reference (for cross-module calls)

## Co-Process FFI Protocol

External function calls are isolated in a separate `nano_cop` process, communicating over pipes with a binary protocol.

### Wire Format

8-byte header: `[version (1B)] [msg_type (1B)] [reserved (2B)] [payload_len (4B)]`

### Message Types

**VM to Co-Process:**
- `COP_MSG_INIT` (0x01) - Send import table
- `COP_MSG_FFI_REQ` (0x02) - Call external function (import index + serialized args)
- `COP_MSG_SHUTDOWN` (0x03) - Terminate

**Co-Process to VM:**
- `COP_MSG_FFI_RESULT` (0x10) - Serialized return value
- `COP_MSG_FFI_ERROR` (0x11) - Error string
- `COP_MSG_READY` (0x12) - Init complete

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

1. VM forks `nano_cop` subprocess (connected via pipes)
2. VM sends `COP_MSG_INIT` with import table
3. Co-process loads FFI modules, sends `COP_MSG_READY`
4. For each FFI call: VM sends `COP_MSG_FFI_REQ`, co-process sends result
5. VM sends `COP_MSG_SHUTDOWN` on exit

If the co-process crashes, the VM detects the broken pipe and can recover gracefully. FFI crashes are fully isolated from VM execution.

## Compiler Backend (Codegen)

The `nano_virt` compiler translates NanoLang AST to NanoISA bytecode in three passes:

**Pass 1 - Type Registration:** Register all struct, enum, union definitions; register function signatures and globals.

**Pass 1.5 - Global Initialization:** Compile `__init__` function for global variable initializers.

**Pass 2 - Function Bodies:** Two sub-passes per function:
- 2a: Scan for loop/branch labels
- 2b: Generate bytecode with forward jump patching

### Native Binary Generation

The wrapper generator (`wrapper_gen.c`) produces standalone native executables:

1. **Full wrapper** (default): Embeds .nvm bytecode + links full VM runtime. Supports all features including closures, cross-module calls, and FFI.
2. **Daemon wrapper** (`--daemon-wrapper`): Thin binary that connects to `nano_vmd` daemon. Much smaller footprint, requires daemon running.

## Source Files

### NanoISA (`src/nanoisa/`)

| File | Lines | Purpose |
|------|-------|---------|
| `isa.h` / `isa.c` | 401 | Instruction set definition, encode/decode |
| `nvm_format.h` / `nvm_format.c` | 618 | Binary format serialization, CRC32 |
| `assembler.h` / `assembler.c` | 736 | Two-pass text assembler |
| `disassembler.h` / `disassembler.c` | 246 | Binary to text with label reconstruction |

### NanoVM (`src/nanovm/`)

| File | Lines | Purpose |
|------|-------|---------|
| `vm.h` / `vm.c` | 1,844 | Core switch-dispatch interpreter |
| `value.h` / `value.c` | 225 | NanoValue constructors, type checking |
| `heap.h` / `heap.c` | 595 | Reference-counting GC |
| `vm_builtins.c` | 297 | Runtime builtins |
| `vm_ffi.h` / `vm_ffi.c` | 611 | Fork/pipe/exec FFI lifecycle |
| `cop_protocol.h` / `cop_protocol.c` | 227 | Co-process wire protocol |
| `cop_main.c` | 212 | `nano_cop` binary |
| `vmd_protocol.h` / `vmd_protocol.c` | 150 | Daemon wire protocol |
| `vmd_client.c` | 275 | Daemon client connector |
| `vmd_server.c` | 430 | Daemon server handler |
| `vmd_main.c` | 52 | `nano_vmd` binary |
| `main.c` | 214 | `nano_vm` binary |

### NanoVirt (`src/nanovirt/`)

| File | Lines | Purpose |
|------|-------|---------|
| `codegen.h` / `codegen.c` | 3,083 | AST to bytecode compiler |
| `wrapper_gen.h` / `wrapper_gen.c` | 574 | Native executable generator |
| `main.c` | 331 | `nano_virt` binary |

**Total: ~11,000 lines of C**

## Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/nanoisa/test_nanoisa.c` | 470 | ISA encoding, NVM format, assembler, disassembler |
| `tests/nanovm/test_vm.c` | 150 | All VM operations, GC, cross-module calls, daemon, co-process |
| `tests/nanovm/test_codegen.c` | 62 | Bytecode generation from AST |

## Design Decisions

- **Division by zero produces 0** - Matches the Coq formal semantics (total division)
- **ADD is polymorphic** - Handles both integer addition and string concatenation
- **Relative jump offsets** - Signed i32, relative to instruction start
- **String pool deduplication** - Compile-time dedup saves .nvm file size
- **Per-frame module tracking** - Each call frame records its module for cross-module resolution
- **Opaque proxy values** - FFI objects represented as integer IDs; actual handles live in co-process address space
