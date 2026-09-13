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
  |       embeds .nvm + VM runtime (wrapper_gen; still a VM process)
  |
  +---> nvm2c (`bin/nvm2c`): structured C11 from a closed .nvm subset
          cc → native binary with no nano_vm / nano_cop in-process

5.0 makes `.nvm` the only compiler product and treats C/LLVM/Wasm as
translators of that module. Contract: [`docs/NANOISA_ONLY.md`](NANOISA_ONLY.md).
That rewrite is not 4.x work. `nano_virt -o` remains a packaged interpreter.
```

## Binaries

| Binary | Description |
|--------|-------------|
| `nano_virt` | My compiler: .nano source to .nvm bytecode or native binary |
| `nano_vm` | My VM executor: loads and runs .nvm files |
| `nano_cop` | My FFI co-process: isolates external function calls |
| `nano_vmd` | My VM daemon: persistent VM process for reduced startup latency |
| `bin/nanoisa` | Assemble `.nasm` and dump `.nvm` |
| `bin/nvm2c` | Translate a closed NanoISA subset to structured C11 (no `nano_vm`) |
| `bin/nanoisa_emit` | Cut A: `src_nano` AST → `.nasm` for the pinned i64 `add`/`main` subset |

### nano_virt (Compiler)

```bash
nano_virt input.nano [-o output] [--run] [--emit-nvm] [--strip-debug] [--daemon-wrapper] [-v]
```

- `-o <path>`: Packaged interpreter (wrapper_gen embeds `nano_vm`), or `.nvm` if `--emit-nvm` / the path ends in `.nvm`. 5.0 native AOT is `bin/nvm2c`, not this default.
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
`GC_RETAIN`, `GC_RELEASE`

**Type Casts (0x88-0x8F):**
`CAST_INT`, `CAST_FLOAT`, `CAST_BOOL`, `CAST_STRING`, `TYPE_CHECK`

**Closures (0x90-0x97):**
`CLOSURE_NEW` (fn_idx + capture_count). Closures are invoked with `CALL_INDIRECT`, which handles both plain function values and closures.

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
That schema follows three rules I hold myself to.

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

Third, every public instruction records *why* it belongs in the ISA rather
than a runtime library. Each family entry carries a `justification` beside its
`meaning`: `representation`, `core-semantics`, `execution-substrate`,
`control-flow`, `host-boundary`, or `encoding`. A library written in NanoLang
and reached through `call.import` or a `trap` could provide anything else by
composing these, so anything else is a library word, not an instruction. The
generator refuses to emit the schema unless every instruction claims one of the
six justifications, and I generate the field into
`NanoisaV2Family.justification`. The full classification and the rule for future
instructions live in
[Why every public instruction belongs in the ISA](superpowers/specs/2026-09-01-nanoisa-public-instruction-rationale.md).

I enforce all three rules in `scripts/gen_nanoisa_schema.py`. The generator validates
the schema before it emits anything, so a design that breaks either rule fails
`make schema-check` instead of shipping.

The portable ISA defined here is deliberately kept separate from the verified
and optimized runtime representations my VM builds from a loaded module; see
[Portable NanoISA vs. Runtime Representations](NANOISA_PORTABLE_ISA.md) for
which layer is the portable contract and which layers are internal.
## .nvm Binary Format

Since 4.0 a `.nvm` file is a NanoISA **v2** container, magic `NVM\x02`. The v1
container (`NVM\x01`) is retired: `.nvm` files are build artifacts rather than
distributed packages, so the loader refuses a v1 module with

```
module was built for NanoISA v1 (NVM\x01); rebuild it with nanoc 4.0 or later
```

rather than carrying a compatibility path that would have to stay correct
forever. `--emit-nvm` writes v2; `--emit-nvm-v2` remains accepted as a retired
alias so build scripts written during the transition keep working.

### Header (40 bytes)

```
[magic: "NVM\x02" (4B)] [format_version (2B)] [isa_version (2B)] [feature_bits (4B)]
[total_size (8B)] [header_size (4B)] [section_count (4B)] [entry_point (4B)]
[flags (4B, reserved, must be zero)] [CRC32 checksum (4B)]
```

`total_size` is what bounds everything else. Every section's range is checked
against it **by subtraction** (`size > total - offset`), never by addition,
which would wrap for an offset near the top of the range and admit a section
that appears to fit.

`feature_bits` names the capabilities a reader needs: `LINKED`, `FFI`, `DEBUG`,
`COPROCESS`. A bit this reader does not implement is refused rather than
ignored, so a module using a feature is never silently half-understood. The
rule is a **floor**: the bits must include everything the module's tables
require, and may include more, because a capability can be needed without
leaving a trace in a table -- a module may declare it needs the FFI with no
import listed, which is what v1's `.flag needs_extern` says.

`entry_point` is a FUNCTIONS index, or `0xFFFFFFFF` for a module with no entry
point. v1 left the field at 0 and marked the absence with a flag, so a module
with no `main` appeared to name function 0.

### Sections

Each section has a 24-byte directory entry:
`[type (4B)] [flags (4B, reserved)] [offset (8B)] [size (8B)]`. Entries are
emitted in ascending type order, which makes the file canonical -- two
producers emitting the same module emit the same bytes -- and lets validation
be a single forward pass.

| Section | Type | Contents |
|---------|------|----------|
| `METADATA` | 0x0001 | Key/value constant-index pairs |
| `CONSTANTS` | 0x0002 | Typed constant pool |
| `SIGNATURES` | 0x0003 | Call shapes, deduplicated |
| `LAYOUTS` | 0x0004 | Struct, tuple, union and enum layouts |
| `FUNCTIONS` | 0x0005 | Function table |
| `CODE` | 0x0006 | Bytecode instructions |
| `GLOBALS` | 0x0007 | Global variable declarations |
| `IMPORTS` | 0x0008 | Foreign functions |
| `LINKS` | 0x0009 | Cross-module dependencies |
| `DEBUG` | 0x000A | Source maps; optional |

### Constants

`[tag: u8] [pad: u8[3]] [length: u32] [payload: length, padded to 4]`

The explicit length is the point. v1's pool was read with `strlen`, so a string
holding an embedded zero was silently truncated; here the bytes survive
verbatim.

### Signatures

`[param_count: u16] [result_count: u16] [param_tags] [result_tags]`, each tag
array padded to 4.

Functions, imports, links and indirect call sites reference a signature by
index rather than each carrying its own shape. Producers must deduplicate, so
two identically-shaped callables share an entry: that is what makes comparing
signature indices a valid equality test, and it is why v2 import entries have
no variable-length type tail the way v1's did.

### Layouts

`[kind: u8] [pad: u8] [field_count: u16] [name_idx: u32]`, then per field
`[type_tag: u8] [pad: u8[3]] [nested_idx: u32] [name_idx: u32]`.

Every nested index refers to a **lower-numbered** layout, which makes the table
acyclic by construction: a decoder validates it in one forward pass and nothing
walking it can recurse forever. A forward or self reference is rejected.

### Function Entries (32 bytes each)

```
[name_idx (4B)] [signature_idx (4B)] [code_offset (8B)] [code_length (8B)]
[local_count (2B)] [upvalue_count (2B)] [max_stack (2B)] [flags (2B, reserved)]
```

Arity and result shape are absent because they live in SIGNATURES. Offsets are
64-bit; v1's were 32.

`max_stack` is the maximum operand-stack depth the function reaches. The
producer computes it and the loader **confirms** it against the verifier: a
module declaring less depth than it uses is rejected, because a disagreement
between producer and verifier otherwise surfaces as a stack overflow at run
time. A declared 0 means the producer had nothing to declare.

### Foreign artifact bindings in 5.0

I encode import kind `0` for logical FFI, `1` for coprocess imports, and `2`
for an exact foreign artifact. For kind `2`, `module_name_idx` names an
absolute library path with no embedded NUL. I retain this kind through my v2
reader and writer; my legacy v1 writer refuses nonzero kinds rather than
erasing them. Older v2 readers reject kind `2` as unknown.

My bytecode CLI retains the generation returned by each imported C-module
build. Both root shadows and production imports use that generation's
library. At runtime I open that path and resolve its entry symbol through
that library handle, without falling back to an unrelated loaded symbol.
A missing library or entry symbol is an execution failure.

These are local absolute bindings, not relocatable packages or authenticated
artifacts. Library dependencies and internal C symbol interposition still
follow the platform loader. My source-level imported-name isolation remains
roadmap work; handle-scoped lookup does not repair merged source declarations.

My packaged interpreter links the prebuilt runtime objects and their crypto
dependency. I retain the build's optional Homebrew OpenSSL library directory
for this link; system installations use the compiler's normal library search.
I test a generated standalone executable from another working directory,
including retention after a foreign rebuild and failure after artifact removal.
Both standalone and daemon wrapper builders compile in a private directory
beside the output and publish by rename only after a successful compiler exit
and executable-file checks. Failed builds do not replace the destination.
Overlapping successes follow last-successful-publication order; a destination
symlink is replaced, not followed. I quote path arguments and encode imported
paths as C-string bytes. `NANO_CC` and `CC` remain trusted shell command
configuration. Killed builds can leave private stages, which later builds
ignore; I retain unknown compiler side artifacts with a diagnostic rather
than recursively deleting them. This is not relocatable deployment,
power-loss durability, or protection from a malicious configured compiler.

For foreign C-module cache generations, I now check ordered `fsync` barriers:
regular generation files, then the generation directory, then the cache
directory and its ancestors after naming the generation, then the cache
directory again after switching `current`.
When removing a failed private build, I open its directory without following
a root symlink and remove entries relative to that descriptor. I unlink
symlink entries themselves, never their targets, and do not recurse into
nested directories. I report retained private files if cleanup cannot finish.
My boundary tests include replacing the directory path with a symlink after
opening it: entry removal stays in the original directory. This does not
isolate arbitrary compiler code or make the enclosing cache namespace safe
against a hostile process with write access.

A released module-build lock does not mean every private stage is abandoned.
My controlled compiler test kills only the builder, obtains a new acknowledgement
from its surviving child, and publishes a replacement through another builder
while that child remains paused. After release, the child successfully writes
its original private output. The old and new published generations remain
byte-for-byte unchanged, and warm reuse keeps the replacement. I test local and
shared caches. Killing the whole process group would not exercise this case.
I therefore do not collect leftover stages merely because I acquired the module
lock, or because their creator PID is gone. Automatic collection needs a
descendant-aware lifetime mechanism or an explicit quiescence contract first.
Published generations also remain retained: copied bytecode can reference their
absolute paths outside the cache owner's inventory. Retention can consume
unbounded disk space; a size or age limit cannot safely infer those references.
Ordinary project and example `make clean` targets therefore preserve local
module caches, the default `obj/module_cache`, and the configured shared cache.
My project clean removes unrelated compiler trees and files but keeps cache
ancestors. It does not recursively invoke another recipe that can delete the
retained cache. I also retain earlier cache locations identified by my lock,
generation or private-stage names; changing the configured cache must not
discard an older generation or a surviving compiler child's stage. This is a
conservative retention hint, not a liveness test. The cleanup helper validates tree boundaries before removing
anything and does not traverse directory symlinks. I test the actual recipes
in disposable workspaces, then load a retained foreign library in a fresh
process and verify warm generation reuse. Manual cache reset requires retiring
dependent artifacts and quiescing compiler descendants; it is not a normal
clean operation.

I do not follow symlinks or recurse into unexpected directories while flushing
generation contents. Interrupted barriers retry; failed barriers fail the build.
Failure before the pointer switch leaves the old current generation intact.
Failure after the switch retains the newly referenced generation and reports
that its cache-directory barrier was not confirmed. I never delete that
generation and leave a dangling `current`. A subsequent warm build retries the
cache-directory and ancestor barriers before returning success. I resolve each
parent with `openat` from the current directory descriptor, proceeding from
child to parent until the root or a filesystem boundary. I repeat this chain
even when the directories already exist: existence does not establish that an
earlier creation was flushed. Establishing mounts belongs to the host.

On Darwin, I require `fcntl(F_FULLFSYNC)` after each successful `fsync`,
including directory barriers. I retry interruptions and fail unsupported or
failed requests; I do not silently fall back to the weaker operation. My v13
build context invalidates generations built before this requirement. Other
hosts retain their `fsync` path. Apple's
[fsync contract](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html)
distinguishes flushing host buffers from requesting that the drive flush its
cache. Successful requests still depend on filesystem and device behavior;
they are not a physical power-loss test.

I test syscall ordering, injected failures, retry and retained artifact reads.
My process-crash matrix sends `SIGKILL` to the actual builder at seven points:
file and generation-directory barriers, generation rename, the first cache
barrier, an ancestor barrier, pointer rename and the final cache barrier.
I cover both first publication and replacement in local and shared caches.
Before the pointer switch, readers see the previous generation or no published
generation. After the switch, fresh processes load a complete new library.
Recovery reacquires the released advisory lock and succeeds; subsequent warm
reuse retains the recovered generation. I also link and execute the native
objects and check retained old libraries in fresh processes, so already-loaded
code cannot hide a missing artifact.
These tests do not simulate device power loss. Physical device durability,
wrapper/output-file persistence and full crash-recovery acceptance remain open;
successful directory barriers alone do
not establish those broader claims.

On Linux, I validate a source-cache hit with one private shared-library link
using the retained native and shared-only objects. I construct that command
with the same recipe as a cold build. A byte-for-byte equal library keeps its
generation without recompiling C; changed bytes trigger rebuilding. A failed
validation link fails the build without retrying past that failure or changing
the current pointer. This catches the tested archive, earlier-search-candidate,
thin-member and response-file changes without interpreting GNU ld's lossy
dependency file. Warm builds now pay for one shared link. Nondeterministic
link output can prevent generation reuse.

This checks a link result, not a complete input inventory or an atomic source
snapshot. It does not pin the external dynamic libraries used later by the
runtime. Other linker modes and the full input-identity contract remain open.
I preserve configured flag fragments as trusted shell text, including quoted
newlines, rather than splitting their bytes on whitespace. My v14 cache context
rebuilds generations created with the earlier flag-splitting behavior.

My foreign-module builder also quotes source, object, dependency, library and
declared include paths. It refuses oversized commands before invoking the
compiler. Compiler commands and explicit flag fragments remain trusted
configuration. I request user and system header dependencies with
[`-MD`](https://clang.llvm.org/docs/ClangCommandLineReference.html#dependency-file-generation),
including transitive headers reached through `-isystem`. Cache reuse requires
all expected dependency records to decode and hash successfully. This checks
reported header contents, not the identity of the whole SDK or toolchain.
Known backslash or newline paths still compile, but I
withhold reuse records because compiler Make-format output can lose their
identity. I also capture `-H` include traces from the original compilation and
hash unambiguously decoded header paths alongside Make dependencies. This
repairs the tested backslash/slash alias without enabling saved-input mode.
Missing, malformed or ambiguous trace evidence withholds reuse; compiler
warnings and errors still reach stderr. Diagnostics mixed into a trace can
also withhold reuse. Lossless capture across all compiler modes remains
roadmap work. My [compiler-input experiment](COMPILER_INPUT_EVIDENCE.md)
records the original failure and explains why enabling saved preprocessed
inputs unconditionally is not a semantics-preserving repair.

For cache-eligible `.c` builds with supported literal flags, including captured
pkg-config flags, I retain Clang assembly (`.s`) or GCC preprocessed C (`.i`) in
private staging and compile those files. I hash the bytes while writing them
and require fresh capture to match before recording reuse evidence. The retained files
cover ordinary, multiple and shared-only C sources. Capture emits the original
dependency records and reports C diagnostics against the original source paths.
Failed or empty capture falls back to original compilation without a reuse
record. A failed retained-input compilation fails the build. My v20 context
invalidates older records. I identify the supported compiler family through
a successful version query; that query is not authentication.

My [configured flag boundary](SOURCE_SNAPSHOT_EVIDENCE.md#configured-scalar-flags)
lists the supported spellings. I retain optimization, standard, debug and
warning flags through GCC C code generation. Simple `-D`, `-U` and `-I` tokens, along with
declared include directories, apply during capture but not compilation of
already preprocessed input. Only common and active-platform flags choose this
mode. I decode literal words, quotes and escapes, including paired `-D`, `-U`
and `-I` arguments within a fragment. I do not evaluate shell expansions,
commands or globs to decode flags. Those forms, unlisted
options and words exceeding 4095 bytes keep their original compilation path.
This eligibility parser does not sandbox the original trusted shell text.

For recognized Clang/GCC drivers, literal `@file` arguments in common,
active-platform and package compiler flags are expanded once into invocation-local
argument strings. I parse GNU-style response words separately from shell words,
then quote each argument for my command runner. Nested response paths resolve
from the compiler working directory. Build and public rebuild checks capture
their own argument sets; selected metadata arguments join cache identity.
Missing, cyclic and nonregular response inputs fail without replacing an old
generation. Caller-owned metadata is not rewritten.

I exclude named `clang-cl` drivers and explicit `--driver-mode` overrides.
If metadata or package flags leave an unresolved response or shell fragment,
I preserve the original argument set instead of mixing response dialects or
partially expanded argument groups.

This path is bounded to 16 response nesting levels, 64 KiB cumulative input,
4095-byte words and a 64 KiB serialized fragment. Over-budget fragments,
unterminated quotes, trailing escapes, embedded NULs and shell-expanded
fragments keep the previous compiler path and do not gain snapshot eligibility
from this capture. Literal fragments over 1024 bytes use GNU response sidecars
under the module cache. Before transport, I coalesce eligible common, active-
platform and package compiler groups above 1024 bytes, within a 64 KiB combined
budget. Argument order stays intact; array slots and native-framework NULLs
remain stable. Allocation failure leaves the original strings untouched.
This handles many short compiler fragments as well as individual long ones.
I also quote and coalesce include-directory arguments in search order for
compilation and returned native flags. Original paths remain in metadata for
dependency validation. For shared-library linking, I combine package libraries,
system libraries, common and platform linker flags, and Darwin framework pairs
in their existing order. Literal groups up to 64 KiB use the same transport;
unexpanded linker `@` arguments stay visible and do not gain linker-cache eligibility.
I capture driver response arguments in common/platform linker metadata and
package libraries with the compiler-flag response set. Returned native linker
flags own those decoded arguments, so removing the original response file does
not change them. Package compiler and linker candidates are published together
or discarded together. This does not decode forwarded `-Wl,@file` syntax.
I publish complete read-only files with content-derived
names, verify their bytes before reuse, and leave them alive until that cache
is removed. Returned native flags therefore remain usable after build-info
cleanup. Decoded arguments remain in cache identity and phase selection;
sidecar paths are only transport. I use the same transport for shared linking,
and Darwin linker observation admits only exact sidecars of captured flags or
the current shared-link group.
Larger budgets, other response dialects and indirect linker response files
remain open. This is argument retention,
not retention of every external input named by an argument.

Clang applies all supported C flags during `-S` capture, then assembles without
C-only flags. Its assembly output expands the tested inline `.incbin`, nested
assembler includes and macros into retained bytes. Warm validation runs C code
generation again; it avoids another object assembly, not another C compilation.
GCC's `-S` output retains external directives and needs the separate literal-file
capture described below. See my [assembly capture evidence](SOURCE_SNAPSHOT_EVIDENCE.md#production-clang-assembly-capture)
for the tested boundary and remaining assembler work.

For supported GCC builds, I also fingerprint the actual ordinary and shared
objects before linking. Fresh validation privately captures C and compiles
objects again, then requires matching combined input/object fingerprints. I
also emit assembly from retained C and copy literal, line-leading `.include`
and `.incbin` inputs recursively into private staging. The assembler reads
those copies; binary offset/count expressions remain its responsibility.
Captured bytes join the fingerprint. Final assembly receives no C-only flags.
The tested restored literal-file edits now produce cold/warm/fresh 42.

This path is bounded: 16 MiB per file, 64 MiB total, 256 file visits and 16
include levels. Backslashes in assembly text, alternate macro/MRI modes, nonliteral or
non-line-leading file directives, unrepresentable private paths, nonregular
inputs and capture failures first try the Linux GNU-as read-capture path below,
then retain the earlier C/object validation path when that is unavailable.
That fallback can still produce a changed cold result; I withhold reuse when
validation differs. Captured assembly contains private staging paths and is
build evidence, not a relocatable replay bundle. Broader assembler semantics
and variants remain an [open boundary](SOURCE_SNAPSHOT_EVIDENCE.md#gcc-literal-assembler-file-capture).
Validation repeats a full C compilation after cold builds and on warm reuse. Private
checks use `TMPDIR` (or `/tmp`) and are removed after normal success or failure;
process termination can leave an orphan. This is not an atomic source snapshot.

On Linux, when literal capture cannot represent the input, I also support a
read-capture/replay path for dynamically linked ELF64 little-endian GNU as 2.40
and 2.42. I require an exact supported version token on the first banner line;
this selects compatibility, not executable authenticity.
GCC identifies its assembler with `-print-prog-name=as` and still supplies the
assembly arguments. A private `-B` wrapper loads my copied helper only in the
assembler child, through an inherited descriptor; paths may contain spaces.
I validate the sealed capture before selecting replay, and require both child
success and replay completion before accepting each replayed object. Fresh
validation repeats this recipe and compares combined capture/object evidence.
Helper bytes and the selected assembler path/content join the fingerprint.

Linux builds and installs place `nano_as_capture.so` beside the compiler/VM
drivers. `NANO_AS_CAPTURE_HELPER` can select another helper file and is part of
build-context identity. Ambient `LD_PRELOAD` or `LD_AUDIT`, unavailable helpers,
unidentified assemblers and unsupported versions keep the prior fallback.
The helper's stdio and size limits, trusted private storage, and remaining
variant boundary are described in my [capture evidence](SOURCE_SNAPSHOT_EVIDENCE.md#production-gnu-assembler-read-replay).
This is not a syscall sandbox or complete toolchain snapshot.

For GCC I add `-fpch-preprocess` to capture and warm validation. A
`#pragma GCC pch_preprocess` marker means the output still references external
PCH bytes. For canonical pragmas with unescaped paths I copy those bytes into
private staging and rewrite the retained input to reference the copies. Original
path spellings and copied bytes join the fingerprint; private destination names
do not. Selected-PCH include traces also record the PCH and root source. This
invalidates an earlier ordinary cache hit when a usable PCH appears, and detects
PCH replacement or removal. The marker check crosses input-buffer boundaries.
Malformed pragmas, escaped paths and failed or oversized copies retain the
original-compilation/no-reuse fallback. PCH copies use the existing regular-file
capture limits: 16 MiB per file, 64 MiB per translation unit, and 256 visits.
This is retained GCC PCH support, not arbitrary compiler-module retention.

Other compilers and configured modes keep their original path. I fingerprint
fresh preprocessing and its include trace before
warm reuse. This detects newly selected headers that an old dependency list
could not name. I use the same configured compile flags, including shared-only
flags; I do not compile the probe output. Cold builds require equal probe
observations before and after compilation before I store reuse evidence.
Failed, empty or changed observations withhold reuse, while normal compilation
still determines whether I can publish code. Each warm-cache validation costs
one preprocessing invocation per source; a cacheable cold build uses two. The
configured compiler remains trusted. Matching observations are not an atomic
snapshot and do not establish complete PCH, module or toolchain identity.
My [source snapshot experiment](SOURCE_SNAPSHOT_EVIDENCE.md) distinguishes the
repaired ordinary-C edit-and-restore cases from the remaining modes. Even
retained translation units do not snapshot assembler inputs, linker inputs,
the compiler itself, or the entire source tree at one instant.

`PKG_CONFIG` can select a pkg-config executable name or path. It is not a
shell-command fragment. When unset or empty, I keep my existing installation
lookup order. I quote the tool, package and search-path arguments, and require
successful query exit, complete reads and NUL-free output of at most 64 KiB.
A successful empty flag response is valid. A failed required flag query aborts
the build instead of supplying partial compile or link flags; an existing
generation stays intact. I capture each required compiler/linker response once
per build and share that set across preprocessing, compilation, shared linking
and returned build information, including source-free modules. Both responses
enter my cache fingerprint, so a link-only response change invalidates reuse.
Before recording a cold build as reusable, I query a fresh set: changed or
failed responses withhold reuse evidence without changing the flags used to
build the published artifacts. Capturing sequential responses is not an atomic
snapshot of the package database, and I do not yet identify selected library
bytes or all toolchain inputs across supported platforms.

On Darwin, my shared link now supplies tagged dependency records. I hash its
regular external inputs, record missing search candidates, and check both
before warm reuse. The same-size/time archive edit and newly earlier library
regressions now produce new generations. If capture is unsupported, I discard
its record and retry ordinary linking without reuse evidence. Invalid or
incomplete records likewise withhold reuse. Link commands containing `@` remain
uncacheable pending indirect response-file capture. Cacheable cold builds use
a private discovery link and an identical final link; their input observations
must agree. I retain and revalidate those hashes before recording reuse, rather
than labeling linked code with later hashes. A failed final link preserves the
previous generation. Warm reuse skips both links. These checks do not pin
runtime dynamic-library bytes or snapshot files during linking; other linker
formats and full toolchain identity remain open. My
[linker evidence](LINKER_INPUT_EVIDENCE.md) records the tested boundary.

### Cross-section validation

A section codec sees one section and cannot check an index into another, so the
whole-module reader checks what none of them can: every signature, constant and
layout index resolves; every function's `[code_offset, code_offset+code_length)`
lies inside CODE, bounded by subtraction so a range near `2^64` cannot wrap back
into it; the entry point names a real function or the sentinel; and the header's
feature bits cover what the sections require.

## Execution Model

### Execution Representations

I keep three separate representations of a program so that serialization,
verification, and dispatch each stay simple and independently testable. Only the
first is the **portable NanoISA** contract; the other two are internal runtime
representations rebuilt on every load and carry no portability promise. See
[Portable NanoISA vs. Runtime Representations](NANOISA_PORTABLE_ISA.md) for the
full separation:

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

I use reference-counted GC:
- `OP_GC_RETAIN` / `OP_GC_RELEASE` - Manual reference counting

Scope lifetime is tracked implicitly by the call stack, so no dedicated
scope-marker opcodes are required.

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
