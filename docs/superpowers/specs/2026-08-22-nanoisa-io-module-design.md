# NanoISA I/O module — design

I am adding a single facade for reading, writing, printing, and
pretty-printing NanoISA modules. The codec stays where it is. Callers
stop talking to it directly.

Date: 2026-08-22
Status: approved in conversation; awaiting spec review before the
implementation plan

## Goal

I expose one in-memory `NvmModule` and two on-disk languages:

- `.nvm` — binary module (header, sections, CRC)
- `.nasm` — labeled text assembly

Round-trip: assemble text → save bytes → load bytes → print text that
the assembler accepts again.

I do not reimplement serialize, assemble, or disassemble in NanoLang.
`modules/nanoisa/` wraps `src/nanoisa/`.

## Non-goals

- Wrapping `vm_init`, the verifier, or codegen
- Dual NanoLang reimplementation of the codec
- ANSI color in pretty-print
- Changing the `.nvm` or `.nasm` formats

## Layout

```
modules/nanoisa/
  nanoisa.h              # public C facade
  nanoisa.c              # file/bytes I/O, pretty-print, error fill
  dump_main.c            # `nanoisa dump [--pretty] <file.nvm>`
  nanoisa.nano           # externs, thin wrappers, shadow tests
  module.manifest.json
```

Codec (unchanged ownership):

- `src/nanoisa/nvm_format.c` — `nvm_serialize` / `nvm_deserialize`
- `src/nanoisa/assembler.c` — `asm_assemble` / `asm_assemble_file`
- `src/nanoisa/disassembler.c` — `disasm_module` / `disasm_function`

Only `modules/nanoisa/nanoisa.c` and those codec files call `nvm_*`,
`asm_*`, and `disasm_*` for I/O. Tests that assert codec internals may
still call the codec; tests of the public contract use `nanoisa_*`.

## C API

Prefix: `nanoisa_`. The loaded object is still `NvmModule *` so
`vm_init(vm, mod)` does not change. Free with `nvm_module_free`.

```c
typedef struct {
    int code;           /* 0 = ok; nonzero = failure class */
    uint32_t line;      /* 1-based assemble line, else 0 */
    char message[256];
} NanoisaErr;

NvmModule *nanoisa_load_file(const char *path, NanoisaErr *err);
NvmModule *nanoisa_load_bytes(const uint8_t *data, uint32_t size,
                              NanoisaErr *err);
int        nanoisa_save_file(const NvmModule *mod, const char *path,
                             NanoisaErr *err);
uint8_t   *nanoisa_save_bytes(const NvmModule *mod, uint32_t *out_size,
                              NanoisaErr *err);

NvmModule *nanoisa_assemble_text(const char *source, NanoisaErr *err);
NvmModule *nanoisa_assemble_file(const char *path, NanoisaErr *err);

char *nanoisa_print(const NvmModule *mod);
char *nanoisa_pretty_print(const NvmModule *mod);
```

Failure: pointer APIs return `NULL`; `nanoisa_save_file` returns
nonzero. If `err` is non-NULL, I fill `code` and `message`. Assemble
maps `AsmResult` into `NanoisaErr` (including `line`). Load failures
that `nvm_deserialize` currently reports only as `NULL` get a concrete
message in the facade (short read, bad magic, bad CRC, I/O).

`nanoisa_print` and `nanoisa_pretty_print` return a heap string the
caller frees. They return `NULL` only on allocation/stream failure.

`nanoisa_save_bytes` returns a heap buffer the caller frees (same
contract as `nvm_serialize` today).

### Print vs pretty-print

- **Print** = `disasm_module_styled(..., DISASM_STYLE_CANONICAL)`:
  `.string`, `.flag needs_extern` / `.flag debug_info` when set,
  `.entry`, `.function` / body / `.end`, reconstructed labels
  (`L0`, …), mnemonics without `[offset|abs]` prefixes or debug/cfg
  comments. The assembler accepts this text again; dump → assemble →
  dump is a text fixed point.
- **Pretty-print** = a preamble, then the detailed listing with a
  bytecode offset on each instruction. Preamble includes: magic, format
  version, flags, entry point, section table (type, offset, size),
  function table (name, arity, locals, upvalues, code offset/length),
  import table, debug-entry count.

## NanoLang API

I do not expose `NvmModule` pointers. File and string operations:

```nano
fn load_print(path: string) -> string
fn load_pretty(path: string) -> string
fn assemble_save(nasm_path: string, nvm_path: string) -> int
fn last_error() -> string
```

`load_*` return `""` on failure. C stores the last `NanoisaErr.message`
in a process-local buffer. NanoLang reads it with
`last_error() -> string` so shadows can assert on both content and
failure. `assemble_save` returns `0` on success, nonzero on failure,
and also sets that last-error buffer.

Every `fn` has a `shadow` block. Fixtures are tiny assembled snippets
written to temp paths or assembled via externs used only in shadows.

## Consumers

| Caller | Change |
|--------|--------|
| `src/nanovm/main.c` | `nanoisa_load_file` instead of local read + `nvm_deserialize` |
| `src/nanovm/vmd_server.c` | `nanoisa_load_bytes` |
| `src/nanovm/cop_main.c` | `nanoisa_load_bytes` |
| `src/nanovirt/main.c` | `nanoisa_save_file` / `nanoisa_save_bytes` instead of `nvm_serialize` + write |
| `src/nanovirt/wrapper_gen.c` | `nanoisa_save_bytes` for the embedded blob |
| `tests/nanoisa/test_nanoisa.c` | public round-trip/print tests call `nanoisa_*` |
| `modules/forth_see/forth_see.c` | `nanoisa_load_file`; drop private NVM parser and opcode tables; keep Forth word-slice search; listing via `disasm_function` / `nanoisa_print` |

Forth-specific logic (find `exec_builtin`, map a word to a code slice,
“interpreted — no compiled body”) stays in `forth_see`.

## Build

Existing VM/compiler binaries already link `src/nanoisa/*.o`. Add
`modules/nanoisa/nanoisa.c`.

`modules/forth_see/.build/libforth_see.so` is currently only
`forth_see.c`. After this, it links NVM format, ISA, assembler,
disassembler, and `nanoisa.c` (or compiles those sources into the
`.so`). `examples/Makefile` `test-forth` must still run.

Compile with `-Wall -Wextra -Werror`.

## Testing

- Shadow tests on the three NanoLang functions (happy path + missing
  file / bad magic for load).
- C: assemble → `save_bytes` → `load_bytes` → `print` contains
  expected mnemonics; pretty-print contains section or offset lines;
  corrupt header and bad CRC return `NULL` with a message.
- `make test-nanoisa` passes.
- `make test-nanoisa-dump` covers canonical dump, `--pretty`, missing
  file, and bad magic.
- `make test-forth` (or the documented Forth smoke target) still
  decompiles a known builtin.

## Success

- Agreed consumers do not call `nvm_deserialize`, `nvm_serialize`,
  `asm_assemble`, or `disasm_module` except from `nanoisa.c` and the
  codec.
- `forth_see.c` has no mirrored NVM magic/section/opcode tables.
- Formats are unchanged; `vm_init` still takes `NvmModule *`.

## CLI

`bin/nanoisa dump [--pretty] <file.nvm>` loads through the facade and
prints canonical assembly or the detailed listing.
