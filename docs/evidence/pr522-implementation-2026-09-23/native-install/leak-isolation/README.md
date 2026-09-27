# My Stage 1 leak isolation

I use the instrumented Stage 1 binary retained from the failed installed-package
bootstrap at `601500e51`; `inputs.json` identifies its SHA-256. I keep
`ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1` and
`UBSAN_OPTIONS=halt_on_error=1` throughout. The compiler route probes compile
`examples/language/nl_hello.nano`:

| Route | Exit | LeakSanitizer result |
| --- | --- | --- |
| `--help` | 0 | No reported leak |
| `--emit-nvm` | 1 | 17,163 bytes in 93 allocations |
| `--target c` | 1 | 82,699 bytes in 94 allocations |
| Native `-o` | 1 | 150,285 bytes in 97 allocations |

I retain an initial invalid `--emit-c` invocation separately in `c.log`; it
never compiled anything. `c-corrected.log` uses the actual `--target c` option.
The C/native difference includes one additional 65,536-byte command capture
and two 1,025-byte physical-path results. The bytecode route still exposes
parser lists, a visited-source hashmap and canonical-path storage.

My standalone `.nano` probes are compiled by `bin/nanoc_c` with
`NANO_CC=/opt/homebrew/opt/llvm/bin/clang`,
`NANO_CFLAGS=-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all`
and `NANO_LDFLAGS=-fsanitize=address,undefined`. Their main functions assert
observable capture/map values; their shadows do not establish ownership.

All three initially exit zero under ordinary leak roots. That does **not**
establish cleanup: stale stack/register values can keep unreachable program
allocations visible to the conservative root scan. I rerun with
`LSAN_OPTIONS=use_stacks=0:use_registers=0` and retain both sets of logs.
The capture probe then leaks 131,072 bytes in two allocations; the hashmap
probe leaks 420 bytes in three allocations. Its explicit `map_free` control
still exits zero. These stricter diagnostic roots supplement the original
unchanged bootstrap failure; they do not replace its acceptance gate.

## Owning code boundaries

- `src/transpiler.c` emits `nl_exec_capture` with an unowned 65,536-byte
  malloc result. I must preserve earlier returned strings across later calls.
- `modules/std/fs.c:path_canonical` returns owned realpath/strdup storage and
  supplies a string-release companion. The C-seed direct call path has not
  consumed that companion in this failing execution.
- Generated hashmaps and `src/runtime/list_AST*.c` allocate with malloc.
  C-seed scope cleanup uses `gc_release` for hashmaps, but that function cannot
  release unregistered malloc storage. Parser records also contain lists that
  need recursive ownership handling.
- C-seed scope tracking explicitly omits strings. Replacing malloc with a
  globally rooted managed allocation alone would hide a leak report without
  proving cleanup. I require an actual lifetime and release path, including
  returned values and nested records, before claiming this gate repaired.

The parent task remains `task_02204077a4d34d4aa5ce20ef7054e113`. No implementation
repair or passing instrumented bootstrap is claimed by this isolation report.
