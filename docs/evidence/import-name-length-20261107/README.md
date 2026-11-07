# My import-name length repair

I reproduce #1055's assembler diagnostic at a precise boundary: an otherwise
identical import module name assembles at 255 bytes and refuses at 256 bytes.
Both module and symbol names previously used fixed 256-byte buffers. My old
assembler from `4bf010dcc` reproduces that refusal and preserves prior output.

I allocate both decoded names within the current encoded line's length and
retain the existing u32 pool bound. I free temporary storage on every parse
and publication refusal, check string-pool and import-table allocation, and
leave the import representation and runtime path resolution unchanged.

My local Darwin arm64 checks pass:

- `PATH="/opt/homebrew/opt/python@3.13/libexec/bin:$PATH" make -j1 test-nanoisa test-disasm-roundtrip`:
  schema controls, all 3,018 NanoISA assertions and 440 round-trip assertions.
  I cover 255, 256, 400 and 4,096-byte module names, symbol names and both
  together, comparing full string bytes and parameter tags after reassembly.
- `make 'CC=cc -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' test-assembler-string-alloc`:
  actual assembler source and failure controls run with ASan/UBSan. I exercise
  name allocation, pool and import-table refusals, malformed declarations,
  cleanup and recovery without leaked temporary buffers.
- My repaired CLI still refuses malformed import input and preserves an
  existing output byte-for-byte.
- `python3 scripts/fast_gate.py --jobs 4`: 223 strict Clang and 223 strict GCC
  compilations pass.

I add `make test-long-path-bootstrap` as the complete regression. It creates
an isolated detached Git worktree under `~/.worktrees/nanolang/` with an actual
absolute source path of at least 300 characters. Sparse checkout omits only
historical `docs/evidence/`; all original native inputs remain in place.
I pin object, binary and module-cache paths inside that new worktree and do
not copy old outputs or bootstrap receipts. I retain commands, logs, source
revision and the actual Stage 1 manifest outside the source directory.

The complete long-path Stage 1 run is not yet qualified at this checkpoint.
Focused import success does not establish the full bootstrap acceptance.
