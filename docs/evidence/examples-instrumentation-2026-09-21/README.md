# My instrumented examples investigation

I investigated `make examples` at `e0c7eb76d415520f7298b9d20a22cba2f4825b00`
on Darwin arm64, macOS 26.7 (25G229), with Apple clang 21. My installed SDL2
is Homebrew `sdl2-compat` 2.32.72 backed by SDL3 3.4.16. I record tool versions
and compiler hashes in [environment.json](environment.json).

## What I observed

1. My original `make examples` exits 2 while compiling `sdl_boids`. Its shadow
   child terminates with signal 6. My native crash report places the abort in
   Apple's Metal compiler, reached through `SDL_CreateRenderer`, `vm_ffi_call`,
   and my callback-aware shadow runner. An earlier report from the same day
   has the same faulting stack. I preserve both extracted reports in
   [native-crash-stacks.json](native-crash-stacks.json).
2. With `NANO_VM_TRACE=1`, `NANO_TRACE_BUILD=1`, and `--verbose`, I again stop
   after signal 6. My trace reaches `run_boids`, shows a successful `SDL_Init`
   and non-null `SDL_CreateWindow`, then ends at the foreign renderer call.
   This is a callback-aware NanoVM shadow inside my C compiler, before output
   publication; it is not execution of the final generated example.
3. My separately built ASan/UBSan compiler initially stops in SDL2-compat's
   modal loader error. LLDB reads its message as `Failed loading SDL3 library.`
   My process sample places it in `dlopen` → `dllinit` → `error_dialog` →
   `NSAlert runModal`. Explicit `DYLD_LIBRARY_PATH=/opt/homebrew/lib` passed
   directly to the compiler resolves this diagnostic setup obstacle. An
   earlier attempt through system Perl did not resolve it; I retain that
   terminal too.
4. After correcting library lookup, ASan reports a **BUS** read at PC
   `0xbad4007`, through SDL3 `ScheduleContextUpdates`, `SDL3View updateLayer`,
   AppKit, and `SDL_CreateWindow`. The compiler reports shadow signal 6 after
   ASan aborts. I reproduce this in the wider instrumented Make sweep for
   `sdl_boids` and `sdl_particles`. These reports do not establish a heap
   overflow or use-after-free in my implementation.
5. My small C SDL control passes both directly and after `fork`. A later
   unmodified `sdl_boids` compile passes with `SDL_RENDER_DRIVER=opengl`, and
   then also passes with the default renderer. I therefore do not claim that
   switching renderers fixes the original failure, that `fork` alone causes
   it, or that later success explains the earlier Metal abort.

I have not established a SIGSEGV in these runs. I have established a native
SIGABRT and sanitizer-detected BUS failures on the graphical shadow path.
The source of the invalid AppKit/SDL control transfer remains unresolved.

My instrumented continuation finishes with Make exit 2 and exactly three
failed targets: `sdl_example_launcher`, `sdl_boids`, and `sdl_particles`.
It records 78 successful native compilations and four successful GPU source
emissions; the earlier interrupted sweep records 96 successful native
compilations. These are log counts, not a claim that every target was freshly
rebuilt or that generated GPU kernels were executed. Neither sweep reports a
UBSan `runtime error`.

My final ordinary **incremental** `make examples COMPILER_FLAGS=--verbose`
passes with exit 0. It recompiles `sdl_boids`, `sdl_forth_ide`, and
`sdl_particles`; other outputs are already current. It does not independently
rerun the timed-out launcher's shadows, and it does not clear my sanitizer
failures. I preserve `make-ordinary-after.log` and its exit status.

## My instrumentation

I build my compiler separately, without replacing `bin/nanoc_c` or its objects:

```sh
make -j4 \
  OBJ_DIR=/tmp/nanolang-examples-investigation/obj-san \
  BIN_DIR=/tmp/nanolang-examples-investigation/bin-san \
  'CFLAGS=-Wall -Wextra -Werror -std=c99 -g -O1 -fPIC -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer' \
  'LDFLAGS=-lm -fsanitize=address,undefined' \
  /tmp/nanolang-examples-investigation/bin-san/nanoc_c
cp /tmp/nanolang-examples-investigation/bin-san/nanoc_c bin/nanoc_c_examples_san
```

I place the diagnostic compiler under `bin/` because my compiler locates its
runtime headers relative to its executable. My first attempt to run it from
`/tmp` failed during generated module compilation; `boids-san.log` preserves
that setup failure.

For the failing example I enable my existing opcode and module-build hooks:

```sh
NANO_VM_TRACE=1 NANO_TRACE_BUILD=1 \
  ./bin/nanoc_c examples/graphics/sdl_boids.nano --verbose \
  --llm-shadow-json /tmp/nanolang-examples-investigation/boids-shadow.json \
  -o /tmp/nanolang-examples-investigation/boids
```

My corrected sweep uses the retained [compiler wrapper](compiler-wrapper),
which supplies SDL3 lookup directly and imposes a 120-second compiler deadline:

```sh
NANO_TRACE_BUILD=1 ASAN_OPTIONS=halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  make -k examples \
  EXAMPLES_COMPILER_PATH=/tmp/nanolang-examples-investigation/compiler-wrapper \
  'COMPILER_FLAGS=--verbose'
```

I keep the normal ten-second shadow deadline. My `sdl_example_launcher`
reaches it under instrumentation; I record this separately from the graphics
crashes. I do not weaken its assertions or count it as a pass.

My sanitizer coverage applies to my compiler, embedded interpreter/VM, and
linked compiler runtime. It does **not** establish sanitizer coverage of
Homebrew SDL, Apple frameworks, every dynamically built module, or the final
generated example executables. I do not claim LeakSanitizer qualification.

## My evidence and follow-up

I retain raw logs compressed with hashes in [log-manifest.json](log-manifest.json),
Make outcomes in [sweep-results.json](sweep-results.json), and my small SDL
control's source and results alongside this report. My first broad sweep was
interrupted deliberately to correct SDL3 lookup and bound each compiler call;
its partial successes are not a complete suite pass. I retain the completed
continuation separately.

My investigation is `task_988bd7b1e22445a1b0635d4b983b6c63`. My unresolved
Darwin graphics qualification/repair is
`task_3b0b630c45884d88beef74d9eb77c3b9`, also recorded in my roadmap. I leave
production source and shadow assertions unchanged. The next investigation
must distinguish graphics initialization and process state from incorrect FFI
or VM behavior, while preserving these first terminals.
