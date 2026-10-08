# My native shadow acceptance through NanoISA

I replace the old `transpiler.nano` import in my shadow-emitter fixture with
program and selected-shadow NanoISA emission. The test assembles and verifies
each emitted module, executes it in NanoVM, translates it through `nvm2c`,
compiles strict C11, and checks native behavior. Both the C-seed and installed
self-hosted producers build the emitter fixture.

I retain failing assertions under `NDEBUG`, callable main, a main that is not
implicitly a test, program/shadow separation, local scope and execution order,
explicit selection, rejected selection preserving output, opaque null arguments,
and record-string lifetime. Raw modules retain their ordinary output stream;
I test compiler-driver supervision separately. My quiet C-seed interpreter
suppresses passing shadow output, as implemented in `src/eval.c`; the self-hosted
driver routes shadow output to stderr. Both keep compiler stdout separate from
product output and preserve an existing binary after a failing shadow.

The first migrated run has three failures: the new harness expects successful
C-seed shadow text despite that quiet policy, and both emitter producers refuse
the retained opaque-null fixture. I retain the terminal. The C-seed bytecode
path already accepts that fixture; its dumped parameter metadata says `opaque`.

I recognize explicitly declared opaque scalar types in self-hosted lowering,
publish opaque parameter tags, and admit literal zero or a value of the exact
declared opaque type at that boundary. Nonzero integer, Boolean, and differently
named opaque arguments refuse emission without replacing output. I check direct
and forwarded null arguments in VM and native execution. Module-aware opaque
nominal identity and non-null foreign-handle contracts remain roadmap work.

The full migrated suite passes 12 methods in 117.071 seconds. After adding
forwarded-null and failed-driver-publication controls, the two affected methods
pass in 110.051 seconds. Commands are `python3 -m unittest -v
 tests.test_native_shadow_emitter` and its two affected method selectors. My
Make target now requires the assembler, VM, native translator and host runtime.
The broader `make -j2 test-nanoisa-src-nano` gate is running; its terminal result,
compiler-product qualification and remaining legacy-emitter migrations stay open.
