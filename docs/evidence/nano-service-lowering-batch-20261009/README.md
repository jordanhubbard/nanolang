# My independent Nano File lowering batch

I continue [#989](https://github.com/jordanhubbard/nanolang/issues/989) and
[#982](https://github.com/jordanhubbard/nanolang/issues/982) from `22cd143af`.
I retain the complete [5.1 scope](../../RELEASE_5.1_SCOPE.md). I have not yet
enabled service source publication in my actual compiler drivers.

I implement `src_nano/compiler/service_lowering.nano` against my Nano parser,
nominal facts, and ownership checks. I retain every function and shadow,
original helper and service identities, scoped locals, ownership declarations,
exclusive references, Result payloads, scalar fields, short-circuit control,
loops, and terminal operand cleanup. I compute operand-stack peaks while
emitting instructions. My consumers separately verify the resulting module.

I implement `src_nano/compiler/service_wire.nano` in Nano: little-endian fields,
string interning, signature interning, all eight catalog layouts, five imports,
function/code tables, ownership and service sections, section alignment, and
CRC32. My only catalog dependency returns immutable descriptive data. Neither
new module delegates AST lowering, assembly, validation, or serialization to C.
My test transports printed byte values to the external consumers; that is not
a compiler binary-output publication path.

## Native compilation and retained checks

I partition cyclic native agreement checks into one helper per instruction.
I keep startup, bytes, layouts, imports, functions, locals, instructions,
successors, components, every variant, values, references and regions checked.
I remove no comparison. The original monolithic generated function exceeded
120 seconds at LLVM `-O1`; my complete generated binding now compiles at
`-O2` in roughly 8–9 seconds per selected shadow. The expanded eight-method
[C source corpus](optimized-source-final.log) passes in 53.640 seconds with
ASan/UBSan native programs. My first seven-method measurement remains in
[its original log](optimized-source-initial.log). I also run the full eight
methods at the actual driver default, `-O1`: the
[production-setting gate](production-opt-source.log) passes in 50.572 seconds.

My [sanitized cyclic dispatch gate](dispatch-corrected.log) passes both methods
in 127.673 seconds. It compares VM/native traces at `-O0` and `-O2` and retains
isolated ABI, later-variant, reference and region corruption refusals. My
[first dispatch attempt](dispatch-first.log) failed compiling `vm_ffi.c`:
the sanitizer recipe omitted the configured compiler/include flags and could
not find Darwin `ffi.h`. I correct the recipe rather than omit that provider.
Listed query/runtime providers and generated C are instrumented; common
objects retain their ordinary build flags.

## Independent serialization and source gates

My [wire gate](wire-qualified.log) passes two methods in 3.811 seconds;
my earlier ASCII-only run remains in [its original log](wire-gate.log). A C-produced and a
self-hosted-produced Nano writer execute in VM and sanitized native C and
produce byte-identical File modules. Python independently checks CRC32,
container lengths and a UTF-8 function name. The real granted VM and native File consumers execute
`temp`, drop its owned result, and return 42. Ungranted and damaged modules
refuse; native emission preserves prior output. The native wrapper also checks
null-grant refusal, cleanup failures and open-descriptor balance.

My [eight-method source corpus](nano-source-qualified.log) passes in 197.418
seconds. It uses both producers to compile the Nano lowerer, then runs
each resulting lowerer in VM and sanitized native execution. It compares all
four outputs byte for byte before executing the File module through both real
consumers. I use the unchanged five generated shadow bodies, helper loops,
Result values and fields, qualified re-exports, short-circuit effects, terminal
arguments, nested argument borrows, assertion cleanup, and explicit limits.
The fixture supplies dependency order while retaining real files and source
line origins; actual driver import discovery remains separate integration work.

## Failures and corrections

- My [first native lowerer translation](nano-lowering-first.log) exposes an
  exact-storage constraint failure. The [trace](native-shape-trace.log) maps
  it to `sl_children` and the `parser_get_call_arg` return. A record-array write
  already has a directed, checked write constraint; the additional equality
  incorrectly constrains an unresolved getter's return to caller constructor
  storage. I remove that redundant equality for record writes while retaining
  payload checks and emitted tag guards. The full Nano source test is a
  regression for this producer/getter path.
- My [record-array neighbor gate](native-shape-neighbors.log) passes tagged
  array fields, wrong runtime tags, incompatible concrete fields, mutable
  aliases, and record globals. Two scalar methods initially select Apple's
  `cc` through older literal-compiler helpers and cannot start LeakSanitizer.
  Both pass with the LLVM `cc` shim in the
  [corrected scalar gate](native-shape-scalar-corrected.log). I do not count
  the first failures as passes. All nine methods have passing results.
  My [shape unit gate](native-shape-unit.log) passes 3,098 checks.
- My first self-hosted lowerer build reports
  [an undeclared transitive `tokenize_string` call](nano-lowering-import-failure.log).
  I add explicit direct imports for the parser/checker functions I use.
- My [first complete paired source run](nano-lowering-branch-failure.log)
  passes seven methods and rejects the new nested-borrow fixture. A saved
  array was restored by reference and then changed by subsequent appends,
  leaking ended loans into another Result error arm. I copy saved entries on
  every restoration. The focused corrected VM consumer returns 7, and the final complete paired
  source gate passes all eight methods.
- I retain initial authoring errors separately: missing array commas in
  [the writer](wire-first-parse.log) and a reserved local name in
  [the lowerer](lowering-first-parse.log). These do not establish product bugs.

## Reproduction and remaining work

```sh
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
NANO_SERVICE_WIRE_DRIVER_MODULE=/private/tmp/nanolang-service-lowering-driver-final.nvm \
  python3 -m unittest -v tests.test_service_wire

NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
NANO_SERVICE_LOWERING_DRIVER_MODULE=/private/tmp/nanolang-service-lowering-driver-final.nvm \
  python3 -m unittest -v tests.test_service_lowering_nano

make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  test-file-cyclic-dispatch-sanitize test-nvm2c-shapes
```

My development driver is C-produced at the preceding batch; it is not a fresh
release Stage 1/Stage 2 fixed point. The ordinary lowerer executables and emitted
native File programs use LLVM sanitizers; the linked File archive is an ordinary
build. I do not claim a fully instrumented installed product from these checks.

Actual-driver explicit grants, supervised selected shadows, binary output and
staged publication remain open. So do complete source/profile coverage,
multiple nominal catalogs, indirect/richer-reference calls, fresh exact-pin
bootstrap, Linux/Darwin installed qualification, Socket/public network work,
and the rest of the full 5.1 release contract. I keep those requirements open
rather than equate this API batch with a release.
