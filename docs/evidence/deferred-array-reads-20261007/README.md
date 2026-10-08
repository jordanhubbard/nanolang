# My deferred array-read checkpoint

I translate the full C-seed compiler module into native C and pass the original
`test_compiler_bytecode_to_native_to_program` acceptance method. The generated
compiler runs its help command, builds and executes hello, emits hello NanoISA,
and produces matching VM and native output. This is compiler-product evidence;
I have not completed the self-hosted fixed point or the 5.1 release.

## My defect and correction

At `d11ae94e2`, an unresolved array read equates its result shape with its stored
element shape. A later tagged consumer makes the stored element optional.
Returning the containing record into exact scalar-array storage then refuses a
valid program. My retained trace identifies this path in `nb_owner_field_tag`
and `nb_expr`; the baseline log and eight small assembly modules reproduce the
refusal independently of the compiler source.

I defer that read relationship until shape solving knows the element kind.
Scalar reads produce optional values whose payload retains the exact element
kind. Record reads use directed field conversion; a record consumer still
requires record element storage without equating the consumer's inferred fields
with the stored fields. The graph releases its pending read relations and
repeated solving does not allocate duplicate wrappers or conversions.

My first implementation loses the backward record-container requirement. The
retained initial adjacency run records two failures in the existing projected
array-read test, and the compiler stops at `parser_get_tuple_index`'s record
return. I restore that requirement and retain the corrected full adjacency run.
I do not relax exact scalar constraints.

The first successful compiler translation then reaches a faulty test assertion:
it rejects the literal `"bin/nano_vm"`, used for separately emitted shadow tests.
I check executable identifiers, admit literals/comments, and retain negative
controls for VM calls and bytecode blobs. The corrected original product method
passes all its compilation and execution assertions. The shortened first-failure
record includes the original verbose log's hash; it omits the assertion's dump
of the entire generated compiler.

## My checked boundary

- All eight present/missing int, bool, float and string regressions fail with the
  pre-change classifier and pass with the correction in NanoVM and native code.
- My optional-read suite plus existing projected-array consumer method passes
  eight methods with Homebrew Clang 23 ASan, UBSan and leak detection enabled.
- My unchanged compiler-adjacent selection passes 78 methods in 113.871 seconds.
  It excludes exactly the two full compiler-product methods, which I run
  separately. The new no-VM assertion control also passes separately.
- `make -j2 -o nvm2c test-nvm2c` passes 2,431 structured-C checks and its
  prerequisites, including 1,448 shape checks. `-o nvm2c` retains the just-built
  translator during concurrent independent tests; it omits no test.
- The original C-seed native compiler-product method passes in 48.232 seconds.
  Its native compiler build uses the method's unchanged strict C11 flags and
  `-O0`; I do not describe that product method as a sanitizer run.

I select `/opt/homebrew/opt/llvm/bin/clang` through the temporary `cc` launcher
recorded in the earlier [Darwin checkpoint](../native-release-20261007/README.md).
The compiler module used for focused translation is that checkpoint's retained
`compiler-seed.nvm`. The original product method independently emits a fresh
module before translation. The first shape-only command used a nonexistent
singular Make target; the corrected plural target and owning core gate pass.

MAC task discovery and both filing attempts return `[Errno 1] Operation not
permitted`. I retain product work in my roadmap and do not claim external task
filing succeeded. GitHub API access is intermittent; Git fetch succeeds.
