# Self-hosted functional callback arrays

I continue #976 from e37c41cca. My retained fresh-product baseline rejects all
sixteen callable-array operation fixtures before publication. The checker
cannot infer generic map/reduce signatures; the emitter requires named scalar
callbacks. I retain each producer terminal in probe.log and results.json.

I infer concrete callback signatures from array elements and the reduce
initializer, including independent variables and a generic callable initializer.
I check map, filter, reduce and array mutations before reachability-based
lowering. I specialize generic array replacements and appends from their
complete element context. Inline generic callback arrays can take their
context from a concrete transform parameter. My declared map path follows
the self-hosted builtin-name contract; the C seed separately reserves that
name, so that one source is qualified through self-hosted routes only.

I capture the callback once, after the source and reduce initializer and before
capturing the array length. I invoke the saved function value inside the loop.
Filter retains its selected element before the predicate can mutate its source.
I store callable map results and reduce accumulators with their complete types.

My global-array audit discovers missing return representations. I retain an
exact array storage hint only when every global store is concrete and agrees.
I use that hint when unboxing an array return; I keep the original tagged
carrier unconstrained for unrelated consumers and retain runtime tag, storage
and presence checks. String-array returns use the same checked unboxing.
My first prototype constrained all global consumers and broke five existing
runtime-check assertions; full-nvm2c.log retains that failed prototype.

My final eleven-method suite passes in 52.413 seconds (byte-final-paired.log).
I cover the sixteen retained functional fixtures, inline generic callbacks,
callback capture and operand order, contextual mutations, ten invalid
unused-body cases, and integer/string/bool/float/callable global-array results.
I also execute named, computed, inline and empty callable loops, retained
bindings after mutation, early return, continue and break. I preserve enum/int
storage compatibility and integer-to-byte narrowing without loosening callable
signature identity.

I execute C frontend products and fresh self-hosted whole/program/shadow
emission through verification, NanoVM and strict C11 generated-program
ASan/UBSan/LSan. My fresh checker rejects each invalid body with exact status 1.
I instrument all three translator objects with ASan/UBSan; translator leak
detection is disabled, while generated-program leak detection remains enabled.
My existing eight scalar functional-array methods pass in 99.747 seconds with
a fresh shadow emitter and the installed source CLI (scalar-regression.log).

I retain initial harness errors: a missing space produced a >= token, and an
inline generic callback array initially misclassified as invalid is now a
positive test. I preserve the initial full-product audit separately: its eleven
stages pass in 526.604 seconds, but source inputs changed during independent
corrections, so its overall audit exits 1. It is not a fixed-source qualification.

My corrected full translator gate passes 2,443 execution, 419 callable and
3,098 shape checks (full-nvm2c-corrected.log). My eight callable-union regression
methods pass with the final checker/emitter and instrumented translator in
17.364 seconds (union-byte-corrected.log).

My adjacent for-loop audit exposed a pre-existing checker gap: a callable loop
binding was treated as int when returned. The previous native compiler and an
intermediate seed rejected valid source while preserving prior output
(loop-product-baseline.json). I now reconstruct the complete declared element
type. The final eleven-method suite executes all three retained loop fixtures
and rejects incompatible calls in unused loops.

My first fixed-source audit stopped during self-compilation after 381.709
seconds (mutation-rejection-product). Its nine parser refusals exposed existing
scalar storage conversions: ParseNodeType enum values are accepted in
array<int>. After restoring those conversions, my focused checker run exposed
u8 being treated as an undeclared record (storage-paired-sanitized.log and
scalar-storage-probe.json). I now recognize that primitive in complete type
checks, and my final focused suite passes both enum and byte mutations.

My final fixed-source compiler-product audit passes all eleven stages in
528.043 seconds (final-product). Its manifest confirms unchanged compiler
inputs. I build the seed, use the canonical self-hosted emitter to compile the
compiler, verify its module, translate and link it, then execute the unchanged
compiler-product acceptance controls under the existing shadow deadline.

The resulting native compiler passes all eleven expanded functional methods
(native-product-functional.log) and seven adjacent scalar source-CLI methods
(native-product-scalar.log). My earlier eight-method scalar run also covers the
bound callback-owner case with a freshly built shadow driver.

I do not claim raw bootstrap equality, exact Linux/Darwin candidate
qualification, or completion of the full 5.1 scope.

I trim trailing whitespace in retained text logs; diagnostic content and terminal
results are unchanged. My source hashes identify the qualified code and tests.
