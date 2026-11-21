# My shared ownership declarations and checked slices

I implement this #1142 boundary in `7cd173b30` on
`feat/5.1-nanolang-translator`, draft PR #1144. My NanoLang
`ownership.validate` matches the shared C
`nvm_ownership_contracts_validate` declaration boundary for versions 1, 2 and 3.
I preserve retained-layout authority, resource and reference descriptors,
signature/local relationships, bounded paths, extension framing, unions,
globals, effects/element types and actual constructor instruction sites. I
publish `requires_verifier` only after complete success; every refusal clears it.

This boundary excludes private ARRAY_FIELDS/MAP_FIELDS projections. I retain
their existing shared-validator refusals and error order. Instruction lifetimes,
private declarations, whole-module co-profile conversion and executable
verification remain required. Declaration success grants no execution authority.

## My paired checks

| Check | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Shared declaration status/authority | 2350 cases on five routes | 2350 cases on five routes |
| Explicit C contract expectations | 1415 pass | 1415 pass |
| Nested byte-slice copy and parse-valid semantic refusals | Pass | Pass |
| Original slicing suite, all assertions retained | Four methods pass | Four methods pass |
| Original C checker/transpiler/evaluator/metadata/scoping gates | Pass | Pass |
| Earlier decoder/scope/passive/call regressions with fresh source compiler | Eleven methods pass | Eleven methods pass |
| Schema gate with system Python | 33 methods pass | 33 methods pass |
| Embedded identifiers | Five methods pass | Five methods pass |

The declaration corpus retains all three versions; every flag, field/local/
mode/global tag; resource and disconnected-layout edge order; descriptor and
signature errors; exact 256-path/32-depth and extension limits; union/global/
effect limits; and the actual 32768/32769 constructor boundary with otherwise
valid sites. Binary and embedded-zero string identities, duplicate effect
keys, typed-array references, exact instruction-prefix decoding, truncations
and mutations remain explicit. I preserve refusal status precedence and clear
authority flags on every error. Both hosts retain byte-identical corpus reports
and identical tested source hashes.

The embedded gate reports 3287 static fragments, zero reserved declarations and
135 existing lexer-refused fragments. Executable gates retain responsibility
for parsing and runtime behavior. My added and changed NanoLang helpers have
useful shadows, exercised during compiler and fixture builds.

The eleven-method regression retains 5489 core decoder cases per route, six
unchanged example products, 256 widths, 256 File scans and both global-scope
fixtures. It also matches 2240 complete passive-module decisions and exact claim
bytes with 472 explicit expectations, 2183 passive-record decisions with 414
explicit expectations, four unchanged par/flow products, and 1498 closed-call
cases plus 512 opcode/profile decisions on all five routes. These checks retain
the earlier components' authority boundaries and unchanged budgets. Four
whole-module profiles remain pending.

## My slice prerequisites

My unchanged ownership driver pushes `array_slice` directly into a byte-array
row. The C seed accepts it, but my preceding full NanoLang compiler refuses
publication because its checker loses the slice's complete receiver type. I
retain that failure and the minimal unchanged `array_slice_nested.nano` witness.
My NanoLang checker now checks exact arity, array receivers and integer bounds,
and carries the receiver's complete type into existing array mutation checks.

Parse-valid negative controls then expose three C-producer publications:
scalar receivers, boolean bounds and an integer child slice in byte-array
storage. Missing arity already refuses. I retain those results, share the C
receiver/bounds check and carry receiver TypeInfo into complete storage checks.
The final controls require refusal before publication and preserve exact prior
output bytes. They reject parse-error masking. I add no cast or weakened array
mutation check.

My original slicing suite retains all four methods and product assertions,
including copy/identity, evaluation order, numeric/string/nested/record and null
boundaries. Its default tools remain actual installed Stage1/Stage2 paths.
Optional selectors name the real development compiler paths for this checkpoint
and reject empty selections. These runs do not establish stage qualification.

## My retained failures

I retain the first ownership helper's reserved `resource` name refusal and
rename it `is_resource`. My first new checker shadow uses an unspaced generic
assignment and fails parsing before compiler publication. I retain that exact
candidate and a parser probe, correct its spelling, and require successful
parsing before positive and negative checking. These are fixture failures.

The first original Linux slicing run also refuses strict Clang compilation
because its GCC installation selection lacks the expected libstdc++ headers.
I retain the diagnostic and select the existing GCC13 directory explicitly for
Clang. I keep `-Werror`, original assertions and sanitizer flags. I do not infer
a product failure from that setup refusal or claim its unexplained history is
resolved.

## My commands

```sh
make -j8 test-typechecker test-transpiler test-eval test-module-metadata \
  test-env-scoping bin/nanoc_c nano_virt obj/nvm2c_ownership_reference
bin/nanoc_c src_nano/nanoc_v06.nano -o .test_output/ownership-source-final
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/ownership-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_ownership
NANOLANG_SLICE_COMPILERS="nanoc_c $PWD/.test_output/ownership-source-final" \
  NANOLANG_SLICE_PRODUCERS="nano_virt $PWD/.test_output/ownership-source-final" \
  NMS_NATIVE_CLANG='clang --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13' \
  python3 -m unittest -v tests.test_selfhost_native_array_slice
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/ownership-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_container \
    tests.test_nanolang_translator_payloads tests.test_nanolang_translator_module \
    tests.test_nanolang_translator_instructions tests.test_eval_global_bindings \
    tests.test_nanolang_translator_passive_module \
    tests.test_nanolang_translator_passive tests.test_nanolang_translator_passive_calls
/usr/bin/python3 -m unittest -v tests.test_nanoisa_schema
make test-embedded-nano-identifiers
PYTHONPATH="$PWD" python3 .test_output/ownership-collect.py --corpus
```

Darwin prepends `/opt/homebrew/bin:/usr/local/bin` to PATH and omits the Linux
GCC-directory selector. My tool manifests record the actual compiler versions;
the command name `gcc` on Darwin does not establish GNU compiler coverage.

## My evidence boundaries

I build each fresh full NanoLang compiler with the C seed at the implementation
pin, then test its actual resulting executable. This is development compiler
qualification, not a raw bootstrap fixed point or installed product cutover.
Both producer routes import and compile the NanoLang declaration implementation.
My five execution routes are C-seed native plus C/NanoLang producers in NanoVM
and instrumented generated C. The generated code uses ASan/UBSan and an ordinary
AOT runtime; my C reference oracle is ordinary and calls the shared validator
directly. I do not claim full-provider sanitization, allocation-failure coverage,
complete leak freedom or Darwin LeakSanitizer coverage.

I independently regenerate every synthetic input and retain its exact bytes,
explicit expectation and actual C status/authority report. My manifests bind
source and post-run tool hashes; they are not complete SDK closures. Decoder
unittest images and native binaries are not archived. I retain original slicing
source/command logs and eight producer modules per host, plus hashes of all
files in those original-suite artifact directories. `retained-inputs.json` binds each
compressed input to its uncompressed length/SHA256 and compressed SHA256.

Private declaration projections, remaining module profiles, representation
analysis, C emission in NanoLang, canonical integration, installed/raw bootstrap
cutover and the complete original #976 release scope remain open. #979 remains
open for its full fresh-stage/integration acceptance. I change no product
selection and close no parent issue at this component boundary.
