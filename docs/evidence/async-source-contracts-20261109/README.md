# My scalar async source compatibility candidate

I retain the existing `tests/test_async.nano` fixture and C parser contract:
`await` binds to the following primary and has its operand's type. I recognize
my existing ASYNC/AWAIT token identities and retain PNODE_ASYNC_FN in the
function table and parser result. I preserve function positions and ordinary
metadata while invalidating retained affine authority in the new declaration
marker. My existing borrow/effect provenance guards remain unchanged.

My new controls exercise the unchanged original fixture, int/float/bool/STRING
values, primary binding, nested scalar awaits, keyword-adjacent identifiers,
evaluation once in order, and seven malformed/type/failed-shadow refusals
with prior-output preservation. I reuse the existing first-creation cache:
source-created controls never silently rebuild a failed source compiler.
I add an emitter shadow for conservative async effect metadata; it remains
unexecuted locally because my checker component does not import that emitter.

My [local checker commands](local-checker.json) pass the existing actual C-to-native
checker component, including the selected parser/lexer shadows and checker
entry assertions. My [native image check](local-checker-product.json) matches
its actual Mach-O bytes with unchanged input and product pins; I retain
[exact logs](local-checker-texts.json.gz). My initial checker result is retained
separately before adding parser-result identity assertions.

My [local controls](local-controls.json) pass all five embedded-identifier
controls and the repository fixture scan, plus three actual C-created async
VM/native methods. I retain [four inspected native products](cseed-products.json)
and [exact commands/input pins/refusals](cseed-texts.json.gz). These local
providers are pinned existing binaries; these observations do not establish
their current C-source provenance, source-created async execution, fresh stages,
paired platforms or native lifecycle acceptance.

My [first source-only C attempt](syntax.json) refuses the existing array ABI
boundary in that profile. The [unchanged baseline](syntax-baseline.json) gives
the same refusal. I preserve both logs and do not count that route as passing
or change its profile to obtain acceptance. My native checker component uses
the original full selected shadows and original deadlines.

Current hosted source/platform, unchanged fresh Stage 1/2 fixture execution,
complete original corpus, profile budgets, record-frame lifetime, security,
native lifecycle and release requirements remain open. Scalar compatibility
does not establish coroutine scheduling or foreign authority.

My [complete pre-async Clang baseline](baseline-7a-clang/README.md) verifies the
original component at exact source pin 7a709ee6b. The same-run GCC failure is
retained under `baseline-7a-gcc/`; reaching its later always-run translator
steps did not establish a source pass. I keep these outcomes separate from
async acceptance.

My [source custody check](custody-precommit.json) preserves the reviewed
code-only baseline 21c3818c0, a real descendant of the prior 91735dabc
checkpoint. It adds the new test file and preserves the prior 92,816 paths.
My [metadata commands](metadata/checks.json) pass the real Git controls, all
eight integration locks, four delegates, current authority review and canonical
validation. The native receipt policy remains unconfigured and refuses
acceptance; no receipt or implementation selection is invented.
