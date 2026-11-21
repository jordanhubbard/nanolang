# My NanoLang payload decoder checkpoint

At `3947d68fb` I implement ten core public payload codecs in NanoLang, retaining
validated row spans, embedded binary zero bytes and complete unsigned fields.
I compare with actual decoded/re-encoded C fields, not only status codes.

| Host and input | Gate | Result |
| --- | --- | --- |
| Linux ARM64, current isolated tree | Initial payload matrix |1585 cases, five routes,8.272s, pass |
| Linux ARM64, current isolated tree | Combined container/payload matrix |165+1585 cases, five routes,12.822s, pass |
| Darwin ARM64, fresh detached3947d68fb worktree | Fresh C tools and default component targets |Both component matrices, three routes each, pass |
| Darwin ARM64, same exact source | Fresh full NanoLang driver and combined matrix |165+1585 cases, five routes,8.424s, pass |

My routes are direct C-seed native execution plus both C and NanoLang producers
through NanoVM and sanitizer-instrumented C AOT. Each host builds its own full
NanoLang driver from `src_nano/nanoc_v06.nano` using the C seed, with the original
shadow selection and deadline. These drivers do not establish raw Stage1/Stage2
equality. The existing AOT runtime objects are ordinary; emitted C is instrumented.

My payload fixtures include785 independent expected results and actual encoder
byte comparisons. They cover all sixteen tags, embedded zeros, unsigned64-bit
endpoints, every truncation of representative tables, impossible counts,
reserved fields, padding and refusal precedence, callback ordering, shared/deep
record DAGs and cycles. Eleven useful payload shadows, seven existing container
shadows and the driver hex shadow remain selected by default. The first driver
fixture used reserved `byte`; I retain its parse refusal and correct the name
to `octet` without changing lexer policy.

The shared route builder replaces duplicate compiler/link setup in the original
container test. I rerun both matrices after that refactor. My embedded fixture
gate passes all five controls and scans3275 fragments with zero reserved names;
the133 previously lexer-refused fragments remain reported. Python compilation
and whitespace checks pass.

I build Darwin in `~/.worktrees/nanolang/translator-payloads-20261120`, leaving
my primary checkout's uncommitted callable work in place. Its final test tree
is clean at3947d68fb. I collect full logs, actual SDK/compiler identification,
and post-run hashes. All ten shared source/fixture/recipe files match Linux
byte-for-byte. The JSON manifests and compressed logs are retained here.

The tests delete their temporary per-case files/modules/native images on
success. This evidence is reproducible component qualification, not an archived
exact-release-candidate product or complete SDK receipt. I preserve that limit
rather than inventing a product receipt.

My next required step is full module cross-section and extension validation,
then instruction/ownership/host admission, representation analysis, C emission
in NanoLang and product/bootstrap cutover. I preserve default-empty sections
instead of inventing a required-section set from normal serializer output.
Public layout transport does not admit private array/map/ownership execution.
The full translator, #1142 and release #976 remain open.
