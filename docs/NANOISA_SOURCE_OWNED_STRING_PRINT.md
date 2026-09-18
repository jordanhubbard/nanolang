# My owned string parameter and print source contract

I track `task_a4326d3ba9a74cdd938b97e777d10132` under unchanged affine-example
`task_c4351c720aee424ea9b90187e51a08f2`. I depend on runtime task
`task_badd6be9c31a6e2eac810b95913b4f84` and paired source graph PR744, merged
at b147e9bb. This document precedes implementation. I await the runtime's
reviewed qualification and canonical merge; I do not edit its branch or admit
source through an unqualified runtime. My static review uses peer contract
0209a4d6 and production checkpoint ae4b064c, without claiming that checkpoint
has passed qualification.

## My separate type boundaries

My current `borrow_tag` and `nb_tag` return only int/bool tags. Their callers
also validate resource fields, constructor fields, scalar locals and results.
I keep these helpers narrow. A separate exact parameter classifier may admit
STRING only for mode-zero helper parameters in the value-call graph. I use it
consistently for signature validation, function parameter tags, initialized
parameter slots, ownership descriptors and actual-argument comparisons.
Strings use NO_INDEX and never acquire a resource layout or move authority.

Resource leaves remain int/bool, with existing finite nested resource local
and parameter support. Resource results remain the existing scalar-leaf
contract. String results, string fields, new string local declarations and
assignments, string collections, conversions and general string operations
remain refused. Merely recognizing a string parameter must not widen any of
those sites. Borrowed CALL_REF remains its separate profile; string/print
admission here cannot widen its instruction or signature contract.

## My expressions, effects and lifetime

I admit string literals and reads of live initialized string parameters as
string expressions for exact string actuals or printing. Literal bytes and
lengths enter the existing constant table and canonical text format without
loss or reinterpretation. Empty strings are valid. Embedded NUL refuses
before publication, matching the bounded VM/native output contract. I do not
silently trim at NUL or claim language-wide string restrictions.

A value call inspects every actual once in source order. Its string argument
must have the exact STRING tag; an owner argument retains all current live,
mode, nominal, borrow and pending-disposal checks and transfers exactly once.
A string is copied as an immutable value, not marked as a consumed owner.
Scalar/owner/string interleaving preserves the complete positional metadata.
No implicit conversion, default argument, or declaration-name fallback applies.

I lower only the parser's checked print statement or an exactly recognized
builtin call. I inspect how each producer retains print versus println and
honor normal declaration/local identity rather than selecting a shadowed
function by spelling. PRINT and PRINTLN each consume one exact string and
produce no stack value. PRINTLN adds one newline. These effects are admitted
in ordinary functions and selected shadows in this bounded value graph.
I do not fabricate a VOID operand or bypass pure-context checking.

The runtime owns string root retention, prepared-argument transfer, print trap
release, proof invalidation and terminal cleanup. This source change supplies
exact validated operands and descriptors; it changes no runtime authority.
All source resource exit/region obligations remain explicit and checked.

## My graph, shadows and publication

I preserve PR744's complete acyclic eight-function graph, source-order indices,
zero-argument factories, exact owned/VOID results, entry0 scalar result and
synthetic shadow entry. Every selected shadow is emitted in the existing
order or the entire publication refuses. A shadow calling main executes main;
I never omit that activation to fit the bound or suppress output.

C-seed and both selfhost stages must emit matching constants, signatures,
ownership/layout descriptors and optional lexical names. Parameter string
names retain their actual slots and function intervals. I do not introduce
hidden named string locals. Generic advisory verification is still insufficient:
specialized publication requires positive owned admission. Failure retains the
previous output and the first diagnostic.

## My acceptance before admission

I use newly built compilers and unchanged ordinary source, without replaying
held failed product artifacts. My paired controls cover:

- literal and parameter printing with and without a newline, empty strings,
  escaped bytes and non-ASCII bytes;
- forwarding a string through multiple helpers, sibling/repeated calls and
  mixed string/int/bool/owner positional arguments;
- observing output before normal and false-assertion cleanup, with mandatory
  false-shadow publication preserving an existing output file;
- exact metadata and name intervals, binary/text roundtrips, stripped-name
  execution, and VM versus strict sanitized native output bytes;
- semantic refusal of string fields/results/new local bindings, unsupported
  string operations, wrong positional tags, embedded NUL, reference mixtures
  and shadowed builtin targets, with no refused-artifact execution; and
- the unchanged `examples/language/nl_affine_resource_demo.nano`, including
  all original open_file, close_file and main shadows, with exact output order.

I retain the complete source-borrow suite and applicable owned result/graph,
consuming-call, borrowed-call, assertion and new string runtime regression gates.
Linux source evidence does not substitute for the peer's required Darwin runtime
qualification. The full product gate and full ownership parents remain separate.
Only measured unchanged-example acceptance can advance c435; this proposal
alone closes nothing.

I integrate runtime PR750 at8b0fa1fd before production. My specialized literal
normalizer preserves the existing source escape rules while rejecting an
unescaped source `\0` before the general C-string decoder could truncate it.
The selfhost path quotes normalized bytes with the existing assembler escape
rules; this is local transport support, not a change to general string syntax.
The literal backslash followed by zero remains distinct from decoded NUL.

## My literal and advisory constant ordering prerequisite

I record `task_c396592893874b81924dea82a9f5c769` before correction. At frozen
source d77c41ac my 42-method gate passed 40 methods in 357.378 seconds. The
new byte fixture and unchanged affine example failed exact canonical dump
equality: my C producer added local-name metadata strings after each function,
while my selfhost/assembler placed executable string constants before advisory
metadata constants. My retained log is
`/tmp/nanolang-source-owned-string-paired.log`; all twelve source/tool hashes
remained unchanged. I demonstrated a serialization mismatch, not a runtime
lifetime or execution defect.

I retain pending local-name descriptors in my specialized C compiler context,
bounded by eight functions times 256 slots. Each descriptor keeps its exact
function, slot, PC interval and AST-owned name pointer until compilation ends.
After all functions emit successfully, I publish those descriptors in existing
function/slot order through the unchanged validated metadata API. This matches
the existing selfhost ordering: layout/function names, executable literals in
emission order, then lexical metadata. No literal moves across a source
evaluation, and neither ownership metadata nor wire format changes.

A failed function prevents all pending name publication and releases the
compiler context normally. My original exact-dump, strip/roundtrip, raw-byte
VM/native and complete shadow assertions remain unchanged. I rerun the affected
paired gate after correction and retain the first 40/42 outcome separately.
