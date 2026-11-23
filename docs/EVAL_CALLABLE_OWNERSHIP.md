# My interpreted callable ownership contract

I track this existing release requirement under #1140, with record-carrier
reclamation kept separate under #1116. My current implementation does not
satisfy this contract. I preserve the strict original nine-route/64-call audit,
its 576 retained names, and the valid escaped-record failure before repair.

My retained baseline53abc0a57a actual-expression probe finds fresh callable names/signatures from
declaration and array reads, borrowed names/signatures from variable and record
field reads, and a call-result expression that retains one intermediate name.
These are observed ownership differences, not interchangeable cleanup rules.

My retained baseline escaped-record fixture returns a callable field referring to a local
binding. The ordinary interpreter later returns VOID with a corrupted function
name. A targeted ASan/UBSan build of actual eval/env and the probe reports the
same name allocated by own_function_identifier, freed by eval_evaluated_call's
activation cleanup and read later by eval_call. The other providers are fresh
normal objects. This trace does not establish complete sanitizer acceptance.

I require one explicit ownership rule for callable metadata:

- My source expression results carry an independently owned function name and
  complete signature. Reading a variable, record/union field or aggregate
  element must preserve its original owner's metadata and logical target.
- An owned binding or container slot adopts an owned expression result, or
  clones borrowed C/API or pattern-bound input before its owner can retire.
  Record/union copies preserve independent callable metadata just as existing
  callable array and tuple copies do. I do not free a callable field merely
  because it is present: its installation/copy boundary must first establish
  ownership.
- Source argument temporaries and computed callees have explicit retirement
  after their last use. C/API arguments remain borrowed; activation-owned
  parameter copies must not invalidate their callers. Early returns and
  failed dispatch preserve this distinction.
- A returned callable transfers owned metadata to its caller. A returned or
  escaped container must retain live callable fields after every defining
  lexical block and activation has ended. Nested signatures remain complete
  and independent. A copy must not leave an unowned intermediate allocation.
- Scope/activation cleanup retires only the metadata it owns. Ignored callable
  results, assignment replacement, handler/pattern/loop bindings and callbacks
  need explicit treatment. I preserve eager evaluation, once-only behavior,
  declared module identity and existing argument/return behavior.

I will audit all corresponding creation, copy, store, read and retirement sites
before enabling broader cleanup. I preserve the existing Value/container
layout and declared foreign ABI; this contract adds no value-kind encoding,
cache, delayed-retirement success exemption or C translator dependency.

I require the original strict counters to reach zero while caller identity and
signatures survive. Escaped records/unions, arrays/tuples, reassignment,
computed calls, nested signatures and ordinary lexical/module callbacks must
retain their behavior under actual sanitizer checks. I keep record/union
carrier reclamation separate rather than claiming a metadata correction
reclaims every carrier.

My original interpreter, typechecker, source/shadow, fresh-provider and paired
platform requirements remain mandatory at their original assertions and
budgets. The targeted trace and future focused passes cannot close #1140 or
the full 69-requirement release by themselves. My older dirty callable audit,
private candidates, stashes and frozen roots remain untouched.

My first repair at d732326d70 copies borrowed callable names and complete signatures in
create_struct and create_union. Six constructor values retain independent
nested signatures after explicit source retirement; this bounded control passes
with my fresh ASan/UBSan provider build and leak detection enabled. My source
record/union fixture and all its shadows also finish, then LeakSanitizer reports
3,920 bytes in169 allocations. I retain that full unsuccessful run. At that constructor-only pin, my original
nine-route audit still retains576 names. I have not enabled additional field,
container or scope cleanup. These constructor copies establish a prerequisite;
they do not satisfy the required contract above.

My subsequent owned-expression foundation gives variable/field/aggregate reads
independent callable metadata, transfers returned callables without a lost
intermediate copy, retires direct callable scope/argument/callee temporaries,
and preserves borrowed C/API arguments. I preserve control-flow flags and
logical module targets. Source constructor/tuple inputs retire after copying;
handler parameters and the two async argument-bundle ownership cases stay
explicit. I keep existing nested-record copy behavior and add no field or
carrier reclamation. Borrowed cached task results are copied; complete spawned
result/handle lifecycle acceptance still remains open.

My unchanged nine-route/64-call audit now retains zero names normally and in
a fresh complete ASan/UBSan provider build with leak detection enabled. Five
raw expression reads independently own their results. Four further 64-call
routes balance discarded results (320/320), reassignment (192/192), effect
parameters (64/64) and async forwarding (192/192); an external diagnostic of
those unchanged four methods also passes ASan/UBSan with leak detection enabled.
The complete interpreter/typechecker suites and record/union plus borrowed
C/API caller-survival controls pass normally. My new expression control fails
against retained constructor-only providers.

I preserve both complete fixture sanitizer failures: expression/globals572bytes
in26allocations and record/union3720bytes in159allocations. Both finish their
assertions and shadows before LeakSanitizer fails. I do not promote the external
four-method diagnostic into acceptance of either complete fixture. My #1205 repair restores the strict standalone REPL build and wires its existing
three callable controls plus fifteen state controls into ordinary units. All
eighteen pass normally, under ASan/UBSan with leak detection enabled, and with
unchanged production build flags. I check executable statements before mutation,
retain parsed source owners through environment teardown, clear variable owners
without retiring declarations, and patch compatible function slots without
replaying top-level effects. I retain the original malformed-escape ASan failure,
module-context leak and typed-assignment failure alongside corrected controls.

This is partial persistent-session acceptance. A rejected imported module still
publishes a declaration whose AST has been retired (#1223), and :save still omits
loaded fragments (#1225). My module-context-string correction (#1222) does not
repair that separate AST lifetime. Complete #1205/#1140/#1116, current-source
bootstrap, paired platform and release acceptance remain open.

Under #1207 I keep TYPE_BORROW_SHARED and TYPE_BORROW_MUT slots non-owning
when I create or rebind views and when I traverse roots for static-array
reclamation. I neither release the old target nor retain the new target for
a borrowed view. An owning root still reclaims its array exactly once, even
when both kinds of borrowed alias reference it. My source checker continues
to require a declared nominal record borrow at a direct-call argument; my
callable/string view controls exercise the existing C/API ownership boundary
and grant no new source permission.

My retained baseline reports actual callable and array use-after-free. Both
original probes pass with fresh candidate providers. My permanent caller
survival, nested aggregate and mixed-owner controls pass normally and with
complete ASan/UBSan providers and leak detection enabled. Full interpreter,
checker, environment and callable neighbors also pass with their known
whole-fixture leak boundary preserved. I retain the diagnostic link refusal
caused by attaching evaluator test hooks to the standalone checker, then
qualify the checker with its ordinary provider objects. My [bounded receipt](evidence/borrowed-value-ownership-linux-20261123/README.md) keeps the exact
commands, source, products and both failures. This repair does not retire
record literals or callable fields and does not close #1116 or #1140.

My next bounded #1140/#1116 step retires an explicit record-literal temporary
after a normal binding has copied it and immediate tracing has finished. I
check the expression kind, actual record result, non-borrowed binding and
independent destination carrier. I do not retire a returned/projected record
or a field-store input through this rule. I release callable copies in this
fresh literal tree before the existing string/record-carrier cleanup; shared
arrays retain their existing lifetime. I separately retire direct callable
fields copied into normalized owned environment records during replacement
and teardown. Generic raw-array cleanup and nested owned-binding record
reclamation remain unchanged.

My unchanged complete expression/globals fixture now passes fresh complete
ASan/UBSan providers with leak detection enabled. The repeated global
initialization/replacement regression balances991 callable allocations and
991 releases; the identical test linked to retained old providers reports
991/394 and refuses. Caller-owned legacy fields, raw-array fields, nested
record lifetime and shared array identity survive the candidate. My original
134 interpreter/88 checker methods and adjacent callable/environment/borrow
controls pass; full interpreter/checker sanitizer runs keep leak detection
disabled for their broader outstanding leaks.

I retain the complete record/union fixture's unchanged3720-byte/159-allocation
LeakSanitizer failure. This step does not close #1116/#1140, qualify all local
record escapes or field replacements, or establish paired release acceptance.
I also retain a preexisting NANO_INTERPRETER syntax failure under #1208:
AST_SET traces an undeclared old_value. A borrowed snapshot read after
replacement can dangle, so its repair requires an actual tracing-enabled
build and runtime old/new-value controls. I make no tracing-enabled acceptance
claim in this [bounded receipt](evidence/record-binding-ownership-linux-20261123/README.md).

Under #1208 I now serialize assignment values synchronously before binding
replacement can retire old storage. I add no snapshot allocation or Value
layout change. The focused native observer checks21 live old/new string,
callable, record and scalar values across enabled, filtered and disabled
tracing, including self-assignment. The real tracer produces the checked
output; all14 runtime kinds use a dedicated ValueType label mapping instead
of incompatible declaration-type ordinals. A private evaluator with retirement
moved ahead of serialization reports an actual heap-use-after-free in the
same unchanged regression.

My tracing-enabled fixture passes complete fresh ASan/UBSan providers with
leak detection enabled. All134 original interpreter methods also pass with
the tracing-enabled evaluator calling production tracing directly, in normal
and instrumented builds; that broad suite retains its existing disabled leak
detection boundary. Ordinary checker/callable/environment neighbors and the
complete expression/globals leak gate still pass. I retain the unchanged
record/union3720-byte/159-allocation failure and the original build refusals
in my [tracing receipt](evidence/tracing-assignment-linux-20261123/README.md).
General carrier/escape ownership and paired release acceptance remain open.

My #1208 production audit also retains a real interpreter link refusal:
NANO_INTERPRETER enables evaluator references, but the common source list
omits tracing.c. I add that source and isolate the explicitly enabled observer
tracer. Default, tracing-enabled ordinary and tracing-enabled instrumented
interpreter products now build and run an assertion-bearing string/callable/
record program, help and version. The focused trace/ownership regressions
and original interpreter/checker neighbors remain green under their stated
leak boundaries; I retain the3720-byte record/union failure.

This [production receipt](evidence/tracing-production-linux-20261123/README.md)
establishes the build dependency and live evaluator/tracer behavior. My
standalone CLI does not configure its documented trace flags or automatically
run shadows; I retain that complete contract under #1209 and mark the current
CLI limitation in FEATURES.md. Neither a product exit0 nor API-level tracing
accepts that separate interface. Broader #1116/#1140 and release work remains.

Under #1210 I make scalar container/result snapshots consistent. Tuple,
successful try, Result unwrap/default and cached task reads return independent
STRING buffers and complete owned callable metadata. I leave non-scalar
identities and low-level borrowed Result/scheduler APIs unchanged. A fresh
unwrap_or STRING default transfers its existing storage; a default aliasing
a live STRING binding is copied. My GC balance control requires exactly one
additional owned STRING and returns to its original count after retirement.
STRING identifiers keep their existing binding-alias rules; I do not rewrite
all STRING expression ownership or change Value/container/FFI layout.

My original tuple STRING witness reports use-after-free after nested lexical
cleanup, then passes unchanged with the candidate. My C/API fixture balances
29 callable allocations/releases, preserves nested signatures/caller inputs
and shared mutable array identity, and passes complete fresh ASan/UBSan
providers with leak detection enabled. The identical new fixture refuses
retained old providers on its first independent-STRING assertion. My initial
record-field hypothesis is disproved: that helper already copies STRING and
its replacement control passes. Heap-bearing mutable borrows remain rejected.

My full source fixture also exposes the existing parsed-prefix ? route
returning VOID; only AST_TRY_OP was implemented. I use the same union/empty/Err
propagation and return-target path for both encodings without parser/checker
admission changes. All seven source shadows complete with zero failures,
including Ok and Err propagation, in normal and instrumented correctness
runs. Original interpreter/checker, tracing and ownership neighbors pass.

I retain the complete new source-fixture1937-byte/82-allocation LSan failure
and original record/union3720-byte/159-allocation failure. The source semantic
ASan/UBSan run explicitly disables leak detection; my C/API ownership unit
does not. These [separate results](evidence/scalar-projection-ownership-linux-20261123/README.md)
do not close general #1116/#1140 carrier/escape/heap or paired release acceptance.
