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

My first repair copies borrowed callable names and complete signatures in
create_struct and create_union. Six constructor values retain independent
nested signatures after explicit source retirement; this bounded control passes
with my fresh ASan/UBSan provider build and leak detection enabled. My source
record/union fixture and all its shadows also finish, then LeakSanitizer reports
3,920 bytes in169 allocations. I retain that full unsuccessful run. My original
nine-route audit still retains576 names. I have not enabled additional field,
container or scope cleanup. These constructor copies establish a prerequisite;
they do not satisfy the required contract above.
