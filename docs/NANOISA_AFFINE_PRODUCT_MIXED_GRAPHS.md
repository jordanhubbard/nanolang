# My mixed affine product graphs

I record two release blockers found by the first fresh PR522 acceptance at
`ae3fc859b1e1707b1e72c29f6ea6496663e7cf3d`. MAC creation is pending because
the supported `default` profile returned `credential_rejected`; the idempotency
keys are `pr522-affine-bound-modules-20260919` and
`pr522-affine-scalar-unions-20260919`. This contract precedes production edits.

## My retained terminal

My fresh bootstrap passes Stage 1, Stage 2, both hello programs, the installed
compiler and the no-C-seed check. My installed-Stage2 quick gate then passes all
three component entries, 17 core programs, 244 eligible VM examples, nested
statement execution and 158 standard-library examples. It stops in the
unchanged affine self-host suite after 1,358.67 seconds.

The complete log has SHA-256
`7b9461abbece0d8d24d2805f4d0bff71554aeb8603be16962d02338772ae41c6`.
The source map before and after is identical, with SHA-256
`1ab3eb552805b2a69b9965eacd316d5fbab60a6ddf7e7601e97708da254b7e59`.
I retain that tree and its tools. I do not execute its refused outputs.

Eighteen Stage 1 and Stage 2 subcases report the same checked refusal:

```
I cannot lower this checked program: I require only resource records, main, one helper and selected shadows in my source borrow profile
```

The failures are the forward and reverse imported plain/owned module cases,
their long-path and nested-record forms, qualified public record identity, a
generic formal that shadows a resource record name, and a match that consumes
one outer owner in every arm. C-seed passes the same unchanged sources. This is
an installed self-host lowering gap, not evidence that those sources became
invalid.

## My static boundary

`bind_merged_functions` and `nb_register` already assign distinct function and
record identities before canonical emission. The narrow owner emitter still
rejects any nonzero import count before reading those bindings. It also admits
ordinary records only in its separate FLOAT-array profile, although these
fixtures need finite INT/BOOL records and ordinary copies alongside exact owner
moves. Removing the import check alone would therefore be incomplete.

The same emitter rejects any union count before lowering. The two accepted
generic cases use concrete owner-free `Box<int>` values. One keeps a separate
resource declaration whose name is shadowed by a generic formal. The other
passes `Box<int>` beside a `Handle` and consumes that handle in every successful
match arm. I do not classify a generic spelling as owner-free merely because
one test uses `int`.

The ordinary emitter already resolves module-qualified direct calls and lowers
unions, but it does not establish affine runtime transfers. Falling through to
it would discard the runtime ownership contract. I will not use that shortcut.

## My bound-module contract

I admit imports only after the complete merged source has passed parsing,
nominal binding, typechecking and affine checking. Every admitted qualified or
unqualified direct call resolves through the existing binding tables to one
defined declaration identity. Local or formal shadowing, duplicate targets,
body-less foreign declarations, an import outside the merged source set and an
unresolved module-qualified call remain refusals.

I retain source-order function indices only after binding has made them unique.
Import order and path length do not affect nominal identity. Every selected
dependency and root shadow remains mandatory. Program publication retains the
existing exact main closure; raw full-module APIs do not acquire permission to
invent an external ABI.

An ordinary record in this mixed profile is finite, acyclic and owner-free by
the existing transitive classifier. Its fields are exact admitted ordinary
scalars or already-qualified ordinary children. Construction evaluates fields
once from left to right and packs declaration order. Copying is allowed only
for these owner-free layouts. A resource or resource-bearing child always uses
the owner path; no spelling collision may change that decision.

## My scalar-union contract

I admit only a fully instantiated union whose substituted payload graph is
proved owner-free and whose field tags and nominal union identity are exact.
Generic formals remain lexical formals during substitution. A concrete resource
argument, a resource-bearing record, unresolved substitution, recursive or
over-budget payload graph, or mismatched instance remains a checked refusal.

Construction evaluates payloads once from left to right and preserves declared
field order and variant tag. A match evaluates its scrutinee once. Arms retain
source order, lexical payload bindings and exact Boolean guards; the first
successful arm wins. `return` inside an arm exits the enclosing function. The
arm's final expression is the match value where a value is required.

Owner state must agree across every continuing arm. An owner consumed in every
terminal arm is consumed after the match. A missing successful arm, an arm that
leaves a live owner, or disagreeing continuing states refuses publication. The
runtime descriptors retain the union tag and exact layout facts without
granting owner transfer to the ordinary union itself.

## My implementation order

1. I add explicit checked facts for already-bound direct module dependencies,
   exact owner-free ordinary record layouts and exact concrete scalar unions.
2. I make the owner emitter consume those facts for direct call identity,
   ordinary record construction/copy/projection and scalar union
   construction/matching. I do not infer authority from names during emission.
3. I publish matching layout, function, local and parameter metadata and require
   the verifier and native translator to accept exactly those facts.
4. I retain precise refusal and prior-output behavior for every unsupported
   import, foreign call, owner-bearing union and mismatched arm state.

## My acceptance

I first qualify focused C-seed, Stage 1 and Stage 2 controls for all six positive
methods that failed the product gate, in both import orders where provided. I
execute every accepted program through NanoVM and strict native translation.
I retain all existing negative methods in both test modules, including foreign
record collisions, duplicate declarations, unresolved resources, use after
move, resource-bearing generic arguments and disagreeing match joins.

I add direct controls for import-order identity, local call-name shadowing,
ordinary-record/resource name collisions, generic formal shadowing, one-time
union scrutinee evaluation, guarded first-success order, no-success refusal and
owner-state agreement. Refused publication preserves a sentinel output and is
never executed.

After focused qualification I require a fresh self-host bootstrap and the
unchanged complete affine self-host gate. Then I restart the full product quick
gate from a new frozen pin. VM/native fixed points, Darwin acceptance, PR522
merge and release publication remain later gates.
