# My shared match selection and completion policy

I define the source semantics required by
`task_477bdd430a1442e7bc19cbacdbac0bde` and
`task_70c5a56802e44142af0f19da2469f654`. This is a review contract, not a
production implementation. I make no release, product-acceptance or backend
coverage claim from this document.

My contract starts from canonical source
`37c68e8a9b68bb301fc0f950b4040337963ee0db`. I keep the frozen 950f product
and platform evidence unchanged.

## My decision

I test match arms in lexical source order. For each arm I:

1. test its pattern;
2. create its payload binding only when that pattern matches;
3. evaluate its guard exactly once, when present, with that binding in scope;
4. discard that arm's scope and continue when the guard is false; and
5. execute the first successful arm exactly once and skip every later arm.

A wildcard is an always-matching pattern. It is not a deferred `default` arm.
An early guarded wildcard may fail and allow later arms to run. Multiple
guarded wildcards are valid and retain their source order. An unguarded
wildcard, or a wildcard guarded by the literal `true`, is unconditional; I
reject every later arm as unreachable. This also rejects a second wildcard
after an unconditional one. I do not silently move a wildcard, select the last
wildcard or let a named arm override an earlier successful wildcard.

```nano
match choice {
    _ if (should_intercept choice) => { (record "intercepted") }
    Some(value) if (is_ready value) => { (use value) }
    _ if (should_defer choice) => { (record "deferred") }
    _ => { (record "ordinary") }
}
```

Each guard above runs only after its pattern matches. The first true guard
wins. The following form is invalid because its second arm can never run:

```nano
match choice {
    _ => { (record "always") }
    Some(value) => { (use value) }
}
```

I require every match to be statically total, whether its selected value is
used or discarded. A statement match does not acquire an implicit no-op case.
For a finite union, an unguarded named/or-pattern arm or one guarded by the
literal `true` covers its named variants. A reachable unconditional wildcard
covers every remaining variant. Other guards do not establish coverage because
they may be false. An integer-literal match requires a reachable unconditional
wildcard. A producer that cannot establish coverage for another pattern domain
must refuse that form rather than guess.

A guard must have the exact checked type `bool`. An unresolved or unknown guard
type is a checked error, not permission to emit code.

No accepted source program can normally reach the no-success path. Every
runtime still keeps a terminal backstop for corrupt values, malformed bytecode
or a compiler/runtime contract violation. That backstop reports a first-person
match failure and stops execution. It never returns `void`, manufactures a
zero-initialized value, falls through to the following statement or marks the
path unreachable in C while permitting execution to reach undefined behavior.
The exact transport may be a VM trap, interpreter failure result or native
fatal diagnostic, but source execution cannot continue.

## My value and control-flow rules

I evaluate the scrutinee once before testing any arm. Pattern tests do not
duplicate it. Failed patterns do not evaluate bindings, guards or bodies.
Failed guards do not evaluate bodies or later effects before their source turn.

An arm block's final expression supplies the match value. `return` inside an
arm or a guard exits the enclosing function; it never means "return from this
arm." A statement match discards the selected arm value. Existing
`break`/`continue` targets remain the surrounding source loop: a lowering must
not introduce a wrapper loop or switch behavior that captures them. Nested
matches apply the same rules independently.

A payload binding is visible only in its matching arm's guard and body. A false
guard ends that arm scope before the next arm begins. It restores shadowed
names and preserves the ownership state required to try the same scrutinee in
the next arm. Resource-bearing matches retain their existing stricter admission
until their checker can prove this behavior; a backend must refuse unsupported
moves or borrows rather than approximate them.

I preserve observable source effects in this order:

```text
scrutinee
pattern 1, matching guard 1
pattern 2, matching guard 2
...
first selected body
```

No later guard or body runs after selection. A terminal no-success backstop has
no source continuation. Each runtime must follow its established failure and
runtime-allocation cleanup contract; this policy does not promise source-level
destructors after a fatal process termination.

## Why I choose this policy

Lexical first-success order is already the natural rule for guarded named arms
and for `cond`. Applying it to wildcards gives one rule instead of a special
"named arms first, last wildcard later" exception. It also makes guard effects
and payload lifetimes readable directly from source.

Requiring totality turns no-success into a compile-time source error and keeps
the runtime path as a defensive invariant check. Returning `void` from a value
match, silently doing nothing in a statement match or relying on native
undefined behavior would give the same source different meanings by route.

I considered requiring exactly one final wildcard. That is easy for a backend,
but it discards useful guarded fallback chains and makes backend limitations
part of the language. I instead permit early and repeated conditional
wildcards, while rejecting the simple and objectively unreachable case after
an unconditional wildcard. Individual targets may remain narrower through a
checked refusal until they implement the shared rule.

## My current route audit

The current implementations do not satisfy this contract:

| Route | Current behavior at the pinned source | Required disposition |
| --- | --- | --- |
| C-seed parser | Accepts named, integer, or-pattern and wildcard arms with optional guards; it imposes no wildcard order. | Keep the syntax and preserve lexical arm order. |
| C-seed checker | Expression coverage is a warning, statement coverage is absent and an unknown guard type is accepted. | Make exact guard typing, reachability and totality checked errors for both value and statement matches. |
| C-seed interpreter | Tries all specific arms first, remembers only the last wildcard and returns `void` after a diagnostic on no success. | Execute one lexical chain and return a terminal failure on an impossible miss. |
| NanoVirt/NanoVM | NanoVirt emits lexical first-success order and ASSERT/HALT on a miss. | Retain ordering, give the trap the shared diagnostic category and admit only checked-total source. |
| C-seed native emitter | Guarded matches use a lexical `_matched` chain; unguarded matches use C `switch`, so an early default loses to a later case and repeated defaults fail C compilation. Some miss paths yield zero or fall through. | Use semantics-preserving ordered control flow or prove a switch is equivalent for the exact checked shape; keep a terminal backstop. |
| Self-hosted parser/checker | The stored match shape has no guards, accepts wildcard position/repetition and does not establish totality. | Represent and check the shared syntax, or issue an explicit capability refusal before emission. |
| Self-hosted native emitter | Emits a C switch with `default`, zero-initializes the result and has no guarded source form. | Align ordered/total semantics before claiming this source profile. |
| Self-hosted NanoISA emitter | Accepts only complete, distinct, named scalar-union arms and emits ASSERT/HALT after them. | Preserve its safe narrow refusal, then widen deliberately; do not claim wildcard/guard support before parser and checker parity. |
| Public C backend | Its bounded guarded-union child accepts a single final wildcard with static coverage and terminal failure; it refuses early/multiple wildcards. | Keep that refusal until its ordered lowering is qualified for the wider shared policy. |
| NanoCore subset/export | The subset walk and exporter retain arm bodies but omit guards, so a guarded match can be modeled as a different program. | Represent guards and ordered miss behavior, or reject guarded matches from the subset before export. |
| NanoISA translators | `nvm2c`, LLVM and Wasm consume branches plus ASSERT/HALT rather than source matches. | Preserve the verified branch order and terminal trap through each selected translation. |

The surrounding analyses are part of the semantic path. Resource and purity
analysis already inspect guards, while current effect and CPS walks omit them;
type inference does not check them; and the PGO clone path does not clone them.
Nominal, folding and DCE walks do retain guards. Every transform that accepts a
match must preserve the scrutinee, ordered patterns, guards, bindings and bodies
as one unit. NanoCore, effect analysis and optimization are not allowed to erase
guard effects merely because the runtime emitter is correct.

## My implementation boundary after review

No production edit starts from this contract until root reviews the semantic
decision. After approval, I require changes in dependency order:

1. Update `docs/SPECIFICATION.md`, `spec.json`, the control-flow guide and the
   canonical match guidance with the shared rule and migration examples.
2. Give both source parsers and AST/schema paths a lossless ordered match form,
   including guards, source locations and arm-local bindings.
3. Put one shared coverage/reachability policy behind C-seed value and statement
   checking, then implement the same decision in the self-hosted checker.
4. Repair every accepting analysis/clone/export path before enabling a producer
   that depends on it. A route may retain a precise checked refusal.
5. Align the interpreter, NanoVirt, both native emitters, the canonical NanoISA
   emitter and public C without broad fallback or fabricated values.
6. Verify terminal trap preservation in NanoVM and the C, LLVM and Wasm NanoISA
   translators.

I will not close either MAC task from a documentation-only decision. Their
roadmap rows remain open until implementation and all-route evidence are merged.

## My qualification matrix

Fresh tests must prove the policy rather than merely compile it.

### Selection and effects

- scrutinee evaluation occurs once;
- an early guarded wildcard true/false pair selects or continues in source
  order;
- two guarded wildcards prove first-success order for both true/false
  combinations;
- repeated named variants and or-patterns preserve the same rule;
- a final wildcard retains the common total case;
- an unconditional early wildcard rejects the first later arm with a stable
  source location; and
- counters/prints in scrutinee, guards and bodies prove skipped effects remain
  skipped.

### Totality and failure

- incomplete union matches reject in value and statement positions;
- one guarded arm per declared variant still rejects without unconditional
  coverage;
- integer-literal matches without an unconditional wildcard reject;
- non-BOOL and unresolved guards reject before emission;
- accepted total matches never reach the miss backstop; and
- low-level malformed-tag/bytecode harnesses prove interpreter, VM and selected
  native translations terminate without a value or following side effect.

### Scope, ownership and control flow

- payload bindings exist in their guards/bodies and nowhere else;
- a false guard restores an outer same-named binding before the next arm;
- nested matches retain independent bindings and private labels;
- arm final expressions yield match values while `return` exits the enclosing
  function;
- `break` and `continue` retain their enclosing loop targets; and
- supported owned/borrowed payloads prove false-guard restoration, branch joins,
  selected cleanup and refusal of unsupported transfers.

### Producers and platforms

The same source cases need ordinary, non-mocked acceptance through the C-seed
interpreter, C-seed native emitter, NanoVirt/NanoVM, self-hosted native emitter,
self-hosted NanoISA emitter and public C for every profile each route admits.
The compiler's Stage1 and Stage2 products, imports, dependency shadows and
installed-tool selection remain visible in the logs. NanoCore gets either a
corresponding model/checker result or an exact refusal test.

NanoISA artifacts must verify and exercise their backstop through NanoVM,
`nvm2c`, LLVM and Wasm where those backends support the value profile. Native
controls run under strict GCC and Clang with the existing sanitizers on Linux
and Darwin. Existing deadlines, shadows and failure checks remain unchanged.
No skipped route, warning-only checker result, old failed artifact, widened
timeout or removed sanitizer qualifies this contract.

Full product acceptance and the publication decision remain separate gates.

## My bounded C-seed interpreter slice

I implement the next dependency-ordered slice from canonical
`8ddba93efe2a888e21c9ab484f60fabea51d02be`. My shared C-seed checker already
requires exact `bool` guards and total `int` or known-union matches. I now make
that checker reject the first arm after an unconditional wildcard, including a
wildcard guarded by literal `true`. A conditional wildcard remains reachable
and may continue to later arms when its guard is false.

My interpreter scans every arm once in lexical order. A wildcard participates
at its written position instead of being remembered as a deferred default.
Named and or-pattern payload bindings remain scoped to their matching guard and
body, and a false guard restores the arm scope before the next arm. I evaluate
the scrutinee once and preserve enclosing `return`, `break` and `continue`
signals.

Checked source cannot normally miss every arm. For an unchecked or corrupted
AST that does, the interpreter prints one first-person invariant diagnostic and
terminates with failure. It does not return `void` to its caller or execute a
later source effect. A forked unit control reaches this backstop without
executing any historical failed artifact.

Fresh qualification covers expression and statement rejection after both bare
and literal-true unconditional wildcards; early true/false conditional
wildcards; repeated conditional wildcards; named and or-pattern fallthrough;
once-only scrutinee/guard/body effects; binding restoration; and the low-level
terminal backstop. I retain the existing C-seed totality matrix and interpreter
suite. This slice does not widen the self-hosted parser, self-hosted checker,
public C profile or NanoCore subset, and it does not close either shared parent
or the release gate.

## My root-reviewed qualification supplement

Before executing PR855 checkpoint `d0fbd9b1`, I add integer early-wildcard
and repeated-conditional-wildcard controls to the existing interpreter fixture.
A marked scrutinee and marked guards/bodies distinguish first-success selection,
false-guard fallthrough, skipped effects and exactly one scrutinee evaluation.
I retain every existing union, binding-restoration and totality assertion.
My forked terminal-backstop fixture drains stderr through EOF with EINTR handling
instead of assuming one read returns a complete diagnostic.

I also retain the independently reviewed handler-order fixture correction
`70c12fd68`: an unconditional fallback calls mark after its conditional wildcard.
Both branches remain forbidden by the existing nonlocal-return result assertion.
The previous fixture-only correction reached the old interpreter ordering defect
and produced4099 instead of7; that terminal stays attributed to the pre855
interpreter. This supplement changes no production and waives no failing gate.
I review the complete fixture delta before fresh evaluator/totality qualification,
retain first terminals and source/tool identities, and keep all-route parents open.

## My unchecked-backstop fixture setup correction

Task `task_3f1a7d4857974ba399007aabc5239b0b` retains first Linux supplement
`5e063cad1`: buildPASS22.270s,88 evaluator controls pass including handler order
and repeated wildcards, then the child-exit assertion fails. Source/tool hashes
are unchanged;155 provider hashes also match in the labeled postfailure map.
The original binary and full terminal remain at `/tmp/nanolang-match-855-linux-first`.

Static inspection finds that run_program does not register AST_FUNCTION: its
evaluator case is intentionally a no-op. The unchecked fixture skipped the
checker, which normally registers its function, then attempted a named call.
Before another execution I explicitly register the parsed zero-argument INT
function and its body through env_define_function, without typechecking the
deliberately incomplete match. I retain the call, following exit91 sentinel,
required normal EXIT_FAILURE and exact invariant diagnostic. I print captured
child status/diagnostic if assertions fail. No production changes or assertion
removals follow. A new corrected test binary may reuse the unchanged, hashed
providers; the original failed binary is never replayed.

## My later integer-reduce fixture correction

Task `task_d4427bf731b14e899228b1443922a066` retains the next independent
evaluator terminal at `cd25b66cd`: the corrected unchecked-match backstop and
earlier match controls pass, then eval_reduce_pure_arithmetic_int fails its
initialization assertion. A fresh compiler diagnostic on its exact call shape
reports E001: reduce requires array<E>, initializer A, then fn(A,E)->A.
The two fixture calls put the function before the initializer.

Before another gate I swap only those fixture arguments to `(reduce arr 0 add_ints)`
and run the existing result15 shadow in this test. I retain all source assertions,
the diagnosed rejected source, both earlier gate terminals and unchanged compiler
providers. No production or type-rule change follows.

## My retained provider-relative fixture path

The fresh12b61 fixture now passes the corrected reduce shadow and subsequent
controls, then ffi_loader_open cannot open its relative
`obj/test_interpreter_ffi_native.so`. My reused provider lives in the original
provider checkout; the newly compiled test ran in the fixture-only checkout.
The library itself remains present and hash-identical in the provider inventory.
I retain this runner terminal separately. Before another fresh binary I set
only the evaluator process working directory to that original provider checkout,
record both checkout pins and paths, and leave compiler, fixtures and assertions
unchanged. Compilation and Python fixture discovery still use the corrected
source checkout. This corrects an input-path omission, not a product failure.
