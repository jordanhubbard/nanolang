# My canonical match guard correction

I retain recovered, unqualified candidate `45abffce66338460d9595095500bd33ff7e55dc6` unchanged. I start corrective child `task_4eaa6395ba014821b123f5332817aed0` under `task_a18a9f752536469faafc4d3ebec01dfd` from canonical `4aa6ce063`. I preserve PR877 parenthesized named-literal and record-first tuple parsing. I do not execute recovered worker binaries.

## My recorded static findings

I found these issues by reading the recovered source, without executing it:

- My borrow emitter initializes layout and ownership counts from records alone, then appends union rows (`nanoisa_borrows.nano:1891–1947`). It also assigns generic formal payloads TAG_INT even when its admission helper permits BOOL, FLOAT or STRING. I need complete concrete instance identities, including phantom arguments, exact tags and counted envelopes.
- My ordinary NanoISA match emitter stores the only scrutinee before testing a guard (`nanoisa_codegen.nano:1398`). A false guard then reaches another DUP without the retained value. My result-type analysis also returns from a rejected guard without restoring its lexical type/name state.
- My native statement emitter still uses unguarded switch cases, including repeated variant labels. My candidate expression chain initializes a zero result and publishes it after total failure instead of terminating.
- My resource-flow and purity match visitors inspect arm bodies but omit guard expressions (`typecheck.nano:1252–1292,6117`). My match checker does not establish ordered reachability and totality, and recognizes integer matches from patterns instead of the actual scrutinee type.
- My borrow expression-match implementation restores the incoming owner state for each arm without joining surviving output states (`nanoisa_borrows.nano:842–861`). Its union-value helper admits only locals and constructors despite broader returned-union claims.
- My new helpers lack meaningful mandatory shadows. One proposed generic guard positive uses global initialization that my borrow emitter still refuses. I preserve that fixture and distinguish its route from separately qualified VM acceptance.

These are corrective obligations, not acceptance inferred from code inspection.

## My ordered checkpoints

1. I preserve every guard's AST identity and source node type through schema, parser and visitors. I check exact BOOL in the lexical payload context, ordered reachability, and unconditional coverage. Conditional wildcard and repeated guarded arms retain source order. I account for guard effects in purity and resource analysis; owner-changing guarded paths remain explicitly refused until my separate affine join checkpoint. I evaluate a scrutinee once, retain it through failed guards, bind payloads before their guards, and emit both statement and expression arms in order. A terminal miss cannot manufacture a result. I preserve enclosing return/break/continue semantics and restore temporary compiler state on refusals. I add meaningful helper shadows. Root reviews this complete source checkpoint before any bootstrap or fixture execution.
2. I construct exact scalar generic-union instance maps and complete envelope counts, with INT/BOOL/FLOAT/STRING payloads, positional parameters/results/calls and precise unresolved/resource-bearing refusals. This remains part of the original full a18 scope.
3. I establish guarded and unguarded ownership joins, false-guard continuation state and enclosing-function control flow in the owned producer. I retain original tests and qualify fresh C seed, Stage1, Stage2, VM and native paths on Linux and Darwin. Full bootstrap and original generic/identity/refusal coverage remain required.

My first checkpoint does not admit integer matches or wildcard union arms into an ordinary NanoISA route that cannot represent them; I retain a checked refusal there while preserving their native semantics. It does not import the unqualified recovered borrow emitter. Neither this child nor a scalar example closes a18 or the full 5.1 release.

## My qualification boundary

Before execution I freeze the reviewed source and harness, record actual provider/compiler identities and preserve first terminals. I require guard false/true traces, repeated patterns and conditional wildcards, lexical payload names, once-only scrutinees, exact BOOL negatives, non-total/unreachable refusals, side-effect/purity controls, meaningful helper shadows, terminal control flow, and unchanged output on refused compilation. I retain parser877 and existing shared match-policy controls. Later checkpoints add concrete generic identities and owner joins without replacing the original acceptance suite.

## My first source checkpoint

I add guard ID/type arrays to ASTMatch and both checked-in schema outputs. My parser records absent guards as -1 and preserves actual parsed expressions for all three arm forms. A shared parser query recognizes only absent and literal-true guards as unconditional. My checker uses the actual INT/UNION domain, verifies declared patterns, rejects arms after an unconditional wildcard, and requires unconditional coverage. I bind named payloads before checking each exact BOOL guard.

I visit guards in resource flow and purity. My resource visitor compares copied ownership and held-place state before and after the real guard traversal; changed or terminal guard flow is refused in that ownership profile. My owned-union match helper explicitly retains its unguarded boundary. The existing passive scalar/parallel validators continue refusing match nodes; they do not silently omit guard dependencies. The declaration-local inventory already visits every stored block, including expression blocks.

I share one native ordered-chain emitter between statement and value matches. I evaluate the scrutinee once and establish each payload scope before its guard. Success prevents all later guards and bodies. I introduce no loop or switch that could capture a source break or continue. A missing successful arm prints a diagnostic and aborts; for statement arms that all return, the trailing backstop is unconditional so there is no synthetic returning path. I retain existing arm-value and enclosing-function return generation.

My ordinary NanoISA route still requires supported named scalar union variants. Its coverage query permits repeated guarded variants and counts literal true correctly. I duplicate before storing a payload binding, retain the original scrutinee on false guard edges, and pop it only after guard success. Result inference restores temporary names/types on a bad guard. Emission snapshots those arrays by value and restores them on refusal; successful lexical scopes retain allocated slots but hide departed binding names, as before. Existing module publication remains staged.

My meaningful shadows cover stored guard identities and absence, exact BOOL refusal, lexical named-payload guards, literal-true coverage, missing/unreachable arms, resource-state change refusal through an actual consuming guard call, guard purity effects, ordered native value/statement structure, return/break/continue, and NanoISA duplicate/guard/backstop plus rejected-guard lexical restoration. I preserve all prior assertions except explicit parser-refusal and switch-spelling expectations superseded by this checkpoint. I have inspected delimiters and the diff only; I have not compiled, bootstrapped or executed any fixture at this pin. Independent source review precedes qualification.
