# I design bounded original-slot rendering for reassignment

I retain the unchanged reassignment anchors' ordinary Nano preflight refusal at a static281-local estimate. I do not raise128, remove assertions, substitute a smaller fixture or treat an estimate as producer acceptance. This separate representation proposal applies only to Nano output of the new proven reassignment profile. C and all prior profile bytes remain unchanged. The original analyzed ownership/control tree stays immutable and authoritative.

## I map a finite rendering plan

I propose a checked rendering plan over the original structured statement lists, not general register allocation. Every source IR statement is accounted for exactly once, either emitted directly or consumed by one explicit same-list pattern. Patterns may not cross branches, loops, pretests, reference regions or unrelated statements. Whole-function def/use counts refer to exact slot/Expr identities, never source names. A failed pattern keeps the existing rendering; failure to fit128 remains an atomic refusal.

| Original-slot rendering form | Observed candidates in the unchanged true/0 anchor | Projected local reduction |
| --- | --- | --- |
| Exact scalar constructor bundle: two original scalar stores, two sole-use LOAD snapshots, scalar-leaf pack. Each scalar slot has exactly one syntactic store/read; each RHS is a constant or already retained scalar temporary. I keep left-to-right field order and feed these exact values into the constructor's existing staging. | Eight bundles,16 original scalar slots and16 snapshots |32 |
| Pack followed immediately by its sole consuming first owner store: emit one typed mutable owner initialization containing the constructor. | Nine |9 |
| Owner move followed immediately by its sole consuming first store or assignment: emit the destination transfer directly. Source and destination slots must differ; otherwise the checked source's consumed-destination rule still needs the intermediate move. |18 first stores,6 assignments |24 |
| Destructure outputs that are scalar and unused: keep the full original-position destructive pattern and its mandatory fields, but omit the later redundant copies from those pattern fields into never-used scalar temporaries. No child origin, pattern binding or consumption point disappears. |18 scalar result copies |18 |
| Scalar owner observation immediately followed by its sole original scalar store: assign that original slot directly from the same read at the same point. I keep the complete observation and its order. |18 |18 |
| Local LOAD snapshot immediately followed by its sole use in a total scalar assertion expression: substitute that exact local leaf. I retain the assertion and allow only comparison/Boolean/NOT/local/constant leaves; no resource read, arithmetic, call or effect is moved. |10 |10 |
| Sequential guarded ADD uses one private per-function typed bank `a,b,low,high,boundary`, plus one lasting result local per original ADD. I retain all eight distinct result lifetimes. | Eight ADD expansions:56 locals become5+8 |43 |

These disjoint candidate counts project154 fewer locals,281→127. The residual estimate is21 original scalar slots,7 LOAD snapshots,18 constructor staging locals,29 first owner declarations,2 unfused owner move temporaries,34 pattern staging/result locals,13 ADD bank/result locals and3 call-result temporaries. This is a feasible estimated envelope, not a proved minimum and not ordinary regenerated acceptance. I must independently count the false/two-iteration variant and compare all estimates with actual regenerated function local counts.

The parser accepts pattern field names but has no field-renaming syntax (`parse_owned_pattern`). I therefore do not invent direct renamed child bindings or erase their staging. My retained34 unpack locals include the full existing nominal patterns. A broader reconstruction of source patterns/lexical scopes is not part of this proposal.

## I preserve exact owner transfers

The origin checker runs on the untouched analyzed tree before rendering. A direct move/pack-to-destination plan carries the exact original origin token and identifies both original statements it replaces. It has one consumer and cannot leave an alias or revive a dead source. Child origins and order remain exact. The plan does not use nominal-layout equality in place of origin identity, merge generations, create uninitialized resources or extend a borrow lifetime. Removing unused scalar result copies leaves the destructive operation and its failure/cleanup position intact.

Declaration dominance remains separate: first-store plans declare; later assignments target the existing mutable declaration. A constructor plan can remove only the proven sole-use scalar slots whose initialization and reads it covers, and must show no remaining read/store in the emitted function. It does not remove guest-visible writes or effectful expressions. Per-iteration constructor bundles remain inside their original loop body, preserving zero iterations and reevaluation.

## I bound reusable scalar scratch

The bank is private to one emitted function, never global or shared with its callee. Constants low/high and mutable a/b/boundary are initialized in that function's entry scope. At each ADD I assign a from the original left operand, then b from the original right operand, before any guard or result write. Every guard reads only the new operands and constants. Each positive/negative boundary is assigned on its dominating arm before use. No operation depends on the bank's value from a previous call or iteration.

Each ADD keeps its own original result variable, initialized and assigned on every existing guarded branch, so a result live across another ADD is not clobbered. There is no recursive/nested bank use within an operation: admitted arithmetic is already a direct statement-boundary ADD with retained scalar operands, and guards contain no helper calls. Branches, loop bodies and materialized pretests run complete sequential operations; no operand or unfinished result may cross a bank reuse. The existing signed-wrap formulas and operand order remain exact. The structural helper-work proof stays conservative; its256 charge is not reduced because scratch locals are reused.

## I require first acceptance before expansion

Before a broad corpus, I require both unchanged existing source variants through ordinary CLI reconstruction and original producer regeneration, exact estimated-versus-actual entry/helper/shadow local counts, original verification and execution. Any result above128 remains a blocker with prior output preserved; diagnostic bypass is measurement only. Full original assertions, zero/two iterations, final field values, aliases and successful owner cleanup remain the semantic oracle.

Focused controls must retain intermediates for multiple consumers, source==destination moves, nonadjacent effects, slot writes between snapshot/use, branch-only initialization, wrong nominal children and owner generation replacement. Scratch tests cover MIN/MAX wrap, operands using a previous ADD result, two operations in each branch, loop/pretest reevaluation and final-false effects. Original source shadows and generated representative consistency shadows remain distinct. All old-profile byte goldens must remain exact. No production rendering change is authorized merely by this design document.

## I retain reviewed implementation evidence

The parent review approved exactly these seven forms before implementation. My [component receipt](evidence/owned-reassignment-20261019/README.md) records ordinary127/9 for both unchanged anchors and60/9 selected shadows, plus128 acceptance and129 atomic output refusal. The earlier281 number remains a static refusal estimate. The pre-qualification review corrected18 scalar copies that the first pruned renderer still emitted; actual source omission and regenerated counts now agree. Fresh aggregate/platform qualification remains separate.
