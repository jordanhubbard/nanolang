# I parse record-first tuples without changing literal identity

I qualify parser source `2132eaba802a3a2a296d654424ed7659b1dafb16` on Linux
ARM64 for `task_d9730c3ab71e45b283e3c38094638d47`. An uppercase named literal
inside parentheses uses my declaration-aware literal selector before my existing
comma/rparen logic decides tuple versus group. Other heads retain expression
parsing, and prefix calls retain their existing route. I add no ownership or
runtime admission.

My first candidate `bbafb83f4` bypassed that selector for qualified names.
Independent review identified the changed record/union identity. Its fresh
bootstrap stopped after 65.219 seconds at the existing ambiguous-union shadow,
with status 2 and no Stage 1 publication. I retain that log and input maps;
I do not replay or count the failed candidate as acceptance.

The corrected source passes:

| Gate | Result |
|---|---|
| Fresh ordinary compiler bootstrap | PASS, 283.921 seconds |
| C-seed, Stage 1 and Stage 2 compiled parser assertions | PASS on all three routes |
| Existing owned-wrapper refusal method, with original record-first case restored | PASS, 187.688 seconds including setup |

The AST fixture retains its original grouped infix, tuple, prefix-call,
qualified-call, malformed-expression, import, loop-control and unary checks.
Three added cases cover a plain record-first tuple, a grouped record and a
qualified record-first tuple; tuple element kinds and complete consumption are
asserted. Mandatory helper shadows and the existing exact/ambiguous union
shadows remain enabled by the bootstrap.

The refusal method checks six wrapper cases against `nano_virt`, both canonical
self-hosted drivers and three freshly built source emitters. The original
`(Leaf { value: 1 }, 2)` resource tuple must reach semantic refusal, preserve
previous output, and report neither a parse error nor an unexpected token.
This is 36 checked refusals, not owned tuple execution or complete affine
acceptance. No test selection, shadow, timeout or expected refusal was weakened.
The external wrapper only retains the existing test's temporary directory.

`reports.json` hashes 36 retained reports; `artifacts.json` records 63 retained
files, including compiled stages, AST executables and the test setup artifacts.
Before/after source maps match in all three runs. The four compiler binaries
remain byte-identical from corrected bootstrap through every acceptance phase.
These ordinary parser gates do not claim sanitizer coverage, Darwin execution,
full product acceptance or a NanoISA-only compiler fixed point. Qualified and
failed trees remain separate from this documentation integration.
