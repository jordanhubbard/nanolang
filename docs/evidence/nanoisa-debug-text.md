# Canonical DEBUG transport evidence

I retain accepted DEBUG triples in their original order through canonical text,
including duplicate and unordered offsets, zero columns and full unsigned
32-bit values. I retain an explicit empty DEBUG section through v2 conversion.
Unknown advisory metadata keys keep their exact bytes; unsupported wire section
types remain refused. These records do not grant executable authority.

My final implementation and test pin is
`940195d4cb4cf42c614720912dd18d34c6e09fd9`, integrated with main PR597. My fresh
three-stage bootstrap passed with compiler source `9a0f4ef8`; later changes add
assembler allocation-path tests, stronger source assertions and main's C native
optional-array-read repair. My NanoLang compiler source is unchanged afterward.
The bootstrap exercises actual canonical host-module linking and installation.

| Gate | Observed result |
| --- | --- |
| DEBUG text, empty presence, strip and legacy controls | 53 checks pass |
| Local-name metadata with original wire equality | 123 checks pass |
| Direct append and v1/v2 allocation failure cleanup | Pass |
| Both assembler DEBUG append failure exits | Pass |
| Existing local-name append / marker allocation boundaries | 5 / 20 pass |
| These focused C suites under ASan/UBSan with leak checks | Pass |
| Paired ordinary source production | 2 methods, 6 C seed/Stage1/Stage2 artifacts pass |
| Exact original canonical v2 bytes and VM/native results | Pass for all six artifacts |
| Existing canonical text / v2 conversion / whole-module tests | 210 / 365 / 43 pass |
| Existing advisory transport / conversion allocation controls | 188 / 18 pass |
| Integrated optional native array reads | 3 methods pass |

My C-source controls explicitly require nonempty DEBUG tables before comparing
complete serialized bytes. The Stage1/Stage2 controls preserve whichever debug
records their producers supply; this transport change does not add source-map
production to those frontends. I also reran the original retained C-source file
that exposed the loss, `/tmp/nanolang-local-binding-c.nvm`, through the stronger
wire-equality probe and under sanitizers.

My append API returns false without adding a record on allocation failure.
Assembly reports a memory error and releases the partial module; v1 loading
returns NULL and v2 conversion releases partial state and uses its existing
INDEX_RANGE allocation-error convention. Existing source producers may ignore
the checked result as before; their general allocation policy is outside this
bounded transport evidence.

I keep full source reconstruction and other producer families open. I do not
claim opaque unknown-section transport or equality for every possible module.
The bounded task is `task_1466d452d48c4a558c3be8a51765dd8f`; its contract is
`../NANOISA_DEBUG_TEXT.md`.

I retain Linux ARM64 logs under `/tmp/nanolang-debug-text-`: `bootstrap.log`,
`focused-final.log`, `adjacent.log`, `integrated.log`, `source-debug-gate.log`,
`asan-integrated.log`, `original-repro.log` and `original-asan.log`. The earlier
failed original-byte comparison remains in the prior local-name evidence.
