# My canonical component shadow completion contract

I qualify task `task_14c8ecbd8aaa484ea5e73f2aa43fa48b` at frozen product
`e9a5f55f9b82b3de8c03b3df240665865fd25e5c`. My earlier C-seed report and
static declaration inventory do not establish canonical shadow execution.

I use a separate checkout and fresh bootstrap. I record the installed Stage2
compiler, VM and assembler identities, source hashes and clean checkout state.
I explicitly compile each unchanged parser, checker and transpiler driver
with `--emit-nvm --test-imports --verbose`. Through the supported `NANO_VM`
hook, I copy the already verified shadow module before delegating the exact
`--check-shadows` arguments to the built VM. I retain the VM exit status and
module hash. I neither change the shadow deadline nor bypass verification.

After successful publication I disassemble the retained module and inspect its
entry calls to establish selection, then execute the published driver. These
are aggregate shadow-suite completion and bounded driver assertions. A count
of selected calls does not prove unique per-source provenance or statement
coverage. Imported dependencies overlap among suites; counts are not additive.

My outer bootstrap/build limit is 1800 seconds, each compile is 1200 seconds,
and each published driver has 60 seconds. I stop on the first failure and
preserve all logs. The VM keeps its own normal shadow supervision. I verify
source and tool identities afterward and report only completed phases. I keep
the broader component audit and release parent open for their remaining scope.

## My completed qualification

I execute the pre-recorded contract at documentation-only `4cbffa9e36eb9340fc6b12238e601ffbd06d9878`, over product `e9a5f55f9b82b3de8c03b3df240665865fd25e5c`. Fresh bootstrap and tool build pass. I select the installed Stage2 compiler explicitly; all three compiler invocations, supervised shadow suites and published driver entries exit zero.

| Driver | Ordered selected shadow calls | Compiler seconds | Supervised VM seconds |
|---|---:|---:|---:|
| parser | 315 | 119.839 | 0.225 |
| typecheck | 513 | 467.167 | 0.853 |
| transpiler | 505 | 469.682 | 0.789 |

My [selection report](evidence/canonical-component-shadows/selection.json) checks
the module entry is exactly the ordered calls to every numbered shadow wrapper,
followed by return zero. Each retained VM receipt reports successful ordinary
`--check-shadows` execution before output publication. Expected diagnostics from
negative shadow assertions appear in compiler logs; every supervising process
and subsequent driver entry succeeds. Counts include overlapping dependencies,
so I do not add them as unique tests or claim statement coverage.

My [manifest](evidence/canonical-component-shadows/manifest.json) records unchanged
C-seed, Stage1, Stage2, VM and assembler hashes and clean unchanged source. I
retain the exact [qualification runner](evidence/canonical-component-shadows/qualification.py)
and [capture hook](evidence/canonical-component-shadows/capture-vm.py), compiler
logs, VM receipts and selected entry assembly. Full verified modules and full
dumps remain under `/tmp/nanolang-canonical-component-shadows`; their hashes are
recorded. I do not claim hermetic relocation of compiler host libraries.

This report qualifies the named e9 product source. Later main changes require
their own gates. It does not supersede that product's independent affine-example
quick-gate failure or establish release readiness. Alongside the static component
inventory, explicit entry assertions and C-seed completion report, it supplies
the previously missing canonical component-shadow execution evidence.

## My independent review and original task closure

An independent read-only review at report head `f9f98d7e` rehashes retained logs,
five tools, driver sources, capture hook, all three shadow modules/full dumps,
and published driver modules. It resolves every selected entry index to the
numbered shadow wrapper and verifies normal VM delegation and exit propagation.
No scoped blocker is found and no test artifact is re-executed during review.

Together with PR691 entry assertions, PR711 inventory and PR716 C-seed reports,
this satisfies the original component execution task
`task_56a065134a6e4394ae5c307c05e9597d`. Its historical one-argument lexer-call and
skipped-import concerns are addressed at the recorded sources. This closes that
bounded evidence task, not general compiler correctness or release acceptance.
