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
