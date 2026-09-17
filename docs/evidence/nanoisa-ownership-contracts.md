# My reference parameter and root declarations

I continue task_dcec6dcfb3d6442fbb9db131c7ebc0a3 under affine IR parent ed702.
I retain declarations needed by future lifetime verification; I do not admit
borrowed execution from those declarations alone.

My OWNERSHIP section (13, required feature bit 8) records complete/resource
layout flags and exact parameter, local and result descriptors. It requires
retained layouts. I validate counts against function/layout tables, parameter
and result tags against signatures, exact nominal indices, transitive resource
classification and my initial finite scalar/record bounds. Shared/exclusive
modes occur only in parameter slots and require fixed scalar-field resource
referents. Reference result and stored-local declarations are refused.

I copy the bytes through both bridge directions and preserve canonical
`.ownership` chunks. Ordinary declarations can reconstruct through the
verified assembler. Resource/reference declarations remain inspectable by
non-executing codec APIs, but verified assembly, VM execution and native
translation refuse them. My first test caught a missing direct nvm2c guard;
I added it before publication. My VM's checked fallback does not implement
reference semantics either, so direct function, invoke, callable, core and
linked-module entry paths also refuse these contracts. A prior native output
survives refusal.

I passed 62 focused declaration/transport checks normally and under focused
ASan/UBSan. The same ordinary typed-root artifact prints `42` in NanoVM and
native translation. I passed existing codec/bridge/end-to-end gates, 2,691
NanoISA checks, 75 place checks and the retained-layout gate. The integrated
VM suite passes 272,612 checks, including direct API refusal and unchanged
activation/result state. The native translator gate passes 2,412 AOT checks
and 1,092 shape checks.

I also exercise the genuine canonical host-module path:

```text
make build
NANO_MODULE_PATH=modules bin/nanoc_c src_nano/nanoc_v06.nano -o /tmp/nanolang-reference-metadata-seed
/tmp/nanolang-reference-metadata-seed --help
NANO_MODULE_PATH=modules /tmp/nanolang-reference-metadata-seed examples/language/nl_hello.nano -o /tmp/nanolang-reference-metadata-hello
/tmp/nanolang-reference-metadata-hello
NANO_MODULE_PATH=modules /tmp/nanolang-reference-metadata-seed --emit-nvm examples/language/nl_hello.nano -o /tmp/nanolang-reference-metadata-hello.nvm
bin/nano_vm --verify-only /tmp/nanolang-reference-metadata-hello.nvm
bin/nano_vm /tmp/nanolang-reference-metadata-hello.nvm
```

All commands pass with ordinary default shadow checks. The seed initially
consumed the exact retained-layout host-manifest hunk from c88a0aa2; after
that repair merged, I restacked onto main b09a16a8 and dropped the temporary
dependency. Both final ownership/reference source dependencies are in the
host manifest. The canonical `.nano` sources did not change during this
restack. These checks do not claim another full compiler fixed point.

My two NanoISA frontends still refuse source borrows. Producers must establish
these declarations from resolved source types; instruction verification must
prove live ownership, provenance, argument-order holds, joins and non-escape;
the VM/native path must then implement genuine references. The declarations
alone establish none of those properties. Full718, ed702 and publication
acceptance remain open.

## My explicit host-source inventory

I searched tracked build consumers of the verifier and v2 bridge. I found
five independent lists and retained all new transitive dependencies:

| Consumer | Required path |
|---|---|
| `Makefile.gnu` | NanoISA tool/runtime object group |
| `modules/nanoisa/module.json` | Canonical compiler host facade |
| `modules/forth_see/module.json` | Native Forth SEE imports |
| `examples/Makefile` | Separately built Forth SEE shared library |
| `src/nanovirt/wrapper_gen.c` | Regular and daemon wrapper object list |

The last three paths were missing retained-layout support from PR535. A normal
Forth SEE import on that source fails to link `nvm_layouts_have_facts`,
`nvm_retained_layouts_valid` and `nvm_retain_layouts`; the log is retained as
`/tmp/nanolang-forth-see-before.log`. Task_f27d1997764b4c19a5cc0f2705ec37bf
records the closure repair before implementation. I add retained-layout,
reference-place and ownership-contract sources to all required lists.

Five wrapper link tests and seven adjacent publication methods pass. Two
additional gates compile and execute a native Forth SEE import with a fresh
private build cache, then build and load the examples shared library and call
its real host function. They pass without changing the ABI or reference
refusals. This is source-closure evidence, not new Forth language acceptance.

Float tags can be retained in my declarations. Actual native float-record
lowering remains task_93574cf9d200459aa16e959baf68201d and must pass before
that borrowed reference case is enabled. My ordinary runtime fixture here
uses integer fields; it does not establish float-record runtime support.
