# Independent Nano File driver publication

I connect my independent Nano source lowerer to the shared publication
transaction under #989, from parent `a44a5423f` on Darwin. I keep 5.1 open.

## Boundary

I retain parsing, source identities, namespace, nominal/ownership checking,
shadow selection, byte lowering and serialization in Nano. My new owned opaque
publication context receives lowercase hex chunks of emitted bytes and copied
origin/name labels only. It has no AST API. I cap chunks, total retained bytes
and selected modules, poison the context after staging failure, and allow one
publication attempt. Every normal return frees the context. Source and immutable
companion aliases are checked in the Nano driver before publication.

I reuse the qualified byte-only C publication transaction after independent
lowering: complete module validation, selected shadows with fresh explicit
grants, staged native/bytecode output, and cleanup before atomic publication.
`--allow-temporary-files` authorizes compile-time shadows; native programs require
a separate runtime flag. I preserve ordinary default module output. File graphs
without explicit output default to native `a.out`; explicit `--emit-nvm` remains
bytecode. Unqualified File output options refuse before execution/publication.

I register exact opaque artifact signatures in both my Nano emitter and native
translator. Missing opaque declarations and wrong signatures refuse. During this
work I found native typed-artifact argument scratch had three slots despite
existing four/five-argument catalog/snapshot hosts. I size it from the declared
signature parameter array and check the bound before writing slots.

## Verification

- My C-produced updated Nano compiler module builds with all selected shadows
  (`final-build2.log`). This includes new byte transport/lifetime shadows and
  positive/negative artifact signature controls.
- `final-cli.log`: four actual Nano CLI methods pass in 75.983 seconds with that
  compiler in VM and native form. The methods cover identical emitted File bytes,
  all five unchanged generated shadows and complete selection/start/completion
  records, real VM/native File results, separate runtime grants, missing compile
  grants, native compilation failure, malformed bodies, imported failing shadows,
  root-only execution with unselected-body checking, source/companion hardlinks,
  prior output, private staging cleanup, invalid deadlines, directory destinations,
  PATH invocation and default native publication.
- `self-build.log`: the updated Nano compiler compiles its own current source,
  including the new artifact ABI, to another compiler module with its selected
  shadows enabled. `self-cli.log` repeats all four actual CLI methods through this
  Nano-produced compiler in VM/native form; they pass in 73.931 seconds.
- `c-neighbor.log`: all four existing C source-driver methods pass (19.043 seconds).
- My byte bridge allocation/refusal harness passes under LLVM ASan/UBSan, with
  allocation-prefix failures, poison-after-malformed-hex, bounds, incomplete
  selections and single-publication controls. I use a publication stub here to
  isolate staging; the actual CLI corpus uses the real transaction/consumers.
- `artifact.log`, `final-translate.log`, `self-translate.log` and
  `final-checked.log`: instrumented native translation accepts compiler modules
  containing the exact opaque and four/five-argument artifact calls. Only
  `nvm2c.c` is instrumented in that tool; other linked objects are ordinary builds.
  Empty logs mean successful commands without diagnostics, not extra assertions.
- `inputs.json` pins implementation/test sources and retained compiler artifacts.
  Native compiler products link the existing host runtime object; they are not
  fully sanitizer-instrumented compilers or an installed-platform qualification.

## Retained first failures

My first Nano module used reserved local names `byte` and `shadow`; `driver.log`
and `driver2.log` retain parser refusal before output. `driver3.log` records the
corrected first build. The first CLI run (`cli.log`) exposed test setup errors:
its output replaced its own temporary compiler wrapper, and my hand-linked native
compiler lacked the normal host runtime object needed by its std artifact.
Separate wrapper paths and the standard host runtime link fix those controls;
`cli2.log` records the passing run before final signature additions.

The older development compiler refuses the new artifact symbol in
`second-producer.log`. I add its exact ABI to the new compiler and seed that
updated compiler through C before the successful Nano self-compilation. My first
new negative ABI shadow then catches the missing opaque declaration guard
(`final-build.log`); I add the guard and retain the passing corrected build.

I normalize trailing log whitespace for repository checks. I do not infer a
release fixed point from the C/Nano CLI tests. Full source/profile coverage,
multiple catalogs, richer/indirect borrows, Linux/Darwin installed execution,
Socket/public network work and the remaining 5.1 gates still apply.
