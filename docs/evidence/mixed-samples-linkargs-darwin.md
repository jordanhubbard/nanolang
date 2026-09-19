# I pass resolved linker flags to my mixed Samples fixture

I qualify [PR820](https://github.com/jordanhubbard/nanolang/pull/820) for
`task_61a21d598620c6ba3a676624e01de9fc`, a child of
`task_793523b9e4c04e389f9d63652758991a`. My code-qualified head is
`9578eec074da017afed3cb2916cd6e168a5db9f3`, based on canonical PR819 at
`d0de3d23a730c66531f2ed2c5d972215302ebe19`.

Before publication I merge canonical PR821 at
`0955dca62e168c212e1c03c844651e2040286995` through integration commit
`57824ba7dfd703dbe7b0003bc1a1a40b11e70dbc`. That change adds service
providers to the shared object closure and refuses mixed preparation only when
service bindings are present. My fixture supplies no service bindings, and
PR821 separately qualifies the combined service/mixed boundary. I therefore
retain the exact `9578eec` runtime evidence instead of relabeling its manifests
as measurements of the integration commit.

My Make target now passes its resolved `LDFLAGS` through the dedicated
`MIXED_SAMPLES_LDFLAGS` environment boundary. The Python fixture splits that
value into exact linker arguments. Direct invocation without the variable still
uses `-lm -lcrypto`. I change no production source, assertion, sanitizer,
ownership rule or deadline.

## What I tested

In a fresh Darwin checkout with no `obj` directory, Homebrew LLVM23.1.1 runs my
mixed query, runtime, allocation and admission targets in22.68seconds. They pass
112,482 composition checks,2,006 lifecycle checks,13,358 heap checks across488
budgets and440 injected failures, and127 admission checks. The Make invocation
uses its resolved `-lm -lcrypto -lffi -L/opt/homebrew/opt/openssl@3/lib` flags.

I separately omit `MIXED_SAMPLES_LDFLAGS` and run the ordinary fixture default.
The existing Homebrew OpenSSL directory is supplied through `LIBRARY_PATH`,
without changing project or host configuration. The fixture passes112,482
checks in1.14seconds using its unchanged `-lm -lcrypto` default.

My [manifest](mixed-samples-linkargs-darwin/manifest.json) names the commands,
counts and boundaries. I retain full hash maps for5,464 tracked sources,10
affected inputs,148 regular build-directory entries,26 actual query link
objects, five selected products and six actual host tools. Source,
affected-input, resolved-flag and host-tool maps agree before and after
execution.

## My evidence precision correction

The original sealed report calls the148-entry after-run map an object count.
That label is wrong. The unchanged map contains144 dependency `.d` files,
three fixture executables and one stamp; it contains zero `.o` files because
the map recorded regular build-directory entries and omitted symlinked object
paths. I retain that report and its hash unchanged, but I make no object-input
coverage or before/after identity claim from this map.

The separate26-entry link map contains the actual symlink-resolved `.o` inputs
used by the mixed query and was independently verified. The five selected
product hashes cover `nano_vm`, `nvm2c` and the three mixed runtime fixture
executables. I do not add a current runtime-object snapshot: it would describe
a later filesystem state and could not establish retroactive before/after
identity for this qualification.

## What I preserve

The earlier Darwin gate at `64db8f922e7a24e5e4b86794f66fe3d1f0c076b9`
remains separate evidence. Its three STRING source methods pass in278.10seconds;
its six-target runtime invocation stops because the mixed fixture cannot find
`-lcrypto`. I do not relabel that first runtime result, replay its failed
artifact or attribute current runtime evidence to its source-producer pin.

The current focused gate includes PR819's public mixed expectations. It does not
repeat bootstrap or the three source methods, close parent793523, establish full
product acceptance or authorize a release.

## What remains before parent793523 closes

PR820 must land in canonical `main`, and the child task must reconcile against
that actual merge. Parent793523 then needs the same canonical-ancestry
reconciliation tying its separately sealed64db source3 result to this current
runtime correction. I find no additional bounded STRING-field test gate left by
this fixture defect: root confirmed that PR819 does not widen the source
producers and authorized preserving the earlier source/bootstrap evidence.
Arrays, complete managed LLVM/Wasm transport, full ownership parents, whole
product acceptance and release publication remain separate open work.
