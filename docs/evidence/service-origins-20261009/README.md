# I retain original service source identity

I bind origins in C `process_imports` before aliases or module declaration names,
and in Nano `compile_program` immediately after parsing, before function/type
binding. C owns up to sixteen canonical UTF-8 paths in its environment; each
retained path is at most 4096 bytes. Nano uses the merger's exact per-line file
and line arrays, retaining original locations alongside parser nodes whose
merged line remains unchanged. Failure validates before changing node indices.
Neither phase grants File authority or replaces the unresolved-service guards.

My focused build and tests pass:

```sh
make -j4 obj/test_service_origins nano_virt nano_vm
python3 -m unittest -v tests.test_service_origins
```

The C fixture exercises the actual recursive import loader, a symlink import,
same-basename distinct files, repeated bindings, environment lifetime, capacity,
missing files and duplicate/rebinding refusal. The Nano fixture compiles current
production helper code through nano_virt and the installed Stage1/Stage2, verifies
each module, and executes it. It covers original lines, reordered files, exact
owner indices, late-failure atomicity, duplicate declarations, missing mappings,
unequal maps and the seventeen-declaration refusal. Imported shadows are selected.
`tests.log` records both methods passing in 102.998 seconds.

The initial fixture run (`initial-tests.log`) exposed helper visibility and
missing direct test imports. I made the existing UTF-8 validator public and
imported parser/lexer explicitly in the fixture; I removed no assertions.

These component results do not establish a rebuilt full driver or source File
execution. Companion acquisition, complete namespace/nominal propagation, both
lowerers, generated shadows, grants and staged publication remain #989 work.
