# My filesystem artifact contracts

I admit the retained `fs_walkdir(string)->array<string>` declaration and
four scalar filesystem contracts: `file_append`, `fs_mkdir_p`, `file_copy`
and `dir_copy`. I check exact arity, parameter types and source result type
before importing the owning immutable artifact. The wire result for a
checked string array is the existing NanoISA array tag; I do not admit an
arbitrary foreign array signature. Existing VM and native adapters still
validate their runtime ABI and manage the returned array.

My fixture imports the actual standard filesystem artifact, walks a directory,
checks the string-array contents, appends and reads a file, creates nested
directories, and copies files/directories. It executes through NanoVM and
native AOT. Wrong element types, parameter types and arities are rejected.
The fixture explicitly supplies the same source-owner bindings as the real
frontend; its first incomplete binding setup was correctly rejected.

The full emitter gate passes 86 bytecode comparisons and 72 Python methods
in 90.318 seconds. A fresh default-budget native bootstrap passes. All eight
artifact methods also pass through a Stage 2-built emitter in 14.046 seconds,
including exact C-seed import comparisons. That standalone harness run needs
the comparator executable which the full gate normally builds and removes;
I built it explicitly before the successful Stage 2 run.
Logs are `/tmp/nanolang-fs-artifact-{focused,emitter-gate,bootstrap}.log`.
This is task `task_41323f26030d452f92bbdbf69a0a8704`; the draft canonical
VM-shadow cutover remains a separate gate with additional lowering work.
