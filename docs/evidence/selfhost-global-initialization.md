# My global initialization lowering

On 2026-09-16 I added owner-local global storage to my self-hosted NanoISA
emitter. Both whole-source and executable APIs emit ordered initializers in
`__init__`, using the existing VM/AOT initialization convention. I register every
global before lowering and preserve parser declaration order. Initializer calls
are executable roots even when main never reads the initialized binding.

I use `LOAD_GLOBAL` and `STORE_GLOBAL`, preserve declared initializer types,
prefer lexical locals, and distinguish globals from different merged-source
owners. Assigning an immutable binding, using an unsupported global type or
colliding with my generated `__init__` refuses output. Record field access uses
the existing expression type information for global and local objects.

`make -j8 test-nanoisa-src-nano` passes 86 checks and 21 Python integration
methods. My scalar/string-array fixture adds 12 cross-compiler bytecode checks
and executes under both NanoVM and strict C11 AOT. It tests ordered dependent
initializers, exactly one printed initialization effect, mutable strings and
arrays, and local shadowing. My aggregate fixture adds 14 bytecode checks and
executes in NanoVM, covering flat-record fields and mutable map contents.
Program-mode tests additionally retain an unread global's effectful initializer,
check the empty string-array tag, and execute distinct same-name module globals.

I do not claim aggregate global AOT support: nvm2c currently refuses map access
after `LOAD_GLOBAL` and aggregate/map storage. Task
`task_95796f5f49564ed4a911fd05a1aac5b4` records that required transport and
lifetime work. Imported global aliases remain under
`task_2713a842846b417fbfd6aa4b8059d0dd`; I do not guess another owner's
binding. C-seed constant map constructor context and empty string-array global
tags remain under `task_026e73d59e9e45b0b732b43883feea9a`. The parity
fixture uses a typed map factory and a nonempty string array; my separate empty
array test requires the correct string tag. Array-element result inference is
recorded as `task_1c4da3f9cf804bbf95005ad3f603ef26`; the fixture uses an
explicit typed local for its string comparison. None of these boundaries closes
full compiler emission or matching Stage 1/Stage 2 bytecode.

A fresh C-seed-hosted canonical driver, copied only to select the new program
API pending its separate integration, reaches `unsupported local type
MergeResult` when emitting `src_nano/nanoc_v06.nano`. That record contains
scalar and scalar-array fields. This is the next measured compiler blocker;
the probe did not publish a compiler module. The temporary source copy is
removed and no canonical driver change is included here.
