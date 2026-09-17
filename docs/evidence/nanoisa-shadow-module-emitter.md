# My bound-Parser shadow module emitter

`nanoisa_emit_shadows_nasm(parser, first_shadow)` lowers the selected shadow
suffix from my caller's merged and checked Parser. It preserves the caller's
module bindings and the Parser. Each selected body becomes a separate void
function through my existing function/block lowering. A synthetic entry calls
those functions in declaration order and returns zero. Synthetic names avoid
all existing function declarations; an ordinary user `main` remains callable.

I initialize globals once and retain their shared state across selected
shadows. Every global initializer and every call reached from selected bodies
roots the same executable function queue used by my program emitter. I retain
recursive and owner-bound imported calls. Unreachable unsupported functions
are omitted; reachable unsupported operations reject the module. Type bounds,
string constants and exact host imports retain my existing assembly contract.

An empty suffix produces an entry with no shadow calls and still retains
global initialization. A negative index or an index beyond the shadow count
rejects. My driver can choose not to request execution when no shadows are
selected; this API does not change that policy.

Five focused methods assemble and verify actual modules in NanoVM. They check
shared mutable globals and initialization order, separate local scopes,
callable user main, failing assertions, suffix/empty/invalid selections,
recursive same-name functions in two bound modules, synthetic-name collisions,
and reachable unsupported-operation refusal. The integrated emitter gate
passes 86 C-seed/self-hosted bytecode comparisons and 69 Python methods,
including these five.

This is an emitter API, not a driver cutover or a security sandbox. My normal
canonical driver still generates and executes native C shadows. A supervised
VM runner must retain the parent deadline and completion protocol, including
rejection of an early `exit(0)`, before that route can change. I do not treat a
bare successful VM process exit as sufficient shadow completion evidence.

The task is `task_57fc2c62eb504a579dc2ca37ce72e2e8`. Integrated output is retained
in `/tmp/nanolang-shadow-emitter-gate.log`.

## My full compiler-shadow boundary

I built a temporary copy of `src_nano/nanoc_v06.nano` whose assembly request
alone calls this shadow API with the existing `first_shadow`. My product
driver remains unchanged. With `NANO_AS_CAPTURE_HELPER` explicitly pointing
to this checkout's `bin/nano_as_capture.so`, the full checked compiler source
rejects before publication:

```
I cannot lower this checked program: I cannot lower shadow __nano_module___7_tokenize_string at merged line 2381: statement outside the pinned subset
```

`src_nano/compiler/lexer.nano` uses a range `for` loop in that shadow; my emitter
has no `PNODE_FOR` lowering. I record the required continuation as
`task_ef6adaee5e3644c8a8218ede4b01e6b3`. The retained probe source, binary and
log are `/tmp/nanolang-full-shadow-probe.nano`,
`/tmp/nanolang-full-shadow-probe`, and `/tmp/nanolang-full-shadow-probe-run2.log`.
The probe exits 1 without producing its `.nvm` output. Its earlier invocation
omitted the capture-helper setting and also reported host capture refusals;
that setup error is not the final product evidence.
