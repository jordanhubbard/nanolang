# My imported and branch-local affine facts receipt

I qualify #998 from `74a565150` using the fresh raw-bootstrap Stage 1/Stage 2
providers pinned in `providers.json`. My changed C producer rebuilds only
`src/nanovirt/codegen.c` and links cloned, hashed compatible aggregate object/archive
inputs. `build.py`, `build-commands.json`, `build-inputs.json` and
`build-outputs.json` retain that exact isolated build. I do not overwrite shared
providers or run another bootstrap. An initial recipe-reader script failed on
Make's continued command lines before compilation; I retain that setup failure.

## Actual failures and narrow correction

`initial.log` records three methods in 4.951 seconds with two failures, both C
import orders: the module contains `main,take,run,take,run`, and canonical
assembly rejects duplicate function symbols. Fresh Nano Stage 1 and Stage 2
pass the same imported cases; all three producers pass branch-local/refusal
cases. `original-reproducer/` retains the actual C source/module/dump from a
bounded reproduction. I do not remove the roundtrip assertion or rename source
functions to conceal the collision.

I reuse the ordinary import producer's existing `imported_wire_name` helper in
the C borrowed producer. First available names stay unchanged; collisions get
unused indexed wire aliases. Source AST lookup, function ordinals, numeric calls,
layouts and advisory indices do not change. Existing allocation failure enters
the compiler error path; assembler duplicate-symbol refusal remains intact.

My first edit used `source->line` unconditionally. Synthetic shadow slot zero has
no source function; three C positive cases crashed with SIGSEGV at address 4.
I retain `synthetic-shadow-crash.log`; independent review confirms the missing
conditional. The final call uses `source ? source->line : 0`. I add an explicit
resource-shadow success/failure control with prior-output preservation. Future
fixture failures copy their artifacts outside TemporaryDirectory automatically.

Before running, I correct an initial test-design assumption using existing code:
C provenance 1 names original imported paths; Nano provenance 2 names the flattened
compilation-unit path. Function-table indices remain the primary site identity;
path and binding spellings are descriptive, not nominal runtime authority.

## Final acceptance

`final.log` records all six methods passing in 6.694 seconds through the changed
C NanoVirt and both fresh retained Nano generations. They cover both import
orders, distinct reachable function/site identities, source-coordinate provenance,
both conditional arms with checked results, ownership misuse refusal, resource
shadows, original VM execution, O2 ASan/UBSan native execution, byte-exact canonical
roundtrips and unchanged native code/VM behavior after stripping advisory entries.
The two existing owned/borrowed metadata methods remain green.

`comparison.log` additionally establishes byte-identical old/new C output for the
existing noncolliding shared-borrow fixture. For the actual colliding module,
function ordinals/signatures, numeric instructions and decoded metadata payloads
are unchanged. Only wire names and their string-pool storage differ; `fixed.nasm`
retains the corrected canonical text. Provider hashes remain unchanged after
qualification.

I leave #998 open. This closes the demonstrated imported wire-name collision and
qualifies this bounded site slice; it does not add a path-summary format, prove
all generic/capture/projection coverage, or replace exact hosted platform gates.
The C edit requires aggregate build/qualification after integration. The Nano
compiler sources are unchanged from the passed raw bootstrap.
