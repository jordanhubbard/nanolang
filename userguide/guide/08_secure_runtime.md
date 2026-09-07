# Secure Runtime

I compile source to C and to verified NanoISA. I also host services.
Those are not the same job. The language is how you write a function.
The runtime is how that function is allowed to touch the host.

Start with the language chapters if you have not yet. This page is the
map of the runtime that 4.1–4.5 added on top of the 4.0 bytecode
contract.

I do not claim a kernel. POSIX is the host. I do not claim GNU Emacs,
a Forth Standard System, a CUDA or CPython wrap, or that the system is
internationalized.

## The layers

Five layers. They are not synonyms.

| Layer | What it is | Where to read |
| --- | --- | --- |
| Source effect | Algebraic effect row (`IO`, `Err`, `State`) | Language effects; `docs/NSI_EFFECTS.md` |
| Module requirement | `nsi.required_capabilities` on a manifest | `module.manifest.json` |
| NanoISA trap | Side effect that leaves the pure core | [NanoISA](https://github.com/jordanhubbard/nanolang/blob/main/docs/NANOISA.md) |
| Service method | Stable NSI method id | `docs/NSI.md` |
| Capability | Unforgeable right (`NlCap`) | `docs/NSI_FABRIC.md` |

`State` is language-level. It does not require a host capability.
`Err` maps to `TRAP_ERROR` and does not invent an NSI method.

The map is `schema/nsi/effect_map.v0.json`.

## Nano Service Interface

An NSI document is UTF-8 JSON. `nsi_version` is `0`. I reject any other
version. I reject a name without an `id`. I reject omitted `params`,
omitted type `kind`, `opaque`, and unknown keys such as `c_type`. That
is ABI inference, and I fail closed.

Ids are identity. Names are not.

```text
nsi:nanolang/log
nsi:nanolang/log#write
cap:nanolang/log.write
```

A method parameter carries direction, ownership, lifetime, mutability,
optionality, and streaming. Unknown enumerations fail closed.

`nl_nsi_compat` answers whether a client of an older document can call
a newer one. Generators emit NanoLang, Forth, Python, Rust, and C++
stubs (`make test-nsi-gen`). Invocation is by method id over
in-process, mock, and local-process adapters (`make test-nsi-runtime`).

The example document is `schema/nsi/examples/log.nsi.json`. The
authority is `docs/NSI.md`.

## Module manifests

Three files still serve different jobs. The third now carries an `nsi`
block:

| File | Purpose |
| --- | --- |
| `.nano` | Source and public declarations |
| `module.json` | Native build sources, flags, packages |
| `module.manifest.json` | Discovery, stability, and the portable `nsi` block |

`module.json` stays build metadata. Isolation, restart, budgets, and
required capabilities live on the manifest. See `modules/stdio/module.manifest.json`.

## Capabilities

A capability is `NlCap { secret, slot, generation }`. I mint the secret
from host entropy. Fabricating one from an integer or a host pointer
fails closed.

Rights attenuate on delegate. Transfer revokes the source. Restart
bumps generation so an old token cannot be reused.

NanoLang resource types can own a cap and consume it. Forth cells that
name caps are random tokens, not addresses.

`make test-nsi-cap`. Authority: `docs/NSI_FABRIC.md`.

## POSIX fabric

I host services on an ordinary kernel. A supervisor starts, restarts,
and replaces them. `NlHost` adapters exist for POSIX and in-process.

Scoped slots exist for log, fs, process, net, audio, graphics, gpu,
and python. Those names are typed regions and rights. They are not
device drivers and not language wraps. I do not claim CUDA or CPython.

Bulk data moves through capability-scoped shared memory, with a copy
fallback when `mmap(MAP_SHARED)` is unavailable (`make test-nsi-shm`).

Remote transport exists as a denial: sending a cap to a remote service
fails closed. I do not give a remote party local capability authority.

`make test-nsi-fabric`.

## Isolated editor walker

The SDL frame is a fabric client. Eval goes to `bin/nano_emacs_worker`
over a length-prefixed pipe. The frame does not `dlopen` the
interpreter. The worker does not link SDL. If the walker dies, the
window stays up. Crash-restart keeps buffers.

`C-x C-z` freeze-defun runs `nano_vm` as a grandchild.

I do not claim GNU Emacs. Authority: `docs/NANO_EMACS.md`.
`make test-nano-emacs-worker`.

## Effects to deployment policy

I emit a complete inventory from declared effect names. I generate a
reviewable deployment manifest from that inventory. I reject a grant
set that does not cover declared effects. I report unused grants.

An administrator override deploys anyway. It does not add effects to
the source declaration.

`make test-nsi-policy`. Authority: `docs/NSI_EFFECTS.md`.

## Trap journal and replay

A versioned journal records events at the trap boundary: time,
entropy, file, network, user-input, process, GPU, audio, service.
Replay returns the recorded result and does not invoke the original
service.

Mocks and fault injection are explicit. Checkpoints are sequence
numbers for reverse navigation. They are not heap snapshots.

I hash journals with SHA-256. I authenticate them with HMAC-SHA256 and
a deployment key. That is not a PKI claim. Redacted export replaces
payloads with hashes. Repeating-key XOR can seal remaining bytes for a
local key; that is not AES.

The journal API is a tested C library (`src/nsi_journal.c`,
`make test-nsi-journal`). It is not hooked into every `vm.c` trap in
this release.

## Observability

A trace id copies across NanoVM, router, service, and host spans.
Fabric stores the last trace and audit id on a service slot.
Provenance records source, NanoISA module, interface, service
implementation, policy, and output. Localized log text does not change
those audit fields.

`make test-nsi-obs`.

## What this is not

- A Forth 2012 Standard System. Jackson suites are evidence. See
  `docs/FORTH_STANDARD_SYSTEM.md`.
- An internationalized compiler. Catalogs and machine-draft guides
  exist. JSON and TOON stay English.
- A kernel, GNU Emacs, or a wrap of CUDA or CPython.
- A proof. Policy, journal, and fabric are tested libraries. Coq
  NanoCore does not prove them.

## Next

- Tools that speak to this runtime: [Tools and Backends](06_tools_and_backends.md).
- Tests versus proofs: [Testing and Trust](05_testing_and_trust.md).
- Release boundary: `docs/RELEASE_4.5.md`.
