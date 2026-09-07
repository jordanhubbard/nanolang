# Effects, Deployment Policy, Replay, and Provenance

I connect declared program effects to deployable least-privilege policy.
I record nondeterminism at the trap boundary. I replay from a journal
without calling the original service. I do not claim a kernel.

`make test-nsi-policy`, `make test-nsi-journal`, `make test-nsi-obs`.

## Layers

Five layers. They are not synonyms.

| Layer | What it is | Where |
| --- | --- | --- |
| Source effect | Algebraic effect row (`IO`, `Err`, `State`) | `src/effects.h` |
| Module requirement | `nsi.required_capabilities` on a manifest | `module.manifest.json` |
| NanoISA trap | Side effect that leaves the pure core | `TRAP_PRINT`, `TRAP_EXTERN_CALL`, `TRAP_ERROR`, `TRAP_NONE` |
| Service method | NSI method id | `nsi:nanolang/log#write` |
| Capability | Unforgeable right | `cap:nanolang/log.write`, `NlCap` rights |

The map is [schema/nsi/effect_map.v0.json](../schema/nsi/effect_map.v0.json).
`State` is language-level. It does not require a host capability.
`Err` maps to `TRAP_ERROR` and does not invent an NSI method.

I emit a complete inventory from declared effect names
(`nl_effect_inventory_from_rows`). I generate a reviewable deployment
manifest from that inventory. I reject a grant set that does not cover
declared effects. I report unused grants. An administrator override
deploys anyway and does not add effects to the source declaration.

## Trap journal

Version 0. Each event has sequence, kind, capability, method, argument
hash, optional payload, result, result schema, timing, service
generation, and implementation version.

Kinds: time, entropy, file, network, user-input, process, GPU, audio,
service. I record at the boundary where the value enters a NanoVM
(`nl_journal_record`). Replay returns the recorded result
(`nl_journal_replay`) and does not invoke the original service.
Validation checks order, argument identity, capability identity, and
result schema.

Mocks and fault injection are explicit (`nl_journal_mock`,
`nl_journal_inject_fault`). Checkpoints are sequence numbers for reverse
navigation (`nl_journal_checkpoint`, `nl_journal_seek`).

I hash journals with SHA-256. I authenticate them with HMAC-SHA256 and a
deployment key. That is not a PKI claim. Redacted export replaces
payloads with hashes so replayability does not require publishing the
bytes. Repeating-key XOR can seal remaining payloads for a local key;
that is not AES.

## Observability

I assign a trace id and copy it across NanoVM, router, service, and host
spans (`nl_obs_record_span`). Fabric stores the last trace and audit id
on a service slot (`nl_fabric_last_trace`, `nl_fabric_last_audit`).

Metrics and traces go through `nl_obs_emit`. Provenance records source,
NanoISA module, interface, service implementation, policy, and output
(`nl_obs_provenance`). Localized log text does not change those audit
fields (`nl_obs_localize_log`).
