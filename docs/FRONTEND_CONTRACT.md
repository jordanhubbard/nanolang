# My Frontend Contract

I accept multiple source languages, but I do not give each one a private
machine. Every frontend lowers into the same versioned NanoISA module, and I
verify that module before I optimize or execute it.

This is the Phase 21 frontend contract. NanoLang remains my native language.
The other frontends are bounded architecture probes until their own stated
conformance goals pass.

## Boundary

A frontend owns these language-specific stages:

1. source decoding, lexing, and parsing;
2. name resolution and source-language type analysis;
3. desugaring into typed functions and control flow;
4. selection of shared NanoISA types, layouts, and operations; and
5. emission of optional source-language metadata.

The shared pipeline owns module validation, NanoISA verification,
language-neutral optimization, serialization, execution, profiling, debugging,
and target translation. A frontend does not optimize serialized NanoISA by
using facts that it failed to preserve in the module.

## Frontend Interface

A frontend accepts source bytes, a stable source identity, its language and
frontend versions, compilation options, and the service-contract registry
available to the compilation. It returns either one NanoISA module or a
deterministically ordered diagnostic list. It does not return a partly valid
module alongside errors.

The emitted module contains:

| Field | Requirement |
|---|---|
| Module identity | NanoISA format version, frontend identity and version, source identities, and reproducible build identity |
| Source locations | Half-open byte spans tied to stable source identities; generated instructions may cite a parent span and a synthetic reason |
| Typed functions | Stable function identity, signature, locals, blocks, instructions, and source mappings |
| Layouts | Shared scalar, aggregate, callable, and resource layouts with size, alignment, fields, and representation constraints |
| Constants | Typed values encoded by NanoISA rules rather than frontend-private object representations |
| Imports | Typed NanoISA library, NSI service, or isolated FFI imports with version constraints |
| Effects | Declared shared effect identifiers and each function's required effect set |
| Capabilities | Required capability kinds and the operation or import that consumes them; never ambient authority |
| Diagnostics | Stable code, severity, primary span, related spans, and notes; rendered text is not the interface |
| Optional metadata | Namespaced, versioned records that may be discarded without changing program behavior |

Unknown required fields or unsupported NanoISA versions are errors. Unknown
optional metadata is ignored. Frontends must not encode behavior solely in
optional metadata.

## One Module Format

Every frontend emits the current versioned NanoISA module format. It uses the
same serializer, loader, and verifier as NanoLang. A successful frontend test
must therefore perform this sequence:

1. compile source to the shared in-memory module;
2. verify the module with the shared verifier;
3. serialize it with the shared serializer;
4. load and verify the serialized module again; and
5. execute or translate only the verified module.

A frontend may reject a source program more strictly than another language.
It may not bypass verification, weaken verifier rules, add a private module
section with executable meaning, or send an unverified module directly to a
backend.

## Shared Runtime Surface

All frontends bind through the same typed import records. That gives them the
same access, subject to their own language subset, to:

- versioned NSI service contracts;
- unforgeable capabilities and explicit effect declarations;
- process-isolated FFI rather than an in-process escape hatch;
- instruction-to-source mappings, names, scopes, and optional language facts
  used by the debugger;
- the NanoVM profiler and its instruction, allocation, call, trap, and FFI
  counters; and
- the maintained NanoVM and target translators.

A frontend-specific runtime helper is an ordinary typed NanoISA library or
service. It receives no authority merely because a frontend generated the
call.

## Language Facts

I preserve useful language facts as namespaced optional metadata. Initial
shared fact kinds cover purity, match exhaustiveness, ownership or affine use,
inferred source types, and effect provenance. Each record identifies its
producer and schema version and points to a function, block, value, layout, or
source span.

An optimization may consume a fact only when it validates the fact or when the
verified NanoISA semantics make the transformation safe without it. Removing
all optional metadata must leave a valid module with the same behavior. This
keeps source-language analysis separate from language-neutral optimization.

## Opcode Rule

I reject a proposed frontend-specific opcode by default. A proposal must:

1. state semantics without referring to source-language syntax;
2. show why existing operations or a library cannot express it adequately;
3. define verifier, interpreter, serializer, debugger, profiler, and translator
   behavior;
4. show a use outside the proposing frontend, or explain why the primitive is
   reusable across the planned language families; and
5. add shared conformance and malformed-module tests.

Frontend bookkeeping, cached dispatch, syntax markers, and private object
layouts do not satisfy this rule. They belong in optional metadata, libraries,
or private optimized dispatch IR.

## Bounded Frontend Intake

Before implementation begins, each frontend records these limits in the
roadmap or its design document:

- supported and intentionally excluded language features;
- the architectural pressure it is meant to test;
- maximum implementation scope, including whether parsing, modules, and an
  interactive environment are included;
- positive, negative, malformed-module, and verifier tests;
- at least one shared-library or service fixture;
- a pinned external conformance subset when one exists, with exclusions; and
- completion evidence and claims that remain out of scope.

No frontend starts with an open-ended compatibility claim. Passing a selected
suite proves only that selected suite.

## Cross-Frontend Conformance

Each frontend after the first must join the shared conformance harness before I
call it complete. The harness compiles equivalent programs from every
applicable frontend and runs their modules against the same NanoISA libraries
and versioned service interfaces. It checks:

- both in-memory and serialized modules pass the shared verifier;
- observable return values, output, service requests, traps, and capability
  failures match the fixture contract;
- imports use identical service contract identities and versions;
- stripping optional language metadata does not change behavior; and
- debugger locations and profiler attribution remain tied to the originating
  source language.

Not every language must express every fixture. An exclusion is explicit and
justified by the bounded subset; it is not silently counted as a pass.

## Contract Acceptance

The shared contract is ready for a frontend when its intake document is
bounded, its module round trip passes the common verifier, its diagnostics are
stable and source-located, and one shared library or service fixture runs
through the common runtime path. Later frontends add themselves to the same
fixtures rather than cloning the harness.
