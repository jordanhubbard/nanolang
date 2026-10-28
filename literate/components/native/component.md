---
namespace: nanolang
version: 0.1.0
display_name: Native C AOT and runtime
profiles:
  - original-integration
sample: false
provides:
  - name: nanolang.native
    version: 0.1.0
    interface:
      uri: interfaces/native.md
      pin: null
requires:
  - requirement_id: isa-interface
    capability: nanolang.isa
    version_range: ">=0.1.0,<0.2.0"
    dependency_kind: generation
    optional: false
    constraints: []
authoring_inputs:
  - kind: specification-to-source-skill
    uri: literate/skills/specification-to-source/original-source/SKILL.md
workflow_definition:
  uri: literate/workflows/original-review.md
routing_policy:
  uri: literate/routing/original-review.json
flavor_slots: []
entrypoints: []
acceptance_contracts:
  - uri: acceptance/original.md
    pin: null
source_dependencies: []
---

# Native C AOT and runtime

I preserve verified NanoISA-to-C translation, ABI, native runtime ownership and VM/C-AOT equivalence. My future native graph must identify actual backend translation units within src rather than treating all src as one producer.

## Retained boundary

I retain these source regions: `src/`. This integration contract is not
a per-translation-unit source or output inventory. Where regions overlap, `literate/source-ownership.json`
provides deterministic preservation ownership, not native build dependency admission.

## Requirement: Preserve original behavior

I SHALL preserve the behavior and refusal contract above through every native lifecycle change.

### Scenario: Adoption metadata changes

- WHEN I add authority, interfaces or locks
- THEN my original implementation paths, modes and bytes remain unchanged
- AND I retain native build/test/install commands and running qualification

## Requirement: Qualify actual native realization

I SHALL bind actual native units, source/header closure, toolchain identities and output ownership before
claiming conversion complete. I require current case-level receipts for `tests/test_one_ir_compiler.py`, `tests/test_native_string_retention.py`
and all applicable Linux/Darwin 5.1 obligations. A lock or original adoption record is insufficient.
