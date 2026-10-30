---
namespace: nanolang
version: 0.1.0
display_name: NanoVM execution and lifetime
profiles:
  - original-integration
sample: false
provides:
  - name: nanolang.vm
    version: 0.1.0
    interface:
      uri: interfaces/vm.md
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

# NanoVM execution and lifetime

I preserve verified execution, service-call provenance, stack behavior, string ownership, garbage collection and safety limits. Invalid or unsupported bytecode must refuse before execution.

## Retained boundary

I retain these source regions: `src/nanovm/`. This integration contract is not
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
claiming conversion complete. I require current case-level receipts for `tests/test_native_string_retention.py`
and all applicable Linux/Darwin 5.1 obligations. A lock or original adoption record is insufficient.
