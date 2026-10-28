---
namespace: nanolang
version: 0.1.0
display_name: Independent qualification and formal evidence
profiles:
  - original-integration
sample: false
provides:
  - name: nanolang.qualification
    version: 0.1.0
    interface:
      uri: interfaces/qualification.md
      pin: null
requires:
  - requirement_id: bootstrap-interface
    capability: nanolang.bootstrap
    version_range: ">=0.1.0,<0.2.0"
    dependency_kind: generation
    optional: false
    constraints: []
  - requirement_id: libraries-interface
    capability: nanolang.libraries
    version_range: ">=0.1.0,<0.2.0"
    dependency_kind: generation
    optional: false
    constraints: []
  - requirement_id: distribution-interface
    capability: nanolang.distribution
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

# Independent qualification and formal evidence

I retain every original 5.1 obligation and platform gate, existing formal proofs with their limits, exact original test cases and independent acceptance. Historical receipts and successful suite exits do not establish current case coverage. My future policy must bind case ownership, exact reports, permitted skips and current source/toolchain identities.

## Retained boundary

I retain these source regions: `tests/`, `formal/`, `docs/`. This integration contract is not
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
claiming conversion complete. I require current case-level receipts for `docs/RELEASE_5.1_SCOPE.md`, `docs/RELEASE_5.1_STATUS.md`
and all applicable Linux/Darwin 5.1 obligations. A lock or original adoption record is insufficient.
