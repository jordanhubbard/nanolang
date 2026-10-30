---
namespace: nanolang
version: 0.1.0
display_name: Standard libraries, FFI and services
profiles:
  - original-integration
sample: false
provides:
  - name: nanolang.libraries
    version: 0.1.0
    interface:
      uri: interfaces/libraries.md
      pin: null
requires: []
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

# Standard libraries, FFI and services

I preserve public File, Socket, WebSocket and service interfaces, generated descriptors, headers, link order, refusal and resource cleanup. Private or optional service evidence does not establish public admission.

## Retained boundary

I retain these source regions: `modules/`, `std/`, `stdlib/`. This integration contract is not
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
claiming conversion complete. I require current case-level receipts for `Makefile.gnu`, `docs/RELEASE_5.1_SCOPE.md`
and all applicable Linux/Darwin 5.1 obligations. A lock or original adoption record is insufficient.
