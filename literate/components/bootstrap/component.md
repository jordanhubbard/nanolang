---
namespace: nanolang
version: 0.1.0
display_name: Compiler bootstrap and toolchain
profiles:
  - original-integration
sample: false
provides:
  - name: nanolang.bootstrap
    version: 0.1.0
    interface:
      uri: interfaces/bootstrap.md
      pin: null
requires:
  - requirement_id: frontend-interface
    capability: nanolang.frontend
    version_range: ">=0.1.0,<0.2.0"
    dependency_kind: generation
    optional: false
    constraints: []
  - requirement_id: vm-interface
    capability: nanolang.vm
    version_range: ">=0.1.0,<0.2.0"
    dependency_kind: generation
    optional: false
    constraints: []
  - requirement_id: native-interface
    capability: nanolang.native
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

# Compiler bootstrap and toolchain

I preserve the existing three-stage bootstrap and raw Stage 1/2 equality, source membership and shadow budgets. Absolute-root-bound checkpoints cannot be copied as valid evidence. I serialize claims on bin, obj and bootstrap state and identify actual compiler/sysroot/header inputs before reuse.

## Retained boundary

I retain these source regions: `scripts/`, `Makefile.gnu`, `GNUmakefile`, `Makefile`. This integration contract is not
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
claiming conversion complete. I require current case-level receipts for `scripts/bootstrap_nanoisa.py`, `tests/test_bootstrap_source_dependencies.py`, `tests/test_bootstrap_tools.py`
and all applicable Linux/Darwin 5.1 obligations. A lock or original adoption record is insufficient.
