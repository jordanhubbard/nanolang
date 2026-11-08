---
namespace: nanolang
version: 0.1.0
display_name: NanoISA formats and verification
profiles:
  - original-integration
sample: false
provides:
  - name: nanolang.isa
    version: 0.1.0
    interface:
      uri: interfaces/isa.md
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

# NanoISA formats and verification

I retain versioned encoding, verifier authority, assembler/disassembler round trips and rejection of malformed instructions, proof facts and unsupported profiles. I preserve old profile bytes and bounds.

## Retained boundary

I retain these source regions: `src/nanoisa/`. This integration contract is not
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
claiming conversion complete. I require current case-level receipts for `tests/nanoisa/test_disasm_roundtrip.c`, `scripts/embed_binary64_parser.py`
and all applicable Linux/Darwin 5.1 obligations. A lock or original adoption record is insufficient.
