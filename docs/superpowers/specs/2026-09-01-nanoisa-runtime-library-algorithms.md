# NanoISA v2 — string and collection algorithms live in runtime libraries

## Status

Accepted for the 4.0 Portable ISA design (Phase 12). This note records the ISA
boundary between primitive operations and the higher-level algorithms that were
moved out of the instruction set.

## Decision

The normative v2 Portable ISA in `spec/nanoisa.yaml` (`instruction_families`)
exposes only primitive string and aggregate operations. Trimming, case
conversion, splitting, replacement, formatting, parsing, and collection
algorithms are **not** portable instructions. They are provided by runtime
libraries reached through `call.import` / `trap`, exactly like any other host
or module function.

Concretely, the v2 families keep:

- primitive string access: length, concatenation, substring, char access,
  equality, and containment (via the existing typed primitives), and
- layout-driven aggregate operations `aggregate.pack`, `aggregate.get`,
  `aggregate.set`, and `aggregate.tag`.

Everything richer is a library function. The algorithms moved out of the ISA
are enumerated in the `runtime_library_algorithms` section of
`spec/nanoisa.yaml` so schema tests can guard against them re-entering the
portable instruction set:

| Category         | Runtime-library functions                                   |
|------------------|-------------------------------------------------------------|
| Trimming         | `str_trim`, `str_trim_left`, `str_trim_right`               |
| Case conversion  | `str_to_lower`, `str_to_upper`                              |
| Splitting        | `str_split`                                                 |
| Replacement      | `str_replace`                                               |
| Matching         | `str_starts_with`, `str_ends_with`, `str_index_of`          |
| Formatting       | `format`                                                    |
| Parsing          | `string_to_int`, `string_to_float`                          |
| Collection       | `str_join`                                                  |

## Compatibility

The v1 `legacy_opcodes` inventory still lists the compound `STR_*` opcodes
(`STR_TRIM`, `STR_TO_LOWER`, `STR_TO_UPPER`, `STR_STARTS_WITH`,
`STR_ENDS_WITH`, `STR_SPLIT`, `STR_REPLACE`) for executable v1 compatibility.
They are intentionally absent from the v2 families and are not part of the
portable ISA surface. New v2 lowering targets the runtime-library functions
rather than these opcodes.

The C backend already lowers these algorithms to runtime-library helpers in
`src/stdlib_runtime.c` (`nl_str_trim`, `nl_str_to_lower`, `nl_str_split`,
`nl_str_replace`, `nl_format`, …); the registry entries in
`src/builtins_registry.c` for these functions carry `OP_NOP` rather than a
dedicated inline VM opcode, confirming they are library calls and not ISA
instructions.

## Verification

`tests/test_nanoisa_schema.py` asserts that:

- no v2 instruction family declares a trimming, case, splitting, replacement,
  formatting, parsing, matching, or collection algorithm, and
- every documented `runtime_library_algorithm` names a category and the
  primitives it is built from, and is not itself a v2 instruction.
