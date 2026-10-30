# Native concat-key compatibility

Base24feed248. Contract f26397882 and baseline0cb02ca2d precede the implementation.
The only production change admits exact STR_CONCAT with ROOT_STRING recipe to
the existing native map target. Every other dynamic producer, embedded-NUL literal,
STRUCT transport and nonidentity ordinal/layout restriction remains. No runtime,
publisher, metadata grammar, VM admission, budget or frontend changes.

## Baseline before the change

unchanged.nano is byte-identical to map_record_fields.nano. concat.nano changes
exactly one main call key from `"second"` to `(+ "sec" "ond")`; every original
assertion is retained. Seed, retained generic-field C-hosted canonical, and installed
Stage1/Stage2 all compile/verify/VM/native-execute both sources. Each concat module
contains exactly one STR_CONCAT, no STRUCT opcodes, and no authority metadata.
32 O0/O2 ASan/UBSan normal/forced-collection executions pass with empty diagnostics.
The C-hosted compiler is pinned to its actual earlier generic-field build; it is
not described as a fresh/current raw bootstrap.

Separate explicit test annotations pass 837 exported verifier/transport/repeated-VM
checks, but all four concat modules receive the original selected native refusal:
`I have not qualified this native map string producer or STRUCT transport.`
This demonstrates the metadata-selected compatibility gap. The raw inputs are
retained separately from annotations; no source publication claim follows.

## Changed-source qualification

`python3 -m unittest -f -v tests.test_record_map_native` passes 2 methods in97.549s,
using the four concat producer modules from qualified-environment.json.

- 645 private/public publication and VM checks, including independent image/source
  lifetimes. Public/private generated C is identical on admitted fixtures.
- 56 generated executions: four annotated originals and ten synthetic controls,
  O0/O2, raw/forced collection, ASan/UBSan. The existing false-ASSERT fixture
  requires exact SIGABRT/assertion diagnostics; success requires zero/empty stderr.
- UTF-8 and empty concat-key controls pass through helper local→TAIL_CALL→STRING
  return, caller local, insertion and distinct equal-content replacement. Length
  remains1 and lookup returns the replacement value9. Existing growth/alias/VOID
  lookup/initializer/nominal/negative controls remain.
- Compiler allocation:195 measured positions, transient and persistent denials
  plus independent retry,1,961 checks.
- Generated allocation: each concat original27 positions, alias13, growth47,
  concat-lifetime31 and concat-empty31, at both optimization levels. All920 denial
  processes require exact allocation-invariant aborts;16 independent retries
  require zero tracked live payload. Process abort is not in-process rollback.
- A changed default nvm2c object passes actual public verification/transport/VM/
  translation on all eight raw-source-derived modules, plus the retained NUL and
  unaudited STR_TRIM negatives:1,032 checks.
- All eight unannotated output files and four previously selected unchanged outputs
  remain byte-identical. Clang/GCC16 strict public/private syntax:four cases PASS.

The existing required `test-record-map-native` target permanently runs both new
synthetic lifetime cases and their full generated allocation sweeps; no optional
test-only target was introduced. The public negative changes from newly supported
CONCAT to still-unqualified TRIM, preserving explicit native refusal coverage.

No failed terminal occurred in this changed-source gate. The earlier expected
native refusal is retained as the baseline, not removed from history. All eight
raw compiler/CLI/Clang pins match before and after. Native/compiler selection and
status files retain full commands; retained-artifacts.json records local source,
output and log hashes without binary bundles.

The NUL-free closure relies on literal leaves plus only audited concatenation and
immutable transport, not a changed STRING contract. nstr_concat copies complete
NUL-free inputs without collection; CALL/TAIL root locals+operands before collection,
return has no intervening collection, and nmap insertion copies the key. Other
STRING producers still need demonstrated accepted-source baselines and audit.
STRUCT/nonidentity remain consumer restrictions without demonstrated current
producer need. Publishers and exact-head hosted qualification remain separate.
