# I retain the Linux native-translation corrections

I reproduce two completed Linux failures under release issue #976, rather than treating an overall queued workflow as unstarted:

- Retirement job `118068552611`, run `39255178567`, candidate `72efc9b0648383b24eb5ef9cdc8b9c4a870eb686`, actual merge checkout `d667437ad2048fecff1a1c27cf49d8b61d248531`.
- Compiler-contract job `118057686481`, run `39251960912`, candidate `421f3c58a487de38f2483a6c7ded6355830fc7a5`, actual merge checkout `81e3bb09bbf6fd8d9c0b4797489b512063c2af68`.

Both jobs use the same NanoVirt and native translator binaries. Their retained manifests prove that both translator source files match my starting aggregate source. Original logs/artifact archives remain in `/private/tmp/nanolang-hosted-*`; I retain freshly reproduced bytecode here because the original harness temporary directories did not survive failure.

## I correct two distinct boundaries

My current parser module verifies and executes in the VM, then native emission rejects a tagged record-array replacement. Instrumentation locates the actual setter at `parser_set_function_affine`, function 164, offset 174 (function 436 in the larger lookup module). Its classified write result is an exact record array, while the emitted receiver is a tagged value. I constrain record writes through the payload's existing directed array-write relation, save the write output shape, require exact record-array storage, check the runtime tag/storage/handle, and use the existing typed record conversion and replacement. I preserve the classified result representation; I do not add arbitrary record boxing to scalar helpers.

The current larger lookup module additionally fails during global discovery: an earlier optional-integer join plan survives after a carried `node_id` parameter gains UNKNOWN provenance. At function 48 target 649, both branches carry the same argument; only a later argument differs. I invalidate only nonfinal optional-scalar plans that gain UNKNOWN and recompute ordinary carrier compatibility. Final-pass stability and finite-variant UNKNOWN refusals remain enforced.

## I retain failures and validation boundaries

`actual-fixtures-fixed.log` records both original tests passing in 100.265 seconds through VM and Homebrew Clang ASan/UBSan native execution. The field test retains its zero-per-lookup allocation assertion and deliberately injected allocation failure control. The preceding `actual-fixtures.log` retains my generated-C field-name typo, corrected from `.string` to the actual `.text` member before this successful run.

I preserve the intermediate shape diagnostic that correctly rejected my first helper's OPTIONAL-only assumption. The actual classified result is directly ARRAY→RECORD; I accept that exact proved result as well as an OPTIONAL array carrier, with the same runtime receiver checks.

`qualified-inputs.json` pins the production source and actual tools. `final-build-commands.json` records the ordinary compiler flags and final source-path compile/link commands. Initial diagnostic builds used temporary source copies and are not represented as production binaries. Fresh hosted Linux confirmation remains pending; local macOS success is not Linux execution evidence.

## I qualify the corrected boundaries

I pass 32 additional methods in 153.361 seconds (`neighbors.log`): the new actual parser setter/alias execution fixture; scalar joins; finite scalar and integer-array variant carriers; conditional record joins; and tagged record-array writes, including wrong runtime tags, element kinds, null handles and unchanged-output refusals. Native tests use Homebrew Clang with ASan/UBSan; the scalar join suite explicitly receives the sanitizer flags. GCC 16 also accepts the production translator under `-Wall -Wextra -Werror -fsyntax-only`.

The new parser fixture executes `parser_set_function_affine` twice through an aliased Parser, preserving function name/parameters/return type and the shared owner array while changing/resetting completion. It therefore exercises the corrected setter, beyond compiling the original visibility fixture. The small carried-argument test is a passing baseline neighbor, not a minimized reproducer; I retain both passing baseline attempts. The full lookup bytecode remains the baseline-proven discovery-plan regression.

My successful original command is `NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang python3 -m unittest -v tests.test_public_global_declarations tests.test_field_metadata_lookup` before adding the new setter method. My subsequent 32-method command uses `CC` and `NANO_NATIVE_TEST_CC` set to that Clang, `NANO_JOIN_CFLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all'`, and the explicit new setter method plus `tests.test_native_scalar_joins`, `tests.test_native_variant_scalar_carriers`, `tests.test_native_variant_array_carriers`, `tests.test_optional_record_join`, and `tests.test_native_record_array_tagged_fields`. No execution assertions were weakened.
