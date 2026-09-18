# My checked optional string hashmap keys

I record task `task_a4ab109d881d432ab1728299006b121c` in contract
`1c73ce42` before production `84d924ec`.

My native classifier previously required a plain string or unresolved key.
String-array reads and map lookups retain optional tagged storage. My emitter
already extracts such values through `nvalue_require_string`, checking both
the string tag and non-NULL payload. I now admit that representation at key
consumers while preserving the runtime check.

I create a separate exact STRING extraction shape. The optional producer's
present-payload shape flows toward that destination; the producer is not
rewritten as a plain string. Map key equality applies to the extracted shape.
Known concrete non-string keys and incompatible present-payload shapes remain
translation refusals. Map value admission, key families, runtime ownership,
collection and map mutation are unchanged.

My focused ordinary modules exercise HM_SET/GET/HAS/DELETE using string-array
reads, string-map results, record projections, locals, aliases, calls and branch
joins. I verify VM/native results and inspect generated checked string
extraction. Concrete integer, boolean, float and optional-integer keys must
refuse translation without replacing prior output. I do not execute refused
modules.

## My product evidence boundary

The stopped Linux full-quick run at `63c26ecd` passed transpiler entry assertions,
then native translation refused a string-map key. The temporary module was not
retained. That log does not establish the actual key storage or blame source
facts. I preserve the original stopped tree and logs unchanged.

My new component acceptance uses a separate detached source tree at that same
pin, copied completed Stage2/capture/runtime tools, and `NANO_NVM2C` pointing to
the repaired translator. `NANO_AOT_RUNTIME` and `NANO_AS_CAPTURE_HELPER` name
those copies. The compiler still uses three absolute imported host libraries;
this is not hermetic relocation. I record and recheck tool/library hashes in
`/tmp/nanolang-map-key-component-manifest.json`. Fresh module and generated C
retention uses `--keep-c`; compile and execution results are separate gates.
