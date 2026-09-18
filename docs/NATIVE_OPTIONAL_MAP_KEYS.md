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

## My completed local gates and setup correction

Three focused VM/GCC ASan/UBSan methods pass in 1.164s. Fourteen focused and
adjacent methods pass in 16.718s: the new key and optional-array harnesses use
Clang, while the existing map-global harness explicitly invokes GCC. Both
include generated native ASan/UBSan and leak checks. My full native gate passes
2,422 checks and 1,365 shape checks at unchanged production `84d924ec`.

The first isolated component setup omitted `bin/nano_vm`, required to run VM
shadows. It exited 1 before invoking the translator at 440.24s, with maximum
RSS 2,804,684 KiB. I retain `/tmp/nanolang-map-key-component.log` and
`/tmp/nanolang-map-key-component-time.log`. This is an explicit missing-tool
setup failure, not evidence about the repaired translator. I copied the
unchanged frozen VM, verified all prior hashes, and recorded the corrected
closure in `/tmp/nanolang-map-key-component-corrected-manifest.json` before a
corrected component gate. Its result remains required before readiness.
