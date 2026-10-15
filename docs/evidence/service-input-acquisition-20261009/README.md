# My actual Nano companion acquisition

At 2c9528488 I integrate my retained-origin binder and immutable companion reader
into the real Nano driver. My outer compilation wrapper owns the snapshot context
and destroys it after the inner compilation returns. I retain normal failure
prefix cleanup; I do not claim recoverable Nano runtime OOM.

My [paired helper test](paired-helper.log) compiles the current source through
nano_virt and the previously bootstrapped Stage1/Stage2 producers. Each resulting
module verifies and runs in NanoVM and through native nvm2c with LLVM address and
undefined sanitizers plus leak detection. Twenty-four runtime combinations cover
two distinct origins, complete valid catalogs, a malformed second catalog, a
missing second companion and a final-component symlink. I require no partial
path/index publication, retain the acquired prefix until its owner frees it, and
retain original bytes after changing the source document. I do not instrument
all dynamic providers in this gate; the underlying reader has its separate
instrumented fixture.

The [actual fresh C-produced Nano driver](seed-driver.log) passes seven transitive
import cases. Missing, malformed and symlink companions fail acquisition. Valid
companions still reach the explicit unresolved-service checker refusal. Output
and diagnostic aliases preserve companion bytes, including a retained service
followed by a parse error. The original requested output survives every refusal.
The default permanent test runs both rebuilt stages; this first gate explicitly
selects the freshly produced driver module with NANO_SERVICE_INPUT_DRIVER_MODULE.

My first full bootstrap compiles and verifies its seed, then refuses the new
host artifact because its explicit retained-host list lacks file_source_inputs.
I retain that terminal and manifest. In 0aae5c3bb I add this exact module to the
bootstrap host closure; all six [bootstrap guard controls](bootstrap-controls.log)
pass. A corrected full bootstrap is still pending at this evidence checkpoint.

C-loader acquisition, complete namespace and nominal resolution, independent
File lowering, whole-driver aggregate accounting, selected File execution,
Linux and full release qualification remain required. Acquisition does not grant
File authority or close #989/#976.

## My corrected full bootstrap

At 0aae5c3bb my corrected bootstrap completes all seventeen steps, compares raw
Stage1/Stage2 modules byte-for-byte, and verifies every retained source hash at
collection. I retain the [complete manifest and logs](bootstrap-0aae5c3bb/manifest.json).
Both freshly rebuilt native driver stages pass all seven actual acquisition and
output-preservation cases each in [the permanent driver test](rebuilt-drivers.log).
This closes this Nano acquisition/bootstrap checkpoint, not C acquisition,
namespace/type binding, executable File lowering or full release acceptance.
