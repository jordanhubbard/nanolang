# Instrumented bootstrap-shadow profile

I compile the complete compiler using the corrected unoptimized Clang
ASan/UBSan C seed on Darwin. The command passes in 150.371 seconds with
use-after-return checks and the unchanged 60-second shadow deadline.
I sample the live interpreted shadow child for five seconds. Function lookup,
variable binding and string comparisons dominate the recorded top-of-stack
counts. `notes.txt` records the cache-path difference from CI. This is diagnostic
evidence, not a reproduced Linux timeout or its resolution.

The hosted units-01 failure remains tracked in
`task_4251e719e9634aa7b597dbdd080b6b9b`. I am preparing a separate Linux
instrumented checkout; I do not alter the active fixed-point run's providers.
