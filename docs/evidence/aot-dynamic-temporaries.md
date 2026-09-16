# Function-sized AOT temporaries

I remove the fixed 256-temporary translation limit. Emitter bookkeeping uses
checked function bounds; record-field facts and branch snapshots own dynamic
storage. Restoring a snapshot copies its field facts while retaining allocation
ownership and temporary high-water counts.

I emit each function body once, then insert temporary declarations sized to its
actual high-water counts. Unused representations reserve one slot for valid
C11. I check output-size addition before growing the output buffer. Generated
temporaries still use automatic storage: this does not promise unlimited host
stack capacity or implement register allocation.

The new tests exposed missing integer/string array constructors when arrays
were built directly on the operand stack without locals. I now select those
helpers from the representation emitted for each constructor, including legacy
integer-tagged constructors whose destination local establishes a string or
record array. I retain existing legacy tests.

## Verification

`make -j1 test-nvm2c` passes 992 checks on Darwin. New tests compile and execute
300 temporaries for integer, string, integer-array, string-array, record and
record-array representations through both branch paths. A separate
string-valued-record case checks restoration of field type facts beyond the
old limit. Tests inspect generated declarations and returned values; the C
compiler uses `-Wall -Wextra -Werror`. `git diff --check` passes.

`make -j1 test-one-ir-compiler` still fails at function 20 with
`AGG_PACK has too many fields`. The 75-field parser state and array-valued
aggregate fields remain unfinished. This change does not establish full
compiler execution or release readiness. MAC
`task_419c47bdc8fc42e4b52eb6af1a0e9a71` remains open.
