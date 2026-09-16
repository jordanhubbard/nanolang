# My native local and argument widths

I accept up to 1,024 locals per native function, matching my NanoVirt producer's
current limit. I reject larger counts and arity greater than local storage
before allocating or indexing inference state.

I size module-wide local, parameter, field and shape tables using the actual
maximum local count, with a minimum stride of one. I retain checked allocation
products. Definite-initialization bitsets use each function's actual width,
rounded to bytes. This removes the fixed 256-slot stride without allocating
1,024 slots for every small module. It is not a claim of optimal memory use:
the largest function still determines the module-wide stride.

I size native direct-call text for my supported arity. Each formatted argument
includes its separator in one checked append. I check the prefix and closing
parenthesis too; a truncated write never advances an offset beyond capacity.
Ordinary and tail calls use the same formatter.

## Verification

`make -j1 test-nvm2c` and `make test-nvm2c-sanitizers` each pass 1,308 AOT and
994 shape checks. The sanitizer run verifies fresh ASan/UBSan translator and
shape objects. Opcode coverage and sanitizer-driver checks also pass.

My new fixtures execute local counts 1, 257, 512 and 1,024 with a high-index
string passed to another function. Ordinary and tail calls pass 1,024
arguments, including a string in the final slot. I reject an uninitialized
local at index 1,023, count 1,025, the maximum encoded count, and arity exceeding
local storage.

`make -j1 test-one-ir-compiler` passes the focused empty-return fixture but
still fails its full compiler case. Translation now passes function 463's
local-count boundary and reaches unsupported generic `LT` (`0x2A`) in function
532 at offset 47. I recorded that next requirement separately; the release
gate remains open.
