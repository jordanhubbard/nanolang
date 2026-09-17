# My tagged array globals

I carry integer and string array handles in my native tagged global storage.
Loading a global, copying it into a local or another global, and replacing the
original slot do not copy or invalidate the array. Push and set mutate the
shared handle. My generated program retains the existing entry-lifetime array
storage model; this change does not establish general resource ownership.

My tagged array operations check the array tag and representation before
reading the handle. Element reads preserve integer and string tags. An
out-of-range read yields void; consuming that void as an integer traps.
Indices narrow to 32 bits, matching my current VM array indexing. Out-of-range
sets leave the array unchanged. Array truthiness, identity equality, length,
type checks and printed values work through tagged globals.

My regression fixtures cover integer and string arrays initialized before
entry, cross-function append, copied aliases, replacing the original global,
tagged set values, empty and negative-index reads, index narrowing, printing,
non-array consumption and mismatched element tags. Generated C is compiled
with warnings as errors. Tagged reads use the emitted scalar getter after
their bounds check so those helpers are not left unused.

## Verification

I pass `make -j1 test-nvm2c` and `make test-nvm2c-sanitizers`: 1,294 AOT
checks and 994 shape checks in each run, with fresh ASan/UBSan objects for the
sanitizer run. Opcode coverage and sanitizer-driver checks also pass. These
tests establish the exercised behavior, not full compiler acceptance.

## Remaining boundary

I do not claim complete global support. Record arrays, general aggregates,
empty-array inference and full compiler acceptance remain open. My existing
ordinary, untagged array getter still traps on an out-of-range read; matching
the VM's void result there is separate roadmap work.

With this change, my full compiler translation passes its earlier global-array
store and stops at function 369, offset 553: conflicting integer/string array
results in `extract_type_args`. Its empty return path requires inference
compatible with the nonempty string-array path. I retain that failure as a
release blocker rather than weakening result checks.
