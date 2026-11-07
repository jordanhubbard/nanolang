# Field metadata coverage linkage

I retain hosted coverage run `39195672088` as failed evidence. My field metadata
harness linked an instrumented `nano_aot_runtime.o` without the coverage runtime,
producing unresolved `__gcov_init`, `__gcov_exit` and `__gcov_merge_add` symbols.
The native semantic/allocation assertion did not execute in that failed run.

I append shared native link flags at both ordinary and injected-allocation
native build callsites. I inherit `LDFLAGS` only when `NANO_LDFLAGS` is absent;
an explicitly empty override remains empty. I split flags with shell quoting
rules without executing a shell. I use the existing explicit host-build budget
and group cleanup, retaining the 180-second guest deadline and every allocation
assertion.

## Local qualification

I built the actual AOT runtime from this source pin with GCC 16 and `--coverage`.
A negative link of that object without coverage flags reproduces unresolved gcov
symbols. I retain that failure in `/private/tmp/nano-field-link-negative.log`.
Four shared helper tests pass in 5.317 seconds, including quoted arguments,
ambient fallback, explicit override/empty override, guest budget and descendant
cleanup controls. My complete field metadata test with GCC coverage and
ASan/UBSan passes in 241.009 seconds. The positive run asserts exit zero and
`lookup_maps=0`; the injected-allocation control asserts exit one and
`lookup_maps=1`. Both keep initialization excluded without hiding guest
allocations. Execution produced real `dyn_array.gcda`, `gc.gcda` and
`gc_struct.gcda` files; I preserve those coverage data.

I retain copied aggregate producer and newly instrumented runtime hashes:

```text
d669f9f7d05117fb4f422d442c558917459a99d9cc9ce5fca34be05648705af2  bin/nano_virt
ff2cc26d336d38a55276a29cfe2da4f9f878f9d13c8312190b4aef88f5b8774b  bin/nano_vm
1bfbcf06ecc393326e2b945cf273beabce12dd0e7ff6eb2c0f419aced00e7198  bin/nvm2c
686eee5a046ddf8175f04a451de54f5f155071b9d048986fd6cdc5a958a93d3c  bin/nano_aot_runtime.o
```

I retain the build and execution logs in `/private/tmp/nano-field-link-runtime-build.log`
and `/private/tmp/nano-field-link-coverage.log`. My retained tools support this
focused local harness check; I do not claim a new full bootstrap or a successful
rerun of the Linux coverage job. Integrated hosted qualification remains open
under #976.
