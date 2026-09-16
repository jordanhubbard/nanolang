# Scoped VM example shadow budget

I accept `NANO_VM_EXAMPLE_SHADOW_TIMEOUT_SECONDS` only in
`tests/test_vm_examples_coverage.sh`. When explicitly set, I forward its exact
value to the existing `NANO_SHADOW_TIMEOUT_SECONDS` supervisor setting inside
that gate. The supervisor retains its 1–300-second validation. An unset
example budget preserves the inherited supervisor setting or its ten-second
default. This does not extend unrelated full-suite deadline tests.

My regression executes the production configuration and compiler loop with
unset, inherited, overridden and explicitly empty settings. It verifies what
the compiler child receives and that the surrounding suite keeps its original
environment. Both reporting test methods pass. I retain every example,
assertion, shadow and exclusion-validation check.

On Linux arm64 at base `b9119e10`, I built fresh C-seed and NanoVirt objects
with GCC coverage flags (`-fprofile-arcs -ftest-coverage`, no optimization
flag). I ran the unchanged `language/nl_pi_calculator.nano` through the
production compiler loop:

- The default budget stopped shadows after 10.02 seconds and published no NVM.
- The scoped 60-second budget passed in 25.27 seconds and published the NVM.

This reproduces the deadline failure and validates the scoped repair with real
instrumentation. It does not replace the complete hosted coverage gate.

Logs: `/tmp/nanolang-vm-example-coverage-build.log`,
`/tmp/nanolang-vm-example-pi-default.log`,
`/tmp/nanolang-vm-example-pi-60.log`; structured results:
`/tmp/nanolang-vm-example-budget-probe.json`.
