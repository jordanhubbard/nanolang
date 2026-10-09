# Integrated owned-union runtime regression gate

I run the full recorded command at `b623525f91c3b72d853dc432b0288ebc1a7f09f2`.
It exits zero in 144.098 seconds with identical before/after HEAD, status,
source inventory and user-file hash. The user file remains untracked.

The gate covers selected union execution and VM heap failures, affine state
and bytecode, scalar-union and record runtime, consuming calls and multiple
calls, result descriptors/DAGs, nested and value results, ownership transport,
private declaration projection, and all 2,435 native translator execution
checks. Original assertions and allocation budgets remain intact.

This pin predates the C source lowering checkpoint; it does not qualify that
later change, self-hosted source lowering, Linux or the final 5.1 release.
