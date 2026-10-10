# My clean compiler-product capture gate

I run `make -j2 test-one-ir-compiler` at exact commit
`ae92c0488ed44da0112f7eb51367a9976054b16a` in an independent APFS copy-on-write
checkout. The gate passes all 109 top-level methods in 899.496 seconds; Make
returns 0 after 912.297 seconds. The checkout is clean before and after, and its
HEAD is unchanged. The manifest binds the complete log hash.

Both generated-compiler routes also run the five-method capture matrix against
their freshly built compiler. I retain the native full-source generation route,
the independent self-hosted emitter route, strict C and sanitizer/collection
checks. I do not substitute the smaller prepared-compiler suite for this gate.

The earlier ordinary checkout failed with disk exhaustion before tests began;
its retained preparation evidence is in `../capture-qualification-space-20261008`.
This success follows that verified storage recovery. It does not qualify later
component/phase changes or the fresh raw bootstrap equality still required for
the release.
