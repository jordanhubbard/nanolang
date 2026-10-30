# Sanitizer checkpoint runner-image mismatch

Run 39276986919 completed sanitizer Stage 1 (job 118149401411), but continuation job 118203797120 refused restoration before Stage 2 or sanitizer tests. The retained logs show producer Ubuntu image `20261018.339.1` and continuation image `20261012.334.1`. The unchanged-environment guard reported: `I require unchanged flags, toolchain, runtime dependencies and runner image`.

This establishes a runner-image mismatch sufficient for rejection; it does not claim that every other recorded dependency matched. It is separate from the earlier coverage directory-timestamp rebuild failure. No guard, flags, tests, archive, or provider was changed to accept this continuation. Exact-candidate sanitizer continuation remains required.

The excerpts preserve original timestamps and image identities. `inputs.json` records SHA-256 and sizes of the complete locally retained official job logs. Stage 2 and sanitizer tests did not execute in this continuation.
