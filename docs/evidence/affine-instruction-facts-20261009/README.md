# Converged affine instruction facts

I export reachable instruction stack depths, receiver tags and destructive
record-unpack counts only after analysis succeeds. Failed analysis, invalid
capacity and injected allocation failure leave the caller buffer unchanged.
My ordinary native ownership emitter consumes these facts; private profiles
retain their separate admission and depth analysis. Complete resource unions
remain refused until their bytecode and runtime transfers are implemented.

My selected scalar-union runtime fixture now includes an unreachable stack
underflow after an exact match. Both VM and sanitized native execution keep
the selected payload, and native allocation failure releases every object.

I retain each run rather than replacing failures:

- `first.log`: existing 546 bytecode and 856 allocation checks pass.
- `final.log`: my new valid resource fixture fails an old fixture helper's
  assertion that resource-bearing modules must be rejected. I construct the
  checked resource fixture after that helper's existing validation instead.
- `second.log`: 601 bytecode and 1004 allocation checks, plus sanitized scalar
  union execution and allocation cleanup pass.
- `qualified.log`: the stronger dead-arm fixture passes the same focused
  checks and all 2,434 native translator execution checks. This run precedes
  the receiver-tag correction below.
- `owned.log`: the broader owned-runtime case 4 fails its existing allocation
  attempt bound. Native `AGG_GET` incorrectly requires a union tag while this
  case observes a resource-record field. I preserve all original assertions.
  The temporary failed native artifact was removed by the existing harness;
  the tracked fixture source and its hash remain available.
- `corrected.log`: the analyzer's exact stack receiver tag replaces the
  union-only runtime guard. 606 bytecode, 1009 bytecode allocation, 184 owned
  transfer, 275 owned transfer allocation, 32 VM allocation and 1,521 owned
  runtime checks pass. All three Python runtime methods pass, including
  strict native compilation, ASan/UBSan, exact results and allocation cleanup.

The final command exits zero:

```
make -j2 test-affine-bytecode test-affine-scalar-union-runtime test-owned-transfers test-owned-runtime CC=/opt/homebrew/opt/llvm/bin/clang
```

I have not claimed full 5.1, owned-union source execution, Linux qualification,
or final installed-candidate acceptance from these component checks.
