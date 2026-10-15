# Cyclic File source state storage

I continue #989 from `f35ae5370`. My new actual-driver corpus retains the
unchanged five generated shadows, adds two useful helper/cycle shadows, and
requires nested loops with repeated File acquisition, borrowed writes, rewind,
read, close and an early-return cleanup branch. Both frontends initially refuse
publication after source checking (`first-c.log`, `first-nano.log`).

My isolated diagnostic build identifies the existing 16 MiB cyclic query budget:
493 retained states of 30,848 bytes exhaust it (`storage-limit.log`). I retain
that budget, the variant/edge limits and every logical check. Each retained node
now allocates its exact declared locals, occupied operand stack, reference extent
and regions in one allocation. I reconstruct zero-initialized scratch before
transfer. Absent reference getters still return the same zero value, and complete
canonical-state comparison remains available for indirect-call candidate checks.

My negative source controls require rejection of consuming one owner repeatedly
across a loop and passing overlapping exclusive borrows. Both preserve the prior
output and produce no shadow START records. These checks do not establish
multi-borrow or indirect source execution.

## Development self-compilation observation

The earlier native self-compilation finishes successfully, but its following
byte comparison fails (`development-self-compile.log`). Both modules are 629,076
bytes. Their eight differing bytes are four checksum bytes and four characters
in the File product artifact generation directory. I do not normalize those
bytes to claim a fixed point. The earlier development builds were not made from
a frozen artifact dependency set; exact release-revision bootstrap remains open.

## Qualification

- `final-c-cli.log`: all six actual C driver methods pass in 50.670 seconds,
  including the original seven-shadow cyclic source and unchanged-output refusals.
- `final-nano-cli.log`: all six actual Nano driver methods pass through VM and
  native compiler execution in 135.580 seconds.
- `final-neighbors.log`: linked and instrumented cyclic, cyclic-hosted and
  indirect-hosted queries pass. Their allocation sweeps retain 47, 376 and 409
  prefixes respectively, including transient refusals and cleanup assertions.
- `dispatch.log`: linked and instrumented VM/native cyclic dispatch pass in
  124.557 seconds. Instrumentation follows each retained runner's provider scope;
  these are not fully instrumented compiler builds.
- `driver-build.log`, `translate.log`, `native.log`: the updated Nano compiler
  builds with its selected shadows and translates to native form. Its artifact
  dependency hash matches the changed validator source in `inputs.json`.

I retain the initial strict build failure in `build.log`: indirect composition
also uses the full-state equality helper. I preserve that helper and add a
separate comparison for compact stored states; `build2.log` and final neighbor
checks cover the correction. The complete source corpus stays unchanged.

Darwin checks here do not replace Linux qualification or the complete 5.1 gates.
