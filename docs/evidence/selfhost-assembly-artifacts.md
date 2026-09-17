# My assembly publication artifact boundary

I lower two exact foreign contracts needed by my compiler's publication path:
`nl_nanoisa_assemble_text_save(string, string) -> int` and
`nl_nanoisa_last_error() -> string`. I preserve the declaration's source owner,
immutable library path and artifact kind. I do not accept arbitrary foreign
result types or broaden unrelated facade exports.

My emitter gate passes 86 checks and 52 regression methods. Five new exact
C-seed comparisons check the integer status, zero-argument diagnostic call,
function bytecode and ordered import metadata. Six malformed signature/call
cases refuse output.

I also call the actual NanoISA facade from emitted VM and native programs. A
successful call publishes verified v2 bytes. A subsequent invalid assembly
call fails, exposes a diagnostic, and preserves the accepted module. Both
backends produce the same retained bytes. Native artifact callers use the
existing host-runtime link contract.

My canonical frontend publication test passes after the separately tracked
external-source manifest-root repair in PR447. A fresh native three-stage
bootstrap passes, followed by all 14 canonical-output, native-module-linking
and compiler-artifact-support methods against Stage 2. My native Stage 1 and
Stage 2 binaries differ; these checks do not establish a fixed point.

After restacking onto `e07fecfb`, I rerun the emitter gate: 86 checks and all
52 methods pass. The retained logs are
`/tmp/nanolang-assembly-restacked-gate.log`,
`/tmp/nanolang-assembly-bootstrap3.log` and
`/tmp/nanolang-assembly-stage2-final-tests.log`.

Actual full compiler emission advances past these publication calls to
`undefined function str_concat`. I record that next lowering slice as
`task_9ab740ec43654cedb0c1de224856735d`. I do not claim a complete compiler
module or a bytecode fixed point from this bounded result.
