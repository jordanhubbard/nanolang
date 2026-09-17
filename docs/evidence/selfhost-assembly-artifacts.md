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

The canonical frontend test initially encounters the separately tracked
external-source manifest-root bug (`task_c6b698326e0f4e6296299ddfdf172ebd`);
its companion repair is being validated. I retain that test without skipping it.

Actual full compiler emission advances past these publication calls to
`undefined function str_concat`. I record that next lowering slice as
`task_9ab740ec43654cedb0c1de224856735d`. I do not claim a complete compiler
module or a bytecode fixed point from this bounded result.
