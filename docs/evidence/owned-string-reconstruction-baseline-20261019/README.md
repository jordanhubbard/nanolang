# I retain the unchanged owned STRING reconstruction baseline

I ran only the reviewed baseline from design97fbde3a4, based on aggregate1e13109d1.
My exact unchanged source is `unchanged-affine-example.nano`. `initial-pins.json`
retains source/provider hashes; `commands.json` and per-command files retain all
23 command results, byte-exact stdout and process-group cleanup. No production,
facts schema or admission change preceded this baseline.

Both retained ordinary producers (NanoVirt04269be4 and nanoc_u8_contextd7c3c136)
emit original-verified modules that return0 and print exactly
`audit.log\nresource closed\n`. Each has three functions: main()->INT with1 local;
open_file(STRING)->FileHandle with2 locals; close_file(FileHandle)->VOID with3
locals. STRING is only open_file parameter/local slot0. Its slot1 stages INT42.
The resource has one INT field. Close consumes its owner through the existing
shared field observation, ASSERT and final destructive unpack. Its STRING
literal print is not a resource field.

Executable PUSH_STR indexes5 and6 carry `audit.log` (9 bytes) and
`resource closed` (15 bytes). Advisory constant strings contain embedded NUL;
they are not executable string literals and must not enter the proposed table,
cap or NUL checks. Original dumps retain complete metadata and signatures.

I built exactly one existing `shadow_module_driver.nano.txt` using the pinned
aggregate C-hosted compiler3ac71b97, with its byte-identical copy at the child
bin/nanoc_string_baseline path. This path preserves the compiler's project-root
derivation. The initial /private/tmp copy was not executed. I preserve the entire
actual imported Nano source hash inventory before build and checked it unchanged
afterward. The900-second compiler budget is distinct from180-second guest/probe
deadlines. The existing retained process-group runner records completion99.269s,
exit0, no timeout, no cleanup errors and absent process group. Root/dependency
shadows use the compiler's ordinary defaults; no suppression flags were added.

That independent original-shadow emitter produces four functions:
synthetic entry()->INT locals3; open_file(STRING)->FileHandle locals2;
close_file(FileHandle)->VOID locals3; main()->INT locals1. Original verification
and VM execution pass0 and stdout is exactly
`audit.log\nresource closed\nresource closed\naudit.log\nresource closed\n`.
Its raw emitter assembly and canonical dump remain separate files. This is one
C-hosted shadow producer, not both selfhost shadow-driver parity.

Every ordinary and original-shadow module fails current facts extraction with
`I require a verified bounded resource module with exact fields.` Both C and
Nano reconstruction repeat that refusal and `I did not publish reconstructed
source`; every preexisting sentinel remains byte-identical. No refused output is
executed. This establishes the missing reconstruction admission, not a source or
runtime failure. Native parity for this baseline is historical evidence only;
new reconstructed native execution belongs to the later implementation gate.

Before launch I held jobs during sustained worker saturation. All initial tool
hashes still matched at the read-only prerequisite audit. I corrected the copied
compiler root path, reused the existing timeout/group runner, and separated raw
assembly from dumps before any baseline execution. Original sources and generated
artifacts are pinned; binaries/modules remain at the recorded /private/tmp path
and are represented here by hashes. No full compiler bootstrap ran. Full #999 and
fresh aggregate/platform reconstruction qualification remain open.
