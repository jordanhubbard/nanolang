# My canonical frontend's explicit NanoISA route

I accept `nanoc source.nano --emit-nvm -o output.nvm`. My default target remains
native C. I require an explicit output path and reject combining this route with
`--target c`.

I merge imports, bind module function identities, check types, and pass that same
Parser to `nanoisa_emit_parser_nasm`. I do not reparse merged text and discard its
resolved bindings. Unsupported lowering fails before publication. My existing
source identity checks cover the root source and imported sources, including
symbolic and hard links.

I retain my existing native shadow pipeline before publishing module bytes.
Dependency shadows run by default; the existing explicit `--root-shadows-only`
option keeps its documented behavior. Failing shadows preserve prior output.
My facade assembles, verifies structure and stack discipline, serializes v2, and
atomically replaces the output only after all preceding checks pass.

This is a frontend integration prerequisite. It does not remove the C toolchain
used for shadows, establish a NanoISA-only bootstrap, resolve foreign hosts, or
prove complete source coverage. My source subset and linked-module requirements
remain explicit.
