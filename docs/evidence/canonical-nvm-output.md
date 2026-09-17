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

My integration exposed a missing native `str_starts_with` helper during Stage 2
shadow linking. I supplied that helper and checked seven prefix boundaries.
The existing bootstrap rebuild, native smoke and no-C-seed smoke pass. My twenty
CLI regression cases and nine existing module-binding cases also pass.

`make test-canonical-nvm-output` passes through the full native bootstrap and
four Stage 2 integration cases. I verify two modules with the same function name,
ordinary calls within an imported module, deterministic v2 bytes, VM/native output
parity, actual dependency/root shadow failures, root/import direct and linked
source preservation, invalid types, unsupported float lowering, and prefix edges.
The same four cases also pass with the C-seed-built Stage 1 driver.

## Executable reachability

I now call `nanoisa_emit_program_nasm` from my explicit canonical `--emit-nvm`
route, after the existing merged/bound/typechecked Parser pipeline. The emitter's
whole-source APIs remain unchanged. This selection can omit an unreachable
unsupported helper from executable bytecode while retaining every dependency
and root shadow check before publication.

My added regression imports an unused float-returning helper, emits
v2 output and runs its result in NanoVM and native AOT. Replacing only that
helper's shadow with `assert false` then rejects compilation and preserves the
previous accepted `.nvm`. My existing import, root-shadow, nominal binding,
source/hardlink/symlink identity and prior-output regressions remain enabled.
I also surface the lowerer's exact refusal for a reachable unsupported result
instead of replacing it with a generic frontend message; this failure preserves
the previous output. The six canonical driver methods pass after a fresh
three-stage bootstrap.

This is MAC `task_771b3fdc89aa42f7bf100ce7bfd0d40d`. It does not complete the
NanoISA-only bootstrap. Ordered globals and additional lowering remain separate
emitter work; native C-hosted shadow validation remains part of this route.
