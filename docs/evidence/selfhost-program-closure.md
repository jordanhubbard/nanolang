# My executable call closure

On 2026-09-16 I added `nanoisa_emit_program_nasm(Parser)` alongside my
whole-source parsed and raw APIs. I root `main`, discover direct callees during
actual lowering, and emit each required definition once. Bound module identities
and recursive calls retain the existing call resolution. I register only
required foreign imports in program mode; whole-source validation is unchanged.

I reject top-level globals in this intermediate API. Their storage, ordered
initializers and references are required work under
`task_5a019d3f83cc43d1beb7ace8c8c7388c`. I also reject unresolved function
values and indirect calls, including a local that shadows a function name.
I do not silently drop initialization effects or substitute direct calls.
My canonical frontend still owns full-scope shadow execution; switching its
output route to this API is a separate integration change.

`make -j8 test-nanoisa-src-nano` passes 86 bytecode checks and 19 Python
integration methods. The added method emits deterministic assembly for two
import owners with recursive helpers, aliases, void calls and a live host import.
Both NanoVM and strict C11 AOT executables print `A\nB\n`. An unused aggregate
foreign declaration is omitted in program mode and still rejected by my
whole-source API. A reachable unsupported foreign call, effectful global, global
function reference, function-valued argument and shadowed indirect target all
refuse output. This is executable-subset coverage, not full compiler emission or
matching Stage 1/Stage 2 bytecode.

The preceding merged-compiler probe first refused unused `fs_walkdir` while
emitting every declaration. This API removes that whole-source requirement
principally through executable dependencies; it does not special-case that name.
The next full-compiler obligation is global initialization lowering.
