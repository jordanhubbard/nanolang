# Callable array loops and inferred function locals

I retain complete element annotations on checked loop bindings, including
inferred inline arrays. My native loop reads a typed function pointer from
sized storage. My evaluator copies each callable before the body so replacing
its original array slot does not invalidate the loop value. I retain returned
callbacks across cleanup and release loop copies on return, break and continue.
I also preserve returned function signatures in inferred local declarations.

My ten focused methods pass (`focused.log`). I execute named, computed, inline
and empty callback arrays; replace a slot while retaining its current callback;
return that callback; and exercise break and continue. I reject incompatible
indirect arguments in unused code and preserve prior output.

My full native suite passes 70 methods (`regression.log`). The same suite passes
70 methods with nine compiler components instrumented by ASan/UBSan
(`sanitized.log`, leak detection disabled, 65520 KiB stack). My C-seed NanoISA
regressions pass 84 methods (`nanoisa.log`), and evaluator/transpiler unit checks
pass (`units.log`). I use the seven native modules listed in my preceding
[storage checkpoint](../native-callable-arrays-20261010/README.md), including
four new loop/inference methods. NanoISA regression modules are
`test_cseed_generic_functions`, `test_cseed_tuple_generics`,
`test_contextual_generic_functions`, `test_higher_order_generic_functions`,
and `test_generic_callback_storage`.

I separately execute all three positive loop fixtures through the current
C-seed NanoISA producer and NanoVM. Named/computed/inline loops and callback
mutation/return also execute through C AOT. My combined break/continue/empty
fixture reaches C AOT but is refused for an unresolved indirect-call target
in its empty loop (`vm-aot.log`, `empty-loop.nano`). I keep that gap open;
these results do not establish complete translator parity.

My targeted C compiler and NanoISA builds pass. My general build fails in
Stage 2 while the installed compiler compiles `typecheck_driver.nano`: its
shadow process exceeds the unchanged ten-second deadline (`build.log`). I do
not establish its cause from this result or claim complete build qualification.
My initial loop/inference failures remain in `first.log`.

I still need callable map/filter/reduce parity, the empty-loop translator
correction, and the complete 5.1 implementation/platform gates under #976.
