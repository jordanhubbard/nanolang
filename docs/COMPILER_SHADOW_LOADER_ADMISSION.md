# My compiler shadow loader boundary

I track this repair as MAC task_579c1031229d46ae95073f98fa53c751. My exact integrated source 6108067d4161746bc9b20508b9f2fce8a82e433e initializes and opens foreign providers before main.c forks its shadow child. That child has no prepared loader token. Its first loader operations refuse the changed PID; the retained original typecheck run then aborts in builtin_path_join. I preserve that terminal and do not call it a timing result.

## My supported entry paths

My C compiler calls check_interpreted_shadows after front-end checking and compile_modules. The latter builds files but does not open providers. Phase 4.6 is the only direct provider initialization/loading site in main.c. Later phases transpile and eventually clean up the parent loader.

My two check_shadows callers are nanovirt/main.c and check_callback_shadows in nanovirt/shadow_runner.c. Both build artifacts and bind descriptors before execution; neither requires opening providers before its shadow fork. The standalone VM driver loads its production imports only after shadow completion. The C compiler's callback selector currently runs inside the outer child and may create this second shadow child.

## My correction

I move the existing interpreter provider initialization and module-loading loop into its executing child, after callback selection returns zero. A selected callback VM keeps its existing selection and failure authority; I do not initialize interpreter providers on that route. Actual provider constructors and callback thread creation therefore happen after the last shadow fork in these audited CLI paths. Provider loading consumes the unchanged child and parent deadline.

I add a shadow-specific prepared loader admission using the existing lock-free admission protocol. I refuse a busy registry or a history of entering a foreign image, including after shutdown and after unsuccessful native resolution. I record this history before native image entry, not only after successful registration, and never clear it at shutdown. The existing general COP token API remains unchanged.

Both actual shadow fork callers use the new admission before fork, with cancellation disabled until parent/child handoff. Every failure closes its own completion descriptors and releases preparation before restoring cancellation. The child adopts the exact live token before loader or evaluator work. No pthread lock is reset, destroyed or unlocked in the child. The parent completion timestamp and timeout acceptance rules remain unchanged.

This is a boundary for the audited compiler CLIs, not arbitrary embedding after third-party dlopen, external callbacks or unrelated background threads. Loader history cannot prove safety for native activity outside this API. I do not admit that embedding as tested or safe; my separate exec-worker obligation remains open. My SDK cache PID ownership and shutdown callback registration checks remain unchanged.

## My acceptance

I preserve the original full typecheck input and deadlines. Before repeating it, I require direct admission controls for pristine preparation, parent and descendant adoption, busy registry, attempted/loaded provider refusal, shutdown without history reset, and fork-failure cleanup. I require an actual compiler shadow importing a provider, an actual retained callback selector route, and a constructor/thread fixture showing initialization occurs only in the final executing child. Ordinary and supported sanitizer results stay separate. I do not weaken original loader, SDK ownership or opaque COP controls.

## My source checkpoint

I record native entry through wrappers at every loader dlopen/dlsym site. My process-namespace fallback marks history before releasing registry admission; preparation cannot race between the registry scan and that fallback. I also test the actual registry count and retained-image list under closed admission. These checks do not alter general COP admission.

My two shadow callers disable cancellation before preparing and keep the parent protected through child supervision and descriptor cleanup. They restore the previous state after no owned child remains. Child adoption precedes evaluator and provider work. My interpreter provider helper retains the original optional per-module loading behavior; failed initial loader admission or failed path allocation refuses the shadow selection.

My additive C controls exercise successful pristine parent/descendant handoff, native load/failed load/process-symbol history and refusal after shutdown, plus both actual compiler caller fork-failure paths with a substituted fork returning EAGAIN. My original callback tests remain, with an added constructor that creates and joins a real thread and records its PID/parent PID. The new interpreter route must load in the direct compiler child; the retained callback route must load in its nested VM child. These constructor controls observe placement, not arbitrary constructor safety.

Strict C11 GNU-source syntax checks with Wall/Wextra/Werror passed for all changed C translation units and both C controls. Python syntax and git diff whitespace checks passed. I have not executed affected product paths at this checkpoint.
