# My threaded COP worker release obligation

MAC: `task_f1cde0aa640f4c1fb33cbdf3ab3fb678`.

My daemon explicitly enables isolated FFI for every extern-bearing module in `vmd_server.c`, inside independently created client threads. Each VM lazily enters `vm_ffi_cop_start`. My retained-callback dispatcher also supports `NVM_FOREIGN_WORKER_THREAD`, and providers can own further threads. These are supported paths, not a hypothetical caller violating a single-threaded contract.

My loader admission repair can prevent inheritance of a held registry lock, but it cannot make arbitrary allocator, dynamic-loader or provider-private state safe in a multithreaded no-exec child. The POSIX fork contract restricts that child to async-signal-safe operations until exec. I must close this boundary before claiming full threaded COP isolation for release.

I propose an exec worker, launched with explicit descriptors and independently reconstructed verified module/import state. My existing `cop_main.c` executable and pipe protocol are starting points; they do not by themselves establish equivalence with today's anonymous shared mailbox or inherited cached descriptors. I must inventory launch/install discovery, immutable module transfer and re-verification, exact artifact owners, declared ABI metadata, callback/service/capture refusals, persistent per-worker provider state, request ordering, large strings, batch behavior, crash/restart, timeout and descriptor cleanup. Either an exec-compatible shared-memory object or the existing pipe protocol must preserve the public contract, with no parent-side foreign calls or inherited pointer authority.

Before implementation I require a concrete protocol/source review. Acceptance must include concurrent daemon clients, callback-worker coexistence, actual first-resolution artifact calls, parent cache survival, descriptor cleanup, worker restart and ordinary/sanitized Linux/Darwin runs. The loader-only focused controls remain a prerequisite and cannot close this obligation.
