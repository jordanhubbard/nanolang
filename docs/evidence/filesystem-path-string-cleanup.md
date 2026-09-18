# My filesystem path-result cleanup

I enroll six exact filesystem providers in the existing versioned artifact
string cleanup contract: `path_normalize`, `path_canonical`, `path_join`,
`path_basename`, `path_dirname` and `path_relpath`. Their results are allocated
by the provider through malloc, strdup or realpath, or are NULL on failure.
Each exported companion releases through that same provider; free(NULL) is
safe. Public declarations document consumption after escaping text is copied.

I change neither the path algorithms nor the VM/native consumers. The existing
same-image lookup, snapshot-before-cleanup order, NULL rejection and missing-
companion borrowed compatibility remain in force. I do not free foreign
results based on contents or names. Facade TLS outputs remain borrowed, and
file_read retains its separately audited private fallback sentinel.

My child task is `task_50d54c4988044e64afa7f83e82d0c9e5`; parent
`task_d5f899966241452a900422938fff3265` remains open pending its full inventory.
I record the contract before code in `8916b19e`. The implementation and tests
are `328a7d72`, based on main `25f528c9`.

The new acceptance builds the real fs.c provider with its runtime closure.
For every provider it checks normal and empty inputs during 2,000 iterations,
then checks aliases retained from before the loop. The same bytecode executes
in NanoVM and native C. A second test replaces allocation operations only in
the provider translation unit and checks all six NULL results through both
consumers; the runtime objects retain their normal allocators. Existing tests
continue to check release counts, intern hits, same-symbol distinct images,
wrong-image refusal, borrowed literals, and copy-failure cleanup.

Fourteen combined artifact/host methods pass in 15.134 seconds after a fresh
CLI build (`/tmp/nanolang-fs-path-ownership-focused.log`). Generated native
programs use ASan, UBSan and LeakSanitizer; the ordinary VM binary is not
sanitizer-instrumented. No compiler-abort case or historical binary is run.

Both new methods also pass with strict Clang in 2.092 seconds (`/tmp/nanolang-fs-path-ownership-clang.log`). I select this host's installed GCC13 support directory through a private cc wrapper, without disabling warnings. A final strengthened NULL-result assertion checks that refusal contains no AddressSanitizer, LeakSanitizer or UBSan error; that six-provider method passes again in 7.722 seconds (`/tmp/nanolang-fs-path-ownership-null-final.log`).
