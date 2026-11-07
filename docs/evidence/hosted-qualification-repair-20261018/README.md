# My hosted qualification repairs

I consolidate PRs #995 and #996 before repeating hosted qualification. I retain the actual failures from runs [39068601600](https://github.com/jordanhubbard/nanolang/actions/runs/39068601600) and [39070512777](https://github.com/jordanhubbard/nanolang/actions/runs/39070512777).

- My instrumented raw bootstrap finishes, then the legacy `typecheck` component hits its default 600-second compilation alarm. I align that job with the existing platform/coverage `CMD_TIMEOUT=1800`; I retain sanitizer instrumentation, the raw bootstrap bound, and guest execution bounds.
- My Linux x64 and ARM64 generated sanitized lowerer compilation hits the shared 120-second subprocess limit. I give explicitly marked native compiler invocations the configured build budget, while guest execution retains 120 seconds.
- My relocated native service products omit ambient `LDFLAGS` and cannot link coverage-instrumented archives. I fall back to ambient flags only when the caller supplies no override; an explicit empty override remains empty. My new test replaces one required object in a private relocated archive with an actual coverage-instrumented object and exercises both compiler drivers, override precedence, linker failure, and prior-output preservation.
- My successful coverage products retain compiler-generated `.gcno` and `.gcda` files. I permit only their exact publication names as nonexecutable regular nonsymlink files; my negative controls still reject staging directories, provisional executables, unrelated names and symlinks.

I pass 444 strict compiles across Apple clang and GCC 16. My two new link/cleanup methods pass both normally and with ambient ASan/UBSan link requirements. These local receipts do not establish the fresh hosted Linux result or release qualification.

My initial direct GCC lowerer invocation completed native compilation but then found a missing `obj/test_service_lowering` test prerequisite. I retain that failure, build the prerequisite through Make, and repeat the actual execution gate rather than reporting the partial invocation as a pass.

I also pass all 21 existing File and WebSocket driver methods with ambient `LDFLAGS=--coverage` in 227.862 seconds, including installed/relocated publication.

After building the missing prerequisite, I pass the complete mixed WebSocket paired-source byte gate with GCC 16 ASan/UBSan in 143.564 seconds. I compare the C lowering against both VM and sanitized native NanoLang lowerers for mixed/repeated catalogs and direct/indirect construction. My guest execution limit remains 120 seconds. This Darwin result is not a substitute for the pending Linux jobs.
