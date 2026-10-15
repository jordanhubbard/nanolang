# My separate sanitizer compile and runtime link

Coverage job113762443291 at 6cc098a0c now receives effective linker flags. Its one-pass C command also applies -fprofile-arcs/-ftest-coverage while compiling the generated emitter and reaches the unchanged 180-second deadline. I retain the exact command and timeout; I do not claim to have recovered its temporary generated C.

I compile generated C to an object with the same strict warnings, ASan/UBSan and no-recovery flags, then link with the configured LDFLAGS and instrumented repository runtime. Each command retains the 180-second deadline, and both VM/native entry assertions remain unchanged. My Linux nested-Make probe using the real emitter and a separately coverage-instrumented repository runtime passes in 92.862 seconds. All five ordinary Darwin component methods pass in 42.410 seconds. Hosted coverage remains required.

The separate Linux x64 job113762443515 stops during dependency installation after three attempts exit137. I retain its log without assigning a resource or network cause. Display copies remove ANSI escapes; the x64 copy also removes NUL padding, while raw downloads remain local.
