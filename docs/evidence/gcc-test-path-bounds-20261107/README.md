# My path bounds and optimized C test checks

I reproduce #1078 with a 32-byte physical buffer containing `occupied` and a declared capacity of four. My previous helper accepts the append and changes the buffer. I now reject it without modifying the buffer. I also test zero capacity, a nonterminated buffer, exact fit, exhausted capacity, literal shell quoting and include prefixes under ASan/UBSan.

I preserve my 4,096-byte source-hash path bound with checked lengths and copies. Absolute source paths do not depend on the directory argument. My native and ASan/UBSan controls cover absolute, relative, changed, missing and oversized source or directory paths.

I separate misleading statement layouts in the owned-union fixture, file-product fault allocators and file-product text copier. I preserve every assertion and allocation-failure position. My native owned-union checks report 2,380 selected assertions; the file-product bridge and source-hash controls pass.

I replay the five affected Make-derived test compilation profiles with GCC 16, `-O3`, `_FORTIFY_SOURCE=2`, `-Wall -Wextra -Werror` and their actual include flags. All five pass. My first replay retains the additional text-copier warning before its correction. These are selected profiles, not the completed #1046 coverage gate. I retain the original full inventory and initial diagnostic report privately under `/tmp/nanolang-fast-plan-probe.json`.

My complete `make check-fast` passes: five real-lexer fixture controls, 3,120 inspected static fragments (133 separately reported lexer refusals), five prerequisite controls, 223 strict Clang plus 223 strict GCC product compilations, and the bootstrap-tool control.
