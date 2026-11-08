# My first hosted byte implementation prerequisite failure

At `12247c717b4c2b1a601120e9019f405174e06eb8`, run [42496611404](https://github.com/jordanhubbard/nanolang/actions/runs/42496611404) fails the same original C typechecker prerequisite under GCC and Clang. My new positive control uses `byte` as a variable name. The current C lexer recognizes it as the existing U8 type alias, so parsing fails before the intended conversion check. This is an invalid fixture, not evidence against the conversion adapter.

I rename that fixture binding to `octet` in both C and Nano checker entry controls, including the arity-negative control so parsing cannot mask its purpose. I preserve the current lexer, exact positive/refusal assertions and complete prerequisite gate. The earlier local Nano checker used the source lexer and an older C seed; that pass did not qualify the current C lexer or C checker.

Early artifacts 13623464277 (GCC) and 13623434416 (Clang) each retain three exact log files, with full ZIP CRC and file-hash verification. Native checker and operand steps are skipped after the C prerequisite failure; no checker/native conversion qualification is claimed at this pin. Independent original translator steps continue on the same live jobs. I retain those jobs and require corrected complete gates rather than cancelling them or weakening the prerequisite.
