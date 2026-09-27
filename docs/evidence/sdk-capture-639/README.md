# I qualify actual checked SDK source capture

I freeze source `63953ce7b0c854370c1e4e5e285088a49a99bbde`, including reviewed source612 and the reviewed unpublished-output correction639. All four Linux/Puck ordinary and ASan/UBSan/LSan configurations pass the new owning target. I have not integrated this isolated branch into the candidate.

| Host | Configuration | Fresh providers | Owning target | Result |
| --- | --- | ---: | ---: | --- |
| Linux | GCC ordinary O2 | 11.430s | 3.003s | PASS |
| Linux | GCC ASan/UBSan/LSan O1 | 17.483s | 5.841s | PASS |
| Puck | Apple clang ordinary O2 | 6.632s | 2.087s | PASS |
| Puck | Homebrew clang ASan/UBSan/LSan O1 | 10.845s | 3.452s | PASS |

Each lane first passes the unchanged canonical SDK inventory check, freshly builds the actual Make-selected COMMON_OBJECTS and RUNTIME_OBJECTS with captured expanded flags, then runs `test-sdk-checked-projection`. I retain the original1200-second provider/target bounds,120-second inventory bound, private TMPDIR, full sanitizer settings and process-group cleanup. No deadline changes or retries were needed.

The fixture parses real source through the actual checker. It covers program and module collectors, exact source occurrence/Environment declaration ordinals, owner context, a generic union with complete array/callable signatures, enum, compatible duplicate extern declarations, opaque origin, async unwrapping, transient invalidation after another checked module, checker failure, exact/minus-one budgets and every measured projection allocation prefix in one-shot and persistent modes with fresh recovery. The non-NULL output control refuses before allocation/checker work while preserving output, budget and function count. Expected invalid-source diagnostics remain visible; no sanitizer diagnostics were suppressed.

`seal.json` authenticates72 reports across `linux.tar.gz` and `puck.tar.gz`. I independently verify exact archive membership/byte counts/SHA256, four source-before/after pairs against the complete Git archive scope, four provider pairs and both tool maps. All12 recorded phases exit zero without timeouts or surviving owned groups. Source archive SHA256 is `e4aa2be922fa54275bc66d5ec1065b623a4e4acddb0881d21f89938023ebd238`; driver SHA256 is `343ec2034816b0895370a0212836ca90505687e990d548d4aa50fdb363788321`. Darwin owning product maps also retain the exact dSYM DWARF member; that is a separate debug artifact, not another gate.

Actual source/providers/products remain under `/home/jkh/nanolang-qualification/sdk-capture-639-linux` and `/Users/jkh/nanolang-qualification/sdk-capture-639-puck`. The driver and4020-member Git scope accompany this seal. Both compute slots were released at terminal.

This qualifies only a transient C checked-source capture. Its caller must keep the actual AST and Environment alive and immutable. Source/Environment ordinals are not emitted graph ordinals, and no retained view grants execution authority. Independent emitted graph association, Nano producer parity, complete semantic C ABI binding, deterministic generated glue, immutable image attachment, callback/COP conversion and full installed canonical SDK acceptance remain required.
