# Union retained-cost helper checkpoint

I qualify helpers only, after designc6d120b15 on aggregate51ad2e624. No image constructor, execution or publication enrollment is included. Byte outputs are atomic; work/output overlap refuses; managed immutable allocation metadata remains a C precondition, not arbitrary-pointer validation.

The existing permanent constructor target passed1method8.849s:842 linked checks across six unchanged original modules and3504 allocation checks. Eight strict syntax cases passed with Homebrew Clang23 and GCC16, covering both changed providers and linked/instrumented harness. Instrumentation uses O1 ASan/UBSan with fatal recovery; exact environment and commands are retained. No first failure occurred.

Six independent allocator oracles reported retained/peak payloads14854/74825,14862/74833,14863/74834,14876/74847,14884/74855,14896/74867 bytes. Cost work was276 for the first three and288 for the last three. These cover absent parameter table, present table/null row, present zero-arity row, empty string, metadata and debug. Helpers allocate nothing; declaration/proof/plan exact work and one-below controls preserve byte outputs and recover. Corrupted count/pointer/signature/range controls restore the original owner before teardown.

Unchanged constructor remains owner32/site72/carrier13384,retained14854,peak74825,reserved67189009,work16859481,54 allocation positions; atomic proof/site output38 positions. Unchanged proof338/393 passed8.845s with37 positions and literal old sizes/reservations. Six freshly rebuilt legacy budget rows match the retained baseline byte-for-byte. No frontend/bootstrap or installed provider replacement occurred.
