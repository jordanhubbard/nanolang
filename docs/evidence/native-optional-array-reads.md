# I preserve optional scalar array reads

I tested production commits `1b05617a` and `115b478a` over main
`3a33b182`. My ordinary native integer, boolean and string array reads now
carry a tagged optional result. Missing indices yield void; valid indices retain
their payload tag. My index remains signed 64-bit, including negative values and
values above UINT32_MAX. I do not adopt the obsolete fleet narrowing contract.

My shape analysis keeps the optional wrapper through locals, calls, branch joins
and record fields. Typed array writes constrain the present payload and check its
tag before unboxing. Record-array reads and the owned execution profile retain
their separate contracts.

I passed all 2,422 structured native checks and 1,092 shape checks. My three
focused methods cover twelve paired VM/native artifacts: three element families
through missing/present reads, ignored results, tag inspection, locals, calls,
empty arrays, typed writes, record fields and both sides of a branch join.
Indices include -1, one past the end, 2^32 and INT64_MAX.
Generated C passes strict warnings and ASan/UBSan/leak checks with GCC
(1.688 seconds) and Clang (2.312 seconds). I do not claim that the VM itself was
sanitizer-instrumented in these paired checks. Typed missing-payload rejection
uses the existing checked unboxing helpers; these checks do not exercise every
consumer or claim full language ownership equivalence.

I retain the initial focused and full native logs under
`/tmp/nanolang-optional-array-`. My repeatable target is
`make test-native-optional-array-reads`.

MAC `task_438ff01101234d6cb3cad5dfeaa0e9f2` replaces the cancelled fleet task
`task_ed0f455484d04f13818362fd857d2889`. PRs 307, 331, 343, 349 and 356 are
obsolete attempts at that older contract; I retire them only after this repair
lands. This bounded repair does not close my compiler publication or release hold.
