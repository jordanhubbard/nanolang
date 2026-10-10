# My TCP document preparation evidence

I qualify the source inputs in `inputs.json` on this Darwin arm64 host from
parent `bbd287db3`. I retain compiler identities in `tools.json`, test entry
logs here, and exact provider commands, process statuses, executable/object
hashes and fault summaries in `runs.json`. Full generated corpora and raw
allocation logs remain at the recorded local artifact paths. This is a
component and consumer qualification, not an exact-candidate release gate.

| Configuration | Checked scope | Result |
| --- | --- | --- |
| Homebrew LLVM, ASan/UBSan | Both full corpora, two linkage modes, legacy NSI/generator/catalog neighbors | 4 methods pass, 31.359s |
| Apple Clang, ordinary | Same corpora plus simultaneous File/Socket isolation | 5 methods pass, 17.437s |
| GCC 16, ordinary | Same corpora plus simultaneous File/Socket isolation | 5 methods pass, 25.106s |
| Homebrew LLVM, ASan/UBSan | Simultaneous File/Socket isolation | 1 method passes, 3.791s |
| Homebrew LLVM, ASan/UBSan | File publisher API/faults, actual publisher Make/CLI, full File binding neighbors, compiler-input snapshot | 4 methods pass, 20.342s |

I run 705 File and 849 Socket document cases in each complete corpus. Each
File instrumented run passes 26,883,819 checks; each Socket instrumented run
passes 41,283,367. These counts include the allocator bookkeeping, not that
many distinct product scenarios. Independently linked runs pass 7,866 and
9,450 checks respectively. Canonical JSON and generated source match independent
expected bytes. File's source fixture is unchanged. My File heap bound remains
7,575,764 bytes; Socket's is 7,575,756 on this host. These exclude caller storage,
stack, allocator overhead and libc internals.

I preserve my first failed File run. When I parameterized the existing corpus,
its allocation-shape loop shadowed the catalog selector, causing the supposed
foreign-catalog input to be File itself. The API correctly accepted it; the
fixture expected refusal. I renamed the loop variable. Corrected runs require
both cross-catalog refusals and Socket's private-resource substitution refusal.
I made no product assertion weaker. The initial Socket run had 848 cases;
corrected Socket runs include the previously omitted substitution case.

I retain File's exact public API names/status values and generated shadows.
The new Socket output is a declaration only: no compiler admission, network
shadow execution, Socket VM/native dispatch, Linux qualification, full
bootstrap, WebSocket integration or release completion is claimed here.
