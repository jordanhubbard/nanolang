# My mixed service lifetime checkpoint

I build on `aa0d63b1e` under #990. I add one private carrier for distinct
File/TCP/repeated-File instances; I do not admit mixed source or bytecode
execution in this checkpoint.

## What I test

My actual Make target passes with Homebrew LLVM Clang on Darwin:

```sh
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-services-values
```

I retain [the terminal](acceptance.log) and [every build/run command](commands.json).
The separately compiled carrier and all five underlying lifetime/host/capability
sources use AddressSanitizer and UndefinedBehaviorSanitizer, with leak detection
and halt-on-UB enabled. The linked fixture passes 873 checks; the instrumented
fixture passes 1,507 checks in this run. Bounded nonblocking polls can change the
number of checks. The complete target takes 1.696 seconds.

I exercise simultaneous real File/TCP/File lifetimes through IPv4 and IPv6,
independent file contents, TCP send/receive, result arms, moves, stale copies,
cross-context/instance/catalog refusal, borrow exclusion, per-instance live
masks and clean close. I end separate invocations with outstanding borrows and
observe EOF at the TCP peer after terminal cleanup. An invalid Endpoint becomes
an owned typed Error. Filling one File instance does not exhaust another.

I also create 64 alternating File/TCP instances, retain an unhandled Result in
each, and finish them. I fail every one of the 193 allocation prefixes before
successful full creation and require no tracked allocation to remain. My fault
hooks close the real resources before returning EIO: two File close errors and
one TCP close error produce exactly three close attempts. I retain four cleanup
records, because TCP also retains terminal ambiguous-close history. Repeated
finish/destroy preserves that report and attempts no second close.

My unchanged adjacent [File and Socket value targets](regressions.log) pass:
1,989 instrumented plus 1,289 linked File checks, and 3,823 instrumented plus
320 linked Socket checks. Those adjacent targets use their ordinary Make flags;
I do not describe them as sanitizer runs. GCC 16 also accepts the new carrier
and instrumented fixture with C11, `-Wall -Wextra -Werror -fsyntax-only`.

## Retained failed attempts

My first fixture compile failed at its 64-instance loop with:
`error: expected ';' after do/while statement`. I used the assertion macro in a
comma expression. I replaced that statement with an explicit block.

My [first Make attempt](make-failure.log) used an undefined `PYTHON` variable.
I use the repository's existing `python3` invocation convention.
My [first linked hook build](hooks-failure.log) rejected instrumented-only
cleanup counters as unused; I guard those hook definitions with their fixture
configuration. My [first cleanup assertion](cleanup-expectation-failure.log)
incorrectly expected one report entry per close. I inspected the existing
Socket finish/disposal behavior and require both retained TCP records while
independently checking the exact three host close attempts. None of these
failures is attributed to infrastructure.

## Remaining acceptance

I have not connected this carrier to checked mixed VM/native frames or public
host grants. The trusted catalog-table input still needs to be derived from
the checked retained nominal plan at that boundary. Paired source lowering,
DNS/WebSocket, full 5.1 semantics, Linux and exact-candidate release gates remain
required. GitHub DNS/API access was unavailable during this local batch;
issue publication and push must be verified separately.
