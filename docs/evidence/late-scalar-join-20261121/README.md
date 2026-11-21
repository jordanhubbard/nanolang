# My unchanged tagged-carrier join correction

Under #1061/#1142, I retain an actual verified Stage1 module from my failed
Darwin bootstrap. My original C translator refuses it at function497,
target649, slot3: the converged join plan has mask0, while both incoming edges
already have tagged storage and identical VOID|INT mask0x3. A private trace
shows caller373/offset637 adding VOID to parameter11 during shape construction.
Discovery had observed INT mask0x2. I retain the raw module, original source,
actual original diagnostic provider, complete traces and private candidates.

I allow late plan completion only when both incoming edges are already tagged
carriers and each has exactly the joined finite mask. I preserve the existing
finite-provenance test and refuse late widening of exact storage, differing
incoming masks or unknown payloads. The initial joined destination already owns
optional storage; this case does not change its representation. I add function,
target, slot and mask context to the remaining stable-join refusal.

My actual1,033,160-byte Stage1 module is retained as a289,617-byte gzip fixture,
SHA-256 `f3b0f758884a6baeec8052c1f946d3ff6bb9a7947fdbf3c308a719098e624d8f`.
The regression verifies those raw bytes and NanoVM verification, then translates
them through the selected product C translator. It does not execute the module's
Darwin host imports or assert a native compiler/fixed point. I wire this real
counterexample into the original native scalar-join and nvm2c Make gates.
A small source reduction translates before correction; I retain it explicitly
as a non-reproducer.

| Check | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Private actual-module regression | Pass37.913s | Pass31.282s |
| Actual original full `make test-nvm2c`, including new product regression | Pass | Pass |
| Original structured-C / shape / callable assertions | 2,458 / 3,890 / 505 | Same |
| Additional original scalar-join/optional-array/optional-float methods | 22 pass13.434s | 22 pass15.434s |

I retain complete original Make terminals, including their expected native trap
messages, all prerequisite suite results and successful terminal status.
Malformed/provenance/prior-output tests remain unchanged. My separate Darwin
private prerequisite first refuses GNU Make's unsupported `--eval` option; its
corrected diagnostic uses an ordinary supplementary Makefile. That setup refusal
is retained and does not replace an actual translation result.

`manifest.json` maps85 retained inputs to lossless gzip members and records raw
and compressed SHA-256 identities. Duplicate byte strings share a member.
I retain77,423,595 logical raw bytes and12,899,163 compressed unique bytes,
including actual original/candidate provider images, source/object inputs,
translated C, commands, fixture bytes, tests and traces. Generated source is a
translation result, not native compiler execution or full sanitizer evidence.
My original sanitizer-driver/opcode-policy controls pass within the Make gate;
I make no newly instrumented full-provider translator claim.

I have not yet repeated the authoritative bootstrap. Its earlier Linux
1,800-second Stage1 generation timeout remains unexplained; this translator
repair does not change that VM generation path. A continuation must collect
bounded phase evidence without changing the original stage/shadow budgets.
Native Stage1/Stage2, raw fixed points, pure NanoLang translator cutover and the
remaining original #1061/#976 requirements remain open.
