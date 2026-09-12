# My Foreign Linker Input Evidence

My package flags now remain consistent within one build. That does not identify
the library bytes those flags select. I reproduce the remaining cache defect
and measure candidate linker records before choosing an implementation.

```sh
make obj/test_module_generation_probe
python3 -m tests.characterize_linker_inputs
```

The optional argument selects a C compiler executable. I create temporary C
objects and archives, exercise my production module builder, and load only the
fixture libraries in fresh subprocesses. I print JSON observations. Exit zero
means the experiment ran, not that cache acceptance passed. I do not require a
known defect to remain present for this experiment to succeed.

## Measured results

On Darwin with Apple clang 21.0.0 (`clang-2100.1.1.101`) and linker `ld-1267`,
against `212e6b10` on 2026-09-12:

| Change | Cached library | Fresh link |
| --- | --- | --- |
| Unchanged inputs | 42; same generation reused | 42 |
| Replace the selected static archive with different bytes, preserving size and timestamp | 42; same generation reused | 43 |
| Add a library in an earlier `-L` directory without changing flags | 42; same generation reused | 44 |

The original cache record exists but does not name the selected archive. These
are actual stale function results, not just metadata differences. The flags,
caller C source, compiler driver and preprocessing observations remain
unchanged. Archive content and search-selection evidence are both needed.

## Candidate records

Adding `-Wl,-t` preserves the fixture's return values, but the output is not a
lossless line-oriented file inventory. I observe archive-member notation such
as `libselected.a(member.o)`. A directory containing a quote, space, literal
backslash and newline links successfully and prints its literal newline in
the trace. Splitting that trace into paths by newline loses the boundary.

Adding `-Xlinker -dependency_info -Xlinker <file>` also preserves all five
measured return values. Its binary records retain the unusual archive path as
one tagged, NUL-terminated value. I observe:

- Tag 0 with the linker version string.
- Tag 16 with the selected archive, caller object and physical SDK `.tbd`
  paths. The archive is named without the member suffix.
- Tag 17 with absent library-search candidates, including the initially
  missing earlier archive. After creating that archive, it appears under tag
  16 instead.
- Tag 64 with the output path.

These are observations of this linker, not a version-independent format
contract. My characterization decoder records the tags without using them to
authorize reuse. JSON contains record counts, relevant fixture records, tool
identity, selected system examples and trace excerpts; it is not a complete
dump of every SDK search attempt.

There is an important omission: when I supply the selected archive through a
linker response file, the archive appears in the dependency record but the
response file itself does not. I cannot treat this record alone as a complete
inventory of flags and indirect inputs.

My host's `ar rcsT` exits successfully but produces a regular archive, not one
with thin-archive magic. I do not count that as thin-archive coverage. The
experiment only runs its thin-archive link case when the produced format is
actually thin. The non-Darwin dependency-file branch is not verified by this
Darwin run.

## Implementation requirements

I need a compiler/linker-mode boundary that can establish selected library
content and search state without silently changing the original link:

- Capture original-link input evidence and hash selected external bytes,
  including archive contents and applicable member dependencies.
- Detect newly available earlier candidates, not only edits to previously
  selected inputs. Negative search records need explicit validation.
- Identify response files and other indirect flag inputs separately when the
  linker omits them. A readable dependency record is not proof of completeness.
- Validate the record format, full reads and literal path boundaries. Unknown
  or incomplete evidence must not authorize reuse.
- Preserve published generations on build/capture failure and verify recovery,
  unchanged reuse and before/after input changes through actual execution.

This is still not a snapshot of input bytes during the link. Nor does a hash
of a linked dynamic library pin the runtime loader to those same bytes for an
old executable. Runtime dependency retention, source snapshots, transitive
tool identity and other compiler modes remain separate unfinished boundaries.

## Darwin cache repair

I request tagged dependency records from a private Darwin discovery link, then
run the same command for the final link. I require equal input observations
around that final link before recording reuse evidence. If discovery fails,
I discard its record and retry the ordinary link; a successful fallback does
not create reuse evidence. A failed final link fails the build and preserves
the previous generation. I do not publish the discovery output instead.
Malformed, truncated, unknown-tag or contradictory records also withhold reuse
evidence. A cacheable cold build now uses two shared links; warm reuse skips
both. Already uncacheable configurations do not acquire discovery links.

I require a linker header, an input in my private build directory and the exact
expected output record. I hash regular external inputs and record absent
search candidates. Duplicate records must agree. I check those hashes and
absences again before warm reuse. Paths remain literal strings, including
newlines and backslashes; I do not parse archive-member display notation.
Record paths are limited to 8191 bytes, with bounded record count and payload.

The reproducer now returns 43 after the archive edit and 44 after the earlier
candidate appears, while unchanged inputs retain their generation. These
results have an acceptance test. Parser tests cover literal unusual paths,
unknown/missing/truncated records, wrong outputs, missing or non-regular
inputs, existing negative candidates and overlong paths. Unsupported capture,
malformed capture, response-file bypass and recovery exercise actual builds.

I also reproduce replacement during linking: after `5f15a029`, changing an
archive after the link or during later preprocessing could store its new hash
beside old code. I now retain the observation checked around the final link
and validate it again before storing it. I never replace it with later hashes.
Tests cover changes after discovery, after the final link and during later
preprocessing, followed by recovery and warm reuse. A failed final link keeps
the previous generation and does not trigger another link attempt.

I conservatively withhold reuse evidence when the link command contains `@`;
response-file inputs are not captured yet. This also excludes literal `@`
characters that are not response-file syntax. I preserve ordinary compilation
and linking for those commands. The check is not a general shell-input audit.

This integration is Darwin-specific. Other linker formats remain open. Matching
before/after observations are not an atomic snapshot of the bytes the linker
read, and I do not yet capture changes that occur and revert during a build.
The linker binary and arbitrary wrapper inputs also need separate identity.
The full implementation requirements above remain in my roadmap and MAC task.
