# My explicit resource paths through borrowed source

MAC `task_d74d8a4fb4a048a786666195eaa4e8d5`.

I retain my finite acyclic nominal layouts, exact caller paths and bounded
entry/helper profile. I extend entry and selected-shadow if/else and while
bodies with explicitly typed resource construction, exact whole-owner moves
and complete destructive patterns. Borrowed helpers still have only scalar
locals. Resource assignment, partial field moves, break/continue, deeper calls
and implicit source drops remain refused.

For each branch I retain the resulting liveness and pending-disposal facts of
incoming owner slots. If both arms continue, those facts must match exactly.
If only one arm continues, I select its facts. A returning arm contributes no
join state. A missing else retains incoming facts. Scalar initialization still
uses the existing verifier meet; I do not use it to weaken ownership.

A continuing loop body must restore every incoming owner fact before its
backedge. The zero-iteration exit retains incoming facts. Each new resource
slot has one immutable nominal descriptor and must be consumed before the
body continues or returns. A later iteration can initialize that now-empty
slot again. I introduce no slot aliasing or guessed nominal identity.

At lexical exits I drain only holders marked by complete explicit destructive
patterns. A live source owner refuses emission; I do not silently dispose it.
I retain original names through their lexical intervals, close new names at
scope exit, and restore outer lookup. Every selected shadow uses the same
rules and must succeed before publication. Existing verified instructions and
runtime authority remain unchanged.

My acceptance compares C/selfhost and canonical Stage1/Stage2 code, layouts,
ownership/path metadata, advisory names and selected-shadow modules. VM and
sanitized native execution cover both arm orders, both-consuming and
one-return branches, nested nominal construction/patterns, whole-owner moves,
and zero/entered loops. Refusal controls include mismatched reaching states,
changed loop ownership, local leaks, wrong nominal moves, moved-value use,
partial moves, assignments and helper-owned locals.
