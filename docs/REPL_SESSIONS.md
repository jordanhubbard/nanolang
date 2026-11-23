# My saved REPL sessions

I save accepted source operations with `:save` and replay them with `:load`:

```text
nano> :load program.nano
nano> :save session.nrepl
```

In a fresh session:

```text
nano> :load session.nrepl
```

I retain source typed interactively, source admitted by `:load`, function source
admitted by `:reload`, and `:clear` operations in their original order. I copy
the admitted source bytes, so a loaded or reloaded root file can change or
disappear afterward. I use its captured absolute source origin for admission and replay and resolve
relative imports there. Replacing the old pathname with a symlink does not
redirect that stored origin. Imported dependencies must remain available.

I replay source evaluation, expression displays and top-level effects. A saved
session records source operations; it does not capture external files, running
processes or a runtime heap. Use `:load` for session journals. My compiler's
ordinary `.nano` input format is separate from the journal's record headers.
Plain NanoLang source and older plain-source saves remain valid `:load` inputs.

## Reload and clear

For `patch.nano`:

```nano
fn value() -> int { return 2 }
shadow value { assert (== (value) 2) }
```

I can retain a value computed before replacing its function:

```text
nano> fn value() -> int { return 1 }
nano> shadow value { assert (== (value) 1) }
nano> let earlier: int = (value)
nano> :reload patch.nano
nano> :save session.nrepl
```

Loading this journal computes `earlier` with the old body, then patches `value`.
`earlier` remains `1`; `(value)` returns `2`. Reload checks compatible signatures
and does not replay the file's top-level variable assignments or effects.
`:clear` retires variables while retaining functions and declarations; I replay
that operation too. Loading and resaving a journal copies its operations into
the new journal without depending on the older journal file.

## Refusals and publication

I omit parse/typecheck refusals, failed loads and incompatible reloads from
accepted history. I check a journal's complete framing, version and available
capacity before applying any of its records. A truncated or malformed tail
cannot apply its valid prefix. I reject embedded NUL bytes and overflowing
length fields.

A well-framed operation can still fail while replaying, for example when a
dependency is unavailable. I stop at that operation, report the number of
accepted records, and retain the already applied prefix in the live session.
The rejected operation does not enter its accepted history. This includes a
failure in the first record, which reports zero accepted records.

I retain at most 4096 parsed REPL root inputs; imported modules retain their
existing cache ownership. Accepted history also has
4096 slots, including replayed operations and clears. I refuse an operation
before applying it when I cannot retain its history. Saving and inspection
commands remain available at that limit.

I stage a save beside its destination, check every write, flush and close, then
publish it with `rename`. Write, close and publication failures report an error
without a saved-success message. My Linux controls preserve prior destination
bytes through actual injected short writes and close failures and preserve an
existing directory through rename failure. These checks do not establish
power-loss durability.

## Journal version 1

My first line is exactly `# nanolang-repl-session v1` followed by a newline.
Each record has an ASCII header:

```text
<operation> <origin-byte-count> <source-byte-count>\n
```

It is followed by exactly those origin bytes, exactly those source bytes, and
one separating newline. Lengths count bytes. An origin can contain Unicode or
newlines. Operations are `input`, `load`, `reload` and `clear`. A clear record
has zero origin and source lengths; the others retain a nonempty absolute
origin and source. I reserve the session-header prefix for version detection
and refuse unknown versions. I flatten nested journal loads into their source
operations.

My [Linux receipt](evidence/repl-session-save-linux-20261123/README.md) records
16 session methods plus the original 18 callable/state methods passing normal,
ASan/UBSan with leak detection, and unchanged production compiler flags.
Additional nominal/admission aggregate controls retain three complete legacy
checker/runtime leak failures under #1149/#1116. Full REPL ownership, paired
platform and release acceptance remain open under #1205 and the release ledger.
