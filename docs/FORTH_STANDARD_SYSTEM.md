# My Forth 2012 System Label

I am not a Forth 2012 Standard System. I am not an ANS Forth Standard System.
Passing a test suite is evidence. It is not, by itself, a conformance claim.

This file is the precise label. `docs/FORTH_2012.md` pins the document,
suites, Gforth differential, licensing, and environmental contract.
`docs/FORTH_CORE_COVERAGE.md` is the Core-name matrix. I will not put
"Forth 2012" or "ANS Forth" on a banner, in `--version`, or in a REPL
greeting as if I were a Standard System.

## What I claim

Nothing in this file is a Standard System claim. Nothing in this file
is a Core, Core Ext, Block, Exception, File Access, Search-Order,
String, Double, or Memory-Allocation banner.

I claim that:

- Jackson v0.15.0 is vendored and pinned
  (`tests/forth/vendor/gerryjackson`, `tests/forth/pins.json`).
- Core through Block evidence files pass under
  `make test-forth-core` through `make test-forth-block` when loaded
  through C `REFILL`. Passing those files is evidence.
- The Core-name matrix is 133 tested names
  (`docs/FORTH_CORE_COVERAGE.md`). That is a FIND-and-exercise record.
  It is not a Core banner.
- File Access `INCLUDE` / `INCLUDED` exist as words.
  Core evidence still loads through C `REFILL`.
  `make test-forth-jackson` records that gap.

## What I deny

- I do not claim a Forth 2012 Standard System.
- I do not claim an ANS Forth Standard System.
- I do not claim Core.
- I do not claim Core Ext or any optional word set as a banner.
- `examples/language/forth/run_tests.fs` is a File Access driver for
  280 T{ cases. Those cases are not the pinned standard suites.

When a word set later has a banner, I will name the word set, the suite
revision, and the cases I still skip. I will not upgrade this system
label until that record exists for every selected word set.

`make test-forth-jackson` fails if this file drops the denial sentences
or if the REPL prints `Forth 2012 Standard System`.
