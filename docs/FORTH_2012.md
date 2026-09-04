# Forth 2012 Pins, Licensing, and Environment

I am not a Forth 2012 Standard System. Passing a test suite is evidence. It is
not, by itself, a conformance claim. I will not put "Forth 2012" or "ANS Forth"
on a banner, in `--version`, or in a label until the pinned suites pass and the
manual cases in this document are written down as tested behavior.

This file is the pin. `tests/forth/pins.json` is the machine-readable copy.
`make test-forth-gforth-diff` fails if they drift.

The architecture contract is
[ANS Forth on NanoISA](superpowers/specs/2026-08-30-ans-forth-nanoisa-design.md).
The compiler that will satisfy these pins is still Phase 13 work. The session
runtime in `src/forth/` is the first slice: one mutable `NvmModule`, one
persistent `VmState`, Forth stacks that are not the NanoVM operand stack, a
virtual address space, and a file-handle table. `make test-forth-session`
covers that slice. It does not compile colon definitions and it is not Core.

Dictionary FIND is case-insensitive for ASCII `A`–`Z`. Redefining a name
creates a new header; the old name token still maps to the old execution
token. `SOURCE` and `>IN` live in the virtual address space. Nested
`EVALUATE`, included files, and blocks restore both on pop. The block image
is 32 disposable 1024-byte blocks. `TIB`, `>IN`, `BLK`, `STATE`, and that
image are pinned: `FREE` of those addresses fails.

## What is pinned

### Standard document

I implement **Forth 2012** as published by the Forth 200x Standardisation
Committee at https://forth-standard.org/standard/intro.

That is the frozen Forth 2012 document. It is not ISO/IEC 15145:1997 (that is
Forth-94 / ANS Forth). It is not the rolling set of later Forth 200x proposals.
A proposal accepted after the 2012 freeze is out of scope until I pin it here
by name and date.

### Test suites

I have not vendored either suite. I record the revisions I will vendor, and the
license facts that decide whether I may.

| Suite | Pin | Vendor? |
| --- | --- | --- |
| Gerry Jackson Forth-2012 tests | `gerryjackson/forth2012-test-suite` tag **v0.15.0** (`9773f84dd12390f342d37195da8848b04e1f4a23`) | Not yet |
| Committee process + later tests | `Forth-Standard/forth200x` `master` at **91f1ed9c756aac27f57e939c270b5f2c84262427** (2026-07-05) | Not yet |

Jackson v0.15.0 is the Forth-2012 Core and optional-word-set suite I will run.
The forth200x snapshot is the committee repository at a recorded commit; it
includes later material (for example recognizer tests) that is not Forth 2012.
I will not treat those later files as 2012 conformance evidence.

### Differential Gforth

I pin **Gforth 0.7.3** (GPL-3.0-or-later). Homebrew formula `gforth` 0.7.3 and
the Debian/Ubuntu `gforth` package are the intended installs. I do not vendor
Gforth.

`make test-forth-gforth-diff` runs `examples/language/forth/pi.fs` for 0, 1, 10,
and 50 places under that Gforth and checks the exact decimal strings below.
That is tested against Gforth 0.7.3. It is not a claim about my own Forth,
which does not compile colon definitions to NanoISA yet.

| Places | Output |
| ---: | --- |
| 0 | `3` |
| 1 | `3.1` |
| 10 | `3.1415926535` |
| 50 | `3.14159265358979323846264338327950288419716939937510` |

If `gforth` is missing, the pin-consistency checks still run. The Gforth cases
fail in CI (`CI` is set) and are skipped locally.

## Licensing — before vendoring

I confirmed these facts before copying any third-party Forth tests into the
tree. I still have not vendored them. I will vendor Jackson v0.15.0 only when
the NanoISA Forth compiler can include `runtests.fth`. I will not vendor
Gforth. I will not vendor `Forth-Standard/forth200x` wholesale.

**Jackson suite.** There is no SPDX `LICENSE` file. The Hayes core tests carry:

> (C) 1995 JOHNS HOPKINS UNIVERSITY / APPLIED PHYSICS LABORATORY
> MAY BE DISTRIBUTED FREELY AS LONG AS THIS COPYRIGHT NOTICE REMAINS.

Jackson-authored files state they are in the public domain and ask that the
notice be kept. Vendoring is allowed if those notices stay in the copied
files. I will keep them.

**forth200x.** There is no repository `LICENSE`. Some reference
implementations say public domain. File headers vary. A wholesale copy would
mix licenses I have not inventoried, so I will not copy that tree until I
list each file I need.

**Gforth.** GPL-3.0-or-later. Invoking an installed Gforth is a differential
run, not a combined work I ship. I will not copy Gforth sources into this
repository.

## Conformance labels

Until the compiler exists and the pinned suites pass:

- I do not claim a Forth 2012 Standard System.
- I do not claim an ANS Forth Standard System.
- I do not claim Core, Core Ext, or any optional word set.
- `examples/language/forth/run_tests.fs` remains a regression harness for the
  current token-string interpreter. Those 280 cases are not the pinned
  standard suites.

When a word set later passes its pinned tests, I will name the word set, the
suite revision, and the cases I still skip. I will not upgrade the system
label until every selected word set has that record.

## Environmental model

This is the contract for the NanoISA Forth I will build. It is **assumed**
until the compiler implements it. Nothing in this section is proved. The pi
digits above are tested only on Gforth.

Values follow Forth 2012 §3 usage requirements and §4 documentation
requirements. Where the standard leaves a choice, I pick one and keep it.

### Cells, characters, addresses

- A cell is 64 bits, two's complement.
- An address unit is 8 bits. `ADDRESS-UNIT-BITS` is 8.
- A character is 8 bits. Text is UTF-8. `MAX-CHAR` is 255.
- I do not expose host pointers as Forth addresses. The Forth address space
  is a byte-addressable virtual space the VM owns.
- Alignment: cells are 8-address-unit aligned. Characters are 1.
- `CHAR+` adds 1. `CELL+` adds 8.

### Double cells

- A double cell is two 64-bit cells, 128 bits logical.
- On the stack the most significant cell is on top, as Forth 2012 3.1.4.1
  requires.

### Division

- Floored division. `/`, `/MOD`, and `FM/MOD` use floored quotient toward
  minus infinity.
- `FLOORED` from `ENVIRONMENT?` is true.
- `SM/REM` is symmetric division, as the standard names it.

### Integer range

- `MAX-N` is `2^63 - 1`.
- `MAX-U` is `2^64 - 1`.
- `MAX-D` is `2^127 - 1`.
- `MAX-UD` is `2^128 - 1`.

### Floating point

- IEEE binary64 on a separate floating-point stack.
- Not host `long double`. Not decimal floating point.
- `FLOATING` and `FLOATING-STACK` are present once the Floating-Point word
  set exists. Until then they are absent, not approximated.

### Files

- File Access words operate on runtime-owned handles, not raw host `FILE*`.
- A handle that is not in the table is an ambiguous condition I treat as
  `THROW`.
- Line endings: I read LF and CRLF. I write LF.
- File names are UTF-8 byte sequences in the host encoding.

### Terminals and input sources

- Nested sources: terminal, evaluated string, included file, and block.
- `SOURCE` and `>IN` restore on `THROW` / `CATCH` and on finishing a nest.
- The interactive terminal is a line-buffered UTF-8 stream. No block editor
  is required for Core.

### Blocks

- Block and Block Ext tests run only against an explicitly disposable image.
- I will not point those tests at a user file. Jackson `blocktest.fth` uses
  blocks 20–29 and overwrites them; I will keep that isolation.

### Limits (implementation-defined)

These are the initial `ENVIRONMENT?` answers I will report. `STACK-CELLS` and
`RETURN-STACK-CELLS` match the session stacks in `src/forth/forth_session.h`.
`ENVIRONMENT?` itself is still absent until Core exists.

| Query | Value |
| --- | ---: |
| `/COUNTED-STRING` | 255 |
| `/HOLD` | 255 |
| `/PAD` | 255 |
| `RETURN-STACK-CELLS` | 1024 |
| `STACK-CELLS` | 1024 |
| `#LOCALS` | 16 |

Dictionary and address-space size are not a single constant. The VM allocates
them. A failed allocation is `THROW`, not a wrap.

### Ambiguous conditions

Where Forth 2012 names an ambiguous condition, I do not leave it undefined
silence. The NanoISA Forth will `THROW` a distinct code, or reject the
definition at compile time, rather than wrap, invent a stack picture, or
continue with a host pointer.

The session runtime already rejects stack overflow and underflow, unaligned or
unallocated addresses, host pointers used as Forth addresses, and stale file
handles. It returns failure from the C API. `THROW` codes come later with the
Exception word set. Remaining cases I will record when the compiler lands
include: division by zero; executing a compile-only word while interpreting;
a `DOES>` body with the wrong picture; redefining during compilation of that
definition; and UTF-8 trailing bytes that do not complete a character.

## Session runtime

`ForthSession` (`src/forth/forth_session.h`) owns:

- one mutable `NvmModule` and one `VmState` for the life of the session;
- data, return, floating-point, and compile-control stacks, separate from
  NanoVM's operand stack (`vm_invoke` must not clear them);
- a byte-addressable virtual space in `VmState` linear memory, with an
  allocation table so host pointers are not Forth addresses;
- a file-handle table with generation-checked ids;
- dictionary headers, name tokens, execution tokens, immediacy, hidden
  (smudged) names, and a search order of word lists;
- nested terminal, evaluated-string, included-file, and block input sources
  with `SOURCE` / `>IN` / `BLK` restoration.

Appending a function without `forth_session_rebuild` leaves decode stale.
`nvm_verify_function` is the check I will require before dictionary publish;
the session tests call it when they append a constant function.

## What this does not do

I do not run Jackson `runtests.fth` yet. There is no NanoISA Forth to include
it. I do not vendor the suites. I do not claim Core.

The next work is colon definitions compiled privately to NanoISA, verified,
then published, with `OP_CALL` early binding and `RECURSE` to the reserved
current definition.
