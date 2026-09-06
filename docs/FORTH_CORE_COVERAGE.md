# Forth 2012 Core Coverage

This is evidence, not a Core pass. I do not claim Core. I do not
claim a Forth 2012 Standard System. Jackson Core evidence files pass
under `make test-forth-core` through C file-source `REFILL`. File
Access `INCLUDE` / `INCLUDED` exist as File Access words. Core evidence
still loads through C `REFILL`, and `make test-forth-jackson` still
records that Core evidence is not Forth `INCLUDED`.

The names are Forth 2012 §6.1 Core (133 words). Core Ext, Exception,
File Access, and other optional word sets are not in this table. `HEX`
is Core Ext; Hayes `core.fr` uses it, which is a suite dependency, not a
Core name.

Status:

- **tested** — `FIND` succeeds and Jackson Core evidence or a session
  unit test has interpreted the word.
- **missing** — `FIND` of the name fails.
- **ambiguous** — `FIND` succeeds, but I have no Jackson Core case
  and no session interpret of that word.

## Counts

| Status | Count |
| --- | ---: |
| tested | 133 |
| ambiguous | 0 |
| missing | 0 |
| total Core names | 133 |

## Matrix

| Word | Status | Note |
| --- | --- | --- |
| `!` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `#` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `#>` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `#S` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `'` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `(` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `*` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `*/` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `*/MOD` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `+` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `+!` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `+LOOP` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `,` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `-` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `.` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `."` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `/` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `/MOD` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `0<` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `0=` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `1+` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `1-` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `2!` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `2*` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `2/` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `2@` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `2DROP` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `2DUP` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `2OVER` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `2SWAP` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `:` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `;` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `<` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `<#` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `=` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `>` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `>BODY` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `>IN` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `>NUMBER` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `>R` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `?DUP` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `@` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ABORT` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `ABORT"` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `ABS` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ACCEPT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ALIGN` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ALIGNED` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `ALLOT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `AND` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `BASE` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `BEGIN` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `BL` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `C!` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `C,` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `C@` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `CELL+` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `CELLS` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `CHAR` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `CHAR+` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `CHARS` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `CONSTANT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `COUNT` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `CR` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `CREATE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `DECIMAL` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `DEPTH` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `DO` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `DOES>` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `DROP` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `DUP` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ELSE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `EMIT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ENVIRONMENT?` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `EVALUATE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `EXECUTE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `EXIT` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `FILL` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `FIND` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `FM/MOD` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `HERE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `HOLD` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `I` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `IF` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `IMMEDIATE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `INVERT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `J` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `KEY` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `LEAVE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `LITERAL` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `LOOP` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `LSHIFT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `M*` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `MAX` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `MIN` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `MOD` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `MOVE` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `NEGATE` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `OR` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `OVER` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `POSTPONE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `QUIT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `R>` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `R@` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `RECURSE` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `REPEAT` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `ROT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `RSHIFT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `S"` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `S>D` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `SIGN` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `SM/REM` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `SOURCE` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `SPACE` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `SPACES` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `STATE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `SWAP` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `THEN` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `TYPE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `U.` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `U<` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `UM*` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `UM/MOD` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `UNLOOP` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `UNTIL` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `VARIABLE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `WHILE` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `WORD` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `XOR` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `[` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `[']` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `[CHAR]` | tested | Jackson Core evidence files pass via C file-source REFILL. |
| `]` | tested | Jackson Core evidence files pass via C file-source REFILL. |

`INCLUDE` and `INCLUDED` are File Access words. They do not belong in
this Core table. Core evidence still loads through C `REFILL`.

