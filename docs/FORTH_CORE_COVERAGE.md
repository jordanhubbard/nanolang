# Forth 2012 Core Coverage

This is evidence, not a Core pass. I do not claim Core. I do not
claim a Forth 2012 Standard System. Jackson Core has not run: File
Access `INCLUDE` / `INCLUDED` are missing, and `make test-forth-jackson`
records that gap instead of executing `core.fr`.

The names are Forth 2012 §6.1 Core (133 words). Core Ext, Exception,
File Access, and other optional word sets are not in this table. `HEX`
is Core Ext; Hayes `core.fr` uses it, which is a suite dependency, not a
Core name.

Status:

- **tested** — the name is in the NanoISA session dictionary and a
  session unit test interprets it. That is not a Jackson result.
- **missing** — `FIND` of the name fails.
- **ambiguous** — `FIND` succeeds, but I have no Jackson Core case
  and no session interpret of that word.

## Counts

| Status | Count |
| --- | ---: |
| tested | 63 |
| ambiguous | 70 |
| missing | 0 |
| total Core names | 133 |

## Matrix

| Word | Status | Note |
| --- | --- | --- |
| `!` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `#` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `#>` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `#S` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `'` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `(` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `*` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `*/` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `*/MOD` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `+` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `+!` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `+LOOP` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `,` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `-` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `.` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `."` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `/` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `/MOD` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `0<` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `0=` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `1+` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `1-` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `2!` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `2*` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `2/` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `2@` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `2DROP` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `2DUP` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `2OVER` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `2SWAP` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `:` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `;` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `<` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `<#` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `=` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `>` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `>BODY` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `>IN` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `>NUMBER` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `>R` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `?DUP` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `@` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ABORT` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `ABORT"` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `ABS` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ACCEPT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ALIGN` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ALIGNED` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `ALLOT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `AND` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `BASE` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `BEGIN` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `BL` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `C!` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `C,` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `C@` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `CELL+` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `CELLS` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `CHAR` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `CHAR+` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `CHARS` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `CONSTANT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `COUNT` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `CR` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `CREATE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `DECIMAL` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `DEPTH` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `DO` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `DOES>` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `DROP` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `DUP` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ELSE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `EMIT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `ENVIRONMENT?` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `EVALUATE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `EXECUTE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `EXIT` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `FILL` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `FIND` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `FM/MOD` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `HERE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `HOLD` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `I` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `IF` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `IMMEDIATE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `INVERT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `J` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `KEY` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `LEAVE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `LITERAL` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `LOOP` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `LSHIFT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `M*` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `MAX` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `MIN` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `MOD` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `MOVE` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `NEGATE` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `OR` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `OVER` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `POSTPONE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `QUIT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `R>` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `R@` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `RECURSE` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `REPEAT` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `ROT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `RSHIFT` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `S"` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `S>D` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `SIGN` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `SM/REM` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `SOURCE` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `SPACE` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `SPACES` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `STATE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `SWAP` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `THEN` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `TYPE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `U.` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `U<` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `UM*` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `UM/MOD` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `UNLOOP` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `UNTIL` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `VARIABLE` | tested | In the session dictionary and exercised by `tests/forth/test_forth_session.c`. |
| `WHILE` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `WORD` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `XOR` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `[` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `[']` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `[CHAR]` | ambiguous | FIND would succeed. No Jackson Core case has run. |
| `]` | ambiguous | FIND would succeed. No Jackson Core case has run. |

`INCLUDE` and `INCLUDED` are File Access words. They are absent.
They do not belong in this Core table.

