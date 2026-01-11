# LLM / Agent-Only Compiler Flags

NanoLang is an LLM-first language, so `nanoc` supports (and will continue to grow) a small set of **agent-focused** debugging and tracing affordances.

## Stable user-facing flags (today)

- **`-v`, `--verbose`**: More verbose compiler output (phase prints, extra debug artifacts).
- **`-k`, `--keep-c`**: Keep the generated C output.
- **`-h`, `--help`**: Help.

## Always-on debug artifacts (today)

- **`/tmp/merged_debug.nano`**: The compiler may write the merged import graph as a single file for inspection. This is especially useful for mapping diagnostics that currently report merged line numbers.

## Reserved namespace for agent-only flags

Flags prefixed with **`--llm-`** are reserved for automated agents and may change quickly.

### Implemented

- **`--llm-diags-json <path>`**: Emit machine-readable diagnostics as JSON (best-effort; does not fail compilation if the file cannot be written).
- **`--llm-shadow-json <path>`**: Emit a machine-readable summary of shadow-test failures as JSON (best-effort).

Schema (stable keys):
- `tool`: `"nanoc_c"` or `"nanoc"` (self-hosted)
- `success`: boolean
- `exit_code`: int
- `input_file`: string
- `output_file`: string
- `diagnostics`: array of objects:
  - `code`, `message`
  - `phase`, `phase_name`
  - `severity`, `severity_name`
  - `location`: `{ file, line, column }`

Shadow-test schema (stable keys):
- `tool`: `"nanoc_c"`
- `success`: boolean
- `failures`: array of objects:
  - `test`: string
  - `fail_count`: int
  - `first_location`: `{ line, column }`

### Planned / expected

Flags in this namespace (implement incrementally as needed):

- **`--llm-dump-merged <path>`**: Control where the merged program is written (instead of a fixed `/tmp/...` path).
- **`--llm-dump-ast-json <path>`**: Emit the parsed AST as JSON (debugging parser/typechecker issues).
- **`--llm-dump-plan-json <path>`**: Emit internal compilation plan / phase outputs as JSON.
- **`--llm-trace <topic>`**: Enable targeted trace streams (e.g. `imports`, `typecheck`, `transpile`) intended for automated consumption.

## Design rule

When adding a new agent-only flag:

- Prefer **machine-readable output** (JSON) over human-only text.
- Keep outputs **deterministic** (stable ordering, stable keys).
- Ensure outputs can be used to **auto-file beads** (fingerprintable + actionable).

