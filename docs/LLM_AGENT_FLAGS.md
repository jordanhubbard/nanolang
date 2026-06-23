# My LLM / Agent-Only Compiler Flags

I am an LLM-first language. My compiler, `nanoc`, supports a set of agent-focused debugging and tracing affordances. I will continue to expand these as I grow.

## Stable user-facing flags (today)

- **`-v`, `--verbose`**: I provide more verbose output, including phase prints and extra debug artifacts.
- **`-k`, `--keep-c`**: I keep the C code I generate.
- **`-fshow-intermediate-code`**: I print the generated C code to stdout.
- **`-h`, `--help`**: I show my help menu.

## Always-on debug artifacts (today)

- **`/tmp/merged_debug.nano`**: I may write my merged import graph as a single file. You can use this to inspect how I map diagnostics that report merged line numbers.

## Reserved namespace for agent-only flags

I reserve flags prefixed with **`--llm-`** for automated agents. These flags may change quickly as I evolve.

### Implemented

- **`--llm-diags-json <path>`**: I emit machine-readable diagnostics as JSON. This is a best-effort operation. I do not fail compilation if I cannot write the file.
- **`--llm-diags-toon <path>`**: I emit diagnostics in TOON format. This uses about 40% fewer tokens than JSON. See https://toonformat.dev/
- **`--llm-shadow-json <path>`**: I emit a machine-readable summary of shadow-test failures as JSON. This is a best-effort operation.

Schema (stable keys):
- `tool`: `"nanoc_c"` or `"nanoc"` (my self-hosted version)
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

I will implement these flags incrementally as I need them:

- **`--llm-dump-merged <path>`**: I will allow you to control where I write the merged program, instead of using a fixed path.
- **`--llm-dump-ast-json <path>`**: I will emit my parsed AST as JSON for debugging parser or typechecker issues.
- **`--llm-dump-plan-json <path>`**: I will emit my internal compilation plan and phase outputs as JSON.
- **`--llm-trace <topic>`**: I will enable targeted trace streams for topics like `imports`, `typecheck`, or `transpile`. These are intended for automated consumption.

## Design rule

When I add a new agent-only flag, I follow these principles:

- I prefer machine-readable output like JSON over text meant only for humans.
- I keep my outputs deterministic with stable ordering and keys.
- I ensure my outputs can be used to auto-file beads. They are fingerprintable and actionable.

