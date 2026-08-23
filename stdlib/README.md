# My Root Standard Library

`stdlib/` contains a small set of legacy and tool-oriented modules. My supported general-purpose modules live under `modules/std/`; built-ins need no import.

## Usage

Import a root module by its path only when you need that specific tool. For example, `stdlib/tidy.nano` backs my source formatter.

## Modules

| Module | Purpose |
|--------|---------|
| `ast.nano` | AST helpers |
| `coverage.nano` | Coverage support |
| `log.nano` | Legacy logging helpers |
| `mac.nano` | MAC integration |
| `tidy.nano` | Source formatting library |
| `tidy_cli.nano` | Source formatting command |
| `timing.nano` | Timing helpers |

## Design Philosophy

I do not provide root `async.nano`, `iter.nano`, `list.nano`, `map.nano`, `option.nano`, `result.nano`, `set.nano`, or `string.nano` modules. Those unsupported dialect files were deleted rather than left where an import could look plausible.

For supported collections and result handling, use `modules/std/collections/` and `modules/std/result.nano`. Strings and arrays are built in. I document what exists; an attractive import path is not an implementation.
