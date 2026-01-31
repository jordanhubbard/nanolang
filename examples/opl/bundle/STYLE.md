# STYLE.md — Formatting and Conventions

## OPL formatting
- One statement per line
- Spaces after commas and after `:`
- Group declarations at top of a block

## JSON formatting
- Pretty-print with 2-space indent
- Preserve ordering:
  - AST nodes in source order
  - Map entries as arrays (`entries`) to preserve insertion order

## Deterministic output
- Sort validation errors by (line, col, code, path)
- Plan steps in source order
