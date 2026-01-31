# Stage 2 Self-Hosting Status

## Goal
The nanolang compiler (written in nanolang) can compile the entire nanolang ecosystem 
without any dependency on the C reference compiler, using all language features.

## Progress Summary

### ✅ Type System Parity (nanolang-alp.5)
- [x] List<T> typing - parse `List<T>` syntax, type check generic operations
- [x] Array typing - `array<T>` with element type tracking
- [x] Union typing - basic union construction and match expression types
- [x] Match expression typing - pattern binding and arm type checking in `src_nano/typecheck.nano`
- [ ] Generic union instantiation - `Result<int, string>` syntax not yet parsed
- [ ] Tuple types - not yet implemented
- [ ] First-class functions - not yet implemented

### ✅ Transpiler Parity (nanolang-alp.8)
- [x] List<T> monomorphization - detect instantiations, generate specialized code
- [x] List<T> codegen - `List_T` struct and functions (new, push, get, length)
- [x] Union codegen - basic union type definitions
- [x] Match expression codegen - **fully implemented** in both transpilers
  - Self-hosted: `src_nano/transpiler.nano` lines 2041-2115
  - C reference: `src/transpiler_iterative_v3_twopass.c` (commit 4dda4a2)
- [ ] Generic union monomorphization - not yet implemented
- [ ] Tuple codegen - not yet implemented

## Current Blocker

**Match arm bindings not in typechecker scope** (nanolang-wsc)

The typechecker doesn't add match arm bindings to the symbol table, causing errors like:
- `Undefined variable 's'` in nested expressions like `(* s.value 2)`
- `Field access requires a struct` for binding field access

This prevents `tests/nl_control_match.nano` from compiling, even though:
- Match expression transpilation works correctly
- Simple match cases (direct field access) compile fine

## What Remains

### High Priority (Blocking Stage 2)
1. **Match arm binding scope** (nanolang-wsc) - typechecker must add bindings to scope

### Medium Priority (Enhances Stage 2)
2. Generic union instantiation parsing (nanolang-nh1) - enables `Result<T,E>` syntax
3. Generic union monomorphization - codegen for instantiated generic unions

### Lower Priority (Full feature parity)
4. Tuple types - typing and codegen
5. First-class functions/closures - typing and codegen

## Testing Status

Current self-hosted compiler source (`src_nano/`) uses:
- ✅ List<T> - working
- ❌ Match expressions - not used in compiler source yet
- ❌ Generic unions - not used in compiler source yet
- ❌ Tuples - not used in compiler source yet
- ❌ First-class functions - not used in compiler source yet

This means the compiler can already compile itself with current features. Additional
features are needed for compiling user programs that use advanced language features.

## Recent Commits

- `4dda4a2` - transpiler: implement match expression codegen in C transpiler ✅
- `c9dcb0f` - docs: add Stage 2 self-hosting status summary
- `83b9911` - selfhost: add union definition stub to transpiler
- `1475c65` - selfhost: add union and match expression type checking
- `a74e816` - selfhost: add List<T> and array typing support

---
Last Updated: 2025-12-24 (Session 2)
