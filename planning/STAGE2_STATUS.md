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
- [x] Match expression codegen - **fully implemented** in `src_nano/transpiler.nano` lines 2041-2115
- [ ] Match codegen in C transpiler - **BLOCKING** (see nanolang-bjd)
- [ ] Generic union monomorphization - not yet implemented
- [ ] Tuple codegen - not yet implemented

## Critical Blocker

**Match expressions cannot be compiled by the C reference transpiler.**

The C transpiler emits `/* unsupported expr type 26 */` for AST_MATCH nodes (see nanolang-bjd).
This prevents compiling any program (including tests) that uses match expressions, even though:
- The self-hosted typechecker fully supports match typing
- The self-hosted transpiler has complete match codegen implementation

Once match codegen is implemented in the C transpiler, match expressions will work end-to-end.

## What Remains

### High Priority (Blocking Stage 2)
1. **Match codegen in C transpiler** (nanolang-bjd) - must be implemented

### Medium Priority (Enhances Stage 2)
2. Generic union instantiation parsing - enables `Result<T,E>` in user code
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

- `83b9911` - selfhost: add union definition stub to transpiler
- `1475c65` - selfhost: add union and match expression type checking
- `a74e816` - selfhost: add List<T> and array typing support

---
Last Updated: 2025-12-24
