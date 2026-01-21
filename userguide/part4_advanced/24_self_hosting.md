# Chapter 24: Self-Hosting

**NanoLang compiler written in NanoLang.**

NanoLang is self-hosting: the compiler is written in NanoLang itself and compiles to C.

## 24.1 Dual Implementation

**ALL language features require 2× implementation:**
- C reference compiler: `src/*.c`
- NanoLang self-hosted: `src_nano/*.nano`

This constraint drives language design:
- ✅ Simple, regular grammar
- ✅ Explicit over implicit
- ❌ Avoid complex features requiring parser changes

## 24.2 Compiler Pipeline

```
NanoLang source (.nano)
    ↓ lexer.nano
Tokens
    ↓ parser.nano
AST
    ↓ typecheck.nano
Typed AST
    ↓ transpiler.nano
C code (.c)
    ↓ gcc/clang
Native binary
```

## 24.3 Bootstrap Process

1. **Stage 0:** C compiler compiles itself
2. **Stage 1:** C compiler compiles NanoLang compiler
3. **Stage 2:** NanoLang compiler compiles itself
4. **Stage 3:** Verify stage 2 and stage 3 are identical

## 24.4 Compiler Modules

- `src_nano/compiler/lexer.nano` - Tokenization
- `src_nano/compiler/parser.nano` - AST construction
- `src_nano/compiler/typecheck.nano` - Type checking
- `src_nano/compiler/transpiler.nano` - C code generation

## Summary

Self-hosting benefits:
- ✅ "Dog-fooding" the language
- ✅ Ensures language is practical
- ✅ Tests every feature
- ✅ Drives simplicity

---

**Previous:** [Chapter 23: LLM Code Generation](23_llm_generation.md)  
**Next:** [Chapter 25: Contributing](25_contributing.md)
