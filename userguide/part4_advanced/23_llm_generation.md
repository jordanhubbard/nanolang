# Chapter 23: LLM Code Generation

**Best practices for AI-generated NanoLang code.**

NanoLang is designed specifically for LLM code generation. Follow these principles for reliable AI-generated code.

## 23.1 Core Subset First

Use ONLY the 50-primitive core subset unless explicitly requested:
- Types: `int`, `float`, `string`, `bool`, `array<T>`, `void`
- Operations: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`
- Control: `if/else`, `while`, `for-in-range`, `cond`
- Functions: `println`, `array_get`, `array_set`, `array_length`

## 23.2 Always Use Shadow Tests

```nano
fn my_function(x: int) -> int {
    return (* x 2)
}

# ✅ ALWAYS include shadow tests
shadow my_function {
    assert (== (my_function 5) 10)
    assert (== (my_function 0) 0)
}
```

## 23.3 Never Invent Syntax

```nano
# ✅ Verify syntax exists
# Check docs/CANONICAL_STYLE.md
# Search examples/

# ❌ Don't guess or invent
# Don't write: arr[i], a + b, if expression
```

## 23.4 Examples-First Development

1. Search for closest example in target module
2. Copy it as starting point
3. Adapt to user's goal
4. Add shadow tests
5. Compile and iterate

## Summary

**LLM Generation Rules:**
- ✅ Core subset by default
- ✅ Shadow tests mandatory
- ✅ Verify syntax, never invent
- ✅ Copy examples, adapt

**See also:** `AGENTS.md`, `docs/LLM_CORE_SUBSET.md`

---

**Previous:** [Chapter 22: Canonical Style](22_canonical_style.html)  
**Next:** [Chapter 24: Self-Hosting](24_self_hosting.html)
