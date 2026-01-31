# Chapter 25: Contributing

**How to contribute to NanoLang.**

Contributions are welcome! Follow these guidelines.

## 25.1 Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/nanolang.git
cd nanolang

# Build
make clean
make

# Run tests
make test
```

## 25.2 Code Standards

**All changes must:**
- ✅ Compile with `-Wall -Wextra -Werror`
- ✅ Pass all existing tests
- ✅ Include shadow tests for new functions
- ✅ Follow canonical style
- ✅ Update documentation

## 25.3 Adding Features

**Before proposing language features, consider:**
- Is it worth 2× implementation cost? (C + NanoLang)
- Can it be a library function instead?
- Does it maintain LLM-first design?

**Prefer:**
- ✅ Library functions
- ✅ Simple, regular grammar
- ✅ Explicit constructs

**Avoid:**
- ❌ Syntax sugar
- ❌ Complex type inference
- ❌ Multiple ways to do same thing

## 25.4 Pull Request Process

1. Create feature branch
2. Make changes
3. Add tests
4. Update docs
5. Commit with descriptive message
6. Open PR with summary

## 25.5 Testing Requirements

```nano
# ✅ Every function needs shadow tests
fn new_feature(x: int) -> int {
    return (* x 2)
}

shadow new_feature {
    assert (== (new_feature 5) 10)
    assert (== (new_feature 0) 0)
    assert (== (new_feature -3) -6)
}
```

## Summary

**Contributing checklist:**
- ✅ Follow code standards
- ✅ Include shadow tests
- ✅ Update documentation
- ✅ Consider 2× implementation cost
- ✅ Maintain LLM-first design

**See also:** `CONTRIBUTING.md`, `.factory/PROJECT_RULES.md`

---

**Previous:** [Chapter 24: Self-Hosting](24_self_hosting.html)  
**Next:** [Appendix A: Examples Gallery](../appendices/a_examples_gallery.html)
