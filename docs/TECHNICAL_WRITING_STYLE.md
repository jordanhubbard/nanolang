# Technical Writing Style Guide for NanoLang Documentation

**Standards for writing clear, consistent, example-driven documentation.**

---

## Core Principles

### 1. Example-Driven Documentation

**Every concept must have working examples with shadow tests.**

❌ **Bad:**
```markdown
The `map` function applies a transformation to each element.
```

✅ **Good:**
```markdown
The `map` function applies a transformation to each element:

```nano
fn double(x: int) -> int {
    return (* x 2)
}

fn map_example() -> array<int> {
    let numbers: array<int> = [1, 2, 3, 4, 5]
    return (array_map numbers double)
}

shadow map_example {
    let result: array<int> = (map_example)
    assert (== (array_get result 0) 2)
    assert (== (array_get result 4) 10)
}
```
```

### 2. Progressive Disclosure

Start simple, build complexity:
1. **Minimal example** - Simplest possible use
2. **Practical example** - Real-world use case
3. **Complete example** - With error handling
4. **Advanced example** - Complex patterns (if needed)

### 3. One Canonical Way

Always show the canonical NanoLang syntax. Never show alternatives or deprecated forms.

❌ **Bad:**
```markdown
You can concatenate strings with `+` or `str_concat`:
```

✅ **Good:**
```markdown
Concatenate strings using the `+` operator:
```nano
let greeting: string = "Hello, " + name
```
```

---

## Voice & Tone

### Active Voice

Prefer active voice to make instructions clear.

❌ Passive: "The file is compiled by the compiler"  
✅ Active: "The compiler compiles the file"

❌ Passive: "An error will be returned if..."  
✅ Active: "The function returns an error if..."

### Second Person

Address the reader directly.

❌ "One should always use shadow tests"  
❌ "We recommend using shadow tests"  
✅ "You should always use shadow tests"

❌ "The developer can choose..."  
✅ "You can choose..."

### Present Tense

Use present tense for current behavior.

❌ "The function will return an integer"  
✅ "The function returns an integer"

❌ "This would cause an error"  
✅ "This causes an error"

### Encouraging & Positive

Focus on what you CAN do, not just restrictions.

❌ "You must use prefix notation for all operations"
✅ "Use either infix `a + b` or prefix `(+ a b)` notation for operators"

❌ "Don't forget to free memory"  
✅ "Free resources when done to prevent memory leaks"

---

## Code Examples

### Every Example Must Have

1. **Imports** - Always show what to import
2. **Types** - Always annotate types explicitly
3. **Shadow tests** - Prove it works
4. **Comments** - Explain non-obvious logic
5. **Canonical syntax** - Infix for operators, prefix for function calls

### Example Structure

```nano
# Import statements at top
from "module/path.nano" import function_name

# Type definitions (if any)
struct MyType {
    field: int
}

# Functions with clear names
fn descriptive_name(param: Type) -> ReturnType {
    # Comment explaining non-obvious logic
    let result: ReturnType = (operation param)
    return result
}

# Shadow test immediately after
shadow descriptive_name {
    # Test happy path
    assert (== (descriptive_name 5) expected_value)
    
    # Test edge cases
    assert (== (descriptive_name 0) edge_case_value)
}
```

### Example Naming

Use descriptive names that explain what the example demonstrates:

❌ `example1`, `test`, `foo`  
✅ `email_validation`, `parse_json_response`, `calculate_fibonacci`

### Comment Guidelines

**When to comment:**
- Non-obvious algorithms
- Magic numbers or special values
- Why something is done a certain way

**When NOT to comment:**
- Self-explanatory code
- Restating what's obvious from names

❌ **Bad:**
```nano
# Add x and y
let sum: int = (+ x y)
```

✅ **Good:**
```nano
# Use Euclidean distance for 2D points
let distance: float = (sqrt (+ (* dx dx) (* dy dy)))
```

---

## Parameter & Return Value Descriptions

### Parameter Descriptions

Include context, constraints, and valid values.

❌ **Bad:**
- `text` - The text parameter
- `n` - An integer
- `callback` - A callback function

✅ **Good:**
- `text` - String to search. Can be empty (returns `false`) or contain newlines (matches line-by-line)
- `n` - Number of elements to process. Must be >= 0, values < 0 are treated as 0
- `callback` - Function called for each element. Receives element value and index

### Return Value Descriptions

Explain what the return value represents and special cases.

❌ **Bad:**
- **Returns:** An integer

✅ **Good:**
- **Returns:** Number of matches found, or -1 if pattern is invalid
- **Returns:** `true` if file exists and is readable, `false` otherwise
- **Returns:** New array with transformed elements. Original array is unchanged

---

## Heading & Structure

### Heading Hierarchy

```markdown
# Chapter Title (H1 - once per file)

## Major Section (H2)

### Function Name or Subsection (H3)

#### Details or Examples (H4 - use sparingly)
```

### Section Order

For API functions, follow this order:

1. Brief description (1-2 sentences)
2. **Signature** block
3. **Parameters** list
4. **Returns** description
5. **Example 1** (minimal)
6. **Example 2** (practical)
7. **See Also** links
8. **Common Pitfalls** section

---

## Visual Elements

### Call-Out Boxes

Use icons consistently:

```markdown
✅ **Try This:** Working example you can run
⚠️ **Watch Out:** Common pitfalls and warnings
💡 **Pro Tip:** Best practices and optimization hints
❌ **Don't Do This:** Anti-patterns to avoid
```

### Code vs Prose

**In code blocks:**
- Use actual code (compilable)
- Include shadow tests
- Show complete examples

**In prose:**
- Use inline `code` for: function names, variable names, keywords, values
- Use **bold** for: emphasis, file names, important terms
- Use *italic* for: book titles, minimal emphasis

### Lists

Use bullets for unordered information:
```markdown
- Item one
- Item two
- Item three
```

Use numbers for sequential steps:
```markdown
1. First, do this
2. Then, do this
3. Finally, do this
```

---

## Common Patterns

### Introducing a Feature

```markdown
## Feature Name

Brief explanation of what it does and why it's useful.

**Basic Usage:**
```nano
[Minimal example]
```

**When to Use:** Bullet points explaining use cases.
```

### Comparing Approaches

Use tables for comparisons:

```markdown
| Approach | Pros | Cons | Use When |
|----------|------|------|----------|
| Method A | Fast | Limited | Small data |
| Method B | Flexible | Slower | Complex needs |
```

### Showing Evolution

Progress from simple to complex:

```markdown
**Version 1: Basic**
[Simple code]

**Version 2: With Error Handling**
[Code with error checks]

**Version 3: Production-Ready**
[Complete implementation]
```

---

## Terminology Standards

Use these terms consistently:

### NanoLang-Specific

- **infix notation** for operators: `a + b` (preferred), prefix `(+ a b)` also supported
- **prefix notation** for function calls: `(println "hello")` (not "S-expressions", "Lisp-like")
- **shadow test** (not "unit test", "spec", "example")
- **function** (not "procedure", "method", "subroutine")
- **module** (not "library", "package")
- **struct** (not "record", "object")
- **enum** (not "enumeration", though "enumeration" in prose is OK)
- **union** (not "tagged union" in headings, though explaining as "tagged union" is OK)
- **transpile** (not "compile to C", though explaining transpilation is OK)

### Avoid Jargon

Replace technical jargon with plain English when possible:

❌ "Memoize the computation"  
✅ "Cache the result to avoid recomputation"

❌ "Idempotent operation"  
✅ "Operation that produces the same result when called multiple times"

When jargon is necessary, define it on first use.

---

## Navigation & Cross-References

### Internal Links

Always use relative paths:

```markdown
See [Chapter 5: Control Flow](../part1_fundamentals/05_control_flow.md)
```

### See Also Sections

```markdown
**See Also:**
- `related_function()` - Brief description
- [Related Concept](link.md) - Chapter reference
- [External Resource](https://example.com) - When necessary
```

### Previous/Next Navigation

Every chapter should end with:

```markdown
---

**Previous:** [Chapter N](link.md)  
**Next:** [Chapter N+2](link.md)
```

---

## Review Checklist

Before submitting documentation, verify:

### Content
- [ ] All examples compile and pass shadow tests
- [ ] At least 2 examples per function (minimal + practical)
- [ ] Parameters have contextual descriptions
- [ ] Return values explained with edge cases
- [ ] "Common Pitfalls" section included
- [ ] "See Also" section has relevant links

### Style
- [ ] Active voice used throughout
- [ ] Second person ("you") used
- [ ] Present tense for current behavior
- [ ] Consistent terminology from this guide
- [ ] No placeholder text (TODO, TBD, etc.)

### Technical
- [ ] Only canonical syntax shown
- [ ] Operators use infix notation (preferred) or prefix notation
- [ ] Imports included in examples
- [ ] Types explicitly annotated
- [ ] Shadow tests for all examples

### Structure
- [ ] Heading hierarchy is correct
- [ ] Navigation links work
- [ ] Code blocks have language markers
- [ ] Visual elements used consistently

---

## Writing Process

### 1. Research Phase
- Review existing examples in `examples/` directory
- Check `docs/CANONICAL_STYLE.md` for syntax
- Look at `modules/index.json` for module info
- Read related documentation

### 2. Outline Phase
- List all functions/concepts to cover
- Organize by complexity (simple → complex)
- Plan example use cases
- Identify related functions for "See Also"

### 3. Draft Phase
- Write minimal examples first
- Add shadow tests immediately
- Write descriptions after code works
- Include "Common Pitfalls" from experience

### 4. Review Phase
- Compile and test all examples
- Check links and navigation
- Verify consistency with this guide
- Get peer review if possible

### 5. Polish Phase
- Add visual elements (✅ ⚠️ 💡 ❌)
- Improve transitions between sections
- Fix typos and grammar
- Final read-through

---

## Example: Complete Function Documentation

Here's a fully-worked example following all guidelines:

### `str_contains()` - Check if Substring Exists

Check if a string contains a substring. This is the simplest way to test for substring presence.

**Signature:**
```nano
fn str_contains(haystack: string, needle: string) -> bool
```

**Parameters:**
- `haystack` - String to search within. Can be empty (always returns `false`)
- `needle` - Substring to search for. Empty string always matches

**Returns:** `true` if `needle` appears anywhere in `haystack`, `false` otherwise

**Example: Simple Substring Check**
```nano
fn has_file_extension(filename: string) -> bool {
    return (str_contains filename ".txt")
}

shadow has_file_extension {
    assert (has_file_extension "document.txt")
    assert (not (has_file_extension "document.pdf"))
    assert (not (has_file_extension "txt"))
}
```

This example shows basic substring detection for file extensions.

**Example: Input Validation**
```nano
fn is_valid_email_domain(email: string) -> bool {
    # Accept common domains
    return (or
        (str_contains email "@gmail.com")
        (str_contains email "@yahoo.com")
    )
}

shadow is_valid_email_domain {
    assert (is_valid_email_domain "user@gmail.com")
    assert (is_valid_email_domain "test@yahoo.com")
    assert (not (is_valid_email_domain "user@example.com"))
}
```

For more complex pattern matching, use the `regex` module.

**See Also:**
- `str_starts_with()` - Check if string starts with prefix
- `str_ends_with()` - Check if string ends with suffix
- `str_index_of()` - Find position of substring
- [Chapter 13: Text Processing](../userguide/part3_modules/13_text_processing/index.md) - Pattern matching with regex

**Common Pitfalls:**
- ⚠️ Case-sensitive: `"Hello"` does not contain `"hello"`
- ⚠️ Empty needle always matches (returns `true`)
- 💡 Tip: For case-insensitive search, convert both strings to lowercase first
- 💡 Tip: For complex patterns, use `regex` module instead

---

## Questions & Answers

**Q: How long should examples be?**  
A: Minimal examples: 5-15 lines. Practical examples: 15-30 lines. Complete applications: can be longer.

**Q: Should I show error cases?**  
A: Yes, in "Common Pitfalls" section. Show what NOT to do.

**Q: How many examples per function?**  
A: Minimum 2 (minimal + practical). Add more for complex functions.

**Q: What if a function has no good practical example?**  
A: Reconsider if the function should exist. Good functions solve real problems.

**Q: Can I reference external documentation?**  
A: Sparingly. Prefer self-contained examples. Link to official docs for C libraries.

**Q: Should I document deprecated features?**  
A: No. Remove deprecated documentation and redirect to canonical approach.

---

## Resources

- `docs/CANONICAL_STYLE.md` - Syntax reference
- `docs/LLM_CORE_SUBSET.md` - Core 50-primitive subset
- `docs/API_REFERENCE_TEMPLATE.md` - Templates for API docs
- `examples/` - Working code examples
- `MEMORY.md` - Patterns and idioms

---

**This style guide is mandatory for all NanoLang documentation contributors.**
