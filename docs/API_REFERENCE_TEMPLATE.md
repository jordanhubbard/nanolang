# API Reference Template

**How to write API documentation following the "NanoLang by Example" format.**

This template shows the standard format for documenting NanoLang functions, types, and modules. Every API reference should follow this structure.

---

## Template Structure

### Function Documentation Template

```markdown
### `function_name()` - Brief One-Line Description

Brief paragraph explaining what this function does and when to use it (2-3 sentences).

**Signature:**
```nano
fn function_name(param1: Type1, param2: Type2) -> ReturnType
```

**Parameters:**
- `param1` - Description with context about what values are valid and what they mean
- `param2` - Description with context (not just "the second parameter")

**Returns:** Description of what the function returns, including any special values (null, -1 for errors, etc.)

**Example: [Use Case Name]**
```nano
from "module/path.nano" import function_name

fn example() -> ReturnType {
    let result: ReturnType = (function_name arg1 arg2)
    return result
}

shadow example {
    assert (condition)
}
```

Brief explanation of what this example demonstrates.

**Example: [Another Use Case]**
```nano
[Another complete, runnable example]
```

Brief explanation.

**See Also:**
- `related_function1()` - Brief description
- `related_function2()` - Brief description
- [Related Chapter](../link.md) - When relevant

**Common Pitfalls:**
- ⚠️ Warning about common mistake #1
- ⚠️ Warning about common mistake #2
- 💡 Tip: Best practice or optimization hint
```

---

## Complete Example: regex.matches()

Here's a complete example following the template:

### `matches()` - Test if Pattern Matches

Check if a regular expression matches anywhere in a string. This is the simplest way to test for pattern presence.

**Signature:**
```nano
fn matches(regex: Regex, text: string) -> bool
```

**Parameters:**
- `regex` - Compiled regular expression (from `compile()`)
- `text` - String to test against the pattern

**Returns:** `true` if the pattern matches anywhere in the text, `false` otherwise

**Example: Email Validation**
```nano
from "modules/regex/regex.nano" import compile, matches, free

fn is_email(s: string) -> bool {
    let pattern: Regex = (compile "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
    let result: bool = (matches pattern s)
    (free pattern)
    return result
}

shadow is_email {
    assert (is_email "user@example.com")
    assert (not (is_email "not-an-email"))
    assert (is_email "test.user+tag@sub.domain.com")
}
```

This example shows basic email validation using a compiled regex pattern.

**Example: Quick Pattern Check**
```nano
from "modules/regex/regex.nano" import quick_match

fn has_digits(s: string) -> bool {
    return (quick_match "[0-9]+" s)
}

shadow has_digits {
    assert (has_digits "abc123")
    assert (not (has_digits "abcdef"))
}
```

For one-time matches, `quick_match()` is more convenient (no manual cleanup).

**See Also:**
- `compile()` - Compile a regex pattern for reuse
- `quick_match()` - One-shot matching without explicit compilation
- `find()` - Find position of first match
- `find_all()` - Find all match positions

**Common Pitfalls:**
- ⚠️ Don't forget to call `free()` on compiled regexes to avoid memory leaks
- ⚠️ Backslashes must be escaped: use `"\\d"` not `"\d"`
- 💡 Tip: Use `quick_match()` for one-time patterns; use `compile()` when matching repeatedly

---

## Module Index Page Template

For module index pages (e.g., `part3_modules/13_text_processing/index.md`):

```markdown
# Chapter X: Module Category Name

**Brief description of what these modules provide.**

This chapter covers [number] modules for [purpose]:

- **module1** - One-line description
- **module2** - One-line description
- **module3** - One-line description

## When to Use These Modules

**Use module1 when:**
- Specific use case 1
- Specific use case 2
- Specific use case 3

**Use module2 when:**
- Specific use case 1
- Specific use case 2

## Quick Start

Brief overview of getting started with these modules.

---

**Sections:**
- [X.1 module1 - Full Name](module1.md)
- [X.2 module2 - Full Name](module2.md)

---

**Previous:** [Chapter Y](../link.md)  
**Next:** [Chapter Z](../link.md)
```

---

## Writing Guidelines

### Voice & Tone
- **Active voice:** "You call the function" not "The function is called"
- **Second person:** "You" not "we" or "one"
- **Present tense:** "The function returns" not "will return"
- **Encouraging:** Focus on what you CAN do

### Example Quality
Every example MUST:
1. ✅ Be complete and runnable (include imports)
2. ✅ Have shadow tests that pass
3. ✅ Use canonical NanoLang syntax only
4. ✅ Include comments for non-obvious code
5. ✅ Demonstrate realistic use cases

### Example Categories
- **Minimal:** Smallest possible demonstration of the feature
- **Practical:** Real-world use case showing why/when to use it
- **Complete:** Full program with error handling and edge cases
- **Advanced:** Complex pattern or optimization technique

### Parameter Descriptions
❌ **Bad:** `text` - The text to search  
✅ **Good:** `text` - String to test against the pattern. Can be empty (returns `false`) or contain newlines (matches line-by-line).

### See Also Links
Always include:
- Related functions in the same module
- Alternative approaches
- Relevant chapters for concepts

### Common Pitfalls Section
Include:
- ⚠️ Warnings about mistakes beginners make
- 💡 Tips for better performance or cleaner code
- Memory management reminders (free, cleanup)
- Edge cases to handle

---

## Type Documentation Template

For documenting structs, enums, and unions:

```markdown
### `TypeName` - Brief Description

Description of what this type represents.

**Definition:**
```nano
struct TypeName {
    field1: Type1,
    field2: Type2
}
```

**Fields:**
- `field1` - Description with constraints or valid values
- `field2` - Description

**Example: Creating and Using TypeName**
```nano
[Complete example with shadow test]
```

**See Also:**
- Related types
- Functions that use this type
```

---

## Module Documentation Template

For complete module documentation files:

```markdown
# X.Y module_name - Full Module Name

**One-line description of the module's purpose.**

## Quick Start

Minimal example showing basic usage:
```nano
[5-10 line example]
```

## Installation

How to install dependencies (if any):
```bash
# Commands to install
```

## Core Concepts

Brief explanation of key concepts (if needed).

## API Reference

### function1()

[Full function documentation using template above]

### function2()

[Full function documentation using template above]

## Complete Examples

### [Real-World Use Case Name]

[Larger, complete example showing typical usage]

## Best Practices

- Best practice 1
- Best practice 2
- Best practice 3

---

**Navigation links**
```

---

## Checklist for API Documentation

Before submitting API documentation, verify:

- [ ] Function signature is correct and compilable
- [ ] All examples have shadow tests
- [ ] All examples use canonical syntax (prefix notation only)
- [ ] Parameters have contextual descriptions (not just types)
- [ ] At least 2 examples provided (minimal + practical)
- [ ] "See Also" section includes related functions
- [ ] "Common Pitfalls" section has useful warnings/tips
- [ ] All code blocks have proper syntax highlighting markers
- [ ] Navigation links (Previous/Next) are correct
- [ ] No placeholder text remains (TODO, TBD, etc.)

---

## File Naming Conventions

- Module files: lowercase, matching module name (`regex.md`, `sqlite.md`)
- Index files: always `index.md`
- Chapter files: numbered (`01_getting_started.md`)
- Appendix files: lettered (`a_examples_gallery.md`)

---

## Markdown Conventions

### Code Blocks
Always specify language:
\`\`\`nano
(+ 1 2)
\`\`\`

\`\`\`bash
make test
\`\`\`

### Internal Links
Use relative paths:
```markdown
[Chapter 5](../part1_fundamentals/05_control_flow.md)
```

### Emphasis
- **Bold** for function names, types, file names in prose
- `code` for inline code, variables, values
- *Italic* sparingly for emphasis

---

**This template is the standard for all NanoLang API documentation.**
