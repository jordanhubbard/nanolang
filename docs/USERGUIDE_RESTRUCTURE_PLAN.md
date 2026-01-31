# NanoLang User Guide Restructure Plan
## Professional Technical Writing & "NanoLang by Example" Theme

**Author**: Technical Documentation Team  
**Date**: 2026-01-20  
**Status**: PROPOSAL - Awaiting Approval

---

## Executive Summary

The current user guide has **bare API references with no examples** and **inconsistent chapter structure**. This plan transforms it into a professional, example-driven reference following the "**NanoLang by Example**" theme.

### Key Problems Identified
1. ✗ API references are auto-generated with NO examples, NO descriptions
2. ✗ Sidebar structure is inconsistent (numbered chapters vs unnumbered API refs)
3. ✗ No progressive learning path for modules
4. ✗ Code samples lack consistent formatting/colorization
5. ✗ Missing "cookbook" style practical examples

---

## Proposed Structure

### Part I: Language Fundamentals (Chapters 1-8)
**Theme**: Learn the core language, one concept at a time

```
1. Getting Started
   1.1. Installation & Setup
   1.2. Your First NanoLang Program
   1.3. Hello World Walkthrough
   1.4. Compilation & Execution

2. Basic Syntax & Types
   2.1. Prefix Notation (The One True Way™)
   2.2. Numbers & Arithmetic
   2.3. Strings & Characters
   2.4. Booleans & Comparisons
   2.5. Type Annotations

3. Variables & Bindings
   3.1. let Bindings (Immutable by Default)
   3.2. mut Variables (When You Need Mutation)
   3.3. set Statements
   3.4. Scope & Shadowing

4. Functions
   4.1. Function Definitions
   4.2. Parameters & Return Types
   4.3. Shadow Tests (Built-in Testing)
   4.4. Recursion by Example

5. Control Flow
   5.1. if/else Expressions
   5.2. cond Multi-way Branches
   5.3. while Loops
   5.4. for-in-range Loops
   5.5. Loop Patterns & Idioms

6. Collections
   6.1. Arrays (Immutable Collections)
   6.2. Array Operations
   6.3. List<T> (Dynamic Collections)
   6.4. Working with Collections

7. Data Structures
   7.1. Structs (Product Types)
   7.2. Enums (Sum Types)
   7.3. Unions (Tagged Unions)
   7.4. Opaque Types
   7.5. Pattern Matching

8. Modules & Imports
   8.1. Importing Modules
   8.2. Selective Imports (from...import)
   8.3. Creating Your Own Modules
   8.4. Module Structure & Best Practices
```

### Part II: Standard Library (Chapters 9-12)
**Theme**: The batteries included with NanoLang

```
9. Core Utilities
   9.1. String Manipulation
   9.2. Character Classification
   9.3. Math Functions
   9.4. Type Conversions

10. Collections Library
    10.1. StringBuilder (Efficient String Building)
    10.2. HashMap<K,V> (Key-Value Storage)
    10.3. HashSet<T> (Unique Values)
    10.4. Array Utilities
    10.5. Sorting & Searching

11. I/O & Filesystem
    11.1. Reading & Writing Files
    11.2. Directory Operations
    11.3. Path Manipulation
    11.4. File Metadata

12. System & Runtime
    12.1. Command Line Arguments
    12.2. Environment Variables
    12.3. Process Execution
    12.4. Error Handling Patterns
```

### Part III: External Modules (Chapters 13-25)
**Theme**: Batteries NOT included - powerful extensions

```
13. Text Processing
    13.1. regex - Regular Expressions by Example
    13.2. log - Structured Logging
    13.3. StringBuilder - Advanced Techniques

14. Data Formats
    14.1. JSON - Parsing & Generation
    14.2. SQLite - Embedded Database

15. Web & Networking
    15.1. curl - HTTP Client
    15.2. http_server - Building Web Services
    15.3. uv - Async I/O (libuv bindings)

16. Graphics Fundamentals
    16.1. SDL - Simple DirectMedia Layer
    16.2. SDL_image - Image Loading
    16.3. SDL_mixer - Audio Playback
    16.4. SDL_ttf - TrueType Fonts

17. OpenGL Graphics
    17.1. OpenGL - Modern Graphics Pipeline
    17.2. GLFW - Window & Input Management
    17.3. GLEW - Extension Wrangler
    17.4. GLUT - Utility Toolkit

18. Game Development
    18.1. event - Event System
    18.2. vector2d - 2D Math
    18.3. bullet - Physics Engine
    18.4. Example: Building a Simple Game

19. Terminal UI
    19.1. ncurses - Terminal Applications

20. Testing & Quality
    20.1. proptest - Property-Based Testing
    20.2. coverage - Code Coverage
    20.3. Testing Best Practices

21. Configuration
    21.1. preferences - User Preferences Management
```

### Part IV: Advanced Topics (Chapters 22-25)
**Theme**: Deep dives and best practices

```
22. Canonical Style Guide
    22.1. The LLM-First Design Philosophy
    22.2. Canonical Forms (Why One Way Matters)
    22.3. The Core 50-Primitive Subset
    22.4. Style Checklist

23. Higher-Level Patterns
    23.1. Builder Patterns
    23.2. Iterator Patterns
    23.3. Resource Management (RAII-style)
    23.4. Error Handling Idioms

24. Performance & Optimization
    24.1. Understanding Compilation
    24.2. Memory Management
    24.3. Profiling Techniques
    24.4. Common Pitfalls

25. Contributing & Extending
    25.1. Creating FFI Modules
    25.2. C Interop Patterns
    25.3. Module Packaging
    25.4. Contributing to NanoLang
```

### Appendices
```
A. Complete Examples Gallery
   (All examples from the examples/ directory, organized by topic)

B. Quick Reference
   B.1. Operator Precedence
   B.2. Built-in Functions
   B.3. Standard Library Overview
   B.4. Module Index

C. Troubleshooting Guide
   C.1. Common Compile Errors
   C.2. Runtime Issues
   C.3. Module Installation
   C.4. Platform-Specific Issues

D. Glossary
   (Technical terms and NanoLang-specific concepts)
```

---

## API Reference Format: "By Example"

### Current Format (❌ BAD)
```markdown
#### `extern fn nl_regex_match(regex: Regex, text: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `regex` | `Regex` |
| `text` | `string` |

**Returns:** `int`
```

### New Format (✅ GOOD)
```markdown
### `matches()` - Test if Pattern Matches

Check if a regular expression matches a string.

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
from "modules/std/regex.nano" import compile, matches, free

fn is_email(s: string) -> bool {
    let pattern: Regex = (compile "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
    let result: bool = (matches pattern s)
    (free pattern)
    return result
}

shadow is_email {
    assert (is_email "user@example.com")
    assert (not (is_email "not-an-email"))
}
```

**Example: Quick One-Off Match**
```nano
from "modules/std/regex.nano" import quick_match

fn has_digits(s: string) -> bool {
    return (quick_match "[0-9]+" s)
}

shadow has_digits {
    assert (has_digits "abc123")
    assert (not (has_digits "abcdef"))
}
```

**See Also:**
- `compile()` - Compile a regex pattern
- `quick_match()` - One-shot matching without explicit compilation
- `find()` - Find position of first match
- `find_all()` - Find all match positions

**Common Pitfalls:**
- ⚠️ Don't forget to call `free()` on compiled regexes to avoid memory leaks
- ⚠️ Backslashes must be escaped: use `"\\d"` not `"\d"`
- 💡 Tip: Use `quick_match()` for one-time patterns; use `compile()` when matching repeatedly
```

---

## Implementation Strategy

### Phase 1: Structure & Templates (Week 1)
1. ✅ Create new markdown structure for all chapters
2. ✅ Design API reference templates with example sections
3. ✅ Update sidebar generation in `userguide_build_html.nano`
4. ✅ Create style guide for technical writers

### Phase 2: Core Content (Weeks 2-3)
1. ✅ Rewrite Part I (Language Fundamentals) with examples
2. ✅ Add code snippets for every concept
3. ✅ Ensure all examples are tested (shadow tests)
4. ✅ Add "Try It Yourself" sections

### Phase 3: Standard Library (Week 4)
1. ✅ Document Part II with comprehensive examples
2. ✅ Create "cookbook" sections for common tasks
3. ✅ Add comparison tables (e.g., "Which collection type?")

### Phase 4: External Modules (Weeks 5-7)
1. ✅ Rewrite all 20+ API references with examples
2. ✅ Create tutorial-style introductions for each module
3. ✅ Add cross-references between related modules
4. ✅ Include "Getting Started" quick-start for each

### Phase 5: Advanced & Polish (Week 8)
1. ✅ Write advanced topics chapters
2. ✅ Create comprehensive examples gallery
3. ✅ Add troubleshooting guide
4. ✅ Final review and polish

### Phase 6: Generation & Deploy (Week 8)
1. ✅ Update HTML generation for new structure
2. ✅ Ensure all code is syntax-highlighted
3. ✅ Test navigation and cross-links
4. ✅ Deploy to GitHub Pages

---

## Code Example Standards

### Every Example Must Include:
1. ✅ **Syntax highlighting** (via `pretty_print_html()`)
2. ✅ **Shadow test** (proves it works)
3. ✅ **Comments** explaining key points
4. ✅ **Complete, runnable code** (no pseudocode)
5. ✅ **Imports** listed explicitly

### Example Categories:
- **Minimal**: Smallest possible demonstration
- **Practical**: Real-world use case
- **Complete**: Full program with error handling
- **Advanced**: Complex pattern or technique

---

## Sidebar Design

### Current (❌ Inconsistent)
```
01_getting_started
02_control_flow
...
api_reference/
  bullet
  coverage
  curl
  ...
```

### New (✅ Professional Book Structure)
```
Part I: Language Fundamentals
  ├─ 1. Getting Started
  │   ├─ 1.1. Installation & Setup
  │   ├─ 1.2. Your First Program
  │   └─ 1.3. Hello World
  ├─ 2. Basic Syntax & Types
  │   ├─ 2.1. Prefix Notation
  │   ├─ 2.2. Numbers
  │   └─ ...

Part II: Standard Library
  ├─ 9. Core Utilities
  ├─ 10. Collections Library
  └─ ...

Part III: External Modules
  ├─ 13. Text Processing
  │   ├─ 13.1. regex Module
  │   │    ├─ Quick Start
  │   │    ├─ compile()
  │   │    ├─ matches()
  │   │    └─ ...
  └─ ...

Part IV: Advanced Topics
  └─ ...

Appendices
  ├─ A. Examples Gallery
  ├─ B. Quick Reference
  └─ ...
```

---

## Visual Design Principles

### Typography
- Headers: Clear hierarchy (h1 → h2 → h3)
- Code: Monospace, syntax-highlighted
- Body: Readable serif or sans-serif

### Color Coding
- **Examples**: Green accent ("✅ Try This")
- **Warnings**: Yellow/Orange ("⚠️ Watch Out")
- **Tips**: Blue ("💡 Pro Tip")
- **Errors**: Red ("❌ Common Mistake")

### Code Blocks
- Line numbers optional (only for long examples)
- Syntax highlighting mandatory
- Copy button for all code blocks
- Output shown separately (when relevant)

---

## Success Metrics

1. **Completeness**: Every public API has ≥2 working examples
2. **Testability**: 100% of examples have shadow tests
3. **Navigation**: <3 clicks to any API function
4. **Searchability**: Full-text search works
5. **Accessibility**: WCAG 2.1 AA compliance

---

## Migration Plan

### Existing Content Preservation
- Current chapters (01-07) will be refactored, not rewritten
- Existing examples stay (new examples added)
- URLs preserved where possible (redirects for moved content)

### Backward Compatibility
- Old API reference URLs → redirect to new structure
- Keep auto-generation as fallback for undocumented modules
- Version selector (future: v0.5, v0.6, etc.)

---

## Technical Writer's Style Guide

### Voice & Tone
- **Active voice**: "You compile the program" not "The program is compiled"
- **Second person**: "You" not "we" or "one"
- **Present tense**: "NanoLang uses prefix notation"
- **Encouraging**: Focus on what you CAN do

### Example Structure
```
[Brief description: one sentence what it does]

[Signature block]

[Parameters: bulleted, with context]

[Returns: what you get back]

[Example 1: MINIMAL - simplest use case]
  - Code block
  - Shadow test
  - Brief explanation

[Example 2: PRACTICAL - real-world scenario]
  - Code block
  - Shadow test
  - Explanation of why/when

[See Also: related functions]

[Common Pitfalls: warnings & tips]
```

### Terminology
- "prefix notation" (not "S-expressions" or "Lisp-like")
- "function" (not "procedure" or "method")
- "module" (not "library" or "package")
- "shadow test" (not "unit test" or "spec")

---

## Review & Approval Process

### Stakeholders
- [ ] @jordanhubbard (Project Lead) - Overall direction
- [ ] Technical Writers - Content & examples
- [ ] Community - Feedback on draft chapters

### Review Checklist
- [ ] All examples compile and pass shadow tests
- [ ] Consistent terminology throughout
- [ ] Clear learning progression (novice → expert)
- [ ] Cross-references verified
- [ ] Accessibility tested
- [ ] Mobile-friendly layout

---

## Next Steps

1. **Get approval** on this plan
2. **Create detailed outlines** for each chapter
3. **Set up tracking** (beads for each section)
4. **Start Phase 1** (structure & templates)

---

## Appendix: File Structure

```
userguide/
├── index.md (new: landing page)
├── part1_fundamentals/
│   ├── 01_getting_started.md
│   ├── 02_syntax_types.md
│   ├── 03_variables.md
│   ├── 04_functions.md
│   ├── 05_control_flow.md
│   ├── 06_collections.md
│   ├── 07_data_structures.md
│   └── 08_modules.md
├── part2_stdlib/
│   ├── 09_core_utilities.md
│   ├── 10_collections_lib.md
│   ├── 11_io_filesystem.md
│   └── 12_system_runtime.md
├── part3_modules/
│   ├── 13_text_processing/
│   │   ├── index.md
│   │   ├── regex.md (rewritten with examples)
│   │   ├── log.md
│   │   └── stringbuilder.md
│   ├── 14_data_formats/
│   │   ├── index.md
│   │   ├── json.md
│   │   └── sqlite.md
│   ├── 15_web_networking/
│   ├── 16_graphics_fundamentals/
│   ├── 17_opengl/
│   ├── 18_game_dev/
│   ├── 19_terminal_ui/
│   ├── 20_testing/
│   └── 21_configuration/
├── part4_advanced/
│   ├── 22_canonical_style.md
│   ├── 23_patterns.md
│   ├── 24_performance.md
│   └── 25_contributing.md
├── appendices/
│   ├── a_examples_gallery.md
│   ├── b_quick_reference.md
│   ├── c_troubleshooting.md
│   └── d_glossary.md
└── assets/
    ├── style.css (enhanced)
    ├── syntax-highlight.css (new)
    └── examples/ (extracted code)
```

---

**END OF PLAN**
