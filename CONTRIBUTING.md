# How You Can Help Me

I am NanoLang. I have clear standards for how I am built, how I test myself, and how I evolve. If you want to contribute to my development, you must follow my rules. I do not compromise on these.

## My Ground Rules

These rules ensure I remain maintainable and capable of compiling myself.

### 1. One Implementation

I am a compiled language. Every feature you add must live in my compiler. I do not have an interpreter because maintaining two implementations is a waste of time. I transpile to C for performance.

If you add a feature, you must implement it in:
- The compiler (`src/transpiler.c`, `src/parser.c`, `src/typechecker.c`)

I expect shadow tests to compile into the final binary and run at execution time. Before you claim a feature is finished, verify those tests pass.

**Example**: Generic `List<T>` support
- Compiler: Generates C code with `list_TypeName_*` functions
- Shadow tests: Compile into binary, validate at runtime
- Result: One clean implementation

### 2. No Warnings

I must compile without warnings. A warning is just a bug I haven't caught yet.

- C code: Must pass `-Wall -Wextra -Werror`. No warnings means no errors.
- My own code: I expect zero type warnings and zero unused variable warnings.
- The build will fail if you introduce a warning.
- If you find a warning that truly cannot be removed, you must document it.

How to check:
```bash
make clean && make
./bin/nanoc file.nano
```

### 3. Bootstrap and Self-Hosting

I have two versions of myself. You must update both.
1. **C Reference Implementation** (`src/`) - This is how I bootstrap.
2. **Self-Hosted Implementation** (`src_nano/`) - This is how I prove I am complete.

I use the C version to build my first stage. Then I use that stage to build myself again. If I cannot express a feature in my own syntax, then I am not finished.

Your workflow:
1. Implement the feature in my C reference code.
2. Verify it passes my tests.
3. Implement the same feature using my own syntax in `src_nano/`.
4. Verify my self-hosted components can use it.
5. Use the feature in my codebase.

My structure:
```
src/          - C reference implementation (bootstrap compiler)
  parser.c    - C parser
  typechecker.c - C typechecker
  transpiler.c  - C transpiler

src_nano/     - NanoLang self-hosted implementation
  parser_mvp.nano - Parser in NanoLang
  typechecker_minimal.nano - Typechecker in NanoLang
  transpiler_minimal.nano - Transpiler in NanoLang
```

### 4. Shadow Tests are Mandatory

I do not trust code that has not been asked to prove itself. Every function you write must have a shadow test. This is a core design principle, not a suggestion.

I require shadow tests for:
- All functions in my libraries
- All functions in your applications
- All functions in my examples
- All helper functions
- All demonstration programs

The only exception is `extern` functions. I cannot test what happens in the C world.

Why I require this:
- Shadow tests prove correctness during compilation.
- They are executable specifications.
- They allow me to validate myself during the bootstrap process.
- They prevent bad design. If a function is hard to test, it is poorly written.

If my compiler warns you about a missing shadow test, it is not a mistake. Add the test. If you are an LLM generating code for me, you will include shadow tests for every function.

My test hierarchy:
1. **Shadow Tests** (Inlined in functions) - Mandatory. These run during compilation.
2. **Integration Tests** (`tests/*.nano`) - These test how features work together.
3. **Self-Hosting Tests** - These verify the bootstrap and fixed-point identity.

**Example**:
```nano
fn list_length(list: List<int>) -> int {
    return (list_int_length list)
}

shadow list_length {
    let xs: List<int> = (list_int_new)
    (list_int_push xs 5)
    (list_int_push xs 10)
    assert (== (list_length xs) 2)
}
```

### 5. Documentation

I prefer code that explains itself. Use comments only when necessary.

What to document:
- Why you chose a complex algorithm.
- Design decisions that are not obvious.
- Workarounds for my current limitations.
- Brief descriptions of my public API.

What not to document:
- Obvious code. I can read `i++` without help.
- Redundant files that repeat what the code already says.

My README policy:
- Update it only when I ask you to.
- Keep it focused on what I am and how to run me.
- Do not document every function. Use my `--help` flags.

### 6. Precise Error Messages

My error messages must tell the user what happened and how to fix it.

Format:
```
Error at line X, column Y: <What went wrong>
  Note: <Additional context>
  Hint: <How to fix it>
```

**Example**:
```
Error at line 42, column 18: Undefined function 'list_Point_new'
  Note: struct or enum 'Point' is not defined
  Hint: Define 'struct Point { ... }' before using List<Point>
```

I require line and column numbers for every error during parsing, type-checking, or transpiling.

### 7. Backward Compatibility

I expect features that I have already released to keep working.

If you want to make a breaking change:
- I will need a major version bump.
- You must provide a migration guide.
- I will issue deprecation warnings first.
- You must discuss it with the community.

Safe changes include fixing bugs, improving my error messages, or optimizing my performance without changing my behavior.

### 8. Efficiency

I value correctness over performance, but I do not tolerate waste.

- Use the right data structures.
- Avoid slow algorithms where input size might grow.
- Profile your changes before you optimize them.
- Memory leaks are bugs. Always free what you allocate.

My compilation should take less than 5 seconds for small files. My shadow tests should take less than 1 second. If you write more than 2000 lines in one file, you may need to split it.

## How You Should Work

### 1. Preparation
- Read these rules.
- Check my existing issues to see if someone is already doing the work.
- Discuss large changes with me in an issue.

### 2. Development
- Follow my ground rules.
- Write your tests while you code, not after.
- Run `make test` often.
- Keep your commits small and focused.

### 3. Submission
Before you submit your work, verify:
```bash
# Clean build
make clean && make

# All tests pass
make test

# No warnings
./bin/nanoc your_file.nano

# Self-hosted components build
make
```

### 4. PR Description
I expect your description to follow this format:
```markdown
## What Changed
Brief description

## Implementation
- C Reference: <files>
- NanoLang Self-Hosted: <files>

## Testing
- Shadow tests: Done / Updated / None
- Integration tests: Done / Updated / None
- All tests passing: Yes / No

## Breaking Changes
Yes / No

## Checklist
- Builds warning-free
- All tests pass
- Shadow tests pass
- Documentation updated
- Self-hosted components updated
```

## How I Evolve

I use a Request for Comments (RFC) process for major changes.

### When an RFC is Needed
- You want to add new syntax, operators, or types.
- You want to change how an existing feature works.
- You want to add large sections to my standard library.
- You want to change my compiler's architecture.

### When an RFC is Not Needed
- You are fixing a bug.
- You are improving my documentation.
- You are optimizing my performance.
- You are adding examples or tests.

### The RFC Process
1. **Draft**: Use my template in `docs/rfcs/0000-template.md`.
2. **Proposal**: Open a Pull Request with your RFC.
3. **Discussion**: The community will provide feedback for 1 to 2 weeks.
4. **Final Comment Period**: A one-week period for final concerns once the design is stable.
5. **Decision**: I will either accept, reject, or postpone the proposal.

Your RFC must explain your motivation, the detailed design, potential drawbacks, and any alternatives you considered.

See `docs/RFC_PROCESS.md` for more details.

## Questions

If you have questions, open an issue or start a discussion.

## Code of Conduct

- Be respectful.
- Focus on my code.
- Assume everyone is trying to help.
- Help others learn how I work.

