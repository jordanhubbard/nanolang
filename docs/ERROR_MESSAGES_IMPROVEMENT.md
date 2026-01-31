# Elm-Style Error Message Improvements for NanoLang

**Status:** Design & Implementation Phase  
**Priority:** P1  
**Bead:** nanolang-v7tu  
**Created:** 2025-12-29

## Goal

Transform NanoLang error messages from cryptic compiler output to **helpful, actionable guidance** that helps users fix problems quickly.

**Inspiration:** Elm compiler (gold standard), Rust compiler (excellent), Swift compiler (good)

## Problem: Current Error Messages

### Example 1: Type Mismatch
```
Error at line 42, column 15: Type mismatch
Expected: int
Got: string
```

**Problems:**
- No context (what code triggered this?)
- No explanation (why is this wrong?)
- No hint (how do I fix it?)

### Example 2: Undefined Variable
```
Error at line 10, column 5: Undefined variable 'countr'
```

**Problems:**
- No suggestion (did you mean 'counter'?)
- No context showing where 'counter' is defined

## Solution: Rich, Contextual Error Messages

### Example 1 Improved: Type Mismatch

```
-- TYPE MISMATCH ----------------------------------------- calculate.nano

The function `calculate` expects an integer, but you gave it a string:

42│     let result: int = (calculate "hello")
                                      ^^^^^^^
This `"hello"` value is a:

    string

But `calculate` needs the argument to be:

    int

Hint: Did you mean to parse the string first?

    Try: (string_to_int "hello")
    Or:  (parse_int "hello" 10)
```

### Example 2 Improved: Undefined Variable with Suggestion

```
-- UNDEFINED VARIABLE ------------------------------------ counter.nano

I cannot find a variable named `countr`:

10│     set countr (+ countr 1)
             ^^^^^^
Did you mean one of these?

    counter  (defined on line 5)
    count    (defined on line 3)
    
Hint: Variable names are case-sensitive in NanoLang.
```

### Example 3: Missing Shadow Test

```
-- MISSING SHADOW TEST ---------------------------------- calculate.nano

The function `calculate` does not have a shadow test:

15│ fn calculate(x: int, y: int) -> int {
16│     return (+ (* x 2) y)
17│ }

Shadow tests verify your function works correctly. Add one like this:

    shadow calculate {
        assert (== (calculate 2 3) 7)
        assert (== (calculate 0 5) 5)
    }

Why? Shadow tests catch bugs early and serve as documentation.

You can disable this warning with: --no-shadow-warnings
```

### Example 4: Common Mistake - Wrong Operator

```
-- OPERATOR ERROR --------------------------------------- strings.nano

You're trying to add strings with `+`, but that's for numbers:

23│     let greeting: string = (+ "Hello, " name)
                                 ^
The `+` operator works on:

    int + int   -> int
    float + float -> float

For strings, use `++` (concatenation):

    (++ "Hello, " name)

Related:
    ++   - concatenate strings
    +    - add numbers
    and  - logical AND
    or   - logical OR
```

## Implementation Architecture

### Phase 1: Error Context System (Week 1)

**1.1 Add Source Context to Errors**

```c
/* src/nanolang.h */
typedef struct ErrorContext {
    char *filename;
    int line;
    int column;
    int span_length;
    char *source_line;         /* The actual source code line */
    char *previous_line;       /* Line before (for context) */
    char *next_line;           /* Line after (for context) */
} ErrorContext;

typedef struct CompilerError {
    char *title;               /* "TYPE MISMATCH" */
    char *message;             /* Main explanation */
    ErrorContext context;      /* Where it happened */
    char **hints;              /* Array of hints */
    int hint_count;
    char **related;            /* Related definitions/uses */
    int related_count;
} CompilerError;
```

**1.2 Update Error Emission Functions**

```c
/* src/typechecker.c */
void emit_type_mismatch_error(
    const char *filename,
    Token *tok,
    Type expected,
    Type got,
    const char *expr_source  /* The actual code that failed */
) {
    CompilerError err = {0};
    err.title = "TYPE MISMATCH";
    
    /* Build context */
    err.context = build_error_context(filename, tok);
    
    /* Build message */
    char message[1024];
    snprintf(message, sizeof(message),
        "This expression has type `%s`, but I expected `%s`:\n\n"
        "%s\n",
        type_to_string(got),
        type_to_string(expected),
        err.context.source_line
    );
    err.message = strdup(message);
    
    /* Add hints */
    err.hints = generate_hints_for_type_mismatch(expected, got);
    err.hint_count = count_hints(err.hints);
    
    /* Print formatted error */
    print_compiler_error(&err);
    
    /* Cleanup */
    free_compiler_error(&err);
}
```

**1.3 Source Line Extraction**

```c
/* src/error_context.c */
ErrorContext build_error_context(const char *filename, Token *tok) {
    ErrorContext ctx = {0};
    ctx.filename = strdup(filename);
    ctx.line = tok->line;
    ctx.column = tok->column;
    ctx.span_length = strlen(tok->value);
    
    /* Read source file and extract lines */
    FILE *f = fopen(filename, "r");
    if (!f) {
        ctx.source_line = strdup("(source not available)");
        return ctx;
    }
    
    char *lines[tok->line + 2];
    int line_num = 0;
    char buffer[4096];
    
    while (fgets(buffer, sizeof(buffer), f) && line_num < tok->line + 2) {
        lines[line_num++] = strdup(buffer);
    }
    fclose(f);
    
    if (tok->line > 0) ctx.previous_line = lines[tok->line - 2];
    ctx.source_line = lines[tok->line - 1];
    if (line_num > tok->line) ctx.next_line = lines[tok->line];
    
    return ctx;
}
```

### Phase 2: Hint Generation (Week 2)

**2.1 Type Mismatch Hints**

```c
/* src/error_hints.c */
char **generate_hints_for_type_mismatch(Type expected, Type got) {
    char **hints = malloc(sizeof(char*) * 10);
    int hint_count = 0;
    
    /* string vs int */
    if (expected == TYPE_INT && got == TYPE_STRING) {
        hints[hint_count++] = strdup(
            "Hint: To convert a string to an integer, use:\n"
            "    (string_to_int \"42\")\n"
            "    (parse_int \"42\" 10)"
        );
    }
    
    /* int vs float */
    if (expected == TYPE_FLOAT && got == TYPE_INT) {
        hints[hint_count++] = strdup(
            "Hint: To convert an integer to a float, use:\n"
            "    (int_to_float 42)"
        );
    }
    
    /* Missing parentheses */
    if (expected == TYPE_INT && got == TYPE_FUNCTION) {
        hints[hint_count++] = strdup(
            "Hint: Did you forget to call the function?\n"
            "    Instead of: my_function\n"
            "    Try:        (my_function)"
        );
    }
    
    hints[hint_count] = NULL;
    return hints;
}
```

**2.2 Typo Detection (Levenshtein Distance)**

```c
/* src/error_hints.c */
char **suggest_similar_names(const char *name, Environment *env) {
    char **suggestions = malloc(sizeof(char*) * 10);
    int suggestion_count = 0;
    
    /* Check all variables in scope */
    for (int i = 0; i < env->var_count; i++) {
        int distance = levenshtein_distance(name, env->vars[i].name);
        if (distance <= 2) {  /* Max 2 character difference */
            char buf[512];
            snprintf(buf, sizeof(buf), "%s (defined on line %d)",
                    env->vars[i].name, env->vars[i].line);
            suggestions[suggestion_count++] = strdup(buf);
        }
    }
    
    /* Check all functions */
    for (int i = 0; i < env->fn_count; i++) {
        int distance = levenshtein_distance(name, env->functions[i].name);
        if (distance <= 2) {
            char buf[512];
            snprintf(buf, sizeof(buf), "%s (defined on line %d)",
                    env->functions[i].name, env->functions[i].line);
            suggestions[suggestion_count++] = strdup(buf);
        }
    }
    
    suggestions[suggestion_count] = NULL;
    return suggestions;
}

int levenshtein_distance(const char *s1, const char *s2) {
    int len1 = strlen(s1), len2 = strlen(s2);
    int matrix[len1 + 1][len2 + 1];
    
    for (int i = 0; i <= len1; i++) matrix[i][0] = i;
    for (int j = 0; j <= len2; j++) matrix[0][j] = j;
    
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            matrix[i][j] = min3(
                matrix[i-1][j] + 1,      /* deletion */
                matrix[i][j-1] + 1,      /* insertion */
                matrix[i-1][j-1] + cost  /* substitution */
            );
        }
    }
    
    return matrix[len1][len2];
}
```

### Phase 3: Formatted Output (Week 3)

**3.1 ANSI Color Support**

```c
/* src/error_printer.c */
#define ANSI_RED     "\x1b[31m"
#define ANSI_GREEN   "\x1b[32m"
#define ANSI_YELLOW  "\x1b[33m"
#define ANSI_BLUE    "\x1b[34m"
#define ANSI_CYAN    "\x1b[36m"
#define ANSI_BOLD    "\x1b[1m"
#define ANSI_RESET   "\x1b[0m"

bool supports_color(void) {
    /* Check if output is a TTY and TERM is set */
    if (!isatty(STDERR_FILENO)) return false;
    const char *term = getenv("TERM");
    if (!term || strcmp(term, "dumb") == 0) return false;
    return true;
}

void print_error_header(const char *title, const char *filename) {
    if (supports_color()) {
        fprintf(stderr, "%s-- %s %s-- %s\n",
                ANSI_CYAN, title, ANSI_RESET, filename);
    } else {
        fprintf(stderr, "-- %s -- %s\n", title, filename);
    }
}
```

**3.2 Formatted Error Printer**

```c
void print_compiler_error(CompilerError *err) {
    /* Header */
    print_error_header(err->title, err->context.filename);
    fprintf(stderr, "\n");
    
    /* Main message */
    fprintf(stderr, "%s\n", err->message);
    
    /* Source context with highlighting */
    print_source_context(&err->context);
    
    /* Hints */
    if (err->hint_count > 0) {
        fprintf(stderr, "\n");
        for (int i = 0; i < err->hint_count; i++) {
            if (supports_color()) {
                fprintf(stderr, "%s%s%s\n", ANSI_GREEN, err->hints[i], ANSI_RESET);
            } else {
                fprintf(stderr, "%s\n", err->hints[i]);
            }
        }
    }
    
    /* Related definitions */
    if (err->related_count > 0) {
        fprintf(stderr, "\nRelated:\n");
        for (int i = 0; i < err->related_count; i++) {
            fprintf(stderr, "    %s\n", err->related[i]);
        }
    }
    
    fprintf(stderr, "\n");
}

void print_source_context(ErrorContext *ctx) {
    if (supports_color()) {
        /* Line number + source */
        fprintf(stderr, "%s%3d│%s %s\n", 
                ANSI_CYAN, ctx->line, ANSI_RESET, ctx->source_line);
        
        /* Underline the error span */
        fprintf(stderr, "   │ %s", ANSI_RED);
        for (int i = 0; i < ctx->column - 1; i++) {
            fprintf(stderr, " ");
        }
        for (int i = 0; i < ctx->span_length; i++) {
            fprintf(stderr, "^");
        }
        fprintf(stderr, "%s\n", ANSI_RESET);
    } else {
        /* Plain text version */
        fprintf(stderr, "%3d│ %s\n", ctx->line, ctx->source_line);
        fprintf(stderr, "   │ ");
        for (int i = 0; i < ctx->column - 1; i++) {
            fprintf(stderr, " ");
        }
        for (int i = 0; i < ctx->span_length; i++) {
            fprintf(stderr, "^");
        }
        fprintf(stderr, "\n");
    }
}
```

### Phase 4: Error Catalog (Week 4)

**4.1 Error Code System**

```c
/* src/error_codes.h */
typedef enum {
    ERR_TYPE_MISMATCH = 1001,
    ERR_UNDEFINED_VAR = 1002,
    ERR_UNDEFINED_FN = 1003,
    ERR_WRONG_ARG_COUNT = 1004,
    ERR_MISSING_SHADOW = 1005,
    ERR_IMMUTABLE_ASSIGN = 1006,
    ERR_SYNTAX_ERROR = 2001,
    ERR_PARSE_ERROR = 2002,
    /* ... */
} ErrorCode;

const char *get_error_explanation(ErrorCode code);
const char *get_error_url(ErrorCode code);
```

**4.2 --explain Flag**

```bash
$ nanoc file.nano
Error E1001: Type mismatch at line 42

$ nanoc --explain E1001
E1001: TYPE MISMATCH

This error occurs when an expression has a different type than expected.

Common causes:
1. Passing wrong type to function
2. Assigning wrong type to variable
3. Returning wrong type from function

Examples:
...

Learn more: https://nanolang.org/errors/E1001
```

### Phase 5: Self-Hosted Compiler (Week 5)

**5.1 Port Error System to NanoLang**

```nano
/* src_nano/errors.nano */

struct ErrorContext {
    filename: string,
    line: int,
    column: int,
    span_length: int,
    source_line: string,
    previous_line: string,
    next_line: string
}

struct CompilerError {
    title: string,
    message: string,
    context: ErrorContext,
    hints: List<string>,
    related: List<string>
}

fn print_error(err: CompilerError) -> void {
    (print_error_header err.title err.context.filename)
    (println "")
    (println err.message)
    (print_source_context err.context)
    
    /* Print hints */
    let mut i: int = 0
    while (< i (list_length err.hints)) {
        let hint: string = (list_get err.hints i)
        (println (++ "Hint: " hint))
        set i (+ i 1)
    }
}
```

### Phase 6: Testing (Week 6)

**6.1 Error Message Tests**

```nano
/* tests/test_error_messages.nano */

/* TEST: Verify type mismatch error format */
fn test_type_mismatch_error() -> void {
    /* This test checks the error message, not compilation success */
    let output: string = (compile_and_capture_errors "
        fn test() -> int {
            return \"hello\"
        }
    ")
    
    assert (string_contains output "TYPE MISMATCH")
    assert (string_contains output "return \"hello\"")
    assert (string_contains output "Hint:")
}

/* TEST: Verify typo suggestion */
fn test_typo_suggestion() -> void {
    let output: string = (compile_and_capture_errors "
        fn test() -> void {
            let counter: int = 0
            set countr 1  // typo: countr vs counter
        }
    ")
    
    assert (string_contains output "Did you mean")
    assert (string_contains output "counter")
}
```

## Error Message Checklist

For every error, ensure:

- [ ] **Context**: Show the actual code that failed
- [ ] **Explanation**: Explain what's wrong and why
- [ ] **Hint**: Suggest how to fix it
- [ ] **Related**: Show relevant definitions/uses
- [ ] **Tone**: Friendly and encouraging, not condescending
- [ ] **Color**: Use colors when available (with fallback)
- [ ] **Formatting**: Clear, scannable layout

## Examples of All Error Types

### Type Errors
- ✅ Type mismatch (shown above)
- ✅ Wrong argument count
- ✅ Undefined variable (with suggestions)
- ✅ Undefined function (with suggestions)
- ✅ Immutable assignment

### Syntax Errors
- ✅ Missing parenthesis
- ✅ Unexpected token
- ✅ Invalid syntax

### Semantic Errors
- ✅ Missing shadow test
- ✅ Unused variable
- ✅ Unreachable code

### Module Errors
- ✅ Module not found
- ✅ Cyclic dependency
- ✅ Symbol not exported

## Command Line Flags

```bash
--no-color              Disable ANSI colors
--explain E1001         Show detailed explanation for error code
--verbose               Show full error context (3 lines before/after)
--json                  Output errors in JSON format (for IDE integration)
--no-shadow-warnings    Disable "missing shadow test" warnings
```

## Timeline

- **Week 1:** Error context system
- **Week 2:** Hint generation + typo detection
- **Week 3:** Formatted output with colors
- **Week 4:** Error catalog + --explain
- **Week 5:** Port to self-hosted compiler
- **Week 6:** Testing + documentation

**Total:** 6 weeks

## Success Criteria

1. ✅ All type errors show source context + hints
2. ✅ Typo detection works for variables and functions
3. ✅ Colors work on terminals (with fallback)
4. ✅ --explain flag provides detailed help
5. ✅ Self-hosted compiler has parity with C version
6. ✅ Test suite validates error message quality
7. ✅ User testing shows improved error fix time

## References

- **Elm Compiler**: https://elm-lang.org/news/compiler-errors-for-humans
- **Rust Error Codes**: https://doc.rust-lang.org/error-index.html
- **Swift Diagnostics**: https://github.com/apple/swift/tree/main/lib/AST
- **GHC Error Messages**: https://downloads.haskell.org/ghc/latest/docs/users_guide/

---

**Next Steps:**
1. Review this design
2. Start Phase 1 implementation  
3. Create test file for each error type
4. User test with beginner programmers

