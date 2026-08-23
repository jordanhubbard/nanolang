# Multi-Line Input REPL Design

## Goal

Support multi-line input in the REPL for complex expressions and function definitions that span multiple lines.

## Use Cases

### 1. Multi-line Function Definitions
```nano
nano> fn factorial(n: int) -> int {
....>     if (<= n 1) {
....>         return 1
....>     } else {
....>         return (* n (factorial (- n 1)))
....>     }
....> }
Defined: factorial(int) -> int
```

### 2. Multi-line Struct Definitions
```nano
nano> struct Point {
....>     x: int,
....>     y: int
....> }
Defined: Point
```

### 3. Multi-line Expressions
```nano
nano> (+
....>     (* 2 3)
....>     (* 4 5))
=> 26
```

## Implementation Strategy

### Detection of Incomplete Input

Track the depth of various brackets/braces:
- `{` / `}` - brace depth
- `(` / `)` - paren depth
- `[` / `]` - bracket depth

**Input is complete when:**
- All depths are zero
- Not ending with an operator or comma
- Not inside a string literal

**Input is incomplete when:**
- Any depth > 0
- Ending with operator: `+`, `-`, `*`, `/`, etc.
- Ending with comma
- Inside unclosed string

### Continuation Prompt

Show different prompt when continuing multi-line input:
- Primary prompt: `nano> `
- Continuation prompt: `....> ` (4 dots + space)

### Buffering

Use StringBuilder to accumulate lines:
1. Read first line with primary prompt
2. Check if complete
3. If incomplete, show continuation prompt and read next line
4. Append to buffer with newline
5. Repeat until complete
6. Evaluate complete input

## Implementation

### Function: `is_input_complete`

```nano
fn is_input_complete(input: string) -> bool {
    let mut brace_depth: int = 0
    let mut paren_depth: int = 0
    let mut bracket_depth: int = 0
    let mut in_string: bool = false

    let len: int = (str_length input)
    let mut i: int = 0

    while (< i len) {
        let ch: int = (char_at input i)

        # Handle string literals
        if (== ch 34) {  # double quote
            set in_string (not in_string)
        } else {
            if (not in_string) {
                if (== ch 123) {  # {
                    set brace_depth (+ brace_depth 1)
                } else {
                    if (== ch 125) {  # }
                        set brace_depth (- brace_depth 1)
                    } else {
                        if (== ch 40) {  # (
                            set paren_depth (+ paren_depth 1)
                        } else {
                            if (== ch 41) {  # )
                                set paren_depth (- paren_depth 1)
                            } else {
                                if (== ch 91) {  # [
                                    set bracket_depth (+ bracket_depth 1)
                                } else {
                                    if (== ch 93) {  # ]
                                        set bracket_depth (- bracket_depth 1)
                                    } else { (print "") }
                                }
                            }
                        }
                    }
                }
            } else { (print "") }
        }

        set i (+ i 1)
    }

    # Complete if all depths are zero and not in string
    return (and (and (== brace_depth 0) (== paren_depth 0))
                (and (== bracket_depth 0) (not in_string)))
}
```

### Function: `read_complete_input`

```nano
fn read_complete_input() -> string {
    let buffer_sb: StringBuilder = (sb_new)
    let mut first_line: bool = true
    let nl: string = (string_from_char 10)

    let mut complete: bool = false

    while (not complete) {
        # Use appropriate prompt
        let line: string = if first_line {
            (rl_readline "nano> ")
        } else {
            (rl_readline "....> ")
        }

        # Check for EOF
        if (== (str_length line) 0) {
            return ""
        } else { (print "") }

        # Add to buffer
        if (not first_line) {
            (sb_append buffer_sb nl)
        } else { (print "") }
        (sb_append buffer_sb line)

        # Check if complete
        set complete (is_input_complete (sb_to_string buffer_sb))
        set first_line false
    }

    return (sb_to_string buffer_sb)
}
```

### Integration into REPL Loop

```nano
while running {
    let input: string = (read_complete_input)

    # Check for EOF
    if (== (str_length input) 0) {
        (println "Goodbye!")
        set running false
    } else {
        # Add complete input to history
        (rl_add_history input)

        # Process input...
    }
}
```

## Edge Cases

### 1. String Literals with Quotes
```nano
nano> let msg: string = "He said \"hello\""
```
Need to handle escaped quotes inside strings.

### 2. Comments
```nano
nano> # This is a comment
nano> (+ 1 2)  # inline comment
```
Comments should be ignored when checking completeness.

### 3. Empty Lines
```nano
nano> fn test() -> int {
....>
....>     return 42
....> }
```
Allow empty lines in multi-line input.

### 4. Mismatched Brackets
```nano
nano> (+ 1 2]
```
Should still be considered complete (will fail at compile time).

## Limitations

### Escaped Characters in Strings
The simple implementation doesn't handle:
- Escaped quotes: `\"` inside strings
- Escaped backslashes: `\\`

**Workaround:** For MVP, document this limitation. Users should complete strings on one line.

### Comments
The simple implementation doesn't strip comments before checking completeness.

**Workaround:** Comments inside multi-line input may confuse the parser.

## Example REPL Session

```bash
$ ./bin/multiline_repl

NanoLang REPL (Multi-Line Support)
==================================

nano> fn double(x: int) -> int {
....>     return (* x 2)
....> }
Defined: double(int) -> int

nano> (double 21)
=> 42

nano> let complicated: int = (+
....>     (* 2 3)
....>     (* 4 5))
Defined: complicated

nano> complicated
=> 26

nano> struct Person {
....>     name: string,
....>     age: int
....> }
Defined: Person

nano> :quit
Goodbye!
```

## Implementation Plan

1. Implement `is_input_complete` function
2. Implement `read_complete_input` function
3. Update REPL loop to use `read_complete_input`
4. Test with various multi-line scenarios
5. Document limitations

## Estimated Effort

2-3 hours for basic implementation with limitations documented.

## File to Create

- `examples/language/multiline_repl.nano` - REPL with multi-line input support
