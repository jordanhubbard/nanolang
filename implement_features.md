# Implementation Plan - All 15 Features

## Quick Assessment

Looking at the parser, I found:
- ✅ **set statement** is already implemented in parse_statement!
- ❌ No token helper functions for: for, print, assert, import, opaque, shadow, match
- ❌ No parsing functions for these features yet

## Implementation Strategy

Since nanolang uses a Lisp-like syntax with parenthesized expressions, many "statements" are actually function calls:
- `(print expr)` - function call, not statement
- `(assert condition)` - function call, not statement

So some features may not need special parsing!

## Revised Plan

### Already Done ✅
1. set statements - Already in parse_statement

### Need Token Helpers (~1 hour)
2. Add token helper functions for: for, import, shadow, opaque, match

### Need Parsing Logic (~17 hours)
3. for loops - parse_statement (1 hour)
4. import - parse_definition (30 min)
5. shadow - parse_definition (1 hour)
6. opaque - parse_definition (30 min)
7. Array literals - parse_primary for [ ] (1 hour)
8. Float handling - parse_primary number check (15 min)
9. Struct literals - parse_primary Identifier{ } (2 hours)
10. Field access - postfix in parse_expression (2.5 hours)
11. Tuple index - postfix in parse_expression (included above)
12. Match expressions - parse_primary or expression (4 hours)
13. Union construction - postfix or primary (2 hours)
14. Tuple literals - parse_primary ( ) handling (2 hours)

### Testing (~2 hours)
15. Test each feature
16. Shadow tests
17. Integration tests

## Total: ~20 hours (adjusted from 18-21)
