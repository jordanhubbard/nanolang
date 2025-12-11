#!/usr/bin/env python3
import re

with open('src_nano/parser_mvp.nano', 'r') as f:
    content = f.read()

# Find the place after token_union shadow test
insertion_point = content.find('shadow token_union { assert (== (token_union) 53) }')
if insertion_point == -1:
    print("Error: Could not find token_union shadow test")
    exit(1)

# Find the end of that line
end_of_line = content.find('\n', insertion_point)
insertion_point = end_of_line + 1

# Skip blank line if present
if content[insertion_point:insertion_point+1] == '\n':
    insertion_point += 1

# Insert the new token functions
new_tokens = '''/* Additional token helper functions for full feature support */
fn token_for() -> int { return 44 }
fn token_in() -> int { return 45 }
fn token_assert() -> int { return 47 }
fn token_shadow() -> int { return 48 }
fn token_match() -> int { return 54 }
fn token_import() -> int { return 55 }
fn token_as() -> int { return 56 }
fn token_opaque() -> int { return 57 }
fn token_lbracket() -> int { return 27 }
fn token_rbracket() -> int { return 28 }
fn token_dot() -> int { return 33 }

'''

content = content[:insertion_point] + new_tokens + content[insertion_point:]

with open('src_nano/parser_mvp.nano', 'w') as f:
    f.write(content)

print("✅ Added token helper functions")
