# Check what parsing functions are missing

import re

with open('src_nano/parser_mvp.nano', 'r') as f:
    content = f.read()

# Find all parse_ function definitions
parse_functions = re.findall(r'^fn (parse_\w+)\(', content, re.MULTILINE)

print("✅ Currently Implemented Parsing Functions:")
for func in sorted(set(parse_functions)):
    print(f"  - {func}")

print("\n❌ Missing Parsing Functions (from architecture):")
missing = [
    "parse_float_literal",
    "parse_array_literal", 
    "parse_set_statement",
    "parse_for_statement",
    "parse_print_statement",
    "parse_assert_statement",
    "parse_struct_literal",
    "parse_field_access",
    "parse_tuple_index", 
    "parse_union_construct",
    "parse_match_expr",
    "parse_import",
    "parse_shadow",
    "parse_opaque_type",
    "parse_tuple_literal",
]

for func in missing:
    if func.replace('_', '') not in ''.join(parse_functions).replace('_', ''):
        print(f"  - {func}")

print(f"\n📊 Summary:")
print(f"  Implemented: {len(set(parse_functions))} functions")
print(f"  Architecture ready for: 31 node types")
