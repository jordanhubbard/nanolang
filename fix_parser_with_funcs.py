#!/usr/bin/env python3
import re

with open('src_nano/parser_mvp.nano', 'r') as f:
    content = f.read()

# Find and fix parser_with_position
old_pattern1 = r'(fn parser_with_position.*?return Parser \{[^}]*?numbers: p\.numbers,)'
new_insert1 = r'\1\n        floats: p.floats,\n        strings: p.strings,\n        bools: p.bools,'

content = re.sub(old_pattern1, new_insert1, content, flags=re.DOTALL)

# Add after calls: p.calls,
old_pattern2 = r'(calls: p\.calls,)'
new_insert2 = r'\1\n        array_literals: p.array_literals,'
content = re.sub(old_pattern2, new_insert2, content)

# Add after whiles: p.whiles,
old_pattern3 = r'(whiles: p\.whiles,)'
new_insert3 = r'\1\n        fors: p.fors,'
content = re.sub(old_pattern3, new_insert3, content)

# Add after blocks: p.blocks,
old_pattern4 = r'(blocks: p\.blocks,)'
new_insert4 = r'\1\n        prints: p.prints,\n        asserts: p.asserts,'
content = re.sub(old_pattern4, new_insert4, content)

# Add after functions: p.functions,
old_pattern5 = r'(functions: p\.functions,)'
new_insert5 = r'\1\n        shadows: p.shadows,'
content = re.sub(old_pattern5, new_insert5, content)

# Add after structs: p.structs,
old_pattern6 = r'(structs: p\.structs,)'
new_insert6 = r'\1\n        struct_literals: p.struct_literals,\n        field_accesses: p.field_accesses,'
content = re.sub(old_pattern6, new_insert6, content)

# Add after unions: p.unions,
old_pattern7 = r'(unions: p\.unions,)'
new_insert7 = r'\1\n        union_constructs: p.union_constructs,\n        matches: p.matches,\n        imports: p.imports,\n        opaque_types: p.opaque_types,\n        tuple_literals: p.tuple_literals,\n        tuple_indices: p.tuple_indices,'
content = re.sub(old_pattern7, new_insert7, content)

# Add after numbers_count
old_pattern8 = r'(numbers_count: p\.numbers_count,)'
new_insert8 = r'\1\n        floats_count: p.floats_count,'
content = re.sub(old_pattern8, new_insert8, content)

# Add after calls_count (but not if it's being set like calls_count: calls_count)
old_pattern9 = r'(calls_count: (?:p\.)?calls_count,)'
new_insert9 = r'\1\n        array_literals_count: p.array_literals_count,'
content = re.sub(old_pattern9, new_insert9, content)

# Add after whiles_count
old_pattern10 = r'(whiles_count: p\.whiles_count,)'
new_insert10 = r'\1\n        fors_count: p.fors_count,'
content = re.sub(old_pattern10, new_insert10, content)

# Add after blocks_count
old_pattern11 = r'(blocks_count: p\.blocks_count,)'
new_insert11 = r'\1\n        prints_count: p.prints_count,\n        asserts_count: p.asserts_count,'
content = re.sub(old_pattern11, new_insert11, content)

# Add after functions_count
old_pattern12 = r'(functions_count: p\.functions_count,)'
new_insert12 = r'\1\n        shadows_count: p.shadows_count,'
content = re.sub(old_pattern12, new_insert12, content)

# Add after structs_count
old_pattern13 = r'(structs_count: p\.structs_count,)'
new_insert13 = r'\1\n        struct_literals_count: p.struct_literals_count,\n        field_accesses_count: p.field_accesses_count,'
content = re.sub(old_pattern13, new_insert13, content)

# Add after unions_count
old_pattern14 = r'(unions_count: p\.unions_count,)'
new_insert14 = r'\1\n        union_constructs_count: p.union_constructs_count,\n        matches_count: p.matches_count,\n        imports_count: p.imports_count,\n        opaque_types_count: p.opaque_types_count,\n        tuple_literals_count: p.tuple_literals_count,\n        tuple_indices_count: p.tuple_indices_count,'
content = re.sub(old_pattern14, new_insert14, content)

with open('src_nano/parser_mvp.nano', 'w') as f:
    f.write(content)

print("✅ Fixed parser_with_* functions")
