#!/usr/bin/env python3
"""
Generate API documentation markdown from NanoLang module reflection.

Usage:
    python3 scripts/generate_module_api_docs.py <module.nano> <output.md>
    
This script:
1. Runs nanoc with --reflect to extract module API as JSON
2. Parses the JSON
3. Generates formatted markdown documentation
4. Writes to output file
"""

import sys
import json
import subprocess
import os
import tempfile

def type_display(param):
    """Format a type for display."""
    type_str = param.get('type', 'unknown')
    struct_name = param.get('struct_name')
    if struct_name:
        return f"{type_str}<{struct_name}>"
    return type_str

def generate_function_docs(exports):
    """Generate markdown for all public functions."""
    lines = ["### Functions\n"]
    
    # Filter for public/extern functions, excluding builtins
    builtin_names = {
        'range', 'abs', 'min', 'max', 'print', 'println', 'sqrt', 'pow', 
        'floor', 'ceil', 'round', 'sin', 'cos', 'tan', 'str_length',
        'str_equals', 'str_concat', 'int_to_string', 'float_to_string',
        'string_to_int', 'string_to_float', 'read_line', 'exit',
        'str_substring', 'str_contains', 'at', 'array_length', 'array_new',
        'array_set', 'array_get', 'array_push', 'array_pop', 'map', 'filter',
        'reduce', 'fold', 'assert', 'getcwd', 'getenv', 'tmp_dir', 'clock',
        'int_to_char', 'char_to_int'
    }
    
    functions = [
        e for e in exports 
        if e['kind'] == 'function' 
        and (e.get('is_public') or e.get('is_extern'))
        and e['name'] not in builtin_names
    ]
    
    if not functions:
        lines.append("*No public functions*\n")
        return lines
    
    for func in functions:
        name = func['name']
        signature = func['signature']
        params = func.get('params', [])
        return_type = func['return_type']
        
        # Function signature as header
        lines.append(f"#### `{signature}`\n")
        
        # Parameters table if any
        if params:
            lines.append("**Parameters:**\n")
            lines.append("| Name | Type |")
            lines.append("|------|------|")
            for p in params:
                param_name = p['name']
                param_type = type_display(p)
                lines.append(f"| `{param_name}` | `{param_type}` |")
            lines.append("")
        
        # Return type
        lines.append(f"**Returns:** `{return_type}`\n")
        lines.append("")
    
    return lines

def generate_struct_docs(exports):
    """Generate markdown for all public structs."""
    lines = ["### Structs\n"]
    
    structs = [e for e in exports if e['kind'] == 'struct' and e.get('is_public')]
    
    if not structs:
        lines.append("*No public structs*\n")
        return lines
    
    for struct in structs:
        name = struct['name']
        fields = struct.get('fields', [])
        
        lines.append(f"#### `struct {name}`\n")
        
        if fields:
            lines.append("**Fields:**\n")
            lines.append("| Name | Type |")
            lines.append("|------|------|")
            for field in fields:
                field_name = field['name']
                field_type = type_display(field)
                lines.append(f"| `{field_name}` | `{field_type}` |")
            lines.append("")
        else:
            lines.append("*Opaque struct (no public fields)*\n")
        
        lines.append("")
    
    return lines

def generate_enum_docs(exports):
    """Generate markdown for all public enums."""
    lines = ["### Enums\n"]
    
    enums = [e for e in exports if e['kind'] == 'enum' and e.get('is_public')]
    
    if not enums:
        lines.append("*No public enums*\n")
        return lines
    
    for enum in enums:
        name = enum['name']
        variants = enum.get('variants', [])
        
        lines.append(f"#### `enum {name}`\n")
        
        if variants:
            lines.append("**Variants:**\n")
            for variant in variants:
                lines.append(f"- `{variant}`")
            lines.append("")
        
        lines.append("")
    
    return lines

def generate_union_docs(exports):
    """Generate markdown for all public unions."""
    lines = ["### Unions\n"]
    
    unions = [e for e in exports if e['kind'] == 'union' and e.get('is_public')]
    
    if not unions:
        lines.append("*No public unions*\n")
        return lines
    
    for union in unions:
        name = union['name']
        variants = union.get('variants', [])
        is_generic = union.get('is_generic', False)
        
        generic_str = "<T>" if is_generic else ""
        lines.append(f"#### `union {name}{generic_str}`\n")
        
        if variants:
            lines.append("**Variants:**\n")
            for variant in variants:
                variant_name = variant['name']
                fields = variant.get('fields', [])
                if fields:
                    field_strs = [f"{f['name']}: {type_display(f)}" for f in fields]
                    lines.append(f"- `{variant_name}({', '.join(field_strs)})`")
                else:
                    lines.append(f"- `{variant_name}`")
            lines.append("")
        
        lines.append("")
    
    return lines

def generate_opaque_docs(exports):
    """Generate markdown for all opaque types."""
    lines = ["### Opaque Types\n"]
    
    opaques = [e for e in exports if e['kind'] == 'opaque']
    
    if not opaques:
        lines.append("*No opaque types*\n")
        return lines
    
    for opaque in opaques:
        name = opaque['name']
        lines.append(f"- `opaque type {name}`")
    
    lines.append("")
    return lines

def generate_constant_docs(exports):
    """Generate markdown for all constants."""
    lines = ["### Constants\n"]
    
    constants = [e for e in exports if e['kind'] == 'constant']
    
    if not constants:
        lines.append("*No constants*\n")
        return lines
    
    lines.append("| Name | Type | Value |")
    lines.append("|------|------|-------|")
    
    for const in constants:
        name = const['name']
        type_str = const['type']
        value = const.get('value', '*not available*')
        if isinstance(value, str):
            value = f'"{value}"'
        lines.append(f"| `{name}` | `{type_str}` | `{value}` |")
    
    lines.append("")
    return lines

def generate_api_docs(module_path, output_path):
    """Generate complete API documentation for a module."""
    
    # Get module name from path
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    
    print(f"Generating API docs for {module_name}...")
    
    # Run reflection to get JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json_path = tmp.name
    
    try:
        result = subprocess.run(
            ['./bin/nanoc', module_path, '--reflect', json_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"Error running reflection: {result.stderr}", file=sys.stderr)
            return False
        
        # Parse JSON
        with open(json_path, 'r') as f:
            api_data = json.load(f)
        
        exports = api_data.get('exports', [])
        
        # Generate markdown
        lines = []
        lines.append(f"# {module_name} API Reference\n")
        lines.append(f"*Auto-generated from module reflection*\n")
        lines.append("")
        
        # Add sections
        lines.extend(generate_function_docs(exports))
        lines.extend(generate_struct_docs(exports))
        lines.extend(generate_enum_docs(exports))
        lines.extend(generate_union_docs(exports))
        lines.extend(generate_opaque_docs(exports))
        lines.extend(generate_constant_docs(exports))
        
        # Write output
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Generated {output_path}")
        return True
        
    finally:
        if os.path.exists(json_path):
            os.unlink(json_path)

def main():
    if len(sys.argv) != 3:
        print("Usage: generate_module_api_docs.py <module.nano> <output.md>", file=sys.stderr)
        sys.exit(1)
    
    module_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(module_path):
        print(f"Error: Module file not found: {module_path}", file=sys.stderr)
        sys.exit(1)
    
    if not generate_api_docs(module_path, output_path):
        sys.exit(1)

if __name__ == '__main__':
    main()
