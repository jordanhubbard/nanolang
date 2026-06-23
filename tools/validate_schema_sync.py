#!/usr/bin/env python3
"""Validate that C and NanoLang AST/IR definitions match the schema."""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "compiler_schema.json"
C_NANOLANG_H = ROOT / "src" / "nanolang.h"
C_GENERATED = ROOT / "src" / "generated" / "compiler_schema.h"
NANO_SCHEMA = ROOT / "src_nano" / "generated" / "compiler_schema.nano"
NANO_AST = ROOT / "src_nano" / "generated" / "compiler_ast.nano"

class ValidationResult:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []
    
    def error(self, msg: str):
        self.errors.append(f"❌ ERROR: {msg}")
    
    def warning(self, msg: str):
        self.warnings.append(f"⚠️  WARNING: {msg}")
    
    def passed_check(self, msg: str):
        self.passed.append(f"✓ {msg}")
    
    def print_results(self):
        print("\n" + "=" * 70)
        print("SCHEMA VALIDATION RESULTS")
        print("=" * 70 + "\n")
        
        if self.errors:
            print("ERRORS:")
            for err in self.errors:
                print(f"  {err}")
            print()
        
        if self.warnings:
            print("WARNINGS:")
            for warn in self.warnings:
                print(f"  {warn}")
            print()
        
        if self.passed:
            print(f"PASSED CHECKS ({len(self.passed)}):")
            for check in self.passed[:10]:  # Show first 10
                print(f"  {check}")
            if len(self.passed) > 10:
                print(f"  ... and {len(self.passed) - 10} more")
            print()
        
        total_checks = len(self.errors) + len(self.warnings) + len(self.passed)
        success_rate = (len(self.passed) / total_checks * 100) if total_checks > 0 else 0
        
        print("=" * 70)
        print(f"SUMMARY: {len(self.passed)}/{total_checks} checks passed ({success_rate:.1f}%)")
        print(f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}")
        print("=" * 70)
        
        return len(self.errors) == 0

def load_schema() -> Dict:
    """Load the canonical schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)

def extract_c_enum(file_path: Path, enum_name: str) -> Set[str]:
    """Extract enum values from C code."""
    content = file_path.read_text()
    
    # Try different patterns
    patterns = [
        rf'typedef\s+enum\s*\{{([^}}]+)\}}\s*{enum_name};',
        rf'enum\s+{enum_name}\s*\{{([^}}]+)\}};',
        rf'typedef\s+enum\s+{enum_name}\s*\{{([^}}]+)\}}\s*{enum_name};',
        rf'\}}\s*{enum_name};\s*/\*.*?\*/',  # With trailing comment
    ]
    
    match = None
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            break
    
    if not match:
        # Try to find just the enum body after enum_name
        pattern = rf'{enum_name}\s*(?:=\s*\d+)?\s*[,;]'
        if re.search(pattern, content):
            # Enum values are inline, extract them all
            pattern = rf'typedef\s+enum.*?\{{(.*?)\}}\s*{enum_name};'
            match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return set()
    
    enum_body = match.group(1)
    values = set()
    for line in enum_body.split('\n'):
        line = line.strip()
        if line and not line.startswith('/*') and not line.startswith('//'):
            # Remove comments
            line = re.sub(r'/\*.*?\*/', '', line)
            # Extract enum value name (before '=' or ',')
            match = re.match(r'(\w+)', line)
            if match:
                values.add(match.group(1))
    
    return values

def extract_nano_enum(file_path: Path, enum_name: str) -> Set[str]:
    """Extract enum values from NanoLang code."""
    content = file_path.read_text()
    pattern = rf'enum\s+{enum_name}\s*\{{([^}}]+)\}}'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return set()
    
    enum_body = match.group(1)
    values = set()
    for line in enum_body.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            # Extract enum value name (before '=' or ',')
            match = re.match(r'(\w+)', line)
            if match:
                values.add(match.group(1))
    
    return values

def extract_c_structs(file_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Extract struct definitions from C code."""
    content = file_path.read_text()
    structs = {}
    
    # Match typedef struct patterns
    pattern = r'typedef\s+struct\s+(\w+)?\s*\{([^}]+)\}\s*(\w+);'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        struct_name = match.group(3)  # Use typedef name
        struct_body = match.group(2)
        
        fields = []
        for line in struct_body.split('\n'):
            line = line.strip()
            if line and not line.startswith('/*') and not line.startswith('//'):
                # Try to parse type and name
                parts = line.rstrip(';').split()
                if len(parts) >= 2:
                    field_type = ' '.join(parts[:-1])
                    field_name = parts[-1].rstrip('*')
                    fields.append((field_name, field_type))
        
        if fields:
            structs[struct_name] = fields
    
    return structs

def check_token_enums(schema: Dict, result: ValidationResult):
    """Check TokenType enums match schema."""
    schema_tokens = set(schema["tokens"])
    
    # Check C enum
    c_tokens = extract_c_enum(C_GENERATED, "TokenType")
    if not c_tokens:
        result.error("Could not find TokenType enum in C generated code")
    else:
        missing_in_c = schema_tokens - c_tokens
        extra_in_c = c_tokens - schema_tokens
        
        if missing_in_c:
            result.error(f"C TokenType missing: {', '.join(sorted(missing_in_c))}")
        if extra_in_c:
            result.warning(f"C TokenType has extra: {', '.join(sorted(extra_in_c))}")
        if not missing_in_c and not extra_in_c:
            result.passed_check(f"C TokenType enum matches schema ({len(schema_tokens)} tokens)")
    
    # Check NanoLang enum
    nano_tokens = extract_nano_enum(NANO_SCHEMA, "LexerTokenType")
    if not nano_tokens:
        result.error("Could not find LexerTokenType enum in NanoLang generated code")
    else:
        missing_in_nano = schema_tokens - nano_tokens
        extra_in_nano = nano_tokens - schema_tokens
        
        if missing_in_nano:
            result.error(f"NanoLang LexerTokenType missing: {', '.join(sorted(missing_in_nano))}")
        if extra_in_nano:
            result.warning(f"NanoLang LexerTokenType has extra: {', '.join(sorted(extra_in_nano))}")
        if not missing_in_nano and not extra_in_nano:
            result.passed_check(f"NanoLang LexerTokenType enum matches schema ({len(schema_tokens)} tokens)")

def check_parse_node_enums(schema: Dict, result: ValidationResult):
    """Check ParseNodeType enums match schema."""
    schema_nodes = set(schema["parse_nodes"])
    
    # Check C enum
    c_nodes = extract_c_enum(C_GENERATED, "ParseNodeType")
    if not c_nodes:
        result.error("Could not find ParseNodeType enum in C generated code")
    else:
        missing_in_c = schema_nodes - c_nodes
        extra_in_c = c_nodes - schema_nodes
        
        if missing_in_c:
            result.error(f"C ParseNodeType missing: {', '.join(sorted(missing_in_c))}")
        if extra_in_c:
            result.warning(f"C ParseNodeType has extra: {', '.join(sorted(extra_in_c))}")
        if not missing_in_c and not extra_in_c:
            result.passed_check(f"C ParseNodeType enum matches schema ({len(schema_nodes)} nodes)")
    
    # Check NanoLang enum
    nano_nodes = extract_nano_enum(NANO_SCHEMA, "ParseNodeType")
    if not nano_nodes:
        result.error("Could not find ParseNodeType enum in NanoLang generated code")
    else:
        missing_in_nano = schema_nodes - nano_nodes
        extra_in_nano = nano_nodes - schema_nodes
        
        if missing_in_nano:
            result.error(f"NanoLang ParseNodeType missing: {', '.join(sorted(missing_in_nano))}")
        if extra_in_nano:
            result.warning(f"NanoLang ParseNodeType has extra: {', '.join(sorted(extra_in_nano))}")
        if not missing_in_nano and not extra_in_nano:
            result.passed_check(f"NanoLang ParseNodeType enum matches schema ({len(schema_nodes)} nodes)")

def check_ast_node_enum(schema: Dict, result: ValidationResult):
    """Check ASTNodeType enum in C header matches schema parse nodes."""
    # Special handling for ASTNodeType which uses } ASTNodeType; pattern
    content = C_NANOLANG_H.read_text()
    # Match from typedef enum to ASTNodeType;
    pattern = r'typedef\s+enum\s*\{(.*?)\}\s*ASTNodeType;'
    match = re.search(pattern, content, re.DOTALL)
    
    c_ast_nodes = set()
    if match:
        enum_body = match.group(1)
        for line in enum_body.split('\n'):
            line = line.strip()
            if line and not line.startswith('/*'):
                # Remove inline comments
                line = re.sub(r'/\*.*?\*/', '', line).strip()
                # Extract enum value name
                m = re.match(r'(AST_\w+)', line)
                if m:
                    c_ast_nodes.add(m.group(1))
    
    if not c_ast_nodes:
        result.warning("Could not parse ASTNodeType enum in C nanolang.h - skipping check")
        return
    
    # Map PNODE_* to AST_* naming
    expected_ast_nodes = set()
    for pnode in schema["parse_nodes"]:
        ast_name = pnode.replace("PNODE_", "AST_")
        expected_ast_nodes.add(ast_name)
    
    missing = expected_ast_nodes - c_ast_nodes
    extra = c_ast_nodes - expected_ast_nodes
    
    if missing:
        result.warning(f"C ASTNodeType missing: {', '.join(sorted(missing))}")
    if extra:
        result.passed_check(f"C ASTNodeType has additional nodes: {', '.join(sorted(extra))}")
    
    # Count how many match
    matches = len(expected_ast_nodes & c_ast_nodes)
    result.passed_check(f"C ASTNodeType matches {matches}/{len(expected_ast_nodes)} schema parse nodes")

def check_ast_structs(schema: Dict, result: ValidationResult):
    """Check AST struct definitions match schema."""
    for struct_def in schema.get("nano_structs", []):
        struct_name = struct_def["name"]
        schema_fields = {fname: ftype for fname, ftype in struct_def["fields"]}
        
        # Check in NanoLang generated AST
        if NANO_AST.exists():
            content = NANO_AST.read_text()
            pattern = rf'struct\s+{struct_name}\s*\{{([^}}]+)\}}'
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                nano_fields = {}
                for line in match.group(1).split('\n'):
                    line = line.strip().rstrip(',')
                    if ':' in line and not line.startswith('#'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            fname = parts[0].strip()
                            ftype = parts[1].strip()
                            nano_fields[fname] = ftype
                
                schema_field_names = set(schema_fields.keys())
                nano_field_names = set(nano_fields.keys())
                
                missing = schema_field_names - nano_field_names
                extra = nano_field_names - schema_field_names
                
                if missing:
                    result.error(f"NanoLang {struct_name} missing fields: {', '.join(sorted(missing))}")
                if extra:
                    result.warning(f"NanoLang {struct_name} has extra fields: {', '.join(sorted(extra))}")
                if not missing and not extra:
                    result.passed_check(f"NanoLang {struct_name} has all {len(schema_fields)} schema fields")

def main():
    """Run all validation checks."""
    result = ValidationResult()
    
    # Load schema
    try:
        schema = load_schema()
        result.passed_check(f"Loaded schema from {SCHEMA_PATH.name}")
    except Exception as e:
        result.error(f"Failed to load schema: {e}")
        result.print_results()
        return 1
    
    # Check all enums and structs
    check_token_enums(schema, result)
    check_parse_node_enums(schema, result)
    check_ast_node_enum(schema, result)
    check_ast_structs(schema, result)
    
    # Print results
    success = result.print_results()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

