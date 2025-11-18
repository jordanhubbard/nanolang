# pt2_module - ProTracker MOD File Module

**Status:** Architecture change in progress

## Original Plan
Copy pt2-clone's module loader/saver code

## Problem Discovered
- pt2-clone code is tightly coupled to editor state
- Requires 20+ header files
- Uses platform-specific Unicode (UNICHAR)
- Not designed as a standalone library

## New Approach
Create standalone clean MOD loader:
1. Simple C API - no dependencies on pt2-clone globals
2. Standard C types (no UNICHAR)
3. Clean module_t structure
4. CRUD operations: load, save, get/set data

## Implementation
See standalone_loader.c for clean implementation

## Files
- `module.json` - Module configuration
- `pt2_module.nano` - Nanolang FFI bindings  
- `pt2_mod_format.h` - MOD file format structures
- `pt2_mod_loader.c` - Standalone MOD loader
- `pt2_mod_api.c` - Nanolang API wrapper
