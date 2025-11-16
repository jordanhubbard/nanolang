# Checkers Module System Demonstration

This example demonstrates nanolang's module system by using SDL2 as an external module.

## Architecture

### Module Structure

```
checkers_simple.nano          # Main application
├── import "sdl.nano"         # SDL2 FFI bindings module
└── import "sdl_helpers.nano" # SDL helper functions module
```

### Module Files

1. **`sdl.nano`** - SDL2 FFI Module
   - Contains all `extern fn` declarations for SDL2 functions
   - FFI-only module (no implementations, just declarations)
   - Automatically skipped during module compilation
   - Symbols are loaded into the environment for type checking

2. **`sdl_helpers.nano`** - SDL Helper Functions Module
   - Declares helper functions that wrap SDL struct operations
   - These helpers are implemented in C (`src/runtime/sdl_helpers.c`)
   - Automatically links `sdl_helpers.c` when this module is imported

3. **`checkers_simple.nano`** - Main Application
   - Imports SDL modules
   - Uses SDL functions directly
   - Demonstrates hybrid C/nanolang application

## How It Works

### 1. Module Loading

When `checkers_simple.nano` imports modules:
```nano
import "sdl.nano"
import "sdl_helpers.nano"
```

The compiler:
1. Resolves module paths relative to the current file
2. Loads and parses each module
3. Type-checks each module (without requiring `main`)
4. Adds all module symbols (functions, extern declarations) to the environment
5. Makes symbols available to the main program

### 2. FFI Module Detection

FFI-only modules (modules with only `extern fn` declarations) are:
- ✅ Loaded and type-checked
- ✅ Symbols added to environment
- ❌ **Not compiled to object files** (they're just declarations)
- ✅ SDL includes automatically added to generated C code

### 3. SDL Type Mapping

The transpiler includes SDL-specific type mapping (`get_sdl_c_type`) which:
- Maps nanolang `int` → `SDL_Window*`, `SDL_Renderer*`, etc. based on function signatures
- Handles SDL-specific types: `Uint32`, `Uint8`, `SDL_Event*`, `SDL_Rect*`
- This is a **feature**, not a hack - necessary for FFI with C libraries that use custom types

### 4. Runtime Helper Linking

When `sdl_helpers.nano` is imported:
- The compiler detects the import
- Automatically includes `src/runtime/sdl_helpers.c` in the compilation
- Links the C helper functions that wrap SDL struct operations

## Compilation

```bash
./bin/nanoc examples/checkers_simple.nano -o checkers \
    -I/opt/homebrew/include/SDL2 \
    -L/opt/homebrew/lib \
    -lSDL2
```

**What happens:**
1. Modules are loaded: `sdl.nano`, `sdl_helpers.nano`
2. FFI-only modules are skipped for compilation
3. SDL includes are added to generated C code
4. SDL helpers runtime is linked (because `sdl_helpers.nano` is imported)
5. Final binary links against SDL2 library

## Key Features Demonstrated

✅ **Module System** - Import external modules  
✅ **FFI Modules** - C library bindings as modules  
✅ **Type Safety** - Full type checking across modules  
✅ **Automatic Linking** - Runtime helpers linked based on module imports  
✅ **FFI Type Mapping** - SDL-specific types handled automatically  
✅ **Hybrid Applications** - Mix nanolang and C libraries seamlessly  

## Benefits

1. **Reusability** - SDL module can be used by any nanolang program
2. **Type Safety** - All SDL function calls are type-checked
3. **Clean Separation** - FFI bindings separate from application logic
4. **No Hacks** - Module system handles everything properly
5. **First-Class** - Modules are a core language feature

