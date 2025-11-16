# SDL Module for nanolang

This module provides FFI bindings to SDL2 and SDL2_ttf libraries.

## Installation

### Build the module package:
```bash
cd modules/sdl
../tools/build_module.sh .
```

### Install the module:
```bash
../tools/install_module.sh sdl.nano.tar.gz
```

Or manually:
```bash
mkdir -p ~/.nanolang/modules
cp sdl.nano.tar.gz ~/.nanolang/modules/
export NANO_MODULE_PATH=~/.nanolang/modules
```

## Usage

```nano
import "sdl.nano"

fn main() -> int {
    let result: int = (SDL_Init (SDL_INIT_VIDEO))
    # ... use SDL functions
    return 0
}
```

## Compilation

When compiling programs that use this module, you need SDL2 include and library paths:

```bash
nanoc program.nano -o program -I/opt/homebrew/include/SDL2 -L/opt/homebrew/lib -lSDL2 -lSDL2_ttf
```

On Linux:
```bash
nanoc program.nano -o program -I/usr/include/SDL2 -L/usr/lib/x86_64-linux-gnu -lSDL2 -lSDL2_ttf
```
