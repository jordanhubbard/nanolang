# Unicode Module - Cross-Platform UTF-8 String Support

NanoLang's Unicode module provides grapheme-aware string operations using the industry-standard utf8proc library.

## Installation

### Prerequisites

The Unicode module requires the `utf8proc` library. Install it using your system's package manager:

#### macOS (Homebrew)
```bash
brew install utf8proc
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libutf8proc-dev
```

#### Fedora/RHEL/CentOS
```bash
sudo yum install utf8proc-devel
# Or on newer systems:
sudo dnf install utf8proc-devel
```

#### Arch Linux
```bash
sudo pacman -S libutf8proc
```

#### Build from Source (Universal)
```bash
git clone https://github.com/JuliaStrings/utf8proc.git
cd utf8proc
make
sudo make install
sudo ldconfig  # Linux only
```

## Verification

After installation, verify utf8proc is available:

```bash
# Should return version information
pkg-config --modversion libutf8proc

# Should return compiler flags
pkg-config --cflags --libs libutf8proc
```

Expected output (paths may vary):
```
-I/usr/include -L/usr/lib -lutf8proc
```

## Usage

### Basic Example

```nano
import "modules/unicode/unicode.nano" as Unicode

fn main() -> int {
    let text: string = "Hello 世界 🌍"
    
    unsafe {
        /* Get byte length (for file I/O, buffers) */
        let bytes: int = (Unicode.str_byte_length text)
        (println bytes)  /* Prints: 17 */
        
        /* Get grapheme length (for user display) */
        let chars: int = (Unicode.str_grapheme_length text)
        (println chars)  /* Prints: 9 */
    }
    
    return 0
}
```

### Emoji Support

```nano
let family: string = "👨‍👩‍👧‍👦"  /* Family emoji */

unsafe {
    (println (Unicode.str_byte_length family))      /* 25 bytes */
    (println (Unicode.str_grapheme_length family))  /* 1 grapheme ✓ */
}
```

## API Reference

| Function | Signature | Description |
|----------|-----------|-------------|
| `str_byte_length` | `(string) -> int` | Get UTF-8 byte count |
| `str_grapheme_length` | `(string) -> int` | Get user-perceived character count |
| `str_codepoint_at` | `(string, int) -> int` | Get Unicode codepoint at byte index |
| `str_grapheme_at` | `(string, int) -> string` | Get grapheme cluster at index |
| `str_to_lowercase` | `(string) -> string` | Unicode-aware lowercase |
| `str_to_uppercase` | `(string) -> string` | Unicode-aware uppercase |
| `str_normalize` | `(string, int) -> string` | Unicode normalization (NFC/NFD/NFKC/NFKD) |
| `str_is_ascii` | `(string) -> bool` | Check if pure ASCII (fast-path) |
| `str_is_valid_utf8` | `(string) -> bool` | Validate UTF-8 encoding |

## Building

### Automatic (via pkg-config)
```bash
cd modules/unicode
./build.sh
```

### Manual Compilation
```bash
# Compile FFI
gcc -std=c99 -fPIC -c unicode_ffi.c \
    $(pkg-config --cflags libutf8proc) \
    -o unicode_ffi.o

# Link into your program
gcc -o my_program my_program.c unicode_ffi.o \
    $(pkg-config --libs libutf8proc)
```

## Platform Notes

### macOS
- Library location: `/opt/homebrew/Cellar/utf8proc/*/lib/` (Apple Silicon) or `/usr/local/Cellar/utf8proc/*/lib/` (Intel)
- Headers: `/opt/homebrew/include/` or `/usr/local/include/`
- Uses `.dylib` shared libraries

### Linux
- Library location: `/usr/lib/` or `/usr/lib/x86_64-linux-gnu/`
- Headers: `/usr/include/`
- Uses `.so` shared libraries

### pkg-config (Universal)
The module uses `pkg-config` for automatic platform detection:
```bash
pkg-config --cflags --libs libutf8proc
```

This works identically on both macOS and Linux if utf8proc is installed correctly.

## Troubleshooting

### "utf8proc.h not found"
- **Solution:** Install utf8proc development package (see Installation above)
- **Verify:** `pkg-config --cflags libutf8proc`

### "library not found -lutf8proc"
- **Solution:** Install utf8proc library package
- **Verify:** `pkg-config --libs libutf8proc`
- **Linux:** May need `sudo ldconfig` after installation

### pkg-config fails
- **macOS:** Ensure Homebrew's pkg-config is in PATH
- **Linux:** Install pkg-config: `sudo apt-get install pkg-config`

### Runtime: "dyld: Library not loaded" (macOS)
- **Solution:** Check library path:
  ```bash
  otool -L your_binary  # See linked libraries
  ```
- **Fix:** Set `DYLD_LIBRARY_PATH` or reinstall utf8proc

### Runtime: "error while loading shared libraries" (Linux)
- **Solution:** Update library cache:
  ```bash
  sudo ldconfig
  ```
- **Verify:** `ldconfig -p | grep utf8proc`

## Testing

Run the built-in tests:
```bash
# Compile and run demo
nanoc examples/unicode_demo.nano -o bin/unicode_demo
./bin/unicode_demo
```

Expected output:
```
=== Unicode String Operations Demo ===

ASCII Tests:
  String: Hello, World!
  Byte length: 13
  Grapheme length: 13
  Is ASCII: true

Emoji Tests:
  String: 😀😁😂🤣
  Byte length: 16
  Grapheme length: 4
  Is ASCII: false

Complex Grapheme Test:
  Family emoji: 👨‍👩‍👧‍👦
  Byte length: 25
  Grapheme length: 1

✓ All Unicode operations completed successfully!
```

## Performance

- **ASCII fast-path:** `O(1)` for pure ASCII strings
- **Grapheme counting:** `O(n)` where n = byte length
- **Normalization:** `O(n)` with some allocation
- **Case conversion:** `O(n)` with UTF-8 encoding/decoding

**Recommendation:** Use `str_is_ascii()` to enable fast paths when possible.

## License

This module uses utf8proc, which is licensed under the MIT License.

## References

- utf8proc: https://github.com/JuliaStrings/utf8proc
- Unicode Standard: https://unicode.org/
- UAX #29 (Text Segmentation): https://unicode.org/reports/tr29/
- UAX #15 (Normalization): https://unicode.org/reports/tr15/

