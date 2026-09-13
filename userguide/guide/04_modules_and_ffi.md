# Modules, FFI, and Resources

A module is a visibility and safety boundary. New code should use `module`, an alias, and qualified public names.

## Import a Module

```nano
module "modules/std/mathx/mathx.nano" as mathx

fn bounded(value: int) -> int {
    return (mathx.mathx_clamp value 0 100)
}

shadow bounded {
    assert (== (bounded -5) 0)
    assert (== (bounded 120) 100)
}
```

The parser still accepts legacy `import` and `from ... import ...` forms. Do not choose them for new code.

## Visibility

Declarations are private by default. Mark the intended surface with `pub`:

```nano
fn internal_scale(value: int) -> int {
    return (* value 2)
}

pub fn transform(value: int) -> int {
    return (+ (internal_scale value) 1)
}
```

Private helpers remain callable within their module. Importers can call only public declarations.

## Foreign Functions

Foreign declarations use `extern fn`. A direct call requires `unsafe` unless the whole imported module is unsafe:

```nano
extern fn c_close(fd: int) -> int

fn close_fd(fd: int) -> int {
    unsafe {
        return (c_close fd)
    }
}
```

Keep unsafe regions narrow. Validate foreign values at the boundary and expose a typed wrapper when one can honestly be provided. Direct `extern` is not the only host boundary: NSI method ids, unforgeable capabilities, and the POSIX fabric are the 4.x runtime path. See [Secure Runtime](08_secure_runtime.md).

## Resource Types

`resource struct` marks an affine resource that should be consumed at most once:

```nano
resource struct FileHandle {
    fd: int
}
```

My resource checker detects important use-after-consume and repeated-consumption cases. It is not a complete ownership proof. Annotate resource locals explicitly and inspect every return path for cleanup.

## Module Metadata

Three files serve different jobs:

| File | Purpose |
| --- | --- |
| `.nano` | Source and public declarations |
| `module.json` | Native build sources, flags, packages, and ownership metadata |
| `module.manifest.json` | Discovery metadata, stability, capabilities, and examples |

`module.json` stays build metadata. Isolation, restart, budgets, and required capabilities live on `module.manifest.json` in a portable `nsi` block. See [Secure Runtime](08_secure_runtime.md) and the generated [module inventory](../generated/modules.md) for what exists now.

For Clang and GCC C builds admitted to my retained-input path, failed capture
stops the build before final object compilation. GCC must retain assembler reads
as well as preprocessed C; Clang's external-assembler mode has the same rule.
I preserve the previous successful generation and report capture diagnostics.
Check missing inputs, assembler support, and tool failures before retrying.
On supported Linux GNU assembler versions, macro reads can require my installed
`nano_as_capture.so` beside the driver. `NANO_AS_CAPTURE_HELPER` selects another
copy; a missing helper is a build failure when literal capture cannot suffice.
I admit literal assembler include paths through `-Wa,-I,dir`, `-Wa,-Idir`,
`-Xassembler -I -Xassembler dir`, and `-Xassembler -Idir`. I preserve their
order across package, common, and platform C flags. Keep paired forms in one
flag fragment, such as `"-Xassembler -I -Xassembler 'assembler includes'"`.
Quote paths containing
spaces; use `-Xassembler` for paths containing commas. I keep these arguments
out of separate C preprocessing and link-only jobs. Integrated Clang capture
preserves the native driver's C-header search order as well as assembler
lookup. This does not admit arbitrary `-Wa` options or extend the supported
compiler/assembler versions.
Source and flag modes outside this path retain their existing compatibility
behavior; they have no retained-input guarantee.
