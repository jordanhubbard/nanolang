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

Keep unsafe regions narrow. Validate foreign values at the boundary and expose a typed wrapper when one can honestly be provided.

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

Pure modules do not always need native build metadata. See the generated [module inventory](../generated/modules.md) for what exists now.
