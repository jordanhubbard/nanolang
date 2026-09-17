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

## Qualified Type Identity

Qualify imported record types as well as their constructors:

```nano
module "geometry.nano" as geometry

fn origin() -> geometry.Point {
    return geometry.Point { x: 0, y: 0 }
}
```

The qualifier is semantic. If two modules export a record with the same short
name, I keep their declarations distinct; matching field layouts do not make
them interchangeable. Long qualified names are preserved rather than silently
truncated. Within the declaring module, the unqualified local name remains the
ordinary spelling.

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
order across package, common, and platform C flags. Paired forms can occupy
one fragment or adjacent entries in the same flag group; I join literal
fragments before selecting compiler phases, regardless of argument-list size.
Quote paths containing spaces; use `-Xassembler` for paths containing commas.
I do not rebase `-I` operands forwarded to the assembler or linker as C header
paths. I keep these arguments out of separate C preprocessing and link-only
jobs. Integrated Clang capture
preserves the native driver's C-header search order as well as assembler
lookup. This does not admit arbitrary `-Wa` options or extend the supported
compiler/assembler versions.
For supported GNU assemblers, I also capture alternate-macro inputs selected
by `-Wa,--alternate` or `-Xassembler --alternate`, including combinations with
assembler include paths. I keep this grammar selector in assembler phases,
not separate C preprocessing or link-only jobs. Apple Clang rejects this GNU
option; admitting its spelling does not add support to that backend.
With GCC or Clang's explicit external-assembler mode, I also retain `.s` and
`.S` sources alongside C sources, including shared-only assembler inputs.
I copy raw `.s` bytes without preprocessing where the driver treats them as
raw. Apple Clang preprocesses lowercase `.s` too; I preserve that default.
I preprocess `.S` as assembler, then apply the selected assembler's capture
path. Raw roots get explicit
dependency records; nested assembler reads remain part of capture evidence.
Integrated-Clang assembler translation-unit capture remains unfinished.
Source and flag modes outside this path retain their existing compatibility
behavior; they have no retained-input guarantee.
