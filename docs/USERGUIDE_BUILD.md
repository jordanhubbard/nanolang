# My User Guide Build

I publish the Markdown sources selected by `userguide/nav.txt`. That manifest is
the navigation order and the complete list of public pages. Old files under
`userguide/` remain historical input until they are removed; they are not
published merely because they exist.

## Build and Check

```bash
make userguide-check
make userguide-html
make -C userguide serve
```

I write the site to `build/userguide/html/`. The build starts from an empty
output directory, generates reference pages, renders Markdown, then validates
every local link and fragment.

## Generated Reference

`scripts/build_userguide.py` generates these pages during each build:

| Page | Source |
| --- | --- |
| Builtins | `docs/STDLIB.md`, checked against `src/builtins_registry.c` |
| Examples | Every `.nano` file under `examples/` and its metadata header |
| Modules | `modules/`, `module.json`, `module.manifest.json`, and declarations |
| Compiler CLI | `bin/nanoc_c --help` |

Generated Markdown lives under `build/userguide/generated/`; it is a build
artifact, not a second hand-maintained reference.

## Snippets

Executable examples use an `nl-snippet` marker immediately before a NanoLang
fence. `make userguide-check` compiles marked snippets and runs those with
`run:true`. Unmarked blocks may be excerpts or conceptual fragments. They are
not claimed as independently runnable programs.

## Publication

`.github/workflows/userguide_pages.yml` builds the same site on relevant pull
requests. Pushes to `main` additionally upload and deploy the validated Pages
artifact. The workflow does not edit tracked API Markdown while publishing.

## Accessibility

The generated pages provide semantic landmarks, a language attribute, skip
navigation, keyboard focus, stable heading anchors, responsive navigation, and
overflow-safe tables. These properties are tested structurally by the build.
WCAG conformance still requires an accessibility audit; I do not claim one has
occurred.
