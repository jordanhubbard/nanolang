# Modules

NanoLang uses explicit imports. Module paths are strings.

## Importing a module

This example uses the built-in `std/json` module.

<!--nl-snippet {"name":"ug_modules_std_json","check":true}-->
```nano
from "modules/std/json/json.nano" import Json, parse, free, get, object_has, as_string

fn extract_name(json_text: string) -> string {
    let root: Json = (parse json_text)
    if (== root 0) {
        return ""
    }
    if (not (object_has root "name")) {
        (free root)
        return ""
    }
    let v: Json = (get root "name")
    let out: string = (as_string v)
    (free v)
    (free root)
    return out
}

shadow extract_name {
    assert (== (extract_name "{\"name\":\"nano\"}") "nano")
    assert (== (extract_name "{\"x\":1}") "")
}

fn main() -> int {
    assert (== (extract_name "{\"name\":\"NanoLang\"}") "NanoLang")
    return 0
}

shadow main { assert true }
```

## Creating a module (high-level)

At a minimum, a module consists of:

- A `module.json` (build inputs: C sources/headers and flags)
- A `module.manifest.json` (metadata for discovery: keywords/capabilities)
- NanoLang entrypoints (usually `*.nano` files)

For real modules, see the existing folders under `modules/`.

## Module catalog
Every built-in module includes a minimal, runnable MVP snippet.

Module MVP snippets live in `modules/*/mvp.md` and are appended to this page during HTML build.
