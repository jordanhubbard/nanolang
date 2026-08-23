# Getting Started

I currently build on Unix-like systems. Windows users should use WSL2.

## Build

You need a C compiler, Make, Git, and pkg-config. Clone and build:

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make build
./bin/nanoc --version
```

`bin/nanoc` is my compiler. `bin/nano` is my tree-walking interpreter.

## First Program

Create `hello.nano`:

<!--nl-snippet {"name":"refresh_getting_started_hello","check":true,"expect_stdout":"Hello, World!\n"}-->
```nano
fn main() -> int {
    (println "Hello, World!")
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

Compile and run it:

```bash
./bin/nanoc hello.nano -o hello
./hello
```

I transpile this program to C and invoke the host C compiler. Shadows execute while I compile; the resulting program then runs its generated shadow harness as part of startup. A failed shadow stops the process.

## Interpreter

Run a source file without producing a native executable:

```bash
./bin/nano hello.nano
```

The interpreter and compiled backend share the language, but they do not have identical implementation boundaries. Test the backend you intend to ship.

## Project Layout

For a small program, one `.nano` file is enough. Packages can use `nano.toml`:

```text
my_program/
  nano.toml
  main.nano
  src/
```

See [`examples/hello_pkg`](https://github.com/jordanhubbard/nanolang/tree/main/examples/hello_pkg) and [`examples/large_project`](https://github.com/jordanhubbard/nanolang/tree/main/examples/large_project) for complete layouts.

## Next

Read [Language](02_language.md) for calls, operators, bindings, functions, and control flow.
