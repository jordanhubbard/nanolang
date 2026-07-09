# My Examples

I keep 257 `.nano` files here. They are not all the same kind of thing.

Start with `docs/EXAMPLES_INDEX.md`. It is my source of truth for the examples catalog, the learning order, module coverage, dependency labels, and known gaps.

## Tracks

| Track | Use it for |
| --- | --- |
| Learn | Small deterministic programs that teach syntax and semantics. |
| Modules | Focused demonstrations of one module or integration surface. |
| Showcases | Larger programs that prove larger designs work. |
| Internal | Shared helper code, generated output, runners, and fixtures. |
| Experimental | Concepts, expected failures, and unfinished boundaries. |

## Start Here

Use this first path:

1. `language/nl_hello.nano`
2. `language/nl_operators.nano`
3. `language/nl_comparisons.nano`
4. `language/nl_logical.nano`
5. `language/nl_types.nano`
6. `language/nl_mutable.nano`
7. `language/nl_functions_basic.nano`
8. `language/nl_factorial.nano`
9. `language/nl_control_if_while.nano`
10. `language/nl_control_for.nano`
11. `language/nl_array_complete.nano`
12. `language/nl_struct.nano`
13. `language/nl_enum.nano`
14. `language/nl_types_union_construct.nano`
15. `advanced/testing_strategies.nano`

After that, use `docs/EXAMPLES_INDEX.md` for the array, function, algorithm, REPL, module, and showcase progressions.

## Commands

```bash
# Build the compiled example set chosen by examples/Makefile.
make -C examples

# Compile one local example.
./bin/nanoc examples/language/nl_hello.nano -o /tmp/nl_hello
/tmp/nl_hello

# Inspect parsed metadata for one example.
./bin/nanoc examples/run_examples.nano -o /tmp/run_examples
/tmp/run_examples --meta examples/language/nl_hello.nano
```

Some examples need SDL, OpenGL, Bullet, MuJoCo, ncurses, curl, GPU drivers, local servers, or API keys. I do not hide that. The catalog marks those boundaries.
