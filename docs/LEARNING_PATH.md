# My Learning Path

This path uses examples that exist and that I can explain without live services or hidden setup.

For the full catalog, dependency labels, and remaining coverage gaps, read `docs/EXAMPLES_INDEX.md`.

## Running One Example

```bash
./bin/nanoc examples/language/nl_hello.nano -o /tmp/nl_hello
/tmp/nl_hello
```

## Level 1: First Programs

Goal: understand my program shape, calls, operators, variables, and shadow tests.

1. `examples/language/nl_hello.nano`
2. `examples/language/nl_operators.nano`
3. `examples/language/nl_comparisons.nano`
4. `examples/language/nl_logical.nano`
5. `examples/language/nl_types.nano`
6. `examples/language/nl_mutable.nano`
7. `examples/language/nl_functions_basic.nano`
8. `examples/advanced/testing_strategies.nano`

I require a shadow test for every function I compile. Read those blocks as executable specifications, not decoration.

```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
    assert (== (double 0) 0)
    assert (== (double -3) -6)
}
```

## Level 2: Control Flow and Data

Goal: use branches, loops, arrays, strings, and records.

1. `examples/language/nl_control_if_while.nano`
2. `examples/language/nl_control_for.nano`
3. `examples/language/nl_array_complete.nano`
4. `examples/language/nl_array_bounds.nano`
5. `examples/language/nl_for_in_array.nano`
6. `examples/language/nl_string_operations.nano`
7. `examples/language/nl_struct.nano`
8. `examples/language/nl_enum.nano`

## Level 3: Composition

Goal: write reusable functions, model error cases, and use modules.

1. `examples/language/nl_function_return_values.nano`
2. `examples/language/nl_function_variables.nano`
3. `examples/language/nl_types_tuple.nano`
4. `examples/language/nl_union_types.nano`
5. `examples/language/nl_types_union_construct.nano`
6. `examples/language/nl_result_propagation.nano`
7. `examples/language/nl_generics_demo.nano`
8. `examples/language/nl_hashmap.nano`
9. `examples/language/nl_demo_selfhosting.nano`
10. `examples/hello_pkg/main.nano`
11. `examples/large_project/main.nano`

## Level 4: Safety and Tooling

Goal: understand where I draw boundaries.

1. `examples/language/nl_affine_resource_demo.nano`
2. `examples/language/nl_unsafe_demo.nano`
3. `examples/debug/logging_demo.nano`
4. `examples/debug/coverage_demo.nano`
5. `examples/debug/property_test_math.nano`
6. `examples/debug/property_test_sorting.nano`
7. `examples/diagnostics/llm_diags_type_mismatch.nano`
8. `examples/cross_backend/hello_cross_backend.nano`
9. `examples/integration/file_pipeline.nano`

`llm_diags_type_mismatch.nano` is an expected-failure example. It exists to inspect structured diagnostics, not to compile cleanly.

## Level 5: Larger Programs

Goal: study complete designs after the core syntax is familiar.

1. `examples/opl/`
2. `examples/language/nl_forth_interpreter.nano`
3. `examples/games/sdl_checkers.nano`
4. `examples/games/sdl_asteroids.nano`
5. `examples/audio/sdl_nanoamp.nano`
6. `examples/graphics/sdl_forth_ide.nano`
7. `examples/playground/playground_server.nano`

These examples may require external libraries, graphics, audio, local servers, or generated artifacts. Check the index before treating one as a portable smoke test.
