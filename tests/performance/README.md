# Performance Tests

Benchmarks and performance validation tests.

## Categories

- `compilation_speed/` - Compiler performance
- `runtime_speed/` - Generated code performance  
- `memory_usage/` - Memory consumption tests
- `stress/` - Large programs and edge cases

## Running

```bash
make test-performance
```

## Benchmarks

Benchmarks track:
- Compilation time
- Lines of code per second
- Memory usage
- Generated binary size
- Runtime performance vs C baseline

Results are tracked over time to catch performance regressions.

