# Chapter 20: Testing & Quality

**Property-based testing and coverage analysis.**

Testing modules for code quality assurance.

## 20.1 Property-Based Testing

```nano
from "modules/proptest/proptest.nano" import forall, Gen

fn test_commutative() -> bool {
    return (forall Gen.int Gen.int (fn (a: int, b: int) -> bool {
        return (== (+ a b) (+ b a))
    }))
}

shadow test_commutative {
    assert (test_commutative)
}
```

## 20.2 Coverage Analysis

```nano
from "stdlib/coverage.nano" import coverage_init, coverage_record, coverage_report

fn tracked_function(x: int) -> int {
    (coverage_record "test.nano" 10 0)
    if (> x 0) {
        (coverage_record "test.nano" 11 0)
        return x
    }
    (coverage_record "test.nano" 14 0)
    return 0
}

shadow tracked_function {
    (coverage_init)
    (tracked_function 5)
    (coverage_report)
}
```

## 20.3 Performance Testing

```nano
from "stdlib/coverage.nano" import timing_start, timing_end

fn benchmark(n: int) -> int {
    (timing_start "operation")
    let mut sum: int = 0
    for i in (range 0 n) {
        set sum (+ sum i)
    }
    (timing_end "operation")
    return sum
}

shadow benchmark {
    assert (== (benchmark 100) 4950)
}
```

## Summary

Testing tools:
- ✅ Property-based testing
- ✅ Coverage tracking
- ✅ Performance benchmarks

---

**Previous:** [Chapter 19: Terminal UI](../19_terminal_ui/index.md)  
**Next:** [Chapter 21: Configuration](../21_configuration/index.md)
