# Case Study: My LLM-Guided Optimization of Prime Counting

**Date:** January 31, 2026
**Tool:** My Profiling + LLM Analysis
**Result:** 5.6x performance improvement

## Summary

I use this case study to show my LLM-powered profiling workflow:
1. I profile existing code.
2. My LLM analyzes the hotspots.
3. I receive algorithmic improvements.
4. I implement and verify them.

## My Original Implementation (Trial Division)

```nano
fn is_prime(n: int) -> bool {
    if (< n 2) { return false }
    if (== n 2) { return true }
    if (== (% n 2) 0) { return false }

    let mut i: int = 3
    while (<= (* i i) n) {
        if (== (% n i) 0) {
            return false
        }
        set i (+ i 2)
    }
    return true
}

fn count_primes(limit: int) -> int {
    let mut count: int = 0
    let mut i: int = 2
    while (< i limit) {
        if (is_prime i) {
            set count (+ count 1)
        }
        set i (+ i 1)
    }
    return count
}
```

**Complexity:** O(n√n) for counting primes up to n

## My Profiling Data (Original)

```json
{
  "binary": "/tmp/primes_heavy",
  "hotspots": [
    {"function": "<Total>", "samples": 1000, "pct_time": 100.0},
    {"function": "nl_is_prime", "samples": 979, "pct_time": 98.0},
    {"function": "<static>@0x1367a8", "samples": 20, "pct_time": 2.0}
  ]
}
```

**Key Finding:** I spent 98% of my time in the `nl_is_prime` function.

**Performance:**
- Task: Count primes up to 1,000,000
- Runtime: **490ms**
- Result: 78,498 primes

## My LLM Analysis

The profiling data showed me:

1. **Bottleneck:** A single function, `is_prime`, dominated my runtime at 98%.
2. **Root Cause:** My trial division algorithm required O(√n) work per number.
3. **Total Complexity:** O(n√n), which is about 1 billion operations for n=1,000,000.
4. **Micro-optimizations:** These would provide less than 2% improvement. I do not waste effort on them.
5. **Solution:** I required an algorithmic change. I chose the Sieve of Eratosthenes.

## My Optimized Implementation (Sieve of Eratosthenes)

```nano
fn count_primes_sieve(limit: int) -> int {
    if (< limit 2) {
        return 0
    }

    # Allocate boolean array for the sieve
    let mut sieve: array<bool> = (array_new limit true)

    # 0 and 1 are not prime
    (array_set sieve 0 false)
    (array_set sieve 1 false)

    # Sieve of Eratosthenes algorithm
    let mut p: int = 2
    while (< (* p p) limit) {
        if (at sieve p) {
            # Mark all multiples of p as not prime
            let mut i: int = (* p p)
            while (< i limit) {
                (array_set sieve i false)
                set i (+ i p)
            }
        }
        set p (+ p 1)
    }

    # Count primes
    let mut count: int = 0
    let mut idx: int = 0
    while (< idx limit) {
        if (at sieve idx) {
            set count (+ count 1)
        }
        set idx (+ idx 1)
    }
    return count
}
```

**Complexity:** O(n log log n), which is about 13 million operations for n=1,000,000.

## My Profiling Data (Optimized)

```json
{
  "binary": "/tmp/primes_opt_prof",
  "hotspots": [
    {"function": "<Total>", "samples": 1000, "pct_time": 100.0},
    {"function": "<static>@0x1367a8", "samples": 705, "pct_time": 70.6},
    {"function": "mcount", "samples": 117, "pct_time": 11.8},
    {"function": "nl_array_set_bool", "samples": 117, "pct_time": 11.8},
    {"function": "dyn_array_push_bool", "samples": 58, "pct_time": 5.9}
  ]
}
```

**Key Finding:** No single hotspot dominates. I distributed the work across my array operations.

**Performance:**
- Task: Count primes up to 1,000,000
- Runtime: **88ms**
- Result: 78,498 primes. I verified this is correct.

## My Results

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Runtime | 490ms | 88ms | **5.6x faster** |
| Algorithm | Trial Division | Sieve of Eratosthenes | |
| Complexity | O(n√n) | O(n log log n) | |
| Operations | ~1 billion | ~13 million | 75x fewer |
| Hotspot % | 98% in one function | Distributed | Better balanced |
| Memory | O(1) | O(n) | Trade-off |

## Lessons I Learned

### 1. Profiling Reveals My Algorithm Choice

Without profiling, you might spend time:
- Micro-optimizing my trial division loop.
- Trying to parallelize my individual primality tests.
- Caching my results, which remains O(n√n) work.

Profiling showed me that my algorithm was the problem.

### 2. My 98% Rule

When one of my functions uses more than 90% of runtime:
- I focus on an algorithmic change.
- I do not micro-optimize, as I would gain less than 10%.

### 3. Memory-Speed Trade-off

My sieve uses 1MB RAM for 1,000,000 booleans, but I gain a 5.6x speedup.
I find this a good trade-off for:
- Batch processing.
- Pre-computation.
- When my speed matters more than my memory usage.

### 4. My LLM-Guided Workflow

My profiling, LLM, and optimization cycle follows these steps:
1. **Automated profiling**: I compile with `-pg` to auto-profile.
2. **JSON output**: I produce structured data for my LLM to analyze.
3. **LLM suggests algorithm**: My LLM recognizes the pattern and recommends a sieve.
4. **Verification**: My re-profile shows a healthy, distributed workload.

## Implementation Notes

### My Array Operations

These are the patterns I use:
```nano
# Create mutable array
let mut arr: array<bool> = (array_new size initial_value)

# Read array element
let val: bool = (at arr index)

# Write array element
(array_set arr index value)
```

### My Profiling Overhead

When I use the `-pg` flag:
- Original: 490ms became 686ms, a 40% overhead.
- Optimized: 88ms. I did not measure this with profiling.

I always benchmark myself without profiling to get accurate measurements.

## Conclusion

I have shown the power of my profiling-guided optimization:
- **Before:** I was guessing what was slow.
- **After:** I knew exactly what was slow.
- **Result:** I achieved a 5.6x speedup from one algorithmic change.

My LLM workflow makes this accessible. My profiling data guides the LLM to the right approach.

## My Next Steps

I could optimize further:
1. **Segmented Sieve:** I would use this for limits over 1 billion to reduce my memory usage.
2. **Wheel Factorization:** I could skip multiples of 2, 3, and 5 for a minor speedup.
3. **Parallelization:** I could segment my sieve across cores.

My profiling showed these are premature. 88ms is fast enough for most of my use cases.

---

**See also:**
- [userguide/08_profiling.md](../userguide/08_profiling.md) ,oldString:
