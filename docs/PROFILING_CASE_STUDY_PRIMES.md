# Case Study: LLM-Guided Optimization of Prime Counting

**Date:** January 31, 2026
**Tool:** NanoLang Profiling + LLM Analysis
**Result:** 5.6x performance improvement

## Summary

This case study demonstrates the complete LLM-powered profiling workflow:
1. Profile existing code
2. LLM analyzes hotspots
3. Recommend algorithmic improvements
4. Implement and verify

## Original Implementation (Trial Division)

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

## Profiling Data (Original)

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

**Key Finding:** 98% of time spent in `nl_is_prime` function

**Performance:**
- Task: Count primes up to 1,000,000
- Runtime: **490ms**
- Result: 78,498 primes

## LLM Analysis

The profiling data immediately revealed:

1. **Bottleneck:** Single function (`is_prime`) dominates runtime at 98%
2. **Root Cause:** Trial division algorithm requires O(√n) work per number
3. **Total Complexity:** O(n√n) ≈ 1 billion operations for n=1,000,000
4. **Micro-optimizations:** Would provide < 2% improvement (wasted effort)
5. **Solution:** Algorithmic change required - use Sieve of Eratosthenes

## Optimized Implementation (Sieve of Eratosthenes)

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

**Complexity:** O(n log log n) ≈ 13 million operations for n=1,000,000

## Profiling Data (Optimized)

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

**Key Finding:** No single hotspot dominates (work distributed across array operations)

**Performance:**
- Task: Count primes up to 1,000,000
- Runtime: **88ms**
- Result: 78,498 primes (verified correct)

## Results

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Runtime | 490ms | 88ms | **5.6x faster** |
| Algorithm | Trial Division | Sieve of Eratosthenes | |
| Complexity | O(n√n) | O(n log log n) | |
| Operations | ~1 billion | ~13 million | 75x fewer |
| Hotspot % | 98% in one function | Distributed | Better balanced |
| Memory | O(1) | O(n) | Trade-off |

## Lessons Learned

### 1. **Profiling Reveals Algorithm Choice**

Without profiling, you might spend time:
- Micro-optimizing the trial division loop
- Trying to parallelize individual primality tests
- Caching results (still O(n√n) work)

Profiling immediately showed: **algorithm itself is the problem**

### 2. **98% Rule**

When one function uses >90% of runtime:
- ✅ Focus on algorithmic change
- ❌ Don't micro-optimize (< 10% possible gain)

### 3. **Memory-Speed Trade-off**

The sieve uses 1MB RAM (1,000,000 bools) but gains 5.6x speedup.
This is usually a good trade-off for:
- Batch processing
- Pre-computation
- When speed matters more than memory

### 4. **LLM-Guided Workflow Works**

The profiling → LLM → optimization cycle:
1. **Automated profiling**: Programs compiled with `-pg` auto-profile
2. **JSON output**: Structured data perfect for LLM analysis
3. **LLM suggests algorithm**: Recognizes the pattern and recommends sieve
4. **Verification**: Re-profile shows distributed workload (healthy)

## Implementation Notes

### Array Operations in NanoLang

Key syntax learned:
```nano
# Create mutable array
let mut arr: array<bool> = (array_new size initial_value)

# Read array element
let val: bool = (at arr index)

# Write array element (use array_set, NOT set arr[i] value)
(array_set arr index value)
```

### Profiling Overhead

With `-pg` flag:
- Original: 490ms → 686ms (40% overhead)
- Optimized: 88ms → (not measured with profiling)

Always benchmark without profiling for accurate measurements.

## Conclusion

This demonstrates the power of profiling-guided optimization:
- **Before:** Guessing what's slow
- **After:** Knowing exactly what's slow
- **Result:** 5.6x speedup from one algorithmic change

The LLM workflow makes this accessible to developers who may not know all algorithms - the profiling data guides the LLM to suggest the right approach.

## Next Steps

Further optimizations could include:
1. **Segmented Sieve:** For limits > 1 billion (reduces memory)
2. **Wheel Factorization:** Skip multiples of 2, 3, 5 (minor speedup)
3. **Parallelization:** Segment the sieve across cores

But profiling showed these are premature - 88ms is already fast enough for most use cases.

---

**See also:**
- [userguide/08_profiling.md](../userguide/part2_features/08_profiling.md) - Full profiling guide
- [docs/PROFILING_ON_UBUNTU.md](PROFILING_ON_UBUNTU.md) - Setup instructions
- [docs/PERFORMANCE.md](PERFORMANCE.md) - General performance guidelines
