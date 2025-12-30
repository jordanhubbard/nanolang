# Property-Based Testing Framework for NanoLang

**Status:** Design & Implementation Phase  
**Priority:** P2  
**Bead:** nanolang-2xry  
**Created:** 2025-12-29

## Goal

Build a QuickCheck-style property-based testing framework to find edge cases that example-based tests miss.

**Inspiration:** QuickCheck (Haskell), Hypothesis (Python), PropEr (Erlang), ScalaCheck

## Problem: Example-Based Tests Are Limited

### Current Approach: Shadow Tests

```nano
fn divide(a: int, b: int) -> int {
    return (/ a b)
}

shadow divide {
    assert (== (divide 10 2) 5)
    assert (== (divide 7 3) 2)
    assert (== (divide 0 5) 0)
}
```

**Limitations:**
- Only tests 3 specific inputs
- Might miss edge cases (MAX_INT, MIN_INT, etc.)
- No systematic exploration of input space
- Requires manually thinking of test cases

## Solution: Property-Based Testing

### Core Idea

Instead of specific examples, define **properties** that must hold for **all** inputs:

```nano
import "modules/std/proptest.nano" as PT

property divide_reconstruction {
    forall a: int, b: int where (!= b 0) {
        let quotient: int = (divide a b)
        let remainder: int = (% a b)
        let reconstructed: int = (+ (* quotient b) remainder)
        assert (== reconstructed a)
    }
}
```

The framework:
1. **Generates** 1000+ random test cases
2. **Runs** property check on each
3. **Shrinks** failures to minimal counterexamples
4. **Reports** the simplest failing input

## Architecture

### Core Types

```nano
/* modules/std/proptest.nano */

/* Generator produces random values of type T */
struct Generator<T> {
    generate: fn(RNG) -> T,
    shrink: fn(T) -> List<T>  /* Shrinking strategy */
}

/* Property is a test with generated inputs */
struct Property {
    name: string,
    test: fn() -> TestResult,
    max_tests: int,          /* Default: 1000 */
    max_shrink_depth: int    /* Default: 100 */
}

/* Test result */
union TestResult {
    Pass,
    Fail(string),          /* Error message */
    Discard(string)        /* Precondition failed, try another input */
}

/* Random number generator */
struct RNG {
    state: int,
    seed: int
}
```

### Built-in Generators

```nano
/* Integer generators */
fn gen_int() -> Generator<int>
fn gen_int_range(min: int, max: int) -> Generator<int>
fn gen_positive_int() -> Generator<int>
fn gen_negative_int() -> Generator<int>

/* Float generators */
fn gen_float() -> Generator<float>
fn gen_float_range(min: float, max: float) -> Generator<float>

/* Boolean generator */
fn gen_bool() -> Generator<bool>

/* String generators */
fn gen_string(max_len: int) -> Generator<string>
fn gen_ascii_string(max_len: int) -> Generator<string>
fn gen_alphanumeric_string(max_len: int) -> Generator<string>

/* Array generators */
fn gen_array<T>(element_gen: Generator<T>, max_len: int) -> Generator<array<T>>

/* Choice generators */
fn gen_choose<T>(generators: List<Generator<T>>) -> Generator<T>
fn gen_one_of<T>(values: array<T>) -> Generator<T>

/* Tuple generators */
fn gen_tuple2<A,B>(gen_a: Generator<A>, gen_b: Generator<B>) -> Generator<(A, B)>
fn gen_tuple3<A,B,C>(gen_a: Generator<A>, gen_b: Generator<B>, gen_c: Generator<C>) 
    -> Generator<(A, B, C)>
```

### Combinators

```nano
/* Map over generator */
fn gen_map<A,B>(gen: Generator<A>, f: fn(A) -> B) -> Generator<B>

/* Filter generator (with max attempts) */
fn gen_filter<T>(gen: Generator<T>, predicate: fn(T) -> bool) -> Generator<T>

/* Bind for dependent generators */
fn gen_bind<A,B>(gen: Generator<A>, f: fn(A) -> Generator<B>) -> Generator<B>
```

## Example Usage

### Example 1: List Reverse Property

```nano
import "modules/std/proptest.nano" as PT

fn list_reverse<T>(list: List<T>) -> List<T> {
    let mut result: List<T> = (list_new)
    let mut i: int = (- (list_length list) 1)
    while (>= i 0) {
        let item: T = (list_get list i)
        (list_push result item)
        set i (- i 1)
    }
    return result
}

property reverse_twice_is_identity {
    let gen: Generator<List<int>> = (PT.gen_array (PT.gen_int) 20)
    
    (PT.forall gen (fn(list: List<int>) -> TestResult {
        let reversed_once: List<int> = (list_reverse list)
        let reversed_twice: List<int> = (list_reverse reversed_once)
        
        if (list_equal list reversed_twice) {
            return Pass
        } else {
            return (Fail "reverse(reverse(x)) != x")
        }
    }))
}
```

### Example 2: Sort Properties

```nano
fn list_sort(list: List<int>) -> List<int> {
    /* Quicksort implementation */
    /* ... */
}

property sort_is_ordered {
    let gen: Generator<List<int>> = (PT.gen_array (PT.gen_int) 50)
    
    (PT.forall gen (fn(list: List<int>) -> TestResult {
        let sorted: List<int> = (list_sort list)
        
        /* Check all adjacent pairs are in order */
        let mut i: int = 0
        while (< i (- (list_length sorted) 1)) {
            let a: int = (list_get sorted i)
            let b: int = (list_get sorted (+ i 1))
            if (> a b) {
                return (Fail "Found out-of-order pair")
            }
            set i (+ i 1)
        }
        return Pass
    }))
}

property sort_preserves_length {
    let gen: Generator<List<int>> = (PT.gen_array (PT.gen_int) 50)
    
    (PT.forall gen (fn(list: List<int>) -> TestResult {
        let sorted: List<int> = (list_sort list)
        if (== (list_length list) (list_length sorted)) {
            return Pass
        } else {
            return (Fail "Sort changed list length")
        }
    }))
}

property sort_preserves_elements {
    let gen: Generator<List<int>> = (PT.gen_array (PT.gen_int_range 0 10) 20)
    
    (PT.forall gen (fn(list: List<int>) -> TestResult {
        let sorted: List<int> = (list_sort list)
        
        /* Check that sorted contains same elements (with same counts) */
        let freq_original: Map<int,int> = (count_frequencies list)
        let freq_sorted: Map<int,int> = (count_frequencies sorted)
        
        if (map_equal freq_original freq_sorted) {
            return Pass
        } else {
            return (Fail "Sort changed element counts")
        }
    }))
}
```

### Example 3: Parser Round-Trip Property

```nano
property parse_print_roundtrip {
    let gen: Generator<AST> = (gen_random_ast 10)  /* Max depth 10 */
    
    (PT.forall gen (fn(ast: AST) -> TestResult {
        let printed: string = (ast_to_string ast)
        let parsed: Result<AST, string> = (parse_string printed)
        
        match parsed {
            Ok(ast2) -> {
                if (ast_equal ast ast2) {
                    return Pass
                } else {
                    return (Fail "Parsed AST differs from original")
                }
            },
            Err(msg) -> {
                return (Fail (++ "Parse failed: " msg))
            }
        }
    }))
}
```

## Shrinking Strategy

When a property fails, shrink the input to find the **minimal counterexample**:

### Shrinking Integers

```nano
fn shrink_int(n: int) -> List<int> {
    if (== n 0) {
        return (list_new)  /* Can't shrink 0 */
    }
    
    let mut shrinks: List<int> = (list_new)
    
    /* Try 0 first (smallest value) */
    (list_push shrinks 0)
    
    /* Try halving */
    let half: int = (/ n 2)
    if (!= half 0) {
        (list_push shrinks half)
        (list_push shrinks (- 0 half))  /* Negative half */
    }
    
    /* Try n-1 (slightly smaller) */
    if (> n 0) {
        (list_push shrinks (- n 1))
    } else {
        (list_push shrinks (+ n 1))
    }
    
    return shrinks
}
```

### Shrinking Lists

```nano
fn shrink_list<T>(list: List<T>, shrink_element: fn(T) -> List<T>) -> List<List<T>> {
    let mut shrinks: List<List<T>> = (list_new)
    
    /* Empty list (smallest) */
    (list_push shrinks (list_new))
    
    /* Remove first element */
    if (> (list_length list) 0) {
        (list_push shrinks (list_tail list))
    }
    
    /* Remove last element */
    if (> (list_length list) 0) {
        (list_push shrinks (list_init list))
    }
    
    /* Remove middle element */
    if (> (list_length list) 2) {
        let mid: int = (/ (list_length list) 2)
        (list_push shrinks (list_remove list mid))
    }
    
    /* Shrink individual elements */
    let mut i: int = 0
    while (< i (list_length list)) {
        let elem: T = (list_get list i)
        let elem_shrinks: List<T> = (shrink_element elem)
        
        let mut j: int = 0
        while (< j (list_length elem_shrinks)) {
            let smaller_elem: T = (list_get elem_shrinks j)
            let shrunk_list: List<T> = (list_set list i smaller_elem)
            (list_push shrinks shrunk_list)
            set j (+ j 1)
        }
        
        set i (+ i 1)
    }
    
    return shrinks
}
```

### Shrinking Algorithm

```nano
fn shrink_until_minimal<T>(
    failing_input: T,
    property_test: fn(T) -> TestResult,
    shrink: fn(T) -> List<T>,
    max_depth: int
) -> T {
    let mut current: T = failing_input
    let mut depth: int = 0
    
    while (< depth max_depth) {
        let candidates: List<T> = (shrink current)
        let mut found_smaller: bool = false
        
        /* Try each shrink candidate */
        let mut i: int = 0
        while (< i (list_length candidates)) {
            let candidate: T = (list_get candidates i)
            let result: TestResult = (property_test candidate)
            
            match result {
                Fail(_) -> {
                    /* This smaller input also fails! */
                    set current candidate
                    set found_smaller true
                    break
                },
                _ -> {
                    /* This candidate passes, keep looking */
                }
            }
            
            set i (+ i 1)
        }
        
        if (not found_smaller) {
            /* No smaller failing input found */
            break
        }
        
        set depth (+ depth 1)
    }
    
    return current
}
```

## Random Number Generation

Simple linear congruential generator (LCG):

```nano
struct RNG {
    state: int
}

fn rng_new(seed: int) -> RNG {
    return (RNG seed)
}

fn rng_next(rng: RNG) -> (int, RNG) {
    /* LCG parameters (same as glibc) */
    let a: int = 1103515245
    let c: int = 12345
    let m: int = 2147483648  /* 2^31 */
    
    let next_state: int = (% (+ (* rng.state a) c) m)
    return (next_state, (RNG next_state))
}

fn rng_int_range(rng: RNG, min: int, max: int) -> (int, RNG) {
    let (rand, rng2) = (rng_next rng)
    let range: int = (- max min)
    let value: int = (+ min (% rand range))
    return (value, rng2)
}
```

## Property Syntax (Language Extension)

Add `property` keyword as sugar over `shadow`:

```nano
property my_property {
    forall x: int, y: int {
        /* test body */
        assert (property_holds x y)
    }
}
```

Desugars to:

```nano
shadow test_my_property {
    let gen_x: Generator<int> = (PT.gen_int)
    let gen_y: Generator<int> = (PT.gen_int)
    
    (PT.forall2 gen_x gen_y (fn(x: int, y: int) -> TestResult {
        if (property_holds x y) {
            return Pass
        } else {
            return (Fail "Property failed")
        }
    }))
}
```

## Implementation Plan

### Phase 1: Core Framework (Week 1-2)

**1.1 RNG Implementation**
- Linear congruential generator
- Seeded from time or env variable
- `NANOLANG_SEED=42 make test` for reproducibility

**1.2 Generator Type**
- Struct with generate + shrink functions
- Built-in generators for primitives

**1.3 Property Runner**
- Run property N times (default 1000)
- Collect failures
- Report statistics

**Files:**
- `modules/std/proptest.nano`
- `modules/std/proptest/rng.nano`
- `modules/std/proptest/generators.nano`

### Phase 2: Shrinking (Week 3)

**2.1 Shrinking Strategies**
- Integer shrinking (toward 0)
- List shrinking (remove elements)
- String shrinking (remove chars)

**2.2 Shrinking Algorithm**
- Breadth-first search for minimal failing input
- Configurable max depth

**Files:**
- `modules/std/proptest/shrink.nano`

### Phase 3: Advanced Generators (Week 4)

**3.1 Combinators**
- `gen_map`, `gen_filter`, `gen_bind`
- `gen_tuple2`, `gen_tuple3`

**3.2 Recursive Generators**
- `gen_tree`, `gen_ast`
- Bounded depth to avoid infinite generation

**Files:**
- `modules/std/proptest/combinators.nano`

### Phase 4: Compiler Integration (Week 5)

**4.1 Add `property` Keyword**
- Lexer: `TOKEN_PROPERTY`
- Parser: recognize property blocks
- Transpiler: generate proptest calls

**4.2 Test Runner Integration**
- `make test-properties`
- Run all property tests
- Report statistics

**Files to modify:**
- `src/lexer.c`
- `src/parser.c`
- `src/transpiler.c`
- `Makefile`

### Phase 5: Documentation & Examples (Week 6)

**5.1 User Guide**
- Writing properties
- Custom generators
- Interpreting failures

**5.2 Example Properties**
- Stdlib functions (sort, reverse, etc.)
- Parser round-trips
- Math properties (commutativity, associativity)

**Files:**
- `docs/PROPERTY_BASED_TESTING.md`
- `examples/proptest_*.nano`

## Use Cases

### 1. Compiler Testing

```nano
property parser_preserves_semantics {
    let gen: Generator<string> = (gen_valid_nanolang_program)
    
    (PT.forall gen (fn(source: string) -> TestResult {
        let ast1: AST = (parse source)
        let printed: string = (ast_to_string ast1)
        let ast2: AST = (parse printed)
        
        if (ast_equal ast1 ast2) {
            return Pass
        } else {
            return (Fail "AST changed after print/parse")
        }
    }))
}
```

### 2. Stdlib Testing

```nano
property string_concat_associative {
    let gen: Generator<string> = (PT.gen_string 20)
    
    (PT.forall3 gen gen gen (fn(a: string, b: string, c: string) -> TestResult {
        let left: string = (++ (++ a b) c)
        let right: string = (++ a (++ b c))
        
        if (== left right) {
            return Pass
        } else {
            return (Fail "String concatenation not associative")
        }
    }))
}
```

### 3. Differential Testing

```nano
property c_compiler_matches_interpreter {
    let gen: Generator<string> = (gen_simple_program)
    
    (PT.forall gen (fn(source: string) -> TestResult {
        let interpreted: string = (run_interpreter source)
        let compiled: string = (compile_and_run source)
        
        if (== interpreted compiled) {
            return Pass
        } else {
            return (Fail "Interpreter output != compiled output")
        }
    }))
}
```

## Command Line Interface

```bash
# Run all properties (default 1000 tests each)
make test-properties

# Run with custom test count
PROPTEST_COUNT=10000 make test-properties

# Run with specific seed (reproducible)
PROPTEST_SEED=42 make test-properties

# Verbose output
PROPTEST_VERBOSE=1 make test-properties

# Run specific property
./bin/nano tests/test_sort_properties.nano --property sort_is_ordered
```

## Output Format

```
Testing property: sort_is_ordered
  Generating 1000 test cases...
  ✓ All tests passed

Testing property: divide_reconstruction
  Generating 1000 test cases...
  ✗ Failed after 347 tests
  
  Counterexample (after shrinking):
    a = 0
    b = 1
  
  Property: divide_reconstruction
  Error: Reconstruction failed
  
  Reproduce with: PROPTEST_SEED=1735516800

Summary:
  ✓ 2 properties passed
  ✗ 1 property failed
```

## Performance Considerations

- **Generation speed**: Use fast RNG (LCG is fine)
- **Shrinking limits**: Cap at 100 iterations to avoid hangs
- **Parallel execution**: Run properties in parallel (future)
- **Caching**: Cache generated values between runs (future)

## Timeline

- **Week 1-2:** Core framework (RNG, generators, runner)
- **Week 3:** Shrinking implementation
- **Week 4:** Advanced generators and combinators
- **Week 5:** Compiler integration (`property` keyword)
- **Week 6:** Documentation and examples

**Total:** 6 weeks

## Success Criteria

1. ✅ Can define properties with custom generators
2. ✅ Runs 1000+ test cases per property
3. ✅ Shrinks failures to minimal counterexamples
4. ✅ Integrated with `make test`
5. ✅ Reproducible with seeds
6. ✅ Documentation with 10+ examples
7. ✅ Used to test stdlib functions

## References

- **QuickCheck (Haskell)**: https://hackage.haskell.org/package/QuickCheck
- **Hypothesis (Python)**: https://hypothesis.readthedocs.io/
- **PropEr (Erlang)**: https://proper-testing.github.io/
- **ScalaCheck**: https://scalacheck.org/
- **fast-check (JavaScript)**: https://github.com/dubzzz/fast-check

---

**Next Steps:**
1. Review design
2. Implement Phase 1 (core framework)
3. Write example properties for stdlib
4. Integrate with CI/CD

