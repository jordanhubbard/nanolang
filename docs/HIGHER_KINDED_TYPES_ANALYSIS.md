# Higher-Kinded Types: Analysis & Recommendation

**Status:** Analysis & Decision  
**Priority:** P2  
**Bead:** nanolang-aik2  
**Created:** 2025-12-29

## The Question

Should NanoLang add higher-kinded types (HKT)?

**Short Answer:** Probably not. The cost outweighs the benefit for NanoLang's goals.

## What Are Higher-Kinded Types?

**Regular types** (kind `*`):
```nano
int          /* kind: * */
string       /* kind: * */
List<int>    /* kind: * */
```

**Type constructors** (kind `* -> *`):
```nano
List         /* kind: * -> * */
Option       /* kind: * -> * */
Result       /* kind: * -> * -> * */
```

**Higher-kinded types** (types parameterized by type constructors):
```nano
/* This is NOT valid NanoLang (no HKT): */
fn map<F<_>, A, B>(fa: F<A>, f: fn(A) -> B) -> F<B>

/* With HKT, this works for List, Option, Result, etc. */
```

## Problem: Code Duplication Without HKT

Currently, we must duplicate code for each container:

```nano
/* Map for List */
fn map_list<A, B>(list: List<A>, f: fn(A) -> B) -> List<B> {
    let mut result: List<B> = (list_new)
    let mut i: int = 0
    while (< i (list_length list)) {
        let item: A = (list_get list i)
        (list_push result (f item))
        set i (+ i 1)
    }
    return result
}

/* Map for Option */
fn map_option<A, B>(opt: Option<A>, f: fn(A) -> B) -> Option<B> {
    match opt {
        Some(x) -> return (Some (f x)),
        None -> return None
    }
}

/* Map for Result */
fn map_result<A, B, E>(res: Result<A, E>, f: fn(A) -> B) -> Result<B, E> {
    match res {
        Ok(x) -> return (Ok (f x)),
        Err(e) -> return (Err e)
    }
}
```

**Pain point:** Can't write generic `map` that works for all functors.

## Solution With HKT (Haskell/Scala)

```haskell
-- Haskell
class Functor f where
    fmap :: (a -> b) -> f a -> f b

instance Functor [] where
    fmap = map

instance Functor Maybe where
    fmap f Nothing = Nothing
    fmap f (Just x) = Just (f x)

-- Now works for any Functor!
fmap (+1) [1,2,3]        -- [2,3,4]
fmap (+1) (Just 5)       -- Just 6
```

```scala
// Scala
trait Functor[F[_]] {
    def map[A, B](fa: F[A])(f: A => B): F[B]
}

implicit val listFunctor: Functor[List] = ...
implicit val optionFunctor: Functor[Option] = ...

// Generic code
def increment[F[_]: Functor](fa: F[Int]): F[Int] =
    Functor[F].map(fa)(_ + 1)
```

## Cost Analysis

### Implementation Cost: **VERY HIGH**

**1. Type System Complexity**
- Add kind system (`*`, `* -> *`, `* -> * -> *`, ...)
- Kind inference
- Kind checking
- Higher-order unification

**Estimated effort:** 6 months (C implementation)

**2. Dual Implementation**
- Implement in C compiler
- Implement in NanoLang self-hosted compiler
- Keep them in sync

**Dual effort:** 12 months total

**3. Error Messages**
- HKT type errors are notoriously cryptic
- Need excellent error messages (see ERROR_MESSAGES_IMPROVEMENT.md)
- Examples from Scala/Haskell show how bad it can get

**Example error:**
```
Error: Kind mismatch
  Expected: * -> *
  Got: *
  
In expression: Functor<List<int>>
             Should be: Functor<List>
```

**4. Type Inference**
- HKT makes type inference much harder
- May require type annotations everywhere
- Conflicts with NanoLang's goal of simplicity

**5. Compilation Time**
- HKT typechecking is expensive
- Can slow down compilation significantly

## Benefit Analysis

### What Do We Gain?

**1. Generic Functor/Monad Operations**
- `map`, `flatMap`, `pure`, `ap`
- Works for List, Option, Result, etc.

**But:** We can provide these per-type without much duplication.

**2. Free Structures**
- Free monads
- Free applicatives
- Church encodings

**But:** NanoLang is not a Haskell. Do we need these abstractions?

**3. Effect Systems**
- Algebraic effects via free monads
- Monad transformers

**But:** This is advanced PL research territory.

### How Often Would We Use HKT?

**Realistic assessment:**
- Stdlib: Maybe 5-10 generic functions
- Application code: Rarely
- Example programs: Almost never

**Most NanoLang users** won't write code that needs HKT.

## Alternatives to HKT

### Alternative 1: Accept Some Duplication

```nano
/* Just provide map_list, map_option, map_result */
/* It's more code, but it's explicit and easy to understand */
```

**Pros:**
- ✅ Simple to implement
- ✅ Easy to understand
- ✅ No type system complexity

**Cons:**
- ❌ Some code duplication
- ❌ Can't write fully generic code

### Alternative 2: Code Generation / Macros

```nano
/* Hypothetical macro system */
@derive_functor
union Option<T> {
    Some(T),
    None
}

/* Automatically generates map_option */
```

**Pros:**
- ✅ Reduces duplication
- ✅ No type system changes
- ✅ Explicit (you see what's generated)

**Cons:**
- ❌ Need macro system
- ❌ Less flexible than HKT

### Alternative 3: Rust-Style GATs (Generic Associated Types)

```nano
/* Simpler than full HKT */
trait Collection {
    type Item;
    type WithItem<U>;
    
    fn map<B>(self, f: fn(Item) -> B) -> WithItem<B>;
}
```

**Pros:**
- ✅ More practical than full HKT
- ✅ Proven in Rust
- ✅ Less complex type system

**Cons:**
- ❌ Still significant complexity
- ❌ Dual implementation cost
- ❌ 6-9 months effort

### Alternative 4: Defunctionalization

Encode HKT using regular types:

```nano
/* Instead of F<A>, use defunctionalized types */
struct ListF {}
struct OptionF {}

struct App<F, A> { /* Represents F<A> */ }

trait Functor<F> {
    fn map<A, B>(fa: App<F, A>, f: fn(A) -> B) -> App<F, B>
}
```

**Pros:**
- ✅ No type system changes
- ✅ Encodes HKT without HKT

**Cons:**
- ❌ Very unergonomic
- ❌ Complex for users
- ❌ Still need trait system

## Recommendation: **DON'T IMPLEMENT HKT**

### Reasons

**1. Cost/Benefit Ratio**
- **Cost:** 12 months dual implementation + ongoing maintenance
- **Benefit:** Slightly less code duplication for advanced users

**2. Misaligned with NanoLang Goals**
- NanoLang prioritizes **simplicity** and **explicitness**
- HKT adds **complexity** and **abstraction**
- C doesn't have HKT (and NanoLang targets C)

**3. Dual Implementation Burden**
- Every feature costs 2× (C + NanoLang)
- HKT is especially complex
- Better to spend effort on practical features

**4. Limited Use Cases**
- Most NanoLang programs don't need HKT
- Even functional languages use HKT sparingly
- Overkill for systems programming

**5. Better Alternatives**
- Accept small amount of duplication
- Add macros later if needed
- Focus on practical features (see other beads)

### What to Do Instead

**Near Term:**
1. ✅ **Linear types** (P0) - prevents resource bugs
2. ✅ **Better error messages** (P1) - improves UX
3. ✅ **Property-based testing** (P2) - finds bugs

**Medium Term:**
1. ✅ **Contracts** (Phase 1 of dependent types) - catches errors
2. ✅ **Macro system** - reduces duplication
3. ✅ **More stdlib containers** - practical utility

**Long Term:**
1. ⚠️ **Refinement types** (Phase 2 of dependent types) - if contracts prove valuable
2. ⚠️ **GATs** - if HKT-like features desperately needed
3. ❌ **Full HKT** - probably never

## Comparison with Other Languages

### Haskell
- **Has HKT:** Yes (core feature)
- **Rationale:** Haskell is pure FP, HKT is essential

### Rust
- **Has HKT:** No (has GATs instead)
- **Rationale:** Practical systems language, HKT too complex

### Go
- **Has HKT:** No
- **Rationale:** Simplicity over abstraction

### Swift
- **Has HKT:** No (has protocols + PATs)
- **Rationale:** Practical mobile development

### Scala
- **Has HKT:** Yes
- **Rationale:** Academic + FP roots, targets JVM

**NanoLang is closer to Rust/Go/Swift than Haskell/Scala.**

## If We Were to Implement HKT Anyway...

### Phase 1: Kind System (6 months)

```c
/* src/nanolang.h */
typedef enum {
    KIND_TYPE,              /* * */
    KIND_ARROW,             /* k1 -> k2 */
} KindTag;

typedef struct Kind {
    KindTag tag;
    struct Kind *arg;       /* For KIND_ARROW */
    struct Kind *result;    /* For KIND_ARROW */
} Kind;

typedef struct Type {
    /* ... existing fields ... */
    Kind *kind;             /* NEW */
} Type;
```

### Phase 2: Type Constructor Parameters (3 months)

```c
typedef struct TypeParam {
    char *name;
    Kind *kind;
} TypeParam;

typedef struct GenericType {
    char *name;
    TypeParam *params;
    int param_count;
} GenericType;
```

### Phase 3: Higher-Kinded Type Variables (3 months)

```nano
/* Allow type variables with kind * -> * */
fn map<F<_>, A, B>(fa: F<A>, f: fn(A) -> B) -> F<B>
```

### Phase 4: Trait System (6 months)

```nano
trait Functor<F<_>> {
    fn map<A, B>(fa: F<A>, f: fn(A) -> B) -> F<B>
}

impl Functor<List> {
    fn map<A, B>(fa: List<A>, f: fn(A) -> B) -> List<B> {
        /* ... */
    }
}
```

**Total:** 18 months (× 2 for dual implementation = 36 months!)

## Conclusion

**Status:** Analysis complete  
**Decision:** **DO NOT IMPLEMENT HKT**  
**Rationale:**
- ❌ Too complex for NanoLang's goals
- ❌ 12+ months dual implementation cost
- ❌ Limited practical benefit
- ✅ Better alternatives exist

**Recommendation:** Close bead as "won't implement"

**Alternative:** If abstraction over containers becomes critical pain point, revisit with:
1. Macro system first (simpler)
2. GATs second (practical middle ground)
3. Full HKT last (if really needed)

## References

### Languages with HKT
- **Haskell**: https://wiki.haskell.org/Higher-order_type_operator
- **Scala**: https://docs.scala-lang.org/tour/higher-order-functions.html
- **Kind**: https://github.com/HigherOrderCO/Kind

### Languages without HKT
- **Rust** (has GATs): https://blog.rust-lang.org/2022/10/28/gats-stabilization.html
- **Go**: https://go.dev/blog/why-generics
- **Swift**: https://docs.swift.org/swift-book/LanguageGuide/Generics.html

### Papers
- "Lightweight Higher-Kinded Polymorphism" (Yallop & White, 2014)
- "Defunctionalization at Work" (Pottier & Gauthier, 2004)
- "First-Class Modules and Composable Type Classes" (Yallop et al., 2017)

---

**Final Verdict:** Not worth it for NanoLang. Close bead.

