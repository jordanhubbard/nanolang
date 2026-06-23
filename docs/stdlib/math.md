# Math Standard Library

Built-in mathematical functions available without any imports.

> Auto-generated from source. Do not edit directly.

---

## Functions

- [`abs(x: int | float) -> int | float`](#abs)
- [`min(a: int | float, b: int | float) -> int | float`](#min)
- [`max(a: int | float, b: int | float) -> int | float`](#max)
- [`sqrt(x: int | float) -> float`](#sqrt)
- [`pow(base: int | float, exp: int | float) -> float`](#pow)

---

### `abs(x: int | float) -> int | float` { #abs }

Returns the absolute value of a number. Works with both int and float.

**Parameters:**

- `x` — The numeric value to take the absolute value of.

**Returns:** The non-negative magnitude of x.

**Example:**

```nano
(abs -5)      # => 5
(abs 3.14)    # => 3.14
(abs 0)       # => 0
```

---

### `min(a: int | float, b: int | float) -> int | float` { #min }

Returns the smaller of two numeric values. Both arguments must be the same type.

**Parameters:**

- `a` — First numeric value.
- `b` — Second numeric value.

**Returns:** The lesser of a and b.

**Example:**

```nano
(min 5 10)    # => 5
(min -3 0)    # => -3
(min 2.5 1.5) # => 1.5
```

---

### `max(a: int | float, b: int | float) -> int | float` { #max }

Returns the larger of two numeric values. Both arguments must be the same type.

**Parameters:**

- `a` — First numeric value.
- `b` — Second numeric value.

**Returns:** The greater of a and b.

**Example:**

```nano
(max 5 10)    # => 10
(max -3 0)    # => 0
(max 2.5 1.5) # => 2.5
```

---

### `sqrt(x: int | float) -> float` { #sqrt }

Returns the square root of a number. Accepts both int and float; always returns float.

**Parameters:**

- `x` — A non-negative numeric value.

**Returns:** The square root of x as a float.

**Example:**

```nano
(sqrt 4.0)   # => 2.0
(sqrt 9)     # => 3.0
(sqrt 2.0)   # => 1.4142135623730951
```

---

### `pow(base: int | float, exp: int | float) -> float` { #pow }

Raises base to the power of exp. Always returns a float.

**Parameters:**

- `base` — The base value.
- `exp` — The exponent.

**Returns:** base raised to the power exp.

**Example:**

```nano
(pow 2.0 10.0)  # => 1024.0
(pow 3 2)       # => 9.0
(pow 2.0 0.5)   # => 1.4142135623730951
```

---
