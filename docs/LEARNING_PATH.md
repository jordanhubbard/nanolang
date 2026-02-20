# My Learning Path

**A structured guide to learning my syntax through examples**

I organize my 50+ examples into a progressive learning path. I move from basic syntax to advanced topics. Each section builds on the previous one. I recommend following this path in order.

---

## How to Use This Guide

1. **Start at your level.** Skip ahead if you're familiar with similar languages.
2. **Run every example.** Compile and execute each example to see it work.
3. **Modify and experiment.** Change values, add features, or break things.
4. **Read shadow tests.** Shadow tests show my expected behavior and edge cases.
5. **Build projects.** Apply what you learn in the project challenges.

### Running Examples

```bash
# Compile and run an example
./bin/nanoc examples/language/nl_hello.nano -o hello
./hello

# Compile and run in one step
./bin/nanoc examples/language/nl_hello.nano --call main
```

---

## Level 1: Absolute Beginner (*)

**Goal:** Understand my syntax, basic types, and functions.

**Prerequisites:** None. Start here.

**Time to complete:** 1 to 2 hours.

### 1.1 Hello World and Basic Output

**File:** `examples/language/nl_hello.nano`

Learn:
- My program structure (fn main)
- My `println` function
- String literals
- Return values

```nano
fn main() -> int {
    (println "Hello, NanoLang!")
    return 0
}
```

**Exercises:**
1. Change the message to your name.
2. Add multiple println statements.
3. Observe what happens if you change the return value.

### 1.2 Variables and Types

**Files:**
- `examples/language/nl_variables.nano`
- `examples/language/nl_integers.nano`
- `examples/language/nl_floats.nano`

Learn:
- Variable declaration with `let`
- Type annotations (int, float, string, bool)
- Arithmetic operations
- My approach to type safety

```nano
let x: int = 42
let y: float = 3.14
let name: string = "Alice"
let flag: bool = true
```

**Exercises:**
1. Create variables of each type.
2. Try mixing types in operations and read the errors I produce.
3. Calculate the area of a rectangle (length * width).

### 1.3 Functions

**Files:**
- `examples/language/nl_functions_basic.nano`
- `examples/language/nl_function_return_values.nano`
- `examples/language/nl_factorial.nano`

Learn:
- Function declaration
- Parameters and return types
- My support for both infix `a + b` and prefix `(+ a b)` notation for operators
- Prefix notation for function calls: `(println "hello")`
- Recursion basics

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 10 20) 30)
}
```

**Exercises:**
1. Write a subtract function.
2. Write a multiply function.
3. Write a function that calculates the average of two numbers.

### 1.4 Shadow Tests (Testing Philosophy)

**File:** `examples/language/nl_shadow_tests_demo.nano`

Learn:
- Shadow test blocks
- The `assert` function
- Test-driven development
- Why I require testing

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

**Exercises:**
1. Add more shadow test cases to your functions.
2. Write a function that intentionally fails a test.
3. Fix the function until all tests pass.

**Checkpoint:** Can you write a tested function that converts Fahrenheit to Celsius?

---

## Level 2: Fundamentals (**)

**Goal:** Master control flow, arrays, and basic data structures.

**Prerequisites:** Complete Level 1.

**Time to complete:** 3 to 4 hours.

### 2.1 Control Flow: If/While

**File:** `examples/language/nl_control_if_while.nano`

Learn:
- if/else statements
- while loops
- Comparison operators: `==`, `<`, `>`, `<=`, `>=`
- Boolean logic: `and`, `or`, `not`

```nano
fn is_even(n: int) -> bool {
    return (== (% n 2) 0)
}

fn count_to_ten() -> void {
    let mut i: int = 1
    while (<= i 10) {
        (println (int_to_string i))
        set i (+ i 1)
    }
}
```

**Exercises:**
1. Write a function that finds the maximum of three numbers.
2. Write a function that counts down from 10 to 1.
3. Implement FizzBuzz. Print "Fizz" for multiples of 3, "Buzz" for 5, and "FizzBuzz" for both.

### 2.2 Control Flow: For Loops

**File:** `examples/language/nl_control_for.nano`

Learn:
- for loops with `range`
- Iterating over sequences
- Loop control

```nano
for i in (range 0 10) {
    (println (int_to_string i))
}
```

**Exercises:**
1. Print the squares of numbers 1-10.
2. Calculate the sum of numbers 1-100.
3. Print a multiplication table.

### 2.3 Arrays

**Files:**
- `examples/language/nl_array_basics.nano`
- `examples/language/nl_array_complete.nano`
- `examples/language/nl_array_bounds.nano`

Learn:
- Array literals: `[1, 2, 3]`
- Array access with `at`
- Array functions: `array_length`, `array_push`, `array_pop`
- My bounds checking

```nano
let numbers: array<int> = [1, 2, 3, 4, 5]
let first: int = (at numbers 0)
let len: int = (array_length numbers)

for i in (range 0 len) {
    let val: int = (at numbers i)
    (println (int_to_string val))
}
```

**Exercises:**
1. Write a function that reverses an array.
2. Find the maximum value in an array.
3. Calculate the average of all numbers in an array.

### 2.4 String Operations

**Files:**
- `examples/language/nl_string_operations.nano`
- `examples/language/nl_string_manipulation_complete.nano`

Learn:
- String concatenation: `(+ "hello" " " "world")`
- String functions: `str_length`, `str_substring`, `str_contains`
- Character access: `char_at`

```nano
let message: string = "Hello, World!"
let len: int = (str_length message)
let first_char: int = (char_at message 0)
let contains_hello: bool = (str_contains message "Hello")
```

**Exercises:**
1. Write a function that counts vowels in a string.
2. Write a function that checks if a string is a palindrome.
3. Implement string reversal.

**Checkpoint:** Can you build a word counter that counts how many times each word appears in a sentence?

---

## Level 3: Intermediate (**)

**Goal:** Use structs, enums, pattern matching, and higher-order functions.

**Prerequisites:** Complete Levels 1 and 2.

**Time to complete:** 4 to 5 hours.

### 3.1 Structs

**Files:**
- `examples/language/nl_struct.nano`
- `examples/language/nl_point.nano`
- `examples/language/nl_struct_methods.nano`

Learn:
- Struct definitions
- Struct initialization
- Field access with dot notation
- Struct methods

```nano
struct Point {
    x: int
    y: int
}

fn distance(p1: Point, p2: Point) -> float {
    let dx: float = (cast_float (- p2.x p1.x))
    let dy: float = (cast_float (- p2.y p1.y))
    return (sqrt (+ (* dx dx) (* dy dy)))
}

let p1: Point = Point { x: 0, y: 0 }
let p2: Point = Point { x: 3, y: 4 }
let dist: float = (distance p1 p2)  # 5.0
```

**Exercises:**
1. Create a Rectangle struct with width and height.
2. Write a function that calculates rectangle area.
3. Create a Person struct with name, age, and email.

### 3.2 Enums

**Files:**
- `examples/language/nl_enum.nano`
- `examples/language/nl_enum_complete.nano`

Learn:
- Enum definitions
- Enum variants
- Using enums for state

```nano
enum Color {
    Red
    Green
    Blue
}

fn color_to_string(c: Color) -> string {
    match c {
        Color::Red => return "red"
        Color::Green => return "green"
        Color::Blue => return "blue"
    }
}
```

**Exercises:**
1. Create a TrafficLight enum with Red, Yellow, and Green variants.
2. Write a function that returns the next state.
3. Create a Direction enum with North, South, East, and West variants and include movement logic.

### 3.3 Pattern Matching

**File:** `examples/language/nl_control_match.nano`

Learn:
- match statements
- Pattern matching on enums
- Exhaustive matching
- Guard clauses

```nano
fn classify_number(n: int) -> string {
    match n {
        0 => return "zero"
        1 => return "one"
        _ => return "many"
    }
}
```

**Exercises:**
1. Write a function that classifies numbers as positive, negative, or zero.
2. Implement rock-paper-scissors winner logic.
3. Create a calculator that matches on operation type.

### 3.4 First-Class Functions

**Files:**
- `examples/language/nl_first_class_functions.nano`
- `examples/language/nl_function_variables.nano`
- `examples/language/nl_function_factories_v2.nano`

Learn:
- Functions as values
- Function parameters
- Higher-order functions
- My support for closures

```nano
fn apply_twice(f: fn(int) -> int, x: int) -> int {
    return (f (f x))
}

fn double(x: int) -> int {
    return (* x 2)
}

let result: int = (apply_twice double 5)  # (double (double 5)) = 20
```

**Exercises:**
1. Write a `map` function that applies a function to every array element.
2. Write a `filter` function that keeps only matching elements.
3. Implement `compose` to chain two functions together.

### 3.5 Map, Filter, Reduce

**File:** `examples/language/nl_filter_map_fold.nano`

Learn:
- Functional programming patterns
- Data transformation pipelines
- Immutable data flow

```nano
fn squares(numbers: array<int>) -> array<int> {
    return (map numbers square)
}

fn square(x: int) -> int {
    return (* x x)
}

fn evens(numbers: array<int>) -> array<int> {
    return (filter numbers is_even)
}
```

**Exercises:**
1. Filter an array to keep only odd numbers.
2. Map an array of temperatures from Celsius to Fahrenheit.
3. Use reduce to calculate the product of all numbers in an array.

**Checkpoint:** Can you implement a data processing pipeline that filters, maps, and reduces in sequence?

---

## Level 4: Advanced (***)

**Goal:** Master generics, unions, advanced types, and FFI.

**Prerequisites:** Complete Levels 1, 2, and 3.

**Time to complete:** 5 to 6 hours.

### 4.1 Generics

**File:** `examples/language/nl_generics_demo.nano`

Learn:
- Generic type parameters: `List<T>`
- Generic functions
- My type inference
- Monomorphization

```nano
struct Box<T> {
    value: T
}

fn box_new<T>(val: T) -> Box<T> {
    return Box { value: val }
}

let int_box: Box<int> = (box_new 42)
let str_box: Box<string> = (box_new "hello")
```

**Exercises:**
1. Create a generic `Pair<T, U>` struct.
2. Write a generic `swap` function that swaps two values.
3. Implement a generic `max` function.

### 4.2 Unions (Tagged Unions)

**File:** `examples/language/nl_union.nano`

Learn:
- Union definitions
- Union variants with data
- Pattern matching on unions
- My Option<T> and Result<T, E> types

```nano
union Option<T> {
    Some(T)
    None
}

fn divide(a: int, b: int) -> Option<int> {
    if (== b 0) {
        return Option::None
    } else {
        return Option::Some((/ a b))
    }
}
```

**Exercises:**
1. Implement a Result<T, E> type for error handling.
2. Write a safe array access function that returns Option<T>.
3. Create a JSON value union with String, Number, Bool, Null, Object, and Array variants.

### 4.3 HashMap

**File:** `examples/language/nl_hashmap_word_count.nano`

Learn:
- HashMap creation and usage
- Key-value storage
- Word counting pattern
- Hash map operations

```nano
let counts: HashMap<string, int> = (map_new)
(map_put counts "hello" 1)
(map_put counts "world" 2)

let hello_count: int = (map_get counts "hello")
```

**Exercises:**
1. Build a phone book that maps names to phone numbers.
2. Count character frequency in a string.
3. Group students by grade.

### 4.4 FFI (Foreign Function Interface)

**Files:**
- `examples/language/nl_extern_math.nano`
- `examples/language/nl_extern_string.nano`
- `examples/language/nl_extern_char.nano`

Learn:
- The `extern` keyword
- Calling C functions
- Type mapping between my world and C
- Safety considerations

```nano
extern fn sqrt(x: float) -> float
extern fn strlen(s: string) -> int

fn hypotenuse(a: float, b: float) -> float {
    return (sqrt (+ (* a a) (* b b)))
}
```

**Exercises:**
1. Use extern to call C math functions like sin, cos, and tan.
2. Create a wrapper for a C string function.
3. Call a C file I/O function.

### 4.5 File I/O

**File:** `examples/language/nl_file_io_complete.nano`

Learn:
- Reading files with `file_read`
- Writing files with `file_write` and `file_append`
- File existence checks
- Error handling

```nano
let content: string = (file_read "data.txt")
let status: int = (file_write "output.txt" "Hello, File!")

if (== status 0) {
    (println "Write successful")
}
```

**Exercises:**
1. Read a file and count its lines.
2. Copy one file to another.
3. Append a log entry to a file with a timestamp.

**Checkpoint:** Can you build a command-line tool that processes a CSV file and outputs statistics?

---

## Level 5: Expert (***)

**Goal:** Understand my internal structures, how I self-host, and advanced patterns.

**Prerequisites:** Complete Levels 1 through 4.

**Time to complete:** 8 to 10 hours.

### 5.1 My Internal Concepts

**Files:**
- `examples/language/nl_lexer.nano`
- `examples/language/nl_parser.nano`
- `examples/language/nl_demo_selfhosting.nano`

Learn:
- Lexical analysis (tokenization)
- Parsing (AST construction)
- Type checking
- Code generation

**Exercises:**
1. Build a simple calculator parser.
2. Implement a basic expression evaluator.
3. Create a mini language interpreter.

### 5.2 Advanced Math

**Files:**
- `examples/language/nl_advanced_math.nano`
- `examples/language/nl_pi_chudnovsky.nano`
- `examples/language/nl_checked_math_demo.nano`

Learn:
- Advanced numerical algorithms
- Floating-point precision
- Overflow checking
- Mathematical series

**Exercises:**
1. Implement Newton's method for square roots.
2. Calculate factorial with overflow detection.
3. Implement a numerical integration function.

### 5.3 Data Structures and Algorithms

**Files:**
- `examples/language/nl_linked_list.nano`
- `examples/language/nl_binary_tree.nano`
- `examples/language/nl_sorting.nano`

Learn:
- Custom data structures
- Recursive algorithms
- Sorting algorithms
- Tree traversal

**Exercises:**
1. Implement quicksort.
2. Build a binary search tree with insert, search, and delete functions.
3. Implement depth-first and breadth-first search.

### 5.4 Simulations and Games

**Files:**
- `examples/language/nl_game_of_life.nano`
- `examples/language/nl_boids.nano`
- `examples/language/nl_falling_sand.nano`

Learn:
- Cellular automata
- Flocking behavior
- Physics simulation
- State management

**Exercises:**
1. Modify Game of Life rules.
2. Add obstacles to the boids simulation.
3. Implement new materials in falling sand.

---

## Project Challenges

Apply your skills to build complete applications.

### Beginner Projects (*)

**Project 1: Todo List**
- Store tasks in memory.
- Add, remove, and list tasks.
- Mark tasks as complete.
- Save to and load from a file.

**Project 2: Number Guessing Game**
- Generate a random number.
- Accept user input.
- Give hints.
- Count guesses.

**Project 3: Basic Calculator**
- Parse expressions like "2 + 3 * 4".
- Handle operator precedence through parentheses.
- Show step-by-step evaluation.

### Intermediate Projects (**)

**Project 4: CSV Analyzer**
- Read CSV files.
- Calculate statistics like mean, median, and mode.
- Filter rows by criteria.
- Export results.

**Project 5: Text Adventure Game**
- Room navigation.
- Item inventory.
- Simple puzzles.
- Save game state.

**Project 6: Markdown to HTML**
- Parse markdown syntax.
- Generate HTML.
- Handle headings, lists, links, bold, and italic text.

### Advanced Projects (***)

**Project 7: Expression Evaluator**
- Tokenizer (lexer).
- Parser (build AST).
- Evaluator (interpret AST).
- Variables and functions.

**Project 8: Simple Web Server**
- Socket programming via FFI.
- HTTP request parsing.
- Route handling.
- Static file serving.

**Project 9: Mini Compiler**
- Lexical analysis.
- Parsing.
- Type checking.
- Code generation (transpile to C).

---

## Learning Resources

### My Documentation
- [SPECIFICATION.md](SPECIFICATION.md). My language spec.
- [STDLIB.md](STDLIB.md). My standard library reference.
- [EXTERN_FFI.md](EXTERN_FFI.md). My foreign function interface.
- [SHADOW_TESTS.md](SHADOW_TESTS.md). My testing philosophy.

### Examples Browser
```bash
make examples-launcher
```

This launches an interactive browser to explore all examples by category.

### Getting Help

1. **Check my examples.** I have 50+ working examples with shadow tests.
2. **Read my error messages.** I provide direct error messages when things go wrong.
3. **Run shadow tests.** They show what I expect to be true.
4. **GitHub Issues.** Open an issue for bugs or questions.

---

## Tips for Success

1. **Write shadow tests first.** Think about expected behavior before coding.
2. **Start simple.** Master my basics before moving to advanced topics.
3. **Experiment freely.** Break things and learn from the errors I produce.
4. **Read other code.** Study my examples to see idiomatic patterns.
5. **Build projects.** Apply what you learn in real programs.
6. **Ask questions.** The community is here to help.

---

## Common Mistakes to Avoid

### Beginner Mistakes

1. **Forgetting type annotations.**
   ```nano
   let x = 5  # Error: must specify type
   let x: int = 5  # Correct
   ```

2. **Assuming operator precedence.**
   ```nano
   let wrong: int = 2 + 3 * 4    # Evaluates left-to-right: (2+3)*4 = 20
   let right: int = 2 + (3 * 4)  # Use parens to group: 2+12 = 14
   ```
   I support both infix `a + b` and prefix `(+ a b)` notation, but all my infix operators have equal precedence and evaluate left-to-right.

3. **Forgetting to declare mut for mutable variables.**
   ```nano
   let x: int = 5
   set x 10  # Error: x is not mutable

   let mut x: int = 5
   set x 10  # Correct
   ```

### Intermediate Mistakes

1. **Not handling errors.**
   ```nano
   let result: int = (divide 10 0)  # Might panic.

   # Better: Check for errors
   let result_opt: Option<int> = (safe_divide 10 0)
   ```

2. **Ignoring shadow test failures.**
   My shadow tests are not optional. They validate correctness. Fix failing tests immediately.

3. **Overcomplicating solutions.**
   Start with simple, working code. Refactor later if you need to. I prefer honest, direct code over premature optimization.

### Advanced Mistakes

1. **Unsafe FFI usage.**
   Always validate inputs before passing to C. Check return values. See [EXTERN_FFI.md](EXTERN_FFI.md) for my safety guidelines.

2. **Ignoring performance.**
   Profile before optimizing. Use appropriate data structures. Batch operations when possible.

3. **Poor error messages.**
   Use descriptive error types. Include context in errors. Make debugging easy for those who follow you.

---

## Next Steps After This Guide

Once you've completed this learning path:

1. **Contribute to me.** See [CONTRIBUTING.md](../CONTRIBUTING.md).
2. **Build your own projects.** Apply what you learned.
3. **Explore self-hosting.** Study how I compile myself.
4. **Join the community.** Share your projects and help others.

---

**Last Updated:** January 25, 2026.
**Examples:** 50+ working examples with shadow tests.
**Path Duration:** 20 to 30 hours total for all levels.

Happy learning.
