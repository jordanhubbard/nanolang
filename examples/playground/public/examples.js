// NanoLang Playground Examples
// Collection of code examples for interactive learning

const EXAMPLES = {
    hello: {
        title: "Hello World",
        description: "Basic NanoLang program with function and shadow test",
        code: `fn greet(name: string) -> string {
    return (+ "Hello, " name)
}

shadow greet {
    assert (str_equals (greet "World") "Hello, World")
    assert (str_equals (greet "NanoLang") "Hello, NanoLang")
}

fn main() -> int {
    (println (greet "World"))
    (println (greet "Playground"))
    return 0
}

shadow main {
    assert (== (main) 0)
}`
    },

    factorial: {
        title: "Factorial (Recursion)",
        description: "Recursive factorial calculation",
        code: `fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    } else {
        return (* n (factorial (- n 1)))
    }
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 1) 1)
    assert (== (factorial 5) 120)
    assert (== (factorial 10) 3628800)
}

fn main() -> int {
    (println (int_to_string (factorial 5)))
    (println (int_to_string (factorial 10)))
    return 0
}

shadow main {
    assert (== (main) 0)
}`
    },

    fibonacci: {
        title: "Fibonacci Sequence",
        description: "Calculate Fibonacci numbers recursively",
        code: `fn fib(n: int) -> int {
    if (<= n 1) {
        return n
    } else {
        return (+ (fib (- n 1)) (fib (- n 2)))
    }
}

shadow fib {
    assert (== (fib 0) 0)
    assert (== (fib 1) 1)
    assert (== (fib 5) 5)
    assert (== (fib 10) 55)
}

fn main() -> int {
    let mut i: int = 0
    while (<= i 10) {
        (println (int_to_string (fib i)))
        set i (+ i 1)
    }
    return 0
}

shadow main {
    assert (== (main) 0)
}`
    },

    prime: {
        title: "Prime Number Checker",
        description: "Check if a number is prime",
        code: `fn is_prime(n: int) -> bool {
    if (< n 2) {
        return false
    } else { (print "") }

    let mut i: int = 2
    while (< i n) {
        if (== (% n i) 0) {
            return false
        } else { (print "") }
        set i (+ i 1)
    }
    return true
}

shadow is_prime {
    assert (not (is_prime 0))
    assert (not (is_prime 1))
    assert (is_prime 2)
    assert (is_prime 3)
    assert (not (is_prime 4))
    assert (is_prime 17)
}

fn main() -> int {
    (println "Prime numbers up to 20:")
    for n in (range 1 21) {
        if (is_prime n) {
            (println (int_to_string n))
        } else { (print "") }
    }
    return 0
}

shadow main {
    assert (== (main) 0)
}`
    },

    arrays: {
        title: "Array Operations",
        description: "Working with arrays in NanoLang",
        code: `fn sum_array(arr: array<int>) -> int {
    let mut sum: int = 0
    for i in (range 0 (array_length arr)) {
        set sum (+ sum (array_get arr i))
    }
    return sum
}

shadow sum_array {
    assert (== (sum_array [1, 2, 3, 4, 5]) 15)
    assert (== (sum_array [10, 20, 30]) 60)
    assert (== (sum_array []) 0)
}

fn find_max(arr: array<int>) -> int {
    let mut max: int = (array_get arr 0)
    for i in (range 1 (array_length arr)) {
        let val: int = (array_get arr i)
        if (> val max) {
            set max val
        } else { (print "") }
    }
    return max
}

shadow find_max {
    assert (== (find_max [1, 5, 3, 9, 2]) 9)
    assert (== (find_max [10, 10, 10]) 10)
}

fn main() -> int {
    let numbers: array<int> = [5, 2, 8, 1, 9, 3]
    (println (+ "Sum: " (int_to_string (sum_array numbers))))
    (println (+ "Max: " (int_to_string (find_max numbers))))
    return 0
}

shadow main {
    assert (== (main) 0)
}`
    },

    strings: {
        title: "String Manipulation",
        description: "String operations and character handling",
        code: `fn reverse_string(s: string) -> string {
    let mut result: string = ""
    let len: int = (str_length s)
    let mut i: int = (- len 1)

    while (>= i 0) {
        let ch: int = (char_at s i)
        set result (+ result (string_from_char ch))
        set i (- i 1)
    }

    return result
}

shadow reverse_string {
    assert (str_equals (reverse_string "hello") "olleh")
    assert (str_equals (reverse_string "abc") "cba")
    assert (str_equals (reverse_string "") "")
}

fn count_vowels(s: string) -> int {
    let mut count: int = 0
    let len: int = (str_length s)

    for i in (range 0 len) {
        let ch: int = (char_at s i)
        if (or (or (or (or (== ch 97) (== ch 101)) (== ch 105)) (== ch 111)) (== ch 117)) {
            set count (+ count 1)
        } else { (print "") }
    }

    return count
}

shadow count_vowels {
    assert (== (count_vowels "hello") 2)
    assert (== (count_vowels "aeiou") 5)
    assert (== (count_vowels "xyz") 0)
}

fn main() -> int {
    let text: string = "NanoLang"
    (println (+ "Original: " text))
    (println (+ "Reversed: " (reverse_string text)))
    (println (+ "Vowels: " (int_to_string (count_vowels text))))
    return 0
}

shadow main {
    assert (== (main) 0)
}`
    },

    struct: {
        title: "Structs",
        description: "Defining and using struct types",
        code: `struct Point {
    x: int,
    y: int
}

fn distance_from_origin(p: Point) -> int {
    return (+ (abs p.x) (abs p.y))
}

shadow distance_from_origin {
    let p1: Point = Point { x: 3, y: 4 }
    assert (== (distance_from_origin p1) 7)

    let p2: Point = Point { x: -5, y: 12 }
    assert (== (distance_from_origin p2) 17)
}

fn move_point(p: Point, dx: int, dy: int) -> Point {
    return Point { x: (+ p.x dx), y: (+ p.y dy) }
}

shadow move_point {
    let p: Point = Point { x: 10, y: 20 }
    let moved: Point = (move_point p 5 -3)
    assert (== moved.x 15)
    assert (== moved.y 17)
}

fn main() -> int {
    let origin: Point = Point { x: 0, y: 0 }
    let p: Point = Point { x: 10, y: 15 }

    (println (+ "Point: (" (+ (int_to_string p.x) (+ ", " (+ (int_to_string p.y) ")")))))
    (println (+ "Distance: " (int_to_string (distance_from_origin p))))

    let moved: Point = (move_point p 5 -5)
    (println (+ "After move: (" (+ (int_to_string moved.x) (+ ", " (+ (int_to_string moved.y) ")")))))

    return 0
}

shadow main {
    assert (== (main) 0)
}`
    },

    cond: {
        title: "Conditionals (cond expression)",
        description: "Using cond for multi-way branches",
        code: `fn classify_number(n: int) -> string {
    return (cond
        ((< n 0) "negative")
        ((== n 0) "zero")
        ((< n 10) "single digit")
        ((< n 100) "double digit")
        (else "large")
    )
}

shadow classify_number {
    assert (str_equals (classify_number -5) "negative")
    assert (str_equals (classify_number 0) "zero")
    assert (str_equals (classify_number 7) "single digit")
    assert (str_equals (classify_number 42) "double digit")
    assert (str_equals (classify_number 999) "large")
}

fn grade_score(score: int) -> string {
    return (cond
        ((>= score 90) "A")
        ((>= score 80) "B")
        ((>= score 70) "C")
        ((>= score 60) "D")
        (else "F")
    )
}

shadow grade_score {
    assert (str_equals (grade_score 95) "A")
    assert (str_equals (grade_score 85) "B")
    assert (str_equals (grade_score 75) "C")
    assert (str_equals (grade_score 65) "D")
    assert (str_equals (grade_score 50) "F")
}

fn main() -> int {
    (println (classify_number -10))
    (println (classify_number 0))
    (println (classify_number 5))
    (println (classify_number 42))
    (println (classify_number 1000))

    (println (grade_score 95))
    (println (grade_score 75))
    (println (grade_score 55))

    return 0
}

shadow main {
    assert (== (main) 0)
}`
    },

    loops: {
        title: "Loops (while & for)",
        description: "Iteration with while and for loops",
        code: `fn sum_while(n: int) -> int {
    let mut sum: int = 0
    let mut i: int = 1
    while (<= i n) {
        set sum (+ sum i)
        set i (+ i 1)
    }
    return sum
}

shadow sum_while {
    assert (== (sum_while 5) 15)  # 1+2+3+4+5
    assert (== (sum_while 10) 55)
    assert (== (sum_while 0) 0)
}

fn sum_for(n: int) -> int {
    let mut sum: int = 0
    for i in (range 1 (+ n 1)) {
        set sum (+ sum i)
    }
    return sum
}

shadow sum_for {
    assert (== (sum_for 5) 15)
    assert (== (sum_for 10) 55)
    assert (== (sum_for 0) 0)
}

fn count_down(n: int) -> void {
    let mut i: int = n
    while (> i 0) {
        (println (int_to_string i))
        set i (- i 1)
    }
    (println "Liftoff!")
}

shadow count_down {
    (count_down 3)
}

fn main() -> int {
    (println (+ "Sum (while): " (int_to_string (sum_while 10))))
    (println (+ "Sum (for): " (int_to_string (sum_for 10))))

    (println "Countdown:")
    (count_down 5)

    return 0
}

shadow main {
    assert (== (main) 0)
}`
    },

    recursion: {
        title: "Recursion Patterns",
        description: "Various recursive algorithms",
        code: `fn sum_recursive(n: int) -> int {
    if (<= n 0) {
        return 0
    } else {
        return (+ n (sum_recursive (- n 1)))
    }
}

shadow sum_recursive {
    assert (== (sum_recursive 0) 0)
    assert (== (sum_recursive 5) 15)
    assert (== (sum_recursive 10) 55)
}

fn power(base: int, exp: int) -> int {
    if (== exp 0) {
        return 1
    } else {
        return (* base (power base (- exp 1)))
    }
}

shadow power {
    assert (== (power 2 0) 1)
    assert (== (power 2 3) 8)
    assert (== (power 5 2) 25)
    assert (== (power 10 3) 1000)
}

fn gcd(a: int, b: int) -> int {
    if (== b 0) {
        return a
    } else {
        return (gcd b (% a b))
    }
}

shadow gcd {
    assert (== (gcd 48 18) 6)
    assert (== (gcd 100 50) 50)
    assert (== (gcd 17 19) 1)
}

fn main() -> int {
    (println (+ "Sum 1..10: " (int_to_string (sum_recursive 10))))
    (println (+ "2^10: " (int_to_string (power 2 10))))
    (println (+ "GCD(48, 18): " (int_to_string (gcd 48 18))))

    return 0
}

shadow main {
    assert (== (main) 0)
}`
    }
};
