---
title: 语言
machine_generated: true
reviewed: false
lang: zh
---

> 本页是机器生成的草稿，尚未经人工审校。代码示例、命令和标识符保持英文。

# 语言

我让普通语法保持显式。我接受少量便利写法，但不会隐藏优先级或函数签名。

## 调用与运算符

调用带括号，且为前缀：

```nano
(println "ready")
(distance x y)
(math.clamp value low high)
```

运算符接受前缀和中缀写法：

```nano
let prefix = (+ 2 (* 3 4))
let infix = 2 + (3 * 4)
```

所有中缀二元运算符优先级相同，并从左到右结合。`2 + 3 * 4` 表示 `(2 + 3) * 4`。分组重要时请加括号。

## 绑定

绑定默认不可变，除非标上 `mut`。变更使用 `set`：

<!--nl-snippet {"name":"refresh_language_bindings","check":true}-->
```nano
fn count_three() -> int {
    let mut count = 0
    set count (+ count 1)
    set count (+ count 1)
    set count (+ count 1)
    return count
}

shadow count_three {
    assert (== (count_three) 3)
}

fn main() -> int {
    assert (== (count_three) 3)
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

当初始化式已经确定类型时，局部注解是可选的：

```nano
let count = 3
let name = "Ada"
```

为空集合、泛型值、外部句柄、资源值，以及推断会掩盖意图的边界加上注解。函数参数和返回类型保持显式。

## 标量类型

常见标量类型是 `int`、`u8` 或 `byte`、`float`、`bool`、`string`、`bstring` 和 `void`。数组、元组、具名记录、枚举、联合、函数类型、泛型类型、开放记录和不透明外部类型扩展了这个集合。

## 控制流

`if` 可以省略 `else`：

```nano
if needs_redraw {
    (draw scene)
}
```

`cond` 在表达式值之间选择，并且必须有 `else`：

```nano
let sign = (cond
    ((< value 0) -1)
    ((> value 0) 1)
    (else 0)
)
```

循环使用 `while` 或 `for`：

```nano
while (< index count) {
    set index (+ index 1)
}

for index in (range 0 count) {
    (visit index)
}
```

循环内部可以使用 `break` 和 `continue`。

## 函数

```nano
fn gcd(a: int, b: int) -> int {
    if (== b 0) {
        return a
    }
    return (gcd b (% a b))
}

shadow gcd {
    assert (== (gcd 48 18) 6)
}
```

我支持递归、一等函数、闭包、泛型、用 `requires` 写的前置条件和用 `ensures` 写的后置条件。契约是对执行的已检查性质；它们不是形式证明。

## 注释

```nano
# ordinary comment
// accepted line comment
/* block comment */
/// documentation comment
```

普通说明用 `#`，供工具消费的文档用 `///`。

