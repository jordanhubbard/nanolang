---
title: 数据与错误
machine_generated: true
reviewed: false
lang: zh
---

> 本页是机器生成的草稿，尚未经人工审校。代码示例、命令和标识符保持英文。

# 数据与错误

我显式表示集合和领域状态。选择一种让无效状态难以表达的类型。

## 数组与元组

```nano
let values: array<int> = [10, 20, 30]
let first = (at values 0)
(array_set values 1 25)

let pair: (int, string) = (7, "seven")
let number = pair.0
```

数组有边界检查。有些数组函数会改写存储并返回 `void`；另一些返回新数组。查阅 [Builtins](../generated/builtins.md)，不要从熟悉的名字推断所有权。

## 结构体与枚举

```nano
struct Point {
    x: int
    y: int
}

enum Direction {
    North
    South
    East
    West
}

let origin = Point { x: 0, y: 0 }
```

具名类型使用 `UpperCamelCase`。值和函数使用 `snake_case`。

## 联合与匹配

```nano
union ParseResult {
    Value(int)
    Error(string)
}

fn unwrap_or(result: ParseResult, fallback: int) -> int {
    return (match result {
        Value(value) => value
        Error(_) => fallback
    })
}

shadow unwrap_or {
    assert (== (unwrap_or Value(7) 0) 7)
    assert (== (unwrap_or Error("bad") 3) 3)
}
```

对调用方可以处理的失败，使用 `Result<T, E>` 或其他显式联合。能恢复或补充上下文时匹配变体。后缀 `?` 运算符会传播兼容的结果错误；需要清理或上下文时使用显式 `match`。

## 字符串与二进制字符串

`string` 是文本。`bstring` 是带显式长度的二进制数据，可以包含零字节。转换和 Unicode 行为按内建函数或模块记载。不要假定字节下标就是 Unicode 字符下标。

## 集合

内建数组和哈希表与以 C 为后端的集合模块是分开的。生成的 [Builtins](../generated/builtins.md) 页列出确切的内建拼写。生成的 [Modules](../generated/modules.md) 页列出模块声明和原生构建边界。

