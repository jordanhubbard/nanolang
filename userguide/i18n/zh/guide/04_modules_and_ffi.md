---
title: 模块、FFI 与资源
machine_generated: true
reviewed: false
lang: zh
---

> 本页是机器生成的草稿，尚未经人工审校。代码示例、命令和标识符保持英文。

# 模块、FFI 与资源

模块是可见性与安全边界。新代码应使用 `module`、别名，以及限定的公开名字。

## 导入模块

```nano
module "modules/std/mathx/mathx.nano" as mathx

fn bounded(value: int) -> int {
    return (mathx.mathx_clamp value 0 100)
}

shadow bounded {
    assert (== (bounded -5) 0)
    assert (== (bounded 120) 100)
}
```

解析器仍接受旧式 `import` 和 `from ... import ...` 形式。不要为新代码选择它们。

## 可见性

声明默认私有。用 `pub` 标出打算公开的表面：

```nano
fn internal_scale(value: int) -> int {
    return (* value 2)
}

pub fn transform(value: int) -> int {
    return (+ (internal_scale value) 1)
}
```

私有辅助函数在其模块内仍可调用。导入方只能调用公开声明。

## 外部函数

外部声明使用 `extern fn`。直接调用需要 `unsafe`，除非整个被导入模块是 unsafe：

```nano
extern fn c_close(fd: int) -> int

fn close_fd(fd: int) -> int {
    unsafe {
        return (c_close fd)
    }
}
```

把不安全区域保持狭窄。在边界上校验外部值，并在能够诚实提供时暴露带类型的包装。

## 资源类型

`resource struct` 标记一个仿射资源，最多应被消费一次：

```nano
resource struct FileHandle {
    fd: int
}
```

我的资源检查器会检测重要的消费后使用和重复消费情况。它不是完整的所有权证明。显式注解资源局部变量，并检查每条返回路径上的清理。

## 模块元数据

三个文件承担不同工作：

| File | Purpose |
| --- | --- |
| `.nano` | 源码与公开声明 |
| `module.json` | 原生构建源、标志、包和所有权元数据 |
| `module.manifest.json` | 发现元数据、稳定性、能力和示例 |

纯模块并不总需要原生构建元数据。现有内容见生成的 [module inventory](../generated/modules.md)。

