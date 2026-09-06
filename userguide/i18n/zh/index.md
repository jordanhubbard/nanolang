---
title: 首页
machine_generated: true
reviewed: false
lang: zh
---

> 本页是机器生成的草稿，尚未经人工审校。代码示例、命令和标识符保持英文。

# NanoLang 用户指南

我是 NanoLang。我使用明确的边界、前缀调用、同等优先级的运算符，以及紧挨着被测函数的测试。
```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    }
    return (* n (factorial (- n 1)))
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 5) 120)
}
```
## 从这里开始

1. [构建我并运行第一个程序](guide/01_getting_started.md)。
2. [学习我的语言](guide/02_language.md)。
3. [使用结构化数据和带类型的错误](guide/03_data_and_errors.md)。
4. [处理模块、外部代码和资源](guide/04_modules_and_ffi.md)。
5. [理解影子测试、测试与已验证边界](guide/05_testing_and_trust.md)。
6. [选择工具或后端](guide/06_tools_and_backends.md)。
7. [根据证据测量原生性能并调整](guide/07_performance_profiling.md)。

## 参考

- [示例](generated/examples.md) 由 `examples/` 下每个 `.nano` 文件生成。
- [内建函数](generated/builtins.md) 来自机械核对的标准库参考。
- [模块](generated/modules.md) 由模块树和清单生成。
- [编译器 CLI](generated/cli.md) 来自构建本指南所用的编译器。
- [NanoISA](https://github.com/jordanhubbard/nanolang/blob/main/docs/NANOISA.md) 是我共享的带类型虚拟机边界。

## 我承诺什么

- 函数调用使用 `(function argument)` 语法。
- 函数参数和返回类型是显式的。
- 当初始化式已经把类型写清楚时，局部绑定可以使用推断。
- 所有中缀运算符优先级相同，并从左到右结合。
- 项目政策要求为改动过的具名函数提供有用的影子测试。编译器强制有文档化的豁免。
- 通过的影子测试是一次测试，不是证明。
- 我的 C 后端是生产编译路径。其他后端有更窄的、已文档化的子集。

当旧文档与解析器、类型检查器、内建注册表和测试不一致时，以后者为准。我宁愿改正指南，也不保留一条自信的错误。

