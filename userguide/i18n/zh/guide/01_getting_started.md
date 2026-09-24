---
title: 入门
machine_generated: true
reviewed: false
lang: zh
---

> 本页是机器生成的草稿，尚未经人工审校。代码示例、命令和标识符保持英文。

# 入门

我目前在类 Unix 系统上构建。Windows 用户应使用 WSL2。

## 构建

你需要 C 编译器、Make、Git 和 pkg-config。克隆并构建：

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make build bootstrap3
./bin/nanoc --help
```

`bin/nanoc` 是我的编译器。`bin/nano` 是我的树遍历解释器。

## 第一个程序

创建 `hello.nano`：

<!--nl-snippet {"name":"refresh_getting_started_hello","check":true,"expect_stdout":"Hello, World!\n"}-->
```nano
fn main() -> int {
    (println "Hello, World!")
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

编译并运行它：

```bash
./bin/nanoc hello.nano -o hello
./hello
```

使用 `-o hello` 时，我通过 C 转译器和宿主 C 编译器，将经过验证的 NanoISA 转换为原生可执行文件。我在发布可执行文件之前单独运行选定的影子测试；它们不会在程序启动时再次运行。影子测试失败会阻止发布。

如果没有指定输出或目标选项，我会改为发布可移植字节码：

```bash
./bin/nanoc hello.nano
./bin/nano_vm hello.nvm
```

## 解释器

在不生成原生可执行文件的情况下运行源文件：

```bash
./bin/nano hello.nano
```

解释器与编译后端共享同一门语言，但它们的实现边界并不完全相同。请测试你打算交付的那个后端。

## 项目布局

对于小程序，一个 `.nano` 文件就够了。包可以使用 `nano.toml`：

```text
my_program/
  nano.toml
  main.nano
  src/
```

完整布局见 [`examples/hello_pkg`](https://github.com/jordanhubbard/nanolang/tree/main/examples/hello_pkg) 和 [`examples/large_project`](https://github.com/jordanhubbard/nanolang/tree/main/examples/large_project)。

## 下一步

阅读 [语言](02_language.md)，了解调用、运算符、绑定、函数和控制流。 接着阅读 [安全运行时](08_secure_runtime.md)，了解程序如何获准访问宿主。
