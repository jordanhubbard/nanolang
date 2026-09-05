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
make build
./bin/nanoc --version
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

我把这个程序转译成 C，并调用宿主 C 编译器。影子测试在我编译时执行；生成的程序随后在启动时运行其生成的影子测试套件。失败的影子测试会终止进程。

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

阅读 [语言](02_language.md)，了解调用、运算符、绑定、函数和控制流。

