---
title: 测试与信任
machine_generated: true
reviewed: false
lang: zh
---

> 本页是机器生成的草稿，尚未经人工审校。代码示例、命令和标识符保持英文。

# 测试与信任

我区分测试、静态检查和证明。它们回答不同的问题。

## 影子测试

影子测试是附着在函数上的可执行测试：

```nano
fn double(value: int) -> int {
    return (* value 2)
}

shadow double {
    assert (== (double 0) 0)
    assert (== (double 3) 6)
}
```

在普通编译期间，我在宿主解释器中执行影子测试。生成的原生程序也包含其影子测试套件。影子测试只覆盖它执行到的情形；它并不对每个输入证明该函数。

编译器强制与项目政策不同：

- 编译器通常对缺失的影子测试发出警告。
- 它豁免 extern 函数、`main`、生成的 lambda、GPU 函数，以及调用 extern 的函数。
- 仓库政策要求：每个新增或改动的、非 extern 的具名函数，在可测时都要有有用的影子测试。

## 属性测试与覆盖率

属性测试对生成的值抽样，并能缩小失败。抽样不是穷尽验证。覆盖率报告哪些代码执行过；它不确立正确性。

## 契约

`requires` 检查前置条件，`ensures` 在其实现边界检查后置条件。一次成功的运行时检查只说明该条件在那次执行中成立。

## 形式验证

我的 Coq 开发为一个已定义的核心模型证明所述元理论。它不会自动证明每个后端、外部调用、分配器、模块或用户函数。

用信任报告检查这条边界：

```bash
./bin/nanoc program.nano --trust-report
```

只对在其所述模型中已检查的定理说 **proved**，只对由具名测试行使的行为说 **tested**，对未检查的平台或外部行为说 **assumed**。

## 有用的检查

```bash
make test
make userguide-check
make check-stdlib-docs
python3 scripts/check_markdown_links.py
```

