---
title: 性能分析
machine_generated: true
reviewed: false
lang: zh
---

> 本页是机器生成的草稿，尚未经人工审校。代码示例、命令和标识符保持英文。

# 性能分析

我包装用 `-pg` 编译的原生二进制，使运行它时收集宿主分析，并打印 LLM 能读的 JSON。我不保证加速。只有当第二次分析和我的测试表明有帮助时，我才保留一项优化。

规范字段列表、再进入环境变量和 `/tmp` 产物名在
[docs/PERFORMANCE_MONITORING.md](https://github.com/jordanhubbard/nanolang/blob/main/docs/PERFORMANCE_MONITORING.md)。

## 采集一份分析

```bash
./bin/nanoc program.nano -o bin/program -pg --profile-output profile.json
./bin/program
```

第一个进程是协调器（`_nl_run_with_profiling`）。它用自己的进程 ID 创建唯一命名的产物，在子进程中启动被测负载，等待，把采集器输出转成 JSON，然后删除临时文件。并发运行不共享同一 `/tmp` 路径。

JSON 写入 `profile.json`，也写入 **stdout**（夹在两行横幅之间）。重定向 stderr 捕获不到它。

## 不是 `-pg` 的标志

| Flag | Role |
| --- | --- |
| `-pg` / `--profile-output` | OS 采集器 + JSON。原生 `main` 变成 `_nl_run_with_profiling`。 |
| `--profile` | 生成 C 中的计时钩子。表格在 stderr。 |
| `--trace` | 生成 C 中的函数调用跟踪钩子。stderr 上缩进的 `-> fn` / `<- fn`。 |
| `--profile-runtime` | 同时写入 `.nano.prof` 折叠栈。仅原生后端。 |
| `--pgo <file>` | 用 `.nano.prof` 做内联。不使用 `-pg` JSON。 |

与 `-pg` 一起加入的 C 标志：`-pg -g -fno-omit-frame-pointer -fno-optimize-sibling-calls`。

## 平台边界

| Platform | Collector | Measurement |
| --- | --- | --- |
| macOS with full Xcode | `xctrace` Time Profiler | 对已启动子进程采样 |
| macOS fallback | `sample` | 对同步子进程周期性采样 |
| Linux | `gprofng collect app` | 子进程中的插装采集 |
| Other OS | none | 我打印不支持分析并运行该程序 |

仅当 `which xctrace` 和 `xctrace version` 都成功时，我才把 xctrace 视为可用。仅有 Command Line Tools 不够。

在 macOS 上，`sample` 回退先创建负载子进程，并把它停在管道上（`_NL_PROFILING_ACTIVE`、`_NL_PROFILING_CWD`）。采样子进程附着到该 PID（`sample <pid> 60 -f /tmp/nanolang_sample_<pid>.txt -mayDie`），然后协调器放开负载。短运行仍可能在 `sample` 附着之前结束。

在 Linux 上，再进入使用 `_NL_PROFILING_CHILD` 或含 `libgp-collector` 的 `LD_PRELOAD`。若缺少 `gprofng`，子进程在没有采集器的情况下运行。gprofng 是插装采集。JSON 字段 `profile_type` 在每个 OS 上仍是字符串 `"sampling"`；那是历史标签，不是声称 Linux 在采样。

## 我实际发出的 JSON

每个热点只有 `function`、`samples` 和 `pct_time`。我不发出源位置或每次调用的微秒数。

```json
{
  "profile_type": "sampling",
  "platform": "macOS",
  "tool": "xctrace",
  "binary": "./bin/program",
  "hotspots": [
    {"function": "nl_hot_function", "samples": 120, "pct_time": 12.0}
  ],
  "analysis_hints": [
    "Functions with high sample counts are hot spots",
    "Look for nl_ prefixed functions (NanoLang generated)",
    "str_ and array_ functions often indicate algorithmic issues",
    "Deep call stacks may indicate recursion or callback chains"
  ]
}
```

`tool` 是 `"gprofng"`、`"xctrace"` 或 `"sample"`。在 Linux 上，`samples` 是独占百分数乘以十，不是真正的样本计数。在 xctrace 上我发出以 `nl_` 开头的前 20 个名字。

把一份分析当作一台机器上一个负载的测量。

## 用 LLM 调优

1. 用 `-pg --profile-output` 编译。
2. 运行有代表性的负载，而不是一行影子测试。
3. 把 JSON、相关源码，以及产生该负载的命令交给 LLM。要求一项绑定到已测热点的改动。
4. 运行测试，然后对同一负载再分析。
5. 仅当测试仍通过且数字改善时保留该改动。

LLM 可以建议优化；分析可以检验其效果。两者不能互相替代。报告速度时比较**没有** `-pg` 的墙钟时间。

跟踪与分析共享一个启动钩子。`--profile` 二进制遵守 `NANO_PROFILE`，`--trace` 二进制遵守 `NANO_TRACE`；把任一设为 `0` 会在运行时禁用该钩子而无需重编译，禁用的钩子不做按事件的工作。

`--pgo` 读取来自 `--profile-runtime` 的 `.nano.prof`。该文件不是来自 `-pg` JSON 的 PGO 输入。

