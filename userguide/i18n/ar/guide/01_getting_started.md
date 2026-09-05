---
title: البداية
machine_generated: true
reviewed: false
lang: ar
---

> هذه الصفحة مسودة مولَّدة آلياً ولم تُراجع بعد. تبقى أمثلة الشفرة والأوامر والمعرّفات بالإنجليزية.

# البداية

أبني حالياً على أنظمة شبيهة بيونكس. ينبغي لمستخدمي Windows استخدام WSL2.

## البناء

تحتاج مترجم C وMake وGit وpkg-config. انسخ وابنِ:

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make build
./bin/nanoc --version
```

`bin/nanoc` هو مترجمي. `bin/nano` هو مفسّري الذي يمشي على الشجرة.

## البرنامج الأول

أنشئ `hello.nano`:

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

ترجمه وشغّله:

```bash
./bin/nanoc hello.nano -o hello
./hello
```

أحوّل هذا البرنامج إلى C وأستدعي مترجم C للمضيف. تُنفَّذ الظلال أثناء ترجمتي؛ ثم يشغّل البرنامج الناتج حزمة الظلال المولَّدة عند البدء. ظل فاشل يوقف العملية.

## المفسّر

شغّل ملف مصدر دون إنتاج تنفيذي أصلي:

```bash
./bin/nano hello.nano
```

يشترك المفسّر والخلفية المترجمة في اللغة، لكن حدود التنفيذ ليست متطابقة. اختبر الخلفية التي تنوي تسليمها.

## تخطيط المشروع

لبرامج صغيرة يكفي ملف `.nano` واحد. يمكن للحزم استخدام `nano.toml`:

```text
my_program/
  nano.toml
  main.nano
  src/
```

انظر [`examples/hello_pkg`](https://github.com/jordanhubbard/nanolang/tree/main/examples/hello_pkg) و[`examples/large_project`](https://github.com/jordanhubbard/nanolang/tree/main/examples/large_project) للتخطيطات الكاملة.

## التالي

اقرأ [اللغة](02_language.md) للاستدعاءات والعوامل والارتباطات والدوال وتدفق التحكم.

