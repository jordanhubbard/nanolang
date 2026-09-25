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
make build bootstrap3
./bin/nanoc --help
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

مع `-o hello`، أُنتج ملفًا تنفيذيًا أصليًا من NanoISA الذي جرى التحقق منه، عبر مترجمي إلى C ومترجم C للمضيف. أشغّل اختبارات الظل المحددة بصورة منفصلة قبل نشر الملف التنفيذي؛ ولا تُشغَّل مجددًا عند بدء البرنامج. يمنع فشل اختبار ظل نشر الملف.

من دون خيار للمخرج أو الهدف، أنشر بدلًا من ذلك شيفرة بايت محمولة:

```bash
./bin/nanoc hello.nano
./bin/nano_vm hello.nvm
```

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

اقرأ [اللغة](02_language.md) للاستدعاءات والعوامل والارتباطات والدوال وتدفق التحكم. بعد ذلك، يشرح [بيئة التشغيل الآمنة](08_secure_runtime.md) كيف يُسمح للبرنامج بالوصول إلى المضيف.
