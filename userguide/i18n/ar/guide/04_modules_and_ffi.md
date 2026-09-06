---
title: الوحدات وFFI والموارد
machine_generated: true
reviewed: false
lang: ar
---

> هذه الصفحة مسودة مولَّدة آلياً ولم تُراجع بعد. تبقى أمثلة الشفرة والأوامر والمعرّفات بالإنجليزية.

# الوحدات وFFI والموارد

الوحدة حد رؤية وأمان. ينبغي للشيفرة الجديدة أن تستخدم `module` ولقباً وأسماء عامة مؤهلة.

## استيراد وحدة

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

المحلل ما زال يقبل شكلي `import` و`from ... import ...` الموروثين. لا تخترهما للشيفرة الجديدة.

## الرؤية

الإعلانات خاصة افتراضياً. علّم السطح المقصود بـ `pub`:

```nano
fn internal_scale(value: int) -> int {
    return (* value 2)
}

pub fn transform(value: int) -> int {
    return (+ (internal_scale value) 1)
}
```

المساعدات الخاصة تبقى قابلة للاستدعاء داخل وحدتها. المستوردون يستدعون الإعلانات العامة فقط.

## الدوال الأجنبية

تستخدم الإعلانات الأجنبية `extern fn`. الاستدعاء المباشر يتطلب `unsafe` ما لم تكن الوحدة المستوردة كلها unsafe:

```nano
extern fn c_close(fd: int) -> int

fn close_fd(fd: int) -> int {
    unsafe {
        return (c_close fd)
    }
}
```

أبقِ المناطق غير الآمنة ضيقة. تحقق من القيم الأجنبية عند الحد واكشف غلافاً ذا نوع عندما يمكن تقديمه بأمانة.

## أنواع الموارد

`resource struct` يعلّم مورداً تآلفياً ينبغي استهلاكه مرة واحدة على الأكثر:

```nano
resource struct FileHandle {
    fd: int
}
```

فاحص مواردي يكتشف حالات مهمة للاستخدام بعد الاستهلاك والاستهلاك المتكرر. ليس برهان ملكية كاملاً. علّق محليات المورد صراحة وافحص كل مسار إرجاع للتنظيف.

## بيانات الوحدة الوصفية

ثلاثة ملفات تؤدي أعمالاً مختلفة:

| File | Purpose |
| --- | --- |
| `.nano` | المصدر والإعلانات العامة |
| `module.json` | مصادر البناء الأصلي والأعلام والحزم وبيانات ملكية وصفية |
| `module.manifest.json` | بيانات اكتشاف واستقرار وقدرات وأمثلة |

الوحدات الصافية لا تحتاج دائماً بيانات بناء أصلي. انظر [module inventory](../generated/modules.md) المولَّد لما يوجد الآن.

