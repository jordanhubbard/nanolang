---
title: اللغة
machine_generated: true
reviewed: false
lang: ar
---

> هذه الصفحة مسودة مولَّدة آلياً ولم تُراجع بعد. تبقى أمثلة الشفرة والأوامر والمعرّفات بالإنجليزية.

# اللغة

أُبقي النحو العادي صريحاً. أقبل بعض التسهيلات، لكنني لا أخفي الأسبقية ولا تواقيع الدوال.

## الاستدعاءات والعوامل

الاستدعاءات بين قوسين وبادئة:

```nano
(println "ready")
(distance x y)
(math.clamp value low high)
```

تقبل العوامل تدوين البادئة والوسط:

```nano
let prefix = (+ 2 (* 3 4))
let infix = 2 + (3 * 4)
```

لكل عوامل الوسط الثنائية الأسبقية نفسها وترتبط من اليسار إلى اليمين. `2 + 3 * 4` تعني `(2 + 3) * 4`. أضف أقواساً عندما يهم التجميع.

## الارتباطات

الارتباطات غير قابلة للتغيير ما لم تُعلَّم `mut`. يستخدم التغيير `set`:

<!--nl-snippet {"name":"refresh_language_bindings","check":true}-->
```nano
fn count_three() -> int {
    let mut count = 0
    set count (+ count 1)
    set count (+ count 1)
    set count (+ count 1)
    return count
}

shadow count_three {
    assert (== (count_three) 3)
}

fn main() -> int {
    assert (== (count_three) 3)
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

التعليقات المحلية اختيارية عندما يحدد المهيئ النوع:

```nano
let count = 3
let name = "Ada"
```

علّق المجموعات الفارغة والقيم العامة والمقابض الأجنبية وقيم الموارد والحدود حيث يحجب الاستنتاج القصد. معاملات الدوال وأنواع الإرجاع تبقى صريحة.

## الأنواع العددية

الأنواع العددية الشائعة هي `int` و`u8` أو `byte` و`float` و`bool` و`string` و`bstring` و`void`. توسّع المصفوفات والصفوف والسجلات المسماة والتعدادات والاتحادات وأنواع الدوال والأنواع العامة والسجلات المفتوحة والأنواع الأجنبية المعتمة ذلك المجموعة.

## تدفق التحكم

يجوز لـ `if` أن يحذف `else`:

```nano
if needs_redraw {
    (draw scene)
}
```

يختار `cond` بين قيم تعبير ويتطلب `else`:

```nano
let sign = (cond
    ((< value 0) -1)
    ((> value 0) 1)
    (else 0)
)
```

تستخدم الحلقات `while` أو `for`:

```nano
while (< index count) {
    set index (+ index 1)
}

for index in (range 0 count) {
    (visit index)
}
```

`break` و`continue` متاحان داخل الحلقات.

## الدوال

```nano
fn gcd(a: int, b: int) -> int {
    if (== b 0) {
        return a
    }
    return (gcd b (% a b))
}

shadow gcd {
    assert (== (gcd 48 18) 6)
}
```

أدعم التكرار والدوال من الدرجة الأولى والإغلاقات والأنواع العامة والشروط المسبقة بـ `requires` والشروط اللاحقة بـ `ensures`. العقود خصائص مفحوصة للتنفيذات؛ ليست براهين صورية.

## التعليقات

```nano
# ordinary comment
// accepted line comment
/* block comment */
/// documentation comment
```

استخدم `#` للتعليق العادي و`///` للتوثيق الذي تستهلكه الأدوات.

