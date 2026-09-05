---
title: البيانات والأخطاء
machine_generated: true
reviewed: false
lang: ar
---

> هذه الصفحة مسودة مولَّدة آلياً ولم تُراجع بعد. تبقى أمثلة الشفرة والأوامر والمعرّفات بالإنجليزية.

# البيانات والأخطاء

أمثّل المجموعات وحالات المجال صراحة. اختر نوعاً يجعل الحالات غير الصالحة صعبة التعبير.

## المصفوفات والصفوف

```nano
let values: array<int> = [10, 20, 30]
let first = (at values 0)
(array_set values 1 25)

let pair: (int, string) = (7, "seven")
let number = pair.0
```

المصفوفات مفحوصة الحدود. بعض دوال المصفوفة تغيّر التخزين وترجع `void`؛ وأخرى ترجع مصفوفة جديدة. راجع [Builtins](../generated/builtins.md) بدلاً من استنتاج الملكية من اسم مألوف.

## البنى والتعدادات

```nano
struct Point {
    x: int
    y: int
}

enum Direction {
    North
    South
    East
    West
}

let origin = Point { x: 0, y: 0 }
```

الأنواع المسماة تستخدم `UpperCamelCase`. القيم والدوال تستخدم `snake_case`.

## الاتحادات والمطابقة

```nano
union ParseResult {
    Value(int)
    Error(string)
}

fn unwrap_or(result: ParseResult, fallback: int) -> int {
    return (match result {
        Value(value) => value
        Error(_) => fallback
    })
}

shadow unwrap_or {
    assert (== (unwrap_or Value(7) 0) 7)
    assert (== (unwrap_or Error("bad") 3) 3)
}
```

استخدم `Result<T, E>` أو اتحاداً صريحاً آخر للإخفاقات التي يمكن للمستدعي معالجتها. طابق المتغيرات حيث يمكنك الاسترداد أو إضافة سياق. عامل اللاحقة `?` ينشر أخطاء النتيجة المتوافقة؛ استخدم `match` صريحاً عندما تهم التنظيف أو السياق.

## السلاسل والسلاسل الثنائية

`string` نص. `bstring` بيانات ثنائية بطول صريح ويمكن أن تحتوي بايتات صفر. التحويل وسلوك يونيكود موثّقان لكل مضمّن أو وحدة. لا تفترض أن فهارس البايت هي فهارس محارف يونيكود.

## المجموعات

المصفوفات وجداول التجزئة المضمنة منفصلة عن وحدات المجموعات ذات الخلفية C. تسرد صفحة [Builtins](../generated/builtins.md) المولَّدة التهجئات الدقيقة للمضمنات. تسرد صفحة [Modules](../generated/modules.md) المولَّدة إعلانات الوحدات وحدود البناء الأصلي.

