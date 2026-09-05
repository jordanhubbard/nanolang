---
title: تحليل الأداء
machine_generated: true
reviewed: false
lang: ar
---

> هذه الصفحة مسودة مولَّدة آلياً ولم تُراجع بعد. تبقى أمثلة الشفرة والأوامر والمعرّفات بالإنجليزية.

# تحليل الأداء

أغلف ثنائياً أصلياً مترجماً بـ `-pg` بحيث يجمع تشغيله ملف تعريف مضيف ويطبع JSON
يمكن لـ LLM قراءته. لا أضمن تسريعاً. أُبقي تحسيناً عندما يُظهر ملف تعريف ثانٍ
واختباراتي أنه أفاد.

قائمة الحقول القانونية ومتغيرات بيئة إعادة الدخول وأسماء آثار `/tmp` في
[docs/PERFORMANCE_MONITORING.md](https://github.com/jordanhubbard/nanolang/blob/main/docs/PERFORMANCE_MONITORING.md).

## التقاط ملف تعريف

```bash
./bin/nanoc program.nano -o bin/program -pg --profile-output profile.json
./bin/program
```

العملية الأولى منسّق (`_nl_run_with_profiling`). تنشئ أثراً باسم فريد من
معرّف عمليتها، وتشغّل حمل العمل المقاس في ابن، وتنتظر، وتحوّل خرج الجامع إلى
JSON، وتحذف الملفات المؤقتة. التشغيلات المتزامنة لا تشترك في مسار `/tmp` نفسه.

يُكتب JSON إلى `profile.json` وأيضاً إلى **stdout** (بين سطري شعار).
إعادة توجيه stderr لن تلتقطه.

## أعلام ليست `-pg`

| Flag | Role |
| --- | --- |
| `-pg` / `--profile-output` | جامع نظام التشغيل + JSON. يصبح `main` الأصلي `_nl_run_with_profiling`. |
| `--profile` | خطاطيف توقيت في C مولَّد. جدول على stderr. |
| `--trace` | خطاطيف تتبع استدعاء الدوال في C مولَّد. `-> fn` / `<- fn` بمسافة بادئة على stderr. |
| `--profile-runtime` | اكتب أيضاً مكدسات مطوية `.nano.prof`. الخلفية الأصلية فقط. |
| `--pgo <file>` | تضمين باستخدام `.nano.prof`. لا يستخدم JSON من `-pg`. |

أعلام C المضافة مع `-pg`: `-pg -g -fno-omit-frame-pointer -fno-optimize-sibling-calls`.

## حد المنصة

| Platform | Collector | Measurement |
| --- | --- | --- |
| macOS with full Xcode | `xctrace` Time Profiler | أخذ عينات لابن مُشغَّل |
| macOS fallback | `sample` | أخذ عينات دوري لابن متزامن |
| Linux | `gprofng collect app` | جمع مزوَّد في عملية ابن |
| Other OS | none | أطبع أن التحليل غير مدعوم وأشغّل البرنامج |

أعامل xctrace كمتاح فقط عندما ينجح `which xctrace` و`xctrace version`
كلاهما. أدوات سطر الأوامر وحدها لا تكفي.

على macOS ينشئ احتياط `sample` ابن الحمل أولاً ويمسكه على أنبوب
(`_NL_PROFILING_ACTIVE`, `_NL_PROFILING_CWD`). ابن عيّنات يتصل بذلك PID
(`sample <pid> 60 -f /tmp/nanolang_sample_<pid>.txt -mayDie`)، ثم يحرّر
المنسّق الحمل. يمكن للتشغيلات القصيرة أن تنتهي قبل أن يتصل `sample`.

على Linux تستخدم إعادة الدخول `_NL_PROFILING_CHILD` أو `LD_PRELOAD` يحتوي
`libgp-collector`. إذا غاب `gprofng` يعمل الابن بلا جامع.
gprofng جمع مزوَّد. حقل JSON `profile_type` ما زال السلسلة `"sampling"` على
كل نظام تشغيل؛ ذلك وسم تاريخي وليس ادّعاء أن Linux يأخذ عينات.

## JSON الذي أصدره فعلاً

لكل نقطة ساخنة فقط `function` و`samples` و`pct_time`. لا أصدر مواقع مصدر
ولا ميكروثانية لكل استدعاء.

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

`tool` هو `"gprofng"` أو `"xctrace"` أو `"sample"`. على Linux، `samples`
نسبة حصرية مضروبة في عشرة، وليست عدّ عينات حقيقياً. على xctrace أصدر أعلى
20 اسماً تبدأ بـ `nl_`.

عامل ملف التعريف قياساً لحمل واحد على آلة واحدة.

## الضبط مع LLM

1. ترجم بـ `-pg --profile-output`.
2. شغّل حمل عمل تمثيلياً، لا ظلاً من سطر واحد.
3. أعط LLM الـ JSON والمصدر ذا الصلة والأمر الذي أنتج
   الحمل. اطلب تغييراً واحداً مربوطاً بنقطة ساخنة مقاسة.
4. شغّل الاختبارات ثم أعد تحليل الحمل نفسه.
5. أبقِ التغيير فقط إذا بقيت الاختبارات ناجحة وتحسّنت الأرقام.

يمكن لـ LLM أن يقترح تحسيناً؛ يمكن لملف التعريف أن يختبر أثره. لا يغني
أحدهما عن الآخر. قارن ساعة الجدار **بدون** `-pg` عندما تبلّغ السرعة.

يشترك التتبع والتحليل في خطاف بدء واحد. ثنائيات `--profile` تحترم
`NANO_PROFILE`، وثنائيات `--trace` تحترم `NANO_TRACE`؛ ضبط أي منهما إلى `0`
يعطّل ذلك الخطاف وقت التشغيل دون إعادة ترجمة، وخطاف معطّل لا يعمل لكل حدث.

`--pgo` يقرأ `.nano.prof` من `--profile-runtime`. ذلك الملف ليس دخل PGO من
JSON الخاص بـ `-pg`.

