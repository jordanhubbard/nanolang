---
title: الأدوات والخلفيات
machine_generated: true
reviewed: false
lang: ar
---

> هذه الصفحة مسودة مولَّدة آلياً ولم تُراجع بعد. تبقى أمثلة الشفرة والأوامر والمعرّفات بالإنجليزية.

# الأدوات والخلفيات

لدي عدة مسارات تنفيذ. تشترك في النحو لكن ليس في تكافؤ الميزات الكامل.

## الأدوات

| Tool | Purpose |
| --- | --- |
| `bin/nanoc` | ترجمة المصدر وإظهار خيارات التحليل أو الخلفية |
| `bin/nano` | تفسير ملف مصدر |
| `bin/nanolang-repl` | تقييم تفاعلي |
| `nano-fmt` | تنسيق المصدر |
| `nano-docs` | البحث في التوثيق المحلي |
| `bin/nanolang-lsp` | دعم Language Server Protocol |
| `bin/nanolang-dap` | دعم Debug Adapter Protocol |
| `bin/nano_virt` | خفض المصدر إلى بايت كود NanoISA |
| `bin/nano_vm` | تنفيذ بايت كود NanoISA |
| `bin/nano_vmd` | تشغيل عفريت NanoVM |
| `bin/nano_cop` | عزل الاستدعاءات الأجنبية المدعومة في عملية مشاركة |

شغّل كل أداة بـ `--help` حيث يُوفَّر. تسجّل صفحة [Compiler CLI](../generated/cli.md) المولَّدة نص مساعدة المترجم الحالي.

## الخلفيات

| Output | Command | Boundary |
| --- | --- | --- |
| Native executable | `nanoc source.nano -o program` | مسار الإنتاج عبر C مولَّد |
| C source | `nanoc source.nano --target c -o program.c` | C مولَّد مستقل |
| NanoISA | `nano_virt source.nano --emit-nvm -o program.nvm` | تمثيل آلة افتراضية مشتركة ذات أنواع |
| PTX | `nanoc source.nano --target ptx -o program.ptx` | مجموعة فرعية لنواة GPU |
| OpenCL C | `nanoc source.nano --target opencl -o program.cl` | مجموعة فرعية لنواة GPU |
| RISC-V assembly | `nanoc source.nano --target riscv -o program.s` | مجموعة فرعية تجريبية |
| NanoISA | `nano_virt source.nano -o program.nvm` | مسار آلة افتراضية مع FFI معزول |

أهداف LLVM وWebAssembly المستقبلية تترجم من NanoISA بدلاً من التفرع من AST مصدري. C11 من NanoISA تجربة مجموعة فرعية مغلقة (`nvm2c`, `make test-nvm2c`)، وليست الغلاف الأصلي الافتراضي لـ `nano_virt` الذي ما زال يضمّن الآلة الافتراضية.

## التشخيصات

تتضمن التشخيصات الموجهة للآلات شكلي JSON وTOON. تشمل خيارات المترجم المفيدة `--llm-diags-json` و`--llm-diags-toon` و`--json-errors` و`--emit-typed-ast-json` و`--reflect`. أختار محلية العملية بـ `--locale <tag>` (ثم `NANO_LOCALE` و`LC_ALL` و`LANG` وإلا `en`) وأطبع المحاور بـ `--print-locale`. ذلك لا يغيّر خرج التشخيص المقروء آلياً. الفهارس موجودة: stderr البشري يبحث في فهارس UTF-8؛ JSON/TOON يبقيان بالإنجليزية؛ `--locale` / `NANO_CATALOG_DIR`. تستخدم تشخيصات مترجم الأنبوب معرّفات مستقرة (`CIO01` و`CSRC01` و`L0003` وبقية `src/diag_id.c`). عناوين مدقق الأنواع التي تمر عبر `emit_context_error` تستخدم `E001`–`E035` فريدة. المعرّفات ASCII؛ غير ASCII يفشل مغلقاً (`CLEX01`). أرفض UTF-8 غير صالح في مصدر `.nano` (`CSRC01`) وعند حدود JSON/TOON/`module.json`/docgen. لا أسمّي النظام مُدَوَّلاً. راجع صفحة CLI المولَّدة لأن الأعلام تتغير أكثر مما ينبغي للنثر أن يدّعي.

`-pg` و`--profile-output` يغلّفان ثنائياً أصلياً بمحلل المضيف ويصدران JSON على stdout. ذلك المسار ليس `--profile-runtime` وليس `--pgo`. أوثّقه في [تحليل الأداء](07_performance_profiling.md).

