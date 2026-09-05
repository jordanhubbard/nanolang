---
title: औज़ार और बैकएंड
machine_generated: true
reviewed: false
lang: hi
---

> यह पृष्ठ मशीन-जनित मसौदा है और अभी मानव समीक्षा नहीं हुई है। कोड उदाहरण, आदेश और पहचानकर्ता अंग्रेज़ी में रहते हैं।

# औज़ार और बैकएंड

मेरे कई निष्पादन पथ हैं। वे वाक्यविन्यास साझा करते हैं पर पूर्ण फीचर समता नहीं।

## औज़ार

| Tool | Purpose |
| --- | --- |
| `bin/nanoc` | स्रोत संकलित करें और विश्लेषण या बैकएंड विकल्प उजागर करें |
| `bin/nano` | स्रोत फ़ाइल व्याख्या करें |
| `bin/nanolang-repl` | इंटरैक्टिव मूल्यांकन |
| `nano-fmt` | स्रोत प्रारूपित करें |
| `nano-docs` | स्थानीय दस्तावेज़ खोजें |
| `bin/nanolang-lsp` | Language Server Protocol समर्थन |
| `bin/nanolang-dap` | Debug Adapter Protocol समर्थन |
| `bin/nano_virt` | स्रोत को NanoISA बाइटकोड तक उतारें |
| `bin/nano_vm` | NanoISA बाइटकोड चलाएँ |
| `bin/nano_vmd` | NanoVM डेमन चलाएँ |
| `bin/nano_cop` | समर्थित विदेशी कॉल सह-प्रक्रिया में अलग करें |

जहाँ `--help` हो वहाँ प्रत्येक औज़ार चलाएँ। जनित [Compiler CLI](../generated/cli.md) पृष्ठ कंपाइलर का वर्तमान सहायता पाठ दर्ज करता है।

## बैकएंड

| Output | Command | Boundary |
| --- | --- | --- |
| Native executable | `nanoc source.nano -o program` | जनित C से उत्पादन पथ |
| C source | `nanoc source.nano --target c -o program.c` | स्वतंत्र जनित C |
| NanoISA | `nano_virt source.nano --emit-nvm -o program.nvm` | साझा टंकित VM निरूपण |
| PTX | `nanoc source.nano --target ptx -o program.ptx` | GPU कर्नेल उपसमुच्चय |
| OpenCL C | `nanoc source.nano --target opencl -o program.cl` | GPU कर्नेल उपसमुच्चय |
| RISC-V assembly | `nanoc source.nano --target riscv -o program.s` | प्रायोगिक उपसमुच्चय |
| NanoISA | `nano_virt source.nano -o program.nvm` | पृथक FFI समर्थन वाला वर्चुअल-मशीन पथ |

भविष्य के LLVM और WebAssembly लक्ष्य मेरे स्रोत AST से शाखा करने के बजाय NanoISA से अनुवाद करते हैं। NanoISA से C11 एक बंद-उपसमुच्चय स्पाइक है (`nvm2c`, `make test-nvm2c`), `nano_virt` का डिफ़ॉल्ट मूल रैपर नहीं जो अभी VM एम्बेड करता है।

## निदान

मशीन-मुखी निदान में JSON और TOON रूप शामिल हैं। उपयोगी कंपाइलर विकल्प `--llm-diags-json`, `--llm-diags-toon`, `--json-errors`, `--emit-typed-ast-json`, और `--reflect` हैं। मैं `--locale <tag>` से प्रक्रिया लोकैल चुनता हूँ (फिर `NANO_LOCALE`, `LC_ALL`, `LANG`, अन्यथा `en`) और `--print-locale` से अक्ष छापता हूँ। यह मशीन-पठनीय निदान आउटपुट नहीं बदलता। कैटलॉग मौजूद हैं: मानव stderr UTF-8 कैटलॉग देखता है; JSON/TOON अंग्रेज़ी रहते हैं; `--locale` / `NANO_CATALOG_DIR`। पाइपलाइन कंपाइलर निदान स्थिर ID उपयोग करते हैं (`CIO01`, `CSRC01`, `L0003`, और `src/diag_id.c` का शेष)। `emit_context_error` से जाने वाले टाइपचेकर शीर्षक अद्वितीय `E001`–`E035` उपयोग करते हैं। पहचानकर्ता ASCII हैं; गैर-ASCII बंद विफल होता है (`CLEX01`)। मैं `.nano` स्रोत में अमान्य UTF-8 अस्वीकार करता हूँ (`CSRC01`) और JSON/TOON/`module.json`/docgen सीमाओं पर भी। मैं इस तंत्र को अंतरराष्ट्रीयकृत नहीं कहता। जनित CLI पृष्ठ देखें क्योंकि फ्लैग गद्य जितना मानने को तैयार है उससे अधिक बार बदलते हैं।

`-pg` और `--profile-output` मूल बाइनरी को मेज़बान प्रोफ़ाइलर से लपेटते हैं और stdout पर JSON छोड़ते हैं। वह पथ `--profile-runtime` नहीं है और `--pgo` नहीं है। मैं इसे [प्रदर्शन प्रोफ़ाइल](07_performance_profiling.md) में दस्तावेज़ करता हूँ।

