---
title: प्रदर्शन प्रोफ़ाइल
machine_generated: true
reviewed: false
lang: hi
---

> यह पृष्ठ मशीन-जनित मसौदा है और अभी मानव समीक्षा नहीं हुई है। कोड उदाहरण, आदेश और पहचानकर्ता अंग्रेज़ी में रहते हैं।

# प्रदर्शन प्रोफ़ाइल

मैं `-pg` से संकलित मूल बाइनरी लपेटता हूँ ताकि चलाने पर मेज़बान प्रोफ़ाइल एकत्र हो और LLM पढ़ सके ऐसा JSON छपे। मैं गति वृद्धि की गारंटी नहीं देता। जब दूसरी प्रोफ़ाइल और मेरे परीक्षण दिखाएँ कि मदद हुई तभी अनुकूलन रखता हूँ।

प्रामाणिक फ़ील्ड सूची, पुनः-प्रवेश पर्यावरण चर, और `/tmp` आर्टिफ़ैक्ट
नाम इसमें हैं
[docs/PERFORMANCE_MONITORING.md](https://github.com/jordanhubbard/nanolang/blob/main/docs/PERFORMANCE_MONITORING.md)।

## प्रोफ़ाइल पकड़ें

```bash
./bin/nanoc program.nano -o bin/program -pg --profile-output profile.json
./bin/program
```

पहली प्रक्रिया समन्वयक है (`_nl_run_with_profiling`)। यह अपनी प्रक्रिया ID से
अद्वितीय नामित आर्टिफ़ैक्ट बनाती है, मापे गए कार्यभार को बच्चे में चलाती है,
प्रतीक्षा करती है, संग्राहक आउटपुट JSON में बदलती है, और अस्थायी फ़ाइलें
हटाती है। समवर्ती चलान एक ही `/tmp` पथ साझा नहीं करते।

JSON `profile.json` में लिखा जाता है और **stdout** पर भी (दो बैनर
पंक्तियों के बीच)। stderr पुनर्निर्देशित करने से यह नहीं पकड़ेगा।

## वे फ्लैग जो `-pg` नहीं हैं

| Flag | Role |
| --- | --- |
| `-pg` / `--profile-output` | OS संग्राहक + JSON। मूल `main` `_nl_run_with_profiling` बन जाता है। |
| `--profile` | जनित C में समय हुक। stderr पर तालिका। |
| `--trace` | जनित C में फलन-कॉल ट्रेस हुक। stderr पर इंडेंट `-> fn` / `<- fn`। |
| `--profile-runtime` | `.nano.prof` संक्षिप्त स्टैक भी लिखें। केवल मूल बैकएंड। |
| `--pgo <file>` | `.nano.prof` से इनलाइन। `-pg` JSON उपयोग नहीं। |

`-pg` के साथ जुड़े C फ्लैग: `-pg -g -fno-omit-frame-pointer -fno-optimize-sibling-calls`।

## प्लेटफ़ॉर्म सीमा

| Platform | Collector | Measurement |
| --- | --- | --- |
| macOS with full Xcode | `xctrace` Time Profiler | चलाए बच्चे का नमूना |
| macOS fallback | `sample` | समकालित बच्चे का आवधिक नमूना |
| Linux | `gprofng collect app` | बच्चे प्रक्रिया में उपकरण संग्रह |
| Other OS | none | मैं छापता हूँ कि प्रोफ़ाइल समर्थित नहीं और कार्यक्रम चलाता हूँ |

मैं xctrace तभी उपलब्ध मानता हूँ जब `which xctrace` और `xctrace version`
दोनों सफल हों। अकेला Command Line Tools पर्याप्त नहीं।

macOS पर `sample` फ़ॉलबैक पहले कार्यभार बच्चा बनाता है और उसे पाइप पर
रोकता है (`_NL_PROFILING_ACTIVE`, `_NL_PROFILING_CWD`)। एक सैंपलर बच्चा
उस PID से जुड़ता है (`sample <pid> 60 -f /tmp/nanolang_sample_<pid>.txt -mayDie`),
फिर समन्वयक कार्यभार छोड़ता है। छोटे चलान `sample` जुड़ने से पहले खत्म हो सकते हैं।

Linux पर पुनः-प्रवेश `_NL_PROFILING_CHILD` या `libgp-collector` वाला
`LD_PRELOAD` उपयोग करता है। यदि `gprofng` गायब है, बच्चा बिना संग्राहक चलता है।
gprofng उपकरण संग्रह है। JSON फ़ील्ड `profile_type` हर OS पर अभी भी
स्ट्रिंग `"sampling"` है; वह ऐतिहासिक लेबल है, यह दावा नहीं कि Linux नमूना लेता है।

## JSON जो मैं वास्तव में छोड़ता हूँ

प्रत्येक हॉटस्पॉट में केवल `function`, `samples`, और `pct_time` हैं। मैं स्रोत
स्थान या प्रति-कॉल माइक्रोसेकंड नहीं छोड़ता।

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

`tool` `"gprofng"`, `"xctrace"`, या `"sample"` है। Linux पर `samples`
अनन्य प्रतिशत गुना दस है, सच्चा नमूना गिनती नहीं। xctrace पर मैं `nl_` से
शुरू होने वाले शीर्ष 20 नाम छोड़ता हूँ।

प्रोफ़ाइल को एक मशीन पर एक कार्यभार का माप मानें।

## LLM से समायोजित करें

1. `-pg --profile-output` से संकलित करें।
2. प्रतिनिधि कार्यभार चलाएँ, एक-पंक्ति शैडो नहीं।
3. LLM को JSON, संबंधित स्रोत, और वह आदेश दें जिससे कार्यभार बना।
   मापे हॉटस्पॉट से जुड़ा एक परिवर्तन माँगें।
4. परीक्षण चलाएँ, फिर उसी कार्यभार को फिर प्रोफ़ाइल करें।
5. परिवर्तन तभी रखें जब परीक्षण पास रहें और संख्याएँ सुधरें।

LLM अनुकूलन सुझा सकता है; प्रोफ़ाइल उसके प्रभाव का परीक्षण कर सकती है। कोई
दूसरे का स्थानापन्न नहीं। गति बताते समय `-pg` **बिना** दीवार-घड़ी तुलना करें।

ट्रेसिंग और प्रोफ़ाइलिंग एक स्टार्टअप हुक साझा करते हैं। `--profile` बाइनरी
`NANO_PROFILE` मानती हैं, और `--trace` बाइनरी `NANO_TRACE`; किसी को `0`
करने से बिना पुनर्संकलन रनटाइम पर वह हुक बंद होता है, और बंद हुक प्रति-घटना
काम नहीं करता।

`--pgo` `--profile-runtime` से `.nano.prof` पढ़ता है। वह फ़ाइल `-pg` JSON
से PGO इनपुट नहीं है।

