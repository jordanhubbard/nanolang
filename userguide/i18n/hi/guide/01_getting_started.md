---
title: शुरुआत
machine_generated: true
reviewed: false
lang: hi
---

> यह पृष्ठ मशीन-जनित मसौदा है और अभी मानव समीक्षा नहीं हुई है। कोड उदाहरण, आदेश और पहचानकर्ता अंग्रेज़ी में रहते हैं।

# शुरुआत

मैं अभी Unix-जैसे तंत्रों पर बनता हूँ। Windows उपयोगकर्ताओं को WSL2 उपयोग करना चाहिए।

## निर्माण

आपको C कंपाइलर, Make, Git, और pkg-config चाहिए। क्लोन करें और बनाएँ:

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make build bootstrap3
./bin/nanoc --help
```

`bin/nanoc` मेरा कंपाइलर है। `bin/nano` मेरा ट्री-वॉकिंग इंटरप्रेटर है।

## पहला कार्यक्रम

`hello.nano` बनाएँ:

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

इसे संकलित करें और चलाएँ:

```bash
./bin/nanoc hello.nano -o hello
./hello
```

`-o hello` के साथ, मैं सत्यापित NanoISA को अपने C अनुवादक और मेज़बान C कंपाइलर से मूल निष्पादन योग्य फ़ाइल में बदलता हूँ। फ़ाइल प्रकाशित करने से पहले मैं चुने हुए शैडो परीक्षण अलग से चलाता हूँ; कार्यक्रम शुरू होने पर वे दोबारा नहीं चलते। किसी शैडो परीक्षण की विफलता प्रकाशन रोकती है।

आउटपुट या लक्ष्य विकल्प दिए बिना, मैं इसके बजाय पोर्टेबल बाइटकोड प्रकाशित करता हूँ:

```bash
./bin/nanoc hello.nano
./bin/nano_vm hello.nvm
```

## इंटरप्रेटर

मूल एक्ज़ीक्यूटेबल बनाए बिना स्रोत फ़ाइल चलाएँ:

```bash
./bin/nano hello.nano
```

इंटरप्रेटर और संकलित बैकएंड भाषा साझा करते हैं, पर उनके कार्यान्वयन सीमाएँ एक समान नहीं हैं। जिस बैकएंड को आप भेजना चाहते हैं, उसी का परीक्षण करें।

## परियोजना लेआउट

छोटे कार्यक्रम के लिए एक `.nano` फ़ाइल पर्याप्त है। पैकेज `nano.toml` उपयोग कर सकते हैं:

```text
my_program/
  nano.toml
  main.nano
  src/
```

पूर्ण लेआउट के लिए [`examples/hello_pkg`](https://github.com/jordanhubbard/nanolang/tree/main/examples/hello_pkg) और [`examples/large_project`](https://github.com/jordanhubbard/nanolang/tree/main/examples/large_project) देखें।

## आगे

कॉल, ऑपरेटर, बंधन, फलन और नियंत्रण प्रवाह के लिए [भाषा](02_language.md) पढ़ें। इसके बाद [सुरक्षित रनटाइम](08_secure_runtime.md) पढ़ें कि कार्यक्रम को मेज़बान तक पहुँचने की अनुमति कैसे मिलती है।
