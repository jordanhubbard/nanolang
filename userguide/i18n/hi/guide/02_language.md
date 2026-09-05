---
title: भाषा
machine_generated: true
reviewed: false
lang: hi
---

> यह पृष्ठ मशीन-जनित मसौदा है और अभी मानव समीक्षा नहीं हुई है। कोड उदाहरण, आदेश और पहचानकर्ता अंग्रेज़ी में रहते हैं।

# भाषा

मैं साधारण वाक्यविन्यास स्पष्ट रखता हूँ। मैं कुछ सुविधाएँ स्वीकार करता हूँ, पर प्राथमिकता या फलन हस्ताक्षर नहीं छिपाता।

## कॉल और ऑपरेटर

कॉल कोष्ठक में और उपसर्ग हैं:

```nano
(println "ready")
(distance x y)
(math.clamp value low high)
```

ऑपरेटर उपसर्ग और इनफिक्स दोनों स्वीकार करते हैं:

```nano
let prefix = (+ 2 (* 3 4))
let infix = 2 + (3 * 4)
```

सभी इनफिक्स द्विआधारी ऑपरेटरों की प्राथमिकता समान है और वे बाएँ से दाएँ जुड़ते हैं। `2 + 3 * 4` का अर्थ `(2 + 3) * 4` है। जब समूहन मायने रखे, कोष्ठक जोड़ें।

## बंधन

बंधन अपरिवर्तनीय हैं जब तक `mut` न हो। परिवर्तन `set` उपयोग करता है:

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

जब आरंभक प्रकार निर्धारित करे तब स्थानीय एनोटेशन वैकल्पिक हैं:

```nano
let count = 3
let name = "Ada"
```

खाली संग्रहों, जेनेरिक मानों, विदेशी हैंडल, संसाधन मानों, और जहाँ अनुमान आशय धुंधला करे उन सीमाओं पर एनोटेट करें। फलन पैरामीटर और वापसी प्रकार स्पष्ट रहते हैं।

## अदिश प्रकार

सामान्य अदिश प्रकार `int`, `u8` या `byte`, `float`, `bool`, `string`, `bstring`, और `void` हैं। सरणियाँ, टपल, नामित रिकॉर्ड, एनम, यूनियन, फलन प्रकार, जेनेरिक प्रकार, खुले रिकॉर्ड, और अपारदर्शी विदेशी प्रकार उस समुच्चय का विस्तार करते हैं।

## नियंत्रण प्रवाह

`if` से `else` छूट सकता है:

```nano
if needs_redraw {
    (draw scene)
}
```

`cond` व्यंजक मानों में से चुनता है और `else` माँगता है:

```nano
let sign = (cond
    ((< value 0) -1)
    ((> value 0) 1)
    (else 0)
)
```

लूप `while` या `for` उपयोग करते हैं:

```nano
while (< index count) {
    set index (+ index 1)
}

for index in (range 0 count) {
    (visit index)
}
```

लूप के अंदर `break` और `continue` उपलब्ध हैं।

## फलन

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

मैं पुनरावृत्ति, प्रथम-श्रेणी फलन, क्लोज़र, जेनेरिक, `requires` से पूर्वशर्तें, और `ensures` से उत्तरशर्तें समर्थित करता हूँ। संविदाएँ निष्पादनों के जाँचे गुण हैं; वे औपचारिक प्रमाण नहीं हैं।

## टिप्पणियाँ

```nano
# ordinary comment
// accepted line comment
/* block comment */
/// documentation comment
```

साधारण टिप्पणी के लिए `#` और टूलिंग द्वारा उपभोग किए जाने वाले दस्तावेज़ के लिए `///` उपयोग करें।

