---
title: डेटा और त्रुटियाँ
machine_generated: true
reviewed: false
lang: hi
---

> यह पृष्ठ मशीन-जनित मसौदा है और अभी मानव समीक्षा नहीं हुई है। कोड उदाहरण, आदेश और पहचानकर्ता अंग्रेज़ी में रहते हैं।

# डेटा और त्रुटियाँ

मैं संग्रहों और डोमेन अवस्थाओं को स्पष्ट दर्शाता हूँ। ऐसा प्रकार चुनें जिससे अवैध अवस्थाएँ व्यक्त करना कठिन हो।

## सरणियाँ और टपल

```nano
let values: array<int> = [10, 20, 30]
let first = (at values 0)
(array_set values 1 25)

let pair: (int, string) = (7, "seven")
let number = pair.0
```

सरणी सीमा-जाँचित हैं। कुछ सरणी फलन भंडारण बदलते हैं और `void` लौटाते हैं; अन्य नई सरणी लौटाते हैं। परिचित नाम से स्वामित्व न अनुमान करें; [Builtins](../generated/builtins.md) देखें।

## स्ट्रक्ट और एनम

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

नामित प्रकार `UpperCamelCase` उपयोग करते हैं। मान और फलन `snake_case` उपयोग करते हैं।

## यूनियन और मैच

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

कॉलर संभाल सकें ऐसी विफलताओं के लिए `Result<T, E>` या अन्य स्पष्ट यूनियन उपयोग करें। जहाँ आप पुनर्प्राप्त कर सकें या संदर्भ जोड़ सकें वहाँ वेरिएंट मैच करें। पोस्टफिक्स `?` ऑपरेटर संगत परिणाम त्रुटियाँ प्रचारित करता है; सफाई या संदर्भ जब मायने रखे तब स्पष्ट `match` उपयोग करें।

## स्ट्रिंग और द्विआधारी स्ट्रिंग

`string` पाठ है। `bstring` लंबाई-स्पष्ट द्विआधारी डेटा है और शून्य बाइट रख सकता है। रूपांतरण और यूनिकोड व्यवहार प्रत्येक बिगिल्टिन या मॉड्यूल पर दस्तावेज़ी हैं। यह न मानें कि बाइट अनुक्रमणिकाएँ यूनिकोड वर्ण अनुक्रमणिकाएँ हैं।

## संग्रह

बिगिल्टिन सरणियाँ और हैश मैप C-आधारित संग्रह मॉड्यूल से अलग हैं। जनित [Builtins](../generated/builtins.md) पृष्ठ सटीक बिगिल्टिन वर्तनी सूचीबद्ध करता है। जनित [Modules](../generated/modules.md) पृष्ठ मॉड्यूल घोषणाएँ और मूल निर्माण सीमाएँ सूचीबद्ध करता है।

