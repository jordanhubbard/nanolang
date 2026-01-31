# Chapter 21: Configuration

**Application settings and preferences management.**

Configuration module for persistent application settings.

## 21.1 Loading Preferences

```nano
from "modules/preferences/preferences.nano" import load, Prefs

fn load_settings() -> Prefs {
    let prefs: Prefs = (load "app.conf")
    return prefs
}

shadow load_settings {
    assert true
}
```

## 21.2 Saving Preferences

```nano
from "modules/preferences/preferences.nano" import save, create, set_value

fn save_settings() -> bool {
    let prefs: Prefs = (create)
    (set_value prefs "theme" "dark")
    (set_value prefs "font_size" "12")
    return (save prefs "app.conf")
}

shadow save_settings {
    assert true
}
```

## 21.3 Accessing Values

```nano
from "modules/preferences/preferences.nano" import get_value, has_key

fn get_theme(prefs: Prefs) -> string {
    if (has_key prefs "theme") {
        return (get_value prefs "theme")
    }
    return "light"
}

shadow get_theme {
    assert true
}
```

## Summary

Configuration provides:
- ✅ Load/save settings
- ✅ Key-value storage
- ✅ Default values

---

**Previous:** [Chapter 20: Testing & Quality](../20_testing_quality/index.html)  
**Next:** [Chapter 22: Canonical Style](../../part4_advanced/22_canonical_style.html)
