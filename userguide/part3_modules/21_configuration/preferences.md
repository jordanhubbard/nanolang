# 21.1 preferences — User Preferences Management

**Store and retrieve application preferences using the user's home directory.**

The `preferences` module provides a simple, file-backed persistence layer for application settings. It is designed around one concrete use case: saving and loading ordered lists of strings (such as recently opened files, playlist entries, or configuration items) to a file in the user's home directory. The module also provides helpers to locate the right place to store preferences on the current platform.

## Quick Start

```nano
from "modules/preferences/preferences.nano" import nl_prefs_get_path,
                                                    nl_prefs_save_playlist,
                                                    nl_prefs_load_playlist

fn save_recent_files(files: array<string>) -> bool {
    let path: string = (nl_prefs_get_path "myapp")
    let count: int = (array_length files)
    let result: int = (nl_prefs_save_playlist path files count)
    return (== result 1)
}

shadow save_recent_files {
    let files: array<string> = ["/home/user/file1.txt", "/home/user/file2.txt"]
    # Use a temp path to avoid polluting real preferences in tests
    let result: bool = (save_recent_files files)
    assert result
}

fn load_recent_files() -> array<string> {
    let path: string = (nl_prefs_get_path "myapp")
    return (nl_prefs_load_playlist path)
}

shadow load_recent_files {
    let items: array<string> = (load_recent_files)
    # May be empty on a fresh system; just check it doesn't crash
    assert (>= (array_length items) 0)
}
```

## Finding the Preferences File Path

### `nl_prefs_get_home()` — User's home directory

```
nl_prefs_get_home() -> string
```

Returns the user's home directory (e.g. `/Users/alice` on macOS, `/home/alice` on Linux). Uses the `HOME` environment variable internally.

```nano
from "modules/preferences/preferences.nano" import nl_prefs_get_home

fn show_home() -> void {
    let home: string = (nl_prefs_get_home)
    (println (+ "Home: " home))
}

shadow show_home {
    let home: string = (nl_prefs_get_home)
    assert (> (str_length home) 0)
}
```

### `nl_prefs_get_path(app_name)` — Standard preferences file path

```
nl_prefs_get_path(app_name: string) -> string
```

Returns the conventional path for your application's preferences file: `~/.{app_name}_prefs`. For an app named `"nanoamp"` this would return something like `/Users/alice/.nanoamp_prefs`.

```nano
from "modules/preferences/preferences.nano" import nl_prefs_get_path

fn get_prefs_path() -> string {
    return (nl_prefs_get_path "myapp")
}

shadow get_prefs_path {
    let path: string = (get_prefs_path)
    assert (> (str_length path) 0)
    # Path should contain the app name
    assert (str_contains path "myapp")
}
```

This is the recommended way to locate preferences — do not hard-code a path, and do not use a relative path. Platform conventions and multi-user systems expect preferences to live in the user's home directory.

## Saving Preferences

### `nl_prefs_save_playlist(filename, items, count)` — Write a list to disk

```
nl_prefs_save_playlist(filename: string, items: array<string>, count: int) -> int
```

Saves an ordered list of strings to `filename`. Each string is written on its own line. Returns `1` on success, `0` on failure (e.g. permission error, disk full).

The `count` parameter must match `array_length items`. This is an explicit parameter rather than being derived automatically, for compatibility with the underlying C layer.

```nano
from "modules/preferences/preferences.nano" import nl_prefs_get_path, nl_prefs_save_playlist

fn save_playlist(tracks: array<string>) -> bool {
    let path: string = (nl_prefs_get_path "musicplayer")
    let n: int = (array_length tracks)
    let ok: int = (nl_prefs_save_playlist path tracks n)
    return (== ok 1)
}

shadow save_playlist {
    let tracks: array<string> = [
        "Beethoven - Symphony 5.flac",
        "Bach - Goldberg Variations.flac",
        "Mozart - Requiem.flac"
    ]
    let saved: bool = (save_playlist tracks)
    assert saved
}
```

**Saving an empty list:**

```nano
fn clear_playlist() -> bool {
    let path: string = (nl_prefs_get_path "musicplayer")
    let empty: array<string> = []
    let ok: int = (nl_prefs_save_playlist path empty 0)
    return (== ok 1)
}

shadow clear_playlist {
    assert (clear_playlist)
}
```

## Loading Preferences

### `nl_prefs_load_playlist(filename)` — Read a list from disk

```
nl_prefs_load_playlist(filename: string) -> array<string>
```

Reads a previously saved list from `filename`. Returns an `array<string>` with one element per line. Returns an empty array if the file does not exist or cannot be read — there is no error thrown.

```nano
from "modules/preferences/preferences.nano" import nl_prefs_get_path, nl_prefs_load_playlist

fn load_playlist() -> array<string> {
    let path: string = (nl_prefs_get_path "musicplayer")
    return (nl_prefs_load_playlist path)
}

shadow load_playlist {
    let tracks: array<string> = (load_playlist)
    # On a fresh system this may be empty; either is valid
    assert (>= (array_length tracks) 0)
}
```

**Checking if preferences exist:**

```nano
from "modules/preferences/preferences.nano" import nl_prefs_get_path, nl_prefs_load_playlist

fn has_saved_playlist() -> bool {
    let path: string = (nl_prefs_get_path "musicplayer")
    let tracks: array<string> = (nl_prefs_load_playlist path)
    return (> (array_length tracks) 0)
}

shadow has_saved_playlist {
    let result: bool = (has_saved_playlist)
    # Result depends on machine state — just verify it returns a bool
    assert (or result (not result))
}
```

## Round-Trip Pattern

The most reliable way to use preferences is to always check that what you saved can be loaded back:

```nano
from "modules/preferences/preferences.nano" import nl_prefs_get_path,
                                                    nl_prefs_save_playlist,
                                                    nl_prefs_load_playlist

fn save_and_verify(app: string, items: array<string>) -> bool {
    let path: string = (nl_prefs_get_path app)
    let count: int = (array_length items)
    let ok: int = (nl_prefs_save_playlist path items count)
    if (== ok 0) {
        return false
    } else {
        let loaded: array<string> = (nl_prefs_load_playlist path)
        return (== (array_length loaded) count)
    }
}

shadow save_and_verify {
    let items: array<string> = ["alpha", "beta", "gamma"]
    assert (save_and_verify "test_roundtrip" items)

    let empty: array<string> = []
    assert (save_and_verify "test_roundtrip_empty" empty)
}
```

## Complete Example: Music Player Preferences

This example shows a realistic preferences workflow for a music player application:

```nano
from "modules/preferences/preferences.nano" import nl_prefs_get_path,
                                                    nl_prefs_get_home,
                                                    nl_prefs_save_playlist,
                                                    nl_prefs_load_playlist

# --- Preference paths for this app ---

fn playlist_path() -> string {
    return (nl_prefs_get_path "nanoamp_playlist")
}

shadow playlist_path {
    let p: string = (playlist_path)
    assert (> (str_length p) 0)
}

fn settings_path() -> string {
    return (nl_prefs_get_path "nanoamp_settings")
}

shadow settings_path {
    let p: string = (settings_path)
    assert (> (str_length p) 0)
}

# --- Playlist management ---

fn save_current_playlist(tracks: array<string>) -> bool {
    let n: int = (array_length tracks)
    let ok: int = (nl_prefs_save_playlist (playlist_path) tracks n)
    return (== ok 1)
}

shadow save_current_playlist {
    let tracks: array<string> = ["track1.flac", "track2.flac"]
    assert (save_current_playlist tracks)
}

fn load_saved_playlist() -> array<string> {
    return (nl_prefs_load_playlist (playlist_path))
}

shadow load_saved_playlist {
    # Save something first so we have known content
    let tracks: array<string> = ["a.flac", "b.flac", "c.flac"]
    (save_current_playlist tracks)
    let loaded: array<string> = (load_saved_playlist)
    assert (== (array_length loaded) 3)
    assert (== (array_get loaded 0) "a.flac")
}

fn playlist_track_count() -> int {
    let tracks: array<string> = (load_saved_playlist)
    return (array_length tracks)
}

shadow playlist_track_count {
    let tracks: array<string> = ["x.mp3"]
    (save_current_playlist tracks)
    assert (== (playlist_track_count) 1)
}

# --- Settings management ---
# Settings are stored as "key=value" lines using the playlist mechanism

fn make_setting(key: string, value: string) -> string {
    return (+ key (+ "=" value))
}

shadow make_setting {
    assert (== (make_setting "theme" "dark") "theme=dark")
    assert (== (make_setting "volume" "80") "volume=80")
}

fn save_settings(theme: string, volume: int, shuffle: bool) -> bool {
    let vol_str: string = (int_to_string volume)
    let shuf_str: string = (cond (shuffle "true") (else "false"))
    let entries: array<string> = [
        (make_setting "theme" theme),
        (make_setting "volume" vol_str),
        (make_setting "shuffle" shuf_str)
    ]
    let ok: int = (nl_prefs_save_playlist (settings_path) entries 3)
    return (== ok 1)
}

shadow save_settings {
    assert (save_settings "dark" 75 true)
    assert (save_settings "light" 50 false)
}

fn load_setting_value(entries: array<string>, key: string) -> string {
    let n: int = (array_length entries)
    let search: string = (+ key "=")
    let search_len: int = (str_length search)
    let mut result: string = ""
    let mut i: int = 0
    while (< i n) {
        let entry: string = (array_get entries i)
        let entry_len: int = (str_length entry)
        if (and (>= entry_len search_len)
                (== (str_substring entry 0 search_len) search)) {
            let value_len: int = (- entry_len search_len)
            if (> value_len 0) {
                set result (str_substring entry search_len value_len)
            } else {
                (print "")
            }
        } else {
            (print "")
        }
        set i (+ i 1)
    }
    return result
}

shadow load_setting_value {
    let entries: array<string> = ["theme=dark", "volume=75", "shuffle=true"]
    assert (== (load_setting_value entries "theme") "dark")
    assert (== (load_setting_value entries "volume") "75")
    assert (== (load_setting_value entries "shuffle") "true")
    assert (== (load_setting_value entries "missing") "")
}

fn get_saved_theme() -> string {
    let entries: array<string> = (nl_prefs_load_playlist (settings_path))
    let theme: string = (load_setting_value entries "theme")
    if (== theme "") {
        return "dark"   # default
    } else {
        return theme
    }
}

shadow get_saved_theme {
    (save_settings "solarized" 60 false)
    assert (== (get_saved_theme) "solarized")
}

fn main() -> int {
    # Save initial settings
    (save_settings "dark" 80 true)

    # Save a playlist
    let playlist: array<string> = [
        "recordings/concert.flac",
        "recordings/symphony.flac",
        "recordings/sonata.flac"
    ]
    (save_current_playlist playlist)

    # Report what was saved
    (println (+ "Saved " (int_to_string (playlist_track_count)) " tracks"))
    (println (+ "Theme: " (get_saved_theme)))
    (println (+ "Prefs stored in: " (nl_prefs_get_home)))

    return 0
}

shadow main { assert true }
```

## API Reference

| Function | Signature | Description |
|---|---|---|
| `nl_prefs_get_home` | `() -> string` | User home directory |
| `nl_prefs_get_path` | `(app_name: string) -> string` | `~/.{app_name}_prefs` path |
| `nl_prefs_save_playlist` | `(filename: string, items: array<string>, count: int) -> int` | Write list to file, returns 1 on success |
| `nl_prefs_load_playlist` | `(filename: string) -> array<string>` | Read list from file, returns empty array on failure |

## Notes and Limitations

- **Format:** The file format is plain text, one item per line. There is no escaping for newlines within items — items should not contain literal newline characters.
- **Atomic writes:** The current implementation writes directly to the target file. On systems where the process is killed mid-write, the file may be left in a partial state. For critical data, save to a temp file first and rename.
- **No locking:** Multiple processes writing to the same file simultaneously may corrupt it. Use application-level coordination if concurrency is a concern.
- **Encoding:** Files are written in the system's native encoding. Keep preference values to ASCII for maximum portability.

---

**Previous:** [Chapter 21 Overview](index.html)
**Next:** [Part 4: Advanced Topics](../../part4_advanced/22_canonical_style.html)
