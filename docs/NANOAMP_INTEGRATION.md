# NanoAmp Integration Guide: Preferences & Directory Browser

This guide shows how to integrate the preferences and filesystem modules into NanoAmp for persistent playlists and directory browsing.

## Modules Available

### 1. Preferences Module (`modules/preferences/`)

Save and load application preferences to text files.

**Functions:**
- `nl_prefs_save_playlist(filename, items, count)` - Save string array to file
- `nl_prefs_load_playlist(filename)` - Load strings from file into array
- `nl_prefs_get_home()` - Get user's home directory  
- `nl_prefs_get_path(app_name)` - Build path like `~/.nanoamp_prefs`

### 2. Filesystem Module (`modules/filesystem/`)

Directory and file operations.

**Functions:**
- `nl_fs_list_files(path, extension)` - List files with optional filter
- `nl_fs_list_dirs(path)` - List subdirectories
- `nl_fs_is_directory(path)` - Check if path is directory
- `nl_fs_file_exists(path)` - Check if file exists
- `nl_fs_join_path(dir, filename)` - Join path components

## Integration Steps

### Step 1: Add Imports

```nano
import "modules/filesystem/filesystem.nano"
import "modules/preferences/preferences.nano"
```

### Step 2: Load Playlist on Startup

Replace command-line directory loading with preferences:

```nano
# Get preference file path
let pref_path: string = (nl_prefs_get_path "nanoamp")

# Load saved playlist
let mut playlist: array<string> = (nl_prefs_load_playlist pref_path)
let mut playlist_count: int = (array_length playlist)

if (== playlist_count 0) {
    (println "No saved playlist - starting empty")
    (println "Use Browse button to add music")
} else {
    (print "✓ Loaded ")
    (print playlist_count)
    (println " tracks from saved playlist")
}
```

### Step 3: Save Playlist on Exit

Add before cleanup code:

```nano
# Save playlist
(println "Saving playlist...")
let save_result: int = (nl_prefs_save_playlist pref_path playlist playlist_count)
if (== save_result 1) {
    (println "✓ Playlist saved successfully")
} else {
    (println "✗ Failed to save playlist")
}
```

### Step 4: Add Directory Browser UI Mode

Add mode constants:

```nano
let MODE_PLAYER: int = 0
let MODE_BROWSER: int = 1
```

Add state variables:

```nano
let mut ui_mode: int = MODE_PLAYER
let mut browser_path: string = (nl_prefs_get_home)
let mut browser_scroll: int = 0
```

### Step 5: Add Browse Button

In player mode, add button:

```nano
if (== (nl_ui_button renderer font "Browse..." 350 200 90 30) 1) {
    set ui_mode MODE_BROWSER
    set browser_path (nl_prefs_get_home)
    set browser_scroll 0
}
```

### Step 6: Implement Browser Mode

In main loop, add browser mode rendering:

```nano
if (== ui_mode MODE_BROWSER) {
    # Title
    (nl_ui_label renderer font "Browse for Music Directory" 20 20 200 200 255 255)
    
    # Current path
    (nl_ui_label renderer small_font browser_path 20 50 180 180 180 255)
    
    # ".." button to go up
    if (== (nl_ui_button renderer font ".." 20 80 80 30) 1) {
        let parent: string = (nl_fs_join_path browser_path "..")
        set browser_path parent
        set browser_scroll 0
    }
    
    # "Select This Folder" button
    if (== (nl_ui_button renderer font "Select This Folder" 110 80 180 30) 1) {
        # Add all MP3 files from this directory
        let files: array<string> = (nl_fs_list_files browser_path ".mp3")
        let file_count: int = (array_length files)
        
        let mut i: int = 0
        while (< i file_count) {
            let filename: string = (at files i)
            let full_path: string = (nl_fs_join_path browser_path filename)
            (array_push playlist full_path)
            set i (+ i 1)
        }
        
        (print "✓ Added ")
        (print file_count)
        (println " MP3 files")
        set ui_mode MODE_PLAYER
    }
    
    # "Cancel" button
    if (== (nl_ui_button renderer font "Cancel" 300 80 80 30) 1) {
        set ui_mode MODE_PLAYER
    }
    
    # List directories
    let dirs: array<string> = (nl_fs_list_dirs browser_path)
    let dir_count: int = (array_length dirs)
    
    if (== dir_count 0) {
        (nl_ui_label renderer small_font "No subdirectories" 30 130 150 150 150 255)
    } else {
        # Draw directory buttons
        let mut dir_y: int = 130
        let mut dir_i: int = browser_scroll
        
        while (< dir_i dir_count) {
            let dir_name: string = (at dirs dir_i)
            
            # Directory button - click to navigate
            if (== (nl_ui_button renderer small_font dir_name 20 dir_y 560 25) 1) {
                let new_path: string = (nl_fs_join_path browser_path dir_name)
                set browser_path new_path
                set browser_scroll 0
            }
            
            set dir_y (+ dir_y 28)
            set dir_i (+ dir_i 1)
            
            # Stop at bottom of viewport
            if (> dir_y 530) {
                set dir_i dir_count  # Break loop
            }
        }
    }
    
} else {
    # Regular player mode UI...
}
```

### Step 7: Update Window Height

To accommodate browser mode:

```nano
let WINDOW_HEIGHT: int = 600  # Was 550
```

## Example Integration

See `examples/sdl_nanoamp.nano` for a working example (once integrated).

## Testing

1. **First run:** `./bin/sdl_nanoamp` - starts empty
2. **Click Browse:** Navigate to music directory
3. **Select folder:** Adds MP3s to playlist
4. **Play music:** Test playback
5. **Exit:** Playlist saved to `~/.nanoamp_prefs`
6. **Run again:** Playlist automatically restored

## Benefits

✅ **Persistent playlists** - Music library saved between sessions  
✅ **No command-line args** - Simpler UX  
✅ **Directory browsing** - Visual file selection  
✅ **Cross-platform** - Works on macOS, Linux, Windows

## File Format

Preferences are stored as plain text, one path per line:

```
/Users/user/Music/song1.mp3
/Users/user/Music/song2.mp3
/Users/user/Music/song3.mp3
```

Easy to edit manually if needed!
