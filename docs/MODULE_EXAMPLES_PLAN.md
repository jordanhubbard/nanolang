# Module Examples Plan - Practical Demonstrations for Every Module

## Executive Summary

**Problem:** 8 modules have NO examples, and several others have only basic API demonstrations rather than practical, problem-solving examples.

**Impact:** Developers can't see how to use modules in real applications.

**Solution:** Create practical, problem-first examples for every module that demonstrate solving real-world problems.

## Audit Results

### Modules WITHOUT Any Examples (8 modules)

| Module | Purpose | Priority | Example Needed |
|--------|---------|----------|----------------|
| `audio_helpers` | Audio processing | Medium | Waveform generator with effects |
| `http_server` | Web server | **HIGH** | ✅ Examples exist! (4 files) |
| `math_ext` | Extended math | High | Scientific calculator |
| `pt2_audio` | ProTracker audio | Medium | MOD file player |
| `pt2_state` | Tracker state | Low | (Integrate with pt2_audio) |
| `stdio` | File I/O | **HIGH** | File processor (read/write/transform) |
| `unicode` | Unicode strings | Medium | Internationalization demo |
| `vector2d` | 2D vector math | High | Physics simulation |

### Modules WITH Examples But Need Enhancement

| Module | Current Examples | Gap | Enhancement Needed |
|--------|------------------|-----|-------------------|
| `curl` | 2 (basic) | No practical REST client | Weather API client, GitHub API demo |
| `sqlite` | 1 (simple) | No real application | Contact manager or todo list |
| `event` | 1 | Unknown if practical | Event-driven chat or notification system |
| `uv` | 1 | Unknown if practical | Async file processor or concurrent downloader |

### Modules WITH Good Examples (24 modules) ✅

These have multiple practical examples:
- `sdl`, `sdl_helpers`, `sdl_ttf` (21+ examples each)
- `bullet` (4 physics examples)
- `audio_viz` (4 visualization examples)
- `ncurses` (3 TUI examples)
- `onnx` (3 ML inference examples)
- `glew`, `glfw`, `opengl` (3D graphics examples)
- `ui_widgets` (10 UI examples)

---

## Priority 1: Critical Missing Examples

### 1. stdio - File Operations (`stdio_file_processor.nano`)
**Problem:** Process a log file - read lines, filter, transform, write results

**Demonstrates:**
- File reading (line by line)
- File writing (buffered output)
- Error handling (file not found, permissions)
- Text transformation pipeline
- Resource cleanup

**Example Pipeline:**
```
Input: access.log (web server logs)
→ Read file line by line
→ Filter (status code >= 400)
→ Extract (timestamp, IP, URL, status)
→ Transform (anonymize IPs, categorize errors)
→ Write to errors.csv

Real-world: Log analysis, data migration, file format conversion
```

**Code Structure:**
```nano
import "modules/stdio/stdio.nano" as IO

struct LogEntry {
    timestamp: string,
    ip: string,
    url: string,
    status: int
}

fn parse_log_line(line: string) -> LogEntry { ... }
fn anonymize_ip(ip: string) -> string { ... }
fn categorize_error(status: int) -> string { ... }

fn main() -> int {
    /* Open input file */
    let input: IO.File = (IO.open "access.log" "r")
    let output: IO.File = (IO.open "errors.csv" "w")
    
    /* Process line by line */
    while (not (IO.eof input)) {
        let line: string = (IO.read_line input)
        let entry: LogEntry = (parse_log_line line)
        
        if (>= entry.status 400) {
            let anon_ip: string = (anonymize_ip entry.ip)
            let category: string = (categorize_error entry.status)
            (IO.write_line output (+ anon_ip (+ "," (+ entry.url (+ "," category)))))
        }
    }
    
    (IO.close input)
    (IO.close output)
    return 0
}
```

---

### 2. math_ext - Scientific Calculator (`math_ext_scientific_calc.nano`)
**Problem:** Implement a scientific calculator with all extended math functions

**Demonstrates:**
- Trigonometric functions (sin, cos, tan, asin, acos, atan)
- Hyperbolic functions (sinh, cosh, tanh)
- Exponential/logarithmic (exp, log, log10, pow)
- Special functions (factorial, combinations, permutations)
- Practical calculations (compound interest, projectile motion, wave equations)

**Example Calculations:**
```
1. Projectile Motion (physics)
   - Given: velocity, angle, gravity
   - Calculate: max height, range, time of flight
   - Uses: sin, cos, pow, sqrt

2. Compound Interest (finance)
   - Given: principal, rate, time, compounds per year
   - Calculate: future value
   - Uses: pow, exp

3. Wave Interference (signal processing)
   - Given: two sine waves
   - Calculate: resultant amplitude, phase
   - Uses: sin, cos, atan2, sqrt

4. Statistical Analysis
   - Given: data set
   - Calculate: mean, std dev, normal distribution
   - Uses: sqrt, exp, pow, log
```

---

### 3. vector2d - Physics Simulation (`vector2d_physics_demo.nano`)
**Problem:** Simulate bouncing balls with gravity, collisions, and friction

**Demonstrates:**
- Vector operations (add, subtract, scale, normalize)
- Dot product (collision detection)
- Distance calculations
- Physics integration (velocity, acceleration)
- Practical game physics

**Example Simulation:**
```
Setup:
- 10 balls with random positions and velocities
- Gravity pulling down
- Walls with bounce (coefficient of restitution)
- Ball-to-ball collisions
- Friction slowing balls

Physics Loop:
1. Apply gravity (acceleration vector)
2. Update velocity (integrate acceleration)
3. Update position (integrate velocity)
4. Check wall collisions (reflect velocity vector)
5. Check ball-ball collisions (elastic collision formula)
6. Apply friction (scale velocity)
7. Render (SDL visualization)

Real-world: Game physics, particle systems, robotics
```

---

## Priority 2: Enhanced Examples for Existing Modules

### 4. curl - REST API Client (`curl_weather_api.nano`)
**Problem:** Fetch weather data from OpenWeatherMap API and display forecast

**Demonstrates:**
- GET requests with query parameters
- JSON response parsing
- Error handling (network errors, API errors)
- Rate limiting and retries
- Practical API integration

**Example Flow:**
```
1. Build API URL with city and API key
2. Make GET request
3. Parse JSON response
4. Extract weather data (temp, humidity, conditions)
5. Display formatted forecast
6. Handle errors (invalid city, API down, rate limit)

Real-world: API integration, web scraping, microservices
```

---

### 5. sqlite - Contact Manager (`sqlite_contact_manager.nano`)
**Problem:** Build a contact management system with CRUD operations

**Demonstrates:**
- Database schema design
- CREATE TABLE with proper types
- INSERT, SELECT, UPDATE, DELETE operations
- Transactions for data consistency
- Query building and result processing
- Error handling

**Schema:**
```sql
CREATE TABLE contacts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT,
    company TEXT,
    notes TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);

CREATE TABLE contact_tags (
    contact_id INTEGER,
    tag_id INTEGER,
    FOREIGN KEY(contact_id) REFERENCES contacts(id),
    FOREIGN KEY(tag_id) REFERENCES tags(id)
);
```

**Operations:**
```
1. Add contact (INSERT with validation)
2. Search contacts (SELECT with WHERE, LIKE)
3. Update contact (UPDATE with transaction)
4. Delete contact (DELETE with CASCADE)
5. Tag contacts (many-to-many relationship)
6. List all contacts (JOIN query)
7. Export to CSV (SELECT + file write)

Real-world: CRM, address book, inventory system
```

---

## Priority 3: Audio/Media Examples

### 6. audio_helpers - Waveform Generator (`audio_waveform_synth.nano`)
**Problem:** Generate musical notes and apply audio effects

**Demonstrates:**
- Waveform generation (sine, square, sawtooth, triangle)
- Frequency to note conversion (A4 = 440Hz)
- ADSR envelope (attack, decay, sustain, release)
- Effects (reverb, echo, low-pass filter)
- Audio mixing (combine multiple sources)

**Example:**
```
Generate a melody:
1. Create sine wave at 440Hz (A4)
2. Apply ADSR envelope (0.1s attack, 0.2s decay, 0.7 sustain, 0.3s release)
3. Add echo effect (delay 0.3s, feedback 0.5)
4. Mix with bass note (110Hz, square wave)
5. Output to WAV file or SDL audio

Real-world: Music synthesis, sound effects, audio processing
```

---

### 7. pt2_audio - MOD Player (`pt2_mod_player.nano`)
**Problem:** Load and play ProTracker MOD files with playback controls

**Demonstrates:**
- MOD file loading
- Audio playback control (play, pause, stop, seek)
- Pattern/position tracking
- Channel visualization
- Real-time audio processing

**Features:**
```
1. Load MOD file from disk
2. Display module info (title, patterns, samples)
3. Play with controls:
   - Play/Pause/Stop
   - Seek to position
   - Volume control
   - Speed/tempo adjustment
4. Visualize:
   - Current pattern
   - Active channels
   - Sample waveforms
5. Export to WAV

Real-world: Music player, game audio engine, tracker software
```

---

### 8. unicode - Text Processor (`unicode_i18n_demo.nano`)
**Problem:** Process multilingual text with proper Unicode handling

**Demonstrates:**
- UTF-8 encoding/decoding
- Character classification (letter, digit, whitespace)
- Case conversion (uppercase, lowercase, titlecase)
- String normalization (NFC, NFD, NFKC, NFKD)
- Grapheme cluster handling
- Practical internationalization

**Example:**
```
Input: Mixed language text (English, Japanese, Arabic, Emoji)
"Hello 世界! مرحبا 🌍"

Operations:
1. Count characters (not bytes)
2. Classify characters (letter, digit, punctuation, emoji)
3. Convert case (handle locale-specific rules)
4. Normalize (compose/decompose accents)
5. Validate UTF-8 (detect invalid sequences)
6. Display character info (codepoint, name, category)

Real-world: Text editors, web applications, data validation
```

---

## Implementation Checklist

### For Each Module Example:

**Documentation:**
- [ ] Problem statement (what real-world problem does it solve?)
- [ ] Module features demonstrated (list all functions/types used)
- [ ] Real-world applications (where would you use this?)
- [ ] Input/output examples (concrete data)
- [ ] Code comments explaining each step

**Code Quality:**
- [ ] Compiles without errors or warnings
- [ ] All functions have shadow tests
- [ ] Error handling for common failure cases
- [ ] Resource cleanup (files, connections, memory)
- [ ] Follows NanoLang idioms (immutable by default, explicit unsafe)

**Practical Value:**
- [ ] Solves a problem developers recognize
- [ ] Uses realistic data (not [1,2,3,4,5])
- [ ] Shows common patterns and best practices
- [ ] Includes performance considerations
- [ ] Suggests extensions and variations

**File Structure:**
```nano
/* =============================================================================
 * [Example Name] - Practical [Module] Demonstration
 * =============================================================================
 * Problem: [Clear problem statement]
 * 
 * Demonstrates:
 * - [Feature 1]
 * - [Feature 2]
 * - [Feature 3]
 * 
 * Real-World Applications:
 * - [Use case 1]
 * - [Use case 2]
 * 
 * Prerequisites:
 * - [Module dependencies]
 * - [External requirements if any]
 * =============================================================================
 */

import "modules/[module]/[module].nano" as [Alias]

/* Data structures */
struct [DataType] { ... }

/* Helper functions with shadow tests */
fn [helper](...) -> ... { ... }
shadow [helper] { ... }

/* Main demonstration */
fn main() -> int {
    /* Clear, commented implementation */
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

---

## Success Metrics

1. **Coverage:** Every module has at least one practical example
2. **Quality:** Examples solve real problems, not just demonstrate syntax
3. **Clarity:** Developers can understand and adapt examples to their needs
4. **Completeness:** All major module features are demonstrated
5. **Testing:** All examples compile, run, and pass shadow tests

---

## Implementation Timeline

### Phase 1: Critical Modules (Week 1)
- [ ] stdio_file_processor.nano
- [ ] math_ext_scientific_calc.nano
- [ ] vector2d_physics_demo.nano

### Phase 2: Enhanced Examples (Week 2)
- [ ] curl_weather_api.nano
- [ ] sqlite_contact_manager.nano

### Phase 3: Audio/Media (Week 3)
- [ ] audio_waveform_synth.nano
- [ ] pt2_mod_player.nano
- [ ] unicode_i18n_demo.nano

### Phase 4: Review & Polish
- [ ] Test all examples
- [ ] Update module READMEs with example links
- [ ] Create examples index/catalog
- [ ] Document common patterns

---

## Related Work

This plan complements:
- **ADVANCED_EXAMPLES_PLAN.md** - High-level language features (map/filter/fold, generics, AST)
- **Module READMEs** - API documentation for each module
- **Example modernization** - Updating existing examples with modern patterns

Together, these ensure:
1. Every language feature has practical examples
2. Every module has practical examples
3. All examples use modern NanoLang idioms
4. Documentation teaches problem-solving, not just syntax

