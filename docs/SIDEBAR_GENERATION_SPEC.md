# Sidebar Generation Specification

**Technical specification for updating `scripts/userguide_build_html.nano` to support the new user guide structure.**

---

## Current Structure

The current sidebar generation (lines 717-767 in `userguide_build_html.nano`) uses a simple flat categorization:

```
User Guide (chapters 01-07)
├─ 01_getting_started.html
├─ 02_control_flow.html
└─ ...

Examples
└─ 07_examples.html

Module API Reference  
├─ api_reference/bullet.html
├─ api_reference/coverage.html
└─ ...
```

This is implemented by:
1. Scanning all `.md` files
2. Categorizing by path (`api_reference/` → API group, etc.)
3. Creating collapsible `<details>` groups

---

## New Structure Required

The new sidebar should follow a professional book structure:

```
Part I: Language Fundamentals
├─ 1. Getting Started
├─ 2. Basic Syntax & Types
├─ 3. Variables & Bindings
├─ 4. Functions
├─ 5. Control Flow
├─ 6. Collections
├─ 7. Data Structures
└─ 8. Modules & Imports

Part II: Standard Library
├─ 9. Core Utilities
├─ 10. Collections Library
├─ 11. I/O & Filesystem
└─ 12. System & Runtime

Part III: External Modules
├─ 13. Text Processing
│   ├─ 13.1 regex
│   ├─ 13.2 log
│   └─ 13.3 StringBuilder
├─ 14. Data Formats
│   ├─ 14.1 JSON
│   └─ 14.2 SQLite
├─ ... (chapters 15-21)

Part IV: Advanced Topics
├─ 22. Canonical Style Guide
├─ 23. Higher-Level Patterns
├─ 24. Performance & Optimization
└─ 25. Contributing & Extending

Appendices
├─ A. Examples Gallery
├─ B. Quick Reference
├─ C. Troubleshooting Guide
└─ D. Glossary
```

---

## Implementation Strategy

### Phase 1: File Path Categorization

Add new categorization logic to detect part directories:

```nano
fn categorize_file_by_path(rel_path: string) -> string {
    if (str_contains rel_path "part1_fundamentals/") {
        return "part1"
    }
    if (str_contains rel_path "part2_stdlib/") {
        return "part2"
    }
    if (str_contains rel_path "part3_modules/") {
        return "part3"
    }
    if (str_contains rel_path "part4_advanced/") {
        return "part4"
    }
    if (str_contains rel_path "appendices/") {
        return "appendices"
    }
    # Legacy paths for backward compatibility
    if (str_contains rel_path "api_reference/") {
        return "api"
    }
    return "guide"
}
```

### Phase 2: Chapter Number Extraction

Extract chapter numbers from filenames:

```nano
fn extract_chapter_number(filename: string) -> int {
    # From "01_getting_started.md" → 1
    # From "22_canonical_style.md" → 22
    if (not (str_starts_with filename "0")) {
        if (not (str_starts_with filename "1")) {
            if (not (str_starts_with filename "2")) {
                return -1
            }
        }
    }
    let mut num: int = 0
    let len: int = (str_length filename)
    let mut i: int = 0
    while (and (< i len) (char_is_digit (char_at filename i))) {
        set num (+ (* num 10) (- (char_at filename i) 48))
        set i (+ i 1)
    }
    return num
}

fn char_is_digit(c: int) -> bool {
    return (and (>= c 48) (<= c 57))
}
```

### Phase 3: Hierarchical Structure Building

Build a tree structure for nested items:

```nano
struct SidebarItem {
    title: string
    rel_html: string
    level: int  # 0=part, 1=chapter, 2=subsection
    children: array<SidebarItem>
    is_active: bool
}

fn build_sidebar_tree(files: array<PageMeta>, active: string) -> array<SidebarItem> {
    # Group files by part
    # Within each part, group by chapter
    # Within each chapter, collect subsections
    # Return hierarchical structure
}
```

### Phase 4: HTML Generation

Generate hierarchical HTML with nested lists:

```nano
fn render_sidebar_item(item: SidebarItem, out_dir: string, page_dir: string) -> string {
    let href: string = (relpath_copy (join out_dir item.rel_html) page_dir)
    let active_class: string = (cond (item.is_active " active") (else ""))
    let level_class: string = (+ "sidebar-item-level-" (int_to_string item.level))
    
    let mut html: array<string> = []
    set html (array_push html (+ "<li class=\"" (+ level_class active_class)))
    set html (array_push html "\">")
    set html (array_push html (+ "<a href=\"" (+ (html_escape href) "\">")))
    set html (array_push html (html_escape item.title))
    set html (array_push html "</a>")
    
    # Render children if any
    if (> (array_length item.children) 0) {
        set html (array_push html "<ul>")
        let mut i: int = 0
        while (< i (array_length item.children)) {
            let child: SidebarItem = (at item.children i)
            set html (array_push html (render_sidebar_item child out_dir page_dir))
            set i (+ i 1)
        }
        set html (array_push html "</ul>")
    }
    
    set html (array_push html "</li>")
    return (sb_join html "")
}
```

### Phase 5: Part Headers

Each part should have a header:

```nano
fn build_part_group(part_title: string, items: array<SidebarItem>, open: bool) -> string {
    let open_attr: string = (cond (open " open") (else ""))
    return (sb_join [
        "<details class=\"sidebar-part\"",
        open_attr,
        ">",
        "<summary class=\"part-title\">",
        (html_escape part_title),
        "</summary>",
        "<ul class=\"part-chapters\">",
        (render_sidebar_items items out_dir page_dir),
        "</ul>",
        "</details>"
    ] "")
}
```

---

## CSS Styling Requirements

The new structure requires additional CSS classes:

```css
/* Part-level grouping */
.sidebar-part {
    margin-bottom: 1.5rem;
}

.sidebar-part > summary.part-title {
    font-weight: 700;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6b7280;
    padding: 0.5rem 0;
    cursor: pointer;
}

.sidebar-part[open] > summary.part-title {
    color: var(--accent, #7aa2f7);
    margin-bottom: 0.5rem;
}

/* Chapter-level items */
.part-chapters {
    list-style: none;
    padding: 0;
    margin: 0;
}

.sidebar-item-level-1 {
    margin-left: 0;
    padding: 0.25rem 0.5rem;
}

.sidebar-item-level-1.active {
    background: var(--accent-light, rgba(122, 162, 247, 0.1));
    border-left: 3px solid var(--accent, #7aa2f7);
    font-weight: 600;
}

/* Subsection items (nested) */
.sidebar-item-level-2 {
    margin-left: 1rem;
    padding: 0.2rem 0.5rem;
    font-size: 0.9rem;
}

.sidebar-item-level-2.active {
    background: var(--accent-light, rgba(122, 162, 247, 0.05));
    border-left: 2px solid var(--accent, #7aa2f7);
}

/* Links within sidebar items */
.sidebar-item-level-1 a,
.sidebar-item-level-2 a {
    text-decoration: none;
    color: #374151;
    display: block;
}

.sidebar-item-level-1 a:hover,
.sidebar-item-level-2 a:hover {
    color: var(--accent, #7aa2f7);
}
```

These should be added to `userguide/assets/style.css`.

---

## Algorithm Outline

Replace the current `build_toc()` function with:

```
fn build_new_sidebar(
    titles: array<string>,
    rels: array<string>,
    active: string,
    out_dir: string,
    page_dir: string
) -> string {
    # 1. Categorize all files by part
    let mut part1_items: array<PageInfo> = []
    let mut part2_items: array<PageInfo> = []
    let mut part3_items: array<PageInfo> = []
    let mut part4_items: array<PageInfo> = []
    let mut appendix_items: array<PageInfo> = []
    
    for each (title, rel) in (titles, rels) {
        let category: string = (categorize_file_by_path rel)
        let info: PageInfo = { title: title, rel: rel }
        
        match category {
            "part1" -> array_push part1_items info
            "part2" -> array_push part2_items info
            "part3" -> array_push part3_items info
            "part4" -> array_push part4_items info
            "appendices" -> array_push appendix_items info
        }
    }
    
    # 2. Build hierarchical structure for each part
    let part1_tree: array<SidebarItem> = (build_part_tree part1_items active)
    let part2_tree: array<SidebarItem> = (build_part_tree part2_items active)
    let part3_tree: array<SidebarItem> = (build_part_tree part3_items active)
    let part4_tree: array<SidebarItem> = (build_part_tree part4_items active)
    let appendix_tree: array<SidebarItem> = (build_part_tree appendix_items active)
    
    # 3. Determine which parts should be open
    let part1_open: bool = (any_active part1_tree)
    let part2_open: bool = (any_active part2_tree)
    let part3_open: bool = (any_active part3_tree)
    let part4_open: bool = (any_active part4_tree)
    let appendix_open: bool = (any_active appendix_tree)
    
    # 4. Generate HTML for each part
    let mut parts: array<string> = []
    
    if (not (== (array_length part1_items) 0)) {
        array_push parts (build_part_group "Part I: Language Fundamentals" part1_tree part1_open)
    }
    
    if (not (== (array_length part2_items) 0)) {
        array_push parts (build_part_group "Part II: Standard Library" part2_tree part2_open)
    }
    
    if (not (== (array_length part3_items) 0)) {
        array_push parts (build_part_group "Part III: External Modules" part3_tree part3_open)
    }
    
    if (not (== (array_length part4_items) 0)) {
        array_push parts (build_part_group "Part IV: Advanced Topics" part4_tree part4_open)
    }
    
    if (not (== (array_length appendix_items) 0)) {
        array_push parts (build_part_group "Appendices" appendix_tree appendix_open)
    }
    
    # 5. Wrap in sidebar container
    return (sb_join [
        "<aside class=\"sidebar\">",
        "<h2>NanoLang by Example</h2>",
        (sb_join parts "\n"),
        "</aside>"
    ] "\n")
}
```

---

## Handling Nested Modules (Part 3)

Part 3 has a special structure with nested modules:

```
part3_modules/
├── 13_text_processing/
│   ├── index.md           (Chapter 13 intro)
│   ├── regex.md           (13.1)
│   ├── log.md             (13.2)
│   └── stringbuilder.md   (13.3)
```

Detection logic:

```nano
fn is_module_index(rel_path: string) -> bool {
    # Check if path is like "part3_modules/13_text_processing/index.md"
    return (and
        (str_contains rel_path "part3_modules/")
        (str_ends_with rel_path "/index.md")
    )
}

fn is_module_subpage(rel_path: string) -> bool {
    # Check if path is like "part3_modules/13_text_processing/regex.md"
    return (and
        (str_contains rel_path "part3_modules/")
        (and
            (not (str_ends_with rel_path "/index.md"))
            (str_contains rel_path "/")
        )
    )
}

fn get_module_chapter_dir(rel_path: string) -> string {
    # From "part3_modules/13_text_processing/regex.md" → "13_text_processing"
    let parts: array<string> = (str_split rel_path "/")
    if (>= (array_length parts) 2) {
        return (at parts 1)
    }
    return ""
}
```

Build module subsections:

```nano
fn build_module_tree(module_items: array<PageInfo>) -> array<SidebarItem> {
    # Group by chapter directory (13_text_processing, 14_data_formats, etc.)
    let mut groups: HashMap = (map_new)
    
    for each item in module_items {
        let chapter_dir: string = (get_module_chapter_dir item.rel)
        if (not (map_has groups chapter_dir)) {
            map_put groups chapter_dir []
        }
        let group: array<PageInfo> = (map_get groups chapter_dir)
        array_push group item
        map_put groups chapter_dir group
    }
    
    # For each group, create a parent item (index.md) with children
    let mut tree: array<SidebarItem> = []
    for each (chapter_dir, items) in groups {
        # Find the index page
        let mut index_item: PageInfo = null
        let mut sub_items: array<PageInfo> = []
        
        for each item in items {
            if (str_ends_with item.rel "/index.md") {
                set index_item item
            } else {
                array_push sub_items item
            }
        }
        
        if (!= index_item null) {
            let mut children: array<SidebarItem> = []
            for each sub in sub_items {
                array_push children (SidebarItem {
                    title: sub.title,
                    rel_html: sub.rel,
                    level: 2,
                    children: [],
                    is_active: (== sub.rel active)
                })
            }
            
            array_push tree (SidebarItem {
                title: index_item.title,
                rel_html: index_item.rel,
                level: 1,
                children: children,
                is_active: (== index_item.rel active)
            })
        }
    }
    
    return tree
}
```

---

## Backward Compatibility

Keep support for old structure during transition:

```nano
fn build_toc_adaptive(
    titles: array<string>,
    rels: array<string>,
    active: string,
    out_dir: string,
    page_dir: string
) -> string {
    # Check if new structure exists
    let has_new_structure: bool = (any_contains rels "part1_fundamentals/")
    
    if has_new_structure {
        return (build_new_sidebar titles rels active out_dir page_dir)
    } else {
        # Fall back to old sidebar logic
        return (build_toc titles rels active out_dir page_dir [])
    }
}
```

---

## Testing Checklist

Before deploying the new sidebar:

- [ ] All Part I chapters appear in correct order
- [ ] All Part II chapters appear in correct order
- [ ] Part III modules are nested correctly
- [ ] Clicking chapter links navigates correctly
- [ ] Active chapter is highlighted
- [ ] Parts open/close with expand/collapse
- [ ] Active part opens by default
- [ ] Nested subsections (13.1, 13.2, etc.) work
- [ ] Navigation persists across page loads
- [ ] Mobile view sidebar is usable
- [ ] Keyboard navigation works
- [ ] Old structure still works (backward compatibility)

---

## Example HTML Output

Expected HTML structure:

```html
<aside class="sidebar">
  <h2>NanoLang by Example</h2>
  
  <details class="sidebar-part" open>
    <summary class="part-title">Part I: Language Fundamentals</summary>
    <ul class="part-chapters">
      <li class="sidebar-item-level-1">
        <a href="part1_fundamentals/01_getting_started.html">1. Getting Started</a>
      </li>
      <li class="sidebar-item-level-1 active">
        <a href="part1_fundamentals/02_syntax_types.html">2. Basic Syntax & Types</a>
      </li>
      <!-- ... more chapters ... -->
    </ul>
  </details>
  
  <details class="sidebar-part">
    <summary class="part-title">Part III: External Modules</summary>
    <ul class="part-chapters">
      <li class="sidebar-item-level-1">
        <a href="part3_modules/13_text_processing/index.html">13. Text Processing</a>
        <ul>
          <li class="sidebar-item-level-2">
            <a href="part3_modules/13_text_processing/regex.html">13.1 regex</a>
          </li>
          <li class="sidebar-item-level-2">
            <a href="part3_modules/13_text_processing/log.html">13.2 log</a>
          </li>
        </ul>
      </li>
    </ul>
  </details>
  
  <!-- ... more parts ... -->
</aside>
```

---

## Implementation Notes

1. **Incremental Development**: Implement one part at a time
2. **Test After Each Change**: Run `make userguide` after each function
3. **Keep Old Code**: Comment out rather than delete during transition
4. **Use Shadow Tests**: Add tests for all new categorization functions
5. **Log Debugging**: Use trace mode to debug tree building

---

**This specification provides the complete algorithm for updating sidebar generation to support the new hierarchical structure.**
