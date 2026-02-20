# My User Guide Build Process

I use this process to generate my HTML user guide from my markdown sources.

## My Build Command

```bash
# From project root
make userguide

# Or directly
./scripts/userguide_build_html.nano
```

## My File Structure

```
userguide/
├── index.md (landing page)
├── part1_fundamentals/
│   ├── 01_getting_started.md
│   ├── 02_syntax_types.md
│   └── ... (08 chapters)
├── part2_stdlib/
│   └── ... (4 chapters)
├── part3_modules/
│   └── ... (9 module groups)
├── part4_advanced/
│   └── ... (4 chapters)
└── appendices/
    └── ... (4 appendices)
```

## My Output Structure

```
docs_html/
├── index.html
├── part1_fundamentals/
├── part2_stdlib/
├── part3_modules/
├── part4_advanced/
├── appendices/
└── assets/
    ├── style.css
    └── syntax-highlight.css
```

## How I Generate My Sidebar

I organize my sidebar hierarchically:
1. **Landing Page**
2. **Part I: Language Fundamentals** (Chapters 1 to 8)
3. **Part II: Standard Library** (Chapters 9 to 12)
4. **Part III: External Modules** (Chapters 13 to 21)
5. **Part IV: Advanced Topics** (Chapters 22 to 25)
6. **Appendices** (A to D)

I have documented the implementation details in `docs/SIDEBAR_GENERATION_SPEC.md`.

## My Syntax Highlighting

I highlight code blocks using `assets/syntax-highlight.css`:
- Keywords: `fn`, `let`, `if`, `while`, etc.
- Functions: `println`, `array_get`, etc.
- Strings: Quoted text
- Numbers: Integer and float literals
- Comments: Lines starting with `#`

## How I Test My Docs

I follow these steps before I deploy:

```bash
# Build HTML
make userguide

# Test locally
cd docs_html
python3 -m http.server 8000

# Visit http://localhost:8000
```

I verify these points:
- All chapters load
- Navigation works
- Syntax highlighting applied
- Mobile-responsive layout
- Cross-references work

## My Deployment Process

### GitHub Pages

```bash
# Build
make userguide

# Deploy
git add docs_html/
git commit -m "docs: Update user guide HTML"
git push

# Configure GitHub Pages to serve from /docs_html
```

### Manual Deployment

```bash
# Copy to web server
scp -r docs_html/* user@server:/var/www/nanolang/
```

## Accessibility Standards

The HTML I generate meets WCAG 2.1 AA standards:
- Semantic HTML5 elements
- Proper heading hierarchy
- Alt text for images
- Keyboard navigation
- Sufficient color contrast
- Mobile-friendly responsive design

## Troubleshooting

### Missing styles
**Problem:** CSS not loading.
**Fix:** Check relative paths in HTML headers.

### Broken links
**Problem:** 404 on chapter links.
**Fix:** Verify file names match exactly. I am case sensitive.

### Code blocks not highlighted
**Problem:** Syntax highlighting missing.
**Fix:** Ensure `syntax-highlight.css` is loaded.

## See Also

- `docs/SIDEBAR_GENERATION_SPEC.md` - My sidebar implementation
- `docs/API_REFERENCE_TEMPLATE.md` - My documentation standards
- `userguide/assets/syntax-highlight.css` - My highlighting styles

