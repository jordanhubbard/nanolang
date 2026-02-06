# User Guide Maintenance

**Keeping the documentation up-to-date.**

## Regular Updates

### When to Update

Update documentation when:
- ✅ Adding new language features
- ✅ Adding new stdlib functions
- ✅ Adding new modules
- ✅ Deprecating features
- ✅ Fixing bugs that affect examples
- ✅ Improving explanations based on user feedback

### Update Process

1. **Identify affected sections**
   - Search for feature mentions
   - Check related examples
   - Review cross-references

2. **Update content**
   - Edit markdown files
   - Update code examples
   - Verify shadow tests still pass

3. **Test changes**
   ```bash
   # Compile affected examples
   ./bin/nanoc examples/affected_example.nano
   
   # Rebuild user guide
   make userguide
   
   # Test locally
   cd docs_html && python3 -m http.server
   ```

4. **Deploy**
   ```bash
   git add userguide/ docs_html/
   git commit -m "docs: Update for feature X"
   git push
   ```

## Adding New Content

### New Chapter

1. **Create markdown file**
   ```bash
   # Follow naming convention
   userguide/partN_name/NN_chapter_name.md
   ```

2. **Follow template structure**
   - Introduction
   - Sections with examples
   - Summary
   - Navigation links

3. **Update sidebar**
   - Modify `scripts/userguide_build_html.nano`
   - Add chapter to table of contents

4. **Add cross-references**
   - Link from related chapters
   - Update navigation (Previous/Next)

### New Example

1. **Create example file**
   ```bash
   examples/category/example_name.nano
   ```

2. **Include shadow tests**
   ```nano
   fn example_function() -> int {
       return 42
   }
   
   shadow example_function {
       assert (== (example_function) 42)
   }
   ```

3. **Reference in documentation**
   - Add to relevant chapter
   - Include in Appendix A (Examples Gallery)

## Quality Standards

### Code Examples

All code examples must:
- ✅ Use canonical syntax
- ✅ Include shadow tests
- ✅ Compile without errors
- ✅ Be self-contained (or clearly reference modules)
- ✅ Follow naming conventions

### Writing Style

Follow `docs/TECHNICAL_WRITING_STYLE.md`:
- ✅ Clear, concise language
- ✅ Active voice
- ✅ Second person ("you")
- ✅ Present tense
- ✅ Example-driven approach

### Formatting

- ✅ Proper heading hierarchy (H1 → H2 → H3)
- ✅ Code blocks with language tags
- ✅ Consistent terminology
- ✅ Call-out boxes (✅ ⚠️ 💡 ❌)

## Review Process

### Self-Review Checklist

Before submitting:
- [ ] Examples compile
- [ ] Shadow tests pass
- [ ] Links work
- [ ] Spelling/grammar checked
- [ ] Formatting consistent
- [ ] Follows style guide

### Peer Review

Request review for:
- Major content additions
- API changes
- Deprecations
- Structural changes

## Version Control

### Branch Strategy

```bash
# Create feature branch
git checkout -b docs/new-feature-guide

# Make changes
git add userguide/
git commit -m "docs: Add guide for new feature"

# Open PR
git push origin docs/new-feature-guide
```

### Commit Messages

Follow convention:
```
docs: <brief description>

<detailed explanation>

Related to <issue-id>
```

## Automation

### Continuous Integration

CI checks should:
- ✅ Compile all examples
- ✅ Run shadow tests
- ✅ Build HTML user guide
- ✅ Check for broken links
- ✅ Verify accessibility

### Automated Deployments

On merge to main:
1. CI builds HTML
2. Tests pass
3. Deploy to GitHub Pages
4. Notify team

## Monitoring

### Analytics

Track:
- Page views by chapter
- Time on page
- Bounce rate
- Search terms

Use data to:
- Identify popular topics
- Find confusing sections
- Prioritize improvements

### User Feedback

Collect feedback via:
- GitHub issues
- Community forums
- User surveys
- Direct emails

## Long-Term Maintenance

### Quarterly Review

Every quarter:
- [ ] Review page analytics
- [ ] Check for outdated content
- [ ] Update dependencies
- [ ] Refresh examples
- [ ] Improve explanations

### Annual Audit

Annually:
- [ ] Comprehensive content review
- [ ] Accessibility audit
- [ ] Performance optimization
- [ ] User survey
- [ ] Major refactoring (if needed)

## Tools

### Useful Commands

```bash
# Find TODOs
grep -r "TODO\|FIXME" userguide/

# Check for broken internal links
./scripts/check_links.sh

# Find duplicate content
./scripts/find_duplicates.sh

# Generate word count
find userguide/ -name "*.md" | xargs wc -w

# List all code blocks
grep -r '```nano' userguide/ | wc -l
```

### Recommended Editor Setup

VS Code extensions:
- Markdown All in One
- Code Spell Checker
- Markdown Preview Enhanced
- Auto Markdown TOC

## Contact

For questions about documentation:
- Open GitHub issue with `documentation` label
- Email: docs@nanolang.org (if available)
- Community forum: (if available)
