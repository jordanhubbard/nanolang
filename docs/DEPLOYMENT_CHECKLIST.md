# My User Guide Deployment Checklist

I use this checklist to verify my user guide before I deploy it. I do not deploy unless every requirement is met.

## Content Verification

I check that all my chapters are present and complete.

### All Chapters Present
- [ ] Landing page (index.md)
- [ ] Part I: Language Fundamentals (8 chapters)
- [ ] Part II: Standard Library (4 chapters)
- [ ] Part III: External Modules (9 chapters)
- [ ] Part IV: Advanced Topics (4 chapters)
- [ ] Appendices (4 appendices)

**Total: 30 chapters/appendices**

### Content Quality
- [ ] All code examples compile
- [ ] All shadow tests pass
- [ ] No broken internal links
- [ ] No TODO/FIXME markers
- [ ] Consistent formatting
- [ ] Proper heading hierarchy

## Build Process

I run these commands to generate my HTML output.

### HTML Generation
```bash
# Clean build
make clean-userguide
make userguide
```

- [ ] Build completes without errors
- [ ] All markdown files converted
- [ ] Sidebar generated correctly
- [ ] Assets copied (CSS, images)

### Local Testing

I test my generated files locally before I push them to production.

```bash
cd docs_html
python3 -m http.server 8000
```

- [ ] Open http://localhost:8000
- [ ] Test all navigation links
- [ ] Verify sidebar structure
- [ ] Check syntax highlighting
- [ ] Test mobile view (browser dev tools)
- [ ] Verify accessibility (browser tools)

## Navigation Testing

I ensure that my users can navigate my documentation without friction.

### Internal Links
- [ ] Chapter-to-chapter navigation works
- [ ] "Previous" links correct
- [ ] "Next" links correct
- [ ] Sidebar links functional
- [ ] Appendix references work

### External Links
- [ ] GitHub repository links
- [ ] Example file references
- [ ] API documentation links

## Visual Quality

I verify that I look correct on different screen sizes.

### Desktop View (1920x1080)
- [ ] Sidebar visible
- [ ] Content readable width
- [ ] Code blocks formatted
- [ ] Syntax highlighting applied
- [ ] Proper spacing/margins

### Tablet View (768x1024)
- [ ] Layout responsive
- [ ] Sidebar toggles properly
- [ ] Text readable
- [ ] Touch targets adequate

### Mobile View (375x667)
- [ ] Single column layout
- [ ] Hamburger menu works
- [ ] Content not cut off
- [ ] Touch navigation works

## Accessibility

I am committed to being accessible to all users.

### WCAG 2.1 AA Compliance
- [ ] Proper heading hierarchy (H1 → H2 → H3)
- [ ] Alt text for images
- [ ] Color contrast ≥ 4.5:1
- [ ] Keyboard navigation works
- [ ] Focus indicators visible
- [ ] No flashing content

### Screen Reader Testing
- [ ] VoiceOver (macOS) - Test major pages
- [ ] NVDA (Windows) - Test major pages
- [ ] Logical reading order
- [ ] Landmarks properly labeled

## Performance

I ensure that my documentation loads quickly and remains responsive.

### Page Load
- [ ] First contentful paint < 2s
- [ ] Time to interactive < 3s
- [ ] No render-blocking resources

### File Sizes
- [ ] HTML files < 500KB each
- [ ] CSS files minified
- [ ] No unnecessary JavaScript

## Cross-Browser Testing

I verify my documentation in these browsers.

### Desktop Browsers
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)

### Mobile Browsers
- [ ] Safari iOS
- [ ] Chrome Android
- [ ] Firefox Android

## Deployment

I use GitHub Pages to host my documentation.

### GitHub Pages
```bash
# Configure repository settings
# Settings → Pages → Source: docs_html/ branch: main
```

- [ ] Repository settings configured
- [ ] Custom domain (if applicable)
- [ ] HTTPS enabled
- [ ] Deploy successful
- [ ] Live site accessible

### Post-Deployment
- [ ] Visit production URL
- [ ] Smoke test critical pages
- [ ] Verify analytics (if enabled)
- [ ] Check for console errors

## Documentation

I update my other documents to point to my new user guide.

### Update References
- [ ] README.md links to user guide
- [ ] CONTRIBUTING.md references docs
- [ ] Changelog updated
- [ ] Release notes written

### Announcement
- [ ] GitHub release created
- [ ] Community notified (if applicable)
- [ ] Social media posts (if applicable)

## Rollback Plan

If I encounter issues, I revert to my previous state.

```bash
# Revert to previous version
git revert HEAD
git push

# Or restore from backup
git checkout <previous-commit> docs_html/
git commit -m "docs: Rollback user guide"
git push
```

## Sign-Off

I do not consider a deployment finished until these reviews are complete.

- [ ] Technical review complete
- [ ] Content review complete
- [ ] Quality assurance complete
- [ ] Ready for production

**Deployed by:** _______________  
**Date:** _______________  
**Version:** _______________

