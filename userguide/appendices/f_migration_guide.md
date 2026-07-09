# Appendix F: Migration Guide

**Upgrading between NanoLang versions.**

## Version 1.0 → 1.1 (Upcoming)

### Deprecated Features
- `str_concat` → Use `+` operator instead
- `at` → Use `array_get` instead (more explicit)

### New Features
- Enhanced error messages with context
- Improved JSON parsing
- Additional stdlib functions

### Breaking Changes
None expected for 1.1

## Best Practices for Updates

1. **Run tests first**
   ```bash
   make test
   ```

2. **Check deprecation warnings**
   ```bash
   ./bin/nanoc --warnings file.nano
   ```

3. **Update gradually**
   - Fix one module at a time
   - Run tests after each change

4. **Read changelog**
   - Check `CHANGELOG.md` for details

---

**Previous:** [Appendix E: Error Reference](e_error_reference.md)  
**Next:** [Appendix D: Glossary](d_glossary.html)
