# Appendix B: Error Reference

**Common errors and how to fix them.**

## Compilation Errors

### "Undefined function"
**Problem:** Function not imported or doesn't exist  
**Fix:** Check module imports and function names

### "Type mismatch"
**Problem:** Incompatible types  
**Fix:** Verify all type annotations match

### "Missing shadow test"
**Problem:** Function lacks shadow test  
**Fix:** Add shadow block with assertions

### "Bounds checking failed"
**Problem:** Array index out of bounds  
**Fix:** Verify array length before access

## Runtime Errors

### "Assertion failed"
**Problem:** Runtime assertion evaluated to false  
**Fix:** Check logic and test data

### "Division by zero"
**Problem:** Attempted division by zero  
**Fix:** Add guard condition

### "Null pointer"
**Problem:** Accessing freed or null opaque type  
**Fix:** Verify resource lifecycle

## Syntax Errors

### "Expected ')'"
**Problem:** Unmatched parentheses  
**Fix:** Count opening and closing parens

### "Unexpected token"
**Problem:** Invalid syntax  
**Fix:** Check canonical style guide

**See also:** `docs/DEBUGGING_GUIDE.md`

---

**Previous:** [Appendix A: Examples Gallery](a_examples_gallery.html)  
**Next:** [Appendix F: Migration Guide](f_migration_guide.md)
