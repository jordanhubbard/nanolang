# Implementation Strategy - Pragmatic Approach

## Token Budget Reality
- Current: ~147K tokens used of 200K
- Remaining: ~52K tokens
- Each feature implementation: ~5-10K tokens
- We can realistically implement: 5-7 features

## Prioritized Features (by value/effort ratio)

### Tier 1: Quick Wins (3 features, ~3 hours, ~15K tokens)
1. ✅ Token helpers - DONE
2. **FOR loops** - High value, medium complexity (1 hour)
3. **Array literals** - High value, medium complexity (1 hour)  
4. **import/opaque/shadow** - Ecosystem features (1 hour)

### Tier 2: Essential (2 features, ~4 hours, ~20K tokens)
5. **Field access** - Critical for objects (2.5 hours)
6. **Float literals** - Language completeness (15 min)

### Tier 3: Advanced (skip for now, do in next session)
- Struct literals (2 hours)
- Match expressions (4 hours)
- Union construction (2 hours)
- Tuple support (2 hours)

## Recommendation

Implement Tier 1 + Tier 2 (5 features) now:
- Estimated time: ~7 hours
- Estimated tokens: ~35K
- Leaves safety margin: ~17K tokens
- Brings parser to ~85% complete

Save Tier 3 for a follow-up session when implementing:
- The actual parsing logic for struct literals
- Match expression handling  
- Union construction
- Tuple literal support

This gives maximum value (critical features) while staying within token budget.
