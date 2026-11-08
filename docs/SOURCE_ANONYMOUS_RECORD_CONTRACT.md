# My anonymous record source contract

I accept a brace-only record literal when a declared record context supplies its nominal identity. My original anonymous-record fixtures exercise return values, typed local bindings and arguments. I keep those complete fixtures and their shadows unchanged.

I require each declared field exactly once, with no extra names and with its declared type. I pass declared context recursively into literal fields. I retain an explicit empty record context. I do not infer a nominal identity by searching for matching field names, or change the identity of an already named record value. Existing spread and named aggregate behavior remain separate.

I preserve the resolved nominal identity in the checked parser record so my existing aggregate lowering sees the same declared record as a named literal. This does not add a new aggregate ABI, structural type coercion or host contract. I test malformed records with the prior output in place and require refusal before publication. Native component, fresh stages and integrated platforms remain separate evidence boundaries.
