# My declared map write contract

I track `task_b19f8bf0527d4a33911be26706629616`. HM_NEW records key and value
tags; HM_KEYS and HM_VALUES use those declarations for their array element
tags. My standalone native string-key maps already require the exact value tag.
I give ordinary VM HM_SET the same declared-content invariant.

Before mutation, I require the key tag and value tag to equal the map's declared
tags. I do not coerce bool, U8, enum or float to int. On refusal I report a type
error, release all three consumed values, leave other aliases and prior map
contents intact, and publish no result. Successful insertion and replacement
retain their existing ownership behavior. The direct internal heap API keeps
its existing caller precondition; this child checks the bytecode dispatcher.

This child does not change HM_GET/HM_HAS/HM_DELETE, add native non-string map
keys, or define new wildcard declarations. I test ordinary VM API lifecycle,
aliases, replacement and checked refusal, plus same-module native supported
string-to-int/string controls. I do not replay retained compiler failures or
use old malformed artifacts.
