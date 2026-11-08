# My existing byte conversion contract

Under #1110/#1081 I preserve `bytes_from_string` and `string_from_bytes`, rather than removing the unchanged `tests/test_bytes_conversion.nano` from my original corpus. Both original seed routes pass that fixture at cd0; both fresh stages refuse selected shadows at an undefined `bytes_from_string` call.

My public conversion result is `array<u8>` for STRING input and STRING for a byte-array input. A coarse NanoISA ARRAY tag does not establish byte element storage. My source checker currently advertises those results without a dedicated exact operand check; my source producer has no conversion host mapping, and my native translator has no exact conversion adapters. I require separate lexical resolution for locally bound names.

My implementations currently differ internally:

| Route | Encoding storage | Decoding storage | Observed implementation boundary |
| --- | --- | --- | --- |
| C backend | `ELEM_U8` | `ELEM_U8` | Encoding uses `strnlen` with a 64 MiB cap; decoding allocates length plus a terminator. |
| C interpreter | `ELEM_INT` | INT or U8 | Encoding uses `strlen`; decoding casts each element to an unsigned byte. |
| NanoVM host bridge | `ELEM_INT` | INT or U8 | Encoding uses `strlen`; decoding casts each word to a character and appends a terminator. |

These are source-inspection facts, not a claim that all routes have the same behavior for every input. Strings use NUL-terminated storage. Embedded zero bytes and invalid UTF-8 require explicit observation of the complete string/foreign boundary before I choose native behavior. I do not call arbitrary binary data a valid STRING or silently widen source arrays to match the historical VM representation.

My baseline controls retain the whole original fixture with its dependency shadows, and add production-main empty/Unicode byte order, detached conversion storage, typed argument/result, once-only evaluation and bound-name behavior. With my older local providers, both production controls pass source creation, verification and VM execution, then refuse the missing exact `vm_bytes_from_string` native ABI. The whole fixture's main only returns zero: its successful old-provider native program does not execute the conversions that belong to selected shadows. Fresh source qualification must retain those shadows, and production-main controls must exercise actual generated native conversion code.

I preserve exact arity/type and unsupported import/storage refusals, prior output, owned lifetimes and allocation bounds when implementing. Native admission is limited to existing builtin identities; generic externs, SDK descriptors, full file-service authority, fresh compiler lifecycle and release acceptance remain separate. I retain the preceding failure before changing code or claiming qualification.
