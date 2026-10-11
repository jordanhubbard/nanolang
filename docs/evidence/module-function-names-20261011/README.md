# My native module function symbol evidence

I continue #976 from 8e75c9979. Native function naming previously copied printable punctuation into C identifiers, discarded non-ASCII module owners and truncated module prefixes. Imported prototypes separately concatenated raw module names. I now use the existing byte-preserving module symbol encoder for definitions, declarations, direct calls and function values. The encoder also encodes leading digits so its result can begin a C identifier. Names resembling its reserved encoding prefix are themselves encoded, preserving distinct owners.

I retain the three original compile failures and the intermediate prototype failures. My maintained tests import punctuation, underscores, spaces, encoded-looking names, UTF-8, leading digits, dots and long common prefixes together. Every module returns a different value through an internal call; the root executes qualified calls, imported callbacks and a selective alias.

All three methods pass through current C-native, C NanoISA and the fresh self-hosted seed retained from the union checkpoint. The C NanoISA route verifies and runs each module, then executes its strict ASan/UBSan C AOT translation. All 18 native module-name/union methods pass. The complete transpiler unit gate passes, including both assert-literal methods.

I separately instrument the changed transpiler and metadata translation units with ASan/UBSan; all three executed filename methods pass. I disable interpreter leak detection and do not claim leak freedom. The symbol helper's UBSan control checks valid identifier bytes, leading digits, pairwise distinct outputs and metadata call mapping, including encoded-looking neighbors.

I do not claim a new full bootstrap or release-candidate platform qualification from this C naming correction. Aggregate union payload lowering, by-value layout-cycle validation and the remaining 5.1 scope stay open.

My final complete module-compilation invocation suite passes all 24 methods in 58.815 seconds, including encoded metadata calls, path quoting, private intermediates and preservation after failed compilation.
