# Source transformation check

I retained the complete bodies of my six resource algorithms under an invertible mechanical change: Parser parameters become ResourceDeclarations, recursive calls use the internal helpers, and the four Parser getter calls become their existing typed-list operations. Reversing those exact substitutions reproduces every original function byte. Original Parser-facing signatures remain wrappers. All191 existing typechecker shadows remain byte-identical and in order; six direct internal-helper shadows are appended.

This is a source correspondence check, not executed semantic qualification. The whole source diff still requires independent review, including public wrapper context construction, typed-list carriers and the exact existing getters. No compiler or shadow has executed this source yet.
