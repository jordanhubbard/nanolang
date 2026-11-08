# My byte type alias source contract

I treat the bare `byte` type spelling as my existing `u8` type. I normalize that spelling within recursive array and function annotations. Qualified nominal names keep their identity. My C lexer already classifies the bare spelling as the same primitive type.

I preserve my existing `u8` literal range checks, computed modulo-256 conversion and exact array storage. This alias correction does not change general scalar compatibility or general function or array identity; I track directed call conversion separately under #1104. I keep the complete original `test_u8_basic.nano` fixture and its shadows unchanged.

I require actual source VM/native acceptance and actual native checker controls, with fresh-stage and integrated platform results recorded separately.

Under #1105 I retain scalar integer arithmetic promotion in my source NanoISA result query: checked U8/INT operands produce INT arithmetic results. My existing scalar emitter converts U8 operands explicitly to INT, then uses the existing integer opcode. A declared U8 destination narrows a computed result modulo 256; direct out-of-range literals still refuse. I keep both operand orders and once-only evaluation, without changing unary, float, BOOL/STRING, generic, array storage or host-import policy. Fresh source, original whole byte shadows and integrated acceptance remain separately required.
