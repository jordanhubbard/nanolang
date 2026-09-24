NVM module
  magic: NVM\x01
  version: 2
  flags: 0x00000001
  entry: 0
  strings: 25
  debug entries: 0

Sections
  [logical] strings      offset=unserialized size=355
  [logical] code         offset=unserialized size=638
  [logical] functions    offset=unserialized size=60

Functions
  [0] main arity=0 locals=14 upvalues=0 result=int/1 code=0x00000000+618
  [1] identity arity=1 locals=1 upvalues=0 result=union/1 code=0x0000026a+4
  [2] __init__ arity=0 locals=0 upvalues=0 result=void/0 code=0x0000026e+16

Imports

Assembly
.string "saved"
.string "-label"
.string "-array"
.string "temporary-"
.string "saved-label"
.string "saved-array"
.string "Choice.Box { "
.string "value: "
.string "Item { "
.string "label: "
.string ", items: "
.string "["
.string ", "
.string "\""
.string "]"
.string " }"
.string "Choice.Text { "
.string "Choice.Empty"
.string "<union>"
.string "Choice.Box { value: Item { label: saved-label, items: [\"saved-array\"] } }"
.string "main"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x02\x00\x00\x00R\x00\x00\x00j\x02\x00\x00i"
.string "identity"
.string "__init__"

.entry 0

.function main 0 14 0 int 1
  [0000|0000] PUSH_STR 0  ; "saved"
  [0005|0005] PUSH_STR 1  ; "-label"
  [0010|0010] STR_CONCAT
  [0011|0011] PUSH_STR 0  ; "saved"
  [0016|0016] PUSH_STR 2  ; "-array"
  [0021|0021] STR_CONCAT
  [0022|0022] ARR_LITERAL 5 1
  [0026|0026] AGG_PACK 0 0 0 2
  [0036|0036] STORE_LOCAL 0
  [0039|0039] LOAD_LOCAL 0
  [0042|0042] AGG_PACK 1 0 0 1
  [0052|0052] CALL 1  ; identity  ; cfg:call
  [0057|0057] STORE_GLOBAL 0
  [0062|0062] LOAD_GLOBAL 0
  [0067|0067] STORE_LOCAL 1
  [0070|0070] PUSH_I64 0
  [0079|0079] STORE_LOCAL 2
L1:  ; <== jump target
  [0082|0082] LOAD_LOCAL 2
  [0085|0085] PUSH_I64 300
  [0094|0094] I64_LT_S
  [0095|0095] JMP_FALSE L0  ; cfg:branch-if-false
  [0100|0100] PUSH_STR 3  ; "temporary-"
  [0105|0105] LOAD_LOCAL 2
  [0108|0108] CAST_STRING
  [0109|0109] STR_CONCAT
  [0110|0110] STORE_LOCAL 3
  [0113|0113] LOAD_LOCAL 3
  [0116|0116] AGG_PACK 1 0 1 1
  [0126|0126] CALL 1  ; identity  ; cfg:call
  [0131|0131] STORE_GLOBAL 0
  [0136|0136] LOAD_LOCAL 2
  [0139|0139] PUSH_I64 1
  [0148|0148] I64_ADD
  [0149|0149] STORE_LOCAL 2
  [0152|0152] JMP L1  ; cfg:jump
L0:  ; <== jump target
  [0157|0157] AGG_PACK 1 0 2 0
  [0167|0167] STORE_GLOBAL 0
  [0172|0172] LOAD_LOCAL 1
  [0175|0175] DUP
  [0176|0176] AGG_TAG
  [0177|0177] PUSH_I64 0
  [0186|0186] EQ
  [0187|0187] JMP_FALSE L2  ; cfg:branch-if-false
  [0192|0192] DUP
  [0193|0193] STORE_LOCAL 4
  [0196|0196] POP
  [0197|0197] LOAD_LOCAL 4
  [0200|0200] AGG_GET 0
  [0203|0203] AGG_GET 0
  [0206|0206] PUSH_STR 4  ; "saved-label"
  [0211|0211] EQ
  [0212|0212] ASSERT
  [0213|0213] LOAD_LOCAL 4
  [0216|0216] AGG_GET 0
  [0219|0219] AGG_GET 1
  [0222|0222] PUSH_I64 0
  [0231|0231] ARR_GET
  [0232|0232] PUSH_STR 5  ; "saved-array"
  [0237|0237] EQ
  [0238|0238] ASSERT
  [0239|0239] JMP L3  ; cfg:jump
L2:  ; <== jump target
  [0244|0244] DUP
  [0245|0245] AGG_TAG
  [0246|0246] PUSH_I64 1
  [0255|0255] EQ
  [0256|0256] JMP_FALSE L4  ; cfg:branch-if-false
  [0261|0261] DUP
  [0262|0262] STORE_LOCAL 5
  [0265|0265] POP
  [0266|0266] PUSH_BOOL 0
  [0268|0268] ASSERT
  [0269|0269] JMP L3  ; cfg:jump
L4:  ; <== jump target
  [0274|0274] DUP
  [0275|0275] AGG_TAG
  [0276|0276] PUSH_I64 2
  [0285|0285] EQ
  [0286|0286] JMP_FALSE L5  ; cfg:branch-if-false
  [0291|0291] DUP
  [0292|0292] STORE_LOCAL 6
  [0295|0295] POP
  [0296|0296] PUSH_BOOL 0
  [0298|0298] ASSERT
  [0299|0299] JMP L3  ; cfg:jump
L5:  ; <== jump target
  [0304|0304] POP
  [0305|0305] PUSH_BOOL 0
  [0307|0307] ASSERT
  [0308|0308] HALT  ; cfg:halt
L3:  ; <== jump target
  [0309|0309] LOAD_LOCAL 1
  [0312|0312] DUP
  [0313|0313] AGG_TAG
  [0314|0314] PUSH_I64 0
  [0323|0323] EQ
  [0324|0324] JMP_FALSE L6  ; cfg:branch-if-false
  [0329|0329] DUP
  [0330|0330] STORE_LOCAL 7
  [0333|0333] POP
  [0334|0334] PUSH_STR 6  ; "Choice.Box { "
  [0339|0339] PUSH_STR 7  ; "value: "
  [0344|0344] STR_CONCAT
  [0345|0345] LOAD_LOCAL 7
  [0348|0348] AGG_GET 0
  [0351|0351] STORE_LOCAL 8
  [0354|0354] PUSH_STR 8  ; "Item { "
  [0359|0359] PUSH_STR 9  ; "label: "
  [0364|0364] STR_CONCAT
  [0365|0365] LOAD_LOCAL 8
  [0368|0368] AGG_GET 0
  [0371|0371] CAST_STRING
  [0372|0372] STR_CONCAT
  [0373|0373] PUSH_STR 10  ; ", items: "
  [0378|0378] STR_CONCAT
  [0379|0379] LOAD_LOCAL 8
  [0382|0382] AGG_GET 1
  [0385|0385] STORE_LOCAL 9
  [0388|0388] PUSH_STR 11  ; "["
  [0393|0393] STORE_LOCAL 10
  [0396|0396] PUSH_I64 0
  [0405|0405] STORE_LOCAL 11
L9:  ; <== jump target
  [0408|0408] LOAD_LOCAL 11
  [0411|0411] LOAD_LOCAL 9
  [0414|0414] ARR_LEN
  [0415|0415] LT
  [0416|0416] JMP_FALSE L7  ; cfg:branch-if-false
  [0421|0421] LOAD_LOCAL 10
  [0424|0424] LOAD_LOCAL 11
  [0427|0427] JMP_FALSE L8  ; cfg:branch-if-false
  [0432|0432] PUSH_STR 12  ; ", "
  [0437|0437] STR_CONCAT
L8:  ; <== jump target
  [0438|0438] PUSH_STR 13  ; """
  [0443|0443] STR_CONCAT
  [0444|0444] LOAD_LOCAL 9
  [0447|0447] LOAD_LOCAL 11
  [0450|0450] ARR_GET
  [0451|0451] CAST_STRING
  [0452|0452] STR_CONCAT
  [0453|0453] PUSH_STR 13  ; """
  [0458|0458] STR_CONCAT
  [0459|0459] STORE_LOCAL 10
  [0462|0462] LOAD_LOCAL 11
  [0465|0465] PUSH_I64 1
  [0474|0474] I64_ADD
  [0475|0475] STORE_LOCAL 11
  [0478|0478] JMP L9  ; cfg:jump
L7:  ; <== jump target
  [0483|0483] LOAD_LOCAL 10
  [0486|0486] PUSH_STR 14  ; "]"
  [0491|0491] STR_CONCAT
  [0492|0492] STR_CONCAT
  [0493|0493] PUSH_STR 15  ; " }"
  [0498|0498] STR_CONCAT
  [0499|0499] STR_CONCAT
  [0500|0500] PUSH_STR 15  ; " }"
  [0505|0505] STR_CONCAT
  [0506|0506] JMP L10  ; cfg:jump
L6:  ; <== jump target
  [0511|0511] DUP
  [0512|0512] AGG_TAG
  [0513|0513] PUSH_I64 1
  [0522|0522] EQ
  [0523|0523] JMP_FALSE L11  ; cfg:branch-if-false
  [0528|0528] DUP
  [0529|0529] STORE_LOCAL 12
  [0532|0532] POP
  [0533|0533] PUSH_STR 16  ; "Choice.Text { "
  [0538|0538] PUSH_STR 7  ; "value: "
  [0543|0543] STR_CONCAT
  [0544|0544] LOAD_LOCAL 12
  [0547|0547] AGG_GET 0
  [0550|0550] CAST_STRING
  [0551|0551] STR_CONCAT
  [0552|0552] PUSH_STR 15  ; " }"
  [0557|0557] STR_CONCAT
  [0558|0558] JMP L10  ; cfg:jump
L11:  ; <== jump target
  [0563|0563] DUP
  [0564|0564] AGG_TAG
  [0565|0565] PUSH_I64 2
  [0574|0574] EQ
  [0575|0575] JMP_FALSE L12  ; cfg:branch-if-false
  [0580|0580] DUP
  [0581|0581] STORE_LOCAL 13
  [0584|0584] POP
  [0585|0585] PUSH_STR 17  ; "Choice.Empty"
  [0590|0590] JMP L10  ; cfg:jump
L12:  ; <== jump target
  [0595|0595] POP
  [0596|0596] PUSH_STR 18  ; "<union>"
L10:  ; <== jump target
  [0601|0601] PUSH_STR 19  ; "Choice.Box { value: Item { label: saved-label, items: ["saved-array"] } }"
  [0606|0606] EQ
  [0607|0607] ASSERT
  [0608|0608] PUSH_I64 0
  [0617|0617] RET  ; cfg:return
.end

.function identity 1 1 0 union 1
  [0000|0618] LOAD_LOCAL 0
  [0003|0621] RET  ; cfg:return
.end

.function __init__ 0 0 0 void 0
  [0000|0622] AGG_PACK 1 0 2 0
  [0010|0632] STORE_GLOBAL 0
  [0015|0637] RET  ; cfg:return
.end

