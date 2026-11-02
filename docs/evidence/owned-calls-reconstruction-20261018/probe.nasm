.string "Left"
.string "value"
.string "yes"
.string "Right"
.string "main"
.string "make"
.string "forward"
.string "consume"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x00\x00J\x00\x00\x00n"
.string "\x00\x00\x00\x00\x01\x00\x00\x00\x1e\x00\x00\x00J\x00\x00\x00owner"
.string "\x00\x00\x00\x00\x02\x00\x00\x00F\x00\x00\x00J\x00\x00\x00result"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x18\x00\x00\x00value"
.string "\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x18\x00\x00\x00yes"
.string "\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00owner"
.string "\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x001\x00\x00\x00owner"
.string "\x03\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x001\x00\x00\x00extra"
.string "\x03\x00\x00\x00\x03\x00\x00\x00\x15\x00\x00\x001\x00\x00\x00value"
.string "\x03\x00\x00\x00\x04\x00\x00\x00$\x00\x00\x001\x00\x00\x00yes"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x01\x00\x00\x00"
.string "\x01\x00\x00\x00\x01\x00\x00\x00"
.string "\x02\x00\x00\x00\x01\x00\x00\x00"
.string "\x03\x00\x00\x00\x01\x00\x00\x00"
.string "nano.affine-use.v1"
.string "\x00\x00\x00\x00\r\x00\x00\x00\x1d\x00\x00\x00\x02\x01\x00\x00<\x00\x00\x00\x05\x00\x00\x00docs/evidence/owned-calls-reconstruction-20261018/probe.nanoowner"
.string "\x02\x00\x00\x00\x05\x00\x00\x00*\x00\x00\x00\x02\x01\x00\x00<\x00\x00\x00\x05\x00\x00\x00docs/evidence/owned-calls-reconstruction-20261018/probe.nanoowner"
.string "\x03\x00\x00\x00\x07\x00\x00\x00H\x00\x00\x00\x02\x01\x00\x00<\x00\x00\x00\x05\x00\x00\x00docs/evidence/owned-calls-reconstruction-20261018/probe.nanoowner"

.metadata 8 9
.metadata 8 10
.metadata 8 11
.metadata 8 12
.metadata 8 13
.metadata 8 14
.metadata 8 15
.metadata 8 16
.metadata 8 17
.metadata 8 18
.metadata 19 20
.metadata 19 21
.metadata 19 22
.metadata 19 23
.metadata 24 25
.metadata 24 26
.metadata 24 27

.types 2 0 0

.layouts "02000000000002000000000001000000ffffffff0100000004000000ffffffff"
.layouts "02000000000002000300000001000000ffffffff0100000004000000ffffffff"
.layouts "02000000"

.ownership "010000000200000003030000040000000300000001000000ffffffff01000000"
.ownership "ffffffff080000000000000001000000ffffffff040002000800000000000000"
.ownership "01000000ffffffff04000000ffffffff01000000ffffffff04000000ffffffff"
.ownership "01000100080000000000000008000000000000000500020001000000ffffffff"
.ownership "080000000000000001000000ffffffff080000000000000001000000ffffffff"
.ownership "04000000ffffffff"

.entry 0

.function main 0 3 0 int 1
  PUSH_I64 3
  STORE_LOCAL 0
  LOAD_LOCAL 0
  PUSH_BOOL 1
  CALL 1
  CALL 2
  OWN_STORE_LOCAL 1
  LOAD_LOCAL 0
  PUSH_I64 17
  ADD
  STORE_LOCAL 0
  OWN_MOVE_LOCAL 1
  LOAD_LOCAL 0
  PUSH_I64 1
  ADD
  CALL 3
  STORE_LOCAL 2
  LOAD_LOCAL 2
  RET
.end

.function make 2 4 0 struct 1
  LOAD_LOCAL 0
  STORE_LOCAL 2
  LOAD_LOCAL 1
  STORE_LOCAL 3
  LOAD_LOCAL 2
  LOAD_LOCAL 3
  OWN_PACK 0
  RET
.end

.parameters 1 int bool

.function forward 1 1 0 struct 1
  OWN_MOVE_LOCAL 0
  RET
.end

.parameters 2 struct

.function consume 2 5 0 int 1
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 2
  REGION_BEGIN
  BORROW_LOCAL_SHARED 2 2
  REF_GET 2 0
  REGION_END
  STORE_LOCAL 3
  REGION_BEGIN
  BORROW_LOCAL_SHARED 2 2
  REF_GET 2 1
  REGION_END
  STORE_LOCAL 4
  LOAD_LOCAL 3
  LOAD_LOCAL 1
  ADD
  OWN_UNPACK_LOCAL 2
  POP
  POP
  RET
.end

.parameters 3 struct int

