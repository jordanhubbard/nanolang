.string "main"
.string "/private/tmp/nanolang-byte-literal-baseline.nano"
.string "nano.source_file"

.metadata 2 1

.flag debug_info

.debug 0 1 18
.debug 16 1 46
.debug 41 1 74

.entry 0

.function main 0 1 0 int 1
  PUSH_I64 300
  ARR_LITERAL 1 1
  STORE_LOCAL 0
  LOAD_LOCAL 0
  PUSH_I64 0
  ARR_GET
  CAST_INT
  PUSH_I64 44
  EQ
  ASSERT
  PUSH_I64 0
  RET
.end

