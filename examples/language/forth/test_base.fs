\ test_base.fs — Numeric base (hex/decimal) tests

testing decimal hex base
T{ base @ -> 10 }T
T{ hex base @ decimal -> 16 }T
T{ base @ -> 10 }T

testing hex number parsing
T{ hex ff decimal -> 255 }T
T{ hex 10 decimal -> 16 }T
T{ hex 1a decimal -> 26 }T
T{ hex a decimal -> 10 }T

testing bl
T{ bl -> 32 }T

test-summary
