open Decimal
open Hexadecimal

type uint =
| UIntDecimal of Decimal.uint
| UIntHexadecimal of Hexadecimal.uint
