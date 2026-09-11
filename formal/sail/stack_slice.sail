// I model a bounded NanoISA primary-opcode slice, not the complete VM.
default Order dec
$include <prelude.sail>

union instruction = {
  Nop : unit,
  Push : bits(64),
  Dup : unit,
  Pop : unit,
  Swap : unit
}

val decode : list(bits(8)) -> option((instruction, list(bits(8))))
function decode(code) = match code {
  opcode :: rest => match opcode {
    0x00 => Some((Nop(), rest)),
    0x01 => match rest {
      b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: tail =>
        Some((Push(b7 @ b6 @ b5 @ b4 @ b3 @ b2 @ b1 @ b0), tail)),
      _ => None()
    },
    0x07 => Some((Dup(), rest)),
    0x08 => Some((Pop(), rest)),
    0x09 => Some((Swap(), rest)),
    _ => None()
  },
  _ => None()
}

val execute : (instruction, list(bits(64))) -> option(list(bits(64)))
function execute(insn, stack) = match (insn, stack) {
  (Nop(), _) => Some(stack),
  (Push(value), _) => Some(value :: stack),
  (Dup(), value :: rest) => Some(value :: value :: rest),
  (Pop(), _ :: rest) => Some(rest),
  (Swap(), first :: second :: rest) => Some(second :: first :: rest),
  _ => None()
}
