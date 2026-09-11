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
  0x00 :: rest => Some((Nop(), rest)),
  0x01 :: b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest =>
    Some((Push(b7 @ b6 @ b5 @ b4 @ b3 @ b2 @ b1 @ b0), rest)),
  0x07 :: rest => Some((Dup(), rest)),
  0x08 :: rest => Some((Pop(), rest)),
  0x09 :: rest => Some((Swap(), rest)),
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

val main : unit -> unit
function main() = {
  let empty : list(bits(64)) = [||];
  assert(match execute(Dup(), empty) { None() => true, _ => false });
  assert(match execute(Pop(), empty) { None() => true, _ => false });
  assert(match execute(Swap(), [|0x0000000000000001|]) { None() => true, _ => false });
  assert(match execute(Swap(), [|0x0000000000000001, 0xffffffffffffffff|]) {
    Some([|0xffffffffffffffff, 0x0000000000000001|]) => true, _ => false });
  assert(match decode([|0x01, 0xff|]) { None() => true, _ => false });
  assert(match decode([|0xff|]) { None() => true, _ => false });
  assert(match decode([|0x01, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00|]) {
    Some((Push(0x0102030405060708), [|0x00|])) => true, _ => false });
  print_endline("I passed the bounded NanoISA Sail smoke checks.")
}
