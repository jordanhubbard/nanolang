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
