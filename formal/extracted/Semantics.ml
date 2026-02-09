open BinInt
open Datatypes
open Syntax

(** val eval_arith_binop :
    binop -> Big_int_Z.big_int -> Big_int_Z.big_int -> coq_val option **)

let eval_arith_binop op n1 n2 =
  match op with
  | OpAdd -> Some (VInt (Z.add n1 n2))
  | OpSub -> Some (VInt (Z.sub n1 n2))
  | OpMul -> Some (VInt (Z.mul n1 n2))
  | OpDiv ->
    if Z.eqb n2 Big_int_Z.zero_big_int
    then None
    else Some (VInt (Z.div n1 n2))
  | OpMod ->
    if Z.eqb n2 Big_int_Z.zero_big_int
    then None
    else Some (VInt (Z.modulo n1 n2))
  | _ -> None

(** val eval_cmp_binop :
    binop -> Big_int_Z.big_int -> Big_int_Z.big_int -> coq_val option **)

let eval_cmp_binop op n1 n2 =
  match op with
  | OpEq -> Some (VBool (Z.eqb n1 n2))
  | OpNe -> Some (VBool (negb (Z.eqb n1 n2)))
  | OpLt -> Some (VBool (Z.ltb n1 n2))
  | OpLe -> Some (VBool (Z.leb n1 n2))
  | OpGt -> Some (VBool (Z.ltb n2 n1))
  | OpGe -> Some (VBool (Z.leb n2 n1))
  | _ -> None

(** val is_arith_op : binop -> bool **)

let is_arith_op = function
| OpAdd -> true
| OpSub -> true
| OpMul -> true
| OpDiv -> true
| OpMod -> true
| _ -> false

(** val is_cmp_op : binop -> bool **)

let is_cmp_op = function
| OpEq -> true
| OpNe -> true
| OpLt -> true
| OpLe -> true
| OpGt -> true
| OpGe -> true
| _ -> false
