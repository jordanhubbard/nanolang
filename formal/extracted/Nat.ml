open Decimal
open Hexadecimal
open Number

(** val add : int -> int -> int **)

let rec add n m =
  (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
    (fun _ -> m)
    (fun p -> succ (add p m))
    n

(** val tail_add : int -> int -> int **)

let rec tail_add n m =
  (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
    (fun _ -> m)
    (fun n0 -> tail_add n0 (succ m))
    n

(** val tail_addmul : int -> int -> int -> int **)

let rec tail_addmul r n m =
  (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
    (fun _ -> r)
    (fun n0 -> tail_addmul (tail_add m r) n0 m)
    n

(** val tail_mul : int -> int -> int **)

let tail_mul n m =
  tail_addmul 0 n m

(** val of_uint_acc : Decimal.uint -> int -> int **)

let rec of_uint_acc d acc =
  match d with
  | Decimal.Nil -> acc
  | Decimal.D0 d0 ->
    of_uint_acc d0
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        0)))))))))) acc)
  | Decimal.D1 d0 ->
    of_uint_acc d0 (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        0)))))))))) acc))
  | Decimal.D2 d0 ->
    of_uint_acc d0 (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        0)))))))))) acc)))
  | Decimal.D3 d0 ->
    of_uint_acc d0 (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        0)))))))))) acc))))
  | Decimal.D4 d0 ->
    of_uint_acc d0 (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        0)))))))))) acc)))))
  | Decimal.D5 d0 ->
    of_uint_acc d0 (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        0)))))))))) acc))))))
  | Decimal.D6 d0 ->
    of_uint_acc d0 (succ (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        0)))))))))) acc)))))))
  | Decimal.D7 d0 ->
    of_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        0)))))))))) acc))))))))
  | Decimal.D8 d0 ->
    of_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        0)))))))))) acc)))))))))
  | Decimal.D9 d0 ->
    of_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        0)))))))))) acc))))))))))

(** val of_uint : Decimal.uint -> int **)

let of_uint d =
  of_uint_acc d 0

(** val of_hex_uint_acc : Hexadecimal.uint -> int -> int **)

let rec of_hex_uint_acc d acc =
  match d with
  | Nil -> acc
  | D0 d0 ->
    of_hex_uint_acc d0
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc)
  | D1 d0 ->
    of_hex_uint_acc d0 (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc))
  | D2 d0 ->
    of_hex_uint_acc d0 (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc)))
  | D3 d0 ->
    of_hex_uint_acc d0 (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc))))
  | D4 d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc)))))
  | D5 d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc))))))
  | D6 d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc)))))))
  | D7 d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc))))))))
  | D8 d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc)))))))))
  | D9 d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc))))))))))
  | Da d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ (succ (succ
      (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc)))))))))))
  | Db d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ (succ (succ
      (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc))))))))))))
  | Dc d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ (succ (succ
      (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc)))))))))))))
  | Dd d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ (succ (succ
      (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc))))))))))))))
  | De d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ (succ (succ
      (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc)))))))))))))))
  | Df d0 ->
    of_hex_uint_acc d0 (succ (succ (succ (succ (succ (succ (succ (succ (succ
      (succ (succ (succ (succ (succ (succ
      (tail_mul (succ (succ (succ (succ (succ (succ (succ (succ (succ (succ
        (succ (succ (succ (succ (succ (succ 0)))))))))))))))) acc))))))))))))))))

(** val of_hex_uint : Hexadecimal.uint -> int **)

let of_hex_uint d =
  of_hex_uint_acc d 0

(** val of_num_uint : uint -> int **)

let of_num_uint = function
| UIntDecimal d0 -> of_uint d0
| UIntHexadecimal d0 -> of_hex_uint d0
