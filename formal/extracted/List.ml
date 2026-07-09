
(** val nth_error : 'a1 list -> int -> 'a1 option **)

let rec nth_error l n =
  (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
    (fun _ -> match l with
              | [] -> None
              | x::_ -> Some x)
    (fun n0 -> match l with
               | [] -> None
               | _::l' -> nth_error l' n0)
    n
