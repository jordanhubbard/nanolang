
(** val eqb : char list -> char list -> bool **)

let rec eqb s1 s2 =
  match s1 with
  | [] -> (match s2 with
           | [] -> true
           | _::_ -> false)
  | c1::s1' ->
    (match s2 with
     | [] -> false
     | c2::s2' -> if (=) c1 c2 then eqb s1' s2' else false)

(** val append : char list -> char list -> char list **)

let rec append s1 s2 =
  match s1 with
  | [] -> s2
  | c::s1' -> c::(append s1' s2)

(** val length : char list -> int **)

let rec length = function
| [] -> 0
| _::s' -> succ (length s')

(** val substring : int -> int -> char list -> char list **)

let rec substring n m s =
  (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
    (fun _ ->
    (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
      (fun _ -> [])
      (fun m' -> match s with
                 | [] -> s
                 | c::s' -> c::(substring 0 m' s'))
      m)
    (fun n' -> match s with
               | [] -> s
               | _::s' -> substring n' m s')
    n
