open String

type ty =
| TInt
| TBool
| TString
| TUnit
| TArrow of ty * ty
| TArray of ty
| TRecord of (char list * ty) list
| TVariant of (char list * ty) list

type binop =
| OpAdd
| OpSub
| OpMul
| OpDiv
| OpMod
| OpEq
| OpNe
| OpLt
| OpLe
| OpGt
| OpGe
| OpAnd
| OpOr
| OpStrCat

type unop =
| OpNeg
| OpNot
| OpStrLen
| OpArrayLen

type expr =
| EInt of Big_int_Z.big_int
| EBool of bool
| EString of char list
| EUnit
| EVar of char list
| EBinOp of binop * expr * expr
| EUnOp of unop * expr
| EIf of expr * expr * expr
| ELet of char list * expr * expr
| ESet of char list * expr
| ESeq of expr * expr
| EWhile of expr * expr
| ELam of char list * ty * expr
| EApp of expr * expr
| EFix of char list * char list * ty * ty * expr
| EArray of expr list
| EIndex of expr * expr
| EArraySet of expr * expr * expr
| EArrayPush of expr * expr
| ERecord of (char list * expr) list
| EField of expr * char list
| ESetField of char list * char list * expr
| EConstruct of char list * expr * ty
| EMatch of expr * ((char list * char list) * expr) list
| EStrIndex of expr * expr

type coq_val =
| VInt of Big_int_Z.big_int
| VBool of bool
| VString of char list
| VUnit
| VClos of char list * expr * env
| VFixClos of char list * char list * expr * env
| VArray of coq_val list
| VRecord of (char list * coq_val) list
| VConstruct of char list * coq_val
and env =
| ENil
| ECons of char list * coq_val * env

(** val env_lookup : char list -> env -> coq_val option **)

let rec env_lookup x = function
| ENil -> None
| ECons (y, v, rest) -> if eqb x y then Some v else env_lookup x rest

(** val env_update : char list -> coq_val -> env -> env **)

let rec env_update x v = function
| ENil -> ENil
| ECons (y, v', rest) ->
  if eqb x y then ECons (y, v, rest) else ECons (y, v', (env_update x v rest))

(** val assoc_lookup : char list -> (char list * 'a1) list -> 'a1 option **)

let rec assoc_lookup x = function
| [] -> None
| p::rest -> let y,v = p in if eqb x y then Some v else assoc_lookup x rest

(** val assoc_update :
    char list -> 'a1 -> (char list * 'a1) list -> (char list * 'a1) list **)

let rec assoc_update x v = function
| [] -> []
| p::rest ->
  let y,v' = p in
  if eqb x y then (y,v)::rest else (y,v')::(assoc_update x v rest)

(** val list_update : int -> 'a1 -> 'a1 list -> 'a1 list **)

let rec list_update n v = function
| [] -> []
| x::rest ->
  ((fun fO fS n -> if n = 0 then fO () else fS (n - 1))
     (fun _ -> v::rest)
     (fun n' -> x::(list_update n' v rest))
     n)

(** val find_branch :
    char list -> ((char list * char list) * expr) list -> (char list * expr)
    option **)

let rec find_branch tag = function
| [] -> None
| p::rest ->
  let p0,body = p in
  let t,x = p0 in if eqb tag t then Some (x,body) else find_branch tag rest
