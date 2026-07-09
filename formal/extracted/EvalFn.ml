open BinInt
open Bool
open Datatypes
open Decimal
open List
open Nat
open Number
open Semantics
open String
open Syntax

(** val eval_fn : int -> env -> expr -> (env * coq_val) option **)

let rec eval_fn fuel renv e =
  (fun fO fS n -> if n = 0 then fO () else fS (n - 1))
    (fun _ -> None)
    (fun n ->
    match e with
    | EInt z -> Some (renv,(VInt z))
    | EBool b -> Some (renv,(VBool b))
    | EString s -> Some (renv,(VString s))
    | EUnit -> Some (renv,VUnit)
    | EVar x ->
      (match env_lookup x renv with
       | Some v -> Some (renv,v)
       | None -> None)
    | EBinOp (op, e1, e2) ->
      (match eval_fn n renv e1 with
       | Some p ->
         let renv1,v1 = p in
         (match eval_fn n renv1 e2 with
          | Some p0 ->
            let renv2,v2 = p0 in
            (match v1 with
             | VInt n1 ->
               (match v2 with
                | VInt n2 ->
                  if is_arith_op op
                  then (match eval_arith_binop op n1 n2 with
                        | Some v -> Some (renv2,v)
                        | None -> None)
                  else if is_cmp_op op
                       then (match eval_cmp_binop op n1 n2 with
                             | Some v -> Some (renv2,v)
                             | None -> None)
                       else None
                | _ -> None)
             | VBool b1 ->
               (match v2 with
                | VBool b2 ->
                  (match op with
                   | OpEq -> Some (renv2,(VBool (Bool.eqb b1 b2)))
                   | OpNe -> Some (renv2,(VBool (negb (Bool.eqb b1 b2))))
                   | OpAnd -> Some (renv2,(VBool ((&&) b1 b2)))
                   | OpOr -> Some (renv2,(VBool ((||) b1 b2)))
                   | _ -> None)
                | _ -> None)
             | VString s1 ->
               (match v2 with
                | VString s2 ->
                  (match op with
                   | OpEq -> Some (renv2,(VBool (eqb s1 s2)))
                   | OpNe -> Some (renv2,(VBool (negb (eqb s1 s2))))
                   | OpStrCat -> Some (renv2,(VString (append s1 s2)))
                   | _ -> None)
                | _ -> None)
             | _ -> None)
          | None -> None)
       | None -> None)
    | EUnOp (op0, e0) ->
      (match eval_fn n renv e0 with
       | Some p ->
         let renv1,v0 = p in
         (match op0 with
          | OpNeg ->
            (match v0 with
             | VInt z -> Some (renv1,(VInt (Z.opp z)))
             | _ -> None)
          | OpNot ->
            (match v0 with
             | VBool b -> Some (renv1,(VBool (negb b)))
             | _ -> None)
          | OpStrLen ->
            (match v0 with
             | VString s -> Some (renv1,(VInt (Z.of_nat (length s))))
             | _ -> None)
          | OpArrayLen ->
            (match v0 with
             | VArray vs ->
               Some (renv1,(VInt (Z.of_nat (Datatypes.length vs))))
             | _ -> None))
       | None -> None)
    | EIf (cond, e_then, e_else) ->
      (match eval_fn n renv cond with
       | Some p ->
         let renv1,v = p in
         (match v with
          | VBool b ->
            if b then eval_fn n renv1 e_then else eval_fn n renv1 e_else
          | _ -> None)
       | None -> None)
    | ELet (x, e1, e2) ->
      (match eval_fn n renv e1 with
       | Some p ->
         let renv1,v1 = p in
         (match eval_fn n (ECons (x, v1, renv1)) e2 with
          | Some p0 ->
            let e0,v2 = p0 in
            (match e0 with
             | ENil -> None
             | ECons (_, _, renv_out) -> Some (renv_out,v2))
          | None -> None)
       | None -> None)
    | ESet (x, e0) ->
      (match eval_fn n renv e0 with
       | Some p ->
         let renv1,v = p in
         (match env_lookup x renv1 with
          | Some _ -> Some ((env_update x v renv1),VUnit)
          | None -> None)
       | None -> None)
    | ESeq (e1, e2) ->
      (match eval_fn n renv e1 with
       | Some p -> let renv1,_ = p in eval_fn n renv1 e2
       | None -> None)
    | EWhile (cond, body) ->
      (match eval_fn n renv cond with
       | Some p ->
         let renv1,v = p in
         (match v with
          | VBool b ->
            if b
            then (match eval_fn n renv1 body with
                  | Some p0 ->
                    let renv2,_ = p0 in eval_fn n renv2 (EWhile (cond, body))
                  | None -> None)
            else Some (renv1,VUnit)
          | _ -> None)
       | None -> None)
    | ELam (x, _, body) -> Some (renv,(VClos (x, body, renv)))
    | EApp (e1, e2) ->
      (match eval_fn n renv e1 with
       | Some p ->
         let renv1,v = p in
         (match v with
          | VClos (x, body, clos_env) ->
            (match eval_fn n renv1 e2 with
             | Some p0 ->
               let renv2,v2 = p0 in
               (match eval_fn n (ECons (x, v2, clos_env)) body with
                | Some p1 -> let _,v0 = p1 in Some (renv2,v0)
                | None -> None)
             | None -> None)
          | VFixClos (f, x, body, clos_env) ->
            (match eval_fn n renv1 e2 with
             | Some p0 ->
               let renv2,v2 = p0 in
               (match eval_fn n (ECons (x, v2, (ECons (f, (VFixClos (f, x,
                        body, clos_env)), clos_env)))) body with
                | Some p1 -> let _,v0 = p1 in Some (renv2,v0)
                | None -> None)
             | None -> None)
          | _ -> None)
       | None -> None)
    | EFix (f, x, _, _, body) -> Some (renv,(VFixClos (f, x, body, renv)))
    | EArray es ->
      let eval_list =
        let rec eval_list fuel0 env0 = function
        | [] -> Some (env0,[])
        | x::rest ->
          ((fun fO fS n -> if n = 0 then fO () else fS (n - 1))
             (fun _ -> None)
             (fun m ->
             match eval_fn m env0 x with
             | Some p ->
               let env1,v = p in
               (match eval_list m env1 rest with
                | Some p0 -> let env2,vs = p0 in Some (env2,(v::vs))
                | None -> None)
             | None -> None)
             fuel0)
        in eval_list
      in
      (match eval_list n renv es with
       | Some p -> let renv',vs = p in Some (renv',(VArray vs))
       | None -> None)
    | EIndex (e1, e2) ->
      (match eval_fn n renv e1 with
       | Some p ->
         let renv1,v = p in
         (match v with
          | VArray vs ->
            (match eval_fn n renv1 e2 with
             | Some p0 ->
               let renv2,v0 = p0 in
               (match v0 with
                | VInt idx ->
                  (match nth_error vs (Z.to_nat idx) with
                   | Some v1 -> Some (renv2,v1)
                   | None -> None)
                | _ -> None)
             | None -> None)
          | _ -> None)
       | None -> None)
    | EArraySet (e1, e2, e3) ->
      (match eval_fn n renv e1 with
       | Some p ->
         let renv1,v = p in
         (match v with
          | VArray vs ->
            (match eval_fn n renv1 e2 with
             | Some p0 ->
               let renv2,v0 = p0 in
               (match v0 with
                | VInt idx ->
                  (match eval_fn n renv2 e3 with
                   | Some p1 ->
                     let renv3,v1 = p1 in
                     Some (renv3,(VArray (list_update (Z.to_nat idx) v1 vs)))
                   | None -> None)
                | _ -> None)
             | None -> None)
          | _ -> None)
       | None -> None)
    | EArrayPush (e1, e2) ->
      (match eval_fn n renv e1 with
       | Some p ->
         let renv1,v = p in
         (match v with
          | VArray vs ->
            (match eval_fn n renv1 e2 with
             | Some p0 ->
               let renv2,v0 = p0 in Some (renv2,(VArray (app vs (v0::[]))))
             | None -> None)
          | _ -> None)
       | None -> None)
    | ERecord fes ->
      let eval_fields =
        let rec eval_fields fuel0 env0 = function
        | [] -> Some (env0,[])
        | p::rest ->
          let fname,fe = p in
          ((fun fO fS n -> if n = 0 then fO () else fS (n - 1))
             (fun _ -> None)
             (fun m ->
             match eval_fn m env0 fe with
             | Some p0 ->
               let env1,v = p0 in
               (match eval_fields m env1 rest with
                | Some p1 -> let env2,fvs = p1 in Some (env2,((fname,v)::fvs))
                | None -> None)
             | None -> None)
             fuel0)
        in eval_fields
      in
      (match eval_fields n renv fes with
       | Some p -> let renv',fvs = p in Some (renv',(VRecord fvs))
       | None -> None)
    | EField (e0, f) ->
      (match eval_fn n renv e0 with
       | Some p ->
         let renv1,v = p in
         (match v with
          | VRecord fvs ->
            (match assoc_lookup f fvs with
             | Some v0 -> Some (renv1,v0)
             | None -> None)
          | _ -> None)
       | None -> None)
    | ESetField (x, f, e0) ->
      (match eval_fn n renv e0 with
       | Some p ->
         let renv1,v = p in
         (match env_lookup x renv1 with
          | Some v0 ->
            (match v0 with
             | VRecord fvs ->
               (match assoc_lookup f fvs with
                | Some _ ->
                  Some
                    ((env_update x (VRecord (assoc_update f v fvs)) renv1),VUnit)
                | None -> None)
             | _ -> None)
          | None -> None)
       | None -> None)
    | EConstruct (tag, e0, _) ->
      (match eval_fn n renv e0 with
       | Some p -> let renv1,v = p in Some (renv1,(VConstruct (tag, v)))
       | None -> None)
    | EMatch (e0, branches) ->
      (match eval_fn n renv e0 with
       | Some p ->
         let renv1,v0 = p in
         (match v0 with
          | VConstruct (tag, v) ->
            (match find_branch tag branches with
             | Some p0 ->
               let x,body = p0 in
               (match eval_fn n (ECons (x, v, renv1)) body with
                | Some p1 ->
                  let e1,v_result = p1 in
                  (match e1 with
                   | ENil -> None
                   | ECons (_, _, renv_out) -> Some (renv_out,v_result))
                | None -> None)
             | None -> None)
          | _ -> None)
       | None -> None)
    | EStrIndex (e1, e2) ->
      (match eval_fn n renv e1 with
       | Some p ->
         let renv1,v = p in
         (match v with
          | VString s ->
            (match eval_fn n renv1 e2 with
             | Some p0 ->
               let renv2,v0 = p0 in
               (match v0 with
                | VInt idx ->
                  Some (renv2,(VString (substring (Z.to_nat idx) (succ 0) s)))
                | _ -> None)
             | None -> None)
          | _ -> None)
       | None -> None))
    fuel

(** val default_fuel : int **)

let default_fuel =
  of_num_uint (UIntDecimal (D1 (D0 (D0 (D0 (D0 Nil))))))

(** val eval_program : expr -> (env * coq_val) option **)

let eval_program e =
  eval_fn default_fuel ENil e
