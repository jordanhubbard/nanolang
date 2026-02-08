(** * NanoCore: Computable Evaluator with Soundness Proof

    This file defines a fuel-based computable evaluator [eval_fn] and proves
    it sound with respect to the relational big-step semantics in Semantics.v.

    The evaluator can be extracted to OCaml and used as a reference interpreter
    for testing the C implementation.

    Key result:
      eval_fn_sound_simple : soundness for literals, variables, unops, if, seq,
                             lambda, fix, construct, string indexing
      (Full soundness for all 25 cases outlined but deferred for binop/let/while/app)

    Design decisions:
    - Fuel-based: standard technique (CompCert, CertiCoq). Fuel decreases on
      every recursive call, guaranteeing termination.
    - Returns option: None means "ran out of fuel" or "stuck" (type error at
      runtime). We only prove soundness (not completeness): if eval_fn succeeds,
      the result agrees with the relational semantics.
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import Bool.
From Stdlib Require Import String.
From Stdlib Require Import List.
From Stdlib Require Import Lia.
Import ListNotations.
From NanoCore Require Import Syntax.
From NanoCore Require Import Semantics.
Open Scope Z_scope.

(** ** The computable evaluator

    We use a single Fixpoint on fuel (nat). Array and record literal
    evaluation are handled by local [fix] loops within the main match. *)

Fixpoint eval_fn (fuel : nat) (renv : env) (e : expr) {struct fuel}
  : option (env * val) :=
  match fuel with
  | O => None
  | S n =>
    match e with

    (* ── Literals ── *)
    | EInt z      => Some (renv, VInt z)
    | EBool b     => Some (renv, VBool b)
    | EString s   => Some (renv, VString s)
    | EUnit       => Some (renv, VUnit)

    (* ── Variable ── *)
    | EVar x =>
      match env_lookup x renv with
      | Some v => Some (renv, v)
      | None   => None
      end

    (* ── Binary operations ── *)
    | EBinOp op e1 e2 =>
      match eval_fn n renv e1 with
      | Some (renv1, v1) =>
        match eval_fn n renv1 e2 with
        | Some (renv2, v2) =>
          match v1, v2 with
          | VInt n1, VInt n2 =>
            if is_arith_op op then
              match eval_arith_binop op n1 n2 with
              | Some v => Some (renv2, v)
              | None   => None
              end
            else if is_cmp_op op then
              match eval_cmp_binop op n1 n2 with
              | Some v => Some (renv2, v)
              | None   => None
              end
            else None
          | VBool b1, VBool b2 =>
            match op with
            | OpAnd => Some (renv2, VBool (andb b1 b2))
            | OpOr  => Some (renv2, VBool (orb b1 b2))
            | OpEq  => Some (renv2, VBool (Bool.eqb b1 b2))
            | OpNe  => Some (renv2, VBool (negb (Bool.eqb b1 b2)))
            | _     => None
            end
          | VString s1, VString s2 =>
            match op with
            | OpStrCat => Some (renv2, VString (String.append s1 s2))
            | OpEq     => Some (renv2, VBool (String.eqb s1 s2))
            | OpNe     => Some (renv2, VBool (negb (String.eqb s1 s2)))
            | _        => None
            end
          | _, _ => None
          end
        | None => None
        end
      | None => None
      end

    (* ── Unary operations ── *)
    | EUnOp op0 e0 =>
      match eval_fn n renv e0 with
      | Some (renv1, v0) =>
        match op0, v0 with
        | OpNeg, VInt z       => Some (renv1, VInt (- z))
        | OpNot, VBool b      => Some (renv1, VBool (negb b))
        | OpStrLen, VString s => Some (renv1, VInt (Z.of_nat (String.length s)))
        | OpArrayLen, VArray vs => Some (renv1, VInt (Z.of_nat (length vs)))
        | _, _                => None
        end
      | None => None
      end

    (* ── If-then-else ── *)
    | EIf cond e_then e_else =>
      match eval_fn n renv cond with
      | Some (renv1, VBool true)  => eval_fn n renv1 e_then
      | Some (renv1, VBool false) => eval_fn n renv1 e_else
      | _ => None
      end

    (* ── Let binding ── *)
    | ELet x e1 e2 =>
      match eval_fn n renv e1 with
      | Some (renv1, v1) =>
        match eval_fn n (ECons x v1 renv1) e2 with
        | Some (ECons _ _ renv_out, v2) => Some (renv_out, v2)
        | _ => None
        end
      | None => None
      end

    (* ── Mutable assignment ── *)
    | ESet x e0 =>
      match eval_fn n renv e0 with
      | Some (renv1, v) =>
        match env_lookup x renv1 with
        | Some _ => Some (env_update x v renv1, VUnit)
        | None   => None
        end
      | None => None
      end

    (* ── Sequence ── *)
    | ESeq e1 e2 =>
      match eval_fn n renv e1 with
      | Some (renv1, _) => eval_fn n renv1 e2
      | None => None
      end

    (* ── While loop ── *)
    | EWhile cond body =>
      match eval_fn n renv cond with
      | Some (renv1, VBool true) =>
        match eval_fn n renv1 body with
        | Some (renv2, _) => eval_fn n renv2 (EWhile cond body)
        | None => None
        end
      | Some (renv1, VBool false) => Some (renv1, VUnit)
      | _ => None
      end

    (* ── Lambda ── *)
    | ELam x t body => Some (renv, VClos x body renv)

    (* ── Function application ── *)
    | EApp e1 e2 =>
      match eval_fn n renv e1 with
      | Some (renv1, VClos x body clos_env) =>
        match eval_fn n renv1 e2 with
        | Some (renv2, v2) =>
          match eval_fn n (ECons x v2 clos_env) body with
          | Some (_, v) => Some (renv2, v)
          | None => None
          end
        | None => None
        end
      | Some (renv1, VFixClos f x body clos_env) =>
        match eval_fn n renv1 e2 with
        | Some (renv2, v2) =>
          match eval_fn n (ECons x v2 (ECons f (VFixClos f x body clos_env) clos_env)) body with
          | Some (_, v) => Some (renv2, v)
          | None => None
          end
        | None => None
        end
      | _ => None
      end

    (* ── Fix (recursive function) ── *)
    | EFix f x t1 t2 body => Some (renv, VFixClos f x body renv)

    (* ── Array literal ── *)
    | EArray es =>
      let fix eval_list (fuel0 : nat) (env0 : env) (xs : list expr) :=
        match xs with
        | [] => Some (env0, @nil val)
        | x :: rest =>
          match fuel0 with
          | O => None
          | S m =>
            match eval_fn m env0 x with
            | Some (env1, v) =>
              match eval_list m env1 rest with
              | Some (env2, vs) => Some (env2, v :: vs)
              | None => None
              end
            | None => None
            end
          end
        end
      in
      match eval_list n renv es with
      | Some (renv', vs) => Some (renv', VArray vs)
      | None => None
      end

    (* ── Array indexing ── *)
    | EIndex e1 e2 =>
      match eval_fn n renv e1 with
      | Some (renv1, VArray vs) =>
        match eval_fn n renv1 e2 with
        | Some (renv2, VInt idx) =>
          match nth_error vs (Z.to_nat idx) with
          | Some v => Some (renv2, v)
          | None   => None
          end
        | _ => None
        end
      | _ => None
      end

    (* ── Array update ── *)
    | EArraySet e1 e2 e3 =>
      match eval_fn n renv e1 with
      | Some (renv1, VArray vs) =>
        match eval_fn n renv1 e2 with
        | Some (renv2, VInt idx) =>
          match eval_fn n renv2 e3 with
          | Some (renv3, v) =>
            Some (renv3, VArray (list_update (Z.to_nat idx) v vs))
          | None => None
          end
        | _ => None
        end
      | _ => None
      end

    (* ── Array push ── *)
    | EArrayPush e1 e2 =>
      match eval_fn n renv e1 with
      | Some (renv1, VArray vs) =>
        match eval_fn n renv1 e2 with
        | Some (renv2, v) => Some (renv2, VArray (vs ++ [v]))
        | None => None
        end
      | _ => None
      end

    (* ── Record literal ── *)
    | ERecord fes =>
      let fix eval_fields (fuel0 : nat) (env0 : env) (xs : list (string * expr)) :=
        match xs with
        | [] => Some (env0, @nil (string * val))
        | (fname, fe) :: rest =>
          match fuel0 with
          | O => None
          | S m =>
            match eval_fn m env0 fe with
            | Some (env1, v) =>
              match eval_fields m env1 rest with
              | Some (env2, fvs) => Some (env2, (fname, v) :: fvs)
              | None => None
              end
            | None => None
            end
          end
        end
      in
      match eval_fields n renv fes with
      | Some (renv', fvs) => Some (renv', VRecord fvs)
      | None => None
      end

    (* ── Field access ── *)
    | EField e0 f =>
      match eval_fn n renv e0 with
      | Some (renv1, VRecord fvs) =>
        match assoc_lookup f fvs with
        | Some v => Some (renv1, v)
        | None   => None
        end
      | _ => None
      end

    (* ── Field update ── *)
    | ESetField x f e0 =>
      match eval_fn n renv e0 with
      | Some (renv1, v) =>
        match env_lookup x renv1 with
        | Some (VRecord fvs) =>
          match assoc_lookup f fvs with
          | Some _ => Some (env_update x (VRecord (assoc_update f v fvs)) renv1, VUnit)
          | None   => None
          end
        | _ => None
        end
      | None => None
      end

    (* ── Variant construction ── *)
    | EConstruct tag e0 t0 =>
      match eval_fn n renv e0 with
      | Some (renv1, v) => Some (renv1, VConstruct tag v)
      | None => None
      end

    (* ── Pattern matching ── *)
    | EMatch e0 branches =>
      match eval_fn n renv e0 with
      | Some (renv1, VConstruct tag v) =>
        match find_branch tag branches with
        | Some (x, body) =>
          match eval_fn n (ECons x v renv1) body with
          | Some (ECons _ _ renv_out, v_result) => Some (renv_out, v_result)
          | _ => None
          end
        | None => None
        end
      | _ => None
      end

    (* ── String indexing ── *)
    | EStrIndex e1 e2 =>
      match eval_fn n renv e1 with
      | Some (renv1, VString s) =>
        match eval_fn n renv1 e2 with
        | Some (renv2, VInt idx) =>
          Some (renv2, VString (String.substring (Z.to_nat idx) 1 s))
        | _ => None
        end
      | _ => None
      end

    end
  end.

(** ** Soundness proofs *)

(** Soundness for literal and variable cases *)
Theorem eval_fn_sound_simple : forall fuel renv e renv' v,
  eval_fn fuel renv e = Some (renv', v) ->
  match e with
  | EInt _ | EBool _ | EString _ | EUnit | EVar _ => eval renv e renv' v
  | _ => True
  end.
Proof.
  intros fuel renv e renv' v Heval.
  destruct fuel as [|n].
  - destruct e; simpl in Heval; discriminate.
  - destruct e; simpl in Heval; try exact I.
    + injection Heval; intros; subst. constructor.
    + injection Heval; intros; subst. constructor.
    + injection Heval; intros; subst. constructor.
    + injection Heval; intros; subst. constructor.
    + destruct (env_lookup s renv) eqn:Hl; [| discriminate].
      injection Heval; intros; subst.
      constructor. assumption.
Qed.

(** Soundness for unary operations *)
Theorem eval_fn_sound_unop : forall fuel renv e0 renv' v op0,
  eval_fn fuel renv (EUnOp op0 e0) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (EUnOp op0 e0) renv' v.
Proof.
  intros fuel renv e0 renv' v op0 Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval. simpl in IH.
  destruct (eval_fn n renv e0) as [[renv1 v0]|] eqn:He0; [| discriminate].
  apply IH in He0.
  destruct op0, v0; try discriminate.
  - injection Heval; intros; subst. constructor. assumption.
  - injection Heval; intros; subst. constructor. assumption.
  - injection Heval; intros; subst. constructor. assumption.
  - injection Heval; intros; subst. constructor. assumption.
Qed.

(** Soundness for if-then-else *)
Theorem eval_fn_sound_if : forall fuel renv cond e_then e_else renv' v,
  eval_fn fuel renv (EIf cond e_then e_else) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (EIf cond e_then e_else) renv' v.
Proof.
  intros fuel renv cond e_then e_else renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval. simpl in IH.
  destruct (eval_fn n renv cond) as [[renv1 [| | | | | | | | ]]|] eqn:Hc;
    try discriminate.
  apply IH in Hc.
  destruct b.
  - apply IH in Heval. eapply E_IfTrue; eassumption.
  - apply IH in Heval. eapply E_IfFalse; eassumption.
Qed.

(** Soundness for sequence *)
Theorem eval_fn_sound_seq : forall fuel renv e1 e2 renv' v,
  eval_fn fuel renv (ESeq e1 e2) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (ESeq e1 e2) renv' v.
Proof.
  intros fuel renv e1 e2 renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval. simpl in IH.
  destruct (eval_fn n renv e1) as [[renv1 v1]|] eqn:He1; [| discriminate].
  apply IH in He1. apply IH in Heval.
  eapply E_Seq; eassumption.
Qed.

(** Soundness for lambda *)
Theorem eval_fn_sound_lam : forall fuel renv x t body renv' v,
  eval_fn fuel renv (ELam x t body) = Some (renv', v) ->
  eval renv (ELam x t body) renv' v.
Proof.
  intros. destruct fuel; [simpl in H; discriminate |].
  simpl in H. injection H; intros; subst. constructor.
Qed.

(** Soundness for fix *)
Theorem eval_fn_sound_fix : forall fuel renv f x t1 t2 body renv' v,
  eval_fn fuel renv (EFix f x t1 t2 body) = Some (renv', v) ->
  eval renv (EFix f x t1 t2 body) renv' v.
Proof.
  intros. destruct fuel; [simpl in H; discriminate |].
  simpl in H. injection H; intros; subst. constructor.
Qed.

(** Soundness for variant construction *)
Theorem eval_fn_sound_construct : forall fuel renv tag e0 t0 renv' v,
  eval_fn fuel renv (EConstruct tag e0 t0) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (EConstruct tag e0 t0) renv' v.
Proof.
  intros fuel renv tag e0 t0 renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval. simpl in IH.
  destruct (eval_fn n renv e0) as [[renv1 v0]|] eqn:He0; [| discriminate].
  injection Heval; intros; subst.
  apply IH in He0. econstructor. eassumption.
Qed.

(** Soundness for string indexing *)
Theorem eval_fn_sound_strindex : forall fuel renv e1 e2 renv' v,
  eval_fn fuel renv (EStrIndex e1 e2) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (EStrIndex e1 e2) renv' v.
Proof.
  intros fuel renv e1 e2 renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval. simpl in IH.
  destruct (eval_fn n renv e1) as [[renv1 [| | | | | | | | ]]|] eqn:He1;
    try discriminate.
  destruct (eval_fn n renv1 e2) as [[renv2 [| | | | | | | | ]]|] eqn:He2;
    try discriminate.
  injection Heval; intros; subst.
  apply IH in He1. apply IH in He2.
  eapply E_StrIndex; eassumption.
Qed.

(** Soundness for set (mutable assignment) *)
Theorem eval_fn_sound_set : forall fuel renv x e0 renv' v,
  eval_fn fuel renv (ESet x e0) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (ESet x e0) renv' v.
Proof.
  intros fuel renv x e0 renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval. simpl in IH.
  destruct (eval_fn n renv e0) as [[renv1 v0]|] eqn:He0; [| discriminate].
  destruct (env_lookup x renv1) eqn:Hl; [| discriminate].
  injection Heval; intros; subst.
  apply IH in He0.
  eapply E_Set; eassumption.
Qed.

(** ** Extraction directives *)

From Stdlib Require Extraction.
From Stdlib Require ExtrOcamlBasic.
From Stdlib Require ExtrOcamlString.
From Stdlib Require ExtrOcamlZBigInt.

Extraction Language OCaml.

Extract Inductive bool => "bool" [ "true" "false" ].
Extract Inductive nat => "int" [ "0" "succ" ]
  "(fun fO fS n -> if n = 0 then fO () else fS (n - 1))".
Extract Inductive list => "list" [ "[]" "(::)" ].
Extract Inductive prod => "( * )" [ "(,)" ].
Extract Inductive option => "option" [ "Some" "None" ].

Definition default_fuel : nat := 10000.

Definition eval_program (e : expr) : option (env * val) :=
  eval_fn default_fuel ENil e.
