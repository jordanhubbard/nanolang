(** * NanoCore: Computable Evaluator with Soundness Proof

    I define a fuel-based computable evaluator [eval_fn] and prove it sound
    with respect to the relational semantics in Semantics.v.

    The evaluator can be extracted to OCaml and used as a reference interpreter
    for testing the C implementation.

    [eval_fn_sound_simple] covers literals and variables. Other named lemmas
    cover selected compound cases, assuming sound recursive evaluations.
    [eval_fn_and_short] and [eval_fn_or_short] establish skipped-right-operand
    behavior directly. Strong fuel induction connects all expression cases
    in [eval_fn_sound], without a recursive-soundness premise.

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

    (* I evaluate a logical right operand only when its value is needed. *)
    | EBinOp OpAnd e1 e2 =>
      match eval_fn n renv e1 with
      | Some (renv1, VBool false) => Some (renv1, VBool false)
      | Some (renv1, VBool true) =>
        match eval_fn n renv1 e2 with
        | Some (renv2, VBool b) => Some (renv2, VBool b)
        | _ => None
        end
      | _ => None
      end
    | EBinOp OpOr e1 e2 =>
      match eval_fn n renv e1 with
      | Some (renv1, VBool true) => Some (renv1, VBool true)
      | Some (renv1, VBool false) =>
        match eval_fn n renv1 e2 with
        | Some (renv2, VBool b) => Some (renv2, VBool b)
        | _ => None
        end
      | _ => None
      end

    (* ── Remaining binary operations ── *)
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

    | ETuple es =>
      let fix eval_list (fuel2 : nat) (renv2 : env) (elems : list expr)
          : option (env * list val) :=
        match elems with
        | [] => Some (renv2, [])
        | e0 :: rest =>
          match eval_fn fuel2 renv2 e0 with
          | Some (renv3, v0) =>
            match eval_list fuel2 renv3 rest with
            | Some (renv4, vs) => Some (renv4, v0 :: vs)
            | None => None
            end
          | None => None
          end
        end
      in
      match eval_list n renv es with
      | Some (renv', vs) => Some (renv', VTuple vs)
      | None => None
      end

    | ETupleIndex e0 i =>
      match eval_fn n renv e0 with
      | Some (renv1, VTuple vs) =>
        match nth_error vs i with
        | Some v => Some (renv1, v)
        | None => None
        end
      | _ => None
      end

    end
  end.

(** I preserve the left evaluation's environment without inspecting the right
    expression when its value cannot affect the logical result. *)
Theorem eval_fn_and_short : forall n r e1 e2 r1,
  eval_fn n r e1 = Some (r1, VBool false) ->
  eval_fn (S n) r (EBinOp OpAnd e1 e2) = Some (r1, VBool false).
Proof. intros n r e1 e2 r1 H. simpl. rewrite H. reflexivity. Qed.

Theorem eval_fn_or_short : forall n r e1 e2 r1,
  eval_fn n r e1 = Some (r1, VBool true) ->
  eval_fn (S n) r (EBinOp OpOr e1 e2) = Some (r1, VBool true).
Proof. intros n r e1 e2 r1 H. simpl. rewrite H. reflexivity. Qed.

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

(** I prove logical-operator soundness assuming sound recursive evaluations. *)
Theorem eval_fn_sound_logic : forall fuel renv op e1 e2 renv' v,
  is_logic_op op = true ->
  eval_fn fuel renv (EBinOp op e1 e2) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (EBinOp op e1 e2) renv' v.
Proof.
  intros fuel renv op e1 e2 renv' v Hop Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in IH.
  destruct op; simpl in Hop; try discriminate;
    simpl in Heval;
    destruct (eval_fn n renv e1) as [[r1 v1]|] eqn:Hleft; try discriminate;
    destruct v1; try discriminate;
    apply IH in Hleft; destruct b.
  - destruct (eval_fn n r1 e2) as [[r2 v2]|] eqn:Hright; [| discriminate].
    destruct v2; try discriminate.
    inversion Heval; subst. apply IH in Hright. eapply E_And_True; eassumption.
  - inversion Heval; subst. apply E_And_Short. assumption.
  - inversion Heval; subst. apply E_Or_Short. assumption.
  - destruct (eval_fn n r1 e2) as [[r2 v2]|] eqn:Hright; [| discriminate].
    destruct v2; try discriminate.
    inversion Heval; subst. apply IH in Hright. eapply E_Or_False; eassumption.
Qed.

(** I cover every binary operator, including arithmetic failure cases. *)
Theorem eval_fn_sound_binop : forall fuel renv op e1 e2 renv' v,
  eval_fn fuel renv (EBinOp op e1 e2) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (EBinOp op e1 e2) renv' v.
Proof.
  intros fuel renv op e1 e2 renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  destruct op;
    try solve [eapply eval_fn_sound_logic; [reflexivity | exact Heval | exact IH]].
  all: simpl in Heval; simpl in IH;
    destruct (eval_fn n renv e1) as [[r1 v1]|] eqn:Hleft; try discriminate;
    destruct (eval_fn n r1 e2) as [[r2 v2]|] eqn:Hright; try discriminate;
    destruct v1, v2; try discriminate;
    apply IH in Hleft; apply IH in Hright.
  all: repeat match goal with
    | H : context [if ?b then _ else _] |- _ =>
      destruct b eqn:?; try discriminate
    end.
  all: inversion Heval; subst;
    try solve [first [eapply E_BinArith; [eassumption | eassumption | unfold eval_arith_binop; congruence]
          |eapply E_BinCmp; [eassumption | eassumption | reflexivity]
          |eapply E_BinEqBool; [eassumption | eassumption | reflexivity]
          |eapply E_BinNeBool; [eassumption | eassumption | reflexivity]
          |eapply E_BinEqStr; [eassumption | eassumption | reflexivity]
          |eapply E_BinNeStr; [eassumption | eassumption | reflexivity]
          |eapply E_StrCat; eassumption]].
  all: eapply E_BinArith; [eassumption | eassumption |];
    unfold eval_arith_binop; rewrite Heqb; reflexivity.
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
  destruct (eval_fn n renv cond) as [[renv1 vc]|] eqn:Hc; [| discriminate].
  destruct vc; try discriminate.
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
  destruct (eval_fn n renv e1) as [[renv1 v1]|] eqn:He1; [| discriminate].
  destruct v1; try discriminate.
  destruct (eval_fn n renv1 e2) as [[renv2 v2]|] eqn:He2; [| discriminate].
  destruct v2; try discriminate.
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

(** I keep binding names and their order while changing stored values. This
    invariant lets a let-binding pop its own slot after evaluating its body. *)
Fixpoint env_names (r : env) : list string :=
  match r with
  | ENil => []
  | ECons x _ rest => x :: env_names rest
  end.

Lemma env_update_names : forall r x v,
  env_names (env_update x v r) = env_names r.
Proof.
  induction r; intros; simpl; [reflexivity |].
  destruct (String.eqb x s); simpl; f_equal; auto.
Qed.

Lemma eval_preserves_env_names : forall r e r' v,
  eval r e r' v -> env_names r' = env_names r.
Proof.
  intros r e r' v Heval. induction Heval; simpl in *;
    try rewrite env_update_names; congruence.
Qed.

Theorem eval_fn_sound_let : forall fuel renv x e1 e2 renv' v,
  eval_fn fuel renv (ELet x e1 e2) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (ELet x e1 e2) renv' v.
Proof.
  intros fuel renv x e1 e2 renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval, IH.
  destruct (eval_fn n renv e1) as [[r1 v1]|] eqn:Hfirst; [| discriminate].
  destruct (eval_fn n (ECons x v1 r1) e2) as [[r2 v2]|] eqn:Hbody; [| discriminate].
  destruct r2 as [|y vy rest]; [discriminate |].
  apply IH in Hfirst. apply IH in Hbody.
  pose proof (eval_preserves_env_names _ _ _ _ Hbody) as Hnames.
  simpl in Hnames. injection Hnames as Hname Htail. subst y.
  inversion Heval; subst. eapply E_Let; eassumption.
Qed.

(** I thread the environment through every executed loop iteration. *)
Theorem eval_fn_sound_while : forall fuel renv cond body renv' v,
  eval_fn fuel renv (EWhile cond body) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (EWhile cond body) renv' v.
Proof.
  intros fuel renv cond body renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval, IH.
  destruct (eval_fn n renv cond) as [[r1 vc]|] eqn:Hcond; [| discriminate].
  destruct vc; try discriminate. apply IH in Hcond. destruct b.
  - destruct (eval_fn n r1 body) as [[r2 vb]|] eqn:Hbody; [| discriminate].
    apply IH in Hbody. apply IH in Heval. eapply E_WhileTrue; eassumption.
  - inversion Heval; subst. apply E_WhileFalse. assumption.
Qed.

(** I evaluate closure bodies lexically and return the caller's environment. *)
Theorem eval_fn_sound_app : forall fuel renv e1 e2 renv' v,
  eval_fn fuel renv (EApp e1 e2) = Some (renv', v) ->
  (forall r e r' v0, eval_fn (pred fuel) r e = Some (r', v0) -> eval r e r' v0) ->
  eval renv (EApp e1 e2) renv' v.
Proof.
  intros fuel renv e1 e2 renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval, IH.
  destruct (eval_fn n renv e1) as [[r1 vf]|] eqn:Hfn; [| discriminate].
  destruct vf; try discriminate.
  all: repeat match type of Heval with
    | context [eval_fn ?fuel0 ?r ?e] =>
      let H := fresh "Hcall" in
      destruct (eval_fn fuel0 r e) as [[? ?]|] eqn:H; try discriminate
    end.
  all: repeat match goal with
    | H : eval_fn _ _ _ = Some (_, _) |- _ => apply IH in H
    end.
  all: inversion Heval; subst; eauto using E_App, E_AppFix.
Qed.

Theorem eval_fn_sound_match : forall fuel renv e branches renv' v,
  eval_fn fuel renv (EMatch e branches) = Some (renv', v) ->
  (forall r e0 r' v0, eval_fn (pred fuel) r e0 = Some (r', v0) -> eval r e0 r' v0) ->
  eval renv (EMatch e branches) renv' v.
Proof.
  intros fuel renv e branches renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval, IH.
  destruct (eval_fn n renv e) as [[r1 vc]|] eqn:Hscrut; [| discriminate].
  destruct vc as [| | | | | | | |tag payload|]; try discriminate.
  destruct (find_branch tag branches) as [[x body]|] eqn:Hbranch; [| discriminate].
  destruct (eval_fn n (ECons x payload r1) body) as [[r2 result]|] eqn:Hbody;
    [| discriminate].
  destruct r2 as [|y vy rest]; [discriminate |].
  apply IH in Hscrut. apply IH in Hbody.
  pose proof (eval_preserves_env_names _ _ _ _ Hbody) as Hnames.
  simpl in Hnames. injection Hnames as Hname Htail. subst y.
  inversion Heval; subst. eapply E_Match; eassumption.
Qed.

(** I split only the computations inspected by the evaluator, then use the
    corresponding relational rules for aggregate reads and updates. *)
Ltac solve_eval_fn_access IH Heval :=
  repeat match type of Heval with
  | context [match ?scrut with _ => _ end] =>
    let H := fresh "Hcase" in destruct scrut eqn:H; try discriminate
  end;
  repeat match goal with
  | H : eval_fn _ _ _ = Some (_, _) |- _ => apply IH in H
  end;
  inversion Heval; subst;
  eauto using E_Index, E_ArraySet, E_ArrayPush, E_Field, E_SetField, E_TupleIndex.

Theorem eval_fn_sound_index : forall fuel renv e1 e2 renv' v,
  eval_fn fuel renv (EIndex e1 e2) = Some (renv', v) ->
  (forall r e0 r' v0, eval_fn (pred fuel) r e0 = Some (r', v0) -> eval r e0 r' v0) ->
  eval renv (EIndex e1 e2) renv' v.
Proof.
  intros fuel renv e1 e2 renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval, IH.
  solve_eval_fn_access IH Heval.
Qed.

Theorem eval_fn_sound_arrayset : forall fuel renv e1 e2 e3 renv' v,
  eval_fn fuel renv (EArraySet e1 e2 e3) = Some (renv', v) ->
  (forall r e0 r' v0, eval_fn (pred fuel) r e0 = Some (r', v0) -> eval r e0 r' v0) ->
  eval renv (EArraySet e1 e2 e3) renv' v.
Proof.
  intros fuel renv e1 e2 e3 renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval, IH.
  solve_eval_fn_access IH Heval.
Qed.

Theorem eval_fn_sound_arraypush : forall fuel renv e1 e2 renv' v,
  eval_fn fuel renv (EArrayPush e1 e2) = Some (renv', v) ->
  (forall r e0 r' v0, eval_fn (pred fuel) r e0 = Some (r', v0) -> eval r e0 r' v0) ->
  eval renv (EArrayPush e1 e2) renv' v.
Proof.
  intros fuel renv e1 e2 renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval, IH.
  solve_eval_fn_access IH Heval.
Qed.

Theorem eval_fn_sound_field : forall fuel renv e f renv' v,
  eval_fn fuel renv (EField e f) = Some (renv', v) ->
  (forall r e0 r' v0, eval_fn (pred fuel) r e0 = Some (r', v0) -> eval r e0 r' v0) ->
  eval renv (EField e f) renv' v.
Proof.
  intros fuel renv e f renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval, IH.
  solve_eval_fn_access IH Heval.
Qed.

Theorem eval_fn_sound_setfield : forall fuel renv x f e renv' v,
  eval_fn fuel renv (ESetField x f e) = Some (renv', v) ->
  (forall r e0 r' v0, eval_fn (pred fuel) r e0 = Some (r', v0) -> eval r e0 r' v0) ->
  eval renv (ESetField x f e) renv' v.
Proof.
  intros fuel renv x f e renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval, IH.
  solve_eval_fn_access IH Heval.
Qed.

Theorem eval_fn_sound_tupleindex : forall fuel renv e i renv' v,
  eval_fn fuel renv (ETupleIndex e i) = Some (renv', v) ->
  (forall r e0 r' v0, eval_fn (pred fuel) r e0 = Some (r', v0) -> eval r e0 r' v0) ->
  eval renv (ETupleIndex e i) renv' v.
Proof.
  intros fuel renv e i renv' v Heval IH.
  destruct fuel as [|n]; [simpl in Heval; discriminate |].
  simpl in Heval, IH.
  solve_eval_fn_access IH Heval.
Qed.

Theorem eval_fn_sound_array : forall es fuel renv renv' v,
  (forall m, (m < fuel)%nat -> forall r e r' v0,
    eval_fn m r e = Some (r', v0) -> eval r e r' v0) ->
  eval_fn fuel renv (EArray es) = Some (renv', v) ->
  eval renv (EArray es) renv' v.
Proof.
  induction es as [|e es IHes]; intros fuel renv renv' v IH Heval;
    destruct fuel as [|n]; try discriminate.
  - destruct n; simpl in Heval; inversion Heval; subst; constructor.
  - destruct n as [|m]; [discriminate |].
    cbn [eval_fn] in Heval.
    destruct (eval_fn m renv e) as [[r1 v1]|] eqn:Hhead; [| discriminate].
    match type of Heval with
    | context [?loop m r1 es] =>
      destruct (loop m r1 es) as [[r2 vs]|] eqn:Htail; [| discriminate]
    end.
    inversion Heval; subst.
    eapply E_ArrayCons.
    + eapply (IH m); [lia | exact Hhead].
    + eapply (IHes (S m)).
      * intros k Hk. apply IH. lia.
      * cbn [eval_fn]. rewrite Htail. reflexivity.
Qed.

Theorem eval_fn_sound_record : forall es fuel renv renv' v,
  (forall m, (m < fuel)%nat -> forall r e r' v0,
    eval_fn m r e = Some (r', v0) -> eval r e r' v0) ->
  eval_fn fuel renv (ERecord es) = Some (renv', v) ->
  eval renv (ERecord es) renv' v.
Proof.
  induction es as [|[f e] es IHes]; intros fuel renv renv' v IH Heval;
    destruct fuel as [|n]; try discriminate.
  - destruct n; simpl in Heval; inversion Heval; subst; constructor.
  - destruct n as [|m]; [discriminate |].
    cbn [eval_fn] in Heval.
    destruct (eval_fn m renv e) as [[r1 v1]|] eqn:Hhead; [| discriminate].
    match type of Heval with
    | context [?loop m r1 es] =>
      destruct (loop m r1 es) as [[r2 vs]|] eqn:Htail; [| discriminate]
    end.
    inversion Heval; subst.
    eapply E_RecordCons.
    + eapply (IH m); [lia | exact Hhead].
    + eapply (IHes (S m)).
      * intros k Hk. apply IH. lia.
      * cbn [eval_fn]. rewrite Htail. reflexivity.
Qed.

Theorem eval_fn_sound_tuple : forall es fuel renv renv' v,
  (forall m, (m < fuel)%nat -> forall r e r' v0,
    eval_fn m r e = Some (r', v0) -> eval r e r' v0) ->
  eval_fn fuel renv (ETuple es) = Some (renv', v) ->
  eval renv (ETuple es) renv' v.
Proof.
  induction es as [|e es IHes]; intros fuel renv renv' v IH Heval;
    destruct fuel as [|n]; try discriminate.
  - simpl in Heval. inversion Heval; subst. constructor.
  - cbn [eval_fn] in Heval.
    destruct (eval_fn n renv e) as [[r1 v1]|] eqn:Hhead; [| discriminate].
    match type of Heval with
    | context [?loop n r1 es] =>
      destruct (loop n r1 es) as [[r2 vs]|] eqn:Htail; [| discriminate]
    end.
    inversion Heval; subst.
    eapply E_TupleCons.
    + eapply (IH n); [lia | exact Hhead].
    + eapply (IHes (S n)); [exact IH |].
      cbn [eval_fn]. rewrite Htail. reflexivity.
Qed.

(** I connect every expression case by strong induction on fuel. No recursive
    soundness premise remains in this theorem. *)
Theorem eval_fn_sound : forall fuel renv e renv' v,
  eval_fn fuel renv e = Some (renv', v) ->
  eval renv e renv' v.
Proof.
  intro fuel. induction fuel using lt_wf_ind.
  intros renv e renv' v Heval.
  destruct fuel as [|n]; [discriminate |].
  assert (IH : forall r e0 r' v0,
    eval_fn n r e0 = Some (r', v0) -> eval r e0 r' v0).
  { apply H. lia. }
  destruct e;
    try exact (eval_fn_sound_simple _ _ _ _ _ Heval).
  all: first [eapply eval_fn_sound_binop; eassumption
          |eapply eval_fn_sound_unop; eassumption
          |eapply eval_fn_sound_if; eassumption
          |eapply eval_fn_sound_let; eassumption
          |eapply eval_fn_sound_set; eassumption
          |eapply eval_fn_sound_seq; eassumption
          |eapply eval_fn_sound_while; eassumption
          |eapply eval_fn_sound_lam; eassumption
          |eapply eval_fn_sound_app; eassumption
          |eapply eval_fn_sound_fix; eassumption
          |eapply eval_fn_sound_index; eassumption
          |eapply eval_fn_sound_arrayset; eassumption
          |eapply eval_fn_sound_arraypush; eassumption
          |eapply eval_fn_sound_field; eassumption
          |eapply eval_fn_sound_setfield; eassumption
          |eapply eval_fn_sound_construct; eassumption
          |eapply eval_fn_sound_match; eassumption
          |eapply eval_fn_sound_strindex; eassumption
          |eapply eval_fn_sound_tupleindex; eassumption
          |eapply eval_fn_sound_array; [exact H | exact Heval]
          |eapply eval_fn_sound_record; [exact H | exact Heval]
          |eapply eval_fn_sound_tuple; [exact H | exact Heval]].
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
