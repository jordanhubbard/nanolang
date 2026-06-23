(* NanoCore Reference Interpreter
 *
 * Reads NanoCore expressions in S-expression format from stdin,
 * evaluates them using the Coq-extracted evaluator, and prints results.
 *
 * This is mathematically verified: the eval_fn function was extracted
 * from Coq and proven sound with respect to the relational semantics.
 *
 * S-expression format:
 *   (EInt 42)
 *   (EBool true)
 *   (EString "hello")
 *   (EBinOp OpAdd (EInt 1) (EInt 2))
 *   (ELet "x" (EInt 42) (EBinOp OpAdd (EVar "x") (EInt 1)))
 *)

(* Alias Stdlib modules that get shadowed by Coq-extracted modules *)
module StdString = Stdlib.String
module StdList = Stdlib.List

(* ── String/char list conversion ───────────────────────────────────────── *)

let string_to_char_list (s : string) : char list =
  let rec aux i acc =
    if i < 0 then acc
    else aux (i - 1) (StdString.get s i :: acc)
  in
  aux (StdString.length s - 1) []

let char_list_to_string (cl : char list) : string =
  let buf = Buffer.create 16 in
  StdList.iter (Buffer.add_char buf) cl;
  Buffer.contents buf

(* ── S-expression tokenizer ────────────────────────────────────────────── *)

type token =
  | TLParen
  | TRParen
  | TSymbol of string
  | TString of string
  | TInt of int
  | TEof

let tokenize (input : string) : token list =
  let len = StdString.length input in
  let pos = ref 0 in
  let tokens = ref [] in
  let skip_whitespace () =
    while !pos < len && (StdString.get input !pos = ' ' ||
                         StdString.get input !pos = '\n' ||
                         StdString.get input !pos = '\r' ||
                         StdString.get input !pos = '\t') do
      incr pos
    done
  in
  let read_string () =
    incr pos; (* skip opening quote *)
    let buf = Buffer.create 32 in
    while !pos < len && StdString.get input !pos <> '"' do
      if StdString.get input !pos = '\\' && !pos + 1 < len then begin
        incr pos;
        (match StdString.get input !pos with
         | 'n' -> Buffer.add_char buf '\n'
         | 't' -> Buffer.add_char buf '\t'
         | '\\' -> Buffer.add_char buf '\\'
         | '"' -> Buffer.add_char buf '"'
         | c -> Buffer.add_char buf '\\'; Buffer.add_char buf c);
        incr pos
      end else begin
        Buffer.add_char buf (StdString.get input !pos);
        incr pos
      end
    done;
    if !pos < len then incr pos; (* skip closing quote *)
    Buffer.contents buf
  in
  let read_symbol_or_int () =
    let start = !pos in
    while !pos < len && StdString.get input !pos <> ' ' &&
          StdString.get input !pos <> '\n' &&
          StdString.get input !pos <> '\r' &&
          StdString.get input !pos <> '\t' &&
          StdString.get input !pos <> '(' &&
          StdString.get input !pos <> ')' do
      incr pos
    done;
    let s = StdString.sub input start (!pos - start) in
    (* Try to parse as integer *)
    match int_of_string_opt s with
    | Some n -> TInt n
    | None -> TSymbol s
  in
  while !pos < len do
    skip_whitespace ();
    if !pos < len then
      match StdString.get input !pos with
      | '(' -> tokens := TLParen :: !tokens; incr pos
      | ')' -> tokens := TRParen :: !tokens; incr pos
      | '"' -> tokens := TString (read_string ()) :: !tokens
      | _ -> tokens := read_symbol_or_int () :: !tokens
  done;
  StdList.rev !tokens

(* ── S-expression parser ──────────────────────────────────────────────── *)

type sexp =
  | SAtom of string
  | SString of string
  | SInt of int
  | SList of sexp list

let parse_sexps (tokens : token list) : sexp list =
  let toks = ref tokens in
  let peek () = match !toks with t :: _ -> t | [] -> TEof in
  let advance () = match !toks with _ :: rest -> toks := rest | [] -> () in
  let rec parse_one () =
    match peek () with
    | TLParen ->
      advance ();
      let items = ref [] in
      while peek () <> TRParen && peek () <> TEof do
        items := parse_one () :: !items
      done;
      (match peek () with TRParen -> advance () | _ -> ());
      SList (StdList.rev !items)
    | TSymbol s -> advance (); SAtom s
    | TString s -> advance (); SString s
    | TInt n -> advance (); SInt n
    | TRParen -> advance (); SAtom ")" (* shouldn't happen *)
    | TEof -> SAtom "eof"
  in
  let results = ref [] in
  while peek () <> TEof do
    results := parse_one () :: !results
  done;
  StdList.rev !results

(* ── Convert S-expressions to NanoCore AST ────────────────────────────── *)

let z_of_int (n : int) : Big_int_Z.big_int =
  Big_int_Z.big_int_of_int n

let parse_binop = function
  | "OpAdd" -> Syntax.OpAdd | "OpSub" -> Syntax.OpSub
  | "OpMul" -> Syntax.OpMul | "OpDiv" -> Syntax.OpDiv
  | "OpMod" -> Syntax.OpMod | "OpEq"  -> Syntax.OpEq
  | "OpNe"  -> Syntax.OpNe  | "OpLt"  -> Syntax.OpLt
  | "OpLe"  -> Syntax.OpLe  | "OpGt"  -> Syntax.OpGt
  | "OpGe"  -> Syntax.OpGe  | "OpAnd" -> Syntax.OpAnd
  | "OpOr"  -> Syntax.OpOr  | "OpStrCat" -> Syntax.OpStrCat
  | s -> failwith ("Unknown binop: " ^ s)

let parse_unop = function
  | "OpNeg" -> Syntax.OpNeg | "OpNot" -> Syntax.OpNot
  | "OpStrLen" -> Syntax.OpStrLen | "OpArrayLen" -> Syntax.OpArrayLen
  | s -> failwith ("Unknown unop: " ^ s)

let rec parse_ty (s : sexp) : Syntax.ty =
  match s with
  | SAtom "TInt" -> Syntax.TInt
  | SAtom "TBool" -> Syntax.TBool
  | SAtom "TString" -> Syntax.TString
  | SAtom "TUnit" -> Syntax.TUnit
  | SList [SAtom "TArrow"; t1; t2] -> Syntax.TArrow (parse_ty t1, parse_ty t2)
  | SList [SAtom "TArray"; t] -> Syntax.TArray (parse_ty t)
  | SList (SAtom "TRecord" :: fields) ->
    Syntax.TRecord (StdList.map parse_ty_field fields)
  | SList (SAtom "TVariant" :: variants) ->
    Syntax.TVariant (StdList.map parse_ty_field variants)
  | _ -> failwith "Unknown type"

and parse_ty_field = function
  | SList [SString name; ty] -> (string_to_char_list name, parse_ty ty)
  | _ -> failwith "Bad type field"

let rec parse_expr (s : sexp) : Syntax.expr =
  match s with
  | SList [SAtom "EInt"; SInt n] -> Syntax.EInt (z_of_int n)
  | SList [SAtom "EBool"; SAtom "true"] -> Syntax.EBool true
  | SList [SAtom "EBool"; SAtom "false"] -> Syntax.EBool false
  | SList [SAtom "EString"; SString s] -> Syntax.EString (string_to_char_list s)
  | SList [SAtom "EUnit"] -> Syntax.EUnit
  | SList [SAtom "EVar"; SString x] -> Syntax.EVar (string_to_char_list x)
  | SList [SAtom "EBinOp"; SAtom op; e1; e2] ->
    Syntax.EBinOp (parse_binop op, parse_expr e1, parse_expr e2)
  | SList [SAtom "EUnOp"; SAtom op; e0] ->
    Syntax.EUnOp (parse_unop op, parse_expr e0)
  | SList [SAtom "EIf"; cond; e_then; e_else] ->
    Syntax.EIf (parse_expr cond, parse_expr e_then, parse_expr e_else)
  | SList [SAtom "ELet"; SString x; e1; e2] ->
    Syntax.ELet (string_to_char_list x, parse_expr e1, parse_expr e2)
  | SList [SAtom "ESet"; SString x; e0] ->
    Syntax.ESet (string_to_char_list x, parse_expr e0)
  | SList [SAtom "ESeq"; e1; e2] ->
    Syntax.ESeq (parse_expr e1, parse_expr e2)
  | SList [SAtom "EWhile"; cond; body] ->
    Syntax.EWhile (parse_expr cond, parse_expr body)
  | SList [SAtom "ELam"; SString x; ty; body] ->
    Syntax.ELam (string_to_char_list x, parse_ty ty, parse_expr body)
  | SList [SAtom "EApp"; e1; e2] ->
    Syntax.EApp (parse_expr e1, parse_expr e2)
  | SList [SAtom "EFix"; SString f; SString x; t1; t2; body] ->
    Syntax.EFix (string_to_char_list f, string_to_char_list x,
                 parse_ty t1, parse_ty t2, parse_expr body)
  | SList (SAtom "EArray" :: es) ->
    Syntax.EArray (StdList.map parse_expr es)
  | SList [SAtom "EIndex"; e1; e2] ->
    Syntax.EIndex (parse_expr e1, parse_expr e2)
  | SList [SAtom "EArraySet"; e1; e2; e3] ->
    Syntax.EArraySet (parse_expr e1, parse_expr e2, parse_expr e3)
  | SList [SAtom "EArrayPush"; e1; e2] ->
    Syntax.EArrayPush (parse_expr e1, parse_expr e2)
  | SList (SAtom "ERecord" :: fields) ->
    Syntax.ERecord (StdList.map parse_record_field fields)
  | SList [SAtom "EField"; e0; SString f] ->
    Syntax.EField (parse_expr e0, string_to_char_list f)
  | SList [SAtom "ESetField"; SString x; SString f; e0] ->
    Syntax.ESetField (string_to_char_list x, string_to_char_list f, parse_expr e0)
  | SList [SAtom "EConstruct"; SString tag; e0; ty] ->
    Syntax.EConstruct (string_to_char_list tag, parse_expr e0, parse_ty ty)
  | SList (SAtom "EMatch" :: scrutinee :: branches) ->
    Syntax.EMatch (parse_expr scrutinee, StdList.map parse_branch branches)
  | SList [SAtom "EStrIndex"; e1; e2] ->
    Syntax.EStrIndex (parse_expr e1, parse_expr e2)
  | _ -> failwith "Unknown expression form"

and parse_record_field = function
  | SList [SString name; e] -> (string_to_char_list name, parse_expr e)
  | _ -> failwith "Bad record field"

and parse_branch = function
  | SList [SString tag; SString x; body] ->
    ((string_to_char_list tag, string_to_char_list x), parse_expr body)
  | _ -> failwith "Bad match branch"

(* ── Value printer ────────────────────────────────────────────────────── *)

let rec print_val (v : Syntax.coq_val) : string =
  match v with
  | Syntax.VInt z ->
    Printf.sprintf "(VInt %s)" (Big_int_Z.string_of_big_int z)
  | Syntax.VBool true -> "(VBool true)"
  | Syntax.VBool false -> "(VBool false)"
  | Syntax.VString cl ->
    Printf.sprintf "(VString %S)" (char_list_to_string cl)
  | Syntax.VUnit -> "(VUnit)"
  | Syntax.VClos _ -> "(VClos <closure>)"
  | Syntax.VFixClos _ -> "(VFixClos <fix-closure>)"
  | Syntax.VArray vs ->
    Printf.sprintf "(VArray %s)" (StdString.concat " " (StdList.map print_val vs))
  | Syntax.VRecord fvs ->
    let fields = StdList.map (fun (name, v) ->
      Printf.sprintf "(%S %s)" (char_list_to_string name) (print_val v)
    ) fvs in
    Printf.sprintf "(VRecord %s)" (StdString.concat " " fields)
  | Syntax.VConstruct (tag, v) ->
    Printf.sprintf "(VConstruct %S %s)" (char_list_to_string tag) (print_val v)

(* ── Main ─────────────────────────────────────────────────────────────── *)

let () =
  (* Read all of stdin *)
  let buf = Buffer.create 4096 in
  (try while true do Buffer.add_char buf (input_char stdin) done
   with End_of_file -> ());
  let input = Buffer.contents buf in

  if StdString.length input = 0 then begin
    Printf.eprintf "nanocore-ref: no input\n";
    exit 1
  end;

  let tokens = tokenize input in
  let sexps = parse_sexps tokens in

  StdList.iter (fun sexp ->
    let expr = parse_expr sexp in
    match EvalFn.eval_program expr with
    | Some (_env, v) ->
      Printf.printf "%s\n" (print_val v)
    | None ->
      Printf.printf "(Error: evaluation failed -- out of fuel or stuck)\n"
  ) sexps
