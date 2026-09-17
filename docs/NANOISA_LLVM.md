# My first LLVM translator boundary

I translate a verified v2 module directly to LLVM IR. My initial contract covers integer/bool/void scalar storage, integer/bool results, locals, direct calls, branches and assertions. I retain value tags through calls and joins and check typed operations at runtime when verification cannot establish their inputs. My integer arithmetic wraps at 64 bits; division by zero returns zero, and INT64_MIN divided by -1 returns INT64_MIN. I use explicit control before division to avoid LLVM poison.

I reject imports, linked modules, heap values, nominal layouts, ownership/passive metadata and unsupported instructions. I do not embed NanoVM or resurrect AST target paths. I publish named output only after verification, profile checks and complete emission succeed, using an exclusive temporary beside the destination. Existing output and source survive failures.

My first same-module gate runs VM, C AOT and LLVM on calls, branch effects, loops, arithmetic boundaries and boolean tags. A passing scalar gate does not complete my required full-language LLVM or Wasm release scope. Wasm remains unimplemented.
