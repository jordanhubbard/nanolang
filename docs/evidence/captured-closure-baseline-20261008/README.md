# My captured-closure baseline

I retain the canonical returned anonymous-closure chain at source pin
`c7a43b0bc6ed3032ec25695f3d56971320736b3a`. My source fixture includes meaningful
shadows for both named functions and checks the final value in production.

- My C-seed bytecode producer passes the selected shadows and publishes the module.
- My NanoVM executes the resulting module, prints `42`, and exits zero.
- My prepared self-hosted producer rejects the inner captured `n` with `E0011`.
- My native translator refuses `CLOSURE_NEW` in the same seed-produced module.

I retain exact commands, statuses, durations, source/module/log hashes and tool
hashes in `manifest.json`. The prepared producer is the one qualified in my
[58-method named-container run](../selfhost-function-containers-20261008/README.md).
These are measured release blockers, not expected-success regressions or an
accepted reduced language profile. Task `task_f9059876cb0b463a8d51244f759e55c9`
tracks the complete repair and is held against automatic worker dispatch.

My parser currently hoists anonymous declarations and returns an identifier.
My native callable implementation handles module-local function IDs, without
owned capture environments. I must preserve lexical identities before enabling
capture lookup; looking up every outer name globally would accept invalid scope
and conflate distinct closure instances. I retain this fixture as the first
end-to-end acceptance case, not as the complete capture contract.
