# My effect binding helper

I register module-owned effect names separately from function and type names. I retain handler visibility per module, import only public effects, and refuse an explicit import of a private effect. My standalone driver build passes 372 shadows, including registration and visibility controls.

I have not yet connected this helper to the compiler binding pass or rewritten perform names. This evidence proves helper behavior only. My private-import and same-name collision reproducers remain open until both frontends and runtime paths pass.
