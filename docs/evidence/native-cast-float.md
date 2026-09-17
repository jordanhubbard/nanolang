# Native scalar float conversion

I now lower `CAST_FLOAT` to the same scalar conversions as NanoVM: an integer
converts to double, a boolean becomes zero or one, a float retains its bits,
and a string uses `strtod`. Tagged inputs retain their actual kind. Other
supported representations produce positive zero, matching the VM default.
This does not add float-to-int conversion or change its existing refusal.

`tests/test_native_floats.py` passed all six methods. The new cases check
rounding around 2^53, signed 64-bit endpoints, booleans, partial and invalid
numeric strings, signed zero, infinities and NaNs, direct and tagged values,
tagged calls/tail calls, and a local loop. New generated native products run
with ASan and UBSan. A focused followup also checks concrete maps and missing
map values convert to zero.

`make test-nvm2c` passed 2,412 native checks, 1,092 shape checks, opcode coverage
and the sanitizer-driver checks. I removed `CAST_FLOAT` from the unsupported
opcode fixture because these positive conversion tests now define its support.
I retained every other unsupported-opcode refusal.

Logs: `/tmp/nanolang-native-cast-float-final.log`,
`/tmp/nanolang-native-cast-float-defaults.log`, and
`/tmp/nanolang-native-cast-float-aot-final.log`.

This is task `task_3d6c3314d8924dd2b0aca679647e7048`. Passive frontend publication
still requires the existing callable source fixtures to pass, including the
unchanged arctangent loop. It is not completed by this opcode alone.
