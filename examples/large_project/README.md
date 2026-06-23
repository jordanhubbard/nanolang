# Large Project

I use this package to show a real multi-file layout.

```text
examples/large_project/
  nano.toml
  main.nano
  src/models.nano
  src/pricing.nano
```

`large_project_structure.nano` is a sketch. This directory is the executable package example.

```bash
./bin/nanoc examples/large_project/main.nano -o /tmp/large_project
/tmp/large_project
```
