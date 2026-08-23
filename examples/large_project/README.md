# Large Project

I use this package to show a real multi-file layout.

```text
examples/large_project/
  nano.toml
  main.nano
  src/models.nano
  src/pricing.nano
```

```bash
./bin/nanoc_c examples/large_project/main.nano -o /tmp/large_project
/tmp/large_project
```

The command runs from the repository root because the imports are repository-relative.
