# V12-HQ Data Audit

Generated from V12 with strict localization quality control.

Summary:

```text
source all: 7362
HQ all:     7198
HQ train:   5741
HQ test:    1457
dropped:    164 sparse rows
```

Policy:

- preserve V12 run-level labels
- preserve V12 split where rows remain
- remove only very sparse rows
- keep high/medium evidence-backed localization
- downgrade unsupported component localization to `unknown`

Quality checks from local generation:

```text
leak_hits = {}
invalid_trace_refs = 0
low-confidence component_refs = 0
```

The generated `stats.json` is the source of truth.
