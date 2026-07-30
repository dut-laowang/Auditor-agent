# V18 data audit

The input projection uses a strict observable-field allowlist. Offline attack
metadata is used only for target construction, redaction checks, and evidence
sufficiency filtering; it is never copied into the user message or used by the
event selector.

Final checks:

- 200/200 QC rows valid
- exact and normalized prompt overlap: 0
- train/test task-group overlap: 0
- raw marker hits: 0
- benchmark-identity hits: 0
- forbidden input-key hits: 0
- invalid observable-event hits: 0
- redacted entity IDs reused across runs: 0
- shallow structural proxy accuracy: 47.33%

The lexical proxy reaches 71.99% because V18 intentionally retains legitimate
natural-language attack evidence. This is not treated as leakage; the strict
field and marker scans remain zero.
