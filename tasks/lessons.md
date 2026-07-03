
## 2026-07-03 — bash special variables in pod scripts
- `GROUPS` (also `UID`, `EUID`, `SECONDS`, `RANDOM`, `LINENO`) are special bash
  variables; assignments to `GROUPS` are silently ignored, so `for g in $GROUPS`
  iterated over group id `0`. Prefix custom script vars (`MM_*`) always.
- Symptom seen live: phase `download-0`, then skip-guards missed (they grep for
  the real names) and smokes ran without models. Caught by phase telemetry —
  keep phase names derived from data, they double as assertions.
