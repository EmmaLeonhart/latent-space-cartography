# queue — current session's concrete work

(Convention: CLAUDE.md "Queue and longer-horizon work". Items are deleted in the
same commit that completes them; long-horizon material lives in `todo.md`.)

- Step 5 (BLOCKED-ON-USER-ACTION — Emma's explicit go-ahead): update every
  number in paper.md from the regenerated frozen-snapshot run (embedding
  counts, discovered/universal op counts, r values, collision totals, geometry
  stats, Tables 2–9; see 2026-07-03 devlog entries for the numbers). At the
  same time compute per-entity collision participation + collapse-geometry
  stats and port `old/scripts/measure_collapse_geometry.py` into `scripts/`.

## Pinned tail (always last two items)

- Ensure the three autonomous-loop crons are running (work-loop :03, auto-flush
  :15, status-report :42) — restart if a planning burst killed them.
- Run the status-report action once more, independently: end-of-session summary
  of everything that happened this session.
