# queue — current session's concrete work

(Convention: CLAUDE.md "Queue and longer-horizon work". Items are deleted in the
same commit that completes them; long-horizon material lives in `todo.md`.)

## Pinned tail (always last two items)

- Ensure the three autonomous-loop crons are running (work-loop :03, auto-flush
  :15, status-report :42) — restart if a planning burst killed them.
- Run the status-report action once more, independently: end-of-session summary
  of everything that happened this session.
