# queue — current session's concrete work

(Convention: CLAUDE.md "Queue and longer-horizon work". Items are deleted in the
same commit that completes them; long-horizon material lives in `todo.md`.)

## Delete the two dead remote branches (BLOCKED-ON-USER-ACTION: classifier denies remote deletion)

Branch-resolution sweep done 2026-07-04 (see devlog): queue-processing-8xsm6o's
one unique devlog commit fast-forwarded into main; master is a pure ancestor.
Emma runs: `git push origin --delete claude/queue-processing-8xsm6o master`
Delete this item once run.

## Pinned tail (always last two items)

- Ensure the three autonomous-loop crons are running (work-loop :03, auto-flush
  :15, status-report :42) — restart if a planning burst killed them.
- Run the status-report action once more, independently: end-of-session summary
  of everything that happened this session.
