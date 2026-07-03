queue.md 

This is the queue for this repository. It is probably going to be a bit of a mess because I am manually creating it, as opposed to using an agent to create it. The basic way, but basically what's going on here is the paper is a bit out of date, and now that Claw Forest is over, I'm not really concerned about maintaining my strong except it my goal is to instead try to work on it. My goal for this paper is to try to work more on it. I would like to try to get the CI/CD back up and running for the paper and start to make some changes in it to change it from being a Claw4S winter to being something that could go on arXiv. 

Oh, and also please make it so that the todo.md should be updated, if it is present at all, to do this clarification stuff. Generally speaking, this entire directory should have its productivity stuff brought up to the current day standards of the CleanVibe repo of the CleanVibe library. 

## Data regeneration on a frozen snapshot (definitive fix for report §2a/§2b)

- Step 5 (BLOCKED on Emma's decision): update every number in paper.md from the new run (embedding counts, discovered/universal op counts, r, collision totals, geometry stats, Tables 2-9). The current published numbers came from the old crawl; regenerated numbers WILL differ because Wikidata grew. Do not start without explicit go-ahead.

## Pinned tail (always last two items)

- Ensure the three autonomous-loop crons are running (work-loop :03, auto-flush :15, status-report :42) — restart if a planning burst killed them.
- Run the status-report action once more, independently: end-of-session summary of everything that happened this session.