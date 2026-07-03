queue.md 

This is the queue for this repository. It is probably going to be a bit of a mess because I am manually creating it, as opposed to using an agent to create it. The basic way, but basically what's going on here is the paper is a bit out of date, and now that Claw Forest is over, I'm not really concerned about maintaining my strong except it my goal is to instead try to work on it. My goal for this paper is to try to work more on it. I would like to try to get the CI/CD back up and running for the paper and start to make some changes in it to change it from being a Claw4S winter to being something that could go on arXiv. 

Oh, and also please make it so that the todo.md should be updated, if it is present at all, to do this clarification stuff. Generally speaking, this entire directory should have its productivity stuff brought up to the current day standards of the CleanVibe repo of the CleanVibe library. 

## Data regeneration on a frozen snapshot (definitive fix for report §2a/§2b)

- Step 1: regenerate base dataset — `random_walk.py Q1342448 --limit 500` with mxbai-embed-large into `data/` (running in background; Wikidata-API-bound, ~1-1.5 h).
- Step 2: add second seed — `import_wikidata.py --instances` P31 country-level sampling into the same store (match the paper's two-seed design).
- Step 3: write `scripts/reembed_frozen.py` — read the frozen `data/items.json`, embed the identical label/alias set with nomic-embed-text and all-minilm into per-model stores (`FOL_DATA_DIR`), so all three models see byte-identical input. This is what makes the §4.6 cross-model claim clean.
- Step 4: re-run `fol_discovery.py` per model + `compare_models.py`; re-run `analyze_collisions.py --threshold 0.95` and geometry on the combined store, all from the one snapshot.
- Step 5 (BLOCKED on Emma's decision): update every number in paper.md from the new run (embedding counts, discovered/universal op counts, r, collision totals, geometry stats, Tables 2-9). The current published numbers came from the old crawl; regenerated numbers WILL differ because Wikidata grew. Do not start without explicit go-ahead.