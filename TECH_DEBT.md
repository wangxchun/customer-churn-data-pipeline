# Technical-debt register

Known correctness issues that are deliberately **not fixed yet**. Each entry stays here until it is fixed *and* a regression test reproduces the original failure scenario.

Planned cleanup: after the MLA-C02 Domain 1–4 study block, as a separate bug-fix change (not mixed with feature work).

| ID | Status | Severity | Blocks Phase 2–4? |
|---|---|---|---|
| TD-1 | Open (re-confirmed 2026-11-05) | Medium | No |
| TD-2 | Open (re-confirmed 2026-11-05) | Low | No |

---

## TD-1: `storage.replace_table()` is not atomic

- **Affected code:** `src/churn_pipeline/storage.py`, `replace_table()`. Called by `pipeline.run()` for the `customers` and `customer_profiles` SQLite tables.
- **Claim vs behavior:** the docstring says "Done in one transaction: readers never see a half-loaded table." That is false. Python's `sqlite3` in its default legacy transaction mode (`isolation_level=''`) only issues an implicit `BEGIN` before DML (`INSERT`/`UPDATE`/`DELETE`). The `DROP TABLE` and `CREATE TABLE` statements run outside a transaction and commit immediately. `with conn:` can only roll back the `INSERT`.
- **Failure scenario (reproduced 2026-11-05, SQLite 3.53.4):** load 2 rows, then call `replace_table()` with a frame containing a duplicate primary key. Result: `IntegrityError: UNIQUE constraint failed: customers.customer_id`, and the table is left **empty** (0 rows). The old 2 rows are gone.
- **Why it does not happen in normal runs:** both callers pass frames that are already deduplicated on the primary key (`validation.deduplicate()`), so the `INSERT` does not fail in practice. Any other failure inside the `INSERT` (for example a disk-full error) would still trigger it.
- **Severity: Medium.** The documented guarantee is wrong and the failure loses data, but it is only reachable via an unexpected error, and every lost table is rebuildable from `data/raw/`.
- **Impact on Phase 2–4:** none on correctness. Phase 2 reads `curated/customer_master.parquet`, not the SQLite `customers` table. If `replace_table()` fails, the exception aborts `pipeline.run()` *before* any Parquet file is written, so the previous successful Parquet output stays intact. A failed run is visible (non-zero exit) and fixed by re-running.
- **Recommended fix:** make the whole refresh one transaction: issue an explicit `BEGIN` before `DROP`, or use `sqlite3.connect(..., autocommit=False)` (Python ≥ 3.12; the project currently claims ≥ 3.11), or load into a staging table and swap it in with `ALTER TABLE ... RENAME` inside one transaction. Then make the docstring accurate.
- **Required regression tests:**
  1. Load valid rows; call `replace_table()` with a frame that fails during `INSERT` (duplicate primary key); assert `IntegrityError` **and** that the original rows are still present and unchanged.
  2. A successful refresh still fully replaces the contents and the schema (existing behavior; guard against regressions from the fix).
  3. Run the full test suite: Phase 1 outputs must be unchanged.

## TD-2: `events.event_id` primary key accepts `NULL`

- **Affected code:** `src/churn_pipeline/storage.py`, `SCHEMA`: `event_id TEXT PRIMARY KEY` in the `events` table.
- **Claim vs behavior:** the database is described as guaranteeing one row per event. Due to a long-standing SQLite quirk, a non-`INTEGER` `PRIMARY KEY` without `NOT NULL` allows `NULL`, and multiple `NULL`s don't conflict with each other.
- **Failure scenario (reproduced 2026-11-05):** two direct `INSERT`s with `event_id = NULL` both succeed. `ON CONFLICT(event_id) DO NOTHING` would not deduplicate them either.
- **Current protection:** the Python validation layer. `sources.EVENT_COLUMNS` declares `event_id` non-nullable with pattern `evt-[0-9a-z]+`, so `events.ingest_event_file()` never inserts a `NULL` id. The database itself does not enforce it.
- **Severity: Low.** It can't happen through the pipeline's own code path. It only matters if another writer bypasses validation or the validation rule is weakened.
- **Impact on Phase 2–4:** none. Event aggregates are excluded from Phase 2 features by default (synthetic, no churn signal, no cutoff date).
- **Recommended fix:** `event_id TEXT NOT NULL PRIMARY KEY`. Because `CREATE TABLE IF NOT EXISTS` doesn't alter an existing table, existing `churn.db` files keep the old schema. Either document "delete `data/processed/` to rebuild" (it is rebuildable) or add a small migration.
- **Required regression tests:**
  1. Inserting an event row with `event_id = NULL` directly into a fresh database raises `IntegrityError`.
  2. Normal ingestion, replay, and conflict tests in `tests/test_events.py` still pass.

---

## How to close an entry

1. Fix it in its own change, separate from feature work.
2. Add the regression tests listed above, and confirm that they **fail on the unfixed code** before the fix is applied.
3. Run the full suite and record the result.
4. Mark the entry *Fixed* with the date and commit. Don't delete it: the history is part of the learning record.
