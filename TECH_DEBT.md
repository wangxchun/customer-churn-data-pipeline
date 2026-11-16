# Technical-debt register

Known correctness issues that are deliberately **not fixed yet**. Each entry stays here until it is fixed *and* a regression test reproduces the original failure scenario.

Planned cleanup: after the MLA-C02 Domain 1–4 study block, as a separate bug-fix change (not mixed with feature work).

| ID | Status | Severity | Blocks Phase 2–4? |
|---|---|---|---|
| TD-1 | Open (re-confirmed 2026-11-05) | Medium | No |
| TD-2 | Open (re-confirmed 2026-11-05) | Low | No |
| TD-3 | Fixed 2026-11-16 (commit: pending, user commit); policy: refuse + move report (recorded 2026-11-06; Phase 3 mitigation 2026-11-08) | Medium | No. Phase 3 refuses cross-split comparisons; see note |
| TD-4 | Open (recorded 2026-11-06) | Low | No |
| TD-5 | Open (recorded 2026-11-06) | Medium | No for Phase 2–3. Must be addressed before any Phase 4 inference/serving |

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

## TD-3: Split changes between re-runs are not detected, and the previous split is overwritten

- **Affected code:** `src/churn_pipeline/features.py`, `run_features()` (writes `split_assignments.parquet`) and `split_customers()`.
- **Gap:** `split_customers()` is deterministic only for identical input rows plus an identical seed (and, presumably, the same scikit-learn version; not tested). It samples by position after sorting by `customer_id`. As a result, any change to the input set reassigns many customers. Nothing in the pipeline compares a new split to the previous one. `run_features()` also overwrites `split_assignments.parquet` in place, so the previous assignment list is lost unless someone copies it beforehand.
- **Failure scenario (measured in memory 2026-11-06 against the current `split_assignments.parquet`):**

  | Change | Customers compared | Customers that changed split |
  |---|---|---|
  | Same data, seed 42 | 7,043 | 0 |
  | Same data, `--seed 7` | 7,043 | 3,285 |
  | One customer removed (`0004-TLHLJ`), seed 42 | 7,042 | 1,613 |

  In the one-row-removed case, 179 former train customers moved into test and 227 former test customers moved out. A test score from before the re-run and one from after are then measured on different test sets. They are not comparable, and the run gives no warning.
- **Current protection:** none automatic. `feature_report.json` has `input_sha256`, which signals that the input bytes changed. It doesn't show whether the assignments changed, and it is a byte fingerprint, not a content fingerprint.
- **Severity: Medium.** The data is not corrupted. However, the gap silently invalidates model comparisons across re-runs. That matters once Phase 3 starts comparing models.
- **Impact on Phase 2–4:** no effect on Phase 2 correctness within a single run. Phase 3 must not compare test metrics across re-runs until this is addressed (or the split is frozen).
- **Possible fixes (as listed before the fix; the chosen policy is under "Fix" below):**
  - Before overwriting, compare against the existing `split_assignments.parquet` and record or report moved, added and removed customers.
  - Alternatively, keep previous assignment files.
  - Alternatively, support reusing a frozen assignment file for existing customers, with new customers assigned separately.
  - Hash-based assignment (split by a hash of `customer_id`) would make each customer's assignment independent of other rows, but it changes the stratification guarantee. That trade-off needs a decision first.
- **Required regression tests:**
  1. Re-running on identical input reports zero moved customers.
  2. Re-running after removing one customer reports the removed customer and the number of moved customers, instead of silently overwriting.
  3. Existing `test_split_is_reproducible_and_independent_of_row_order` still passes.
- **Phase 3 mitigation (2026-11-08, decision D7; this is not a fix):**
  - Every Phase 3 run manifest records `split_content_sha256`, a hash of the sorted `(customer_id, split)` pairs. Unlike the file checksum, this content hash doesn't depend on Parquet byte layout.
  - `train` refuses to run if `customer_master.parquet` no longer matches `feature_report.json` `input_sha256`, or if the assignment IDs don't match the master one-to-one.
  - `evaluate` refuses if the current split hash differs from the hash the run was trained on.
  - `compare` refuses two runs whose split hashes differ.
  - Tests: `test_training_refuses_a_master_that_phase2_did_not_build_from` and `test_compare_refuses_runs_made_on_different_splits`.
- **Remaining risk (before the 2026-11-16 fix; kept as history):**
  - `run_features()` still overwrites `split_assignments.parquet` silently, and the previous assignment list is still lost.
  - Phase 3 only *detects* the change afterwards. Old runs then become unusable for comparison; nothing preserves the old split or reports which customers moved.
  - A split change between `train` and `evaluate` is caught. A split change that happens before any run exists can't be detected.
- **Evidence required to close:** the three regression tests listed above, implemented against `run_features()` itself and failing on the current code, plus the chosen policy (move report, keep history, or frozen assignments).
- **Fix (2026-11-16; commit: pending, user commit). Policy: refuse + move report.**
  - If `split_assignments.parquet` exists and the new split's content hash differs from it, `run_features()` raises `SplitDriftError` (a `DataContractError`).
  - The refusal happens before any filesystem write, including `mkdir`. All 7 artifacts are left byte-for-byte unchanged.
  - An existing split file that is unreadable, is missing `customer_id` or `split`, or has null or duplicate IDs also raises, and also writes nothing.
  - `build-features` prints the report to stderr and exits with code 2.
  - The report contains no raw customer IDs; the decision was made 2026-11-16, D2-Q2. It holds old and new content hashes, old and new customer counts, and the counts added, removed, unchanged and moved. It also holds moves by direction, per-split sizes, change types, and the files that would have been created or overwritten.
  - The report is not written to any file. `feature_report.json` is unchanged.
  - The comparison casts IDs to `str`, so an object vs. string dtype difference is not treated as a change.
  - Identical re-runs proceed as before.
  - Tests are in `tests/test_split_drift.py`: 15 tests, all on synthetic data in `tmp_path`. Before the fix, 13 failed. The 2 that passed were the deliberate over-correction guards: an identical re-run, and a dtype-only difference.
  - Full suite after the fix: 110 passed.
  - Mapping to the required tests:
    - **Test 1** is covered by `test_identical_rerun_succeeds_and_keeps_content_hash` and `test_compare_assignments_ignores_dtype_and_row_order`, which asserts 0 moved.
    - **Test 2** is covered by `test_removed_customer_is_refused_with_counts_and_nothing_changes`. It reports the removed customer as a count (`removed: 1`), not as an ID, per D2-Q2.
    - **Test 3:** `test_split_is_reproducible_and_independent_of_row_order` still passes.
  - **Not covered by this fix:**
    - There is no programmatic resplit path: no override flag, no archiving of the previous split.
    - A legitimate split change, for example new customers, is refused. It needs a separately planned and approved design.
    - The previous assignment list is still not archived. The fix only prevents losing it by refusing to overwrite it.
    - Multi-file write atomicity and partial-failure recovery are not addressed. TD-4 stays Open.

## TD-4: Feature artifacts are overwritten one file at a time (not atomic as a set)

- **Affected code:** `src/churn_pipeline/features.py`, `run_features()`, the write sequence. It writes 3 split Parquets, then `split_assignments.parquet`, then `preprocessor.joblib`, then `feature_report.json` / `.md`, each directly to its final path.
- **Gap:** the 7 files are meant to be used together. They are written directly to their final names in sequence, with no temporary directory and no rename. If the process is interrupted, or a write fails partway, `data/processed/features/` can contain a mix of new and old files. For example, there could be new `train.parquet` files with an old `preprocessor.joblib` and an old report, and no marker says the set is inconsistent. This is the same class of limitation as Phase 1's direct Parquet overwrite.
- **Failure scenario:** not reproduced. It is reasoned from the code: an exception raised after the `train.parquet` write but before `feature_report.json` leaves mismatched artifacts.
- **Current protection:** all computation happens in `build_features()` before any write, so validation and transform errors leave the previous artifacts untouched. Only I/O failures during the write sequence are exposed.
- **Severity: Low.** It requires an I/O failure or interruption mid-write. The output is rebuildable by re-running.
- **Possible fix:** write all artifacts to a temporary sibling directory, then swap it in with a rename. Alternatively, write a manifest (file names plus hashes) last, and have readers check it.
- **Required regression tests:**
  1. Simulate a failure after the first artifact is written (for example, monkeypatch `joblib.dump` to raise). Assert that the previous artifact set is still complete and mutually consistent.
  2. `test_run_features_writes_reproducible_outputs` still passes.

## TD-5: No inference entry point applies the training-time input guards; `preprocessor.joblib` alone accepts invalid input silently

- **Affected code:** `src/churn_pipeline/features.py`. `run_features()` saves only the fitted `ColumnTransformer` (`preprocessor.joblib`). During training, `build_features()` wraps it with three guards that are plain functions in `features.py` and are **not** saved in the artifact:
  - `check_input()`: schema, IDs, labels, the `total_charges` null rule, and nulls in other features.
  - `_to_model_frame()`: converts nullable pandas dtypes to the plain dtypes sklearn expects.
  - The NaN check in `_transform()`.

  There is no function that applies the same guards to new, unlabeled data.
- **Gap:** anyone who loads `preprocessor.joblib` and calls `transform()` directly on new data skips all three guards. `check_input()` also cannot be reused as-is for inference, because it requires the `churn` label column, which inference data doesn't have.
- **Failure scenario (reproduced 2026-11-06, using customer `0002-ORFBO`, `tenure_months = 9`):**

  | Modified input | Through `check_input()` | `preprocessor.joblib` alone |
  |---|---|---|
  | `total_charges` set to missing | `DataContractError` | Silently filled with 0 → feature value −1.0058 (looks normal) |
  | `monthly_charges` set to missing | `DataContractError` | Outputs `NaN` for that feature |
  | `contract = "Three year"` (unseen value) | Not caught either (Phase 2 does not re-validate category values) | Encoded as all zeros, no warning |

- **Related:** Phase 2 guide Part 0 findings #1 (an unknown value in a binary column encodes the same as the dropped category), #2 and #3. The pickle format of the artifact also requires loading only self-produced files with the same scikit-learn version. The version is recorded in `feature_report.json` under `versions`, but nothing checks it at load time.
- **Severity: Medium.** It doesn't affect Phase 2 outputs, which are built through the guarded path. However, the first inference code would silently produce wrong features for invalid input.
- **Impact on Phase 2–4:**
  - Phase 2–3: none. Training and evaluation data come from `build_features()`.
  - Phase 4: must be addressed before any batch or online inference.
- **Possible fix (not decided):**
  - Add an inference function (for example `transform_for_inference(df, preprocessor, report)`). It would:
    1. Run a label-free variant of `check_input()`.
    2. Check category values against `categories_from_train` and count or reject unknowns.
    3. Apply `_to_model_frame()`.
    4. Transform.
    5. Check for NaN.
    6. Assert that the output columns equal `output_feature_columns`.
  - Also check the installed scikit-learn version against `versions` before loading.
  - Alternatively, save guards and transformer together as one artifact.
- **Required regression tests:**
  1. Each of the three scenarios above raises (or reports, for unknown categories, per the chosen policy) through the inference function.
  2. For valid rows, the inference function's output equals the corresponding rows of `train` / `validation` / `test.parquet` exactly.
- **Phase 3 note (2026-11-08):**
  - `data/processed/models/<run_id>/model.joblib` is a full `Pipeline` (or `VotingClassifier` of Pipelines). It wraps the same `ColumnTransformer`, so it **inherits this gap**: `model.predict_proba(df)` applies none of `check_input()`'s guards.
  - Phase 3 has no inference entry point. `train`, `evaluate` and `compare` only score rows that went through `load_verified_inputs()` (`check_input()` + `_to_model_frame()`).
  - Two small additions: `evaluate` refuses to unpickle a model if the installed scikit-learn version differs from the one recorded in the run manifest, and `diagnostics.drift_experiment` *demonstrates* the unseen-category case (20% `payment_method = "Crypto wallet"`). Feature PSI flags it (1.62), the champion's score distribution barely moves (score PSI 0.002), and no error is raised.
  - TD-5 stays Open and still blocks Phase 4.

---

## How to close an entry

1. Fix it in its own change, separate from feature work.
2. Add the regression tests listed above, and confirm that they **fail on the unfixed code** before the fix is applied.
3. Run the full suite and record the result.
4. Mark the entry *Fixed* with the date and commit. Don't delete it: the history is part of the learning record.
