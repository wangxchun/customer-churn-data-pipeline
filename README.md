# Customer Churn Data Pipeline: Phase 1 (Ingestion and Storage) and Phase 2 (Features)

A small, local, reproducible pipeline that collects, validates, merges, and stores data for a customer-churn classification task. I built it as a learning project for **AWS Certified Machine Learning Engineer – Associate (MLA-C02), Domain 1, Task 1.1 "Collect and store data"**. It uses no AWS or other paid services.

Phase 1 covers ingestion and storage. Phase 2 covers transformation and feature engineering: missing values, encoding, scaling, a reproducible split, and leakage prevention (see [Phase 2](#phase-2-features-and-leakage-prevention)). No model is trained yet, and Phase 3 (data quality and bias) hasn't been implemented. Known open bugs are tracked in [`TECH_DEBT.md`](TECH_DEBT.md).

---

## Quick start

Requires Python ≥ 3.11 (verified with 3.13.7 on macOS).

```bash
cd customer-churn-data-pipeline
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # exact pinned versions
pip install -e . --no-deps             # installs the `churn-pipeline` command

churn-pipeline download                # fetch IBM CSV, verify SHA-256
churn-pipeline generate-synthetic      # write synthetic Sources B and C
churn-pipeline run                     # validate → store → ingest events → merge → summarise
churn-pipeline build-features          # Phase 2: split, fit preprocessing on train only, write features
pytest                                 # run the tests
```

To simulate events arriving over time:

```bash
churn-pipeline generate-synthetic --batches 1 && churn-pipeline run   # 400 events inserted
churn-pipeline generate-synthetic --batches 3 && churn-pipeline run   # 553 new, 50 replays skipped
churn-pipeline run                                                    # 0 new, store unchanged
```

To start over, delete `data/processed/`. Everything in it can be rebuilt from `data/raw/`.

### Inspecting the outputs

```bash
cat data/processed/ingestion_summary.md
sqlite3 data/processed/churn.db "SELECT source, reason, COUNT(*) FROM rejected_records GROUP BY 1,2;"
sqlite3 data/processed/churn.db "SELECT * FROM ingested_files;"
python -c "import pandas as pd; print(pd.read_parquet('data/processed/curated/customer_master.parquet').info())"
```

---

## Data provenance

| Source | Format | Origin | Real or synthetic |
|---|---|---|---|
| A: Customers and churn label | CSV | IBM Telco Customer Churn | **Real** (public sample dataset) |
| B: Customer profiles | JSON | `churn_pipeline/synthetic.py`, seed 42 | **Synthetic** |
| C: Usage events | JSONL, 3 batches | `churn_pipeline/synthetic.py`, seed 42 | **Synthetic** |

### Source A: verified on 2026-11-04

- Repository: <https://github.com/IBM/telco-customer-churn-on-icp4d> (archived; last push 2020-06-25)
- File: `data/Telco-Customer-Churn.csv`, pinned to commit `d5371f5d83a446ad5673cbcca3b814b926491f8a`
- URL: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv`
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`, 970,457 bytes
- Verified schema: 7,043 rows × 21 columns. `customerID` is unique. `Churn` has 5,174 No and 1,869 Yes.
- Known quirk: `TotalCharges` is a text column. It has 11 values of `" "` (a single space), and all of them belong to customers with `tenure == 0`.
- License: the repository is published under **Apache License 2.0**. Its README doesn't state a separate license for the dataset or say where the data originally came from. Attribute it as "IBM, telco-customer-churn-on-icp4d", and check the terms yourself before any use beyond learning.

`config.py` pins the URL and checksum. Both `download` and `run` refuse to continue if the file's bytes differ.

### Sources B and C (synthetic)

These sources were generated for this project. They are **not IBM data and not real customer data**. They reuse IBM `customerID` values only so that the joins have something to match. Every file says so: the JSON has a `metadata.synthetic: true` field, and `data/raw/events/SYNTHETIC_DATA_NOTICE.txt` sits next to the event files. Each source contains deliberate defects, listed in `INJECTED_PROFILE_DEFECTS` and `INJECTED_EVENT_DEFECTS`, so that the reject handling can be checked.

---

## What the pipeline does

```
data/raw/ (immutable)                         data/processed/ (rebuildable)
──────────────────────                        ─────────────────────────────
ibm/*.csv  ──validate──► customers  ─────────► curated/customers.parquet
                         │ (full refresh)      churn.db: customers
profiles/*.json ─validate─► dedupe ──────────► curated/customer_profiles.parquet
                         │ (full refresh)      churn.db: customer_profiles
events/*.jsonl ──manifest check──► validate ─► churn.db: events  (append-only, PK)
                (incremental)        │         churn.db: ingested_files
                                     └───────► churn.db: rejected_records
churn.db ──SQL aggregate──► join ────────────► curated/customer_master.parquet
                                               curated/events.parquet
                                               ingestion_summary.{md,json}
```

| Module | Responsibility |
|---|---|
| `config.py` | Paths, pinned source URL and checksum |
| `download.py` | Downloads to a temp file, verifies it, then renames it into place |
| `validation.py` | Declarative column specs. Returns valid rows plus rejects with reasons; also does deduplication |
| `sources.py` | Per-source schemas and loaders (CSV, JSON, JSONL) |
| `storage.py` | SQLite schema and Parquet writer |
| `events.py` | Idempotent incremental event ingestion |
| `merge.py` | Customer master table and join diagnostics |
| `summary.py`, `pipeline.py`, `cli.py` | Reporting, orchestration, command-line interface |

### Results of a full run

From `ingestion_summary.md` after ingesting all three batches:

| | Read | Valid | Rejected |
|---|---|---|---|
| IBM customers | 7,043 | 7,043 | 0 |
| Profiles | 6,350 | 6,343 | 7 (1 not an object, 1 missing ID, 1 bad region, 1 naive timestamp, 1 exact duplicate, 2 superseded) |
| Events, batch 001 | 400 | 400 | 0. 400 inserted |
| Events, batch 002 | 350 | 350 | 0. 300 inserted, **50 replays counted as duplicates** |
| Events, batch 003 | 261 | 255 | 7 (6 fail validation, plus **1 conflicting ID** found against the store). 253 inserted, 1 in-file duplicate |

Join results: 6,338 customers have a profile and 705 don't (kept, with null profile fields). 5 orphan profiles and 3 orphan events were reported. 484 customers have events. **The master table has 7,043 rows, one per labelled customer.**

---

## Design decisions

### 1. Read everything as strings, then type it in one place

**Problem.** `pd.read_csv` infers types. Because of the 11 blank values, `TotalCharges` silently becomes `object`. With different settings, `"NA"` would silently become NaN.
**Approach.** Every source is read as raw values (`dtype=str, na_filter=False` for CSV; untouched JSON values for JSON). `validation.py` then parses each column against a declared spec: kind, nullable, allowed values, regex, minimum value, and required timezone.
**Alternatives.** pandera, Great Expectations, pydantic, JSON Schema. They're better at scale, but they add dependencies and hide the mechanics this project is meant to teach.
**Trade-off.** Python-level, row-by-row parsing is slow on millions of rows. Real pipelines use vectorised checks or Spark.

### 2. Contract errors stop the run; record errors are quarantined

A missing column, or a JSON document that won't parse, raises `SchemaError`, because continuing would produce silently wrong data. A bad value sends that one row to `rejected_records`, along with the source, line or record reference, key, reason, and raw content. The rest of the batch continues. Nothing is dropped silently: `read = valid + rejected` holds for every source.

A cross-field rule handles the Telco quirk: a blank `TotalCharges` is accepted only when `tenure == 0`, since a brand-new customer hasn't been billed yet. Those rows are stored as null and flagged. Imputing them is Phase 2's job, because imputation choices affect leakage.

### 3. Duplicate policy depends on the source

| Situation | Policy | Reason |
|---|---|---|
| Exact duplicate row | Keep one, record the others | Harmless repetition |
| Same customer ID, different content, no timestamp (IBM) | Reject all copies | No safe way to choose, especially when the churn label disagrees |
| Same profile ID, different `updated_at` | Latest wins; older ones rejected as superseded | Profiles are mutable state |
| Same profile ID **and** same `updated_at`, different content | Reject all | A genuine conflict |
| Same `event_id`, same payload | Count as a duplicate; don't store | At-least-once delivery is normal |
| Same `event_id`, different payload | First seen wins; reject the newcomer with lineage | Events are immutable facts; a changed payload is an upstream bug |

### 4. Idempotent incremental ingestion (`events.py`)

- **File level:** `ingested_files` is keyed by the file's SHA-256, not its name. A renamed copy is skipped. An edited file has a new hash, so it is processed again.
- **Record level:** `events.event_id` is the `PRIMARY KEY`, inserted with `ON CONFLICT DO NOTHING`. The database guarantees exactly one row per event, however many times it is replayed. This is the real guarantee; the file check is only a shortcut.
- **Payload hash:** computed on *normalised* values, so `10:00Z` and `12:00+02:00` count as the same event, while a different `event_type` counts as a conflict.
- **Atomicity:** each file's events, its rejects, and its manifest row commit in **one transaction**. A crash halfway through rolls everything back, and the file is retried cleanly next run. `test_crash_mid_file_rolls_back_and_retry_succeeds` checks this.
- **Raw files are never moved or deleted.** Processing state lives in the manifest, so `data/raw` stays an audit trail that can be replayed.

The common alternative is "move the file to `processed/` after loading". The move and the database write can't be made atomic together, and the move destroys the replay trail.

### 5. Joins that neither multiply nor lose rows (`merge.py`)

- The **IBM customers are the spine** and the join is a `left` join, because they carry the label. An `inner` join would silently drop the 705 customers without a profile (`test_inner_join_would_silently_lose_labelled_customers`).
- Profiles are deduplicated *before* the join, and the join uses `validate="one_to_one"`. A single duplicate profile would otherwise turn 7,043 rows into 7,044 (`test_duplicate_profiles_would_multiply_rows_and_are_blocked`).
- Events are **aggregated to one row per customer in SQL first** (counts by type, total duration, first and last timestamps), then joined. Joining raw events directly would produce one row per event.
- After the joins, the code asserts that master rows equal IBM customers. I tested the guards by removing them: three `validate=` checks plus this assertion all have to be removed before a multiplied table gets through.
- Unmatched keys are **measured, not dropped silently**. An outer-join probe with `indicator=True` counts orphans on both sides. Orphan events stay in the store, because a late-arriving customer could match them later; they are only excluded from the features.
- A customer with no events gets counts of `0` and null timestamps. "No events observed" is a fact, not missing data.

### 6. Which storage for what

| Format | Use it for | Avoid it for | Here |
|---|---|---|---|
| **CSV** | Exchange with humans and spreadsheets; simple flat tables | Types (everything is text), nested data, large scans | Source A as published (948 KiB) |
| **JSON** | One document with nesting and metadata, such as configs or API payloads | Large or appendable data: one syntax error invalidates the whole file, and the whole file must be parsed | Source B: metadata plus a records array |
| **JSONL** | Append-only logs and event streams | Analytics: still text, still row-oriented | Source C: a bad line costs one line, and files can be appended to |
| **Parquet** | Analytical and ML reads: columnar, typed, compressed, readable column by column | Single-row updates, frequent small appends | Curated outputs. `customers.parquet` is **191 KiB versus 948 KiB of CSV**, with types preserved |
| **SQLite** | State that needs keys, constraints, transactions, and point lookups | Concurrent writers, very large analytical scans | Event store, manifest, rejects |

The index `idx_events_customer_ts (customer_id, event_ts)` matches the main access pattern, "events for customer X in time order". `EXPLAIN QUERY PLAN` shows `SEARCH events USING INDEX idx_events_customer_ts`, with no separate sort step.

---

## Equivalent AWS services (not implemented)

| Here | AWS equivalent |
|---|---|
| `data/raw/` kept immutable, `data/processed/` | S3 raw and curated zones of a data lake |
| Pinned URL plus SHA-256 | S3 object versioning; checksums on upload (`x-amz-checksum-sha256`) |
| `validation.py` | AWS Glue Data Quality (DQDL rules), SageMaker Data Wrangler, Glue DataBrew |
| Rejected-records table | Dead-letter queues (SQS), quarantine prefix in S3 |
| JSONL batches in a landing folder | Kinesis Data Firehose delivering micro-batches to S3; Kinesis Data Streams / Amazon MSK for real streams |
| `ingested_files` manifest | AWS Glue job bookmarks |
| `event_id` primary key, `ON CONFLICT DO NOTHING` | DynamoDB conditional writes (`attribute_not_exists`); `MERGE` in Athena with Iceberg tables |
| SQLite | Amazon RDS / Aurora (relational state); DynamoDB (key-value) |
| Parquet curated tables | S3 plus Glue Data Catalog plus Athena |
| `merge.py` | AWS Glue ETL / Spark on EMR |
| `customer_master` (one row of features per customer) | SageMaker Feature Store offline store (conceptually) |

---

## MLA-C02 Task 1.1 coverage

The skills below are taken from the official [MLA-C02 Domain 1 guide](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/machine-learning-engineer-associate-02-domain1.html).

| Skill | Status | Where |
|---|---|---|
| 1.1.1 Extract data from data sources | ◐ Concept | Local files plus SQLite stand in for S3 and RDS. No AWS APIs are used |
| 1.1.2 Storage decisions based on cost, performance, structure, and compliance | ✅ | Storage table; Parquet vs CSV size; provenance, license, and checksum. ◐ Compliance is limited to provenance; there is no PII handling |
| 1.1.3 Troubleshoot ingestion and storage issues | ◐ | Rejects with line-level lineage, checksum failures, schema-drift warnings. Capacity and scalability aren't exercised at this scale |
| 1.1.4 Ingest from streaming sources | ◐ Simulated | Micro-batches of JSONL files with at-least-once replay. No real stream |
| 1.1.5 Use appropriate data formats for the access pattern | ✅ | CSV, JSON, JSONL in; Parquet and SQLite out |
| 1.1.6 Merge data from multiple sources | ✅ | `merge.py`, with join-cardinality guards and orphan reporting |
| 1.1.7 Vector databases | ❌ | Out of scope |
| 1.1.8 Diverse data types (text, image, audio) | ❌ | Tabular data only |
| 1.1.9 Ingest into SageMaker Feature Store | ◐ Concept | `customer_master` is a feature-store-style table. There is no point-in-time logic yet |

---

## Limitations

- **Synthetic sources carry no signal.** Profiles and events are random, so they won't help predict churn. They exist only to exercise the pipeline mechanics.
- **There is no observation date.** The IBM data doesn't say *when* each customer churned, so we can't guarantee that event aggregates come only from before churn. Phase 2 must define a cutoff date before any event features are used, or they risk leaking the label.
- **Full-refresh sources** (customers, profiles) are re-validated and replaced on every run. That's fine at this size, but not at scale.
- **File arrival isn't coordinated.** A file that is still being written could be ingested half-finished. Real systems write to a temp name and atomically rename it, or use `_SUCCESS` marker files.
- **Single writer.** SQLite serialises writes, so the pipeline must not run twice concurrently.
- **Conflict policy is "first seen wins".** A deliberate correction from upstream would be rejected and need manual handling.
- **Orphan events are stored but not re-checked automatically.** If a matching customer arrives later, the features pick the events up on the next run, but nothing raises an alert about it.
- Validation runs row by row in Python, which is clear but slow above roughly 10⁶ rows.
- **Known open bugs:** `replace_table()` isn't atomic, and `events.event_id` accepts `NULL` at the database level. Neither affects Phase 2 outputs. Details, failure scenarios, and planned fixes are in [`TECH_DEBT.md`](TECH_DEBT.md).

## Phase 2: features and leakage prevention

`churn-pipeline build-features` reads `curated/customer_master.parquet` and writes to `data/processed/features/`:

| File | Content |
|---|---|
| `train.parquet`, `validation.parquet`, `test.parquet` | `customer_id`, 40 feature columns, `churn` (0/1). Use `features.load_split()` to get `(ids, X, y)` |
| `split_assignments.parquet` | `customer_id` → `train` / `validation` / `test` |
| `preprocessor.joblib` | The fitted `ColumnTransformer` (pickle: load only files you produced, with the same scikit-learn version) |
| `feature_report.{json,md}` | Input checksum, split sizes and churn rates, fill counts, train-only scaling parameters and categories, excluded columns with reasons |

| Decision | Choice | Reason |
|---|---|---|
| Feature selection | 19 IBM columns. Every input column must be classified as feature, ID, target, or excluded-with-reason, or the run fails | A new column can't become a model input by accident |
| Excluded | `customer_id`, `churn`, all synthetic profile and event columns | IDs don't generalise; the target can't be an input; the synthetic sources were generated independently of churn |
| Missing `total_charges` | Filled with the constant 0.0, only when `tenure_months == 0`; any other null fails the run | All 11 blanks have zero tenure, so "not billed yet" is inferred, **not confirmed** by IBM. A constant learns nothing from the data, so it can't leak |
| Split | Stratified on churn, 70/15/15, seed 42, rows sorted by `customer_id` first | Same seed and data give the same split, whatever the row order |
| Encoding | `OneHotEncoder(drop="if_binary", handle_unknown="ignore")` | Yes/No becomes one 0/1 column; a category unseen in training becomes all zeros |
| Scaling | `StandardScaler` on `tenure_months`, `monthly_charges`, `total_charges`; `senior_citizen` passed through | It's already 0/1 |
| Leakage control | `fit` is called once, on training rows only; validation and test are only transformed | Tests check that changing holdout rows doesn't change the fitted parameters or the training features |

Result on the real data: 4,929 / 1,057 / 1,057 rows, churn rate 26.54% / 26.58% / 26.49%, and the 11 filled values fall 9 / 1 / 1 across the splits.

**What this split does not show.** The IBM data has no snapshot date and no churn date, so the split is random, not temporal. It estimates performance on unseen customers from the same snapshot. It says nothing about how a model would do on a future period. Event features stay excluded until a cutoff date can be defined.

## Tests

```bash
pytest -v
```

The tests use small handcrafted fixtures and don't need the download. They cover normal operation, malformed CSV, JSON, and JSONL, missing columns, duplicate and conflicting events, crash rollback, unmatched IDs, join multiplication, and idempotent re-runs. `tests/test_real_dataset.py` also checks the real file's checksum, row count, and the quirk above; it is skipped if the file hasn't been downloaded.
