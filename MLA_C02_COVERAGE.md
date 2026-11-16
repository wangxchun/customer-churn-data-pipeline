# MLA-C02 coverage register

This register tracks, skill by skill, what this local learning project actually covers for the AWS Certified Machine Learning Engineer – Associate (MLA-C02) exam. Each status rests on code, tests, or recorded experiments in this repository. Documentation alone doesn't count as implementation.

- **Source of skill text and numbering:** the official exam guide pages, retrieved 2026-11-08:
  - [Domain 1](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/machine-learning-engineer-associate-02-domain1.html)
  - [Domain 2](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/machine-learning-engineer-associate-02-domain2.html)
  - [Domain 3](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/machine-learning-engineer-associate-02-domain3.html) and [Domain 4](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/machine-learning-engineer-associate-02-domain4.html). These were retrieved 2026-11-16. Their numbering (30 + 25 skills) matches the Phase 4 plan.
  
  Skill text is quoted verbatim, and numbers are the official ones; none are invented here.
- **AWS hands-on:** none. The project runs only locally, so every AWS service below is at most *conceptual*.
- **Last verified:** 2026-11-08. That day's checks:
  - `pytest`: 95 passed.
  - Phase 3 runs `20261108T104225Z-cdbbaf` (seed 42), `20261108T104330Z-cfbf6f` (seed 42, repeat) and `20261108T104409Z-4dbf37` (seed 7).
  - Diagnostics `20261108T104524Z-908690`, and test evaluation `evaluation-01` of the seed-42 run.

  All run outputs live under `data/processed/models/`, which git ignores.
- **Phase 4 verification:** 2026-11-16. That day's checks:
  - Commit `ac450e0`.
  - The restricted `pytest` run passed: 477 passed, 2 deselected. It excludes the real-data tests `test_features.py::test_real_master_feature_build` and `test_modeling.py::test_real_data_preprocessor_reproduces_train_parquet_exactly`, and ignores `test_real_dataset.py`.
  - Real-data runs:
    - W1 release `rel-20261116T123424Z-d93c9b` was created, checked and promoted.
    - D7 traffic batch `d7-none-n1000-seed42` was generated.
    - The stage C acceptance batch `20261116T161427Z-3c5357` was predicted and monitored.
  - W4, commit `4287613`: the restricted run passed, 555 passed, 2 deselected (78 of them in `test_cost.py` and `test_benchmark.py`). The benchmark ran on two **synthetic** releases (LR, RF) in a temporary directory; no real release and no real data were used.
  - W6, commit `4324cd3`: the restricted run passed, 609 passed, 2 deselected (54 of them in `test_ab.py`). `test_ab.py` uses synthetic IDs only. There was no real-data run, and none is needed: `ab.py` reads no data.
  - W5, commit `521089e9f7672b9a382be0a999d39021fecae696`: `scripts/ci_check.sh` ran for real and exited 0: compileall passed, the secret scan had 0 findings in 62 files, and pytest gave 671 passed, 5 deselected (`-m "not real_data"`). It read no real data and checked no real release. Release permissions (0700/0600) were tested on synthetic releases only; the existing real release was not changed. The `os.fchmod` fallback (Windows before Python 3.13) was tested only by simulation (monkeypatch), not on Windows.

## Status codes

| Code | Meaning |
|---|---|
| **I** | Implemented and tested locally |
| **P** | Partially implemented: some named techniques or services are implemented, others are not |
| **S** | Simulated locally: the mechanism is imitated, but the real setting (for example real traffic or time) is absent |
| **C** | Conceptual study only: explained in the guides, with no code |
| **M** | Deferred to a mini-project: M1 = local RAG + GenAI evaluation; M2 = small-FM LoRA fine-tuning |
| **X** | Out of scope for this project by decision (Phase 4 A1: agent skills; no fake agent is built). Neither concept coverage nor implementation |
| **—** | Not yet assessed |

Guide references: P1 = `docs/INTERNAL_GUIDE_ZH-TW.md`, P2 = `docs/PHASE2_GUIDE_ZH-TW.md`, P3 = `docs/PHASE3_GUIDE_ZH-TW.md` (chapter letters), P4 = `docs/PHASE4_GUIDE_ZH-TW.md` (work-item chapter and section, e.g. "P4 W4 §5"). The `docs/` folder is intentionally git-ignored (`.gitignore`), so guide references point to local files only.

---

## Domain 1: Data Preparation for ML and AI

Moved here from the README ("MLA-C02 Task 1.1 coverage") and Phase 2 guide chapter L, with each skill's original status kept. Two entries gained Phase 3 evidence (1.3.3 and 1.3.5); each such change is marked **[Phase 3]**.

### Task 1.1: Collect and store data.

| Skill | Official text | Status | Evidence | Limitations |
|---|---|---|---|---|
| 1.1.1 | Extract data from data sources (for example, Amazon S3, Amazon EBS, Amazon EFS, Amazon RDS, Amazon DynamoDB, Amazon OpenSearch Service). | C (local stand-ins only) | Local files and SQLite (`sources.py`, `storage.py`) | No AWS APIs used |
| 1.1.2 | Make storage decisions and configure storage services based on cost, performance, data structure, and data compliance. | P | README storage table; Parquet 191 KiB vs CSV 948 KiB; provenance and checksum (`config.py`) | Compliance limited to provenance; no PII handling |
| 1.1.3 | Troubleshoot and debug data ingestion and storage issues that involve capacity and scalability. | P | Rejects with lineage, checksum failures, schema-drift checks; Phase 1 tests | Capacity and scalability not exercised |
| 1.1.4 | Use AWS streaming data sources to ingest data (for example, by using Amazon Kinesis, Apache Flink, Apache Kafka). | S | JSONL micro-batches with at-least-once replay (`events.py`, `tests/test_events.py`) | No real stream |
| 1.1.5 | Ingest from and write by using appropriate data formats (for example, Apache Parquet, JSON, CSV, ORC) based on data access patterns. | I | CSV, JSON, JSONL in; Parquet and SQLite out | ORC not used |
| 1.1.6 | Merge data from multiple sources (for example, by using programming techniques, AWS Glue, Apache Spark). | I (pandas) / C (Glue, Spark) | `merge.py`, `tests/test_merge.py` | |
| 1.1.7 | Configure scalable vector databases for AI applications (for example, OpenSearch Service, Amazon RDS with pgvector, Amazon S3) based on specifications. | M (M1) | — | Tabular project |
| 1.1.8 | Ingest and store diverse data types (for example, text, images, audio) for AI and ML applications. | M (M1) | — | Tabular data only |
| 1.1.9 | Ingest data into SageMaker Feature Store. | C | `customer_master` is a feature-store-style table | No Feature Store, no point-in-time logic |

### Task 1.2: Perform data transformation, feature engineering, and pre-processing.

| Skill | Official text | Status | Evidence | Limitations |
|---|---|---|---|---|
| 1.2.1 | Transform data by using AWS tools (for example, AWS Glue, AWS Glue DataBrew, Spark on Amazon EMR, SageMaker Data Wrangler). | P (local transform) / C (AWS tools) | `features.py`; 18 tests in `tests/test_features.py` | The four AWS tools not studied systematically |
| 1.2.2 | Create and manage features by using AWS tools (for example, SageMaker Feature Store). | C (local analogue) | `FEATURE_COLUMNS` / `EXCLUDED_COLUMNS` / `feature_report.json`; column-classification tests | No feature store |
| 1.2.3 | Transform streaming data (for example, by using AWS Lambda, Spark). | — | — | Micro-batch ingestion exists, streaming transformation does not |
| 1.2.4 | Perform feature engineering (for example, scaling, standardization, feature splitting, binning, log transformation, normalization). | P | Scaling and standardization (`build_preprocessor()`), tested | Feature splitting, binning, log transformation, and normalization not implemented |
| 1.2.5 | Configure and use embedding models to transform text and image data into numerical representations. | M (M1) | — | |
| 1.2.6 | Apply advanced text pre-processing techniques (for example, tokenization, domain-specific augmentation). | M (M1) | — | |
| 1.2.7 | Prepare documents for Retrieval Augmented Generation (RAG) applications (for example, chunking strategies, metadata extraction). | M (M1) | — | |
| 1.2.8 | Mask, redact, and anonymize data. | — | — | No PII handling |
| 1.2.9 | Prepare data for FM fine-tuning, continuous pre-training, and model distillation. | M (M2) | — | |

### Task 1.3: Validate data quality and manage bias.

| Skill | Official text | Status | Evidence | Limitations |
|---|---|---|---|---|
| 1.3.1 | Validate data quality (for example, by using DataBrew, AWS Glue Data Quality). | P (local) / C (AWS) | Phase 1 `validation.py`; Phase 2 `check_input()`; Phase 3 `load_verified_inputs()` refuses stale inputs | DataBrew and DQDL not studied systematically |
| 1.3.2 | Label and annotate data. | C | — | IBM data is already labelled; Ground Truth not studied |
| 1.3.3 | Identify and mitigate sources of bias in data by using AWS tools and techniques (for example, dataset splitting, shuffling, augmentation). | P | Stratified, shuffled split (Phase 2). **[Phase 3]** Per-group base rates are reported in the subgroup tables: in validation, seniors 44.1% vs non-seniors 23.4%; in test, 40.9% vs 23.3% | No augmentation; Clarify pre-training metrics (CI, DPL) only conceptual (P3 L) |
| 1.3.4 | Optimize multimodal data distributions by applying bias metrics across numeric, text, and image assets. | — | — | Numeric part possible locally; text and image parts → M1 |
| 1.3.5 | Resolve class imbalance in numeric, text, and image datasets. | **P [Phase 3]** (numeric) | `class_weight ∈ {None, "balanced"}` is in every applicable search space (`modeling.candidate_specs()`), so CV decides whether reweighting helps. The cost-based threshold (D4) handles imbalance at decision time. Seed-42 run: LR and RF chose `None`, HGB chose `"balanced"` | Text and image parts → M; no resampling (SMOTE etc.) implemented; P3 F explains why balanced weights shift probabilities |
| 1.3.6 | Validate AI training data integrity (for example, prompt-response pair validation, content safety screening). | M (M1/M2) | — | |
| 1.3.7 | Clean data (for example, by detecting outliers, imputing missing data, deduplication). | P | Missing values (Phase 2), deduplication (Phase 1), both tested | No outlier detection. Earlier copies of the task text read "inputting"; the official page now reads "imputing" |

---

## Domain 2: ML Model and Foundation Model (FM) Development

Each row has these columns:

- **Concept:** whether the Phase 3 guide teaches it.
- **Implementation:** the status code.
- **Evidence:** files and tests.
- **Limitations.**
- **Next step.**
- **MP:** whether a mini-project is needed.

### Task 2.1: Choose appropriate modeling approaches for ML and AI solutions.

| Skill | Official text | Concept | Implementation | Evidence | Limitations | Next step | MP |
|---|---|---|---|---|---|---|---|
| 2.1.1 | Evaluate and select appropriate FMs from Amazon Bedrock based on task requirements and performance criteria. | ◐ P3 Q (why an FM is the wrong tool for this tabular task; selection criteria) | **C + M** | — | No Bedrock, no FM | M1: compare local LLMs on one task with fixed metrics | M1 |
| 2.1.2 | Identify fine-tuning strategies for pre-trained FMs to meet business needs. | ◐ P3 Q (full FT vs PEFT/LoRA vs prompt; continued pre-training; distillation) | **C + M** | — | | M2 | M2 |
| 2.1.3 | Compare and select appropriate ML models, generative AI (GenAI) models, algorithms, and solution templates to meet business needs (for example, interpretability, domain-specific performance, latency). | ✅ P3 B, E | **I** (ML) / **C** (GenAI, solution templates) | `modeling.candidate_specs()`, `train_models()`, `select_champion()`; `selection.json` candidate table with CV/validation PR AUC + CI, ROC AUC, log loss, refit time, latency, size, interpretability. Tests: `test_selection_prefers_the_simpler_model_when_the_gap_is_noise`, `test_selection_keeps_the_better_model_when_the_gap_is_significant` | The champion is seed-sensitive: RF (seed 42) vs LR (seed 7). Interpretability enters only as a tie-breaker within noise. No GenAI models | Decided 2026-11-08: per-customer interpretability is not a hard constraint; RF stays champion (its robustness across seeds is not established) | M1 (GenAI part) |
| 2.1.4 | Evaluate tradeoffs between custom solutions, managed services, pre-trained models, and FMs to meet business needs. | ✅ P3 Q (decision matrix: custom sklearn vs Autopilot/Canvas vs AI services vs FM) | **C** | — | No managed service used | — | — |
| 2.1.5 | Select Retrieval Augmented Generation (RAG) architecture patterns based on use case requirements. | ◐ P3 Q | **M** | — | No documents to retrieve | M1 | M1 |
| 2.1.6 | Assess tradeoffs between ML model performance, training time, and cost. | ✅ P3 D, H | **I** | `search_seconds` / `refit_seconds` per candidate; `diagnostics.hpo_budget_experiment` (HGB: 1 → 30 trials, 3 s → 61 s of fit time, best CV PR AUC +0.003, below CV std 0.014); early stopping (9.7 s → 0.41 s) | Local CPU seconds, not instance-hours or dollars | Map to SageMaker instance pricing conceptually | — |
| 2.1.7 | Assess tradeoffs between AI model performance, latency, and cost. | ◐ P3 B | **P** | `modeling.serving_cost()`: median ms per 1k rows (LR 7.5, RF 31.9) and model size (LR 12 KB, RF 5.6 MB) | Tabular models only; in-process latency, not endpoint latency; no LLM token cost | M1 for LLM latency and cost | M1 |
| 2.1.8 | Apply AWS AI services to solve specific business problems (for example, Amazon Textract, Amazon Rekognition, Amazon Comprehend, Amazon Transcribe). | ◐ P3 Q (service-to-problem table) | **C** | — | No meaningful local substitute | — | — |

### Task 2.2: Train, fine-tune, and customize models for ML and AI solutions.

| Skill | Official text | Concept | Implementation | Evidence | Limitations | Next step | MP |
|---|---|---|---|---|---|---|---|
| 2.2.1 | Apply Amazon SageMaker AI built-in algorithms and common ML libraries. | ✅ P3 B, Q (built-in algorithm table; XGBoost ↔ HGB hyperparameter mapping) | **P** | scikit-learn is the common library (I). Built-in XGBoost and Linear Learner are C (D2: XGBoost not installed) | HGB is the same algorithm family as XGBoost, not the same implementation | — | — |
| 2.2.2 | Configure SageMaker AI script mode with supported frameworks for simplicity and performance. | ◐ P3 Q (`SM_CHANNEL_*` / `SM_MODEL_DIR` contract) | **C** | — | D6 declined the local interface simulation | — | — |
| 2.2.3 | Implement hyperparameter optimization (for example, SageMaker AI automatic model tuning [AMT]). | ✅ P3 D | **I** (local) / **C** (AMT) | `modeling.tune()`: `RandomizedSearchCV`, 20 trials per family, `StratifiedKFold(5)`, preprocessing re-fitted per fold; `cv_results.csv`. Tests: `test_every_candidate_is_cross_validated_on_identical_folds`, `test_preprocessor_is_refitted_on_each_folds_training_rows_only` | Random search only; AMT Bayesian, Hyperband, and warm start are conceptual | — | — |
| 2.2.4 | Implement training time reduction techniques (for example, early stopping, distributed training). | ✅ P3 H | **P** | Early stopping: `diagnostics.early_stopping_experiment` (HGB 1000 rounds 9.70 s vs 41 rounds 0.41 s; validation PR AUC 0.588 vs 0.647); test `test_early_stopping_stops_before_max_iter` | Distributed training is conceptual. `n_jobs` parallelism is not distributed training | — | — |
| 2.2.5 | Prevent model overfitting, underfitting, and catastrophic forgetting. | ✅ P3 G | **P** | Learning curves and regularization sweeps (`diagnostics.learning_curves`, `regularization_sweeps`): unregularized RF train PR AUC 1.000 vs CV 0.629; `min_samples_leaf=20` gives CV 0.664; `min_samples_leaf=300` underfits (0.637) | Catastrophic forgetting is C + M2 (no sequential or fine-tuning setting) | M2 | M2 (forgetting) |
| 2.2.6 | Combine multiple ML models to improve performance or reduce cost. | ✅ P3 I | **I** | Soft-voting LR + HGB (`modeling.build_ensemble`): validation PR AUC 0.656 vs best single 0.680, at 19.9 ms/1k rows and 116 KB. RF is itself a bagging ensemble | No stacking; the ensemble's members aren't re-tuned, and they use different `class_weight` settings in the seed-42 run | — | — |
| 2.2.7 | Adjust fundamental hyperparameters (for example, epoch, steps, batch size). | ✅ P3 H | **I** (diagnostics) | `diagnostics.epoch_batch_experiment`: 30 epochs × batch 16 / 64 / 256 / 4929 = 9270 / 2340 / 600 / 30 steps; test `test_steps_are_epochs_times_batches_per_epoch` | Done with an MLP that is not a candidate model; small network | — | — |
| 2.2.8 | Apply customization techniques for AI solutions (for example, task-specific prompt engineering, fine-tuning). | ◐ P3 Q | **M** | — | | M1 (prompting), M2 (fine-tuning) | M1/M2 |
| 2.2.9 | Optimize retrieval components and embedding models. | ◐ P3 Q | **M** | — | | M1 | M1 |

### Task 2.3: Analyze and evaluate the performance of ML and AI systems.

| Skill | Official text | Concept | Implementation | Evidence | Limitations | Next step | MP |
|---|---|---|---|---|---|---|---|
| 2.3.1 | Perform reproducible experiments (for example, by using MLflow on SageMaker AI, Amazon Bedrock evaluations, Amazon Bedrock Prompt Management). | ✅ P3 M | **I** (local manifest) / **C** (MLflow, Bedrock) | `tracking.build_manifest()`: MLflow-shaped params, metrics, tags, artifacts, plus input and split hashes, git state, versions. Two seed-42 runs produced a byte-identical `model.joblib` (sha256 `30149fa1…`). Test: `test_same_seed_gives_identical_metrics_and_predictions` | Byte identity measured once, not guaranteed. All runs were made with uncommitted code (`dirty: true`). No tracking server | Code is now committed (`bd38deb`), but the recorded runs still reference `14af8d0` + dirty; a manifest pointing to a clean commit would require a new run | — |
| 2.3.2 | Create model performance baselines and implement drift detection. | ✅ P3 N | **I** (baseline) + **S** (drift) | `baseline.json` (train feature bins, validation score bins, validation performance); `evaluation.feature_psi`, `score_psi`. Real train → validation/test max PSI 0.015 / 0.011; simulated shifts detected. Tests: `test_psi_*`, `test_drift_scenarios_are_detected_by_psi` | No time column, so drift over time can't be measured; PSI thresholds are a convention; no scheduled monitoring | Phase 4 (Model Monitor concepts) | — |
| 2.3.3 | Compare the performance of shadow variants to production variants. | ✅ P3 O | **S** | `evaluation.compare_runs()`: paired bootstrap, agreement, McNemar, cost; refuses mismatched splits. Seed-42 RF vs seed-7 LR: PR AUC gap −0.034 [−0.060, −0.008], cost 443 vs 428. Tests: `test_compare_*` | Offline replay of one validation set, not shadow traffic | Phase 4 | — |
| 2.3.4 | Explain model outputs. | ✅ P3 J | **I** | `evaluation.lr_coefficients`, `lr_decomposition` (exact; `test_lr_decomposition_sums_exactly_to_the_logit`), `permutation_importance_table` on validation. `explanations.json` | The seed-42 champion is RF, so the exact decomposition applies to the LR candidate, not the champion; the champion has only global permutation importance. SHAP and Clarify are conceptual (D3) | — | — |
| 2.3.5 | Debug model convergence issues. | ✅ P3 H | **I** | `diagnostics.convergence_experiment`: LR converges in 45 iterations on scaled inputs and still warns at 1000 on unscaled inputs; MLP learning rate 1.0 diverges (PR AUC 0.266). Test: `test_too_few_iterations_trigger_a_recorded_convergence_warning` | Small models; no deep-learning tooling | — | — |
| 2.3.6 | Apply comprehensive model evaluation techniques for traditional ML and GenAI models. | ✅ P3 F, K | **P** | ML is I: ROC/PR AUC, log loss, Brier, confusion matrix, cost, bootstrap CIs, calibration (ECE), subgroup metrics; one-time test evaluation (`run_final_evaluation`, `test_test_split_is_read_once_unless_forced_and_every_read_is_logged`) | GenAI evaluation is M1 | M1 | M1 |
| 2.3.7 | Implement integrated human evaluation frameworks (for example, human-in-the-loop workflows, text generation quality assessment). | ◐ P3 Q | **C + M** | — | Not forced into a churn project | M1 (annotation agreement, Cohen's κ) | M1 |
| 2.3.8 | Apply natural language processing (NLP) evaluation metrics (for example, bilingual evaluation understudy [BLEU], Recall-Oriented Understudy for Gisting Evaluation [ROUGE], BERTScore, semantic similarity). | ◐ P3 Q | **M** | — | | M1 | M1 |
| 2.3.9 | Perform AI evaluation (for example, model output assessment, content quality validation, bias detection, LLM-as-a-judge frameworks). | ✅ P3 L (bias) / ◐ Q (rest) | **P** | Tabular bias detection is I: `evaluation.subgroup_metrics` (Clarify-style DPPL, DI, recall difference with CIs; small groups flagged). Sensitive-feature-removal experiment (D6): no change in PR AUC or disparities; senior status is reconstructable from the other inputs (CV ROC AUC 0.787). Tests: `test_subgroup_metrics_and_disparities_hand_example`, `test_without_sensitive_removes_exactly_the_sensitive_model_inputs` | Only two attributes; senior group (n = 161 validation, 193 test) is below the reliability threshold; LLM-as-a-judge and content-quality validation are M1 | M1 | M1 (GenAI part) |
| 2.3.10 | Configure RAG system monitoring, including retrieval accuracy assessment. | ◐ P3 Q | **M** | — | | M1 | M1 |

### Domain 2 summary

| Code | Skills |
|---|---|
| I (fully, for this project's scope) | 2.1.6, 2.2.6, 2.2.7, 2.3.4, 2.3.5 |
| I with a C part (local done, AWS/GenAI conceptual) | 2.1.3, 2.2.3, 2.3.1 |
| I + S | 2.3.2 |
| P | 2.1.7, 2.2.1, 2.2.4, 2.2.5, 2.3.6, 2.3.9 |
| S | 2.3.3 |
| C | 2.1.4, 2.1.8, 2.2.2 |
| C + M / M | 2.1.1, 2.1.2, 2.1.5, 2.2.8, 2.2.9, 2.3.7, 2.3.8, 2.3.10 |

Every Domain 2 row has AWS hands-on status "none".

---

## Domain 3: Deployment and Orchestration of ML and AI Workflows

The columns are the same as in Domain 2. **Concept** refers to a Phase 4 guide, which doesn't exist yet, so it is "—" everywhere.

In Domains 3 and 4:
- **"—" in Implementation** means nothing is implemented yet; the Next step column names the planned work item.
- **"Real data"** in Evidence means a recorded run on this repository's real data. Everything else is synthetic-fixture tests only.
- **Every row has AWS hands-on status "none".** Local mechanisms are not AWS services.

Code and tests:

| Item | Code | Tests | Commit |
|---|---|---|---|
| TD-5 | `inference.py` | `test_inference.py` | `180714d` |
| W1 | `release.py` | `test_release.py` | `0c4f6a7`, `e60c747` |
| D7 | `traffic.py`, `simulate.py` | `test_traffic.py`, `test_simulate.py` | `286a6b1`, `5831092`, `d7c7d22` |
| W2 | `predict.py` | `test_predict.py` | `ac450e0` |
| W3 | `monitoring.py` | `test_monitoring.py` | `ac450e0` |

Real-data runs (all 2026-11-16):

| Name | What it is |
|---|---|
| **B2** | W1 release `rel-20261116T123424Z-d93c9b`: created, check-release, promoted |
| **D7-run** | Synthetic-ID traffic `d7-none-n1000-seed42`, resampled from train rows |
| **Stage C** | `predict` and `monitor-report` on that batch, producing batch `20261116T161427Z-3c5357` |

### Task 3.1: Manage deployment infrastructure for ML and AI model types.

| Skill | Official text | Concept | Implementation | Evidence | Limitations | Next step | MP |
|---|---|---|---|---|---|---|---|
| 3.1.1 | Select appropriate compute environments and deployment targets. | ✅ P4 W4 §3–4 | **C** (local evidence) | Real-time / serverless / async / batch decision matrix in the guide. W4 `benchmark` (`benchmark.py`, `test_benchmark.py`) measures guarded vs raw latency, load time and model size; **Synthetic releases only** (LR, RF; 2026-11-16), not real data | No endpoint deployed; in-process latency only; local CPU is not an AWS instance | — | — |
| 3.1.2 | Select deployment orchestrators and multi-model or multi-container deployment strategies. | — | **—** | — | Single tabular model; no multi-model case | Guide (planned C) | — |
| 3.1.3 | Select model inference strategies (for example, real-time and batch processing). | — | **I** (batch) **+ S** (single record) | Batch: `predict` CLI → `predict.run_predict`, `test_predict.py` (31 tests); **real data: Stage C**. Single record: `release.predict_one` / `inference.predict_one`, synthetic tests only | No real-time endpoint; single-record path is in-process, not a served endpoint | Guide: real-time vs batch vs async vs serverless | — |
| 3.1.4 | Evaluate and select appropriate foundation model (FM) deployment options. | — | **M** | — | No FM | M1 (+ guide C) | M1 |
| 3.1.5 | Deploy models that were built outside of AWS into AWS environments (for example, Amazon SageMaker AI, Amazon Bedrock Custom Model Import). | — | **—** | — | `model.joblib` not packaged; no SageMaker container (D14) | D14 guide: `model_fn` contract, `model.tar.gz` (planned C) | — |
| 3.1.6 | Deploy and configure agents for specific tasks, integration with other services and tools, and agent communication protocols. | — | **X** | — | Out of scope (A1) | — | — |
| 3.1.7 | Configure FM deployment, model hosting, and resource allocation. | — | **M** | — | | M2 (+ guide C) | M2 |
| 3.1.8 | Apply Retrieval Augmented Generation (RAG) system configurations (for example, retrieval strategies, reranking). | — | **M** | — | | M1 | M1 |

### Task 3.2: Provision and configure resources for ML and AI workloads based on existing architecture and requirements.

| Skill | Official text | Concept | Implementation | Evidence | Limitations | Next step | MP |
|---|---|---|---|---|---|---|---|
| 3.2.1 | Optimize resource provisioning between on-demand and provisioned resources for performance and cost efficiency. | ✅ P4 W4 §2, §6 | **C + S** | `cost.py`: always-on vs serverless vs batch cost formulas and `compare` (cheapest first); hand-computed exact tests and refusal of ≤ 0 / NaN / inf / bool (`test_cost.py`) | Prices are user inputs (guide examples are made up); no AWS pricing or billing data | — | — |
| 3.2.2 | Automate compute resource provisioning with integrated communication between stacks and orchestration services. | — | **—** | — | No IaC | Guide (planned C) | — |
| 3.2.3 | Build and maintain containers for ML and AI workloads. | — | **—** | — | No Dockerfile by decision (D15) | D15 guide (planned C) | — |
| 3.2.4 | Configure SageMaker AI endpoints within VPC network environments. | — | **—** | — | | Guide (planned C) | — |
| 3.2.5 | Deploy and host models programmatically (for example, by using the SageMaker AI SDK for Python, AWS CLI, Boto3). | — | **—** | — | No AWS SDK by decision (D14) | Guide (planned C) | — |
| 3.2.6 | Select specific metrics for auto scaling implementations. | ✅ P4 W4 §3, §7 | **C + S** | `cost.invocations_per_instance_target` turns measured single-row latency into an `InvocationsPerInstance` target; guide works it out from the W4 numbers (**Synthetic releases only** (LR, RF; 2026-11-16), not real data) | No Application Auto Scaling; target not validated by a load test | — | — |
| 3.2.7 | Create and manage Amazon Bedrock knowledge bases with vector database configurations, document indexing, and retrieval optimization. | — | **M** | — | | M1 | M1 |
| 3.2.8 | Implement retrieval pipelines to meet business needs. | — | **M** | — | | M1 | M1 |
| 3.2.9 | Implement agent state management systems. | — | **X** | — | Out of scope (A1) | — | — |
| 3.2.10 | Implement AI-specific resource scaling for GPU workloads. | — | **—** | — | No GPU workload | Guide (planned C; M2 may touch it) | — |
| 3.2.11 | Deploy agentic workflow infrastructure. | — | **X** | — | Out of scope (A1) | — | — |

### Task 3.3: Implement automated orchestration and continuous integration and continuous delivery (CI/CD) pipelines for MLOps and AI workloads.

| Skill | Official text | Concept | Implementation | Evidence | Limitations | Next step | MP |
|---|---|---|---|---|---|---|---|
| 3.3.1 | Implement automated deployment strategies and rollback actions. | — | **S** | `release.py`: gates G1–G5, G7–G9 (G6 removed by decision W1-Q2), atomic `promote` with a post-deploy canary and automatic pointer restore, and `rollback`. Tests: `test_release.py`. **Real data: B2** (release created, check-release and promote passed) | A local pointer, not a deployment; real `rollback` not exercised (no previous release); canary/blue-green traffic shifting not implemented | Guide: SageMaker deployment guardrails, blue/green, canary | — |
| 3.3.2 | Configure and troubleshoot AWS CodeBuild, AWS CodeCommit, AWS CodeDeploy, AWS CodePipeline, and AWS CodeConnections. | ✅ P4 W5 §5 | **C** | Guide maps each `ci_check.sh` step to a CodePipeline stage and a CodeBuild buildspec phase | No AWS CI/CD service used or configured | — | — |
| 3.3.3 | Configure training and inference jobs. | — | **P** | Training: Phase 3 `churn-pipeline train`. Batch inference job: `predict` (**real data: Stage C**) | Local jobs only; SageMaker training/processing/batch-transform job configuration not covered | Guide (C part) | — |
| 3.3.4 | Configure automated testing strategies within CI/CD pipelines for traditional ML and AI workloads. | ✅ P4 W5 §1–3 | **I** (local) | `scripts/ci_check.sh`: compileall → secret scan → pytest `-m "not real_data" --strict-markers`, stopping at the first failure (exit 0/1/2). The `real_data` marker excludes exactly the 5 real-data tests, which `test_repo_hygiene.py` checks by collection. Orchestration tested with a stub interpreter (`test_ci_check.py`). Real run: 671 passed, 5 deselected | Local script, not a hosted pipeline; no trigger on push; real-data and real-release checks are separate manual steps (R1, R2); no lint or type checking (D8-Q1) | — | — |
| 3.3.5 | Build and integrate mechanisms to re-train models. | — | **P** | W3 alerts set `retrain_review_recommended` (`monitoring.batch_record`). Tests prove that no training code is ever called (`test_monitoring.py`) | Retraining is deliberately manual; no trigger, pipeline or schedule | Guide: retraining triggers and pipelines (C) | — |
| 3.3.6 | Manage model versions for repeatability and audits (for example, SageMaker Model Registry, MLflow on SageMaker AI). | — | **I** (local) | Phase 3 run manifests (`tracking.py`); W1 immutable releases, `production.json` pointer, append-only `audit_log.jsonl`, D6 waiver. **Real data: B2** | MLflow-shaped JSON, not MLflow; no SageMaker Model Registry; manifests unsigned | Guide (C part) | — |
| 3.3.7 | Manage prompts (for example, Amazon Bedrock Prompt Management). | — | **M** | — | | M1 | M1 |
| 3.3.8 | Implement automated agent deployment pipelines and agent version management. | — | **X** | — | Out of scope (A1) | — | — |
| 3.3.9 | Implement AI model testing frameworks, including prompt testing. | — | **M** | — | ML testing part counted under 3.3.4 | M1 | M1 |
| 3.3.10 | Configure FM deployment automation with fine-tuned model versioning. | — | **M** | — | | M2 | M2 |
| 3.3.11 | Configure AI-specific pipeline orchestration for RAG system updates and knowledge base refresh cycles. | — | **M** | — | | M1 | M1 |

### Domain 3 summary

| Code | Skills |
|---|---|
| I (local) | 3.3.4, 3.3.6 |
| I + S | 3.1.3 |
| S | 3.3.1 |
| P | 3.3.3, 3.3.5 |
| C + S | 3.2.1, 3.2.6 |
| C | 3.1.1, 3.3.2 |
| — (planned; not done) | 3.1.2, 3.1.5, 3.2.2, 3.2.3, 3.2.4, 3.2.5, 3.2.10 |
| M | 3.1.4, 3.1.7, 3.1.8, 3.2.7, 3.2.8, 3.3.7, 3.3.9, 3.3.10, 3.3.11 |
| X | 3.1.6, 3.2.9, 3.2.11, 3.3.8 |

---

## Domain 4: Operating, Monitoring, and Securing ML and AI Solutions

The columns and conventions are the same as in Domain 3.

### Task 4.1: Monitor ML and AI model inference and performance.

| Skill | Official text | Concept | Implementation | Evidence | Limitations | Next step | MP |
|---|---|---|---|---|---|---|---|
| 4.1.1 | Monitor model performance in production by using Amazon CloudWatch generative AI observability, Amazon Bedrock Model Evaluation, and drift detection pipelines. | — | **P** | Drift detection pipeline: `monitoring.py` (feature PSI vs train baseline, score PSI vs validation baseline), `test_monitoring.py`. **Real data: Stage C** (max feature PSI 0.0105, score PSI 0.027, no alerts) | No labels, so performance itself is not monitored; CloudWatch and Bedrock evaluation not covered; score-PSI baseline mismatch (D7 traffic from train rows vs validation score baseline) | Guide (CloudWatch C); Bedrock → M1 | M1 |
| 4.1.2 | Monitor workflows to detect anomalies or errors in data processing or model inference. | — | **I** (local) | Reject rate, per-column null rate, unseen share and data-quality alerts in `monitoring.batch_record`; fail-closed errors and exit codes 0/2/3 in `predict`. **Real data: Stage C** | Local JSONL, not CloudWatch alarms | Guide (C part) | — |
| 4.1.3 | Detect changes in data distribution that can affect model performance. | — | **I + S** | PSI with P8 thresholds (> 0.25). Scenario tests on synthetic traffic: `price_up`, `contract_mix`, `unseen_payment` and `nulls` raise the expected alerts; `none` raises none. **Real data: Stage C** (no-drift batch, no alerts) | Drift alerts are proven on synthetic shifts only; PSI ignores sample size; score PSI must not be read alone (validation baseline vs train-resampled traffic) | — | — |
| 4.1.4 | Monitor model performance in production by using A/B testing. | ✅ P4 W6 | **C + I** (router and sample size only, D10) | `ab.assign_variant`: sha256-salted sticky assignment, verified against an independent hashlib computation, stable across `PYTHONHASHSEED`, proportions within 4 SE on 100k synthetic IDs. `ab.required_sample_size`: two-sided two-proportion z-test, matching the textbook hand example (0.20 vs 0.30 → 294 per group). Tests: `test_ab.py` (synthetic IDs only) | No online A/B test, no live traffic, no outcome collection or analysis, and no outcomes are simulated (D10); no SageMaker production variants; equal-size groups only | — | — |
| 4.1.5 | Monitor and automate the management of agent performance and coordination (for example, coordination failure detection, truncated streaming, tool failures). | — | **X** | — | Out of scope (A1) | — | — |
| 4.1.6 | Configure AI-specific performance monitoring for foundation models (FMs), such as Amazon Bedrock evaluations. | — | **M** | — | | M1 | M1 |

### Task 4.2: Optimize and manage ML and AI infrastructure costs and performance.

| Skill | Official text | Concept | Implementation | Evidence | Limitations | Next step | MP |
|---|---|---|---|---|---|---|---|
| 4.2.1 | Select inference instance families to optimize performance and cost. | ✅ P4 W4 §5 | **C** (local evidence) | Guide: instance families vs this workload (small CPU tabular models). W4 benchmark gives relative evidence (LR vs RF, guarded vs raw); **Synthetic releases only** (LR, RF; 2026-11-16), not real data | Local arm64 CPU maps to no `ml.*` instance; no Inference Recommender | — | — |
| 4.2.2 | Configure and use tools to troubleshoot and analyze resources (for example, Amazon CloudWatch, Amazon Bedrock AgentCore Observability, AWS X-Ray). | ◐ P4 W4 §8 | **P** | Coarse stage timings in `batch_manifest.json` and monitoring records (`load_release`, `read_input`, `predict_batch`, `write`). **Real data: Stage C**. W4 benchmark separates guarded from raw latency (synthetic releases only) | Validate/transform/predict are not timed separately in `predict` (accepted P5); no CloudWatch or X-Ray; AgentCore Observability not covered | Guide: W1–W3 chapter | — |
| 4.2.3 | Set up dashboards to monitor performance metrics. | — | **S** | `monitor-report`: a static Markdown report across batches (prints only). **Real data: Stage C** | A static report, not a dashboard | Guide: CloudWatch dashboards (C) | — |
| 4.2.4 | Optimize capacity for cost, performance, and reliability. | ✅ P4 W4 §7 | **C + S** | Scaling target with headroom (`invocations_per_instance_target`) and cost comparison (`compare`); guide covers min/max instances, cooldown, and load time vs scale-out | Nothing is actually scaled; no reliability or load test | — | — |
| 4.2.5 | Optimize costs and set cost quotas by using appropriate cost management tools. | — | **—** | — | | Guide (planned C) | — |
| 4.2.6 | Optimize infrastructure costs by selecting purchasing options. | ✅ P4 W4 §6 | **C** | Guide: On-Demand, Savings Plans, Spot (training), serverless provisioned concurrency. `cost.py` takes the hourly price as an input, so a discounted rate can be plugged in | Purchasing options are not modelled in code (no commitment term or discount logic); downgraded from the planned C + S | — | — |
| 4.2.7 | Evaluate cost implications of using FMs for inference in production. | — | **M** | — | | M1 (+ guide C) | M1 |
| 4.2.8 | Monitor agent resource consumption patterns. | — | **X** | — | Out of scope (A1) | — | — |
| 4.2.9 | Manage FM inference costs with usage optimization. | — | **M** | — | | M1 | M1 |
| 4.2.10 | Monitor AI-specific cost patterns (for example, token usage optimization, embedding computation costs, vector database storage optimization). | — | **M** | — | | M1 | M1 |

### Task 4.3: Secure ML and AI workloads and model endpoints.

| Skill | Official text | Concept | Implementation | Evidence | Limitations | Next step | MP |
|---|---|---|---|---|---|---|---|
| 4.3.1 | Secure continuous integration and continuous delivery (CI/CD) pipelines by checking for code and image vulnerabilities (for example, by using Amazon CodeGuru, Amazon Inspector). | ✅ P4 W5 §4–5 | **P** | Secret scan as a CI step (`scripts/secret_scan.py`, stdlib regex for 11 credential formats, tracked plus untracked-not-ignored files, values never printed); tests with runtime-built fake credentials (`test_secret_scan.py`); the repo itself scans clean (`test_repo_hygiene.py`). CodeGuru and Inspector are conceptual only (guide) | Regex scan of known formats only: no entropy detection, no git history, no ignored files; no dependency or image vulnerability scan (D9); no container image exists | — | — |
| 4.3.2 | Configure least privilege access to ML and AI artifacts. | ✅ P4 W5 §4b | **P** | Holdout guard file 0600; guard key file must be owner-only. W5: a new release directory is created 0700 and its files 0600, set exactly whatever the umask, and checked before the atomic rename. A looser or unexpected entry blocks the release (`release.create_release`, `tracking.atomic_directory(mode=)`, `tracking.write_json(mode=)`). Tests: `test_release_permissions.py`, synthetic releases under umasks 000, 022, 027, 077 and 277; failures leave no release | Local POSIX file modes only, not IAM. Releases created before W5, including the real `rel-20261116T123424Z-d93c9b` (0755), are not changed. The production pointer, audit log and `releases/` parent keep umask defaults. Same-user processes can still read everything. Not enforced on Windows | Guide: IAM (C) | — |
| 4.3.3 | Configure IAM policies and roles for users and applications in ML and AI systems. | — | **—** | — | No IAM locally | Guide (planned C) | — |
| 4.3.4 | Configure comprehensive monitoring, auditing, compliance, and logging for ML and AI systems (for example, AWS CloudTrail, AWS Config). | — | **S** | Append-only `audit_log.jsonl` (release, gate failure, promote, rollback events); release manifests; monitoring JSONL. **Real data: B2, Stage C** | No CloudTrail or Config; logs are local files | Guide (C part) | — |
| 4.3.5 | Troubleshoot and debug security issues in ML and AI systems. | — | **P** | Actionable fail-closed errors for hash, version, key, guard and baseline problems (`inference.py`, `release.py`, `monitoring.py`), with tests | No AWS access-denied, KMS or VPC troubleshooting | Guide (C part) | — |
| 4.3.6 | Create VPCs, subnets, and security groups to securely isolate ML and AI systems. | — | **—** | — | | Guide (planned C) | — |
| 4.3.7 | Identify and mitigate security risks and vulnerabilities in ML and AI systems. | ◐ P4 W5 §4 | **I** (local) | sha256 checked before unpickling (TD-5); input validation; HMAC holdout guard with external key (W1); no IDs or raw values in logs, records or reports (tests). **Real data: B2, Stage C**. W5: secret scan in CI; tests that `data/`, `docs/` and `.venv/` stay ignored and no data file is tracked (`test_repo_hygiene.py`) | No signing (manifest and model can be replaced together); holdout guard matches exact strings only (accepted M3) | Guide: W1–W3 chapter | — |
| 4.3.8 | Select the appropriate credential type to access FMs (for example, Amazon Bedrock API keys, IAM credentials). | — | **—** | — | | Guide (planned C) / M1 | M1 |
| 4.3.9 | Implement safeguards and sensitive data protection to meet application requirements and responsible AI policies (for example, by using Amazon Bedrock Guardrails). | — | **P** | Customer IDs and raw feature values excluded from monitoring records and reports (tests); holdout IDs stored only as HMACs; D2-Q2 counts-only drift report | Bedrock Guardrails not covered; rejects files keep `customer_id` locally by decision (D3) | Guardrails → M1 | M1 |

### Domain 4 summary

| Code | Skills |
|---|---|
| I (local) | 4.1.2, 4.3.7 |
| I + S | 4.1.3 |
| S | 4.2.3, 4.3.4 |
| P | 4.1.1, 4.2.2, 4.3.1, 4.3.2, 4.3.5, 4.3.9 |
| C + I | 4.1.4 |
| C + S | 4.2.4 |
| C | 4.2.1, 4.2.6 |
| — (planned; not done) | 4.2.5, 4.3.3, 4.3.6, 4.3.8 |
| M | 4.1.6, 4.2.7, 4.2.9, 4.2.10 |
| X | 4.1.5, 4.2.8 |

Every Domain 3 and Domain 4 row has AWS hands-on status "none".
