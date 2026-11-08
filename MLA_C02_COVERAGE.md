# MLA-C02 coverage register

This register tracks, skill by skill, what this local learning project actually covers for the AWS Certified Machine Learning Engineer – Associate (MLA-C02) exam. Each status rests on code, tests, or recorded experiments in this repository. Documentation alone doesn't count as implementation.

- **Source of skill text and numbering:** the official exam guide pages, retrieved 2026-11-08:
  - [Domain 1](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/machine-learning-engineer-associate-02-domain1.html)
  - [Domain 2](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/machine-learning-engineer-associate-02-domain2.html)
  
  Skill text is quoted verbatim, and numbers are the official ones; none are invented here.
- **AWS hands-on:** none. The project runs only locally, so every AWS service below is at most *conceptual*.
- **Last verified:** 2026-11-08. That day's checks:
  - `pytest`: 95 passed.
  - Phase 3 runs `20261108T104225Z-cdbbaf` (seed 42), `20261108T104330Z-cfbf6f` (seed 42, repeat) and `20261108T104409Z-4dbf37` (seed 7).
  - Diagnostics `20261108T104524Z-908690`, and test evaluation `evaluation-01` of the seed-42 run.

  All run outputs live under `data/processed/models/`, which git ignores.

## Status codes

| Code | Meaning |
|---|---|
| **I** | Implemented and tested locally |
| **P** | Partially implemented: some named techniques or services are implemented, others are not |
| **S** | Simulated locally: the mechanism is imitated, but the real setting (for example real traffic or time) is absent |
| **C** | Conceptual study only: explained in the guides, with no code |
| **M** | Deferred to a mini-project: M1 = local RAG + GenAI evaluation; M2 = small-FM LoRA fine-tuning |
| **—** | Not yet assessed |

Guide references: P1 = `docs/INTERNAL_GUIDE_ZH-TW.md`, P2 = `docs/PHASE2_GUIDE_ZH-TW.md`, P3 = `docs/PHASE3_GUIDE_ZH-TW.md` (chapter letters). The `docs/` folder is intentionally git-ignored (`.gitignore`), so guide references point to local files only.

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
