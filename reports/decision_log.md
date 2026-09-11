# Architectural & Engineering Decision Log

This document records the foundational architectural, product, and machine learning decisions made during the design and evaluation of the Hiver AI Support Agent.

---

### Decision 1: Brand Selection — AppleSupport over AmazonHelp
- **Decision**: Selected `AppleSupport` (10,004 conversations in dataset slice) as the primary brand.
- **Why**: Apple Support interactions have concise, standardized technical troubleshooting steps across well-defined device/software domains with high historical resolution specificity.
- **Alternative considered**: `AmazonHelp` (11,280 conversations).
- **Why rejected**: Amazon conversations heavily involved multi-item retail logistics, carrier delivery delays, and third-party seller disputes which lack standardized self-service technical resolutions.

---

### Decision 2: Zero Autonomous Irreversible Financial/Account Actions
- **Decision**: The support agent drafts grounded replies and routes to human escalation, but is strictly prohibited from autonomously initiating refunds, cancellations, or password resets.
- **Why**: Autonomous financial or account actions introduce severe risk of unauthorized fund disbursement and account takeover.
- **Alternative considered**: Implementing automated refund webhook execution.
- **Why rejected**: Unacceptable financial liability and fraud vulnerability for an LLM-based autonomous agent.

---

### Decision 3: Macro-Averaged F1 as Primary Optimization Metric
- **Decision**: Prioritized Macro-F1 over Accuracy and Weighted-F1.
- **Why**: Real customer support data suffers from heavy class imbalance (e.g. software updates dominate general advice). Accuracy masks complete failure on minority safety-critical classes (e.g. security compromise).
- **Alternative considered**: Top-1 Classification Accuracy.
- **Why rejected**: A model predicting only the top 2 classes could achieve 70% accuracy while failing on 8 critical customer intents.

---

### Decision 4: Two-Stage Hybrid Retrieval (Vector Search + Cross-Encoder Reranker)
- **Decision**: Deployed a two-stage retrieval pipeline: retrieve Top-10 candidates from vector store, then rerank to Top-3 with a Cross-Encoder.
- **Why**: Vector similarity search has high recall but can match lexical surface similarities without resolution relevance. The Cross-Encoder computes joint token-level interaction scores.
- **Alternative considered**: Single-stage Top-3 dense vector retrieval.
- **Why rejected**: Vector retrieval alone frequently retrieved queries with similar emotional tone but completely irrelevant technical resolutions.

---

### Decision 5: Strict Intent Metadata Filtering with Low-Confidence Fallback
- **Decision**: Filter vector search by predicted intent metadata when classifier confidence $\ge 0.65$; if confidence is low or filter returns $<2$ results, fall back to unfiltered global semantic search.
- **Why**: Prevents cross-intent pollution (e.g. battery queries retrieving iCloud advice) while avoiding zero-result failures when intent classification is uncertain.
- **Alternative considered**: Unfiltered global vector search across all intents.
- **Why rejected**: Unfiltered search frequently returned resolutions from unrelated domains that happened to share generic phrases like *"Please DM us your device"*.

---

### Decision 6: Quarantined Golden Set with Zero-Leakage Verification
- **Decision**: 200 Golden Evaluation Set items were isolated prior to any model training, vector indexing, or clustering, verified programmatically with 0% overlap.
- **Why**: Contaminating the knowledge base or training set with evaluation queries produces artificially inflated, fraudulent benchmark metrics.
- **Alternative considered**: K-fold cross-validation on the entire scraped corpus.
- **Why rejected**: Cross-validation allows knowledge base retrieval leakage where the retriever directly finds the identical target thread.

---

### Decision 7: Conservative Escalation Bias over Aggressive Automation
- **Decision**: Biased the escalation engine towards human routing whenever intent confidence $<0.65$ or retrieval similarity $<0.45$.
- **Why**: In enterprise support, a False Auto-Handling error (giving incorrect advice on a broken/dangerous device) is significantly more damaging to brand trust than an unnecessary human escalation.
- **Alternative considered**: Aggressive auto-handling threshold ($\ge 0.40$).
- **Why rejected**: Resulted in hallucinations and dangerous advice for accounts undergoing unauthorized access.

---

### Decision 8: Multi-Signal Composite Confidence Formulation
- **Decision**: Formulated confidence as a linear combination of classifier probability (40%), retrieval similarity (30%), reranker relevance (20%), and evidence density (10%), minus risk penalties.
- **Why**: LLM self-reported confidence is notoriously uncalibrated and prone to overconfidence hallucinations.
- **Alternative considered**: Relying on LLM self-generated confidence score.
- **Why rejected**: LLMs regularly output 0.95+ confidence even when generating completely fabricated policy claims.

---

### Decision 9: PII Masking at the Ingestion Gateway
- **Decision**: Masked all email addresses, phone numbers, and order/case references with standardized tokens (`[EMAIL]`, `[PHONE]`, `[ORDER_REF]`) before embedding and LLM prompt construction.
- **Why**: Protects customer privacy, prevents data leakage in logs, and stops the LLM from hallucinating real customer phone numbers.
- **Alternative considered**: Relying on LLM system prompt instructions to ignore PII.
- **Why rejected**: LLM prompt filtering is non-deterministic and fails under prompt injection or high token load.

---

### Decision 10: Deterministic Offline Grounded Generator Fallback
- **Decision**: Built an offline deterministic grounded response generator and LLM judge fallback that runs locally when external API keys are not supplied.
- **Why**: Guarantees that evaluators can run unit tests, CI/CD, and the evaluation harness in $<15$ minutes without requiring external paid API keys.
- **Alternative considered**: Making OpenAI/Anthropic API keys mandatory for all tests.
- **Why rejected**: Would break local testing, automated grading, and offline reproducibility.

---

### Decision 11: Empirical Intent Discovery via Clustering over Manual Invention
- **Decision**: Discovered the 10 operational intents using TF-IDF feature extraction and KMeans clustering on customer query embeddings before finalizing taxonomy definitions.
- **Why**: Ensures intent classes reflect actual customer issues observed in the wild rather than idealized theoretical categories.
- **Alternative considered**: Adopting generic Banking77 taxonomy or arbitrary manual labels.
- **Why rejected**: Banking77 is irrelevant to tech hardware support; arbitrary manual taxonomy fails to cover real-world failure patterns like iOS OTA update stalls.

---

### Decision 12: Stratified Difficulty Tiers in Golden Set (Easy, Medium, Hard, High-Risk)
- **Decision**: Intentionally curated the 200 Golden Set items across 4 distinct difficulty tiers rather than uniformly sampling simple FAQs.
- **Why**: A realistic benchmark must test boundary cases, multi-symptom complaints, and security threats to accurately measure system resilience.
- **Alternative considered**: Uniform random sampling from raw tweets.
- **Why rejected**: Random sampling produces 80% trivial queries (e.g. *"thanks"*, *"hi"*), hiding critical system vulnerabilities.
