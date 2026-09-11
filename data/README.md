# Data Directory Structure

This directory houses the raw, processed, and sampled evaluation datasets for the Hiver AI Support Agent.

## Structure:
- `raw/`: Raw downloaded datasets (e.g. `twcs.csv` / `train.parquet`). Not tracked in git.
- `processed/`: Reconstructed conversations, cleansed interactions, and feature tables.
  - `conversations.parquet`: Full multi-turn reconstructed conversations for the selected brand.
  - `support_pairs.parquet`: Flattened `(customer_message, brand_response)` knowledge base pairs.
- `sample/`: Self-contained, compact evaluation & demonstration artifact (~5MB) allowing full 100% reproduction of all evaluation benchmarks in under 15 minutes without needing raw 3M tweet processing.
