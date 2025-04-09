# IR Project – Query Expansion Techniques

This is a reproducibility package for the IR project. It contains all scripts, notebooks, and results used to run and evaluate different query expansion techniques on two datasets (TREC and NF).

---

## 📁 Project Structure

### 🔹 `query-expansion-reproducibility-package.ipynb`

Contains the first four parts of the project:

- **Part 1: Setup + Baseline**
  - Index each corpus.
  - Split datasets into training/test.
  - Run baseline retrieval experiments.

- **Part 2: LLM-based Query Expansion**
  - Apply LLMs to generate expanded queries.
  - Run experiments with LLM-expanded queries.

- **Part 3: Similarity-Based Expansion (Synonyms)**
  - Generate query expansions using word-level similarity.
  - Evaluate the retrieval performance.

- **Part 4: Ontology-Based Expansion**
  - Use domain ontology (e.g., SCTO.owl) to expand queries.
  - Run ontology-based query expansion experiments.

---

### 🔹 `ir-project-2.ipynb`

Contains:

- **Part 5: Word2Vec-Based Query Expansion**
  - Trains or uses pre-trained Word2Vec embeddings.
  - Expands queries using similar word vectors.

- **Part 6: Pseudo-Relevance Feedback (PRF) Expansion**
  - Uses top-ranked documents from baseline results.
  - Performs local feedback-based query reformulation.

---

### 🔹 `Results.xlsx`

An Excel sheet containing the **results of all six query expansion techniques**, for both DPH and BM25 retrieval models.

---

### 🔹 `vis.py` and `vis2.py`

Python scripts to generate visualizations of key metrics (`P@5`, `nDCG@10`, `MAP`, `MRR`, etc.) from the results stored in `Results.xlsx`.

---

## 📂 Additional Files

- `query_expansion.py`: Core functions or helper utilities for query expansion logic.
- `SCTO.owl`: Ontology used for query expansion in the ontology-based method.

---

## 📈 Evaluation Metrics

Each experiment measures:
- Precision at rank 5 and 10 (P@5, P@10)
- Normalized Discounted Cumulative Gain (nDCG@10)
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Recall@5, Recall@10

---

## 🔁 Reproducibility

To reproduce the experiments:
1. Run each part of the notebook in order.
2. Ensure all datasets and dependencies are correctly configured.
3. Use the provided visualizations and metrics in the Excel file to compare methods.

---

## 📊 Summary

This project evaluates the impact of six different query expansion techniques on information retrieval performance using traditional ranking models.
