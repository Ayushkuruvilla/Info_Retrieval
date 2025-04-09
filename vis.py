import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the Excel sheets
nf_df = pd.read_excel("IR_New_Index_Results.xlsx", sheet_name="overall_nf_output")
trec_df = pd.read_excel("IR_New_Index_Results.xlsx", sheet_name="overall_trec_output")

# Add dataset labels
nf_df["Dataset"] = "NF"
trec_df["Dataset"] = "TREC"

# Normalize and extract Model and Query Technique
for df in [nf_df, trec_df]:
    df["name"] = df["name"].astype(str)
    df["Model"] = df["name"].str.extract(r"(DPH|BM25)", flags=re.IGNORECASE)[0].str.upper()
    df["Query Technique"] = df["name"].str.replace("DPH", "", case=False)
    df["Query Technique"] = df["Query Technique"].str.replace("BM25", "", case=False).str.strip().str.title()

# Combine both DataFrames
combined_df = pd.concat([nf_df, trec_df], ignore_index=True)

# Rename columns for consistency
combined_df.rename(columns={
    "P.5": "P@5",
    "P.10": "P@10",
    "ndcg_cut.10": "nDCG@10",
    "map": "MAP",
    "recip_rank": "MRR",
    "recall_5": "Recall@5",
    "recall_10": "Recall@10"
}, inplace=True)

# Define all metrics to plot
metrics = ["nDCG@10", "MAP", "MRR", "Recall@10"]
datasets = ["TREC", "NF"]
models = ["DPH", "BM25"]

# Plotting for each dataset
for dataset in datasets:
    plt.figure(figsize=(14, 7))
    for model in models:
        data_subset = combined_df[(combined_df["Model"] == model) & (combined_df["Dataset"] == dataset)]
        for metric in metrics:
            plt.plot(
                data_subset["Query Technique"],
                data_subset[metric],
                marker='o',
                label=f"{model} - {metric}"
            )

    plt.title(f"All Metrics Comparison by Query Expansion Method and Model on {dataset}")
    plt.xlabel("Query Expansion Method")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
