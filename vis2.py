import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Load the Excel file with multiple sheets
file_path = "C:\\Users\\91948\\Documents\\Q3\\Info_Retrieval\\Info_Retrieval\\IR_New_Index_Results.xlsx"
nf_df = pd.read_excel(file_path, sheet_name="overall_nf_output")
trec_df = pd.read_excel(file_path, sheet_name="overall_trec_output")

# Add Dataset labels
nf_df["Dataset"] = "NF"
trec_df["Dataset"] = "TREC"

# Normalize technique names and extract model
for df in [nf_df, trec_df]:
    df["name"] = df["name"].astype(str)  # Ensure all entries are strings
    df["Technique"] = df["name"].str.replace("DPH", "", case=False).str.replace("BM25", "", case=False)
    df["Technique"] = df["Technique"].str.strip().str.title()
    df["Model"] = df["name"].str.extract(r"(DPH|BM25)", flags=re.IGNORECASE)[0].str.upper()

# Combine datasets
combined_df = pd.concat([nf_df, trec_df], ignore_index=True)

# Drop rows with missing Technique or Model
combined_df.dropna(subset=["Technique", "Model"], inplace=True)

# Rename metric columns if needed
combined_df.rename(columns={"P.5": "P@5", "P.10": "P@10"}, inplace=True)

# Define consistent technique order
techniques = sorted(combined_df["Technique"].dropna().unique())
x = np.arange(len(techniques))
width = 0.18

# Define custom colors for each (Dataset, Model) pair
custom_colors = {
    ("NF", "BM25"): "#FFA726",
    ("NF", "DPH"): "#EF5350",
    ("TREC", "BM25"): "#42A5F5",
    ("TREC", "DPH"): "#AB47BC"
}

# Group by Dataset and Model
grouped = combined_df.groupby(["Dataset", "Model"])

# Plotting
plt.figure(figsize=(14, 6))

for i, ((dataset, model), group) in enumerate(grouped):
    # Align each group's techniques with full list for consistent bar positions
    values = group.set_index("Technique").reindex(techniques)["P@5"]
    color = custom_colors.get((dataset, model), "#888888")
    plt.bar(x + i * width, values, width, label=f"{dataset}-{model}", color=color)

# Final touches
plt.xticks(x + width * 1.5, techniques, rotation=45, ha='right')
plt.ylabel("P@5")
plt.title("P@5 Comparison by Query Technique")
plt.legend()
plt.tight_layout()
plt.show()
