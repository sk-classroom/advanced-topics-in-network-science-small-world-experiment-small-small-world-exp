# %% Load the network
import pandas as pd
import numpy as np
from scipy import sparse

file_path = "APS_author2DOI.dat"

results = []
with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        # Split the line by tab to get publication IDs
        publication_ids = line.strip().split('\t')
        # Skip lines with less than 5 publication IDs
        if len(publication_ids) < 5:
            continue
        # Create a list of dictionaries with author_id and doi
        results +=[{"author_id":i, "doi":p} for p in publication_ids]

# Create a DataFrame from the results list
df = pd.DataFrame(results)

# Get unique DOIs and create a DataFrame with paper IDs
doi_list = df["doi"].unique()
doi_table = pd.DataFrame({"doi":doi_list, "paper_id":np.arange(len(doi_list))})

# Merge the original DataFrame with the DOI table to get paper IDs
df = pd.merge(df, doi_table, on="doi")

# Get the number of authors and papers
n_authors = df["author_id"].max() + 1
n_papers = df["paper_id"].max() + 1

# Create a sparse matrix for author-paper relationships
author2paper = sparse.csr_matrix((np.ones(len(df)), (df["author_id"], df["paper_id"])), shape=(n_authors, n_papers))
# Create a sparse matrix for author-author relationships by multiplying the author-paper matrix with its transpose
author2author = author2paper @ author2paper.T

# Set all non-zero entries in the author-author matrix to 1
author2author.data = author2author.data * 0 + 1
min_deg = 1

# Copy the author-author matrix to A
A = author2author.copy()

# Iteratively remove nodes with degree less than min_deg
while True:
    # Calculate the degree of each node
    deg = np.array(A.sum(axis=1)).flatten()
    # Break the loop if all nodes have degree >= min_deg
    if np.all(deg >= min_deg):
        break
    # Keep only nodes with degree >= min_deg
    keep = deg >= min_deg
    A = A[keep, :]
    A = A[:, keep]

# Get the source and target nodes from the final matrix
src, trg, _ = sparse.find(sparse.triu(A))

# Save the edges to a CSV file
pd.DataFrame({"src":src, "trg":trg}).to_csv("edge_coauthorship.csv", index=False)
