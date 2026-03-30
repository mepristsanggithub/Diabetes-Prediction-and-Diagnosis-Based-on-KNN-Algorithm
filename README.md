# Diabetes Prediction and Diagnosis Based on KNN Algorithm

This project implements and evaluates K-Nearest Neighbor (KNN) algorithms for diabetes prediction using the Pima Indians Diabetes dataset.

## Files

| File | Description |
|------|-------------|
| `Dataset.txt` | Diabetes dataset (2000 samples, 8 features + 1 label) |
| `knearestneighbor.m` | Standard KNN — iterates K from 1 to 50 using 5-fold cross-validation to find the optimal K |
| `Mutual.m` | Mutual Weighted KNN — uses the same K-selection process with a dual-weight voting scheme |
| `k15.m` | Model evaluation at K=15 — reports accuracy, recall, specificity, F1-score, and AUC-ROC |

## Workflow

1. **PCA dimensionality reduction** — standardize features, compute the correlation matrix, extract principal components. Based on the cumulative contribution rates output, **5 principal components** were selected as they explain sufficient variance of the dataset.
2. **K selection** — run `knearestneighbor.m` or `Mutual.m` to sweep K from 1 to 50 and identify the optimal value (K=15)
3. **Model evaluation** — run `k15.m` with K=15 to evaluate full performance metrics via 5-fold cross-validation

> When prompted `Enter the number of principal components to keep`, enter `5`. This value was manually selected by inspecting the cumulative contribution rates printed by the PCA step — the first 5 principal components were found to explain sufficient variance of the dataset.

## Algorithms

### Standard KNN (`knearestneighbor.m`)
Computes Euclidean distance from each test sample to all training samples, selects the K nearest neighbors, and assigns the majority label.

### Mutual Weighted KNN (`Mutual.m` / `k15.m`)
Extends standard KNN with two-layer weighted voting:
- **Weight 1** — based on distance from test sample to each neighbor
- **Weight 2** — based on the local density of each neighbor among other training samples

## Adapting to Other Datasets

All three scripts hardcode the dataset structure in two places (line 3 and line 52):
```matlab
x = y(:,1:8);            % assumes 8 feature columns
iris(:,m1+1) = y(:,9);  % assumes label is in column 9
```
To use a different dataset, update these two lines to match your data's feature count and label column. Everything else — sample size, number of principal components, and K selection — is handled dynamically.

