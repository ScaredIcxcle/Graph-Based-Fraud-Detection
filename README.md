# Graph-Based-Fraud-Detection
Course Project for CS328 Introduction to Data Science

To run-
Download all the .py files provided in the repo.
Download the Credit Card Transactions Fraud Detection Dataset from Kaggle https://www.kaggle.com/datasets/kartik2112/fraud-detection
Put the extracted CSVs from the dataset and the 3 .py files in the same folder together.
First run "python 01_build_graph_features.py" in the terminal to generate graph based parameters.
Then run "python 02_train_models.py" to train the models.
Result will be visible in "outputs/model_comparision_report.txt"

Description-
Fraud detection systems typically evaluate transactions using attributes such as transaction amount,
time, and merchant identity, treating each transaction as an independent event. This project investigates
whether representing transactions as a customer–merchant graph and extracting structural features from
it can improve detection of coordinated fraud. We build a bipartite graph from a public credit-card
fraud dataset, extract structural signals (PageRank, degree, Louvain community membership, commu-
nity density, and a community-level fraud-rate feature), and compare an XGBoost model trained on
conventional transaction features alone (Model A) against an XGBoost model trained on the same fea-
tures plus graph-derived features (Model B). Both models use identical hyperparameters and an identical
5-fold cross-validation procedure, and are evaluated on the same held-out test set, both overall and on
subsets of transactions exhibiting specific coordinated-fraud patterns. After resolving a series of method-
ological issues – most significantly a train/test graph-scale mismatch that initially caused catastrophic
model failure – the final graph-enhanced model achieves a held-out PR-AUC of 0.908 versus 0.903 for
the baseline, with the largest improvement on the fan-out pattern (0.851 vs. 0.700) and the chain pattern
(0.998 vs. 0.993). We report the design decisions, bugs, and corrections that shaped the final pipeline,
since arriving at a trustworthy graph feature set was itself a substantial part of this project’s contribution.
