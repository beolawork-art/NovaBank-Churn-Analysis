🏦 NovaBank Churn Analysis and Prediction

🌟 Project Overview

This project focuses on building a high-performance machine learning model to predict customer churn (attrition) for NovaBank. Given the severe imbalance in the customer data (less than 6% churn rate), the primary goal was to maximize the model's Recall for the minority churn class.

Final Result: An Optimized Random Forest Classifier achieved 73% Recall for the churn class, providing the bank with a highly actionable list of high-risk customers.

🎯 Key Findings & Business Impact

| Metric | Score | Impact |
| :--- | :---: | --- |
| Recall (Churn Class) | 0.73 (73%) | The model correctly identifies 73 out of every 100 customers who are about to leave. |
| Precision (Churn Class) | 0.42 (42%) | When the model flags a customer as high-risk, it is correct 42% of the time. (A necessary trade-off for high Recall). |

Top 3 Drivers of Churn (Feature Importance)

The model revealed that behavioral engagement is the overwhelming factor, not demographics.

1. transactions_per_month (0.292 Importance) 💥: The single strongest predictor.

2. account_balance (0.165 Importance) 💰: The second strongest predictor.

3. age (0.049 Importance) 🧑‍🦳: A distant third, confirming behavioral data is key.

🛠️ Methodology and Technical Pipeline

1. Preprocessing and Feature Engineering
- Data Cleaning: Handled missing values and ensured correct data types.

- Feature Binning: Created more meaningful categorical features from numerical ones (e.g., transactions_per_month was binned into Txn_Activity_Group) using pd.cut and pd.qcut.

- Encoding & Scaling: Features were One-Hot Encoded and numerical features were

 2. Standard Scaled using StandardScaler.

| Stage | Model Used | Technique | Purpose |
| :--- | :---: | :---: | --- |
| Baseline | Logistic Regression | Standard Train/Test Split | Established baseline performance (Recall 0.09). |
| Imbalance Fix | Random Forest | SMOTE (Synthetic Minority Oversampling) | Balanced the training data to force the model to learn the churn patterns. |
| Optimization | Random Forest | Grid Search (scoring='recall') | Fine-tuned the best model to maximize the identification of churners. |

💻 How to Run the Project
This project requires a Python environment with the following libraries installed.

Dependencies
Install all required packages using pip:

pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

Key Python Files

1. 01_data_cleaning_and_engineering.ipynb: Contains the code for feature creation, binning, and train/test split.

2. 02_model_training_and_optimization.ipynb: Contains the code for SMOTE, Grid Search, and training the final Random Forest Classifier.

3. 03_final_analysis_and_scoring.ipynb: Contains the code for generating Feature Importance and the final Churn Probability Score list.

Running the Analysis
Execute the notebooks in sequential order. The core logic for modeling is:

# 1. Balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# 2. Train the best model
best_rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
best_rf_model.fit(X_train_smote, y_train_smote)

# 3. Generate probabilities for all customers
churn_probabilities = best_rf_model.predict_proba(X_final)[:, 1]
