
■ Customer Churn Prediction — EDA & Machine
Learning
■ Project Overview
Customer churn (attrition) refers to when a customer stops using a company’s service. Reducing
churn is critical for subscription-based businesses like telecom, banking, and SaaS. This project
builds an end-to-end churn prediction model — from data exploration to predictive modeling — to
identify customers most likely to churn and help businesses take preventive actions.

■ Objectives
• Perform Exploratory Data Analysis (EDA) to uncover churn trends.
• Preprocess the dataset (cleaning, encoding, scaling).
• Build and compare classification models for churn prediction.
• Evaluate model performance using standard metrics.
• Derive business insights and potential interventions.

■ Dataset
The dataset contains customer-level information such as demographics, account info, services, and
churn status. If sourced from Kaggle, refer to the Telco Customer Churn Dataset:
https://www.kaggle.com/blastchar/telco-customer-churn

■ Exploratory Data Analysis (EDA)
EDA visualizes and interprets churn behavior using Pandas, NumPy, Matplotlib, and Seaborn. It
includes distribution analysis, feature correlation, service usage patterns, and churn probability.
■■ Data Preprocessing
Handled missing values, encoded categorical features, standardized numerical features, and split
data into training and testing sets.
■ Model Building
Implemented Logistic Regression, Random Forest, Decision Tree, KNN, and SVM. Evaluation
metrics include Accuracy, Precision, Recall, F1-Score, and ROC-AUC Curve.
■ Results & Insights
Random Forest performed best (~85–90% accuracy). Key predictors include tenure, contract type,
and monthly charges. Month-to-month contracts and high charges show higher churn likelihood

■ Tech Stack
• Programming: Python
• Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
• IDE: Jupyter Notebook
• Version Control: Git, GitHub
■■■■ How to Run Locally
1. Clone the repository: `git clone
https://github.com/yourusername/churning_customers_EDA_churn_prediction.git` 2. Navigate to
the folder: `cd churning_customers_EDA_churn_prediction` 3. Install dependencies: `pip install -r
requirements.txt` 4. Run the notebook: `jupyter notebook
churning_customers_EDA_churn_prediction.ipynb`


■ Future Work
• Use XGBoost or LightGBM for improved accuracy.
• Implement hyperparameter tuning using GridSearchCV or Optuna.
• Deploy the model as a Streamlit or Flask app.
• Add real-time churn prediction dashboard.



■■■ Author
Aadarsh Kushwaha
■ LinkedIn Profile
