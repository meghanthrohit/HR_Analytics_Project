import pandas as pd

# Load the dataset
df = pd.read_csv("IBM_HR_Analytics.csv")

# Show the first 5 rows
print("🔹 First 5 Rows:")


# Shape of the dataset
print("🔹 Dataset Shape (rows, columns):")
print(df.shape)

# Column names
print("🔹 Column Names:")
print(df.columns.tolist())

# Data types of each column
print("🔹 Data Types:")
print(df.dtypes)

# Summary statistics
print("🔹 Summary Statistics:")


# Check missing values
print("🔹 Missing Values:")
print(df.isnull().sum())

# Check target variable balance
print("🔹 Attrition Value Counts:")
print(df['Attrition'].value_counts())


## 3 Check missing values
# Assuming df is already loaded
print("🔍 Missing Values in Each Column:")
print(df.isnull().sum())

#Remove columns that don’t help the prediction
cols_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print("Dropped columns:", cols_to_drop)

#Fix categorical column formats
object_cols = df.select_dtypes(include='object').columns
df[object_cols] = df[object_cols].astype(str)

print("Categorical columns:", list(object_cols))

#Ensure numeric columns are actually numeric
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')


# Shape of the new dataset
print("🔹 Dataset Shape (rows, columns):")
print(df.shape)


## 4 Exploratory Data Analysis (EDA)
#Basic Understanding of Attrition
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Attrition')
plt.title("Attrition Count")
plt.show()


#Attrition by Gender
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Gender', hue='Attrition')
plt.title("Attrition by Gender")
plt.show()


#Attrition by Department
plt.figure(figsize=(8,4))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title("Attrition by Department")
plt.xticks(rotation=45)
plt.show()

#Attrition by Job Role
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='JobRole', hue='Attrition')
plt.title("Attrition by Job Role")
plt.xticks(rotation=45)
plt.show()

#Attrition by Age Distribution
plt.figure(figsize=(8,4))
sns.histplot(data=df, x='Age', hue='Attrition', kde=True)
plt.title("Age Distribution by Attrition")
plt.show()


#Attrition vs Monthly Income
plt.figure(figsize=(8,4))
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome')
plt.title("Monthly Income vs Attrition")
plt.show()


#Correlation Heatmap (Numeric Columns Only)
numeric_df = df.select_dtypes(include=['int64','float64'])
plt.figure(figsize=(14,10))
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

## 5 Feature Engineering - This step prepares the dataset for ML modeling
#Encode the Target (Attrition)
# Convert Attrition back to Yes/No if needed
df['Attrition'] = df['Attrition'].astype(str)

# Convert to binary labels
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print(df['Attrition'].unique())


#Identify categorical & numeric columns
categorical_cols = df.select_dtypes(include='object').columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("Categorical Columns:", list(categorical_cols))
print("Numeric Columns:", list(numeric_cols))

#One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Feature scaling (Standardization)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols = numeric_cols.drop('Attrition', errors='ignore')
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])


#Split dataset into train/test
from sklearn.model_selection import train_test_split

X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Set Shape:", X_train.shape)
print("Test Set Shape:", X_test.shape)

## 6: Model Building & Evaluation for the HR Attrition Prediction Project.
#We will build and evaluate:
#✔️ Logistic Regression
#✔️ Random Forest
#✔️ XGBoost
#️✔️ Decision Tree
#✔️ Compare all models

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_log))
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

print("Decision Tree Results:")
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_dt))

#XGBoost (optional but best model usually)
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost Results:")
print(classification_report(y_test, y_pred_xgb))
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))


#Compare All Models
results = {
    "Model": ["Logistic Regression", "Random Forest", "Decision Tree", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_xgb)
    ],
    "ROC_AUC": [
        roc_auc_score(y_test, y_prob_log),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_dt),
        roc_auc_score(y_test, y_prob_xgb)
    ]
}

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)
