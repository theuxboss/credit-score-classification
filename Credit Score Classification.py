
# Commented out IPython magic to ensure Python compatibility.
# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %matplotlib inline


# Preprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler


# Models

import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Model Evaluation

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import RocCurveDisplay

# XAI Models

# explainability

import shap

# print the JS visualization code to the notebook

shap.initjs()

"""# Dataset

### Data Dictionary

* ID: Represents a unique identification of an entry
* Customer_ID: Represents a unique identification of a person
* Month: Represents the month of the year
* Name: Represents the name of a person
* Age: Represents the age of the person
* SSN: Represents the social security number of a person
* Occupation: Represents the occupation of the person
* Annual_Income: Represents the annual income of the person
* Monthly_Inhand_Salary: Represents the monthly base salary of a person
* Num_Bank_Accounts: Represents the number of bank accounts a person holds
* Num_Credit_Card: Represents the number of other credit cards held by a person
* Interest_Rate: Represents the interest rate on credit card
* Num_of_Loan: Represents the number of loans taken from the bank
* Type_of_Loan: Represents the types of loan taken by a person
* Delay_from_due_date: Represents the average number of days delayed from the payment date
* Num_of_Delayed_Payment: Represents the average number of payments delayed by a person
* Changed_Credit_Limit: Represents the percentage change in credit card limit
* Num_Credit_Inquiries: Represents the number of credit card inquiries
* Credit_Mix: Represents the classification of the mix of credits
* Outstanding_Debt: Represents the remaining debt to be paid (in USD)
* Credit_Utilization_Ratio: Represents the utilization ratio of credit card
* Credit_History_Age: Represents the age of credit history of the person
* Payment_of_Min_Amount: Represents whether only the minimum amount was paid by the person
* Total_EMI_per_month: Represents the monthly EMI payments (in USD)
* Amount_invested_monthly: Represents the monthly amount invested by the customer (in USD)
* Payment_Behaviour: Represents the payment behavior of the customer (in USD)
* Monthly_Balance: Represents the monthly balance amount of the customer (in USD)
* Credit_Score: The Outcome
"""

data_train = pd.read_csv("/content/Dissertation/train.csv")

data_train.shape

data_train.head()

"""## Information about Dataset"""

data_train["Credit_Score"].value_counts()

data_train["Month"].value_counts()

data_train["Occupation"].value_counts()

pd.crosstab(data_train["Occupation"], data_train["Credit_Score"])

data_train.info()

"""#### Converting all numeric data to int64 or Float"""

columns_int = ['Age', 'Num_of_Loan']

# data_train[columns_int] = data_train[columns_int].str.replace('_', '')

# Loop through each column and replace characters

for col in columns_int:
    data_train[col] = data_train[col].str.replace('_', '')

# Convert specified columns to integers
data_train[columns_int] = data_train[columns_int].astype(int)
data_train[columns_int] = data_train[columns_int].astype('int64')

columns_float = ['Changed_Credit_Limit', 'Outstanding_Debt',
                 'Amount_invested_monthly', 'Monthly_Balance', 'Annual_Income', 'Num_of_Delayed_Payment']

# Remove underscores from the specified columns
for col in columns_float:
    data_train[col] = data_train[col].str.replace('_', '')

# Replace empty strings with NaN
data_train[columns_float] = data_train[columns_float].replace('', np.nan)

# Convert specified columns to floats
data_train[columns_float] = data_train[columns_float].astype(float)

data_train.info()

data_train["Age"].value_counts()

pd.crosstab(data_train["Age"], data_train["Credit_Score"])

"""# Data Cleaning"""

data_train.isnull().sum()

"""* ID, SSN, Name, CustomerID will not contribute to the credit score, and are unique to each data point. Let's drop these columns
* Occupation, TypeofLoan need to be dropped owing to the number of missing values
* The rest of the features contribute to the credit score according to this domain. So, rather than dropping the feature, lets remove the datapoints that have null values
"""

data_train = data_train.drop(['ID', 'SSN', 'Name', 'Customer_ID', 'Type_of_Loan', 'Occupation', 'Month'], axis = 1)

data_train = data_train.dropna()

data_train.isnull().sum()

data_train.describe().T

"""# Data Preprocessing"""

data_train.info()

"""### Handling Categorical Data"""

Categorical_features = ['Credit_Mix','Credit_History_Age','Payment_of_Min_Amount','Payment_Behaviour','Credit_Score']

data_train[Categorical_features]

data_train["Credit_Mix"].value_counts()

data_train["Credit_Score"].value_counts()

data_train["Payment_of_Min_Amount"].value_counts()

data_train["Payment_Behaviour"].value_counts()

"""* CreditMix' : Lets use label encoding to this, as there is a inherent relationship between the values
* CreditHistoryAge' : Let's replace with the total number of months
* PaymentofMinAmount' : Let's use one hot encoding as there is no inherent relationship
* PaymentBehaviour' : Let's use one hot encoding as there is no inherent relationship
* CreditScore' : Lets use label encoding to this, as there is a inherent relationship between the values

### Label Encoding for the Credit Score
"""

le_Cs = LabelEncoder()

data_train['Credit_Score'] = le_Cs.fit_transform(data_train['Credit_Score'])

le_Cs.classes_

"""### One-Hot Encoding for 'PaymentofMinAmount'"""

# Instantiate OneHotEncoder without specifying the 'sparse' parameter

encoder = OneHotEncoder(drop='first', sparse_output=False)

# Reshape the input data

encoded_data = encoder.fit_transform(data_train['Payment_of_Min_Amount'].values.reshape(-1, 1))

encoded_data_train = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Payment_of_Min_Amount']))

data_train = data_train.reset_index(drop=True) # Resetting the index to avoid issues while concatenating

data_train= pd.concat([data_train, encoded_data_train], axis=1)

"""### 'CreditHistoryAge' : Let's replace with the total number of months"""

data_copy = data_train

def getMonths(duration):
    years, months = int(duration.split()[0]), int(duration.split()[3])
    months += years * 12
    return months

data_copy['Credit_History_Age'] = data_copy['Credit_History_Age'].apply(getMonths)

"""#### Credit_Mix : Lets use label encoding to this, as there is a inherent relationship between the values"""

le_Cm = LabelEncoder()
data_copy['Credit_Mix'] = le_Cm.fit_transform(data_copy['Credit_Mix'])

le_Cm.classes_

"""#### Payment_Behaviour : Let's use one hot encoding as there is no inherent relationship"""

encoder = OneHotEncoder(sparse_output=False, drop='first')

encoded_data = encoder.fit_transform(data_copy['Payment_Behaviour'].values.reshape(-1, 1))

encoded_data_train = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Payment_Behaviour']))

data_train = data_copy.reset_index(drop=True) # Resetting the index to avoid issues while concatenating

data_train= pd.concat([data_train, encoded_data_train], axis=1)

data_train.head()

data_train["Credit_Mix"].value_counts()

data_train["Credit_Score"].value_counts()

data_train.info()

"""#### Dropping the left over categorical values post one hot encoding"""

data_train = data_train.drop(['Payment_of_Min_Amount', 'Payment_Behaviour'], axis = 1)

data_train.info()

"""## Data Balancing"""

arr_0, arr_1, arr_2 = [], [], []
for i in data_train['Credit_Score']:
    if i == 0:
        arr_0.append(i)
    if i == 1:
        arr_1.append(i)
    else:
        arr_2.append(i)

Class_1, Class_0, Class_2 = len(arr_1), len(arr_0), len(arr_2)


plt.pie([Class_1, Class_0, Class_2], labels=['Class 1', 'Class 0', 'Class 2'], autopct='%1.1f%%', startangle=140)
plt.title('Target Variable Before Over sampling')
plt.show()

"""#### Dataset is imbalanced. Let's use oversampling to increase the number of datapoints with the classes 0 and 1"""

# Class_2 represents the size of the majority class

Class_2 = len(data_train[data_train['Credit_Score'] == 2])

# Splitting the dataset based on 'Credit_Score'

data_class_1 = data_train[data_train['Credit_Score'] == 1]
data_class_2 = data_train[data_train['Credit_Score'] == 2]
data_class_0 = data_train[data_train['Credit_Score'] == 0]

# Oversampling the minority classes

data_class_1_oversampled = data_class_1.sample(n=Class_2, replace=True)
data_class_0_oversampled = data_class_0.sample(n=Class_2, replace=True)

# Concatenating the oversampled data with the majority class

data_oversampled = pd.concat([data_class_2, data_class_1_oversampled, data_class_0_oversampled], axis=0)

# Optional: Shuffle the oversampled data

data_oversampled = data_oversampled.sample(frac=1).reset_index(drop=True)

arr_0, arr_1, arr_2 = [], [], []
for i in data_oversampled['Credit_Score']:
    if i == 0:
        arr_0.append(i)
    if i == 1:
        arr_1.append(i)
    else:
        arr_2.append(i)

Class_1, Class_0, Class_2 = len(arr_1), len(arr_0), len(arr_2)


plt.pie([Class_1, Class_0, Class_2], labels=['Class 1', 'Class 0', 'Class 2'], autopct='%1.1f%%', startangle=140)
plt.title('Target Variable After over sampling')
plt.show()

"""## Feature Analysis"""

plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

# Use Seaborn to create boxplots for all columns in df

sns.boxplot(data=data_oversampled, orient='h')  # orient='h' for horizontal boxplots

plt.title('Boxplots for Multiple Columns')

"""##### There is an outlier in the MonthlyBalance that is effecting this plot. Before we handle it, lets check the skewness of the feature distributions"""

data_oversampled.info()

"""### Outlier Removal"""

df = data_oversampled.drop('Credit_Score', axis = 1)

num_cols = 3

num_features = len(df.columns)
num_rows = (num_features - 1) // num_cols + 1

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

# Flattenning the axes

axes = axes.ravel()

i = -1
for feature in df.columns:
    i += 1
    ax = axes[i]
    sns.boxplot(x=df[feature], ax = ax)
    ax.set_title(f'Outliers of {feature}')
    ax.set_xlabel('')
    ax.set_ylabel('')

for i in range(num_features, num_cols * num_rows):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

"""#####  Adding back the target column after plotting"""

df['Credit_Score'] = data_oversampled['Credit_Score']

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# print(IQR)

print(df < (Q1 - 1.5 * IQR) |(df > (Q3 + 1.5 * IQR)))

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

print(df < (Q1 - 1.5 * IQR) |(df > (Q3 + 1.5 * IQR)))

s = set([
    'Payment_of_Min_Amount_No',
    'Payment_of_Min_Amount_Yes',
    'Payment_Behaviour_High_spent_Large_value_payments',
    'Payment_Behaviour_High_spent_Medium_value_payments',
    'Payment_Behaviour_High_spent_Small_value_payments',
    'Payment_Behaviour_Low_spent_Large_value_payments',
    'Payment_Behaviour_Low_spent_Medium_value_payments',
    'Payment_Behaviour_Low_spent_Small_value_payments',
    'Age',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Num_of_Delayed_Payment',
    'Num_Credit_Inquiries'
])

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
for column in df.columns:
    if column in s:
        continue
    df = df[(df[column] >= lower_bound[column]) & (df[column] <= upper_bound[column])]

print(df)

num_cols = 3

num_features = len(df.columns)
num_rows = (num_features - 1) // num_cols + 1

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

# Flattenning the axes

axes = axes.ravel()

i = -1

for feature in df.columns:

    i += 1
    ax = axes[i]
    sns.boxplot(x=df[feature], ax = ax)
    ax.set_title(f'Boxplot of {feature}')
    ax.set_xlabel('')
    ax.set_ylabel('')

for i in range(num_features, num_cols * num_rows):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

df.reset_index(drop=True)

plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

# Use Seaborn to create boxplots for all columns in df

sns.boxplot(data=df, orient='h')  # orient='h' for horizontal boxplots
plt.title('Boxplots for Multiple Columns')

"""## Feature Scaling

* Scaling post outlier removal
"""

df_without_outliers  = df.copy()

sc = StandardScaler()
sc.fit(df_without_outliers.drop('Credit_Score', axis = 1))
data_oversampled_normalized_without_outliers = sc.transform(df_without_outliers.drop('Credit_Score', axis = 1))
data_oversampled_normalized_without_outliers = pd.DataFrame(data_oversampled_normalized_without_outliers, columns = df_without_outliers.drop('Credit_Score', axis = 1).columns)

plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

# Use Seaborn to create boxplots for all columns in df

sns.boxplot(data=data_oversampled_normalized_without_outliers, orient='h')  # orient='h' for horizontal boxplots
plt.title('Boxplots for Multiple Columns')

data_oversampled_normalized_without_outliers = pd.DataFrame(data_oversampled_normalized_without_outliers,
                                                            columns=df_without_outliers.columns.difference(['Credit_Score']))


data_oversampled_normalized_without_outliers['Credit_Score'] = df_without_outliers.reset_index()['Credit_Score']

data_oversampled_normalized_without_outliers

df2 = data_oversampled_normalized_without_outliers.copy()

df2

"""# Model Building

## 1. Decision Trees
"""

X = df2.drop('Credit_Score', axis=1)
y = df2['Credit_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# setup random seed

np.random.seed(42)

decision_tree_model = DecisionTreeClassifier(random_state=42)

decision_tree_model.fit(X_train, y_train)

y_pred_decision = decision_tree_model.predict(X_test)

decision_tree_model.score(X_test, y_test)

accuracy_decision = accuracy_score(y_test, y_pred_decision)

print(f'Accuracy: {accuracy_decision:.2f}')
print(classification_report(y_test, y_pred_decision))

de_confusion = confusion_matrix(y_test, y_pred_decision)

plt.figure(figsize=(8, 6))
sns.heatmap(de_confusion, annot=True, fmt='d', cmap='Blues', cbar=False)

plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix for Decision Tree Classifier Model')

plt.show()

"""## XGBoost"""

# setup random seed

np.random.seed(42)

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print(f'Accuracy: {accuracy_xgb:.2f}')
print(classification_report(y_test, y_pred_xgb))

xgb_confusion = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(8, 6))
sns.heatmap(xgb_confusion, annot=True, fmt='d', cmap='Blues', cbar=False)

plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix for XGBoost Classifier Model')
plt.show()

"""## 3. Random Forest"""

# setup random seed

np.random.seed(42)

# Instantiate Random Forest Classifier

rdf_model = RandomForestClassifier(n_estimators=100)

# Fit the model to the data

rdf_model.fit(X_train, y_train)

y_preds_rdf = rdf_model.predict(X_test)

accuracy_rdf = accuracy_score(y_test, y_preds_rdf)

print(f'Accuracy: {accuracy_rdf:.2f}')
print(classification_report(y_test, y_preds_rdf))

rdf_confusion = confusion_matrix(y_test, y_preds_rdf)

plt.figure(figsize=(8, 6))
sns.heatmap(rdf_confusion, annot=True, fmt='d', cmap='Blues', cbar=False)

plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix for Random Forest Classifier Model')

plt.show()

"""## Baseline model comparison"""

model_scores = {"Decision Trees": 0.87,
                 "XGBoot": 0.84,
                 "Random Forest": 0.90}

model_compare = pd.DataFrame(model_scores, index=["Accuracy"])

model_compare.T.plot.bar()
plt.xticks(rotation = 0);

"""# Hyperparameter Tuning with GridSearchCV

we're going to tune:

* Decision Tree
* XGBoost
* Random Forest Classifier

...GridSearchCV
"""

# Hyperparameter grid for Random Forest

rf_grid = {
    "n_estimators": np.arange(10, 100, 50),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2)
}

# Hyperparameter grid for Decision Tree (similar to Random Forest without n_estimators)

dt_grid = {
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2)
}

# Hyperparameter grid for XGBoost (a popular gradient boosting library)

xgb_grid = {
    "n_estimators": np.arange(10, 100, 50),
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0]
}

"""
# Printing the hyperparameter grids
print("Random Forest Grid:")
print(rf_grid)
print("\nDecision Tree Grid:")
print(dt_grid)
print("\nXGBoost Grid:")
print(xgb_grid)

"""

"""## Grid Search on Decision Tree"""

# Setup grid hyperparameter search for DT

de_rscv = RandomizedSearchCV(DecisionTreeClassifier(random_state=42),
                         param_distributions=dt_grid,
                         cv=5, n_iter=100, verbose=True)


# Fit random hyperparameter search model for Decision Tree

de_rscv.fit(X_train, y_train)

de_rscv.best_params_

y_preds_de_rscv = de_rscv.predict(X_test)

accuracy_de_rscv = accuracy_score(y_test, y_preds_de_rscv)

print(f'Accuracy: {accuracy_de_rscv:.2f}')
print(classification_report(y_test, y_preds_de_rscv))

"""## Grid Search on XGBoost"""

# Setup grid hyperparameter search for XGBoost

xgb_rscv = RandomizedSearchCV(xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42),
                         param_distributions=xgb_grid,
                         cv=5, n_iter=50, verbose=True)


# Fit random hyperparameter search model for XGBoost

xgb_rscv.fit(X_train, y_train)

xgb_rscv.best_params_

y_preds_xgb_rscv = xgb_rscv.predict(X_test)

accuracy_xgb_rscv = accuracy_score(y_test, y_preds_xgb_rscv)

print(f'Accuracy: {accuracy_xgb_rscv:.2f}')
print(classification_report(y_test, y_preds_xgb_rscv))

"""## Grid Search on Random Forest"""

# setup random seed

np.random.seed(42)

# Setup grid hyperparameter search for Random Forest

rf_rscv = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                         param_distributions=rf_grid,
                         cv=5, n_iter=100, verbose=True)


# Fit random hyperparameter search model for Random Forest

rf_rscv.fit(X_train, y_train)

rf_rscv.best_params_

y_preds_rf_rscv = rf_rscv.predict(X_test)

accuracy_rf_rscv = accuracy_score(y_test, y_preds_rf_rscv)

print(f'Accuracy: {accuracy_rf_rscv:.2f}')
print(classification_report(y_test, y_preds_rf_rscv))

"""## GridSearchCV model comparison"""

cv_model_scores = {"Decision Trees": 0.87,
                 "XGBoot": 0.83,
                 "Random Forest": 0.89}

cv_model_compare = pd.DataFrame(cv_model_scores, index=["Accuracy"])

cv_model_compare.T.plot.bar()
plt.xticks(rotation = 0);

"""# Explainable AI Techniques

* I will need to implement local and global interpretability
* Local using SHAP: To identify features that affect loan approval
* Global using GRIP: To explain the importance of each features outcomes on the average for the entire dataset

## 1. SHAP on Random Forest Baseline Model

* Best performer, with very good accuracy, precision and fi-score
"""

explainer = shap.TreeExplainer(rdf_model)

shap_values = explainer.shap_values(X_test)

# Create an explainer object for the RandomForestClassifier model

# explainer = shap.Explainer(rdf_model)

# Compute SHAP values

# shap_values = explainer(X)

# Visualize the SHAP values (for example, using a bar plot)

# shap.plots.bar(shap_values)

# summarize the effects of all the features

# shap.plots.beeswarm(shap_values)

# visualize the first prediction's explanation

# shap.plots.waterfall(shap_values[0])
