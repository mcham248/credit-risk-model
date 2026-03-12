import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train = pd.read_csv('cs-training.csv').drop('Unnamed: 0', axis=1)
test = pd.read_csv('cs-test.csv').drop('Unnamed: 0', axis=1)

# Display basic information about the dataset
# print(train.duplicated().sum())
# print(train.shape)


train_redup = train.drop_duplicates()

# print(train_redup.duplicated().sum())
# print(train_redup.shape)
# print(train_redup.isnull().sum())

print(train_redup['SeriousDlqin2yrs'].value_counts())
print(train_redup['SeriousDlqin2yrs'].value_counts(normalize=True))


# Handle missing values by filling them with the median of the respective columns
train_redup['MonthlyIncome'] = train_redup['MonthlyIncome'].fillna(
    train_redup['MonthlyIncome'].median()
)

train_redup['NumberOfDependents'] = train_redup['NumberOfDependents'].fillna(
    train_redup['NumberOfDependents'].median()
)

print(train_redup.isnull().sum())

# winsorise variables to handle outliers
train_redup['RevolvingUtilizationOfUnsecuredLines'] = \
train_redup['RevolvingUtilizationOfUnsecuredLines'].clip(upper=train_redup['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99))

train_redup['DebtRatio'] = train_redup['DebtRatio'].clip(
    upper=train_redup['DebtRatio'].quantile(0.99)
)

train_redup['MonthlyIncome'] = train_redup['MonthlyIncome'].clip(
    lower=train_redup['MonthlyIncome'].quantile(0.01),
    upper=train_redup['MonthlyIncome'].quantile(0.99)
)

# exploratory data analysis
print(train_redup.describe())

sns.histplot(train_redup['age'], bins=50)
plt.title("Age Distribution")
plt.show()

print(train_redup['DebtRatio'].describe())
sns.histplot(train_redup['DebtRatio'], bins=50)
plt.title("Debt Ratio Distribution")
plt.show()

# realistic debt ratio values are typically less than 5, so we can focus on that range for better insights
sns.histplot(train_redup[train_redup['DebtRatio'] < 5]['DebtRatio'], bins=50)
plt.title("Debt Ratio Distribution (<5)")
plt.show()

sns.violinplot(
    x='SeriousDlqin2yrs',
    y='DebtRatio',
    data=train_redup[train_redup['DebtRatio'] < 5]
)

plt.title("Debt Ratio Distribution by Default")
plt.show()

# analyzing the relationship between late payments and default rates
late_90 = train_redup.groupby('NumberOfTimes90DaysLate')['SeriousDlqin2yrs'].mean()

plt.figure(figsize=(8,5))
late_90.plot(marker='o')
plt.title("Default Rate vs 90+ Days Late")
plt.xlabel("Number of 90-Day Late Payments")
plt.ylabel("Default Rate")
plt.show()

late_30 = train_redup.groupby('NumberOfTime30-59DaysPastDueNotWorse')['SeriousDlqin2yrs'].mean()

plt.figure(figsize=(8,5))
late_30.plot(marker='o')
plt.title("Default Rate vs 30-59 Days Late")
plt.xlabel("Number of Late Payments")
plt.ylabel("Default Rate")
plt.show()

late_90 = train_redup.groupby('NumberOfTimes90DaysLate')['SeriousDlqin2yrs'].mean()

late_90[late_90.index <= 10].plot(marker='o')
plt.title("Default Rate vs 90+ Days Late (0–10)")
plt.show()


# corrolation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(train_redup.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")   
plt.show()

# risk curve plots
plt.figure(figsize=(10, 6))
sns.kdeplot(train_redup[train_redup['SeriousDlqin2yrs'] == 0]['age'], label='No Default', shade=True)
sns.kdeplot(train_redup[train_redup['SeriousDlqin2yrs'] == 1]['age'], label='Default', shade=True)
plt.title("Age Distribution by Default Status") 
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(train_redup[train_redup['SeriousDlqin2yrs'] == 0]['DebtRatio'], label='No Default', shade=True)
sns.kdeplot(train_redup[train_redup['SeriousDlqin2yrs'] == 1]['DebtRatio'], label='Default', shade=True)
plt.title("Debt Ratio Distribution by Default Status")
plt.xlabel("Debt Ratio")
plt.ylabel("Density")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(train_redup[train_redup['SeriousDlqin2yrs'] == 0]['RevolvingUtilizationOfUnsecuredLines'], label='No Default', shade=True)
sns.kdeplot(train_redup[train_redup['SeriousDlqin2yrs'] == 1]['RevolvingUtilizationOfUnsecuredLines'], label='Default', shade=True)
plt.title("Revolving Utilization Distribution by Default Status")
plt.xlabel("Revolving Utilization")
plt.ylabel("Density")
plt.legend()
plt.show()

# prepare the data for modeling
from sklearn.model_selection import train_test_split   
X = train_redup.drop('SeriousDlqin2yrs', axis=1)
y = train_redup['SeriousDlqin2yrs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# train a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
y_pred_proba = model.predict_proba(X_test)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='Logistic Regression (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_proba)))
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# train a random forest model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# evaluate the random forest model
y_rf_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_rf_pred))
print(confusion_matrix(y_test, y_rf_pred))
y_rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_rf_pred_proba))
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_rf_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.2f})'.format(roc_auc_score(y_test, y_rf_pred_proba)))
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# train XGBoost model
from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# evaluate the XGBoost model
y_xgb_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_xgb_pred))
print(confusion_matrix(y_test, y_xgb_pred))
y_xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_xgb_pred_proba))
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_xgb_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = {:.2f})'.format(roc_auc_score(y_test, y_xgb_pred_proba)))
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# feature importance
importance = xgb_model.feature_importances_
importance_df = pd.Series(importance, index=X.columns)

importance_df.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title("Feature Importance from XGBoost")
plt.ylabel("Importance Score") 
plt.show()


# decile risk plot
risk_df = pd.DataFrame({'pd': y_xgb_pred_proba, 'actual_default': y_test.values})
risk_df['decile'] = pd.qcut(risk_df['pd'], 10, labels=False)
decile_summary = risk_df.groupby('decile').agg(borrowers=('actual_default','count'),
    defaults=('actual_default','sum'),
    default_rate=('actual_default','mean'))

print(decile_summary)

decile_summary['default_rate'].plot(kind='bar', figsize=(10, 6))
plt.title("Default Rate by Decile")
plt.xlabel("Decile")
plt.ylabel("Default Rate")
plt.show()


policy_df = pd.DataFrame({
    'pd': y_xgb_pred_proba,
    'actual_default': y_test.values
})

policy_df['decision'] = np.where(policy_df['pd'] > 0.5, 'Deny', 'Approve')

policy_summary = policy_df.groupby('decision').agg(
    borrowers=('pd', 'count'),
    avg_pd=('pd', 'mean'),
    actual_default_rate=('actual_default', 'mean')
)

print(policy_summary)

approval_rate = (policy_df['decision'] == 'Approve').mean()
print("Approval Rate:", approval_rate)

policy_summary['actual_default_rate'].plot(kind='bar', figsize=(10, 6))
plt.title("Actual Default Rate by Decision")
plt.xlabel("Decision")
plt.ylabel("Actual Default Rate")
plt.show()

thresholds = [0.03, 0.05, 0.08, 0.10, 0.15]

results = []

for t in thresholds:
    approved = policy_df[policy_df['pd'] < t]
    
    results.append({
        'threshold': t,
        'approval_rate': len(approved) / len(policy_df),
        'approved_default_rate': approved['actual_default'].mean() if len(approved) > 0 else np.nan
    })

threshold_df = pd.DataFrame(results)
print(threshold_df)

plt.figure(figsize=(10, 6))
plt.plot(threshold_df['threshold'], threshold_df['approval_rate'], marker='o', label='Approval Rate')
plt.plot(threshold_df['threshold'], threshold_df['approved_default_rate'], marker='o', label='Approved Default Rate')
plt.title("Approval Rate and Approved Default Rate vs Threshold")
plt.xlabel("PD Threshold")
plt.ylabel("Rate")
plt.legend()
plt.show()
