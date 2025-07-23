# %%
import pandas as pd

# %%
df = pd.read_excel(r'C:\Users\User\Desktop\Project Original\Stroke Risk Assessment Survey for TASUED Lecturers.xlsx')

# %%
df.head()

# %%
df.tail()

# %%
binary_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
df = df.rename(columns={
    'Gender': 'gender',
    'Age': 'age',
    'Weight (kmg)': 'weight',
    'Heigt (M)': 'height',
    'Smoking status': 'smoking_status',
    'Have you ever had a stroke?': 'stroke',
    'Have you been diagnosed with hypertension (high blood pressure)?': 'hypertension',
    'Do you have a history of heart disease?': 'heart_disease',
    'Have you ever been married?': 'ever_married',
    'Work Type': 'work_type',
    'Residence Type': 'Residence_type',
    'Have you ever been diagnosed with high blood sugar or diabetes?': 'diabetes'
})

# %%
df.drop(columns=[
    'Timestamp',
    'Do you consent to participate in this research and provide your responses for academic purposes?'
], inplace=True)

# %%
df['height'] = pd.to_numeric(df['height'], errors='coerce')
df.dropna(subset=['weight', 'height'], inplace=True)

# %%
print(df.columns)


# %%

df.columns = df.columns.str.strip().str.lower()

# %%
binary_map = {'Yes': 1, 'No': 0}


# %%
df['stroke'] = df['stroke'].map(binary_map)
df['hypertention'] = df['hypertention'].map(binary_map)
df['heart disease?'] = df['heart disease?'].map(binary_map)
df['married'] = df['married'].map(binary_map)
df['diabetes'] = df['diabetes'].map(binary_map)
df['do you have a family history of stroke?\n(i.e., parents, siblings, or grandparents)'] = df['do you have a family history of stroke?\n(i.e., parents, siblings, or grandparents)'].map(binary_map)
df['do you often experience delay in salary payments?'] = df['do you often experience delay in salary payments?'].map(binary_map)
df['do you consume alcohol?'] = df['do you consume alcohol?'].map(binary_map)
df['do you engage in regular physical exercise?'] = df['do you engage in regular physical exercise?'].map(binary_map)
df['are you currently without children (biological or adopted)?'] = df['are you currently without children (biological or adopted)?'].map(binary_map)
df['have you ever been diagnosed with high cholesterol?'] = df['have you ever been diagnosed with high cholesterol?'].map(binary_map)


# %%
df.dropna(subset=['stroke'], inplace=True)

# %%
df['bmi'] = df['weight'] / (df['height'] ** 2)

# %%

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Figure 4.4.1: Pearson Correlation Heatmap of Features")
plt.show()


# %%
# === FIGURE 4.4.3: Stroke-Specific Correlation Ranking ===
# This shows which features are most correlated with 'stroke'

stroke_corr = corr['stroke'].drop('stroke').sort_values(key=abs, ascending=False)

# Bar chart of correlations
plt.figure(figsize=(10, 6))
stroke_corr.plot(kind='barh', color='skyblue')
plt.title("Figure 4.4.3: Feature Correlation Strength with Stroke")
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# %%
print(df.columns.tolist())


# %%
# === FIGURE 4.2.2: Histograms and Boxplots for Numeric Columns ===
numeric_cols = ['age', 'bmi', 'how would you rate your current stress level?', 'how many hours of sleep do you get on average per night?']

plt.figure(figsize=(18, 10))

# Histograms
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, len(numeric_cols), i)
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col[:30]}...")  # truncate long titles

# Boxplots
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, len(numeric_cols), i + len(numeric_cols))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col[:30]}...")  # truncate long titles

plt.tight_layout()
plt.show()


# %%
categorical_cols = ['gender', 'work_type', 'residence_type', 'smoking_status']


# %%
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['gender', 'work_type', 'residence_type', 'smoking_status']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Check columns
categorical_cols = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']

# Plot
plt.figure(figsize=(12, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    sns.countplot(data=df, x=col)
    plt.title(f"Distribution of {col.capitalize()}")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# %%
df = df.rename(columns={
    'hypertention': 'hypertension',
    'heart disease?': 'heart_disease',
    'married': 'ever_married',
})


# %%
features = ['age', 'gender', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'residence_type', 'diabetes', 'bmi', 'smoking_status']
X = df[features]
y = df['stroke']

# %%
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt


X = X.copy()
X[X.select_dtypes(include='number').columns] = X.select_dtypes(include='number').fillna(X.median(numeric_only=True))

for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].fillna(X[col].mode()[0])

# If categorical variables are not yet encoded, encode them (LabelEncoding for mutual_info_classif)
from sklearn.preprocessing import LabelEncoder

X_encoded = X.copy()
for col in X_encoded.select_dtypes(include='object').columns:
    X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])

# Compute mutual information
mi_scores = mutual_info_classif(X_encoded, y, discrete_features='auto', random_state=42)

# Create and display DataFrame of results
mi_scores_df = pd.DataFrame({'Feature': X_encoded.columns, 'MI Score': mi_scores})
mi_scores_df.sort_values(by='MI Score', ascending=True, inplace=True)

# Display results
print(mi_scores_df)

# Optional: Plot
plt.figure(figsize=(10, 6))
plt.barh(mi_scores_df['Feature'], mi_scores_df['MI Score'], color='skyblue')
plt.title("Mutual Information Scores")
plt.xlabel("MI Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



# %%
X = X.dropna()
y = y[X.index]  # Keep labels aligned after dropping


# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Step 1: Copy your original features again if needed
features = ['age', 'gender', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'residence_type', 'diabetes', 'bmi', 'smoking_status']
X = df[features].copy()

# Step 2: Label encode any object columns
label_cols = X.select_dtypes(include='object').columns
le = LabelEncoder()
for col in label_cols:
    X[col] = le.fit_transform(X[col])

# Step 3: Ensure no missing values
X = X.dropna()

# Step 4: Generate Pearson Correlation Heatmap
corr_matrix = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Figure 4.5: Pearson Correlation Heatmap of Features')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# %%
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt

# Ensure y is aligned with X
y = df['stroke']
y = y[X.index]

# Compute mutual information
mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_df = mi_df.sort_values('MI Score', ascending=True)

# Plot the mutual information scores
plt.figure(figsize=(10, 6))
plt.barh(mi_df['Feature'], mi_df['MI Score'], color='skyblue')
plt.xlabel('Mutual Information Score')
plt.title('Figure 4.6: Feature Importance Using Mutual Information')
plt.tight_layout()
plt.show()



# %%
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)


# %%
# === Print shape of resampled dataset ===
print("Resampled feature set shape (X):", X_resampled.shape)
print("Resampled label set shape (y):", y_resampled.shape)

# Optionally print class balance
import numpy as np
unique, counts = np.unique(y_resampled, return_counts=True)
print("Class distribution after SMOTE:", dict(zip(unique, counts)))


# %%
df.head()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Bar plots for selected categorical features
categorical_features = ['gender', 'hypertension', 'diabetes', 'smoking_status', 'ever_married', 'do you engage in regular physical exercise?']
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

for i, col in enumerate(categorical_features):
    sns.countplot(data=df, x=col, ax=axs[i])
    axs[i].set_title(f'Distribution of {col}')
    axs[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# %%
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=df, x='smoking_status', ax=axs[0])
axs[0].set_title('Smoking Status Distribution')

sns.countplot(data=df, x='stroke', ax=axs[1])
axs[1].set_title('Stroke Diagnosis Distribution')
axs[1].set_xticklabels(['No Stroke', 'Stroke'])

plt.tight_layout()
plt.show()


# %%
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Age
sns.boxplot(data=df, x='stroke', y='age', ax=axs[0][0])
axs[0][0].set_title('Age Distribution by Stroke Class')

sns.histplot(data=df, x='age', hue='stroke', multiple='stack', ax=axs[0][1])
axs[0][1].set_title('Age Histogram by Stroke Class')

# BMI
sns.boxplot(data=df, x='stroke', y='bmi', ax=axs[1][0])
axs[1][0].set_title('BMI Distribution by Stroke Class')

sns.histplot(data=df, x='bmi', hue='stroke', multiple='stack', ax=axs[1][1])
axs[1][1].set_title('BMI Histogram by Stroke Class')

plt.tight_layout()
plt.show()


# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# %%
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


# %%
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)


# %%
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# %%
print(sm)


# %%
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)


# %%
from xgboost import XGBClassifier
model = XGBClassifier(eval_metric='logloss')


# %%
print(model)


# %%
from xgboost import XGBClassifier

# Create model without deprecated arguments
model = XGBClassifier(eval_metric='logloss')

# Train the model
model.fit(X_resampled, y_resampled)

# Optional: Evaluate
print(model)


# %%
from xgboost import XGBClassifier

# Create the model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Train the model
xgb_model.fit(X_train, y_train)

# Then evaluate
evaluate_model("XGBoost", xgb_model)


# %%
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(name, model):
    print(f"\n--- {name} ---")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {name}")
    plt.legend()
    plt.grid(True)
    plt.show()

evaluate_model("Logistic Regression", lr_model)
evaluate_model("Random Forest", rf_model)
evaluate_model("XGBoost", xgb_model)



