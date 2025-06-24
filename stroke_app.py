#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[6]:


# df = pd.read_csv(r'C:\Users\User\Desktop\MY PROJECT WORK\healthcare-dataset-stroke-data.csv')


# In[7]:


# print(df.info())
# print(df.isnull().sum())


# In[8]:


# df = df.drop(columns=['id'], errors='ignore')


# In[9]:


# df['bmi'] = df['bmi'].fillna(df['bmi'].median())


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
# df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)


# In[12]:


# X = df.drop('stroke', axis=1)
# y = df['stroke']


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# In[14]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# In[15]:


from sklearn.ensemble import RandomForestClassifier
importances = RandomForestClassifier().fit(X_train_res, y_train_res).feature_importances_
feature_scores = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print(feature_scores)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# In[17]:


from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train_res, y_train_res)


# In[18]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))


# In[19]:


import pickle

with open("stroke_nb_model.pkl", "wb") as f:
    pickle.dump(model, f)

