import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("Student Mental health.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()
df.dropna(inplace=True)
df['Choose your gender'] = df['Choose your gender'].map({'Male':0, 'Female':1})

df['Do you have Depression?'] = df['Do you have Depression?'].map({'Yes':1, 'No':0})
df['Do you have Anxiety?'] = df['Do you have Anxiety?'].map({'Yes':1, 'No':0})
df['Do you have Panic attack?'] = df['Do you have Panic attack?'].map({'Yes':1, 'No':0})
df['What is your CGPA?'] = df['What is your CGPA?'].astype(str)
df['What is your CGPA?'] = df['What is your CGPA?'].apply(lambda x: float(x.split('-')[0].strip()))
print(df.columns)
df.columns = df.columns.str.strip()
df.drop(['Timestamp', 'What is your course?', 'Your current year of Study'], axis=1, inplace=True, errors='ignore')
# First, check the actual column names in your DataFrame
print(df.columns.tolist())

# Then drop the columns using the correct column names
# For example, if the actual names are slightly different:

# Alternatively, if you're unsure about exact names, you can use a safer approach:
# This will only drop columns that exist in the DataFrame
columns_to_drop = ['Timestamp', 'What is your course?', 'Your current year of Study']
existing_columns = [col for col in columns_to_drop if col in df.columns]
df.drop(existing_columns, axis=1, inplace=True)
df['Marital status'] = df['Marital status'].map({'No':0, 'Yes':1})

df['Did you seek any specialist for a treatment?'] = df['Did you seek any specialist for a treatment?'].map({'No':0, 'Yes':1})
X = df.drop('Do you have Depression?', axis=1)
y = df['Do you have Depression?']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight='balanced',n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
importances = model.feature_importances_

sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance")
plt.show()
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_scaled, y, cv=5)

print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
import pickle

pickle.dump(model, open("mental_health_model.pkl", "wb"))
