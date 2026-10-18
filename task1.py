import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("titanic.csv")
cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
df = df[cols]

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

print("--- Jumlah Sampel per Kelas ---")
print(df["survived"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x="survived", data=df, palette="Set2")
plt.title("Distribution of Survived vs Not Survived")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n--- Hasil Evaluasi ---")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))