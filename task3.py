import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("Machine Learning — Section 3: Multiple & Polynomial Regression")
print("=" * 60)

try:
    df = pd.read_csv("insurance.csv")
except FileNotFoundError:
    print("Error: 'insurance.csv' not found. Please place it in the same directory.")
    exit()

print(f"\nOriginal Dataset Shape: {df.shape}")
print(df.head(3))


df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

X = df.drop(columns=['charges'])
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")


print("\n--- [1] Multiple Linear Regression Model ---")
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
y_pred_mlr = mlr_model.predict(X_test)

mae_mlr = mean_absolute_error(y_test, y_pred_mlr)
rmse_mlr = np.sqrt(mean_squared_error(y_test, y_pred_mlr))
r2_mlr = r2_score(y_test, y_pred_mlr)

print(f"MAE : {mae_mlr:,.2f}")
print(f"RMSE: {rmse_mlr:,.2f}")
print(f"R²  : {r2_mlr:.3f}")


print("\n--- [2] Polynomial Regression Model (Degree = 2) ---")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

mae_poly = mean_absolute_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
r2_poly = r2_score(y_test, y_pred_poly)

print(f"MAE : {mae_poly:,.2f}")
print(f"RMSE: {rmse_poly:,.2f}")
print(f"R²  : {r2_poly:.3f}")


coefs = pd.Series(mlr_model.coef_, index=X.columns)
print("\nMultiple Linear Regression Coefficients:")
print(coefs.round(2))
print(f"Intercept: {round(mlr_model.intercept_, 2)}")

plt.figure(figsize=(10, 6))
coefs.plot(kind="barh", color=["#0E7C7B" if c > 0 else "#C8553D" for c in coefs])
plt.title("Learned Coefficients (Effect on Charges)")
plt.xlabel("Coefficient Value")
plt.axvline(0, color="black", lw=1)
plt.tight_layout()

plt.show()