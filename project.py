import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class DataLoader:
    def __init__(self, file_path: str, feature: str, target: str, sep: str = ',') -> None:
        self.file_path: str = file_path
        self.feature: str = feature
        self.target: str = target
        self.sep: str = sep

        self._X_train: pd.DataFrame | None = None
        self._X_test: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None
        self._y_test: pd.Series | None = None

        self._X_full: pd.DataFrame | None = None
        self._y_full: pd.Series | None = None

    def load_and_split(self, test_size: float = 0.2, random_state: int = 42) -> None:
        try:
            df: pd.DataFrame = pd.read_csv(self.file_path, sep=self.sep)
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' was not found.")
            return

        initial_rows = len(df)
        df = df.dropna(subset=[self.feature, self.target])
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"{dropped_rows} row(s) deleted due to missing values.")
            
        X: pd.DataFrame = df[[self.feature]].copy()
        y: pd.Series = df[self.target]

        if X[self.feature].dtype == 'object':
            print(f"Feature '{self.feature}' is categorical, encoding to numerical...")
            encoder = LabelEncoder()
            X[self.feature] = encoder.fit_transform(X[self.feature])

        if y.dtype == 'object':
            print(f"Target '{self.target}' is categorical, encoding to numerical...")
            encoder_y = LabelEncoder()
            y = pd.Series(encoder_y.fit_transform(y), index=y.index)

        self._X_full, self._y_full = X, y

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def plot_first_data(self, title: str = "Step 1 - Data Visualization") -> None:
        if self._X_full is None:
            print("Data has not been loaded yet.")
            return
        
        plt.figure(figsize=(6, 4))
        plt.scatter(self._X_full[self.feature], self._y_full, s=16, alpha=0.6, color="#7A8B99")
        plt.xlabel(self.feature)
        plt.ylabel(self.target)
        plt.title(title)
        plt.show()

    @property
    def X_train(self) -> pd.DataFrame | None:
        return self._X_train

    @property
    def X_test(self) -> pd.DataFrame | None:
        return self._X_test

    @property
    def y_train(self) -> pd.Series | None:
        return self._y_train        

    @property
    def y_test(self) -> pd.Series | None:
        return self._y_test

    @property
    def summary(self) -> str:
        if self._X_train is not None:
            return (
                f"Feature (X)      : {self.feature}\n"
                f"Target (y)       : {self.target}\n"
                f"Training Samples : {self._X_train.shape[0]}\n"
                f"Testing Samples  : {self._X_test.shape[0]}\n"
            )
        return "Dataset has not been loaded or split yet."


class ModelTrainer:
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
        print("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model


class ModelEvaluator:
    def __init__(self, model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)

    def evalaute_metrics(self) -> dict:
        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2 = r2_score(self.y_test, self.y_pred)

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R²   : {r2:.4f}")

        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    def plot_results(self, title: str = "Model Predictions vs Actual Data") -> None:
        plt.figure(figsize=(6, 4))
        plt.scatter(self.X_test, self.y_test, color="#7A8B99", s=16, alpha=0.7, label="Actual Data")
        plt.plot(self.X_test, self.y_pred, color="#C8553D", linewidth=2, label="Regression Line")
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.title(title)
        plt.legend()
        plt.show()

def run_multiple_and_polynomial_regression(file_path: str = "insurance.csv") -> None:
    print("\n" + "=" * 60)
    print("PROJECT 2: Multiple & Polynomial Regression")
    print("=" * 60)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print
        