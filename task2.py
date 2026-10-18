import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. Data Loader Class
# ==========================================
class DataLoader:
    def __init__(self, file_path: str, features: list[str], target: str, sep: str = ',') -> None:
        self.file_path: str = file_path
        self.features: list[str] = features
        self.target: str = target
        self.sep: str = sep

        self._X_train: pd.DataFrame | None = None
        self._X_test: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None
        self._y_test: pd.Series | None = None

        self._X_full: pd.DataFrame | None = None
        self._y_full: pd.Series | None = None

    def load_and_split(self, test_size: float = 0.2, random_state: int = 42) -> None:
        df: pd.DataFrame = pd.read_csv(self.file_path, sep=self.sep)

        required_cols = self.features + [self.target]
        df = df.dropna(subset=required_cols)

        X: pd.DataFrame = df[self.features].copy()
        y: pd.Series = df[self.target]

        for col in self.features:
            if X[col].dtype == 'object':
                print(f"Feature: {col} is categorical, encoding to numerical...")
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])

        if y.dtype == 'object':
            print(f"Target: {self.target} is categorical, encoding to numerical...")
            encoder_y = LabelEncoder()
            y = pd.Series(encoder_y.fit_transform(y), index=y.index)

        self._X_full, self._y_full = X, y

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def plot_first_data(self, title: str = "Step 1 - Just the data"):
        if self._X_full is None:
            print('You have to load the data first.')
            return
        
        num_features = len(self.features)
        fig, axes = plt.subplots(1, num_features, figsize=(4 * num_features, 4))
        if num_features == 1:
            axes = [axes]

        for ax, col in zip(axes, self.features):
            ax.scatter(self._X_full[col], self._y_full, s=16, alpha=0.6, color="#7A8B99")
            ax.set_xlabel(col)
            ax.set_ylabel(self.target)
            ax.set_title(f"{col} vs {self.target}")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show(block=False)

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
                f"Features (X)     : {', '.join(self.features)}\n"
                f"Target (y)       : {self.target}\n"
                f"Training Samples : {self._X_train.shape[0]}\n"
                f"Testing Samples  : {self._X_test.shape[0]}"
            )
        return "Dataset has not been loaded or split yet."


# ==========================================
# 2. Model Trainer Class
# ==========================================
class ModelTrainer:
    def __init__(self) -> None:
        self.model: LinearRegression = LinearRegression()
        self.is_trained: bool = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
        self.model.fit(X_train, y_train)
        self.is_trained = True
        b0: float = float(self.model.intercept_)
        coefficients = self.model.coef_

        print(f"Model has been trained successfully!")
        print(f"Intercept (b0): {b0:.2f}")
        for col, coef in zip(X_train.columns, coefficients):
            print(f"Coeff ({col}): {coef:.4f}")

        return self.model


# ==========================================
# 3. Model Evaluator Class
# ==========================================
class ModelEvaluator:
    def __init__(self, model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        self.model: LinearRegression = model
        self.X_test: pd.DataFrame = X_test
        self.y_test: pd.Series = y_test

        self.y_pred: np.ndarray = self.model.predict(self.X_test)
        self.residuals: np.ndarray = self.y_test - self.y_pred

    def predict(self) -> np.ndarray:
        self.y_pred = self.model.predict(self.X_test)
        print("Predicted successfully.")
        return self.y_pred

    def evaluate_metrics(self) -> dict[str, float]:
        if self.y_pred is None:
            self.predict()

        mse: float = float(mean_squared_error(self.y_test, self.y_pred))
        rmse: float = float(np.sqrt(mse))
        mae: float = float(mean_absolute_error(self.y_test, self.y_pred))
        r2: float = float(r2_score(self.y_test, self.y_pred))

        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAE  : {mae:.2f}")
        print(f"R2   : {r2:.2f}")

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def plot_results(self) -> None:
        if self.y_pred is None:
            self.predict()
 
        r2 = r2_score(self.y_test, self.y_pred)
 
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        
        lims = [min(self.y_test.min(), self.y_pred.min()), max(self.y_test.max(), self.y_pred.max())]
        ax[0].scatter(self.y_test, self.y_pred, s=20, alpha=0.7, color='#7A8B99')
        ax[0].plot(lims, lims, "r--", label="Perfect Prediction")
        ax[0].set_xlabel("Actual Sales")
        ax[0].set_ylabel("Predicted Sales")
        ax[0].set_title(f"Actual vs Predicted (R2 = {r2:,.2f})")
        ax[0].legend()

        ax[1].scatter(self.y_pred, self.residuals, s=20, alpha=0.7, color='#C8553D')
        ax[1].axhline(y=0, color='black', linestyle='--')
        ax[1].set_xlabel("Predicted Sales")
        ax[1].set_ylabel("Residuals (Error)")
        ax[1].set_title("Residuals Analysis")

        plt.tight_layout()
        plt.show(block=False)

        print(f"Mean Residual (Error Bias): {self.residuals.mean():.4f}")


# ==========================================
# 4. Pipeline Execution
# ==========================================
if __name__ == "__main__":
    features_list = ['TV', 'Radio', 'Newspaper']
    
    dt = DataLoader(file_path="Advertising.csv", features=features_list, target='Sales', sep=',')
    dt.load_and_split(test_size=0.2, random_state=42)
    dt.plot_first_data()

    print(dt.summary)
    print('=' * 40)

    trainer = ModelTrainer()
    trained_model = trainer.train(dt.X_train, dt.y_train)

    print('=' * 40)

    evaluator = ModelEvaluator(model=trained_model, X_test=dt.X_test, y_test=dt.y_test)
    metrics = evaluator.evaluate_metrics()

    print('=' * 40)
    evaluator.plot_results()

    plt.show()
