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
        df: pd.DataFrame = pd.read_csv(self.file_path, sep=self.sep)

        initial_rows = len(df)
        df = df.dropna(subset=([self.feature, self.target]))
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"{dropped_rows} rows have been deleted due to missing values.")
            
        X: pd.DataFrame = df[[self.feature]].copy()
        y: pd.Series = df[self.target]

        if X[self.feature].dtype == 'object':
            print(f"Feature: {self.feature} is categorical, encoding to numerical...")
            encoder = LabelEncoder()
            X[self.feature] = encoder.fit_transform(X[self.feature])

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
        
        plt.figure(figsize=(6, 4))
        plt.scatter(self._X_full[self.feature], self._y_full,
                    s=16, alpha=0.6, color="#7A8B99")
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
        b1: float = float(self.model.coef_[0])

        print(f"Model has been trained successfully! Intercept: {b0:.2f}, Slope: {b1:.2f}")
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
        ax[0].scatter(self.X_test, self.y_test, s=15, alpha=0.6, color='#7A8B99', label='test data')
        ax[0].plot(self.X_test, self.y_pred, color='#C8553D', lw=3, label='model')
        ax[0].set_xlabel(self.X_test.columns[0])
        ax[0].set_ylabel(self.y_test.name)
        ax[0].legend()
 
        lims = [self.y_test.min(), self.y_test.max()]
        ax[1].scatter(self.y_test, self.y_pred, s=15, alpha=0.7, color='#7A8B99')
        ax[1].plot(lims, lims, "r--", label="perfect prediction")
        ax[1].set_xlabel("actual")
        ax[1].set_ylabel("predicted")
        ax[1].set_title(f"Actual vs Predicted (R2 = {r2:,.2f})")
        ax[1].legend()
        plt.tight_layout()
        plt.show()

        print(f"Mean Residual (Error Bias): {self.residuals.mean():.4f}")


# ==========================================
# 4. Pipeline Execution
# ==========================================
if __name__ == "__main__":
    # Load and prepare data
    dt = DataLoader(file_path="Advertising.csv", feature='TV', target='Sales', sep=';')
    dt.load_and_split(test_size=0.2, random_state=42)
    dt.plot_first_data()

    print(dt.summary)
    print('=' * 40)

    # Train the model
    trainer = ModelTrainer()
    trained_model = trainer.train(dt.X_train, dt.y_train)

    print('=' * 40)

    # Evaluate the model
    evaluator = ModelEvaluator(model=trained_model, X_test=dt.X_test, y_test=dt.y_test)
    metrics = evaluator.evaluate_metrics()

    print('=' * 40)
    evaluator.plot_results()