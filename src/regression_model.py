from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class RegressionModel:
    def __init__(self, model_type='linear'):
        self.model_type = model_type
        self.model = self.build_model()

    def build_model(self):
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            return Ridge()
        elif self.model_type == 'lasso':
            return Lasso()
        elif self.model_type == 'random_forest':
            return RandomForestRegressor()
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor()
        else:
            raise ValueError('Invalid model type')

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return self.evaluate(X_val, y_val)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        return {'mse': mse, 'r2': r2}
