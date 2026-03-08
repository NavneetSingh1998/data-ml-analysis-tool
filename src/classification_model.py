import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense

class ClassificationModel:
    def __init__(self, model_type='logistic', random_state=42):
        self.model = self._initialize_model(model_type)

    def _initialize_model(self, model_type):
        if model_type == 'logistic':
            return LogisticRegression(random_state=random_state)
        elif model_type == 'svm':
            return SVC(probability=True, random_state=random_state)
        elif model_type == 'random_forest':
            return RandomForestClassifier(random_state=random_state)
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=random_state)
        elif model_type == 'neural_network':
            model = Sequential()
            model.add(Dense(32, activation='relu', input_shape=(None,)))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        else:
            raise ValueError('Model type not recognized')

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        if isinstance(self.model, Sequential):
            self.model.fit(X_train, y_train, epochs=10, batch_size=32)
        else:
            self.model.fit(X_train, y_train)
        return self.evaluate(X_val, y_val)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        return accuracy, report