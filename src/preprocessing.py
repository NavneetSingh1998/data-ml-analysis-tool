import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def handle_missing_values(df):
    """Handle missing values in DataFrame."""
    imputer = SimpleImputer(strategy='mean')
    df.iloc[:, :] = imputer.fit_transform(df)
    return df


def detect_outliers(df, column):
    """Detect outliers in a DataFrame column using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def scale_features(df, method='standard'): 
    """Scale features using StandardScaler or MinMaxScaler."""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method. Choose 'standard' or 'minmax'.")
    df.iloc[:, :] = scaler.fit_transform(df)
    return df


def encode_categorical(df, categorical_cols):
    """Encode categorical variables using OneHotEncoder."""
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    df_encoded = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, df_encoded], axis=1)
    return df
