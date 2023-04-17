import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess(data):
    target_var = data.columns[-1]
    data = handle_null_missing_values(data)
    data = encode(data)
    input_var = data.drop(target_var, axis=1)
    input_var = remove_outliers(input_var)
    input_var = scale_data(input_var)
    data = pd.concat([input_var, data[target_var]], axis=1)
    return data


def handle_null_missing_values(data):
    for col in data.columns:
        if data[col].isna().sum() > 0:
            if type(data[col].loc[0]) == 'str':
                data[col].fillna(method='bfill', inplace=True)
                data[col].fillna(method='ffill', inplace=True)
            else:
                data[col].fillna(np.round(np.mean(data[col]), decimals=0), inplace=True)
    return data


def encode(data):
    encoder = LabelEncoder()
    for col in data.columns:
        if type(data[col].loc[0]) == 'str':
            if not data[col].loc[0].isnumeric():
                data[col] = encoder.fit_transform(data[col])
            else:
                data[col] = int(data[col])
    return data


def remove_outliers(data):
    for col in data.columns:
        Q1 = data[col].quantile(.25)
        Q3 = data[col].quantile(.75)
        IQR = (Q3 - Q1) * 1.5
        lower = Q1 - IQR
        upper = Q3 + IQR
        if np.sum(((data[col] < lower) | (data[col] > upper))) > 0:
            # data[col] = np.clip(lower=data[col].quantile(.25), upper=data[col].quantile(.75))
            mean_without_outliers = np.mean(data[col][(data[col] >= lower) & (data[col] <= upper)])
            indices = data.index[(data[col] < lower) | (data[col] > upper)]
            data.loc[indices, col] = mean_without_outliers

    return data


def scale_data(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)


def split(input_, output):
    return train_test_split(input_, output, test_size=.2, random_state=42)
