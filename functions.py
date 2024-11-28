import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(df, feature_col='Close', train_ratio=0.70, lookback=100):
    """
    Splits the data into training and testing datasets and scales the data.
    Returns training arrays (x_train, y_train), testing arrays (x_test, y_test), 
    and the scaler used.
    """
    data = df[[feature_col]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * train_ratio)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - lookback:]

    # Preparing x_train and y_train
    x_train, y_train = [], []
    for i in range(lookback, len(train_data)):
        x_train.append(train_data[i - lookback:i, 0])
        y_train.append(train_data[i, 0])

    # Preparing x_test and y_test
    x_test, y_test = [], []
    for i in range(lookback, len(test_data)):
        x_test.append(test_data[i - lookback:i, 0])
        y_test.append(test_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    return x_train, y_train, x_test, y_test, scaler
