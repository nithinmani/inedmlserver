import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, LSTM, Flatten, Input
from sklearn.ensemble import RandomForestRegressor
import json
from keras.callbacks import EarlyStopping
import joblib

from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import newssent
import sentiment


def load_data(ticker):
    data = yf.download(ticker, start="2018-01-01", end="2023-05-31")

    data.to_csv('company.csv')
    input_col = 'Close'

    # Set the number of time steps to predict into the future
    n_steps = 365

    # Preprocess the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[[input_col]])
    n_features = scaled_data.shape[1]

    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data)-sequence_length-1):
            X.append(data[i:(i+sequence_length), :])
            y.append(data[(i+sequence_length), :])
        return np.array(X), np.array(y)

    sequence_length = 50
    X, y = create_sequences(scaled_data, sequence_length)
    train_size = int(0.80 * len(X))
    return X, y, n_features, scaler


def train_and_predict(ticker):
    sequence_length = 50
    n_steps = 365
    X, y, n_features, scaler = load_data(ticker)
    train_size = int(0.80 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Define the LSTM model
    # Define model architecture
    from keras.optimizers import Adam

    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
              input_shape=(sequence_length, n_features)))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model_path = f"{ticker}_model_lstm"
    # Train the LSTM model

    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
    save_model(model, model_path)

    last_sequence = X_test[-1]
    print("last", last_sequence)
    predictions = []
    for i in range(n_steps):
        predicted_price = model.predict(
            last_sequence.reshape(1, sequence_length, n_features))[0, 0]

        predictions.append(predicted_price)

        last_sequence = np.append(
            last_sequence[1:], predicted_price.reshape((1, 1)), axis=0)

    print(predictions)

    predicted_prices = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1))

    # Reshape the input and target arrays
    X_train = X_train.reshape((X_train.shape[0], -1))
    y_train = y_train.ravel()

    # Create and train a random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    save_path = f"rf_{ticker}_model.pkl"
    joblib.dump(rf_model, save_path)
    # save_model(rf_model, "{ticker}_model_rf")

    rf_predictions = []
    last_sequence = X_test[-1][-n_steps:]
    # reshape to match the size of last_sequence
    last_sequence = last_sequence.reshape((1, last_sequence.shape[0]))
    for i in range(n_steps):
        rf_prediction = rf_model.predict(
            last_sequence.reshape((1, last_sequence.shape[1])))
        rf_predictions.append(rf_prediction[0])
        last_sequence = np.append(
            last_sequence[:, 1:], rf_prediction.reshape((1, 1)), axis=1)

    predicted_prices2 = scaler.inverse_transform(
        np.array(rf_predictions).reshape(-1, 1))

    ensemble_predictions = []
    for lstm_pred, rf_pred in zip(predictions, rf_predictions):
        ensemble_pred = (lstm_pred + rf_pred) / 2.0
        ensemble_predictions.append(ensemble_pred)
    print(ensemble_predictions)
    ess = scaler.inverse_transform(ensemble_predictions)
    print(ess)
    return predicted_prices, predicted_prices2, ess


# test_model_save("tsla")


def load_and_predict(ticker):
    try:
        sequence_length = 50
        n_steps = 365
        X, y, n_features, scaler = load_data(ticker)
        train_size = int(0.80 * len(X))
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        model_path = f"{ticker}_model_lstm"
        model = load_model(model_path)
        print(model.summary())
        last_sequence = X_test[-1]
        predictions = []
        for i in range(n_steps):
            predicted_price = model.predict(
                last_sequence.reshape(1, sequence_length, n_features))[0, 0]

            predictions.append(predicted_price)

            last_sequence = np.append(
                last_sequence[1:], predicted_price.reshape((1, 1)), axis=0)

        print(predictions)

        predicted_prices = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1))

        # Reshape the input and target arrays
        X_train = X_train.reshape((X_train.shape[0], -1))
        y_train = y_train.ravel()

        # Create and train a random forest model
        # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        # rf_model.fit(X_train, y_train)
        rf_model = joblib.load(f"rf_{ticker}_model.pkl")
        rf_predictions = []
        last_sequence = X_test[-1][-n_steps:]
        # reshape to match the size of last_sequence
        last_sequence = last_sequence.reshape((1, last_sequence.shape[0]))
        for i in range(n_steps):
            rf_prediction = rf_model.predict(
                last_sequence.reshape((1, last_sequence.shape[1])))
            rf_predictions.append(rf_prediction[0])
            last_sequence = np.append(
                last_sequence[:, 1:], rf_prediction.reshape((1, 1)), axis=1)

        predicted_prices2 = scaler.inverse_transform(
            np.array(rf_predictions).reshape(-1, 1))

        ensemble_predictions = []
        for lstm_pred, rf_pred in zip(predictions, rf_predictions):
            ensemble_pred = (lstm_pred + rf_pred) / 2.0
            ensemble_predictions.append(ensemble_pred)
        print(ensemble_predictions)
        ess = scaler.inverse_transform(ensemble_predictions)
        print(ess)
        return predicted_prices, predicted_prices2, ess
    except IOError as e:
        print(e)
        return train_and_predict(ticker)
