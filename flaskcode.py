from flask import Flask, jsonify,request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import joblib
import requests

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Input
from sklearn.ensemble import RandomForestRegressor
import json
from keras.callbacks import EarlyStopping

from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import newssent
import subprocess

app = Flask(__name__)
CORS(app)
# load the NLTK VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()


def get_stocks(ticker):
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/hi/history/{ticker}/15m"
    querystring = {"diffandsplits": "false"}
    headers = {
        "X-RapidAPI-Key": "0a47ceaf09msh3f22cd364c17590p160ab2jsnfe62bb61ad68",
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }
    response = requests.request(
        "GET", url, headers=headers, params=querystring)
    data = response.json()
    ret_list = []
    items = data.get("items")
    for key in items.keys():
        item = items[key]
        data = [item["date"], item["close"]]
        ret_list.append(data)
    return ret_list


@app.route('/get_recommendation_trend/<ticker>', methods=['GET'])
def get_recommendation_trend(ticker):
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{ticker}/recommendation-trend"

    headers = {
        "X-RapidAPI-Key": "0a47ceaf09msh3f22cd364c17590p160ab2jsnfe62bb61ad68",
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    return jsonify(data)


@app.route('/api/stock-data/<ticker>')
def get_stock_data(ticker):
    df = yf.download(ticker, period='1d', interval='1m')
    json_string = df.tail(1).to_json()
    return jsonify(json_string)




@app.route('/api/screener')
def get_screener_data():
    url = "https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/growth_technology_stocks"
    querystring = {"start": "0"}
    headers = {
        "X-RapidAPI-Key": "0a47ceaf09msh3f22cd364c17590p160ab2jsnfe62bb61ad68",
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }
    response = requests.request(
        "GET", url, headers=headers, params=querystring)
    data = response.json()
    return jsonify(data)


@app.route('/api/undervalue')
def get_screener1_data():
    url = "https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/undervalued_growth_stocks"
    querystring = {"start": "0"}
    headers = {
        "X-RapidAPI-Key": "0a47ceaf09msh3f22cd364c17590p160ab2jsnfe62bb61ad68",
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }
    response = requests.request(
        "GET", url, headers=headers, params=querystring)
    data = response.json()
    return jsonify(data)


@app.route('/api/news/<ticker>')
def get_news_data(ticker):
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/ne/news/{ticker}"
    headers = {
        "X-RapidAPI-Key": "0a47ceaf09msh3f22cd364c17590p160ab2jsnfe62bb61ad68",
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }
    response = requests.request(
        "GET", url, headers=headers)
    data = response.json()
    return jsonify(data)


@app.route('/get_yahoo_finance_data/<ticker>', methods=['GET'])
def get_yahoo_finance_data(ticker):
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/mo/module/{ticker}"

    querystring = {
        "module": "asset-profile,financial-data,earnings,institution-ownership"}

    headers = {
        "X-RapidAPI-Key": "0a47ceaf09msh3f22cd364c17590p160ab2jsnfe62bb61ad68",
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }

    response = requests.request(
        "GET", url, headers=headers, params=querystring)

    return jsonify(response.json())


@app.route('/getDailydata/<ticker>', methods=['GET'])
def get_daily_data(ticker):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-summary"

    querystring = {"symbol": f"{ticker}"}

    headers = {
        "X-RapidAPI-Key": "080d5391a5msh8465fd7b03a93c4p1725e9jsnb512929f94b5",
        "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }

    response = requests.request(
        "GET", url, headers=headers, params=querystring)
    return jsonify(response.json())


@app.route("/stocks/<ticker>")
def return_stocks(ticker):
    data = get_stocks(ticker)
    return jsonify(data)


@app.route('/api/aggressive_small_cap', methods=['GET'])
def get_aggressive_small_cap():
    url = "https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/aggressive_small_caps"
    querystring = {"start": "0"}
    headers = {
        "X-RapidAPI-Key": "080d5391a5msh8465fd7b03a93c4p1725e9jsnb512929f94b5",
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    return jsonify(response.json())


@app.route('/api/predict/<ticker>', methods=['GET'])
def predict(ticker):

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
    # Train the LSTM model

    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

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
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

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
    return jsonify({'predicted_prices': predicted_prices.tolist(), 'random': predicted_prices2.tolist(), 'combine': ess.tolist()})



@app.route('/news_sentiment/<ticker>', methods=['GET'])
def get_sentiment(ticker):
    # Call the newssent function to get the sentiment scores
    sentiment_df = newssent.newssentiment(ticker)

    # Convert the DataFrame to a JSON object
    sentiment_json = sentiment_df.to_json(orient='records')
    print(sentiment_json)

    # Return the JSON object
    return jsonify(sentiment_json)

if __name__ == '__main__':

    app.run(port=5000)
