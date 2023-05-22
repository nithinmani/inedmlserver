from flask import Flask, jsonify,request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import joblib
import requests
import numpy as np
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
from model_test import load_and_predict
import newssent
import sentiment

app = Flask(__name__)
CORS(app)
# load the NLTK VADER lexicon for sentiment analysis

def get_stocks(ticker):
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/hi/history/{ticker}/15m"
    querystring = {"diffandsplits": "false"}
    headers = {
        "X-RapidAPI-Key": "080d5391a5msh8465fd7b03a93c4p1725e9jsnb512929f94b5",
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
        "X-RapidAPI-Key": "080d5391a5msh8465fd7b03a93c4p1725e9jsnb512929f94b5",
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
    predicted_prices, predicted_prices2, ess = load_and_predict(ticker)
    return jsonify({'predicted_prices': predicted_prices.tolist(), 'random': predicted_prices2.tolist(), 'combine': ess.tolist()})







@app.route('/hist_sentiment/<ticker>', methods=['GET'])
def hist_sentiment(ticker):
    # Call the newssent function to get the sentiment scores
    sentiment_df = newssent.newssentiment(ticker)

    # Convert the DataFrame to a JSON object
    sentiment_json = sentiment_df.to_json(orient='records')
    print(sentiment_json)

    # Return the JSON object
    return jsonify(sentiment_json)

import json

import numpy as np

@app.route('/news_sentiment/<ticker>', methods=['GET'])
def get_sentiment(ticker):
    # Call the newssent function to get the sentiment scores
    sentiment_df = sentiment.newssentiment(ticker)

    # Convert the NumPy array to a Python list
    sentiment_list = sentiment_df.tolist()

    # Return the JSON object
    return json.dumps(sentiment_list)


if __name__ == '__main__':
    app.run(port=5000)

