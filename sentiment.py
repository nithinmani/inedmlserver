from datetime import datetime, timedelta
import requests

def newssentiment(ticker):
    import pandas as pd
    company = ticker  # Replace with the name of the company you're interested in
    api_key = 'chad2kpr01qhe0ra0vv0chad2kpr01qhe0ra0vvg'  # Get your API key from https://finnhub.io

    # Set the date range for the news articles you want to retrieve
    month_span=12
    to_date = datetime.today()

    # Initialize an empty list to store the news articles
    news_data = []
    # Iterate over the months
    for i in range(month_span):
        from_date=to_date - timedelta(days=30)
        print(from_date.strftime("%Y-%m-%d"),end="\t")
        print(to_date.strftime("%Y-%m-%d"))

        # Use a while loop to paginate through the results and retrieve more news articles
        cursor = ''
        while True:
            url = f'https://finnhub.io/api/v1/company-news?symbol={company}&from={from_date.strftime("%Y-%m-%d")}&to={to_date.strftime("%Y-%m-%d")}&token={api_key}&cursor={cursor}'
            response = requests.get(url)
            data = response.json()
            # Append the news articles to the list
            news_data.extend(data)

            # Check if there are more news articles to retrieve
            if 'next' in response.headers:
                cursor = response.headers['next']
            else:
                break

        to_date=from_date

    # Convert the news data to a pandas DataFrame
    df = pd.DataFrame(news_data)

    df.head()
    df.drop_duplicates(subset=['summary'], inplace=True)

    # convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
    #
    # extract date from datetime
    df['datetime'] = df['datetime'].dt.date

    # Save the DataFrame to an Excel file
    df.to_csv('finance.csv')

    print(f"Successfully saved {len(df)} financial news articles to {company}_financial_news.xlsx")


    # In[49]:


    import pandas as pd
    from textblob import TextBlob
    import nltk
    nltk.download('vader_lexicon')

    from nltk.sentiment import SentimentIntensityAnalyzer
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    # load the tweet dataset
    tweet_df = pd.read_csv('finance.csv')

    # convert the 'created_at' column to datetime format
    tweet_df['Date'] = pd.to_datetime(tweet_df['datetime'])

    # set the 'created_at' column as the index
    tweet_df.set_index('Date', inplace=True)

    # group the tweets by day and concatenate the text of each tweet
    #tweet_df = tweet_df.groupby(pd.Grouper(freq='D')).agg({'headline': ' '.join})

    # reset the index
    tweet_df = tweet_df.reset_index()
    # drop unwanted columns
    # create a list of columns to drop
    columns_to_drop = [col for col in tweet_df.columns if col not in ['datetime', 'headline']]

    # drop the columns
    tweet_df.drop(columns_to_drop, axis=1, inplace=True)
    # rename the columns
    tweet_df.columns = ['Date', 'sentiment_score']

    # convert the 'date' column to datetime format
    tweet_df['Date'] = pd.to_datetime(tweet_df['Date']).dt.date

    # convert sentiment_score column to string format
    tweet_df['sentiment_score'] = tweet_df['sentiment_score'].astype(str)
    # calculate the sentiment score for each day
    tweet_df['sentiment_score'] = tweet_df['sentiment_score'].apply(lambda x: TextBlob(x).sentiment.polarity)

    print(tweet_df['sentiment_score'])

    tweet_df.to_csv('teslasentiment.csv')
   
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense, LSTM,Dropout
    from sklearn.ensemble import RandomForestRegressor

    # Load the dataset
    data = pd.read_csv('teslasentiment.csv', index_col=0)
    # Choose the column to use as input
    input_col = 'sentiment_score'

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
    # Split the data into training and testing sets
    train_size = int(0.80 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    # Define the LSTM model
    # Define model architecture
    from keras.optimizers import Adam

    # Reshape the input and target arrays
    X_train = X_train.reshape((X_train.shape[0], -1))
    y_train = y_train.ravel()

    # Create and train a random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions using the Random Forest model
    rf_predictions = []
    last_sequence = X_test[-1][-n_steps:]
    last_sequence = last_sequence.reshape((1, last_sequence.shape[0])) # reshape to match the size of last_sequence
    for i in range(n_steps):
        rf_prediction = rf_model.predict(last_sequence.reshape((1, last_sequence.shape[1])))
        rf_predictions.append(rf_prediction[0])
        last_sequence = np.append(last_sequence[:,1:], rf_prediction.reshape((1,1)), axis=1)

    print(rf_predictions)
    print(last_sequence)
    return last_sequence



