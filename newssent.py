

from datetime import datetime, timedelta
import requests
import pandas as pd
def newssentiment(ticker):
    company = ticker  # Replace with the name of the company you're interested in
    api_key = 'chad2kpr01qhe0ra0vv0chad2kpr01qhe0ra0vvg'  # Get your API key from https://finnhub.io

    # Set the date range for the news articles you want to retrieve
    month_span=5
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


    # In[10]:


    news_data

    df.head()

    # Drop any summary duplicates
    df.drop_duplicates(subset=['summary'], inplace=True)

    # convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')

    # extract date from datetime
    df['datetime'] = df['datetime'].dt.date

    # Save the DataFrame to an Excel file
    df.to_csv('finance.csv')

    print(f"Successfully saved {len(df)} financial news articles to {company}_financial_news.xlsx")

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
    return tweet_df
