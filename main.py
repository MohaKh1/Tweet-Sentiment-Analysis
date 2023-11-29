import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from ast import literal_eval

# Load the dataset
file_path = 'elon_musk_model_filtered_and_original.csv'
elon_musk_tweets = pd.read_csv(file_path)

# Convert 'Datetime' to DateTime format and extract sentiment scores
elon_musk_tweets['Datetime'] = pd.to_datetime(elon_musk_tweets['Datetime'])

def extract_sentiment_score(sentiment_str):
    try:
        sentiment_list = literal_eval(sentiment_str)
        if sentiment_list and 'score' in sentiment_list[0]:
            return sentiment_list[0]['score']
    except:
        return None

elon_musk_tweets['original_sentiment_score'] = elon_musk_tweets['Sentiment'].apply(extract_sentiment_score)
elon_musk_tweets['filtered_sentiment_score'] = elon_musk_tweets['filtered_sentiment'].apply(lambda x: {'positive': 1, 'neutral': 0.5, 'negative': 0}.get(x, None))
elon_musk_tweets['model_sentiment_score'] = elon_musk_tweets['model_sentiment'].apply(lambda x: {'positive': 1, 'neutral': 0.5, 'negative': 0}.get(x, None))

# Sentiment trend analysis over time
sns.set(style="whitegrid")
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.lineplot(x='Datetime', y='original_sentiment_score', data=elon_musk_tweets)
plt.subplot(1, 3, 2)
sns.lineplot(x='Datetime', y='filtered_sentiment_score', data=elon_musk_tweets)
plt.subplot(1, 3, 3)
sns.lineplot(x='Datetime', y='model_sentiment_score', data=elon_musk_tweets)
plt.tight_layout()
plt.show()

# Word frequency analysis (word clouds)
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    return text

elon_musk_tweets['cleaned_text'] = elon_musk_tweets['Text'].apply(clean_text)
all_tweets = ' '.join(elon_musk_tweets['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_tweets)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for All Tweets')
plt.show()

def generate_wordcloud_for_sentiment(sentiment):
    filtered_tweets = elon_musk_tweets[elon_musk_tweets['filtered_sentiment'] == sentiment]['cleaned_text']
    combined_tweets = ' '.join(filtered_tweets)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_tweets)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment.capitalize()} Tweets')
    plt.show()

generate_wordcloud_for_sentiment('positive')
generate_wordcloud_for_sentiment('neutral')
generate_wordcloud_for_sentiment('negative')

# Overall sentiment distribution (pie chart)
sentiment_counts = elon_musk_tweets['filtered_sentiment'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Overall Sentiment Distribution in Tweets')
plt.show()
