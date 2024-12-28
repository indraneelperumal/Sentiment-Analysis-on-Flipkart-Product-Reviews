#!/usr/bin/env python
# coding: utf-8

# In[2]:


from time import sleep
from random import random
import pandas as pd
import requests
from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib.parse as urlparse
from urllib.parse import parse_qs

import warnings
warnings.filterwarnings('ignore')


# In[47]:


BASE_URL = 'https://www.flipkart.com/'
SEARCH_QUERY = "iphone14"
TOP_N_PRODUCTS = 10
REVIEW_PAGES_TO_SCRAPE_FROM_PER_PRODUCT = 100


# In[51]:


SAMPLE_URL = "https://www.flipkart.com/boat-rockerz-400-bluetooth-headset/product-reviews/itm14d0416b87d55?pid=ACCEJZXYKSG2T9GS&lid=LSTACCEJZXYKSG2T9GSVY4ZIC&marketplace=FLIPKART&page=1"
r = requests.get(SAMPLE_URL)    
soup = BeautifulSoup(r.content, 'html.parser') 
print(soup.prettify()[:500])


# In[85]:


SAMPLE_URL = str(input("Enter the reviews page URL of the product:", ))
r = requests.get(SAMPLE_URL)    
soup = BeautifulSoup(r.content, 'html.parser') 
print(soup.prettify()[:500])

rows = soup.find_all('div',attrs={'class':'col _2wzgFH K0kLPL'})
print(f"Count of rows(reviews):{len(rows)}\n\n\n")
df_ = pd.DataFrame(columns=['Product name', 'Rating', 'Summary', 'Review'])
product_name = soup.find('div',attrs={'class':'_2s4DIt _1CDdy2'})
product_name_text = product_name.text
for row in rows:
    
    sub_row = row.find_all('div',attrs={'class':'row'})
    rating = sub_row[0].find('div').text
    summary = sub_row[0].find('p').text
    review = sub_row[1].find_all('div')[1].text
    df_ = df_.append({'Rating': rating, 'Summary': summary, 'Review': review}, ignore_index=True)
    df_['Product name']=(product_name_text)


# In[86]:


df_


# In[103]:


df = pd.DataFrame(columns=['Product name', 'Rating', 'Summary', 'Review'])
df


# In[104]:


page_num = 1
url = "https://www.flipkart.com/apple-iphone-14-blue-128-gb/product-reviews/itmdb77f40da6b6d?pid=MOBGHWFHSV7GUFWA&lid=LSTMOBGHWFHSV7GUFWAC4ZPNA&marketplace=FLIPKART&page="
for page_num in range(900):
    r = requests.get(url+str(page_num))
    soup = BeautifulSoup(r.content, 'html.parser')
    rows = soup.find_all('div', attrs={'class':'col _2wzgFH K0kLPL'})
    
    for row in rows:
        sub_row = row.find_all('div', attrs={'class':'row'})
        rating = sub_row[0].find('div').text
        summary = sub_row[0].find('p').text
        review = sub_row[1].find_all('div')[1].text
        product_name = soup.find('div',attrs={'class':'_2s4DIt _1CDdy2'})
        df = df.append({'Product name': product_name, 'Rating': rating, 'Summary': summary, 'Review': review}, ignore_index=True)
        


# In[149]:


from wordcloud import WordCloud
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re


# In[125]:


nltk.downloader.download('vader_lexicon')


# In[129]:


sia = SentimentIntensityAnalyzer()


# In[142]:


def get_sentiment_score(review):
    
    review = review.lower()
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    
    return sia.polarity_scores(review)['compound']
df = df[df['Review'].apply(lambda x: len(x.split()) > 2)]
df['sentiment_score'] = df['Review'].apply(get_sentiment_score)
df['sentiment'] = df['sentiment_score'].apply(lambda score: 'Positive' if score > 0.2 else 'Negative' if score < -0.2 else 'Neutral')
average_sentiment = df['sentiment_score'].mean()

rating_5 = (average_sentiment + 1)*2.5


# In[143]:


print(average_sentiment)
print(rating_5)


# In[145]:


df['Rating'].astype(float).mean()


# In[146]:


df


# In[147]:


df.to_csv(r'C:\Users\Abhiram Pollur\Desktop\FlipkartReviews\Iphone14.csv')


# In[150]:


pos_reviews = ' '.join(df[df['sentiment_score'] > 0]['Review'])
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_reviews)


# In[152]:


neg_reviews = ' '.join(df[df['sentiment_score'] < 0]['Review'])
wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(neg_reviews)


# In[153]:


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title('Word Cloud for Positive reviews')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title('Word Cloud for Negative reviews')
plt.axis('off')

plt.show()


# In[ ]:




