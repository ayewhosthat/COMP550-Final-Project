#!/usr/bin/env python
# coding: utf-8

# In[23]:


import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import seaborn as sns
from collections import Counter


# In[5]:


df = pd.read_csv('final.csv', index_col=0)


# In[6]:


df


# In[7]:


df['subreddit'].value_counts().head(5)
# explore 5 most popular subreddits in depth


# In[8]:


top5 = ['AskReddit', 'politics', 'worldnews', 'leagueoflegends', 'pcmasterrace']
top5_filter = df['subreddit'].isin(top5)
subset = df.loc[top5_filter]


# In[9]:


subset


# ## Class/Label Distribution by Subreddit

# In[10]:


class_distributions = subset.groupby(by=['subreddit', 'label']).size()


# In[11]:


sns.countplot(data=subset, x='subreddit', hue='label')
plt.title('Label Distribution among Top 5 Most common Subreddits')
plt.xticks(rotation=75)


# ### Insert discussion/analysis here

# In[ ]:





# ## Average comment length by subreddit

# In[12]:


sns.barplot(data=subset, x='subreddit', y='comment length', estimator='mean')
plt.xticks(rotation=75)
plt.title('Average comment length by subreddit')
plt.ylabel('Average comment length')


# In[ ]:





# ## Average comment length by subreddit & label

# In[13]:


sns.barplot(data=subset, x='subreddit', y='comment length', hue='label', estimator='mean')
plt.title('Average comment length by subreddit + label')
plt.xticks(rotation=75)
plt.ylabel('average comment length')


# ## Average compound sentiment score difference b/w comment and parent comment by subreddit

# In[14]:


sns.barplot(data=subset, x='subreddit', y='compound difference', estimator='mean')
plt.xticks(rotation=75)
plt.title('Average compound sentiment difference by subreddit')
plt.ylabel('Average compound sentiment difference')


# In[ ]:





# ## Average comment polarity different by subreddit + label

# In[15]:


sns.barplot(data=subset, x='subreddit', y='compound difference', hue='label', estimator='mean')
plt.title('Average compound sentiment score difference by subreddit + label')
plt.xticks(rotation=75)
plt.ylabel('Average compound sentiment score difference')


# ### Most common unigram/word

# In[17]:


sarcastic_filter = subset['label'] == 1
sarcastic = subset.loc[sarcastic_filter]


# In[21]:


stopwords_eng = stopwords.words('english')
word_counts = Counter()
for i, row in sarcastic.iterrows():
    text = row['tokenized']
    text = text[1:-1].replace("'", '').split(', ')
    valid = [word.lower() for word in text if word.lower() not in stopwords_eng]
    word_counts.update(valid)


# In[22]:


word_counts


# ## Most common bigram

# In[28]:


bigram_counts = Counter()
for j, row in sarcastic.iterrows():
    text = row['tokenized'][1:-1].replace("'", '').split(', ')
    text = [word.lower() for word in text if word.lower() not in stopwords_eng]
    bigrams = zip(text, text[1:])
    bigram_counts.update(bigrams)


# In[29]:


bigram_counts


# In[ ]:




