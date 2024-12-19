#!/usr/bin/env python
# coding: utf-8

# In[81]:


import re
import pandas as pd
import string
import swifter
from nltk.tag import pos_tag
pd.set_option('display.max_colwidth', None)


# In[66]:


df = pd.read_csv('Sarcasm on Reddit dataset/train-balanced-sarcasm.csv')


# In[67]:


df


# In[68]:


df[df['comment'].isna() == 1]
# there are some null value comments that we should drop


# In[69]:


df.dropna(subset=['comment'], inplace=True)


# In[70]:


def remove_punctuation(text):
#     no_apostrophe = string.punctuation.replace("'", "")
    pattern = f"[{re.escape(string.punctuation)}]"
    return re.sub(pattern, '', text)


# In[71]:


df['no punctuation'] = df['comment'].apply(lambda comment: remove_punctuation(comment))


# In[72]:


df


# In[73]:


# now we tokenize/split
from nltk.tokenize import word_tokenize
df['tokenized'] = df['no punctuation'].apply(lambda comment: word_tokenize(comment))


# In[74]:


df


# In[75]:


def is_sticky_caps(text):
    # Filter out non-alphabetic characters
    filtered_chars = [char for char in text if char.isalpha()]

    # If there are fewer than 2 letters, we can't evaluate sticky caps
    if len(filtered_chars) < 2:
        return False

    # Check for sticky caps alternation
    for i in range(1, len(filtered_chars)):
        if filtered_chars[i].isupper() == filtered_chars[i - 1].isupper():
            return False  # Two consecutive letters have the same case

    return True


# In[76]:


df['sticky caps'] = df['no punctuation'].apply(lambda comment: is_sticky_caps(comment))


# In[77]:


df


# In[82]:


df['pos'] = df['tokenized'].swifter.apply(pos_tag)


# In[83]:


df


# In[84]:


df.to_csv('tokenized_tagged.csv')


# In[85]:


token_tag = pd.read_csv('tokenized_tagged.csv')


# In[86]:


token_tag


# In[ ]:




