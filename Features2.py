#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import swifter
import re
import string
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
pd.set_option('display.max_colwidth', None)


# In[2]:


tokenized_tagged = pd.read_csv('tokenized_tagged.csv', index_col=0)


# In[3]:


tokenized_tagged


# Now we want to remove all words that are tagged as proper nouns (NNP)

# In[4]:


# now add a feature column that indicates whether an emoticon is present in the comment
def contains_emoticon(text):
    # Define a regex pattern for common emoticons
    emoticon_pattern = r"""
        [:;=8xX]          # Eyes: :, ;, =, 8, x, X
        [-~^]?            # Optional nose: -, ~, ^
        [\)DpPoO/\|\\\(\[\]@#*$] # Mouth or other features
    """
    # Compile the pattern with re.VERBOSE for readability
    emoticon_regex = re.compile(emoticon_pattern, re.VERBOSE)

    # Search for the pattern in the text
    return bool(emoticon_regex.search(text))


# In[5]:


tokenized_tagged['contains emoticon'] = tokenized_tagged['comment'].apply(lambda comment: contains_emoticon(comment))


# In[6]:


tokenized_tagged


# In[7]:


tokenized_tagged[tokenized_tagged['contains emoticon'] == 1]


# In[8]:


intensity = SentimentIntensityAnalyzer()
tokenized_tagged['comment positive score'] = tokenized_tagged['comment'].apply(lambda comment: intensity.polarity_scores(comment)['pos'])





# In[ ]:


tokenized_tagged['comment negative score'] = tokenized_tagged['comment'].apply(lambda comment: intensity.polarity_scores(comment)['neg'])


# In[ ]:


tokenized_tagged['comment neutral score'] = tokenized_tagged['comment'].apply(lambda comment: intensity.polarity_scores(comment)['neu'])


# In[ ]:


tokenized_tagged['comment compound score'] = tokenized_tagged['comment'].apply(lambda comment: intensity.polarity_scores(comment)['compound'])


# In[ ]:


tokenized_tagged


# In[ ]:


tokenized_tagged['parent comment positive score'] = tokenized_tagged['parent_comment'].apply(lambda comment: intensity.polarity_scores(comment)['pos'])


# In[ ]:


tokenized_tagged['parent comment negative score'] = tokenized_tagged['parent_comment'].apply(lambda comment: intensity.polarity_scores(comment)['neg'])


# In[ ]:


tokenized_tagged['parent comment neutral score'] = tokenized_tagged['parent_comment'].apply(lambda comment: intensity.polarity_scores(comment)['neu'])


# In[ ]:


tokenized_tagged['parent comment compound score'] = tokenized_tagged['parent_comment'].apply(lambda comment: intensity.polarity_scores(comment)['compound'])


# In[ ]:


tokenized_tagged['compound difference'] = abs(tokenized_tagged['comment compound score'] - tokenized_tagged['parent comment compound score'])


# In[ ]:


tokenized_tagged


# In[ ]:





# In[19]:


tokenized_tagged.to_csv('all_features.csv')

