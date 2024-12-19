#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import matplotlib


# In[20]:


df = pd.read_csv('all_features.csv', index_col=0)


# In[21]:


df


# In[22]:


tokenized_tagged = pd.read_csv('tokenized_tagged.csv', index_col=0)


# In[23]:


tokenized_tagged


# In[24]:


df['tokenized'] = tokenized_tagged['tokenized'].astype(str)


# In[29]:


df['sticky caps'] = tokenized_tagged['sticky caps'].astype(int)


# In[26]:


# average word length
def compute_avg_word_length(sequence):
    return sum([len(word) for word in sequence])/len(sequence)

df['avg word length'] = df['comment'].apply(lambda comment: compute_avg_word_length(comment))


# In[34]:


df['comment length'] = df['comment'].apply(lambda comment: len(comment))


# In[35]:


df['parent comment length'] = df['parent_comment'].apply(lambda comment_par: len(comment_par))


# In[37]:


df['avg word length'] = df['tokenized'].apply(lambda seq: compute_avg_word_length(seq))


# In[ ]:





# In[38]:


df


# In[39]:


df.to_csv('final.csv')

