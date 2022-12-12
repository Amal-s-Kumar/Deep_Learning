#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv(r'C:\Users\91623\Downloads\COVIDSenti-A.csv')


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.columns


# In[8]:


data.describe()


# In[9]:


data.label.value_counts()


# In[10]:


data[data['label']=='neg']


# In[11]:


data[data['label']=='pos']


# In[12]:


data[data['label']=='neu']


# In[13]:


data[data['label']=='neu'].loc[17,'tweet']


# In[14]:


data[data['label']=='neg'].loc[16,'tweet']


# In[15]:


data[data['label']=='pos'].loc[70,'tweet']


# In[16]:


data.isnull().sum()


# In[17]:


import re 
import numpy as np


# In[18]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt


# In[19]:


# create new column with removed @user
data['tweet'] = np.vectorize(remove_pattern)(data['tweet'], '@[\w]*')


# In[20]:


# to remove HTTP and urls from tweets
data['tweet'] = data['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])


# In[21]:


# remove special characters, numbers, punctuations
data['tweet'] = data['tweet'].str.replace('[^a-zA-Z#]+',' ')


# In[22]:


data['tweet'] = data['tweet'].str.replace('#',' ')


# In[23]:


# Making all the words in lower case
data['tweet']=data['tweet'].str.lower() 


# In[24]:


data.head()


# In[25]:


# remove short words
data['tweet'] = data['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))


# In[26]:


data.head()


# In[27]:


data[data['label']=='neu'].loc[17,'tweet']


# In[28]:


# create new variable tokenized tweet 
tokenized_tweet = data['tweet'].apply(lambda x: x.split())


# In[29]:


#Importing required resources
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[30]:


# import these modules
from nltk.stem import WordNetLemmatizer
  
lemmatizer = WordNetLemmatizer()
  
# apply lemmatizer for tokenized_tweet
tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])


# In[31]:


tokenized_tweet


# In[32]:


# join tokens into one sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
# change df['Tweet'] to tokenized_tweet


# In[33]:


data['Tweet']  = tokenized_tweet


# In[34]:


data.drop('tweet',axis=1,inplace=True)


# In[35]:


data.head()


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x="label", data=data)
plt.show()


# In[38]:


X = data['Tweet']
y = data['label']


# In[39]:


# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 0)


# In[40]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(max_features = 5000)
tfidf_vect.fit(data['Tweet'])
X_train_tfidf = tfidf_vect.transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)


# In[41]:


print(X_train_tfidf)


# In[42]:


print(X_test_tfidf)


# In[43]:


print(tfidf_vect.vocabulary_)


# In[44]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay,classification_report
from sklearn.preprocessing import LabelBinarizer


# In[45]:


svm_model = SVC(probability = True, kernel = 'linear')
svm_model.fit(X_train_tfidf, y_train )


# In[46]:


svm_predictions = svm_model.predict(X_test_tfidf)
Predicted_data = pd.DataFrame()
Predicted_data['Tweet'] = X_test
Predicted_data['Label'] = svm_predictions
Predicted_data


# In[47]:


Predicted_data['Label'].value_counts()


# In[48]:


ConfusionMatrixDisplay.from_predictions(y_test, svm_predictions)


# In[49]:


svm_accuracy = accuracy_score(svm_predictions, y_test)*100
svm_accuracy


# In[50]:


print("Classification Report:")
print(classification_report(y_test, svm_predictions))


# In[ ]:




