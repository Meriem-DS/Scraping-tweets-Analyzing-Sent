import twint
# Importing pandas
import pandas as pd
#import nltk et ses librairies
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
nltk.download("stopwords")
stopwords = stopwords.words("english")
ps = nltk.PorterStemmer()

#Visualisation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#Configure search requierement :
c = twint.Config()
c.Search = "vaccin astrazeneca "  #subject of research
c.Lang = "en"  #Searching tweets which are only english language.
c.Pandas = True  #nable Pandas integration.
c.Limit = 2   #2tweets
c.Since = "2020-12-30"
c.Until = "2021-03-26"
#c.Username = "nom de compte twiter d'un user"

#Run
twint.run.Search(c)
#InFO
   #we can save our unstructed data in csv file
     #Stor it in csv FIle
     #c.Store_csv = True # tell Twint to save tweets in csv file
     #c.Output = "test" #name de fichier csv

#saving the results in pandas by writing my own function.
def columne_names():
  return twint.output.panda.Tweets_df.columns
#SWITCH twint to pandas :dataframe
def twint_to_pd(columns):
  return twint.output.panda.Tweets_df[columns]
#afficher tous les columns
data = twint_to_pd(['id', 'conversation_id', 'created_at', 'date', 'timezone','place', 'tweet', 'hashtags', 'cashtags', 'user_id', 'user_id_str', 'username', 'name', 'day', 'hour', 'link', 'retweet', 'nlikes', 'nreplies', 'nretweets', 'quote_url', 'search', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src', 'trans_dest']).sort_values('date', ascending=False)  # afficher resultat selon ordre déccroissant
#Dispaly rows
data.head()
#Save it as html file
data.to_html("Tweets.html")


      #Analysinz tweets with NLTK
#A-Clean Data
#1Removing punctuations /lowercase
data["tweet"] = data["tweet"].str.replace("[^a-zA-Z0-9]"," ")


#2:(optionel)removing any tweet that has less than 3 character.
data["tweet"] = data["tweet"].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))

#3:Tokenization and give us as output lowecase caracter
def tokenize(text):
  tokens = re.split("\W+", text)
  return tokens
data["tweet"] = data["tweet"].apply (lambda x: tokenize(x.lower()))


#Removing stopwords = les mots vides
def remove_stopword(text):
  text_nostopword= [char for char in text if char not in stopwords]
  return text_nostopword
data["tweet"] = data["tweet"].apply(lambda x: remove_stopword(x))


#4- Stemming / Lemmatize
#A stemmer will return the stem of a word, which needn’t be identical to the morphological root of the word.
# Lemmatisation, it will return the dictionary form of a word, which must be a valid word
#using stemmer as this is fast compared to lemmatize using Porter stemmer as this is most popular and commonly used stemmer.
def stem(tweet_no_stopword):
  text = [ps.stem ( word) for word in tweet_no_stopword]
  return text
data["tweet"] = data["tweet"].apply(lambda x: stem(x))

#B-Word frequency counts
#To count word frequency I need to first put the clean tweet into a list
data_list = data.loc[:,"tweet"].to_list()
#To put in a flat list I need to use
flat_data_list = [item for sublist in data_list for item in sublist]
#Now my tweet is in a list. I can count the word frequency
data_count= pd.DataFrame(flat_data_list)
data_count= data_count[0].value_counts()

#C-Visualise the result
data_count = data_count[:5, ]
plt.figure(figsize=(10, 5))
#type of our graph is a bar plot
sns.barplot(data_count.values, data_count.index, alpha=0.8)
plt.title("Top Words Overall")
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
#save the graph's photo
plt.savefig("Barplot.png")
plt.show()


















