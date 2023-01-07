# %%
import pandas as pd

# %%
df = pd.read_csv("train_500.csv",index_col="Unnamed: 0")

#%%
df.columns

#%%
df = df.drop(columns=['id','date', 'lang', 'language_2', 'word_tokenize',
       'sent_tokenize', 'count_word', 'count_sent',
       'word_tokenize_num_of_stopwords', 'word_tokenize_without_stopwords',
       'word_tokenize_without_stopwords_port', 'msg_lemmatized', 'cleanLinks',
       'laenge_saetze', 'word_per_sent_mean', 'gender', 'topic',
       'sign'])

df.rename(columns = {'age':'labels'}, inplace = True)
# %%
for ind in df.index:
       if df.labels[ind] < 20:
              df.labels[ind] = 0
       elif df.labels[ind] < 25 and df.labels[ind] >= 20:
              df.labels[ind] = 1
       elif df.labels[ind] < 30 and df.labels[ind] >= 25:
              df.labels[ind] = 2
       elif df.labels[ind] < 35 and df.labels[ind] >= 30:
              df.labels[ind] = 3
       elif df.labels[ind] < 40 and df.labels[ind] >= 35:
              df.labels[ind] = 4
       elif df.labels[ind] >= 40:
              df.labels[ind] = 5
df
# %%
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)

# %%
train.to_csv("train.csv",index = False,sep="°")
test.to_csv("test.csv",index = False,sep="°")
# %%
