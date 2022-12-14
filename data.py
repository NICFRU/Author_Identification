# %%
import pandas as pd

# %%
df = pd.read_csv("train_500.csv",index_col="Unnamed: 0")

#%%
df.columns

#%%
df = df.drop(columns=['date', 'lang', 'language_2', 'word_tokenize',
       'sent_tokenize', 'count_word', 'count_sent',
       'word_tokenize_num_of_stopwords', 'word_tokenize_without_stopwords',
       'word_tokenize_without_stopwords_port', 'msg_lemmatized', 'cleanLinks',
       'laenge_saetze', 'word_per_sent_mean', 'gender', 'topic',
       'sign'])
# %%
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)

# %%
#train.to_csv("train.csv")
#test.to_csv("test.csv")