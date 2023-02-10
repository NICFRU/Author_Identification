#%%
import pandas as pd
# %%
df1 = pd.read_csv("df_tokenized_port_2.csv")
#%%
df1
# %%
df1 = df1.drop(columns=['Unnamed: 0', 'id', 'gender', 'topic', 'sign', 'date', 'text',
       'lang', 'language_2', 'word_tokenize', 'count_word',
       'count_sent', 'word_tokenize_num_of_stopwords',
       'word_tokenize_without_stopwords',
       'word_tokenize_without_stopwords_port'])
# %%
df1 = df1.dropna()
#%%
import ast
def str_list_str(x):
    r = ast.literal_eval(x)[0]
    return r
#%%
df1.sent_tokenize = df1.apply(lambda row: str_list_str(row["sent_tokenize"]), axis=1)
#%%
df1 = df1.drop_duplicates()
# %%
df1.rename(columns = {'age':'labels','sent_tokenize':'text'}, inplace = True)
# %%
for ind in df1.index:
        if df1.labels[ind] < 15:
            df1.labels[ind] = 0
        elif df1.labels[ind] < 20 and df1.labels[ind] >= 15:
            df1.labels[ind] = 1
        elif df1.labels[ind] < 25 and df1.labels[ind] >= 20:
            df1.labels[ind] = 2
        elif df1.labels[ind] < 30 and df1.labels[ind] >= 25:
            df1.labels[ind] = 3
        elif df1.labels[ind] < 35 and df1.labels[ind] >= 30:
            df1.labels[ind] = 4
        elif df1.labels[ind] < 40 and df1.labels[ind] >= 35:
            df1.labels[ind] = 5
        elif df1.labels[ind] < 45 and df1.labels[ind] >= 40:
            df1.labels[ind] = 6
        elif df1.labels[ind] >= 45:
            df1.labels[ind] = 7

#%%
#df1.to_csv("data.csv",index=False)

#%%
import pandas as pd
#df1 = pd.read_csv("data.csv")
#%%
df1 = df1.drop(index = df1.loc[df1['text'].str.contains("http", na=True)].index).reset_index(drop=True)
#%%
df1 = df1.drop(index = df1.loc[df1['text'].str.contains("urlLink", na=True)].index).reset_index(drop=True)
df1 = df1.drop(index = df1.loc[df1['text'].str.contains("urllink", na=True)].index).reset_index(drop=True)
#%%
df1 = df1.drop(index = df1.loc[df1['text'].str.contains("&nbsp;", na=True)].index).reset_index(drop=True)
#%%
#df1.to_csv("data.csv",index=False)
#%%
#import pandas as pd
#df1 = pd.read_csv("data.csv")

#%%
df1 = df1.drop(index = df1.loc[df1["text"].apply(lambda n: len(n.split()))<6].index).reset_index(drop=True)

#%%
df1 = df1.drop(index = df1.loc[df1['text'].str.contains(".org", na=True)].index).reset_index(drop=True)

#%%
import pandas as pd
#df1 = pd.read_csv('data.csv')
#%%
df1.labels.value_counts()
#%%
df1.labels.value_counts().sort_index().plot(kind='bar')
#%%
df1.labels.value_counts()
#%%
df = df1.groupby('labels').head(8100)
indexlist = df.index
df = df.reset_index(drop=True)
# %%
dfvali = df.groupby('labels').head(8100*0.2)
# %%
dftrain = df.drop(index=dfvali.index).reset_index(drop=True)
dfvali = dfvali.reset_index(drop=True)
#%%
dftest = df1.drop(index=indexlist).reset_index(drop=True)
#%%
dftest = dftest.groupby('labels').head(24).reset_index(drop=True)
# %%
dftrain.to_csv("train.csv",index=False)
dfvali.to_csv("vali.csv",index=False)
dftest.to_csv("test.csv",index=False)
#%%
df1.to_csv("data.csv",index=False)