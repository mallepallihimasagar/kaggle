
# coding: utf-8

# In[101]:

import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from sklearn import *
import sklearn
import numpy as np
import xgboost as xgb


# In[103]:

#help("modules")


# ### Reading Datasets

# In[14]:

tr_variants_df = pd.read_csv("training_variants", index_col=False )
tst_variants_df = pd.read_csv("test_variants", index_col=False )


# In[6]:

tr_text_df = pd.read_csv("training_text", delimiter= "\|\|", skiprows=1, 
                   engine = 'python', header = None, names = ["ID","Text"])


# In[16]:

tst_text_df = pd.read_csv("test_text", delimiter= "\|\|", skiprows=1, 
                   engine = 'python', header = None, names = ["ID","Text"])


# In[7]:

tr_text_df.head()


# In[8]:

tr_variants_df.head()


# ### Ploting data

# In[ ]:

plt.figure(1)
plt.subplot(221)
g = sns.countplot(x="Class", data=tr_variants_df)
plt.subplot(222)
sns.countplot(x="Gene", data=tr_variants_df)
plt.show()


# ### Merging datasets

# In[10]:

train_df = pd.merge(tr_variants_df, tr_text_df, how='left', on='ID').fillna('')


# In[17]:

test_df = pd.merge(tst_variants_df, tst_text_df, how='left', on='ID').fillna('')


# In[18]:

train_df.head()


# In[19]:

test_df.head()


# In[114]:

y = train_df['Class'].values
train = train_df.drop(['Class'], axis=1)


# In[115]:

y


# In[116]:

pid = test_df['ID'].values


# In[117]:

df_all = pd.concat((train, test_df), axis=0, ignore_index=True)


# In[118]:

df_all.head()


# In[119]:

df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') 
                                                   if w in r['Text'].split(' ')]), axis=1)


# In[120]:

df_all.head(50)


# In[48]:

df_all["Text"][41]


# In[57]:

df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].upper().split(' ')
                                                        if w in r['Text'].upper().split(' ')]), axis=1)


# In[58]:

df_all.head(50)


# In[60]:

df_all["Text"][23]


# In[121]:

gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print(len(gen_var_lst))


# In[67]:

gen_var_lst[1:10]


# In[122]:

gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
print(len(gen_var_lst))


# In[70]:

gen_var_lst[1:10]


# In[123]:

i_ = 0
#commented for Kaggle Limits
#for gen_var_lst_itm in gen_var_lst:
#    if i_ % 100 == 0: print(i_)
#    df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
#    i_ += 1

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]

class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)



# In[124]:

y


# In[125]:

print('Pipeline...')
fp = pipeline.Pipeline([ #Setting paramenters
    ('union', pipeline.FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            #commented for Kaggle Limits
            #('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))
        ])
    )])

train = fp.fit_transform(train); print(train.shape)
test = fp.transform(test); print(test.shape)



# In[126]:

pipeline.Pipeline([('Gene', cust_txt_col('Gene')),
                   ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                   ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])


# In[127]:

np.unique(y)


# In[128]:

y = y - 1 #fix for zero bound array

denom = 0
fold = 1 #Change to 5, 1 for Kaggle Limits
for i in range(fold):
    params = {
        'eta': 0.03333,
        'max_depth': 4,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print(score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[97]:




# In[129]:

preds /= denom
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission_xgb.csv', index=False)


# In[130]:

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=model); plt.show()


# In[81]:



