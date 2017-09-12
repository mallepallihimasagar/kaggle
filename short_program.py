import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
from sklearn import *
import sklearn
import numpy as np
import xgboost as xgb

tr_variants_df = pd.read_csv("training_variants", index_col=False)
tst_variants_df = pd.read_csv("test_variants", index_col=False)
tr_text_df = pd.read_csv("training_text", delimiter="\|\|", skiprows=1,
                         engine='python', header=None, names=["ID", "Text"])
tst_text_df = pd.read_csv("test_text", delimiter="\|\|", skiprows=1,
                          engine='python', header=None, names=["ID", "Text"])
train_df = pd.merge(tr_variants_df, tr_text_df, how='left', on='ID').fillna('')
test_df = pd.merge(tst_variants_df, tst_text_df, how='left', on='ID').fillna('')
y = train_df['Class'].values
print(np.unique(y))
train = train_df.drop(['Class'], axis=1)
pid = test_df['ID'].values
df_all = pd.concat((train, test_df), axis=0, ignore_index=True)
# df_all["Text"] = df_all["Text"].apply(lambda x: x[0:2000]) + df_all["Text"].apply(lambda x: x[len(x) - 1000:len(x)])
gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print(len(gen_var_lst))
train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]


class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = x.drop(['Gene', 'Variation', 'ID', 'Text'], axis=1).values
        return x


class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.key].apply(str)


print('Pipeline...')
fp = pipeline.Pipeline([  # Setting paramenters
    ('union', pipeline.FeatureUnion(
        n_jobs=-1,
        transformer_list=[('standard', cust_regression_vals()),
                          ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene',
                                                                                      feature_extraction.text.CountVectorizer
                                                                                      (analyzer=u'char',
                                                                                       ngram_range=(1, 8))),
                                                     ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25,
                                                                                          random_state=12))])),
                          ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')),
                                                     ('count_Variation', feature_extraction.text.CountVectorizer
                                                     (analyzer=u'char', ngram_range=(1, 8))),
                                                     ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25,
                                                                                          random_state=12))])),
                          # commented for Kaggle Limits
                          ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text',
                                                                                      feature_extraction.text.TfidfVectorizer(
                                                                                          ngram_range=(1, 3))),
                                                     ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25,
                                                                                          random_state=12))]))
                          ])
     )])

train = fp.fit_transform(train);
print(train.shape)
test = fp.transform(test);
print(test.shape)
pipeline.Pipeline([('Gene', cust_txt_col('Gene')),
                   ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                   ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])
y = y - 1
denom = 0
fold = 5  # Change to 5, 1 for Kaggle Limits
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
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000, watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit),
                              labels=list(range(9)))
    print(score1)
    # if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit + 80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit + 80)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class' + str(c + 1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb_fold_' + str(i) + '.csv', index=False)
    preds /= denom
submission = pd.DataFrame(preds, columns=['class' + str(c + 1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission_xgb.csv', index=False)
