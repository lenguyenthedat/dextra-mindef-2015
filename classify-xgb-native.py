import pandas as pd
import datetime
import csv
import numpy as np
import os
import scipy as sp
import xgboost as xgb
import itertools
import operator
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import cross_validation
from matplotlib import pylab as plt

sample = True
plot = True

goal = 'RESIGNED'
myid = 'PERID'

def entropyloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    el = sum(act*sp.log10(pred) + sp.subtract(1,act)*sp.log10(sp.subtract(1,pred)))
    el = el * -1.0/len(act)
    return el

# http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] # mode
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], # mean
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

# LOAD DATA
features = ['GENDER','COUNTRY_OF_BIRTH','NATIONALITY','AGE','AGE_GROUPING','MARITAL_STATUS','RANK_GRADE','RANK_GROUPING',
            'YEARS_IN_GRADE','EMPLOYEE_GROUP','PARENT_SERVICE','SERVICE_SUB_AREA','SERVICE_TYPE','YEARS_OF_SERVICE',
            'VOC','UNIT','NO_OF_KIDS','MIN_CHILD_AGE','AVE_CHILD_AGE','HSP_ESTABLISHMENT','HSP_CERTIFICATE','HSP_CERT_RANK',
            'HSP_CERT_DESC','UPGRADED_LAST_3_YRS','UPGRADED_CERT_3_YRS','UPGRADED_CERT_DESC_3_YRS','MARRIED_WITHIN_2_YEARS',
            'DIVORCE_WITHIN_2_YEARS','DIVORCE_REMARRIED_WITHIN_2_YEARS','PROMO_LAST_5_YRS','PROMO_LAST_4_YRS','PROMO_LAST_3_YRS',
            'PROMO_LAST_2_YRS','PROMO_LAST_1_YR','UNIT_CHG_LAST_3_YRS','UNIT_CHG_LAST_2_YRS','UNIT_CHG_LAST_1_YR','AWARDS_RECEIVED',
            'HOUSING_TYPE','HOUSING_GROUP','HOUSING_RANK','PREV_HOUSING_TYPE','MOVE_HOUSE_T_2','HOUSE_UPG_DGRD','IPPT_SCORE',
            'PES_SCORE','HOMETOWORKDIST','SVC_INJURY_TYPE','TOT_PERC_INC_LAST_1_YR','BAS_PERC_INC_LAST_1_YR']
features_non_numeric = ['GENDER','COUNTRY_OF_BIRTH','NATIONALITY','AGE_GROUPING','MARITAL_STATUS','RANK_GRADE','RANK_GROUPING',
            'EMPLOYEE_GROUP','PARENT_SERVICE','SERVICE_SUB_AREA','SERVICE_TYPE',
            'VOC','UNIT','HSP_ESTABLISHMENT','HSP_CERTIFICATE',
            'HSP_CERT_DESC','UPGRADED_LAST_3_YRS','UPGRADED_CERT_3_YRS','UPGRADED_CERT_DESC_3_YRS','MARRIED_WITHIN_2_YEARS',
            'DIVORCE_WITHIN_2_YEARS','DIVORCE_REMARRIED_WITHIN_2_YEARS','UNIT_CHG_LAST_3_YRS','UNIT_CHG_LAST_2_YRS','UNIT_CHG_LAST_1_YR',
            'HOUSING_TYPE','HOUSING_GROUP','PREV_HOUSING_TYPE','MOVE_HOUSE_T_2','SVC_INJURY_TYPE']

noisy_features = ['RANK_GRADE','RANK_GROUPING','COUNTRY_OF_BIRTH','DIVORCE_WITHIN_2_YEARS',
                  'DIVORCE_REMARRIED_WITHIN_2_YEARS', 'UPGRADED_LAST_3_YRS', 'MARRIED_WITHIN_2_YEARS', 'MOVE_HOUSE_T_2',
                  'PROMO_LAST_5_YRS','PROMO_LAST_4_YRS','PROMO_LAST_3_YRS', 'PROMO_LAST_2_YRS','PROMO_LAST_1_YR']
features = [c for c in features if c not in noisy_features]
features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]

# Load data
train = pd.read_csv('./data/20150803115609-HR_Retention_2013_training.csv')
test = pd.read_csv('./data/20150803115608-HR_Retention_2013_to_be_predicted.csv')

# # FEATURE ENGINEERING
# # Gender and promotion
train['promo1_gender'] = train['PROMO_LAST_1_YR'].map(str) + train['GENDER']
test['promo1_gender'] = test['PROMO_LAST_1_YR'].map(str) + test['GENDER']
features = features + ['promo1_gender']
features_non_numeric = features_non_numeric + ['promo1_gender']

train['promo2_gender'] = train['PROMO_LAST_2_YRS'].map(str) + train['GENDER']
test['promo2_gender'] = test['PROMO_LAST_2_YRS'].map(str) + test['GENDER']
features = features + ['promo2_gender']
features_non_numeric = features_non_numeric + ['promo2_gender']

train['promo3_gender'] = train['PROMO_LAST_3_YRS'].map(str) + train['GENDER']
test['promo3_gender'] = test['PROMO_LAST_3_YRS'].map(str) + test['GENDER']
features = features + ['promo3_gender']
features_non_numeric = features_non_numeric + ['promo3_gender']

train['promo4_gender'] = train['PROMO_LAST_4_YRS'].map(str) + train['GENDER']
test['promo4_gender'] = test['PROMO_LAST_4_YRS'].map(str) + test['GENDER']
features = features + ['promo4_gender']
features_non_numeric = features_non_numeric + ['promo4_gender']

train['promo5_gender'] = train['PROMO_LAST_5_YRS'].map(str) + train['GENDER']
test['promo5_gender'] = test['PROMO_LAST_5_YRS'].map(str) + test['GENDER']
features = features + ['promo5_gender']
features_non_numeric = features_non_numeric + ['promo5_gender']

# # Age and Gender
train['age_gender'] = train['GENDER'].map(str) + train['AGE_GROUPING']
test['age_gender'] = test['GENDER'].map(str) + test['AGE_GROUPING']
features = features + ['age_gender']
features_non_numeric = features_non_numeric + ['age_gender']

# # Rank grouping
train['Rank_1'] = train['RANK_GROUPING'].apply(lambda x: x.split(' ')[0])
train['Rank_2'] = train['RANK_GROUPING'].apply(lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 else '')
test['Rank_1'] = test['RANK_GROUPING'].apply(lambda x: x.split(' ')[0])
test['Rank_2'] = test['RANK_GROUPING'].apply(lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 else '')
features = features + ['Rank_1', 'Rank_2']
features_non_numeric = features_non_numeric + ['Rank_1', 'Rank_2']

# # Salary increment. Max to set = 101. It doesn't matter anyone getting more than this or not.
train['TOT_PERC_INC_LAST_1_YR'] = train['TOT_PERC_INC_LAST_1_YR'].apply(lambda x: 101 if x > 101 else x)
test['TOT_PERC_INC_LAST_1_YR'] = test['TOT_PERC_INC_LAST_1_YR'].apply(lambda x: 101 if x > 101 else x)
train['BAS_PERC_INC_LAST_1_YR'] = train['BAS_PERC_INC_LAST_1_YR'].apply(lambda x: 101 if x > 101 else x)
test['BAS_PERC_INC_LAST_1_YR'] = test['BAS_PERC_INC_LAST_1_YR'].apply(lambda x: 101 if x > 101 else x)

# train['HOMETOWORKDIST'] = train['HOMETOWORKDIST'].apply(lambda x: 18 if x == 0 else x)
# test['HOMETOWORKDIST'] = test['HOMETOWORKDIST'].apply(lambda x: 18 if x == 0 else x)

# # These are yes / no columns which might contain NaN that doesn't have a significant propotion of yes or no
# for col in ['UNIT_CHG_LAST_3_YRS','UNIT_CHG_LAST_2_YRS','UNIT_CHG_LAST_1_YR','MOVE_HOUSE_T_2','UPGRADED_LAST_3_YRS']:
#     train[col] = train[col].fillna('UNKNOWN')
#     test[col] = test[col].fillna('UNKNOWN')

# # SVC Injury Type
# train['SVC_INJURY_TYPE'] = train['SVC_INJURY_TYPE'].fillna(-1)
# test['SVC_INJURY_TYPE'] = test['SVC_INJURY_TYPE'].fillna(-1)

# # HSP_ESTABLISHMENT
# train['HSP_ESTABLISHMENT'] = train['HSP_ESTABLISHMENT'].fillna('NONE')
# test['HSP_ESTABLISHMENT'] = test['HSP_ESTABLISHMENT'].fillna('NONE')

# # HSP_CERTIFICATE
# train['HSP_CERTIFICATE'] = train['HSP_CERTIFICATE'].fillna('NONE')
# test['HSP_CERTIFICATE'] = test['HSP_CERTIFICATE'].fillna('NONE')

# # UPGRADED_CERT_DESC_3_YRS - this has too many values

# # HOUSING_TYPE
# train['HOUSING_TYPE'] = train['HOUSING_TYPE'].fillna('NONE')
# test['HOUSING_TYPE'] = test['HOUSING_TYPE'].fillna('NONE')

# # HOUSING_GROUP
# train['HOUSING_GROUP'] = train['HOUSING_GROUP'].fillna('NONE')
# test['HOUSING_GROUP'] = test['HOUSING_GROUP'].fillna('NONE')

# # PREV_HOUSING_TYPE
# train['PREV_HOUSING_TYPE'] = train['PREV_HOUSING_TYPE'].fillna('UNKNOWN')
# test['PREV_HOUSING_TYPE'] = test['PREV_HOUSING_TYPE'].fillna('UNKNOWN')

# Fill NA
train = DataFrameImputer().fit_transform(train)
test = DataFrameImputer().fit_transform(test)

# Pre-processing non-number values
le = LabelEncoder()
for col in features_non_numeric:
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in set(features) - set(features_non_numeric) - \
  set(['TOT_PERC_INC_LAST_1_YR','BAS_PERC_INC_LAST_1_YR','AGE',
       'YEARS_IN_GRADE', 'YEARS_OF_SERVICE', 'NO_OF_KIDS', 'MIN_CHILD_AGE',
       'AVE_CHILD_AGE','HSP_CERT_RANK','HOUSING_RANK']):
    scaler.fit(list(train[col])+list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

# XGB Params
# params = {'max_depth':8, 'eta':0.05, 'silent':1,
#           'objective':'multi:softprob', 'num_class':2, 'eval_metric':'mlogloss',
#           'min_child_weight':3, 'subsample':1,'colsample_bytree':0.6, 'nthread':4}
# num_rounds = 180
params = {'max_depth':6, 'eta':0.01, 'silent':1,
          'objective':'multi:softprob', 'num_class':2, 'eval_metric':'mlogloss',
          'min_child_weight':3, 'subsample':1,'colsample_bytree':0.55, 'nthread':4}
num_rounds = 990


# TRAINING / GRIDSEARCH
if sample:
    cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True, indices=False, random_state=1337)
    results = []
    for traincv, testcv in cv:
        xgbtrain = xgb.DMatrix(train[traincv][list(features)], label=train[traincv][goal])
        classifier = xgb.train(params, xgbtrain, num_rounds)
        score = entropyloss(train[testcv][goal].values, np.compress([False, True],\
            classifier.predict(xgb.DMatrix(train[testcv][features])), axis=1).flatten())
        print score
        results.append(score)
    print "Results: " + str(results)
    print "Mean: " + str(np.array(results).mean())

# EVAL OR EXPORT
if not sample: # Export result
    print str(datetime.datetime.now())
    xgbtrain = xgb.DMatrix(train[features], label=train[goal])
    classifier = xgb.train(params, xgbtrain, num_rounds)
    if not os.path.exists('result/'):
        os.makedirs('result/')
    csvfile = 'result/' + classifier.__class__.__name__ + '-submit.csv'
    with open(csvfile, 'w') as output:
        predictions = []
        print str(datetime.datetime.now())
        for i in test[myid].tolist():
          # National ServiceMen always resigned
          if test[test[myid] == i]['EMPLOYEE_GROUP'].item() == 2:
            predictions += [[i,1]]
          else:
            predictions += [[i,classifier.predict(xgb.DMatrix(test[test[myid]==i][features])).tolist()[0][1]]]
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow([myid,goal])
        writer.writerows(predictions)
        print str(datetime.datetime.now())
    # Feature importance
    if plot:
      outfile = open('xgb.fmap', 'w')
      i = 0
      for feat in features:
          outfile.write('{0}\t{1}\tq\n'.format(i, feat))
          i = i + 1
      outfile.close()
      importance = classifier.get_fscore(fmap='xgb.fmap')
      importance = sorted(importance.items(), key=operator.itemgetter(1))
      df = pd.DataFrame(importance, columns=['feature', 'fscore'])
      df['fscore'] = df['fscore'] / df['fscore'].sum()
      # Plotitup
      plt.figure()
      df.plot()
      df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
      plt.title('XGBoost Feature Importance')
      plt.xlabel('relative importance')
      plt.gcf().savefig('Feature_Importance_xgb.png')
