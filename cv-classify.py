import pandas as pd
import time
import csv
import numpy as np
import os
import scipy as sp
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sknn.mlp import Classifier, Layer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.base import TransformerMixin
from sklearn import cross_validation


def entropyloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    el = sum(act*sp.log10(pred) + sp.subtract(1,act)*sp.log10(sp.subtract(1,pred)))
    el = el * -1.0/len(act)
    return el

el_scorer = make_scorer(entropyloss, greater_is_better = False)

# http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        # self.fill = pd.Series([X[c].value_counts().index[0]
        #     if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
        #     index=X.columns)
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        # Treat N/A uniquely
        # self.fill = pd.Series(['-1'
        #     if X[c].dtype == np.dtype('O') else -1 for c in X],
        #     index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

features = ['GENDER','COUNTRY_OF_BIRTH','NATIONALITY','AGE',
            'YEARS_IN_GRADE','EMPLOYEE_GROUP','PARENT_SERVICE','SERVICE_SUB_AREA','SERVICE_TYPE','YEARS_OF_SERVICE',
            'VOC','UNIT','NO_OF_KIDS','MIN_CHILD_AGE','AVE_CHILD_AGE','HSP_ESTABLISHMENT','HSP_CERTIFICATE','HSP_CERT_RANK',
            'HSP_CERT_DESC','UPGRADED_LAST_3_YRS','UPGRADED_CERT_3_YRS','UPGRADED_CERT_DESC_3_YRS','MARRIED_WITHIN_2_YEARS',
            'DIVORCE_WITHIN_2_YEARS','DIVORCE_REMARRIED_WITHIN_2_YEARS','PROMO_LAST_5_YRS','PROMO_LAST_4_YRS','PROMO_LAST_3_YRS',
            'PROMO_LAST_2_YRS','PROMO_LAST_1_YR','UNIT_CHG_LAST_3_YRS','UNIT_CHG_LAST_2_YRS','UNIT_CHG_LAST_1_YR','AWARDS_RECEIVED',
            'HOUSING_TYPE','HOUSING_GROUP','HOUSING_RANK','PREV_HOUSING_TYPE','MOVE_HOUSE_T_2','HOUSE_UPG_DGRD','IPPT_SCORE',
            'PES_SCORE','HOMETOWORKDIST','SVC_INJURY_TYPE','TOT_PERC_INC_LAST_1_YR','BAS_PERC_INC_LAST_1_YR']
features_non_numeric = ['GENDER','COUNTRY_OF_BIRTH','NATIONALITY',
            'EMPLOYEE_GROUP','PARENT_SERVICE','SERVICE_SUB_AREA','SERVICE_TYPE',
            'VOC','UNIT','HSP_ESTABLISHMENT','HSP_CERTIFICATE',
            'HSP_CERT_DESC','UPGRADED_LAST_3_YRS','UPGRADED_CERT_3_YRS','UPGRADED_CERT_DESC_3_YRS','MARRIED_WITHIN_2_YEARS',
            'DIVORCE_WITHIN_2_YEARS','DIVORCE_REMARRIED_WITHIN_2_YEARS','UNIT_CHG_LAST_3_YRS','UNIT_CHG_LAST_2_YRS','UNIT_CHG_LAST_1_YR',
            'HOUSING_TYPE','HOUSING_GROUP','PREV_HOUSING_TYPE','MOVE_HOUSE_T_2']

goal = 'RESIGNED'
myid = 'PERID'

train = pd.read_csv('./data/20150803115609-HR_Retention_2013_training.csv')

train = DataFrameImputer().fit_transform(train)

# Pre-processing non-number values
le = LabelEncoder()
for col in features_non_numeric:
    le.fit(list(train[col]))
    train[col] = le.transform(train[col])

# Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in set(features)-set(features_non_numeric):
    scaler.fit(list(train[col]))
    train[col] = scaler.transform(train[col])

cv = cross_validation.KFold(len(train), n_folds=5, indices=False)

# Define classifiers
classifiers = [
    ExtraTreesClassifier(n_estimators=1024, max_features=None,
                       oob_score=False, bootstrap=True, min_samples_leaf=1,
                       min_samples_split=2, max_depth=19),
    RandomForestClassifier(n_estimators=1024, max_features=23,
                       oob_score=False, bootstrap=True, min_samples_leaf=1,
                       min_samples_split=2, max_depth=32),
    XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7)
]

# Train
for classifier in classifiers:
    print classifier.__class__.__name__
    start = time.time()
    results = []
    for traincv, testcv in cv:
        classifier.fit(np.array(train[traincv][list(features)]), train[traincv][goal])
        results.append( entropyloss(train[testcv][goal].values, np.compress([False, True],\
            classifier.predict_proba(np.array(train[testcv][features])), axis=1).flatten()) ) 
    print "Results: " + str( np.array(results).mean() )
