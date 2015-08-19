import pandas as pd
import time
import csv
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
import scipy as sp
import warnings
warnings.filterwarnings("ignore")

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sknn.mlp import Classifier, Layer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.base import TransformerMixin
from scipy.stats import randint
from numpy.random import uniform
from sklearn import cross_validation

sample = True
gridsearch = False
randomsearch = False

goal = 'RESIGNED'
myid = 'PERID'

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

noisy_features = ['RANK_GRADE','RANK_GROUPING']
features = [c for c in features if c not in noisy_features]
features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]

# Load data
train = pd.read_csv('./data/20150803115609-HR_Retention_2013_training.csv')
test = pd.read_csv('./data/20150803115608-HR_Retention_2013_to_be_predicted.csv')

# # FEATURE ENGINEERING
# # Rank grouping
train['Rank_1'] = train['RANK_GROUPING'].apply(lambda x: x.split(' ')[0])
train['Rank_2'] = train['RANK_GROUPING'].apply(lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 else '')
test['Rank_1'] = test['RANK_GROUPING'].apply(lambda x: x.split(' ')[0])
test['Rank_2'] = test['RANK_GROUPING'].apply(lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 else '')
features = features + ['Rank_1', 'Rank_2']
features_non_numeric = features_non_numeric + ['Rank_1', 'Rank_2']

# # Salary increment. Max to set = 100. It doesn't matter anyone getting more than this or not.
train['TOT_PERC_INC_LAST_1_YR'] = train['TOT_PERC_INC_LAST_1_YR'].apply(lambda x: 100 if x > 100 else x)
test['TOT_PERC_INC_LAST_1_YR'] = test['TOT_PERC_INC_LAST_1_YR'].apply(lambda x: 100 if x > 100 else x)
train['BAS_PERC_INC_LAST_1_YR'] = train['BAS_PERC_INC_LAST_1_YR'].apply(lambda x: 100 if x > 100 else x)
test['BAS_PERC_INC_LAST_1_YR'] = test['BAS_PERC_INC_LAST_1_YR'].apply(lambda x: 100 if x > 100 else x)

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
for col in set(features)-set(features_non_numeric):
    scaler.fit(list(train[col])+list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

# CLASSIFIERS DEFINED
if sample:
    classifiers = [
        GradientBoostingClassifier(),
        XGBClassifier(),
        RandomForestClassifier(),
        ExtraTreesClassifier()
        # ExtraTreesClassifier(oob_score=True,bootstrap=True,min_samples_leaf=1,
        #                      n_estimators=432,min_samples_split=1,max_features=29,max_depth=16), #0.0204004223355
        # GradientBoostingClassifier(n_estimators=1000,max_depth=3,learning_rate=0.02), # 0.0165834057975
        # XGBClassifier(n_estimators=719,subsample=1,max_depth=9,min_child_weight=3,learning_rate=0.01), # 0.0153317811336
        # RandomForestClassifier(n_estimators=512, max_features=16,
        #                        oob_score=True, bootstrap=True, min_samples_leaf=1,
        #                        min_samples_split=2, max_depth=16), #0.0167591927224
    ]
else:
    classifiers = [# Other methods are underperformed yet take very long training time for this data set
        GradientBoostingClassifier(n_estimators=1000,max_depth=3,learning_rate=0.02), # 0.0165834057975
        RandomForestClassifier(n_estimators=512, max_features=16,
                               oob_score=True, bootstrap=True, min_samples_leaf=1,
                               min_samples_split=2, max_depth=16), #0.0167591927224
        XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7,learning_rate=0.1) # 0.0156365191636
    ]

# TRAINING / GRIDSEARCH
for classifier in classifiers:
    print classifier.__class__.__name__
    start = time.time()
    if (gridsearch & sample): # only do gridsearch if we run with sampled data.
        if (classifier.__class__.__name__ == "GradientBoostingClassifier"):
            print "Attempting GridSearchCV for GB model"
            gscv = GridSearchCV(classifier, {
                'max_depth': [2, 3, 5, 7, 8, 12, 16, 20, 32],
                'n_estimators': [256,512,1024],
                'learning_rate': [0.1, 0.02, 0.04],
                'subsample': [1, 0.5, 0.1, 0.25]},
                verbose=1, n_jobs=1, cv=3, scoring=el_scorer)
        if (classifier.__class__.__name__ == "XGBClassifier"):
            print "Attempting GridSearchCV for XGB model"
            gscv = GridSearchCV(classifier, {
                'max_depth': [12,16,20,24,28],
                'n_estimators': [256, 300, 365, 512],
                'min_child_weight': [1,3,5,7,9],
                'subsample': [1, 0.5, 0.1, 0.25],
                'learning_rate': [0.1, 0.05, 0.01]},
                verbose=1, n_jobs=1, cv=3, scoring=el_scorer)
        if (classifier.__class__.__name__ == "RandomForestClassifier"):
            print "Attempting GridSearchCV for RF model"
            gscv = GridSearchCV(classifier, {
                'max_depth': [12,16,20,24,28],
                'max_features' : [2, 8, 16, 24, 32],
                'min_samples_split': [1,2,4,6,8],
                'min_samples_leaf': [1,2,3,4],
                'n_estimators': [256, 512, 1024],
                'bootstrap':[True],
                'oob_score': [True,False]},
                verbose=1, n_jobs=1, cv=3,scoring=el_scorer)
        if (classifier.__class__.__name__ == "ExtraTreesClassifier"):
            print "Attempting GridSearchCV for ExtraTrees model"
            gscv = GridSearchCV(classifier, {
                'max_depth': [12,16,20,24,28],
                'max_features' : [2, 8, 16, 24, 32],
                'min_samples_split': [1,2,4,6,8],
                'min_samples_leaf': [1,2,3,4],
                'n_estimators': [256, 512, 1024],
                'bootstrap':[True],
                'oob_score': [True,False]},
                verbose=1, n_jobs=1, cv=3,scoring=el_scorer)
        classifier = gscv.fit(np.array(train[list(features)]), train[goal])
        print(classifier.best_score_)
        print(classifier.best_params_)
    elif (randomsearch & sample):
        if (classifier.__class__.__name__ == "GradientBoostingClassifier"):
            print "Attempting RandomizedSearchCV for GB model"
            rscv = RandomizedSearchCV(classifier, {
                'max_depth': randint(3, 32),
                'n_estimators': randint(256, 1024),
                'learning_rate': [0.1, 0.05, 0.01, 0.025],
                'subsample': [1, 0.5, 0.1, 0.25]},
                verbose=1, n_jobs=1, cv=3, scoring=el_scorer, n_iter=200)
        if (classifier.__class__.__name__ == "XGBClassifier"):
            print "Attempting RandomizedSearchCV for XGB model"
            rscv = RandomizedSearchCV(classifier, {
                'max_depth': randint(3, 32),
                'n_estimators': randint(256, 1024),
                'min_child_weight': randint(1, 9),
                'subsample': [1, 0.5, 0.1, 0.25],
                'learning_rate': [0.1, 0.05, 0.01, 0.025]},
                verbose=1, n_jobs=1, cv=3, scoring=el_scorer, n_iter=200)
        if (classifier.__class__.__name__ == "RandomForestClassifier"):
            print "Attempting RandomizedSearchCV for RF model"
            rscv = RandomizedSearchCV(classifier, {
                'max_depth': randint(3, 32),
                'criterion': ['gini', 'entropy'],
                'max_features' : randint(3, 32),
                'min_samples_split': randint(1, 12),
                'min_samples_leaf': randint(1, 6),
                'n_estimators': randint(256, 1024),
                'bootstrap':[True],
                'oob_score': [True,False]},
                verbose=1, n_jobs=1, cv=3,scoring=el_scorer, n_iter=200)
        if (classifier.__class__.__name__ == "ExtraTreesClassifier"):
            print "Attempting RandomizedSearchCV for ExtraTrees model"
            rscv = RandomizedSearchCV(classifier, {
                'max_depth': randint(3, 32),
                'max_features' : randint(3, 32),
                'min_samples_split': randint(1, 12),
                'min_samples_leaf': randint(1, 6),
                'n_estimators': randint(256, 1024),
                'bootstrap':[True],
                'oob_score': [True,False]},
                verbose=1, n_jobs=1, cv=3,scoring=el_scorer, n_iter=200)
        classifier = rscv.fit(np.array(train[list(features)]), train[goal])
        print(classifier.best_score_)
        print(classifier.best_params_)
    elif sample:
        # perform cross validation
        cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True, indices=False, random_state=1337)
        results = []
        for traincv, testcv in cv:
            classifier.fit(np.array(train[traincv][list(features)]), train[traincv][goal])
            score = entropyloss(train[testcv][goal].values, np.compress([False, True],\
                classifier.predict_proba(np.array(train[testcv][features])), axis=1).flatten())
            print score
            results.append(score) 
        print "Results: " + str( np.array(results).mean() )
    else:
        classifier.fit(np.array(train[list(features)]), train[goal])
    print '  -> Training time:', time.time() - start
    try:
        print classifier.feature_importances_
    except:
        pass

# EVAL OR EXPORT
if not sample: # Export result
    count = 0
    for classifier in classifiers:
        count += 1
        if not os.path.exists('result/'):
            os.makedirs('result/')
        predictions = classifier.predict_proba(np.array(test[features]))
        try: # try to flatten a list that might be flattenable.
            predictions = list(itertools.chain.from_iterable(predictions))
        except:
            pass
        csvfile = 'result/' + classifier.__class__.__name__ + '-'+ str(count) + '-submit.csv'
        with open(csvfile, 'w') as output:
            predictions = np.column_stack((test[myid], classifier.predict_proba(np.array(test[features])))).tolist()
            predictions = [[int(i[0])] + i[2:3] for i in predictions]
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([myid,goal])
            writer.writerows(predictions)
        try:
            with PdfPages('feature_importances_' + classifier.__class__.__name__ +".pdf") as pdf:
                # Plot feature importance
                feature_importance = classifier.feature_importances_
                # make importances relative to max importance
                feature_importance = 100.0 * (feature_importance / feature_importance.max())
                sorted_idx = np.argsort(feature_importance)
                pos = np.arange(sorted_idx.shape[0]) + .5
                plt.subplot(1, 2, 2)
                plt.barh(pos, feature_importance[sorted_idx], align='center')
                plt.yticks(pos, features)
                plt.xlabel('Relative Importance')
                plt.title('Variable Importance')
                pdf.savefig()
                plt.show()
        except:
            pass