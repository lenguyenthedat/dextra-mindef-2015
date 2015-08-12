import pandas as pd
import time
import csv
import numpy as np
import os
import itertools

from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sknn.mlp import Classifier, Layer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

pd.options.mode.chained_assignment = None

sample = True
gridsearch = False

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

# Load data
if sample:
    if gridsearch:
        train = pd.read_csv('./data/20150803115609-HR_Retention_2013_training.csv')
        test = pd.read_csv('./data/20150803115608-HR_Retention_2013_to_be_predicted.csv')
    else:
        df = pd.read_csv('./data/20150803115609-HR_Retention_2013_training.csv')
        df['is_train'] = (df[myid] % 17) >= 9
        # df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
        train, test = df[df['is_train']==True], df[df['is_train']==False]
else:
    # To run with real data
    train = pd.read_csv('./data/20150803115609-HR_Retention_2013_training.csv')
    test = pd.read_csv('./data/20150803115608-HR_Retention_2013_to_be_predicted.csv')

train=train.fillna(-1)
test=test.fillna(-1)

# Pre-processing non-number values
le = LabelEncoder()
for col in features_non_numeric:
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in features:
    scaler.fit(list(train[col])+list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

MyNNClassifier = Classifier(
                    layers=[
                        Layer("Rectifier", units=100),
                        Layer("Rectifier", units=100),
                        Layer("Rectifier", units=100),
                        Layer("Rectifier", units=100),
                        Layer('Softmax')],
                    learning_rate=0.1,
                    learning_rule='momentum',
                    learning_momentum=0.9,
                    batch_size=20,
                    valid_size=0.01,
                    n_stable=50,
                    n_iter=200,
                    verbose=True)

# Define classifiers
if sample:
    classifiers = [
        # KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025, probability=True),
        # SVC(gamma=2, C=1, probability=True),
        # RandomForestClassifier(),
        # RandomForestClassifier(max_depth=16, n_estimators=1024, max_features=None),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # LDA(),
        # QDA(),
        # MyNNClassifier,
        # DecisionTreeClassifier(criterion='entropy', min_samples_split=2,
        #     min_samples_leaf=1, max_depth=5, min_weight_fraction_leaf=0.0,
        #     max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None),
        # AdaBoostClassifier(DecisionTreeClassifier(max_depth=16), algorithm="SAMME",n_estimators=200),
        # RandomForestClassifier(max_depth=16, n_estimators=1024, max_features=None),
        # XGBClassifier(n_estimators=128,subsample=1,max_depth=16,min_child_weight=3),
        # XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7)
        XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7)
        # XGBClassifier()
    ]
else:
    classifiers = [# Other methods are underperformed yet take very long training time for this data set
        # MyNNClassifier1,
        # MyNNClassifier2,
        # GradientBoostingClassifier(max_depth=16,n_estimators=1024),
        # RandomForestClassifier(max_depth=16,n_estimators=1024),
        # XGBClassifier(n_estimators=128,subsample=1,max_depth=16,min_child_weight=3),
        # XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7),
        XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7)
    ]

# Train
for classifier in classifiers:
    print classifier.__class__.__name__
    start = time.time()

    if (gridsearch & sample): # only do gridsearch if we run with sampled data.
        try: # depth & estimator: usually fits for RF and XGB
            if (classifier.__class__.__name__ == "GradientBoostingClassifier"):
                print "Attempting GridSearchCV for GB model"
                gscv = GridSearchCV(classifier, {
                    'max_depth': [2, 8, 16],
                    'n_estimators': [32, 64, 128, 256, 512],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.6,0.8,1]},
                    verbose=1, n_jobs=2, scoring='log_loss')
            if (classifier.__class__.__name__ == "XGBClassifier"):
                print "Attempting GridSearchCV for XGB model"
                gscv = GridSearchCV(classifier, {
                    'max_depth': [1, 2, 5, 8, 10, 16, 25, 32],
                    'n_estimators': [64, 128, 256, 365, 512, 712, 1024],
                    'min_child_weight': [1,3,5,7, 9, 11, 13],
                    'subsample': [0.6,0.8,1,1.2]},
                    verbose=1, n_jobs=2, scoring='log_loss')
            if (classifier.__class__.__name__ == "RandomForestClassifier"):
                print "Attempting GridSearchCV for RF model"
                gscv = GridSearchCV(classifier, {
                    'max_depth': [2, 8, 16, 32],
                    'n_estimators': [32, 64, 128, 256, 512],
                    'bootstrap':[True,False],
                    'oob_score': [True,False]},
                    verbose=1, n_jobs=2, scoring='log_loss')
            if (classifier.__class__.__name__ == "Classifier"): # NN Classifier
                print "Attempting GridSearchCV for Neural Network model"
                gscv = GridSearchCV(classifier, {
                    'hidden0__units': [4, 16, 64, 128],
                    'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]},
                    verbose=1, n_jobs=1)
            classifier = gscv.fit(np.array(train[list(features)]), train[goal])
            print(classifier.best_score_)
            print(classifier.best_params_)
        except:
            classifier.fit(np.array(train[list(features)]), train[goal]) # just fit regular one
    else:
        classifier.fit(np.array(train[list(features)]), train[goal])
    print '  -> Training time:', time.time() - start
    # print classifier.feature_importances_
# Evaluation and export result
if sample:
    if ~gridsearch:
        for classifier in classifiers:
            print "===" + classifier.__class__.__name__
            print 'Log Loss:'
            print log_loss(test[goal].values, classifier.predict_proba(np.array(test[features])))
else: # Export result
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