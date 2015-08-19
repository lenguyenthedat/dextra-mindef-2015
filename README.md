XGBClassifier(n_estimators=128,subsample=0.6,max_depth=16,min_child_weight=3)  0.0202
XGBClassifier(n_estimators=128,subsample=1,max_depth=16,min_child_weight=3) 0.0191784
XGBClassifier(n_estimators=128,subsample=1,max_depth=16,min_child_weight=3) No scaling 0.0202145
XGBClassifier(n_estimators=128,subsample=0.4,max_depth=16,min_child_weight=3) No scaling 0.0198647
XGBClassifier(n_estimators=128,subsample=1,max_depth=16,min_child_weight=3) No scaling No "RANK GROUPING" 0.0194125
XGBClassifier(n_estimators=128,subsample=1,max_depth=16,min_child_weight=4) No scaling 0.0201803
XGBClassifier(n_estimators=128,subsample=0.4,max_depth=16,min_child_weight=4) 0.0204031
XGBClassifier(n_estimators=128,subsample=1,max_depth=16,min_child_weight=3) No "RANK GROUPING" 0.0191475
XGBClassifier(n_estimators=512,subsample=0.8,max_depth=8,min_child_weight=5) 0.0201772
XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7) 0.0190156
XGBClassifier(n_estimators=512,subsample=1,max_depth=10,min_child_weight=9) 0.0193612
XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7) 0.018724 (No rank grade nor marital status)
XGBClassifier(n_estimators=512,subsample=1,max_depth=10,min_child_weight=9) 0.0191667 (No child age info)
XGBClassifier(n_estimators=512,subsample=1,max_depth=10,min_child_weight=9) 0.0192915 (No parent service)
XGBClassifier(n_estimators=512,subsample=2,max_depth=16,min_child_weight=7) 0.0198633 (upgraded scikit, fix bug and stuff)
XGBClassifier(n_estimators=365,subsample=1,max_depth=16,min_child_weight=9,learning_rate=0.1) 0.01987 with age grouping
XGBClassifier(n_estimators=512,subsample=1,max_depth=8,min_child_weight=11,learning_rate=0.1) 0.0195721
Blending / Stacking (see ens* folders)
XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7) - removed ['GENDER','COUNTRY_OF_BIRTH','NATIONALITY','AGE'] - 0.0206067
XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7) - fill w median, 0.0192577, rank grouping 1, 2 and both
XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7) - fill w -1, 0.019219, rank grouping 1, 2 and both
XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7) - fill w -1, 0.018724, no rank grouping at all
XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7) - fill w -1, 0.0188001, no rank grouping at all, no scaler
XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7) - fill w -1, and mean(if number), 0.0196049  no rank grouping at all
XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7) - fill w 0,  0.0189772 no rank grouping at all
XGBClassifier(n_estimators=256,subsample=2,max_depth=16,min_child_weight=7) - fill w -10,0.0190489 no rank grouping at all

--- mod median, with rank groupings
RandomForestClassifier(n_estimators=1024, max_features=23,oob_score=False, bootstrap=True, min_samples_leaf=1,min_samples_split=2, max_depth=32): 0.026679
XGBClassifier(n_estimators=256,subsample=0.5,max_depth=10,min_child_weight=3,learning_rate=0.1) 0.0200443
XGBClassifier(n_estimators=256,subsample=0.5,max_depth=10,min_child_weight=3,learning_rate=0.05) 0.0193743
XGBClassifier(n_estimators=2048,subsample=0.5,max_depth=10,min_child_weight=3,learning_rate=0.03) 0.0204512

--- native xgb
    xgbtrain = xgb.DMatrix(train[features], label=train[goal])
    params = {}
    params["objective"] = "multi:softprob"
    params["num_class"] = 2
    params["eval_metric"]="mlogloss"
    params["eta"] = 0.05
    params["min_child_weight"] = 3
    params["subsample"] = 1
    params["colsample_bytree"] = 0.6
    #params["scale_pos_weight"] = 1.0
    params["silent"] = 0
    params["max_depth"] = 8
    params["nthread"] = 4
    plst = list(params.items())
    num_rounds = 180

Results: [0.01601425233360964, 0.020371078696600001, 0.010461164126409057, 0.014025222435671015, 0.014524528938030926]
Mean: 0.0150792493061

params = {'max_depth':8, 'eta':0.05, 'silent':1,
          'objective':'multi:softprob', 'num_class':2, 'eval_metric':'logloss',
          'min_child_weight':3, 'subsample':1,'colsample_bytree':0.6, 'nthread':4}
num_rounds = 180
Results: [0.015967713220674192, 0.019908556421256548, 0.0099656954301119999, 0.01385761412214254, 0.014662429553165731]
Mean: 0.0148724017495

Removed some scales
0.0158728863676
0.020110657513
0.00969442441583
0.0140470703463
0.0143980670541
Results: [0.015872886367572073, 0.020110657513012147, 0.0096944244158308698, 0.014047070346340977, 0.01439806705409912]
Mean: 0.0148246211394

No rank group but rank 1 and 2
0.0158892333181
0.0200519979571
0.00957516652104
0.0140163609598
0.0141398236705
Results: [0.01588923331809481, 0.020051997957098781, 0.0095751665210400162, 0.014016360959825841, 0.014139823670548318]
Mean: 0.0147345164853

--- Hall of fucking fame:
cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True, indices=False, random_state=1337)
XGBClassifier
0.016305351037
0.0214994732043
0.0109005534736
0.0157298640789
0.0163817216885
Results: 0.0161633926965


cv = cross_validation.KFold(len(train), n_folds=10, shuffle=True, indices=False, random_state=1337)
XGBClassifier
0.0132394294141
0.0190374850443
0.0180839631635
0.02466821637
0.0114606088516
0.0103901016906
0.0182169675627
0.0135496847102
0.0164289318806
0.0154071139885
Results: 0.0160482502676