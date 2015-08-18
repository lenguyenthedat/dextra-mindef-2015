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