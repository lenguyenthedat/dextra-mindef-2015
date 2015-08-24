Ministry of Defence Data Analytics Challenge
============================================
![img](http://i.imgur.com/TSTPnaI.png)

Challenge URL: https://challenges.dextra.sg/challenge/44

Quick analysis with Tableau Public: https://public.tableau.com/profile/le.nguyen.the.dat#!/vizhome/MINDEFDataScienceChallenge2015/MINDEFDataScienceChallenge2015

Libraries used: Scikit-Learn, Pandas, XGBoost, Mathplotlib

Scores:
-------
[Public Leader Board (5th/158 Participants)](https://challenges.dextra.sg/challenge/44)

- 0.0169515: best Single XGBoost model
- 0.0168939: blending multiple XGBoost models with different Features Set.

[Private Leader Board (1st/158 Participants)](http://www.dextra.sg/mindef-challenge-results/)

- 0.0141351 

Feature Importance:
-------------------
![img](https://raw.githubusercontent.com/lenguyenthedat/dextra-mindef-2015/master/Feature_Importance_xgb.png)

Submission History (only the best one):
---------------------------------------
Only Native XGBoost was recorded since it just dominated everything.

1) Public Leader Board 0.0171364

    $ python classify-xgb-native.py # 990r depth6
    0.0155765992602
    0.019516592639
    0.00988590074655
    0.0141124661651
    0.014303086534
    Mean: 0.014678929069 (Local Score)

2) Public Leader Board 0.0172253

    $ python classify-xgb-native.py # 180r
    0.0157726389016
    0.0201645979107
    0.0095532522597
    0.013888759618
    0.0139117869773
    Mean: 0.0146582071335 (Local Score)

3) Public Leader Board 0.0171475

    $ python classify-xgb-native.py #added age_gender, rm a bunch of features
    0.015551655811
    0.019148557532
    0.00965389534226
    0.0139233429833
    0.0139280448029
    Mean: 0.0144410992943 (Local Score)

4) Public Leader Board 0.0171112

    $ python classify-xgb-native.py # promo - gender
    0.0155083548415
    0.0189263516813
    0.00951782504063
    0.0140093232169
    0.014178032663
    Mean: 0.0144279774887 (Local Score)

5) Public Leader Board 0.0170703

    $ python classify-xgb-native.py # cap salary 101%
    0.0153414063482
    0.0189991328711
    0.00959486331913
    0.0139794582592
    0.0140253377611
    Mean: 0.0143880397117 (Local Score)

6) Public Leader Board 0.0170369

    $ python classify-xgb-native.py # INJURY TYPE as String
    0.0153022751895
    0.0189944794534
    0.00957494483944
    0.0139220394066
    0.014069437855
    Mean: 0.0143726353488 (Local Score)

7) Public Leader Board 0.0169515

    $ python classify-xgb-native.py # better minchildage # treat as str
    0.0152455036731
    0.0189285563506
    0.00961418416464
    0.0139189502782
    0.0139664367926
    Mean: 0.0143347262518 (Local Score)
