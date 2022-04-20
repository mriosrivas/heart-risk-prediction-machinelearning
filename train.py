#!/usr/bin/env python
# coding: utf-8

# # Final Model Selection

import argparse

import pandas as pd
import pickle as pkl


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a machine learning model for inference.')
    parser.add_argument('--model', help='Select model to train, if not specified logistic regression is used. '
                                        'Can select between logistic, random_forrest, xgboost or all.', default='logistic')
    args = parser.parse_args()
    #print(args.model)

    # Load and clean data
    df = pd.read_csv('heart_2020_cleaned.csv')
    df.columns = df.columns.str.lower()

    numerical = list(df.dtypes[df.dtypes == 'float'].index.values)

    categorical = list(df.dtypes[df.dtypes == 'object'].index.values)
    categorical.remove('heartdisease')

    for c in categorical:
        df[c] = df[c].str.lower()

    df['heartdisease'] = df['heartdisease'].str.lower()

    # Split train and test data
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=1)

    y_train = (df_train['heartdisease']=='yes').astype('int').values
    y_test = (df_test['heartdisease']=='yes').astype('int').values

    df_train = df_train.drop(columns='heartdisease')
    df_test = df_test.drop(columns='heartdisease')

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Format data for out model
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train.to_dict(orient='records'))
    X_test = dv.transform(df_test.to_dict(orient='records'))

    # 1. Logistic Regression
    if args.model == 'logistic' or args.model == 'all':
        from sklearn.linear_model import LogisticRegression
        print('Training logistic regression \n')
        lr = LogisticRegression(random_state=1, max_iter=10000, C=0.1)
        lr.fit(X_train, y_train)
        y_pred = lr.predict_proba(X_test)[:,1]
        score = roc_auc_score(y_test, y_pred)
        print('auc = {} \n'.format(score))

    # 2. Random Forrest
    if args.model == 'random_forrest' or args.model == 'all':
        print('Training random forrest \n')
        from sklearn.ensemble import RandomForestClassifier

        best_estimator = 300
        best_depth = 15
        best_min_samples_leaf = 4

        rf = RandomForestClassifier(n_estimators=best_estimator,
                                    max_depth=best_depth,
                                    min_samples_leaf=best_min_samples_leaf,
                                    random_state=1)

        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_test)[:,1]
        score = roc_auc_score(y_test,y_pred)
        print('auc = {} \n'.format(score))

    # 3. XGBoost
    if args.model == 'xgboost' or args.model == 'all':
        import xgboost as xgb
        print('Training xgboost \n')

        feature_names = list(dv.get_feature_names_out())
        dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=feature_names)

        best_eta = 0.01
        best_max_depth = 6
        best_min_child_weights = 1

        xgb_params = {'eta':best_eta,
                     'max_depth':best_max_depth,
                     'min_child_weight':best_min_child_weights,

                     'objective':'binary:logistic',
                     'nthread':8,
                      'eval_metric':'auc',

                     'seed':1,
                     'verbosity':1}

        model = xgb.train(xgb_params, dtrain, num_boost_round=1000)
        y_pred = model.predict(dtest)
        score = roc_auc_score(y_test, y_pred)
        print('auc = {} \n'.format(score))

    # Save Model

    with open('dict_vectorizer.bin', 'wb') as file:
        pkl.dump(dv, file)
        print('Saved dict_vectorizer.bin \n')

    if args.model == 'logistic' or args.model == 'all':
        with open('logistic_regression.bin', 'wb') as file:
            pkl.dump(lr, file)
            print('Saved logistic_regression.bin \n')

    if args.model == 'random_forrest' or args.model == 'all':
        with open('random_forrest.bin', 'wb') as file:
            pkl.dump(rf, file)
            print('Saved random_forrest.bin \n')

    if args.model == 'xgboost' or args.model == 'all':
        with open('xgboost.bin', 'wb') as file:
            pkl.dump(model, file)
            print('Saved xgboost.bin \n')




