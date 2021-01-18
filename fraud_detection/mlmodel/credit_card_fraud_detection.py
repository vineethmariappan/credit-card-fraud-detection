import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest 
from sklearn.neighbors import LocalOutlierFactor #lly to k nearest neighs but here it gets a score
import pickle
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
def classification_model(model, X_train, y_train):
    #Fit the model:
    # time_start = time.perf_counter() #start counting the time
    model.fit(X_train,y_train)
    n_cache = []
    
    train_predictions = model.predict(X_train)
    precision = precision_score(y_train, train_predictions)
    recall = recall_score(y_train, train_predictions)
    f1 = f1_score(y_train, train_predictions)
    
    print("Precision ", precision)
    print("Recall ", recall)
    print("F1 score ", f1)
    
    cr_val = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
    
    # time_end = time.perf_counter()
    
    # total_time = time_end-time_start
    print("Cross Validation Score: %f" %np.mean(cr_val))
    # print("Amount of time spent during training the model and cross validation: %4.3f seconds" % (total_time))
# filename = 'model.sav'
# pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
data = pd.read_csv('fraud_detection/mlmodel/creditcard.csv')
# print(data.columns)
# print(data.shape)
# print(data.describe())
data = data.sample(frac = 1, random_state=1)
# print(data.shape)
# data.hist(figsize = (20,20))
# plt.show()
Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]

outlier_fraction = float(len(Fraud)) / len(Valid)
# print(outlier_fraction)
# print('Fraud Cases: {}'.format(len(Fraud)))
# print('Valid Cases: {}'.format(len(Valid)))

# corrmat = data.corr()
# fig = plt.figure(figsize = (12,9)) #increase figure size
# sns.heatmap(corrmat, vmax = .8, square = True)
# plt.show()

columns = data.columns.tolist()
columns = [c for c in columns if c not in ['Class']] #every columns except Class column name

target = 'Class'
X = data[columns]
Y = data[target]

# print(X.shape)
# print(Y.shape)



state = 1

classifiers = {
    "Isolation Forest" : IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor" : LocalOutlierFactor(
    n_neighbors = 20,
    contamination=outlier_fraction)
}
# trn_data = lgb.Dataset(X,Y,feature_name = columns)
# val_data = lgb.Dataset(Y)
# params = {
#  'task': 'train'
#  , 'boosting_type': 'gbdt'
#  , 'objective': 'regression'
#  , 'num_class': 1
#  , 'metric': 'rmsle' 
#  , 'min_data': 1
#  , 'verbose': -1
# }
# clf_lgb = lgb.train(trn_data,10000,valid_sets=[trn_data,val_data],verbose_eval=1000,early_stopping_rounds=5000)
# clf_lgb = lgb.train(params, trn_data, num_boost_round=50)
# gbm = lgb.train(params, trn_data, num_boost_round=50)
lgb_model = lgb.LGBMClassifier(num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
classification_model(lgb_model, X, Y)
n_outliers = len(Fraud)
def find(data):
    for i,(clf_name, clf) in enumerate(classifiers.items()): #runs twice since there are 2 classifiers
        #fit the data
        if clf_name=="Local Outlier Factor":
            # y_pred = clf.fit_predict(X)
            # scores_pred=clf.negative_outlier_factor_
            pass
        else:
            clf.fit(X)
            # scores_pred = clf.decision_function(X)
            # y_pred = clf.predict(X)
            ans=clf.predict(data)
            # ans = clf.predict()
        # y_pred[y_pred==1] = 0
        # y_pred[y_pred==-1] = 1
        # clf_lgb.fit(X)
        # ans=clf_lgb.predict(data)
    test_predictions_lgb = lgb_model.predict(data)
    print(test_predictions_lgb)
    return test_predictions_lgb.tolist()
    # -1 => fraud
    # 1 => not fraud
    # print("ans : ",ans)
    # n_errors = (y_pred!=Y).sum()
    # print('{}: {}'.format(clf_name, n_errors))
    # print(accuracy_score(Y,y_pred))
    # print(classification_report(Y,y_pred))
print("Train Complete!")

