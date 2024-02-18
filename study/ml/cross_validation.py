from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X,y = load_iris(return_X_y=True)

print(X.shape, y.shape) #(150, 4) (150,)

####################################################
# #Train, test 데이터셋 분할
# X_train, X_test, y_train, y_test = train_test_split(
#     X,y, test_size=0.4, random_state=42
# )

# print(X_train.shape, y_train.shape) #(90, 4) (90,)
# print(X_test.shape, y_test.shape) #(60, 4) (60,)

# #Train 에서 Validation 데이터셋 분할
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.4, random_state=42
# )

# print(X_train.shape, y_train.shape) #(54, 4) (54,)
# print(X_val.shape, y_val.shape) #(36, 4) (36,)


######################################################
#cross val score 
# from sklearn.model_selection import cross_val_score
# from sklearn.svm import SVC
# clf = SVC(kernel='linear', C=1, random_state=42)
# scores = cross_val_score(clf, X, y, cv=5)
# print("scores : ", scores)
# #scores :  [0.96666667 1.         0.96666667 0.96666667 1.        ]
# print("acc 평균 : %0.2f" % scores.mean()) #acc 평균 : 0.98
# print("acc 표준편차 : %0.2f" % scores.std()) #acc 표준편차 : 0.02

#######################################################
# from sklearn import metrics
# scores = cross_val_score(
#     clf, X, y, cv=5, scoring='f1_macro')
# print(scores)

#######################################################
# from sklearn.model_selection import cross_validate
# from sklearn.metrics import recall_score
# scores = cross_validate(
#     clf, X, y, cv=5, scoring=['precision_macro', 'recall_macro'])
# print(scores)

########################################################
#KFlod
# import numpy as np
# from sklearn.model_selection import KFold

# X = np.array(['a', 'b', 'c', 'd','e','f','g','h','i','j'])
# kf = KFold(n_splits=5)
# for train, test in kf.split(X):
#     print(train, test)
# '''
# [2 3 4 5 6 7 8 9] [0 1]
# [0 1 4 5 6 7 8 9] [2 3]
# [0 1 2 3 6 7 8 9] [4 5]
# [0 1 2 3 4 5 8 9] [6 7]
# [0 1 2 3 4 5 6 7] [8 9]
# '''

###########################################################
#Repeated KFold
# import numpy as np
# from sklearn.model_selection import RepeatedKFold
# X = np.array(['a', 'b', 'c', 'd','e','f','g','h','i','j'])
# rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
# for train, test in rkf.split(X):
#     print(train, test)
# '''
# [2 3 4 5 6 7 8 9] [0 1]
# [0 1 4 5 6 7 8 9] [2 3]
# [0 1 2 3 6 7 8 9] [4 5]
# [0 1 2 3 4 5 8 9] [6 7]
# [0 1 2 3 4 5 6 7] [8 9]
# [0 2 3 4 5 6 7 9] [1 8]
# [1 2 3 4 6 7 8 9] [0 5]
# [0 1 3 4 5 6 8 9] [2 7]
# [0 1 2 3 5 6 7 8] [4 9]
# [0 1 2 4 5 7 8 9] [3 6]
# [2 3 4 5 6 7 8 9] [0 1]
# [0 1 2 3 4 6 7 9] [5 8]
# [0 1 2 5 6 7 8 9] [3 4]
# [0 1 2 3 4 5 6 8] [7 9]
# [0 1 3 4 5 7 8 9] [2 6]
# '''

##############################################################
#LOO
# import numpy as np
# from sklearn.model_selection import LeaveOneOut
# X = np.array([1,2,3,4])
# loo = LeaveOneOut()
# for train, test in loo.split(X):
#     print(train, test)
# '''
# [1 2 3] [0]
# [0 2 3] [1]
# [0 1 3] [2]
# [0 1 2] [3]
# '''
##############################################################
#LPO
# import numpy as np
# from sklearn.model_selection import LeavePOut
# X = np.array([1,2,3,4])
# lpo = LeavePOut(p=2)
# for train, test in lpo.split(X):
#     print(train, test)
# '''
# [2 3] [0 1]
# [1 3] [0 2]
# [1 2] [0 3]
# [0 3] [1 2]
# [0 2] [1 3]
# [0 1] [2 3]
# '''
##############################################################
#ShuffleSplit
# import numpy as np
# from sklearn.model_selection import ShuffleSplit
# X = np.arange(10)
# ss = ShuffleSplit(n_splits=5, test_size=0.4, random_state=42)
# for train_index, test_index in ss.split(X):
#     print("%s %s" % (train_index, test_index))
# '''
# [7 2 9 4 3 6] [8 1 5 0]
# [3 4 7 9 6 2] [0 1 8 5]
# [8 5 3 7 1 4] [9 2 0 6]
# [8 0 3 4 5 9] [1 7 6 2]
# [0 7 6 3 2 9] [1 5 4 8]
# '''

################################################################
#Stratified k-fold
# import numpy as np
# from sklearn.model_selection import StratifiedKFold, KFold
# X, y = np.ones((50,1)), np.hstack(([0] * 45, [1] * 5)) #label의 개수 0 : 45개  1: 5개
# skf = StratifiedKFold(n_splits=3)
# print("StrartifiedKFold")
# for train, test in skf.split(X, y):
#     print(f'train : {np.bincount(y[train])} | test : {np.bincount(y[test])}')
# print("="*30)
# print("kFold")
# kf = KFold(n_splits=3)
# for train, test in kf.split(X, y):
#     print(f'train : {np.bincount(y[train])} | test : {np.bincount(y[test])}')
# '''
# StrartifiedKFold
# train : [30  3] | test : [15  2]
# train : [30  3] | test : [15  2]
# train : [30  4] | test : [15  1]
# ==============================
# kFold
# train : [28  5] | test : [17]
# train : [28  5] | test : [17]
# train : [34] | test : [11  5]
# '''
#####################################################################
# import numpy as np
# from sklearn.model_selection import GroupKFold
# X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
# y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
# groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
# gf = GroupKFold(n_splits=3)
# for train, test in gf.split(X,y,groups= groups):
#     print(train, test)
# '''
# [0 1 2 3 4 5] [6 7 8 9]
# [0 1 2 6 7 8 9] [3 4 5]
# [3 4 5 6 7 8 9] [0 1 2]
# '''

######################################################################
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv) #TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, test_size=None)
TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, test_size=None)
for train, test in tscv.split(X):
    print(train, test)
'''
[0 1 2] [3]
[0 1 2 3] [4]
[0 1 2 3 4] [5]
'''
#gap = 1
TimeSeriesSplit(gap=1, max_train_size=None, n_splits=3, test_size=None)
for train, test in tscv.split(X):
    print(train, test)
'''
[0 1 2] [3]
[0 1 2 3] [4]
[0 1 2 3 4] [5]
'''