import pandas as pd 
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import ttest_ind
from create_dataset import create_dataset,create_X_Y
from classes import SVM,KNN,NB

df_data=create_dataset()
X,Y=create_X_Y(df_data)



print("(ΕΡΩΤΗΜΑ 3)")
print("\n\n")
svm=SVM(X,Y)
svm.best_EXCE()
print("\n\n")
knn=KNN(X,Y)
knn.best_EXCE()
print("\n\n")
nb=NB(X,Y)
nb.best_EXCE()
print("\n\n")