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

def create_dataset():
    path="/home/lefteris/Ceid/calc_theory/Project_ΘΑ_2023-24/BreastTissue.xlsx"

    df_data = pd.read_excel(path, sheet_name="Data")

    df_data.loc[df_data['Class'] == 'car', 'Class'] = 1
    df_data.loc[df_data['Class'] == 'fad', 'Class'] = 2
    df_data.loc[df_data['Class'] == 'mas', 'Class'] = 3
    df_data.loc[df_data['Class'] == 'gla', 'Class'] = 4
    df_data.loc[df_data['Class'] == 'con', 'Class'] = 5
    df_data.loc[df_data['Class'] == 'adi', 'Class'] = 6

    df_data['Class'] = df_data['Class'].astype(int)

    columnsForNormalization = df_data.columns[2:]

    scaler = MinMaxScaler(feature_range=(-1, 1))

    df_data[columnsForNormalization] = scaler.fit_transform(df_data[columnsForNormalization])

    return df_data

def create_X_Y(dataset): 
    X = dataset.drop(['Case #', 'Class'], axis=1)
    Y = dataset['Class']
    return X,Y


