import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class GM:
    def __init__(self, estimator,X,Y):
        self.estimator = estimator
        self.X=X
        self.Y=Y
    def calculate_metrics(self, y_true, y_pred, class_label):
        true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
        false_negatives = np.sum((y_true == class_label) & (y_pred != class_label))
        false_positives = np.sum((y_true != class_label) & (y_pred == class_label))
        true_negatives = np.sum((y_true != class_label) & (y_pred != class_label))

        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

        return sensitivity, specificity

    def cross_val_with_metrics(self, n_splits=5, shuffle=True, random_state=1):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        gm_scores = []

        for train_index, test_index in skf.split(self.X, self.Y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            Y_train, Y_test = self.Y.iloc[train_index], self.Y.iloc[test_index]

            self.estimator.fit(X_train, Y_train)
            predicted_labels = self.estimator.predict(X_test)

            
            sensitivity_scores = []
            specificity_scores = []
            for class_label in np.unique(self.Y):
                sensitivity, specificity = self.calculate_metrics(Y_test, predicted_labels, class_label)
                sensitivity_scores.append(sensitivity)
                specificity_scores.append(specificity)

            gm_scores.append(np.sqrt(np.mean(sensitivity_scores) * np.mean(specificity_scores)))

        return gm_scores


class SVM:
    def __init__(self,X,Y):
       self.best_c=0.0
       self.best_g=0.0
       self.svm=SVC(kernel='rbf')
       self.X=X
       self.Y=Y
    def best_C(self):
        find_c = {'C': list(range(1, 201, 5))}
        search = GridSearchCV(self.svm, find_c, cv=5, scoring='accuracy')
        search.fit(self.X, self.Y)
        print("Best Parameters: ", search.best_params_)
        self.best_c=search.best_params_['C']
    
    def best_G(self):
        find_g = {'gamma': list(np.arange(0.0, 10.0, 0.5))}
        search = GridSearchCV(self.svm, find_g, cv=5, scoring='accuracy')
        search.fit(self.X, self.Y)
        print("Best Parameters: ", search.best_params_)
        self.best_g=search.best_params_['gamma']
        
    def best_EXCE(self):
        self.best_C()
        self.best_G()
        
        self.svm = SVC(kernel='rbf', C=self.best_c, gamma=self.best_g)
        test=GM(self.svm,self.X,self.Y)
        geometric_mean_scores = test.cross_val_with_metrics()
        mean_value = np.mean(geometric_mean_scores)
        print("-------------THIS IS SVM-------------")
        print("Geometric Mean Scores for each fold:", geometric_mean_scores)
        print("The mean values is :", mean_value)

class KNN:
    def __init__(self,X,Y):
        self.best_k=3
        self.knn=KNeighborsClassifier(n_neighbors=self.best_k)
        self.X=X
        self.Y=Y
    def find_bk(self):
        find_k={'n_neighbors': list(range(3, 15, 1))}
        search = GridSearchCV(self.knn, find_k, cv=5, scoring='accuracy')
        search.fit(self.X, self.Y)
        print("Best Parameters: ", search.best_params_)
        self.best_k=search.best_params_['n_neighbors']
    def best_EXCE(self):
        self.find_bk()
        self.knn = KNeighborsClassifier(n_neighbors=self.best_k)

        test=GM(self.knn,self.X,self.Y)
        geometric_mean_scores = test.cross_val_with_metrics()
        mean_value = np.mean(geometric_mean_scores)
        print("-------------THIS IS KNN-------------")
        print("Geometric Mean Scores for each fold:", geometric_mean_scores)
        print("The mean values is :", mean_value)

class NB:
    def __init__(self,X,Y):
        self.nb = GaussianNB()
        self.X=X
        self.Y=Y
    def best_EXCE(self):
        test=GM(self.nb,self.X,self.Y)
        geometric_mean_scores = test.cross_val_with_metrics()
        mean_value = np.mean(geometric_mean_scores)
        print("-------------THIS IS NAIVE BAYES-------------")
        print("Geometric Mean Scores for each fold:", geometric_mean_scores)
        print("The mean values is :", mean_value)

