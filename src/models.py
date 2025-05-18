
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def lista_modelos():
    return {
        'SVM_Linear': SVC(kernel='linear', probability=True, random_state=42),
        'SVM_Radial': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM_Poly': SVC(kernel='poly', degree=3, probability=True, random_state=42),
        'MLP': MLPClassifier(max_iter=1000, random_state=42),
        'Adaboost': AdaBoostClassifier(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(),
        'ExtraTrees': ExtraTreesClassifier(random_state=42),
        'NaiveBayes': GaussianNB(),
        'Bagging': BaggingClassifier(random_state=42),
        'SGD': SGDClassifier(loss='modified_huber', max_iter=1000, random_state=42)
    }
