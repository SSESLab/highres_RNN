#import ML algorithms
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def fun_SVM(X_t, Y_t, X_e):
    clf = svm.SVC() #declaring classifier
    clf.fit(X_t, Y_t)
    Y_e = clf.predict(X_e)
    S = clf.support_vectors_

    return Y_e, S


def multiclass_SVM(X_t, Y_t, X_e):
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_t, Y_t)
    Y_e = clf.predict(X_e)

    return Y_e


def linear_model(X_t, Y_t, X_e):
    lin_model = LinearRegression()
    lin_model.fit(X_t, Y_t)
    Y_e = lin_model.predict(X_e)

    beta = lin_model.coef_


    return Y_e, beta

def fun_kNN(X_t, Y_t, X_e, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh_model = neigh.fit(X_t, Y_t)
    Y_e = neigh_model.predict(X_e)

    return Y_e