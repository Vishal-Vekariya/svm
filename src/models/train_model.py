
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def logistic_regression(Y, X_scaled):
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)
    lg=LogisticRegression()
    lg.fit(x_train,y_train)
    
    return lg,x_train,y_train,x_test,y_test

def support_vm(Y, X_scaled):
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)
    svm = SVC(kernel = 'linear') #linear kernal or linear decision boundary
    model = svm.fit(X = x_train, y = y_train)
    
    return model,x_train,x_test

def rbf_k(Y, X_scaled):
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)
    svm = SVC(kernel = 'rbf') #linear kernal or linear decision boundary
    model = svm.fit(X = x_train, y = y_train)
    
    return model,x_train,x_test

def poly_n(Y, X_scaled):
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)
    svm = SVC(kernel = 'poly', degree=3) #linear kernal or linear decision boundary
    model = svm.fit(X = x_train, y = y_train)
    
    return model,x_train