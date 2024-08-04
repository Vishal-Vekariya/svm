from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
def evaluate_model(lg,x_train,y_train,x_test,y_test):
        y_pred_train = lg.predict(x_train)
        
        
        y_pred_test = lg.predict(x_test)
        
        
        return y_pred_train,y_pred_test 
    
def evaluate_model_svm(model,x_train,x_test):
    y_pred_train_svm = model.predict(x_train)
    y_pred_test_svm = model.predict(x_test)
    
    return y_pred_train_svm,y_pred_test_svm

def evaluate_model_rbf(model,x_train,x_test):
    y_pred_train_rbf = model.predict(x_train)
    y_pred_test_rbf = model.predict(x_test)
    return y_pred_train_rbf,y_pred_test_rbf

def evaluate_model_poly(model,x_train):
    y_pred_train_poly = model.predict(x_train)
    return y_pred_train_poly