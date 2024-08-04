from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy_vars
from src.models.train_model import logistic_regression, support_vm,rbf_k,poly_n
from src.models.predict_model import evaluate_model, metrics_score, evaluate_model_svm,evaluate_model_rbf,evaluate_model_poly


if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/HR_Employee_Attrition.xlsx"
    df = load_and_preprocess_data(data_path)
    
    
    Y, X_scaled = create_dummy_vars(df)
    
    lg,x_train,y_train,x_test,y_test = logistic_regression(Y, X_scaled)
    
    y_pred_train,y_pred_test  = evaluate_model(lg,x_train,y_train,x_test,y_test)

    
    print("Training Data:\n")
    tr = metrics_score(y_train, y_pred_train)
    print(f"Testing Data:\n")
    te = metrics_score(y_test, y_pred_test)
    
    model,x_train,x_test = support_vm(Y, X_scaled)
    
    y_pred_train_svm,y_pred_test_svm = evaluate_model_svm(model,x_train,x_test)
    
    print("Training Data SVM:\n")
    tr_SVM= metrics_score(y_train, y_pred_train_svm)
    print(f"Testing Data SVM:\n")
    te_SVM = metrics_score(y_test, y_pred_test_svm)
    
    model,x_train,x_test = rbf_k(Y, X_scaled)
    y_pred_train_rbf,y_pred_test_rbf = evaluate_model_rbf(model,x_train,x_test)
    print("Training Data rbf:\n")
    tr_rbf = metrics_score(y_train, y_pred_train_rbf)
    print(f"Testing Data rbf:\n")
    te_rbf= metrics_score(y_test, y_pred_test_svm)
    
    model,x_train = poly_n(Y, X_scaled)
    y_pred_train_poly = evaluate_model_poly(model,x_train)
    
    print("Training Data poly:\n")
    te_poly = metrics_score(y_train, y_pred_train_poly)
    
    
    