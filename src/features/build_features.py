import pandas as pd
from sklearn.preprocessing import StandardScaler

# create dummy features
def create_dummy_vars(df):
    
    to_get_dummies_for = ['BusinessTravel', 'Department','Education', 'EducationField','EnvironmentSatisfaction', 'Gender',  'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus' ]

#creating dummy variables
    df = pd.get_dummies(data = df, columns= to_get_dummies_for, drop_first= True)

#mapping overtime and attrition
    dict_OverTime = {'Yes': 1, 'No':0}
    dict_attrition = {'Yes': 1, 'No': 0}


    df['OverTime'] = df.OverTime.map(dict_OverTime)
    df['Attrition'] = df.Attrition.map(dict_attrition)
    Y= df.Attrition
    X= df.drop(columns = ['Attrition'])
    sc=StandardScaler()
    X_scaled=sc.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled, columns=X.columns)
    
    return Y, X_scaled