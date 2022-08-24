'''Pipiline for the Random Forest Regressor model for predicting the final star score outcome.'''

import pandas as pd
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from models.star_outcome_model.train_star_model import prepare_data


def create_components(df1):
    '''
    Save label encoders and one-hot-encoders into pickle files.

    Parameters:
        df1: main dataset
    '''
    # Standard scaling numeric data

    ss = StandardScaler()
    df1[['Age.at.Start','Num.Sessions','Time.per.Session']] = ss.fit_transform(df1[['Age.at.Start','Num.Sessions','Time.per.Session']])
    output = open('StandardScaler.pkl', 'wb')
    pickle.dump(ss, output)
    output.close()

    # One Hot Encoding Category data

    categ = ['EET.status','Area','Scheme']
    ohe = OneHotEncoder()
    X_object = df1[categ]
    df1 = df1.drop(categ,1)
    ohe.fit(X_object)
    codes = ohe.transform(X_object).toarray()
    feature_names = ohe.get_feature_names(categ)
    df1 = pd.concat([df1,  pd.DataFrame(codes,columns=feature_names).astype(int)], axis=1)

    output = open('OneHotEncoder.pkl', 'wb')
    pickle.dump(ohe, output)
    output.close()


def get_model():
    '''
    Save the tuned model into pickle file.

    Return:
        best_model: model to predict the star outcome
    '''
    output = open('model_star/model.pkl','rb')
    best_model = pickle.load(output)
    output.close()
    return best_model


def transform_input(x):
    '''
    Transform dictionary received from html form into the dictionary to feed to the model.
    
    Parameters:
        x: dictionary from html model
    
    Return:
        transformed: dictionary to feed to the model
    '''
    transformed = []

#   'med_issues', 'G1', 'A1', 'Age.at.Start', 'Num.Sessions',
#   'Time.per.Session', 'D1', 'avgscore', 'EET' 'Area', 'Scheme'

    output = open('model_star/StandardScaler.pkl', 'rb')
    le = pickle.load(output)
    output.close()
    a = le.transform([[x['Age.at.Start'],x['Num.Sessions'],x['Time.per.Session']]])
    
    output = open('model_star/OneHotEncoder.pkl', 'rb')
    ohe = pickle.load(output)
    output.close()
    codes = ohe.transform([[x['EET.status'],x['Area'],x['Scheme']]]).toarray()

    transformed.append(x['med_issues'])
    transformed.append(x['G1'])
    transformed.append(x['A1'])
    transformed.append(a[0][0]) #age
    transformed.append(a[0][1]) #num
    transformed.append(a[0][2]) #time

    transformed.append(x['D1'])
    transformed.append(x['avgscore'])

    transformed = np.asarray(transformed)
    transformed = np.append(transformed,codes[0]) #eet, area, scheme

    transformed = np.reshape(transformed,(1,28))

    return transformed


def GenerateStarPrediction(x):
    '''
    Run the model training process.

    Parameters:
        x: the dictionary of values gotten from the html page
    Return:
        prediction: predicted final star score
    '''
    model = get_model()

    tx = transform_input(x)
    prediction = model.predict(tx)
    return list(prediction)[0]


if __name__ == '__main__':
    x = {'EET.status': 'EET', 'Area': 'Warwickshire', 'Scheme': 'AT', 'avgscore': 4, 'Age.at.Start': 20,
       'Num.Sessions': 13, 'Time.per.Session': 80, 'A1': 1, 'med_issues': 1, 'D1': 0, 'G1': 1}
    predicted_star = GenerateStarPrediction(x)
