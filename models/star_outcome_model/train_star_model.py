'''Developing and training the Random Forest Regressor for predicting the final star outcome.'''

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from dython import nominal

import warnings
warnings.filterwarnings('ignore')


def prepare_data():
    '''
    Load the datasets for training.

    Return:
        df: clean dataframe for training with selected columns
    '''
    yp_data = pd.read_csv('/home/workspace/files/aryan/data_files_raw/yp_background.csv')
    yp_data = yp_data[['Client Number', 'Do They Have Any Medical Issues', 'Mental Health/Learning Disabilities',
                    'A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'F2', 'F3', 'G1', 'G2', 'H1', 'H2',
                    'H3', 'I1', 'I2', 'J1', 'K1', 'L1']]

    yp_data['med_issues'] = (yp_data['Do They Have Any Medical Issues'] == 'Yes').astype(int)
    yp_data['mental_issues'] = (yp_data['Mental Health/Learning Disabilities'].isna()).astype(int)

    for col in ['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'F2', 'F3', 'G1', 'G2', 'H1', 'H2',
                    'H3', 'I1', 'I2', 'J1', 'K1', 'L1']:
        yp_data[col] = (yp_data[col].isin([None, 'nan', 'No', '-', 'N/A - Not Applicable', 'No benefits',
                        'No benefits', 'Don\'t Know / Refused', 'No benefits claims made / refused to answer',
                        'Not Known', '\n\n'])).astype(int)

    yp_data['Client.Number'] = yp_data['Client Number']
    yp_data = yp_data.drop(['Do They Have Any Medical Issues', 'Mental Health/Learning Disabilities', 'Client Number'], axis=1)



    raw_data = pd.read_csv('/home/workspace/files/aryan/Results/data_2224_withSTAR.csv')
    raw_data = raw_data.drop(['Unnamed: 0', 'starmin_noncollab', 'starfinal_noncollab', 'starmin', 'starfinal','stardup'], 1)
    starbase_df = pd.read_csv('/home/workspace/files/stbasil_prj/hannah/star engineered.csv')
    starbase_df['Client.Number'] = starbase_df['Client Number']
    starbase_df['starfinal'] = starbase_df['avgscore_final']
    starbase_df = starbase_df.drop(['Client Number', 'Unnamed: 0', 'avgscore_final', 'Star Date_final',
                                    'Star Date_first', 'Star Date', 'avgscore_first', 'flag_first',
                                    'flag_final', 'Outcome Area 1 - Score_final',
                                    'Outcome Area 2 - Score_final', 'Outcome Area 3 - Score_final',
                                    'Outcome Area 4 - Score_final', 'Outcome Area 5 - Score_final',
                                    'Outcome Area 6 - Score_final', 'Outcome Area 7 - Score_final',
                                    'Outcome Area 8 - Score_final'], 1)
    raw_data = raw_data.merge(starbase_df, on='Client.Number', how='left')

    raw_data = raw_data[raw_data['stardup'] == 0]
    raw_data = raw_data.drop('stardup', 1)

    data = raw_data.merge(yp_data, how='left', on='Client.Number')

    df = data[['EET.status', 'Area', 'med_issues', 'G1', 'A1',
       'Age.at.Start', 'Scheme', 'Num.Sessions', 'Time.per.Session',
       'D1', 'avgscore', 'starfinal']]
    
    return df


def draw_correlations(dfcorr):
    '''
    Print the matrix of correlations between all categorical and numerical columns.

    Parameters:
        dfcorr: dataframe to analyse
    '''
    nominal.associations(dfcorr,figsize=(30,30),mark_columns=True)


def load_dataset(df):
    '''
    Split the dataset into train and test set.

    Parameters:
        df: dataframe with both dependent and independent features
    Return:
        X_train: dataset of independent features for training
        X_test: X_train: dataset of independent features for testing
        y_train: X_train: dataset of dependent features for training
        y_test: dataset of dependent features for testing
    '''
    cat = ['EET.status', 'Area', 'Scheme']
    dfc = pd.get_dummies(df, columns=cat)

    y = dfc['starfinal'].values
    dfc_x = dfc.drop('starfinal', axis=1)
    feature_list = list(dfc_x.columns)

    X = dfc_x.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training Features Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', X_test.shape)
    print('Testing Labels Shape:', y_test.shape)
    return X_train, X_test, y_train, y_test


def hp_tuning(X_train, y_train):
    '''
    Tune the hyperparameters of the model, Random Forest Regressor.

    Parameters:
        X_train: dataset of independent features for training
        y_train: X_train: dataset of dependent features for training
    Return:
            BEST_NE: best n_estimators parameter
            BEST_MD: best max_depth parameter
            BEST_RS: best random_state parameter
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    n_estimators = [int(x) for x in np.linspace(start = 1, stop = 50, num = 50)]
    random_grid = {'n_estimators': n_estimators}
    rf = RandomForestRegressor(max_depth=4, random_state=42)
    rf_best_ne = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                                n_iter = 100, cv = kf, verbose=2, random_state=42, n_jobs = -1)
    rf_best_ne.fit(X_train, y_train)
    BEST_NE = rf_best_ne.best_params_['n_estimators']

    max_depth = [int(x) for x in np.linspace(2, 6, num = 5)]
    # max_depth.append(None)
    random_grid = {'max_depth': max_depth}
    rf = RandomForestRegressor(n_estimators=BEST_NE, random_state=42)
    rf_best_depth = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                                n_iter = 100, cv = kf, verbose=2, random_state=42, n_jobs = -1)
    rf_best_depth.fit(X_train, y_train)
    BEST_MD = rf_best_depth.best_params_['max_depth']

    random_state = [int(x) for x in np.linspace(1, 100, num = 100)]
    random_grid = {'random_state': random_state}

    rf = RandomForestRegressor(n_estimators=BEST_NE, max_depth=BEST_MD)
    rf_best_rs = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                                n_iter = 100, cv = kf, verbose=2, random_state=42, n_jobs = -1)
    rf_best_rs.fit(X_train, y_train)
    BEST_RS = rf_best_rs.best_params_['random_state']

    return BEST_NE, BEST_MD, BEST_RS


def fit_best_model(BEST_NE, BEST_MD, BEST_RS, X_train, X_test, y_train):
    '''
    Fit the test dataset into the tuned model.

    Parameters:
            BEST_NE: best n_estimators parameter
            BEST_MD: best max_depth parameter
            BEST_RS: best random_state parameter
            X_train: dataset of independent features for training
            X_test: X_train: dataset of independent features for testing
            y_train: X_train: dataset of dependent features for training
    Return:
        yp: list of predicted values based on X_test
    '''
    rf_best = RandomForestRegressor(n_estimators=BEST_NE, random_state=BEST_RS, max_depth=BEST_MD)
    rf_best.fit(X_train, y_train)
    yp = rf_best.predict(X_test)
    return yp


def evaluate(y_test, pred_outcomes):
    '''
    Helper function to calculate the errors of the predictions of the model.

    Parameters:
        y_test: dataset of dependent features for testing
        pred_outcomes: values predicted by the model
    Return:
        mse: Mean Square Error value
        rmse: Root Mean Square Error value
        mae: Mean Absolute Error value
    '''
    err_sum, mse, rmse = 0, 0, 0
    for real, pred in zip(y_test, pred_outcomes):
        err_sum += (pred - real) ** 2

    mse = err_sum / len(y_test)
    rmse = np.sqrt(mse)

    mae, err_sum_mae = 0, 0
    for real, pred in zip(y_test, pred_outcomes):
        err_sum_mae += abs(pred - real)
    mae = err_sum_mae / len(y_test)
    return round(mse, 3), round(rmse, 3), round(mae, 3)


def model_eval(model_type, X_train, X_test, y_train, y_test, BEST_NE, BEST_RS, BEST_MD):
    '''
    Evaluate the tuned model performance by comparing it to the base model performance.

    Parameters:
        model_type: type of the model to produce results for, either base or best
        X_train: dataset of independent features for training
        X_test: X_train: dataset of independent features for testing
        y_train: X_train: dataset of dependent features for training
        y_test: dataset of dependent features for testing
        BEST_NE: best n_estimators parameter
        BEST_MD: best max_depth parameter
        BEST_RS: best random_state parameter
    Return:
        mse: Mean Square Error value
        rmse: Root Mean Square Error value
        mae: Mean Absolute Error value
    '''
    if model_type == 'base':
        model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    elif model_type == 'best':
        model = RandomForestRegressor(n_estimators=BEST_NE, random_state=BEST_RS, max_depth=BEST_MD)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    curr_df = pd.DataFrame({'test': y_test, 'pred': y_pred})

    mse, rmse, mae = evaluate(curr_df['test'], curr_df['pred'], 2)
    return mse, rmse, mae


def print_evaluation(X_train, X_test, y_train, y_test, BEST_NE, BEST_RS, BEST_MD):
    '''
    Display table with results of base model, best model, and the improvement achieved.

    Parameters:
        X_train: dataset of independent features for training
        X_test: X_train: dataset of independent features for testing
        y_train: X_train: dataset of dependent features for training
        y_test: dataset of dependent features for testing
        BEST_NE: best n_estimators parameter
        BEST_MD: best max_depth parameter
        BEST_RS: best random_state parameter
    Return:
        df_performance: dataframe with results
    '''
    base = model_eval('base', X_train, X_test, y_train, y_test, BEST_NE, BEST_RS, BEST_MD)
    best = model_eval('best', X_train, X_test, y_train, y_test, BEST_NE, BEST_RS, BEST_MD)

    error_types = ['mse', 'rmse', 'mae']

    df_performance = pd.DataFrame({'error type': error_types, 'base model': base, 'best model': best, 'improvement': [x-y for x, y in zip(best, base)]})
    print('\t\tPERFORMANCE\n', df_performance, '\n')
    return df_performance


def main():
    '''
    Run the model training process.
    '''
    df = prepare_data()
    X_train, X_test, y_train, y_test = load_dataset(df)

    BEST_NE, BEST_MD, BEST_RS = hp_tuning(X_train, X_test, y_train, y_test)
    yp = fit_best_model(BEST_NE, BEST_MD, BEST_RS, X_train, X_test, y_train, y_test)

    df_performance = print_evaluation(X_train, X_test, y_train, y_test, BEST_NE, BEST_RS, BEST_MD)


if __name__ == '__main__':
    main()
