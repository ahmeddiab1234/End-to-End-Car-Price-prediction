import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from utils.helper_fun import *
from Preproessing.Preprocessing import *
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor as nnr
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


RANDOM_STATE = 42

class Prepare():
    def __init__(self, path):
        self.df = load_df(path)
        self.x_train=None
        self.x_val=None
        self.t_train=None
        self.t_val=None
    
    def prepare(self):
        preprocess = Preprocessing(self.df)
        self.df = preprocess.prepare_data(True)
        _, x,t=load_x_t(self.df)

        self.x_train, self.x_val, self.t_train, self.t_val = split_data(x, t)
        self.x_train, _, poly, scaler = preprocess.fit_transform(self.x_train, None, 1, 2, True)
        self.x_val = preprocess.transform(self.x_val, poly, scaler)
        
        return self.x_train, self.x_val, self.t_train, self.t_val, poly, scaler



class Train():
    def __init__(self,x_train, x_val, t_train, t_val):
        self.x_train=x_train
        self.x_val=x_val
        self.t_train=t_train
        self.t_val=t_val

    def try_linear_regression(self, fit_intercept=True):
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(self.x_train, self.t_train)
        log_result(f'Training Path', 'LinearRegression')
        # log_result(f'Weights: {model.coef_}')
        # log_result(f'Intercept: {model.intercept_}')

        train_score,train_error = eval_model(model,self.x_train,self.t_train, 'train')
        val_score,val_error = eval_model(model,self.x_val,self.t_val, 'val')
        log_result(f'MSE for Train: {train_error}','LinearRegression')
        log_result(f'R2-score for Train: {train_score}','LinearRegression')
        log_result(f'MSE for Val: {val_error}','LinearRegression')
        log_result(f'R2-score for Val: {val_score}','LinearRegression')
        log_result('--'*40,'LinearRegression')
        return model 


    def try_ridge(self, alpha=1, fit_intercept=True, solver='auto'):

        log_result(f'Training Path', 'Ridge')
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver, random_state=RANDOM_STATE)
        model.fit(self.x_train,self.t_train)

        train_score,train_mse =eval_model(model,self.x_train,self.t_train, 'train')
        val_score, val_mse = eval_model(model,self.x_val,self.t_val, 'val')

        log_result(f'MSE for Train: {train_mse}', name='Ridge')
        log_result(f'R2-score for Train: {train_score}', name='Ridge')
        log_result(f'MSE for Val: {val_mse}', name='Ridge')
        log_result(f'R2-score for Val: {val_score}', name='Ridge')
        log_result(f'parameters: alpha = {alpha}, fit-intercept = {fit_intercept}', name='Ridge')
        log_result('--'*40, name='Ridge')

        return model

    def try_neural_network(self, hiden_layers=1, solver=1, learning_rate=0.1, alpha=0.1, max_iter=300):
        log_result(f'Training Path', 'Neural_Network')
        model = nnr(
            hidden_layer_sizes=hiden_layers,
            activation='identity',
            solver=solver,
            learning_rate='adaptive',
            learning_rate_init=learning_rate,
            alpha=alpha,
            max_iter=max_iter,
            random_state=RANDOM_STATE
        )

        model.fit(self.x_train, self.t_train)
        train_score, train_error = eval_model(model, self.x_train, self.t_train, 'train')
        val_score, val_error = eval_model(model, self.x_val, self.t_val, 'val')

        log_result(f'Score train: {train_score}', name='Neural_Network')
        log_result(f'MSE train: {train_error}', name='Neural_Network')
        log_result(f'Score val: {val_score}', name='Neural_Network')
        log_result(f'MSE val: {val_error}', name='Neural_Network')

        current_params = {
            'hidden_layers': hiden_layers,
            'solver': solver,
            'learning_rate_init': learning_rate,
            'max_iter': max_iter,
            'alpha': alpha
        }

        log_result(f'Parameters: {current_params}', name='Neural_Network')
        log_result('--'*40, name='Neural_Network')

        return model


    def try_xgboost(self, n_estimators=1, learning_rate=1, max_depth=1, gamma=1, reg_alpha=1, reg_lambda=1):
        log_result(f'Training Validaion path', 'XGboost')

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=RANDOM_STATE
        )

        model.fit(self.x_train, self.t_train)
        train_score, train_error = eval_model(model, self.x_train, self.t_train, 'train')
        val_score, val_error = eval_model(model, self.x_val, self.t_val, 'val')

        log_result(f'MSE for Train: {train_error}', name='XGboost')
        log_result(f'R2-score for Train: {train_score}', name='XGboost')
        log_result(f'MSE for Val: {val_error}', name='XGboost')
        log_result(f'R2-score for Val: {val_score}', name='XGboost')
        param = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda
        }
        log_result(f'parameters: {str(param)}', name='XGboost')
        log_result('--'*40, name='XGboost')

        return model



def eval_model(model, x, t, name='val'):
    pred = model.predict(x)
    mse_error = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)
    return r2score,mse_error


if __name__ == '__main__':
    pre=Prepare('data/train_car_price.csv')
    x_train, x_val, t_train, t_val, poly, scaler = pre.prepare()
    print(x_train.shape, x_val.shape, t_train.shape, t_val.shape) 

    train =Train(x_train, x_val, t_train, t_val)
    model = train.try_ridge(3, True, 'auto')
    # save_model(model, poly, scaler, 'LinearRegression', 'val')

    print("Successful")

