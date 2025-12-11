import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.helper_fun import *
from Preproessing.Preprocessing import Preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

config = load_config()
model_config = config['Model']
linear_config = model_config['LinearRegression']
ridge_config = model_config['Ridge']
xgboost_config = model_config['XGBoost']

RANDOM_STATE = config.get('RANDOME_STATE', 42)


class Prepare:
    def __init__(self, path):
        self.df = load_df(path)
        self.x_train = None
        self.x_val = None
        self.t_train = None
        self.t_val = None

    def prepare(self):
        preprocess = Preprocessing(self.df)
        df_processed = preprocess.prepare_data(fit_encoders=True)
        _, x, t = load_x_t(df_processed)
        self.x_train, self.x_val, self.t_train, self.t_val = split_data(x, t)

        self.x_train, _, poly, scaler = preprocess.fit_transform(self.x_train, None)

        self.x_val = preprocess.transform(self.x_val, poly, scaler)

        encoders = preprocess.encoders
        return self.x_train, self.x_val, self.t_train, self.t_val, poly, scaler, encoders


class Train:
    def __init__(self, x_train, x_val, t_train, t_val):
        self.x_train = x_train
        self.x_val = x_val
        self.t_train = t_train
        self.t_val = t_val

    def try_linear_regression(self):
        model = LinearRegression(fit_intercept=linear_config['fit_intercept'])
        model.fit(self.x_train, self.t_train)
        log_result('Training Path', 'LinearRegression')

        train_score, train_error = eval_model(model, self.x_train, self.t_train)
        val_score, val_error = eval_model(model, self.x_val, self.t_val)

        log_result(f'MSE for Train: {train_error}', 'LinearRegression')
        log_result(f'R2-score for Train: {train_score}', 'LinearRegression')
        log_result(f'MSE for Val: {val_error}', 'LinearRegression')
        log_result(f'R2-score for Val: {val_score}', 'LinearRegression')
        log_result('--' * 40, 'LinearRegression')
        return model

    def try_ridge(self):
        model = Ridge(alpha=ridge_config['alpha'],
                      fit_intercept=ridge_config['fit_intercept'],
                      solver=ridge_config.get('solver', None),
                      random_state=RANDOM_STATE)
        model.fit(self.x_train, self.t_train)
        log_result('Training Path', 'Ridge')

        train_score, train_mse = eval_model(model, self.x_train, self.t_train)
        val_score, val_mse = eval_model(model, self.x_val, self.t_val)

        log_result(f'MSE for Train: {train_mse}', name='Ridge')
        log_result(f'R2-score for Train: {train_score}', name='Ridge')
        log_result(f'MSE for Val: {val_mse}', name='Ridge')
        log_result(f'R2-score for Val: {val_score}', name='Ridge')
        log_result(f"parameters: alpha = {ridge_config['alpha']}, fit-intercept = {ridge_config['fit_intercept']}", name='Ridge')
        log_result('--' * 40, name='Ridge')
        return model

    def try_xgboost(self):
        model = XGBRegressor(
            n_estimators=xgboost_config['n_estimators'],
            learning_rate=xgboost_config['learning_rate'],
            max_depth=xgboost_config['max_depth'],
            gamma=xgboost_config['gamma'],
            reg_alpha=xgboost_config['reg_alpha'],
            reg_lambda=xgboost_config['reg_lambda'],
            random_state=RANDOM_STATE
        )
        model.fit(self.x_train, self.t_train)
        log_result('Training Validaion path', 'XGboost')

        train_score, train_error = eval_model(model, self.x_train, self.t_train)
        val_score, val_error = eval_model(model, self.x_val, self.t_val)

        log_result(f'MSE for Train: {train_error}', name='XGboost')
        log_result(f'R2-score for Train: {train_score}', name='XGboost')
        log_result(f'MSE for Val: {val_error}', name='XGboost')
        log_result(f'R2-score for Val: {val_score}', name='XGboost')
        param = {
            'n_estimators': xgboost_config['n_estimators'],
            'learning_rate': xgboost_config['learning_rate'],
            'max_depth': xgboost_config['max_depth'],
            'gamma': xgboost_config['gamma'],
            'reg_alpha': xgboost_config['reg_alpha'],
            'reg_lambda': xgboost_config['reg_lambda']
        }
        log_result(f'parameters: {str(param)}', name='XGboost')
        log_result('--' * 40, name='XGboost')
        return model


def eval_model(model, x, t):
    pred = model.predict(x)
    mse_error = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)
    return r2score, mse_error


if __name__ == '__main__':
    pre = Prepare('data/train_car_price.csv')
    x_train, x_val, t_train, t_val, poly, scaler, encoders = pre.prepare()
    print(x_train.shape, x_val.shape, t_train.shape, t_val.shape)

    trainer = Train(x_train, x_val, t_train, t_val)
    model = trainer.try_xgboost()
    save_model(model, poly, scaler, encoders, model_config['model_name'], 'test')

    print("Successful")
