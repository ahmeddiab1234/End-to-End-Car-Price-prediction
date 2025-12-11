import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Preproessing.Preprocessing import Preprocessing
from utils.helper_fun import *
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

config = load_config()

test_path = 'data/test_car_price.csv'
NAME = config['Model']['model_name']

if __name__ == '__main__':
    df_test = load_df(test_path)

    model, poly, scaler, encoders = load_model(NAME, 'test')

    config['preprocessing']['fit_encoders'] = False
    prepare = Preprocessing(df_test)

    prepare.encoders = encoders
    if "_impute" in encoders:
        prepare._impute = encoders["_impute"]
    if "_outlier_bounds" in encoders:
        prepare._outlier_bounds = encoders["_outlier_bounds"]

    df_test = prepare.prepare_data(fit_encoders=False)

    _, x, t = load_x_t(df_test)
    x = prepare.transform(x, poly, scaler)

    pred = model.predict(x)

    mse_error = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)
    print(f'r2score: {r2score}, mse_error: {mse_error}')

    """
    r2score: 0.3433338403701782, mse_error: 11.997185707092285
    """