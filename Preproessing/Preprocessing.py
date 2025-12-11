import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils.helper_fun import *
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler, Normalizer

config = load_config()
process_config = config['preprocessing']


class Handling:
    def __init__(self, df):
        self.df = df.copy()
        self.impute_values = {}
        self.outlier_bounds = {}

    def fix_data_type(self):
        self.df['Price'] = self.df['Price'].str.replace('$', '', regex=False).astype(int)

        self.df['Levy'] = self.df['Levy'].str.replace('-', '0', regex=False).astype(float).astype('Int64')

        self.df['Mileage'] = self.df['Mileage'].str.replace('KM', '', regex=False)
        self.df['Mileage'] = pd.to_numeric(self.df['Mileage'], errors='coerce')

        self.df['Prod. year'] = self.df['Prod. year'].astype(str).replace('unknown', '0').astype('Int64')

        self.df['Cylinders'] = self.df['Cylinders'].astype('Int64')

        self.df['Engine volume'] = self.df['Engine volume'].str.replace(' Turbo', '', regex=False).astype(float)

        return self.df

    def remove_useless_features(self):
        for col in ['ID', 'Random_notes']:
            if col in self.df.columns:
                self.df.drop(col, axis=1, inplace=True)
        return self.df

    def handling_missing(self, fit_mode=True, impute_values=None):
        if fit_mode:
            mileage_mean = int(self.df['Mileage'].mean().round())
            self.df['Mileage'] = self.df['Mileage'].fillna(mileage_mean).astype('int64')
            self.impute_values['Mileage'] = mileage_mean

            levy_mean = int(self.df['Levy'].mean())
            self.df['Levy'] = self.df['Levy'].fillna(levy_mean)
            self.impute_values['Levy'] = levy_mean

            color_mode = self.df['Color'].mode()[0]
            self.df['Color'] = self.df['Color'].fillna(color_mode)
            self.impute_values['Color'] = color_mode
        else:
            if impute_values is None:
                raise ValueError("impute_values must be provided when fit_mode=False")
            if 'Mileage' in self.df.columns:
                self.df['Mileage'] = self.df['Mileage'].fillna(impute_values['Mileage']).astype('int64')
            if 'Levy' in self.df.columns:
                self.df['Levy'] = self.df['Levy'].fillna(impute_values['Levy'])
            if 'Color' in self.df.columns:
                self.df['Color'] = self.df['Color'].fillna(impute_values['Color'])

        return self.df

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates(keep='first')
        return self.df

    def handling_outlier(self, fit_mode=True, outlier_bounds=None):
        feats = ['Levy', 'Engine volume']
        if fit_mode:
            for col in feats:
                # ensure float
                self.df[col] = self.df[col].astype('float64')
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                self.df[col] = self.df[col].clip(lower, upper)
                self.outlier_bounds[col] = (lower, upper)
        else:
            if outlier_bounds is None:
                raise ValueError("outlier_bounds must be provided when fit_mode=False")
            for col in feats:
                if col in outlier_bounds:
                    lower, upper = outlier_bounds[col]
                    self.df[col] = self.df[col].astype('float64').clip(lower, upper)
                else:
                    self.df[col] = self.df[col].astype('float64')
        return self.df

    def remove_negative_price(self):
        if 'Price' in self.df.columns:
            self.df = self.df[self.df['Price'] > 0].copy()
        return self.df

    def apply_price_log(self, apply=True):
        if apply and 'Price' in self.df.columns:
            self.df['Price'] = np.log(self.df['Price'])
        return self.df


class Scaling:
    def __init__(self, x, x_val=None):
        self.x = x
        self.x_val = x_val
        self.option = process_config['scaling']['option']
        self.degree = process_config['polynomial']['degree']
        self.include_bias = process_config['polynomial']['include_bias']

    def polynomial_feature(self):
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
        self.x = self.poly.fit_transform(self.x)
        if self.x_val is not None:
            self.x_val = self.poly.transform(self.x_val)
            return self.poly, self.x, self.x_val
        return self.poly, self.x, None

    def scaling(self):
        if self.option == 1:
            self.scaler = MinMaxScaler()
        elif self.option == 2:
            self.scaler = StandardScaler()
        elif self.option == 3:
            self.scaler = Normalizer()
        else:
            self.scaler = None
            return None, self.x, self.x_val

        self.x = self.scaler.fit_transform(self.x)
        if self.x_val is not None:
            self.x_val = self.scaler.transform(self.x_val)
            return self.scaler, self.x, self.x_val
        return self.scaler, self.x, None


class Preprocessing:
    def __init__(self, df):
        self.df = df.copy()
        self.encoders = None         
        self._impute = None           
        self._outlier_bounds = None   
        self.poly = None
        self.scaler = None

    def prepare_data(self, fit_encoders=True):
        handle = Handling(self.df)

        self.df = handle.fix_data_type()
        self.df = handle.remove_useless_features()

        if fit_encoders:
            self.df = handle.handling_missing(fit_mode=True)
            self._impute = handle.impute_values.copy()
        else:
            if self._impute is None:
                raise ValueError("Preprocessing._impute is None — set it before calling prepare_data(fit_encoders=False)")
            self.df = handle.handling_missing(fit_mode=False, impute_values=self._impute)

        self.df = handle.remove_duplicates()

        if fit_encoders:
            self.df = handle.handling_outlier(fit_mode=True)
            self._outlier_bounds = handle.outlier_bounds.copy()
        else:
            if self._outlier_bounds is None:
                raise ValueError("Preprocessing._outlier_bounds is None — set it before calling prepare_data(fit_encoders=False)")
            self.df = handle.handling_outlier(fit_mode=False, outlier_bounds=self._outlier_bounds)

        self.df = handle.remove_negative_price()

        apply_log = process_config.get('apply_log', False)
        if fit_encoders:
            self.df = handle.apply_price_log(apply=apply_log)
        else:
            self.df = handle.apply_price_log(apply=apply_log)

        feats = ['Manufacturer', 'Model', 'Category',
                 'Leather interior', 'Fuel type', 'Gear box type',
                 'Drive wheels', 'Doors', 'Wheel', 'Color']

        if fit_encoders:
            encoders = {}
            for col in feats:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                encoders[col] = le

            encoders["_impute"] = self._impute
            encoders["_outlier_bounds"] = self._outlier_bounds
            encoders["_apply_log"] = apply_log

            self.encoders = encoders
        else:
            if self.encoders is None:
                raise ValueError("self.encoders is None. Set preprocess.encoders from training before calling prepare_data(fit_encoders=False)")
            encoders = self.encoders
            for col in feats:
                le = encoders[col]
                def safe_map(x):
                    if x in le.classes_:
                        return int(le.transform([x])[0])
                    return -1
                self.df[col] = self.df[col].apply(safe_map)

            if "_impute" in encoders and self._impute is None:
                self._impute = encoders["_impute"]
            if "_outlier_bounds" in encoders and self._outlier_bounds is None:
                self._outlier_bounds = encoders["_outlier_bounds"]

        return self.df

    def fit_transform(self, x, x_val=None):
        scale = Scaling(x, x_val)
        self.poly, x, x_val = scale.polynomial_feature()
        self.scaler, x, x_val = scale.scaling()

        self.poly = self.poly
        self.scaler = self.scaler

        return x, x_val, self.poly, self.scaler

    def transform(self, x, poly, scaler):
        if poly is None or scaler is None:
            raise ValueError("poly and scaler must be provided to transform()")
        x = poly.transform(x)
        if scaler is not None:
            x = scaler.transform(x)
        return x


if __name__ == '__main__':
    df = load_df(config['dataset']['path'])
    preprocess = Preprocessing(df)
    df_train = preprocess.prepare_data(fit_encoders=True)
    df_train, x, t = load_x_t(df_train)
    x_train, x_val, t_train, t_val = split_data(x, t)
    x_train, _, poly, scaler = preprocess.fit_transform(x_train, None)
    print(df_train.shape, x_train.shape, t_train.shape)
    x_val_ = preprocess.transform(x_val, poly, scaler)
    print(x_val.shape, t_val.shape)
