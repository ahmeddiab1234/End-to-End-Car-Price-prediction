"""  
 - handling wrong data type for numerical features
 - remove id and random notes features
 - drop negative values from sales
 - handling missing values for Levy, Mileage and Color features
 - remove dublicates
 - cap outliers to acceptable limit
 - try to apply log on price / use as it is
 - label encoding for categorical features

- Scaling (minmax scaler, standar scaler, normalize)
- polynomial feature

"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils.helper_fun import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler, Normalizer


class Handling:
    def __init__(self, df):
        self.df=df

    def fix_data_type(self):
        self.df['Price'] = self.df['Price'].str.replace('$','',regex=False).astype(int)
        self.df['Levy'] = self.df['Levy'].str.replace('-','0',regex=False).astype(float).astype('Int64') 
        self.df['Mileage'] = self.df['Mileage'].str.replace('KM', '', regex=False)
        self.df['Mileage'] = pd.to_numeric(self.df['Mileage'], errors='coerce')
        self.df['Prod. year'] = (
        self.df['Prod. year'].astype(str).replace('unknown', '0').astype('Int64'))
        self.df['Cylinders'] = self.df['Cylinders'].astype('Int64')
        self.df['Engine volume'] = self.df['Engine volume'].str.replace(' Turbo','',regex=False).astype(float)
        return self.df

    def remove_usless_featus(self):
        self.df.drop('ID', axis=1, inplace=True)
        self.df.drop('Random_notes', axis=1, inplace=True)
        return self.df
    
    def handling_missing(self):
        self.df['Mileage'] = self.df['Mileage'].fillna(self.df['Mileage'].mean().round()).astype('int64')
        self.df['Levy'] = self.df['Levy'].fillna(int(self.df['Levy'].mean()))
        self.df['Color'] = self.df['Color'].fillna(self.df['Color'].mode()[0])
        return self.df

    def remove_dublicates(self):
        self.df = self.df.drop_duplicates(keep='first')
        return self.df

    def handling_outlier(self):
        feats = ['Levy', 'Engine volume']
        for col in feats:
            self.df[col] = self.df[col].astype('float64')
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            self.df[col] = self.df[col].clip(lower, upper)
            
            if 'int' in str(self.df[col].dtype):
                self.df[col] = self.df[col].round().astype('Int64')
        return self.df

    def remove_negative_price(self):
        self.df = self.df[self.df['Price'] > 0].copy()
        return self.df

    def apply_price_log(self):
        self.df['Price'] = np.log(self.df['Price'])
        return self.df

    def label_encoding(self):
        feats = ['Manufacturer', 'Model', 'Category', 
                 'Leather interior', 'Fuel type', 'Gear box type', 
                 'Drive wheels', 'Doors', 'Wheel', 'Color']

        le = LabelEncoder()
        for col in feats:
            self.df[col] = le.fit_transform(self.df[col])
        return self.df

class Scaling():
    def __init__(self, x, x_val, option=2, degree=2, include_bias=True):
        self.x=x
        self.x_val=x_val
        self.option=option
        self.degree=degree
        self.include_bias=include_bias

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
            return None, self.x, self.x_val

        self.x = self.scaler.fit_transform(self.x)
        if self.x_val is not None:
            self.x_val = self.scaler.transform(self.x_val)
            return self.scaler, self.x, self.x_val
        return self.scaler, self.x, None

class Preprocessing():
    def __init__(self, df):
        self.df = df
        self.poly=None
        self.scaler=None
        
    def prepare_data(self, apply_log=True) :
        handle = Handling(self.df)
        self.df = handle.fix_data_type()
        self.df = handle.remove_usless_featus()
        self.df = handle.handling_missing()
        self.df = handle.remove_dublicates()
        self.df = handle.handling_outlier()
        self.df = handle.remove_negative_price()
        if apply_log:
            self.df = handle.apply_price_log()
        self.df = handle.label_encoding()

        return self.df


    def fit_transform(self, x, x_val=None, option=2, degree=2, include_bias=True):
        scale = Scaling(x, x_val, option, degree, include_bias)
        self.poly, x, x_val = scale.polynomial_feature()
        self.scaler, x, x_val = scale.scaling()

        return x, x_val, self.poly, self.scaler

    def transform(self, x, poly, scaler):
        x = poly.transform(x)
        x = scaler.transform(x)
        return x


if __name__ == '__main__':
    df = load_df('data/train_car_price.csv')
    preprocess = Preprocessing(df)

    df = preprocess.prepare_data(False)

    df, x,t = load_x_t(df)
    x_train, x_val, t_train, t_val = split_data(x, t)
    x_train_, _, _, _ = preprocess.fit_transform(x_train, None)
    print(df.shape ,x_train.shape, t_train.shape)

    x_val_ = preprocess.transform(x_val)
    print(x_val.shape, t_val.shape)


    # for col in df.columns:
    #     print(f'{col} : {df[col].dtype}')


