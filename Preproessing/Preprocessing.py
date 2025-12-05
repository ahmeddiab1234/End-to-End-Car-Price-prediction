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
        self.df['Prod. year'] = self.df['Prod. year'].str.replace('unknown','0',regex=False).astype('Int64')
        self.df['Cylinders'] = self.df['Cylinders'].astype('Int64')
        self.df['Engine volume'] = self.df['Engine volume'].str.replace(' Turbo','',regex=False).astype(float)


    def remove_usless_featus(self):
        self.df.drop('ID', axis=1, inplace=True)
        self.df.drop('Random_notes', axis=1, inplace=True)
    

    def handling_missing(self):
        self.df['Mileage'] = self.df['Mileage'].fillna(self.df['Mileage'].mean().round()).astype('int64')
        self.df['Levy'] = self.df['Levy'].fillna(int(df['Levy'].mean()))
        self.df['Color'] = self.df['Color'].fillna(self.df['Color'].mode()[0])

    def remove_dublicates(self):
        self.df = self.df.drop_duplicates(keep='first', inplace=True)

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


    def remove_negative_price(self):
        self.df['Price'] = self.df[self.df['Price'] > 0]

    def apply_price_log(self):
        self.df['Price'] = np.log(self.df['Price'])

    def label_encoding(self):
        feats = ['Manufacturer', 'Model', 'Category', 
                 'Leather interior', 'Fuel type', 'Gear box type', 
                 'Drive wheels', 'Doors', 'Wheel', 'Color']

        le = LabelEncoder()
        for col in feats:
            self.df[col] = le.fit_transform(self.df[col])

class Scaling(df):
    def polynomial_feature(self, x, x_val=None, degree=2, include_bias=True):
        self.poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        x = self.poly.fit_transform(x)
        if x_val is not None:
            x_val = self.poly.transform(x_val)
            return self.poly, x, x_val
        return self.poly, x

    def scaling(self, x, x_val=None, option=2):
        if option == 1:
            self.scaler = MinMaxScaler()
        elif option == 2:
            self.scaler = StandardScaler()
        elif option == 3:
            self.scaler = Normalizer()
        else:
            return None, x, x_val
                                            
        x = self.scaler.fit_transform(x)
        if x_val is not None:
            x_val = self.scaler.transform(x_val)
            return self.scaler, x, x_val
        return self.scaler, x




if __name__ == '__main__':
    df = load_df('data/car_price_Dataset.csv')
    preprocess = Preproccesing(df)
    preprocess.fix_data_type()
    for col in df.columns:
        print(f'{col} : {df[col].dtype}')




