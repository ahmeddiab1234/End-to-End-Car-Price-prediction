
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
import pickle
import yaml

import matplotlib
matplotlib.use("Qt5Agg") 
import matplotlib.pyplot as plt

RANDOM_STATE=42

def load_df(path):
    if path is None:
        raise ValueError('Path is don`t defined correctly')
    df = pd.read_csv(path)
    
    return df
    
def load_x_t(df:pd.DataFrame):
    x,y = df.iloc[:,:-1], df.iloc[:,-1]
    return df, x,y

def split_train_val(df:pd.DataFrame, split_sz=0.8):
    sz = int(split_sz*df.shape[0])
    train,val = df.iloc[:sz,:],df.iloc[sz:,:]
    return train,val

def split_data(x,t, split_sz=0.2, train_val=False):
        x_train, x_val, t_train, t_val = train_test_split(x,t, test_size=split_sz, random_state=RANDOM_STATE)
        return x_train, x_val, t_train, t_val


def save_model(model, encode_season, encode_store, poly, scaler, name='Ridge', type='val'):
    model_dict = {
        "model": model,
        "encode_season": encode_season,
        "encode_store": encode_store,
        "poly": poly,
        "scaler": scaler
    }
    if type=='val':
        filename = fr'val_pkl/{name}.pkl'
    else:
        filename = fr'{name}.pkl'

    with open(filename, 'wb') as file:
        pickle.dump(model_dict, file)

def load_model(name='Ridge',type='val'):
    if type=='val':
        filename = fr'val_pkl/{name}.pkl'
    else:
        filename = fr'{name}.pkl'

    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)

    return loaded_dict['model'], loaded_dict['encode_season'], loaded_dict['encode_store'], loaded_dict['poly'], loaded_dict['scaler']


def load_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


def log_result(text, name='Ridge', filename=None):
    if filename is None:
        filename=fr'logs/model_results_{name}.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'a') as f:
        f.write(text+'\n')


if __name__=='__main__':
    pass

    