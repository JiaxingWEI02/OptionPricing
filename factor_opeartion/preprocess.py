# -*- coding: utf-8 -*-
# reading local data
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def read_merge(path):
    '''
    param path: the dir that stored all data
    can handle with txt or csv files
    '''
    for file in os.listdir(path):
        if file.endswith('.txt'):
            files = os.path.join(path, file)
        elif file.endswith('.csv'):
            files = os.path.join(path, file)
        else:
            continue
    df_list = []
    for f in files:
        date = os.path.basename(f).replace('.txt', '')
        df = pd.read_csv(f, delimiter='\t', header=None, names=['INS', 'Weight'])
        df['DAY'] = date
        df_list.append(df)
    merge = pd.concat(df_list)
    return merge


def convert_to_pivot(data, date, stockid, value):
    df = data.copy()
    df = df.pivot(index=date, columns=stockid, values=value)
    return df

def adjust_kline(ohlc, mode='REAL'):
    '''
    param pos: pivot_table, target position
    param ohlc: pivot_table, price and amt, volume data
    param fq: if take the qfq/hfq into account
    '''
    if 'adj_factor' not in ohlc:
        raise ValueError('Need adj_factor to perform adjustment')
    
    res = {}
    for i in ohlc.columns:
        res[i] = ohlc[i].unstack()
    
    if mode == 'REAL':
        return res

    elif mode == 'BEFORE':
        adj = ohlc.adj_factor.unstack()
        adj = adj.div(adj.fillna(method='ffill').iloc[-1],axis=1)
        for i in ohlc.keys():
            if i in ['OPEN', 'CLOSE', 'HIGH', 'LOW']:
                res[i] *= adj
            elif i in ['VOLUME']:
                res[i] /= adj
        return res
    
    elif mode == 'BEFORE':
        adj = ohlc.adj_factor.unstack()
        for i in ohlc.keys():
            if i in ['OPEN', 'CLOSE', 'HIGH', 'LOW']:
                res[i] *= adj
            elif i in ['VOLUME']:
                res[i] /= adj
        return res
    

def get_pos(path):
    res=[]
    for i in os.listdir(path):
        if ".txt" in i:
            tmp=pd.read_table(f"{path}/{i}",header=None)
            tmp.columns=['Contract',i.split(".")[0]]
            res.append(tmp.set_index(['Contract']).T)
        elif ".csv" in i:
            tmp=pd.read_csv(f"{path}/{i}",header=None)
            tmp.columns=['Contract',i.split(".")[0]]
            res.append(tmp.set_index(['Contract']).T)
    return pd.concat(res,axis=0).sort_index()

