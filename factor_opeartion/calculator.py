import numpy as np


def MAD(factor,n: int = 3):
    m=factor.median(axis=1)
    x = 1.4826*factor.sub(m,axis=0).abs().median(axis=1)
    f = m - n*x
    u = m + n*x
    return np.clip(factor,f,u,axis=0)

def zScore(data):
    return data.sub(data.mean(axis=1),axis=0).div(data.std(axis=1),axis=0)


def log(data):
    return np.log(data)