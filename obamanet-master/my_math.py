import numpy as np

'''
numpyが扱える量を超えるデータを扱うため、平均分散を求めるプログラムを実装
'''

def my_mean(arr): #(data数,キーポイント数)を想定
    block=10000
    sum=np.zeros((arr.shape[1],),dtype=np.float32)

    i=0
    for i in range(arr.shape[0]//block):
        sum+=np.sum(arr[i*block:(i+1)*block],axis=0)

    sum+=np.sum(arr[(i+1)*block:],axis=0)
    mean=sum/arr.shape[0]
    #print('mean:',mean)

    return mean


def my_var(arr,mean=None,only=False): #(data数,キーポイント数)を想定
    if only: #平均を持っているときのみその平均から分散を求める。
        mean=my_mean(arr)
    squ_mean=my_mean(arr**2)
    var=squ_mean-mean**2
    #print('var:',var)
    return var

def my_standardization_fit(arr): #標準化のための平均と分散を取得
    mean=my_mean(arr=arr)
    var=my_var(arr=arr,mean=mean)
    std=np.sqrt(var)
    return mean,std
