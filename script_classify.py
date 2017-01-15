import csv
import random
import numpy as np
import pandas as pd
import utilities as utils
import math
 
def splitdataset(dataset, trainindex, testindex):
    customertarget = 'Customers'
    salestarget = 'Sales'

    del dataset['Store']
    del dataset['cury']

    features = list(dataset.columns)
    features.remove(customertarget)
    features.remove(salestarget)
    Xtrain = dataset.loc[trainindex,features]
    yCtrain = dataset.loc[trainindex,customertarget]
    yStrain = dataset.loc[trainindex,salestarget]
    Xtest = dataset.loc[testindex,features]
    yCtest = dataset.loc[testindex,customertarget]
    yStest = dataset.loc[testindex,salestarget]
    if Xtest.shape[0] == 0:
        (Xtest, yCtest, yStest) = (Xtrain, yCtrain, yStrain)

    return ((Xtrain.values,list(map(float,yCtrain.values)),list(map(float,yStrain.values))), (Xtest.values,list(map(float,yCtest.values)),list(map(float,yStest.values))))

def multisplitdataset(dataset, trainsize=300, ftestsize = 100, testsize=100):
    # Now randomly split into train and test
    totalrows = dataset.shape[0]
    indices = list(range(totalrows))
    random.shuffle(indices)
    trainindices = indices[0:trainsize]
    testindices = indices[trainsize:trainsize+testsize]
    ftestindices = indices[trainsize+testsize:trainsize+testsize+ftestsize]
    customertarget = 'Sales'
    salestarget = 'Customers'

    del dataset['Store']
    del dataset['cury']

    features = list(dataset.columns)
    features.remove(customertarget)
    features.remove(salestarget)
    Xtrain = dataset.loc[trainindices,features]
    yStrain = dataset.loc[trainindices,salestarget]
    Xftest = dataset.loc[testindices,features]
    yfStest = dataset.loc[testindices,salestarget]
    Xtest = dataset.loc[ftestindices,features]
    yStest = dataset.loc[ftestindices,salestarget]

    return ((Xtrain.values,list(map(float,yStrain.values))), (Xftest.values,list(map(float,yfStest.values))), (Xtest.values,list(map(float,yStest.values))))


def processtrainset(dataset):
    customertarget = 'Customers'
    salestarget = 'Sales'
    del dataset['Store']
    del dataset['cury']
    features = list(dataset.columns)
    features.remove(customertarget)
    features.remove(salestarget)
    Xtrain = dataset.loc[:,features]
    yCtrain = dataset.loc[:,customertarget]
    yStrain = dataset.loc[:,salestarget]
    #print(dataset.columns)
    return (Xtrain.values,list(map(float,yCtrain.values)),list(map(float,yStrain.values)))

def processtestset(dataset):
    del dataset['Store']
    del dataset['Id']
    del dataset['Sales']
    del dataset['cury']
    #print(dataset.columns)
    return dataset.values

def observevar(fullcleanedtrain):
    varlist = []
    for i in range(1,1116):
        if i in fullcleanedtrain.Store:
            cleanedtrain = fullcleanedtrain[fullcleanedtrain.Store== i].copy()
        else:
            continue
        varlist.append(cleanedtrain['Sales'].var())
        print(i)
    ofile = open("variances","w")
    ofile.write(str(varlist))
    ofile.close()
    print(np.percentile(varlist,0))
    print(np.percentile(varlist,20))
    print(np.percentile(varlist,40))
    print(np.percentile(varlist,60))
    print(np.percentile(varlist,80))
    print(np.percentile(varlist,100))
    return
 
def geterror(predictions, ytest):
    # Can change this to other error values
    return math.sqrt(sum(((ytest-predictions)/ytest)**2)/len(ytest))