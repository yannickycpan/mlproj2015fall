import numpy as np
import script_classify as sc
import classalgorithms as cl
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from operator import add
import matplotlib.pyplot as plt
import salespreprocessing as pp
import pandas as pd
import sys
from scipy.stats.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.svm import SVR
import random
import copy
import math

## the basic idea is to use multiple learning algorithms to form an ensemble method
## first idea is try to predict number of customers and see whether it would be easier
## second idea is to predict customers first and then predict sales; then use sales to predict customers until converge or certain number of iterations reached
def findcenters(trainset,targettocombine, target, factor):
    trainset = np.column_stack((trainset,targettocombine))
    return KMeans(n_clusters=int(factor*trainset.shape[0])).fit(trainset,target).cluster_centers_

def findrandomcenters(trainset,targettocombine, factor):
    trainset = np.column_stack((trainset,targettocombine))
    centers = trainset[random.sample(list(range(trainset.shape[0])),int(factor*trainset.shape[0])),:]
    return centers

'''def findrandomcenters(trainset,factor):
    centers = trainset[random.sample(list(range(trainset.shape[0])),int(factor*trainset.shape[0])),:]
    return centers'''

def predictsales():
    fullcleanedtrain = pd.read_csv("tidysalesdata.csv")
    fullcleanedtrain = fullcleanedtrain[fullcleanedtrain.Sales != 0]
    fullnewtest = pd.read_csv("tidysalesdata_test.csv")
    fullnewtest['Sales'] = 0
    numofstores = 1115
    start = 2
    for storeindex in range(start,numofstores+1):
        if storeindex not in list(fullnewtest.Store):
            continue
        cleanedtrain = fullcleanedtrain[fullcleanedtrain.Store==storeindex].copy()
        cleanedtrain.index = range(cleanedtrain.shape[0])
        newtest = fullnewtest[fullnewtest.Store==storeindex].copy()
        newtest.index = range(newtest.shape[0])
        ##regularizer
        regl = chooseregularizer(cleanedtrain['Sales'].var())
        frac = choosefrac(cleanedtrain['Sales'].var())

        (cleanedtrain, x_scaler, scaledfeatures, Xtrainbool_index, Xtestbool_index) = pp.datascaling(cleanedtrain)
        newtest[scaledfeatures] = x_scaler.transform(newtest[scaledfeatures])
        ((Xtrain,yCtrain,yStrain), (Xftest,yfCtest,yfStest)) = sc.splitdataset(cleanedtrain.copy(),Xtrainbool_index, Xtestbool_index) #create a false test set to do model evaluation
        #(Xtrain,yCtrain,yStrain) = sc.processtrainset(cleanedtrain)
        Xtest = sc.processtestset(newtest)

        ##scale target variable data to [0,1] interval for training purpose, features are already scaled
        yc_scaler = preprocessing.MinMaxScaler().fit(yCtrain)
        ys_scaler = preprocessing.MinMaxScaler().fit(yStrain)
        scaled_trainyc = yc_scaler.fit_transform(yCtrain)
        scaled_trainys = ys_scaler.fit_transform(yStrain)

        ##num of iteration
        i = 0
        numofiteration = 5
        ##num of times to find good training samples
        numoffinding = 4
        j = 0
        ##find centers
        centerswithcustomers = findrandomcenters(Xtrain.copy(),scaled_trainyc,frac)
        centerswithsales = findrandomcenters(Xtrain.copy(),scaled_trainys,frac)
        ##to record best model
        bestmodel = None
        lowesterror = float('inf')
        bestcustomers = None
        ## iteratively change training samples
        while i < numofiteration:
            print(str(i)+' '+ str(lowesterror) + ' ' + str(j))
            centers = centerswithsales if i>0 else centerswithsales[:,0:centerswithsales.shape[1]-1]
            kernellearner = cl.kernelregression(centers)
            kernellearner.learn(Xtrain,scaled_trainyc,regularizer = regl)

            '''predictedcustomers_train = kernellearner.predict(Xtrain)
            predictedcustomers_train[predictedcustomers_train>2*max(scaled_trainyc)] = np.mean(scaled_trainyc)
            predictedcustomers_train[predictedcustomers_train<0] = np.mean(scaled_trainyc)'''
            Xtrain = Xtrain[:,0:Xtrain.shape[1]-1] if i>0 else Xtrain
            Xtrain = np.column_stack((Xtrain,scaled_trainyc))

            predictedcustomers_ftest = kernellearner.predict(Xftest)
            predictedcustomers_ftest[predictedcustomers_ftest>2*max(scaled_trainyc)] = np.mean(scaled_trainyc)
            predictedcustomers_ftest[predictedcustomers_ftest<0] = np.mean(scaled_trainyc)
            Xftest = Xftest[:,0:Xftest.shape[1]-1] if i>0 else Xftest
            Xftest = np.column_stack((Xftest,predictedcustomers_ftest))

            '''predictedcustomers_test = kernellearner.predict(Xtest)
            predictedcustomers_test[predictedcustomers_test>2*max(scaled_trainyc)] = np.mean(scaled_trainyc)
            predictedcustomers_test[predictedcustomers_test<0] = np.mean(scaled_trainyc)
            Xtest = Xtest[:,0:Xtest.shape[1]-1] if i>0 else Xtest
            Xtest = np.column_stack((Xtest,predictedcustomers_test))'''

            centers = centerswithcustomers
            kernellearner = cl.kernelregression(centers)
            kernellearner.learn(Xtrain,scaled_trainys,regularizer = regl)
            scaled_predictions = kernellearner.predict(Xftest)
            predictions = ys_scaler.inverse_transform(scaled_predictions)
            error = sc.geterror(predictions,yfStest)
            print('The predicted error of my poly kenel regression on False testing set is: ')
            print(error)
            print('The predicted error of mean predictor is: ')
            temp = np.empty(len(yfStest))
            temp.fill(np.mean(yStrain))
            print(sc.geterror(temp,yfStest))
            if error < lowesterror and i<numofiteration-1:
                lowesterror = error
                bestmodel = copy.copy(kernellearner)
                bestcustomers = Xtest[:,Xtest.shape[1]-1].copy()
                #print(scaled_trainyc)
            elif error > 2:
                if j < numoffinding:
                    Xtrain = Xtrain[:,0:Xtrain.shape[1]-1]
                    Xtest = Xtest[:,0:Xtest.shape[1]-1]
                    Xftest = Xftest[:,0:Xftest.shape[1]-1]
                    centerswithcustomers = findrandomcenters(Xtrain.copy(),scaled_trainyc,frac)
                    centerswithsales = findrandomcenters(Xtrain.copy(),scaled_trainys,frac)
                    print("****************error too large******************")
                    j = j + 1
                    i = 0
                    continue
                else:
                    i = numofiteration
            elif lowesterror > 0.1 and i == numofiteration-1 and j<numoffinding:
                Xtrain = Xtrain[:,0:Xtrain.shape[1]-1]
                Xtest = Xtest[:,0:Xtest.shape[1]-1]
                Xftest = Xftest[:,0:Xftest.shape[1]-1]
                centerswithcustomers = findrandomcenters(Xtrain.copy(),scaled_trainyc,frac)
                centerswithsales = findrandomcenters(Xtrain.copy(),scaled_trainys,frac)
                print("*******************error not low enough***************")
                j = j + 1
                i = 0
                continue
            '''predictedsales_train = kernellearner.predict(Xtrain)
            predictedsales_train[predictedsales_train>2*max(scaled_trainys)] = np.mean(scaled_trainys)
            predictedsales_train[predictedsales_train<0] = np.mean(scaled_trainys)'''
            Xtrain = Xtrain[:,0:Xtrain.shape[1]-1]
            Xtrain = np.column_stack((Xtrain,scaled_trainys))

            '''predictedsales_test = kernellearner.predict(Xtest)
            predictedsales_test[predictedsales_test>2*max(scaled_trainys)] = np.mean(scaled_trainys)
            predictedsales_test[predictedsales_test<0] = np.mean(scaled_trainys)
            Xtest = Xtest[:,0:Xtest.shape[1]-1]
            Xtest = np.column_stack((Xtest,predictedsales_test))'''

            predictedsales_ftest = kernellearner.predict(Xftest)
            predictedsales_ftest[predictedsales_ftest>2*max(scaled_trainys)] = np.mean(scaled_trainys)
            predictedsales_ftest[predictedsales_ftest<0] = np.mean(scaled_trainys)
            Xftest = Xftest[:,0:Xftest.shape[1]-1]
            Xftest = np.column_stack((Xftest,predictedsales_ftest))

            i = i+1

        Xtest = Xtest[:,0:Xtest.shape[1]-1]
        Xtest = np.column_stack((Xtest,bestcustomers))
        scaled_predictions = bestmodel.predict(Xtest)
        predictions = ys_scaler.inverse_transform(scaled_predictions)
        '''print('The final predicted error of my poly kenel regression on testing sales is: ')
        print(sc.geterror(predictions,yStest))
        print('The predicted error of mean predictor is: ')
        temp = np.empty(len(yStest))
        temp.fill(np.mean(yStrain))
        print(sc.geterror(temp,yStest))'''
        fullnewtest.loc[fullnewtest.Store == storeindex,'Sales'] = predictions
        print("currently processed store "+ str(storeindex))
        f = open('resultfile.csv', 'a')
        fullnewtest.loc[fullnewtest.Store == storeindex].to_csv(f, header=False)
        f.close()
    #fullnewtest.to_csv("allresultfile.csv")

def simplified_predictsales():
    fullcleanedtrain = pd.read_csv("tidysalesdata.csv")
    fullcleanedtrain = fullcleanedtrain[fullcleanedtrain.Sales != 0]
    fullnewtest = pd.read_csv("tidysalesdata_test.csv")
    fullnewtest['Sales'] = 0
    numofstores = 1115
    start = 305
    for storeindex in range(start,numofstores+1):
        if storeindex not in list(fullnewtest.Store):
            continue
        print("current processing: "+ str(storeindex))
        cleanedtrain = fullcleanedtrain[fullcleanedtrain.Store==storeindex].copy()
        cleanedtrain.index = range(cleanedtrain.shape[0])
        newtest = fullnewtest[fullnewtest.Store==storeindex].copy()
        newtest.index = range(newtest.shape[0])
        testid = newtest['Id']
        ##regularizer
        regl = chooseregularizer(cleanedtrain['Sales'].var())
        #regl = 0
        #frac = choosefrac(cleanedtrain['Sales'].var())
        (cleanedtrain, x_scaler, scaledfeatures, Xtrainbool_index, Xtestbool_index) = pp.datascaling(cleanedtrain)
        newtest[scaledfeatures] = x_scaler.transform(newtest[scaledfeatures])
        ((Xtrain,yCtrain,yStrain), (Xtest,yfCtest,yStest)) = sc.splitdataset(cleanedtrain.copy(),Xtrainbool_index, Xtestbool_index) #create a false test set to do model evaluation
        #(Xtrain,yCtrain,yStrain) = sc.processtrainset(cleanedtrain.copy())
        NXtest = sc.processtestset(newtest.copy())
        ##suse log transformation on target variable
        logy = np.array(list(map(math.log,yStrain)))
        errorlist = []
        modellist = []

        svrp = SVR(kernel='poly', C=1e3, degree=2)
        scaled_predictions = svrp.fit(Xtrain, logy).predict(Xtest)
        predictionsvp = np.array(list(map(math.exp,scaled_predictions)))
        errorvp = sc.geterror(predictionsvp,yStest)
        print('The final predicted error of my SVM poly kenel regression is: ')
        print(errorvp)
        errorlist.append(errorvp)
        modellist.append(svrp)

        svrl = SVR(kernel='linear', C=1e3)
        scaled_predictions = svrl.fit(Xtrain, logy).predict(Xtest)
        predictionsvl = np.array(list(map(math.exp,scaled_predictions)))
        errorvl = sc.geterror(predictionsvl,yStest)
        print('The final predicted error of my SVM l kenel regression is: ')
        print(errorvl)
        errorlist.append(errorvl)
        modellist.append(svrl)

        kernellearner = cl.kernelregression("l",regularizer = regl)
        kernellearner.fit(Xtrain,logy)
        scaled_predictions = kernellearner.predict(Xtest)
        predictionsl = np.array(list(map(math.exp,scaled_predictions)))
        errorl = sc.geterror(predictionsl,yStest)
        print('The final predicted error of my l kenel regression is: ')
        print(errorl)
        errorlist.append(errorl)
        modellist.append(cl.kernelregression("l",regularizer = regl))

        kernellearner = cl.kernelregression("p",regularizer = regl)
        kernellearner.fit(Xtrain,logy)
        scaled_predictions = kernellearner.predict(Xtest)
        predictionsp = np.array(list(map(math.exp,scaled_predictions)))
        errorp = sc.geterror(predictionsp,yStest)
        print('The final predicted error of my p kenel regression is: ')
        print(errorp)
        errorlist.append(errorp)
        modellist.append(cl.kernelregression("p",regularizer = regl))

        kernellearner = cl.kernelregression("g",regularizer = regl)
        kernellearner.fit(Xtrain,logy)
        scaled_predictions = kernellearner.predict(Xtest)
        predictionsg = np.array(list(map(math.exp,scaled_predictions)))
        errorg = sc.geterror(predictionsg,yStest)
        print('The final predicted error of my g kenel regression is: ')
        print(errorg)
        errorlist.append(errorg)
        modellist.append(cl.kernelregression("g",regularizer = regl))

        lreg = linear_model.Ridge(alpha = 0.1)
        lreg.fit(Xtrain, logy)
        scaled_predictions = lreg.predict(Xtest)
        predictionslr = np.array(list(map(math.exp,scaled_predictions)))
        errorlr = sc.geterror(predictionslr,yStest)
        print('The final predicted error of my linear regression is: ')
        print(errorlr)
        errorlist.append(errorlr)
        modellist.append(linear_model.LinearRegression())

        breg = linear_model.BayesianRidge(compute_score=True)
        breg.fit(Xtrain, logy)
        scaled_predictions = breg.predict(Xtest)
        predictionsb = np.array(list(map(math.exp,scaled_predictions)))
        errorb = sc.geterror(predictionsb,yStest)
        print('The final predicted error of my bayes regression is: ')
        print(errorb)
        errorlist.append(errorb)
        modellist.append(linear_model.BayesianRidge(compute_score=True))

        preg = cl.poissonreg()
        preg.fit(Xtrain, logy)
        scaled_predictions = preg.predict(Xtest)
        predictionspr = np.array(list(map(math.exp,scaled_predictions)))
        errorpr = sc.geterror(predictionspr,yStest)
        print('The final predicted error of my poisson regression is: ')
        print(errorpr)
        errorlist.append(errorpr)
        modellist.append(copy.copy(cl.poissonreg()))

        errorlist = np.array(errorlist)
        sorted_index = np.argsort(errorlist)
        #print(errorlist[sorted_index[0:3]])
        ensemble = 0
        numofmd = 5
        for i in range(numofmd):
            (Xtrain,yCtrain,yStrain) = sc.processtrainset(cleanedtrain.copy())
            logy = np.array(list(map(math.log,yStrain)))
            #ys_scaler = preprocessing.MinMaxScaler().fit(yStrain)
            #scaled_trainsy = ys_scaler.transform(yStrain)
            modellist[sorted_index[i]].fit(Xtrain,logy)
            scaled_predictions = modellist[sorted_index[i]].predict(NXtest)
            temp_predictions = np.array(list(map(math.exp,scaled_predictions)))
            #temp_predictions = ys_scaler.inverse_transform(scaled_predictions)
            ensemble = ensemble + temp_predictions * (1/errorlist[sorted_index[i]])/sum(1/(errorlist[sorted_index[0:numofmd]]))
            #print((1/errorlist[sorted_index[i]])/sum(1/(errorlist[sorted_index[0:numofmd]])))
        if min(errorlist) > 0.15:
            ofile = open("record.txt","a")
            ofile.write(" "+str(storeindex))
            ofile.close()
        #ensemble = predictionsl + predictionsp + predictionsg + predictionslr + predictionsb + predictionspr
        ensembleprediction = ensemble
        '''error = sc.geterror(ensembleprediction,yStest)
        print('The final predicted error of my ensemble is: ')
        print(error)'''

        submission = pd.Series(ensembleprediction, index=testid)
        submission = pd.DataFrame({ "Id": submission.index, "Sales": submission.values})
        print("currently processed store "+ str(storeindex))
        f = open('resultfile.csv', 'a')
        submission.to_csv(f, header=False,index=False)
        f.close()
    #fullnewtest.to_csv("allresultfile.csv")

def choosefrac(v):
    if v <= 331721:
        return 0.5
    elif v<= 1762504:
        return 0.6
    elif v<=2635979:
        return 0.7
    elif v<=3652159:
        return 0.8
    elif v<=5149050:
        return 0.9
    else:
        return 1

def chooseregularizer(v):
    if v <= 331721:
        return 0.00000001
    elif v<= 1762504:
        return 0.00000002
    elif v<=2635979:
        return 0.00000004
    elif v<=3652159:
        return 0.00000006
    elif v<=5149050:
        return 0.00000008
    else:
        return 0.01

def learntopredict():
    fullcleanedtrain = pd.read_csv("tidysalesdata.csv")
    print(fullcleanedtrain.columns)
    cleanedtrain = fullcleanedtrain[fullcleanedtrain.Store.isin([5])].copy()
    cleanedtrain.index = range(cleanedtrain.shape[0])
    ##regularizer
    regl = chooseregularizer(cleanedtrain['Sales'].var())
    frac = choosefrac(cleanedtrain['Sales'].var())

    (cleanedtrain, x_scaler, scaledfeatures)= pp.datascaling(cleanedtrain)
    dim = cleanedtrain.shape
    ((Xtrain,yCtrain,yStrain), (Xtest,yCtest,yStest)) = sc.splitdataset(cleanedtrain,int(0.8*dim[0]),int(0.2*dim[0]))

    ##scale target variable data to [0,1] interval for training purpose, features are already scaled
    yc_scaler = preprocessing.MinMaxScaler().fit(yCtrain)
    ys_scaler = preprocessing.MinMaxScaler().fit(yStrain)
    scaled_trainyc = yc_scaler.fit_transform(yCtrain)
    scaled_trainys = ys_scaler.fit_transform(yStrain)

    ##num of iteration
    numofiteration = 5
    ##initialization
    print(Xtrain.shape)
    centerswithcustomers = findrandomcenters(Xtrain.copy(),scaled_trainyc,frac)
    centerswithsales = findrandomcenters(Xtrain.copy(),scaled_trainys,frac)
    print(Xtrain.shape)
    bestmodel = None

    ## iteratively change cluster centers
    for i in range(numofiteration):
        centers = centerswithsales if i>0 else centerswithsales[:,0:centerswithsales.shape[1]-1]
        kernellearner = cl.kernelregression(centers)
        kernellearner.fit(Xtrain,scaled_trainyc,regularizer = regl,kerneltype="poly")

        predictedcustomers_train = kernellearner.predict(Xtrain)
        Xtrain = Xtrain[:,0:Xtrain.shape[1]-1] if i>0 else Xtrain
        Xtrain = np.column_stack((Xtrain,predictedcustomers_train))
        predictedcustomers_test = kernellearner.predict(Xtest)
        Xtest = Xtest[:,0:Xtest.shape[1]-1] if i>0 else Xtest
        Xtest = np.column_stack((Xtest,predictedcustomers_test))

        centers = centerswithcustomers
        kernellearner = cl.kernelregression(centers)
        kernellearner.fit(Xtrain,scaled_trainys,regularizer = regl,kerneltype="poly")
        scaled_predictions = kernellearner.predict(Xtest)
        predictions = ys_scaler.inverse_transform(scaled_predictions)
        error = sc.geterror(predictions,yStest)
        #print(predictions)
        #print(yStest)
        print('The predicted error of my poly kenel regression on sales is: ')
        print(error)

        predictedsales_train = kernellearner.predict(Xtrain)
        Xtrain = Xtrain[:,0:Xtrain.shape[1]-1]
        Xtrain = np.column_stack((Xtrain,predictedsales_train))
        predictedsales_test = kernellearner.predict(Xtest)
        Xtest = Xtest[:,0:Xtest.shape[1]-1]
        Xtest = np.column_stack((Xtest,predictedsales_test))

    predictedcustomers_train = kernellearner.predict(Xtrain)
    Xtrain = Xtrain[:,0:Xtrain.shape[1]-1]
    Xtrain = np.column_stack((Xtrain,predictedcustomers_train))
    predictedcustomers_test = kernellearner.predict(Xtest)
    Xtest = Xtest[:,0:Xtest.shape[1]-1]
    Xtest = np.column_stack((Xtest,predictedcustomers_test))

    kernellearner = cl.kernelregression(centerswithcustomers)
    kernellearner.fit(Xtrain,scaled_trainys,regularizer = regl,kerneltype="poly")
    scaled_predictions = kernellearner.predict(Xtest)
    predictions = ys_scaler.inverse_transform(scaled_predictions)
    error = sc.geterror(predictions,yStest)
    print('The final predicted error of my poly kenel regression is: ')
    print(error)

    print('The predicted error of mean predictor is: ')
    temp = np.empty(len(yStest))
    temp.fill(np.mean(yStrain))
    print(sc.geterror(temp,yStest))


def readtrain():
    cleanedtrain = pp.saledata_preprocess("train.csv", "store.csv")
    cleanedtrain.to_csv("tidysalesdata.csv",index=False)
    return cleanedtrain

def readtest():
    cleanedtest = pp.saledata_preprocess("test.csv", "store.csv")
    cleanedtest.to_csv("tidysalesdata_test.csv",index=False)
    return cleanedtest

def write_predicted_testfile():
    resultfile = pd.read_csv("resultfile.csv")
    print(resultfile.shape)
    test = pd.read_csv("test.csv")
    test['Sales'] = 0
    test['Open'].fillna(0, inplace=True)
    test = test[test.Open==0]
    print(test.shape)
    test = test[['Id','Sales']]

    frames = [resultfile,test]
    result = pd.concat(frames)
    result = result.sort('Id')
    print(result.shape)
    result['Sales'] = list(map(int,result.Sales))
    result.to_csv("finalsubmission7.csv",index=False)

if __name__ == '__main__':
    #readtrain()
    #readtest()
    ##learntopredict()
    #predictsales()
    simplified_predictsales()
    '''cleaned = pd.read_csv("test.csv")
    print(cleaned.shape)
    cleaned.fillna(0, inplace=True)
    print(cleaned[cleaned.Open==1].shape)
    print(cleaned[cleaned.Open==0].shape)
    print(cleaned.loc[cleaned.Open=='Open', :].shape)
    print(cleaned.shape)
    print(pd.unique(cleaned['Open'].values.ravel()))'''
    #write_predicted_testfile()
    '''fullcleanedtrain = pd.read_csv("tidysalesdata.csv")
    plt.hist(fullcleanedtrain[fullcleanedtrain.Store==3].Sales.values)
    plt.savefig("sales_histogram.png")
    plt.show()
    plt.close()'''
    if len(sys.argv)>1:
        if str(sys.argv[1]) == "pretrain":
            readtrain()
        elif str(sys.argv[1]) == "pretest":
            readtest()
        elif str(sys.argv[1]) == "simple_pred":
            simplified_predictsales()
        elif str(sys.argv[1]) == "self_pred":
            predictsales()
        elif str(sys.argv[1]) == "salesplot":
            ind = int(sys.argv[2])
            fullcleanedtrain = pd.read_csv("tidysalesdata.csv")
            plt.hist(fullcleanedtrain[fullcleanedtrain.Store==ind].Sales.values)
            print(np.var(fullcleanedtrain[fullcleanedtrain.Store==ind].Sales.values))
            plt.show()
            plt.close()

