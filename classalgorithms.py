from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import pandas as pd
import math
import random
import script_classify as sc
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
from sklearn.cluster import KMeans
import copy


class kernelregression:

    def __init__(self, kernel = "p", centers = None, regularizer = 1):
        self.weights = None
        self.kerneltype = self.polynomialkn
        self.regularizer = regularizer
        if kernel == "l":
            self.kerneltype = self.linearkn
        elif kernel == "p":
            self.kerneltype = self.polynomialkn
        elif kernel == "g":
            self.kerneltype = self.gaussiankn
        self.features = None
        self.scalerx = None
        self.scalery = None
        self.scalerkn = None
        self.centers = centers

    def linearkn(self,xi,xj):
        return np.dot(xi,xj)

    def polynomialkn(self,xi,xj):
        return (1 + np.dot(xi,xj))**2

    def gaussiankn(self,xi,xj,sigma = 20):
        return math.e**(-sum((xi-xj)**2)/(2*(sigma**2)))

    def fit(self, Xtrain, ytrain, regularizer = 10):
        regularizer = self.regularizer
        if self.centers == None:
            self.centers = Xtrain.copy()
        knrep = []
        for xi in Xtrain:
            knrep.append(list(map(self.kerneltype, np.repeat(np.array([xi]),len(self.centers),axis=0),self.centers)))

        knrep = np.matrix(knrep)
        kn_scaler = preprocessing.MinMaxScaler().fit(knrep)
        knrep = kn_scaler.transform(knrep)
        self.scalerkn = kn_scaler

        knrep = np.hstack((knrep, np.ones((knrep.shape[0],1))))

        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(knrep.T,knrep) + regularizer*np.identity(knrep.shape[1])), knrep.T),ytrain)
        #print(list(self.weights))

    def sgdlearn(self, Xtrain, ytrain, kerneltype = "linear", regularizer = 10, numofpasses = 2):
        self.kerneltype = kerneltype
        stepsize = 0.00000001
        self.weights = np.random.rand(self.centers.shape[0]+1)
        for i in range(numofpasses):
            for exid in range(Xtrain.shape[0]):
                kernelrep_exid = list(map(self.polynomialkn, np.repeat(np.array([list(Xtrain[exid,:])]),len(self.centers),axis=0),self.centers))
                kernelrep_exid.append(1)
                kernelrep_exid = np.array(kernelrep_exid)
                #print(kernelrep_exid.shape)
                #print(self.weights.shape)
                print(exid)
                self.weights = self.weights - stepsize * ( np.dot(np.dot(kernelrep_exid, self.weights)-ytrain[exid], kernelrep_exid)+ regularizer * self.weights)
                #print(self.weights)
        print(list(self.weights))

    def predict(self, Xtest):
        knreptest = []
        for xi in Xtest:
            knreptest.append(list(map(self.kerneltype, np.repeat(np.array([xi]),len(self.centers),axis=0),self.centers)))
        knreptest = np.matrix(knreptest)
        knreptest = self.scalerkn.transform(knreptest)
        knreptest = np.hstack((knreptest, np.ones((knreptest.shape[0],1))))
        ytest = np.dot(self.weights,knreptest.T)
        return ytest

    def sgdpredict(self, Xtest):
        ytest = []
        for xi in Xtest:
            kernelrep_exid = list(map(self.polynomialkn, np.repeat(np.array([xi]),len(self.centers),axis=0),self.centers))
            kernelrep_exid.append(1)
            kernelrep_exid = np.array(kernelrep_exid)
            ytest.append(np.dot(self.weights,kernelrep_exid))
        return ytest

class poissonreg:
    def __init__(self):
        self.weights = None

    def predict(self, Xtest):
        ytestpower = np.dot(Xtest, self.weights)
        ytest = math.e**(ytestpower)
        return ytest

    def fit(self, Xtrain, ytrain, lamda = 0.5):
        ##require normalization on both x and y
        Xless = Xtrain
        dim = Xless.shape
        numofsample = dim[0]
        ##initialize weight vector
        weights = np.random.rand(dim[1])*0.001
        weightsdif = 10000
        ##design stop condition
        tolerance = 0.0001

        while weightsdif > tolerance:
           ##compute diagonal matrix C
           cii = []
           for i in range(0,numofsample,1):
               cii.append(math.e**np.dot(Xless[i,],weights))
           Cmat = np.diag(cii)

           oldweights = weights
           weights = weights + np.dot(np.dot(np.linalg.inv(np.dot(np.dot(Xless.T,Cmat),Xless) + lamda*np.identity(dim[1])),Xless.T),ytrain-cii) ##
           weightsdif = np.linalg.norm(np.subtract(weights,oldweights),ord=2)
           #print('the weight difference is:')
           #print(weightsdif)

        self.weights = weights