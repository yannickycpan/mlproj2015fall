import pandas as pd
import script_classify as sc
import classalgorithms as cl
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

def week_of_month(date):
        month = date.month
        week = 0
        while date.month == month:
            week += 1
            date -= timedelta(days=7)
        return week

def saledata_preprocess(original,store):
    originalset = pd.read_csv(str(original))
    storeset = pd.read_csv(str(store))
    rawset = pd.merge(originalset, storeset, on = 'Store')
    
    ## to decrease data set size, delete the rows where the store does not open or sales is zero, which is not useful for our prediction
    rawset = rawset[rawset.Open == 1]
    rawset.reset_index(drop=True)
    
    ## Separate the date column to two columns: month in a year, day in a month
    rawset['MonthInYear'] = [int(month[5:7]) for month in rawset['Date']]
    rawset['DayInMonth'] = [int(day[-2:]) for day in rawset['Date']]
    rawset['Date'] = pd.to_datetime(rawset['Date'])
    
    ## Handle missing value in column 'Promo2SinceYear' and 'Promo2SinceWeek'
    rawset.loc[pd.isnull(rawset.Promo2SinceYear),'Promo2SinceYear'] = 3000
    rawset.loc[pd.isnull(rawset.Promo2SinceWeek),'Promo2SinceWeek'] = 60
    
    ## Create a new column to handle how long a store has started promotion 2, one unit is one week
    ## To achieve this, create a new column 'CurWeekNum' and 'cury' to help computation
    rawset['cury'] = rawset['Date'].dt.year
    rawset['CurWeekNum'] = rawset['Date'].dt.week
    promo2duration = lambda p2wk,p2y,curwk,cury: curwk-p2wk+(cury-p2y)*52 if curwk-p2wk+(cury-p2y)*52>=0 else 0
    rawset['Promo2Duration'] = list(map(promo2duration,rawset['Promo2SinceWeek'],rawset['Promo2SinceYear'],rawset['CurWeekNum'],rawset['cury']))
    
    ## Handle missing values in 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' columns
    #rawset.loc[pd.isnull(rawset.CompetitionOpenSinceMonth),'CompetitionOpenSinceMonth'] = 13
    rawset['CompetitionOpenSinceMonth'].fillna(rawset['CompetitionOpenSinceMonth'].median())
    rawset['CompetitionOpenSinceYear'].fillna(rawset['CompetitionOpenSinceYear'].median())
    #rawset.loc[pd.isnull(rawset.CompetitionOpenSinceYear),'CompetitionOpenSinceYear'] = 3000
    
    ## Add a column to denote how long a competition opened, unit is month
    copenduration = lambda cpmonth,cpyear,curmonth,curyear: curmonth - cpmonth + (curyear - cpyear)*12 if (curmonth - cpmonth + (curyear - cpyear)*12)>=0 else 0
    rawset['CompDuration'] = list(map(copenduration,rawset['CompetitionOpenSinceMonth'], rawset['CompetitionOpenSinceYear'], rawset['Date'].dt.month,rawset['cury']))
    
    ## Handle missing values in 'PromoInterval' column
    rawset.loc[pd.isnull(rawset.PromoInterval),'PromoInterval'] = 'no'
    rawset['PromoInterval'] = rawset.PromoInterval.apply(str)
    
    ## Create a column denote whether current date is in promotion interval
    monthDict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    promointerval = lambda x,y: 1 if (x in monthDict) and (monthDict[x] in y) else 0
    rawset['InPromoInterval'] = list(map(promointerval, rawset['Date'].dt.month, rawset['PromoInterval']))

    ## Handling missing values in CompetitionDistance
    #rawset['CompetitionDistance'].fillna(rawset['CompetitionDistance'].median())

    ## Replace characters with integers in columns Assortment, StoreType, StateHoliday
    #replacechar = lambda x: 1 if x == 'a' else (2 if x == 'b' else (3 if x == 'c' else 4))
    stateholiday = lambda x: 1 if (x == 'a' or x == 'b' or x == 'c') else 0
    rawset['StateHoliday'] = list(map(stateholiday,rawset['StateHoliday']))
    #rawset['StoreType'] = list(map(replacechar,rawset['StoreType']))
    #rawset['Assortment'] = list(map(replacechar,rawset['Assortment']))

    # Create dummy varibales for DayOfWeek
    day_dummies_rossmann  = pd.get_dummies(rawset['DayOfWeek'], prefix='Day')
    day_dummies_rossmann.drop(['Day_7'], axis=1, inplace=True)
    rawset = rawset.join(day_dummies_rossmann)
    del rawset['DayOfWeek']

    # Create dummy variables for Month 8 and 9
    dummymonth8 = lambda x: 1 if x == 8 else 0
    dummymonth9 = lambda x: 1 if x == 9 else 0
    rawset['Month_8'] = list(map(dummymonth8,rawset['MonthInYear']))
    rawset['Month_9'] = list(map(dummymonth9,rawset['MonthInYear']))

    # Create a column to show which week it is in a month
    rawset['WeekInMonth'] = list(map(week_of_month, rawset['Date']))
    # Create dummy variable for index of week in a month
    weekinmonth_dummies  = pd.get_dummies(rawset['WeekInMonth'], prefix='WeekM')
    weekinmonth_dummies.drop(['WeekM_5'], axis=1, inplace=True)
    rawset = rawset.join(weekinmonth_dummies)
    del rawset['WeekInMonth']

    del rawset['CompetitionOpenSinceMonth']
    del rawset['CompetitionOpenSinceYear']
    del rawset['Promo2SinceWeek']
    del rawset['Promo2SinceYear']
    del rawset['PromoInterval']
    del rawset['CurWeekNum']
    del rawset['Open']
    del rawset['Date']
    del rawset['Promo2']
    del rawset['StoreType']
    del rawset['MonthInYear']
    del rawset['Assortment']
    del rawset['CompetitionDistance']
    return rawset

def datascaling(dataset):
    #print(dataset.columns)
    scaledfeatures = ['Promo2Duration', 'CompDuration','DayInMonth']

    Xtrainbool_index = list(((dataset.Month_8!=1) & (dataset.Month_9!=1)) | (dataset.cury!=2014))
    Xtestbool_index = list(((dataset.Month_8==1) | (dataset.Month_9==1)) & (dataset.cury==2014))

    x_scaler = preprocessing.MinMaxScaler().fit(dataset[scaledfeatures])
    dataset[scaledfeatures] = x_scaler.transform(dataset[scaledfeatures])
    return (dataset, x_scaler, scaledfeatures, Xtrainbool_index, Xtestbool_index)

if __name__ == '__main__':
    fullcleanedtrain = pd.read_csv("tidysalesdata.csv")
    plt.hist(fullcleanedtrain[fullcleanedtrain.Store==1112].Sales.values)
    print(np.var(fullcleanedtrain[fullcleanedtrain.Store==1112].Sales.values))
    plt.savefig("sales_histogram1112.png")
    plt.show()
    plt.close()