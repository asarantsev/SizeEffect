#Author: Taran Grove
#Start Date: 7/5/2019
#Purpose: This code runs the tests described in "Task8.pdf"



#Important Global Variables
numPortfoliosPerQuarter = 100
NSTOCKS = 500
NQTRS = 120

#Importing Libraries
import pandas
import math
import numpy
from numpy import random
import scipy
from scipy import stats
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from statsmodels import api

def Regression2(B, F, Y):
    n = numpy.size(Y)
    P = [B[k] * F[k] for k in range(n)]
    X = pandas.DataFrame({'Benchmark': B, 'Factor': F, 'Product': P})
    X = api.add_constant(X)
    Reg = api.OLS(Y, X).fit()
    Y_Predictions = Reg.predict(X)
    print(Reg.summary())
    #print([Decimal(Reg.params[k]) for k in range(3)])
    residuals = Y - Y_Predictions
    stderr = math.sqrt(1/(n-4)*numpy.dot(residuals, residuals))
    return (stderr)

#Helper functions
#This function returns two lists where both are defined
def intersect(list1, list2, time):
    listA = []
    listB = []
    for i in range(NSTOCKS):
        #Only if both are defined, should we add to respective lists
        if (not math.isnan(list1[i][time])) and (not math.isnan(list2[i][time])):
            listA.append(list1[i][time])
            listB.append(list2[i][time])
    return [listA, listB]

#This functions returns three lists where all three lists
#simultaneously defined for respective times.
def tripleIntersect(l1, l2, l3, t):
    lA = []
    lB = []
    lC = []
    for i in range(NSTOCKS):
        #Only if all three are defined, should we add to respective lists
        if (not math.isnan(l1[i][t])) and (not math.isnan(l2[i][t])) and (not math.isnan(l3[i][t])):
            lA.append(l1[i][t])
            lB.append(l2[i][t])
            lC.append(l3[i][t])
    return [lA, lB, lC]

#This functions returns a vector with N elements
#All the elements sum to 1 and each element is randomly generated
def randomPiVector(N):
    #First fill a vector with N random exponential variables
    temp = [random.exponential() for k in range(N)]
    piVectorSum = sum(temp)
    #Now divide each by total sum so that this new vector sums to 1
    piVector = [item / piVectorSum for item in temp]
    return(piVector)


#Section 1A: Raw Data

#Obtaining data from 'Data500.xlsx'
#Note one will need to adjust the file path for their own computer.
file1 = pandas.ExcelFile('Data500.xlsx')
df1 = file1.parse('Return')
df2 = file1.parse('Cap')
df3 = file1.parse('PriceBook')


file2 = pandas.ExcelFile('VFINX.xlsx')
dfV = file2.parse('VFINX')
file3 = pandas.ExcelFile('TR3month-quarterly.xlsx')
dfR = file3.parse('Rates')

#Create 2-dimensional lists of the data frames
tempReturns = df1.values
tempCaps = df2.values
tempFactors = df3.values
tempIndices = dfV.values
tempRates = dfR.values

#Create 2-dimensional lists that are better suited for our analysis
#prices[i][t] = S_i(t), divs[i][t] = D_i(t), caps[i][t] = C_i(t)
returns = [[math.log(1 + tempReturns[stock][NQTRS - t]) for t in range(NQTRS)] for stock in range(NSTOCKS)]
caps = [[tempCaps[stock][NQTRS + 1 - t] for t in range(NQTRS)] for stock in range(NSTOCKS)]
factors = [[tempFactors[stock][NQTRS + 1 - t] for t in range(NQTRS)] for stock in range(NSTOCKS)]


caps = numpy.array(caps)
factors = numpy.array(factors)
returns = numpy.array(returns)
    
#Parsing tempIndices data frames into 2 1-Dimensional lists
TreasuryRates = tempRates[:NQTRS, 1]
TreasuryRates = TreasuryRates[::-1]
TreasuryReturns = [math.log(1 + item/4) for item in TreasuryRates]

prems = returns - [TreasuryReturns for stock in range(NSTOCKS)]
prems = numpy.array(prems)

AverageCap = []
sumCap = []
AverageRet = []
WeightedRet = []
for t in range(NQTRS):
    column1 = caps[:, t]
    cleanColumn1 = [item for item in column1 if not math.isnan(item)]
    AverageCap.append(numpy.mean(cleanColumn1))
    s = sum(cleanColumn1)
    sumCap.append(s)
    column2 = numpy.array(prems[:, t])
    cleanColumn2 = [item for item in column2 if not math.isnan(item)]
    AverageRet.append(numpy.mean(cleanColumn2))
    column3 = [prems[k, t]*caps[k, t] for k in range(NSTOCKS)]
    cleanColumn3 = [item for item in column3 if not math.isnan(item)]
    WeightedRet.append(sum(cleanColumn3)/s)
    
Vs = [[numpy.log(caps[stock][t]) - numpy.log(AverageCap[t]) for t in range(NQTRS)] for stock in range(NSTOCKS)]

PRegression = []
VRegression = []
AvgRet = []
for t in range(NQTRS):
    #Only if we have P_0(t), we may compute this because we need
    #V_pi(t) * P_0(t)
    intersection = intersect(prems, Vs, t)
    numStocks = len(intersection[0])
    #Generate random vector of portfoio weights
    for iteration in range(numPortfoliosPerQuarter):
        piVector = randomPiVector(numStocks)
        #Compute P_pi(t), V_pi(t) and V_pi(t) * P_0(t)
        PPit = numpy.dot(piVector, intersection[0])
        VPit = numpy.dot(piVector, intersection[1])
        #Append everything to its respective list
        PRegression.append(PPit)
        VRegression.append(VPit)
        AvgRet.append(AverageRet[t])


print("")
print("")
print("Random Portfolios: Equally-Weighted Portfolio Benchmark")
print("Standard Error:", Regression2(AvgRet, VRegression, PRegression))

Ptemp = numpy.array([])
Vtemp = numpy.array([])
ARet = numpy.array([])
for t in range(NQTRS):
    intersection = intersect(prems, Vs, t)
    numStocks = len(intersection[0])
    Ptemp = numpy.append(Ptemp, intersection[0])
    Vtemp = numpy.append(Vtemp, intersection[1])
    ARet = numpy.append(ARet, [AverageRet[t] for i in range(numStocks)])
print("")
print("")
print("Individual Stocks: Equally-Weighted Portfolio Benchmark")
print("standard error:", Regression2(ARet, Vtemp, Ptemp))

