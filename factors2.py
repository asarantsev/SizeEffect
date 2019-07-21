# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:01:29 2019

@author: UNR Math Stat
"""

#Author: Taran Grove
#Start Date: 7/5/2019
#Purpose: This code runs the tests described in "Task8.pdf"



#Important Global Variables
numPortfoliosPerQuarter = 10
NSTOCKS = 500
NQTRS = 40

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



#Code for computing linear regressions.
#This code is adapted from https://github.com/asarantsev/BayesianLongTermFactorModeling
#which was written by the author "UNR Math Stat"
def Regression1(X1, Y):
    n = numpy.size(Y)
    X = pandas.DataFrame({'1': X1})
    X = api.add_constant(X)
    Reg = api.OLS(Y, X).fit()
    Y_Predictions = Reg.predict(X)
    print(Reg.summary())
    residuals = Y - Y_Predictions
    stderr = math.sqrt((1/(n-2))*numpy.dot(residuals, residuals))
    print('standard error = ', stderr)
    return(stderr)

def Regression2(X1, X2, Y):
    n = numpy.size(Y)
    X = pandas.DataFrame({'1': X1, '2': X2})
    X = api.add_constant(X)
    Reg = api.OLS(Y, X).fit()
    Y_Predictions = Reg.predict(X)
    print(Reg.summary())
    #print([Decimal(Reg.params[k]) for k in range(3)])
    residuals = Y - Y_Predictions
    stderr = math.sqrt(1/(n-3)*numpy.dot(residuals, residuals))
    return (stderr)

def Regression(B, F, Y):
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

def Regression3(F1, F2, B, Y):
    n = numpy.size(Y)
    X1 = [F1[k] * B[k] for k in range(n)]
    X2 = [F2[k] * B[k] for k in range(n)]
    X = pandas.DataFrame({'factor 1': F1, 'factor 2': F2, 'Benchmark': B, 'Product 1': X1, 'Product 2': X2})
    X = api.add_constant(X)
    Reg = api.OLS(Y, X).fit()
    Y_Predictions = Reg.predict(X)
    print(Reg.summary())
    residuals = Y - Y_Predictions
    stderr = math.sqrt(1/(n-6)*numpy.dot(residuals, residuals))
    return (stderr)
    
#End adapted code



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
returns = [[tempReturns[stock][NQTRS - t] for t in range(NQTRS)] for stock in range(NSTOCKS)]
caps = [[tempCaps[stock][NQTRS + 1 - t] for t in range(NQTRS)] for stock in range(NSTOCKS)]
factors = [[tempFactors[stock][NQTRS + 1 - t] for t in range(NQTRS)] for stock in range(NSTOCKS)]


caps = numpy.array(caps)
factors = numpy.array(factors)
returns = numpy.array(returns)
    
#Parsing tempIndices data frames into 2 1-Dimensional lists
VanguardPrices = tempIndices[:NQTRS+1, 1]
VanguardPrices = VanguardPrices[::-1]
VanguardDivs = tempIndices[:NQTRS, 2]
VanguardDivs = VanguardDivs[::-1]
TreasuryRates = tempRates[:NQTRS, 1]
TreasuryRates = TreasuryRates[::-1]
TreasuryReturns = [math.log(1 + item/4) for item in TreasuryRates]

#First Benchmark: Vanguard 500
VanguardReturns = []
for t in range(NQTRS):
    #This is just using standard formula for equity premium
    benchRet = math.log((VanguardPrices[t+1] + VanguardDivs[t]) / VanguardPrices[t])
    VanguardPremium = benchRet - TreasuryReturns[t]
    VanguardReturns.append(VanguardPremium)

VanguardReturns = numpy.array(VanguardReturns)

#Section 1B: Computing other data from raw data

#Nominal Return of Stocks
#stockReturns[i][t] = Q_i(t)
prems = returns - [TreasuryReturns for stock in range(NSTOCKS)]
prems = numpy.array(prems)
#Average Capitalization
#avgCaps[t] = C^_(t)
# avgCaps = []
# for t in range(133):
#     count = 0
#     sum = 0
#     #Add up all elements and increment the count
#     for i in range(2905):
#         if (not math.isnan(caps[i][t])):
#             sum += caps[i][t]
#             count += 1
#     if (count == 0):
#         avgCaps.append(np.nan)
#     #Append the sum divided by count
#     else:
#         avgCaps.append(sum / count)

#Relative Log but with avg capitalzation instead of sum of capitalization
# V2s = []
# for i in range(2905):
#     tempArray = []
#     for t in range(133):
#         #Checking to make sure all inputs are well defined
#         if (not math.isnan(caps[i][t])) and (caps[i][t] > 0) and (not math.isnan(avgCaps[t])) and (avgCaps[t] > 0):
#             #Using the given formula for relative log cap but with avg
#             tempArray.append(np.log(caps[i][t]) - np.log(avgCaps[t]))
#         else:
#             tempArray.append(np.nan)
#     V2s.append(tempArray)

AverageCap = []
for t in range(NQTRS):
    column = caps[:, t]
    cleanColumn = [item for item in column if not math.isnan(item)]
    AverageCap.append(numpy.mean(cleanColumn))

#Equity Premuims of Benchmark (Average)
AverageRet = []
WeightedRet = []
for t in range(NQTRS):
    column = numpy.array(prems[:, t])
    cleanColumn = [item for item in column if not math.isnan(item)]
    AverageRet.append(numpy.mean(cleanColumn))
    

    

#Relative Log of Capitalization
#Vs[i][t] = V_i(t) = ln(C_i(t)) - ln(C(t))
Vs = [[numpy.log(caps[stock][t]) - numpy.log(AverageCap[t]) for t in range(NQTRS)] for stock in range(NSTOCKS)]

#Section 3: Random Portfolios
#We will compute two regressions, Q_pi(t) vs E_pi(t) and Q_pi(t) vs V_pi(t)
QRegression1 = []
FRegression1 = []
VRegression1 = []
for t in range(NQTRS):
    #Take the triple intersection
    intersection = tripleIntersect(prems, factors, Vs, t)
    #The number of stocks is the length of any of our three elements of
    #intersection since all three are equal
    numStocks = len(intersection[0])
    for i in range(numPortfoliosPerQuarter):
        #Generate random vector of portfolio weights
        piVector = randomPiVector(numStocks)
        #Compute Q_pi(t), E_pi(t), and V_pi(t)
        QPit = numpy.dot(piVector, intersection[0])
        FPit = numpy.dot(piVector, intersection[1])
        VPit = numpy.dot(piVector, intersection[2])
        #Append each to their respective lists
        QRegression1.append(QPit)
        FRegression1.append(FPit)
        VRegression1.append(VPit)

#Run Q vs E regression
print("")
print("")
print("Portfolio equity premium versus F")
print(Regression1(FRegression1, QRegression1))

#Run Q vs V regression
print("")
print("")
print("Portfolio equity premium versus V")
print(Regression1(VRegression1, QRegression1))

VReturns = []
#Section 4A: Capital Asset Pricing Model
#In this section we will use Vanguard VFINX in 1976
PRegression2 = []
VRegression2 = []
for t in range(NQTRS):
    intersection = intersect(prems, Vs, t)
    numStocks = len(intersection[0])
    #Generate random vector of portfoio weights
    for iteration in range(numPortfoliosPerQuarter):
        piVector = randomPiVector(numStocks)
        #Compute P_pi(t), V_pi(t) and V_pi(t) * P_0(t)
        PPit = numpy.dot(piVector, intersection[0])
        VPit = numpy.dot(piVector, intersection[1])
        #Append everything to its respective list
        PRegression2.append(PPit)
        VRegression2.append(VPit)
        VReturns.append(VanguardReturns[t])

#Run regression
print("")
print("")
print("Multiple Regression for P(t) vs V(t) and Vanguard")
print("Standard Error:", Regression(VReturns, VRegression2, PRegression2))

#PRegression3 = []
#FRegression3 = []
#for t in range(NQTRS):
#    #Only if we have P_0(t), we may compute this because we need
#    #V_pi(t) * P_0(t)
#    intersection = intersect(prems, factors, t)
#    numStocks = len(intersection[0])
#    #Generate random vector of portfoio weights
#    for iteration in range(numPortfoliosPerQuarter):
#        piVector = randomPiVector(numStocks)
#        #Compute P_pi(t), E_pi(t) and E_pi(t) * P_0(t)
#        PPit = numpy.dot(piVector, intersection[0])
#        FPit = numpy.dot(piVector, intersection[1])
#        #Append everything to its respective list
#        PRegression3.append(PPit)
#        FRegression3.append(FPit)
#
##Run multiple regression
#print("")
#print("")
#print("Multiple Regression for P(t) vs F(t) and Vanguard")
#print("Standard Error:", Regression(VReturns, FRegression3, PRegression3))

#Section 4B: Capital Asset Pricing Model
#In this section we will use an equally wegihted portfolio
#as our benchmark
PRegression3 = []
VRegression3 = []
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
        PRegression3.append(PPit)
        VRegression3.append(VPit)
        AvgRet.append(AverageRet[t])

#Run regression
print("")
print("")
print("Multiple Regression for P(t) vs V(t) and avg P(t)")
print("Standard Error:", Regression(AvgRet, VRegression3, PRegression3))

#PRegression4 = []
#FRegression4 = []
#for t in range(NQTRS):
#    #Only if we have P_0(t), we may compute this because we need
#    #V_pi(t) * P_0(t)
#    intersection = intersect(prems, factors, t)
#    numStocks = len(intersection[0])
#    #Generate random vector of portfoio weights
#    for iteration in range(numPortfoliosPerQuarter):
#        piVector = randomPiVector(numStocks)
#        #Compute P_pi(t), E_pi(t) and E_pi(t) * P_0(t)
#        PPit = numpy.dot(piVector, intersection[0])
#        FPit = numpy.dot(piVector, intersection[1])
#        #Append everything to its respective list
#        PRegression4.append(PPit)
#        FRegression4.append(FPit)
#
##Run multiple regression
#print("")
#print("")
#print("Multiple Regression for P(t) vs F(t) and avg P(t)")
#print("Standard Error:", Regression(AvgRet, FRegression4, PRegression4))

#PRegression = []
#FRegression = []
#VRegression = []
#for t in range(NQTRS):
#    intersection = tripleIntersect(prems, factors, Vs, t)
#    numStocks = len(intersection[0])
#    #Generate random vector of portfoio weights
#    for iteration in range(numPortfoliosPerQuarter):
#        piVector = randomPiVector(numStocks)
#        #Compute P_pi(t), V_pi(t) and V_pi(t) * P_0(t)
#        PPit = numpy.dot(piVector, intersection[0])
#        FPit = numpy.dot(piVector, intersection[1])
#        VPit = numpy.dot(piVector, intersection[2])
#        #Append everything to its respective list
#        PRegression.append(PPit)
#        FRegression.append(FPit)
#        VRegression.append(VPit)
#
##Run regression
#print("")
#print("")
#print("Multiple Regression for P(t) vs V(t), F(t), and Vanguard")
#print("standard error:", Regression3(VRegression, FRegression, VReturns, PRegression))

#Run regression
#print("")
#print("")
#print("Multiple Regression for P(t) vs V(t), F(t), and avg P(t)")
#print("stderr:", Regression3(VRegression, FRegression, AvgRet, PRegression))

Ptemp = numpy.array([])
Vtemp = numpy.array([])
VReturnsIndiv = numpy.array([])
for t in range(NQTRS):
    intersection = intersect(prems, Vs, t)
    numStocks = len(intersection[0])
    Ptemp = numpy.append(Ptemp, intersection[0])
    Vtemp = numpy.append(Vtemp, intersection[1])
    VReturnsIndiv = numpy.append(VReturnsIndiv, [VanguardReturns[t] for i in range(numStocks)])
print("")
print("")
print("Individual Stocks: Regression vs V and Vanguard")
print("std error:", Regression(VReturnsIndiv, Vtemp, Ptemp))

#Ptemp = numpy.array([])
#Ftemp = numpy.array([])
#VReturnsIndiv = numpy.array([])
#for t in range(NQTRS):
#    intersection = intersect(prems, factors, t)
#    numStocks = len(intersection[0])
#    Ptemp = numpy.append(Ptemp, intersection[0])
#    Ftemp = numpy.append(Ftemp, intersection[1])
#    VReturnsIndiv = numpy.append(VReturnsIndiv, [VanguardReturns[t] for i in range(numStocks)])
#print("")
#print("")
#print("Individual Stocks: Regression vs F and Vanguard")
#print("standard error:", Regression(VReturnsIndiv, Ftemp, Ptemp))
#
#Ptemp = numpy.array([])
#Ftemp = numpy.array([])
#Vtemp = numpy.array([])
#VReturnsIndiv = numpy.array([])
#for t in range(NQTRS):
#    intersection = tripleIntersect(prems, factors, Vs, t)
#    numStocks = len(intersection[0])
#    Ptemp = numpy.append(Ptemp, intersection[0])
#    Ftemp = numpy.append(Ftemp, intersection[1])
#    Vtemp = numpy.append(Vtemp, intersection[2])
#    VReturnsIndiv = numpy.append(VReturnsIndiv, [VanguardReturns[t] for i in range(numStocks)])
#print("")
#print("")
#print("Individual Stocks: Regression vs V, F, and Vanguard")
#print("standard error:", Regression3(Vtemp, Ftemp, VReturnsIndiv, Ptemp))

    
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
print("Individual Stocks: Regression vs V and avg ret")
print("standard error:", Regression(ARet, Vtemp, Ptemp))

#Ptemp = numpy.array([])
#Ftemp = numpy.array([])
#ARet = numpy.array([])
#for t in range(NQTRS):
#    intersection = intersect(prems, factors, t)
#    numStocks = len(intersection[0])
#    Ptemp = numpy.append(Ptemp, intersection[0])
#    Ftemp = numpy.append(Ftemp, intersection[1])
#    ARet = numpy.append(ARet, [AverageRet[t] for i in range(numStocks)])
#print("")
#print("")
#print("Individual Stocks: Regression vs F and avg ret")
#print("standard error:", Regression(ARet, Ftemp, Ptemp))
#
#Ptemp = numpy.array([])
#Ftemp = numpy.array([])
#Vtemp = numpy.array([])
#ARet = numpy.array([])
#for t in range(NQTRS):
#    intersection = tripleIntersect(prems, factors, Vs, t)
#    numStocks = len(intersection[0])
#    Ptemp = numpy.append(Ptemp, intersection[0])
#    Ftemp = numpy.append(Ftemp, intersection[1])
#    Vtemp = numpy.append(Vtemp, intersection[2])
#    ARet = numpy.append(ARet, [AverageRet[t] for i in range(numStocks)])
#print("")
#print("")
#print("Individual Stocks: Regression vs V, F, and average return")
#print("standard error:", Regression3(Vtemp, Ftemp, ARet, Ptemp))

    

